#!/usr/bin/env python3
"""GaussianGPT faithful *deterministic-position* speculative decoder smoke test.

Dependencies: eval_lean_candidate_to_draft.py, eval_lean_ranker_cpu.py,
model checkpoints and profiling.s3.s3_onpolicy_feature_distill_train.make_kv_cache.

Frozen on-policy trained ranker predicts *one* next Position from the Lean set.
The Target verifies [previous Feature, proposed Position] together. Rejections
sample from exact target residual (p - delta_proposal)_+ / normalization and
rewind the cache by one token before repairing it. The Target alone samples all
Features.  Sampled outputs need not match vanilla with the same seed because
accept/reject consumes random numbers; compare distributions and timing.

FIXED 1024-token training horizon: the ranker feature i/511 is intentionally
unchanged, even for shorter benchmark output; this prevents future EOS leakage.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from joblib import load
from threadpoolctl import threadpool_limits

from eval_lean_candidate_to_draft import FrozenLean, draft_weights, ids_from_gaps, mode, topk, target_policy


class CausalRanker:
    """Exactly reproduce the training feature schema from build_examples()."""
    def __init__(self, rank_map, p_vocab):
        self.predictor = FrozenLean()
        self.m3 = defaultdict(Counter)
        self.m3_phase = defaultdict(Counter)
        self.last_seen = {}
        self.n_finalized_positions = 1  # P0 already sampled; proposing P1
        self.rank_map = rank_map
        self.p_vocab = p_vocab
        self.pending = None

    def propose(self, prev):
        if self.pending is not None:
            raise RuntimeError('Previous proposal has not been finalized')
        i = self.n_finalized_positions - 1
        gs, zp, state, phase, mh, zh, ph = self.predictor.propose(prev)
        ids, invalid = ids_from_gaps(prev, gs, self.p_vocab, self.rank_map)
        if not ids:
            # No legal draft: still track the causal state for a Target fallback.
            past = self.predictor.gaps
            tri = tuple(past[-3:]) if len(past) >= 3 else None
            self.pending = (prev, state, phase, tri, i)
            return None  # EOS/terminal edge case; fall back to vanilla
        q = draft_weights(ids, prev, zp, mh, zh, ph)
        hist_id = max(ids, key=lambda c: (q[c], c == prev + 1, -c))
        ng, nz, nph = sum(mh.values()), sum(zh.values()), sum(ph.values())
        past = self.predictor.gaps
        last4 = [0] * max(0, 4 - len(past)) + list(past[-4:])
        tri = tuple(past[-3:]) if len(past) >= 3 else None
        t3 = self.m3[tri] if tri is not None else Counter()
        t3ph = self.m3_phase[(tri, phase)] if tri is not None else Counter()
        n3, n3ph = sum(t3.values()), sum(t3ph.values())
        qm = max(q.values())
        sorted_zh = topk(zh, 3)
        sorted_ph = topk(ph, 2)
        rows = []
        for c in ids:
            g = c - prev
            seen = self.last_seen.get((state, phase, g))
            age = i - seen if seen is not None else 1024
            rows.append([
                g, phase, (prev // 20) % 20, prev // 400, prev / 8000,
                i / 511, *last4,
                len(ids), int(g == 1), int(g == zp),
                int(g == mode(mh)), int(g in sorted_zh), int(g in sorted_ph),
                q[c], qm - q[c],
                zh[g], mh[g], ph[g],
                np.log1p(nz), np.log1p(ng), np.log1p(nph),
                zh[g] / nz if nz else 0.,
                mh[g] / ng if ng else 0.,
                ph[g] / nph if nph else 0.,
                t3[g] / n3 if n3 else 0.,
                t3ph[g] / n3ph if n3ph else 0.,
                np.log1p(n3ph), np.log1p(n3),
                min(age, 1024) / 1024,
            ])
        x = np.asarray(rows, dtype=np.float32)
        self.pending = (prev, state, phase, tri, i)
        return ids, x, hist_id, invalid

    def finalize(self, position):
        if self.pending is None:
            raise RuntimeError('No pending proposal to finalize')
        prev, state, phase, tri, i = self.pending
        gap = position - prev
        if not (0 <= position < self.p_vocab and int(self.rank_map[position]) > int(self.rank_map[prev])):
            raise AssertionError(f'Finalized illegal Position: {prev} -> {position}')
        if tri is not None:
            self.m3[tri][gap] += 1
            self.m3_phase[(tri, phase)][gap] += 1
        self.last_seen[(state, phase, gap)] = i
        self.predictor.observe(prev, gap)
        self.n_finalized_positions += 1
        self.pending = None

    def drop_eos(self):
        self.pending = None


@torch.inference_mode()
def sample(logits, rng, temp, top_p):
    probs = target_policy(logits, temp=temp, top_p=top_p)
    if not bool(torch.isfinite(probs).all()) or not abs(float(probs.sum().item()) - 1) < 1e-4:
        raise AssertionError('Non-normalized / nonfinite Target policy')
    return int(torch.multinomial(probs, 1, generator=rng).item())


@torch.inference_mode()
def choose_position(ranker, causal, prev):
    out = causal.propose(prev)
    if out is None:
        return None
    ids, feats, hist, invalid = out
    if len(ids) == 1:
        guess = ids[0]
    else:
        scores = ranker.predict_proba(feats)[:, 1]
        guess = ids[int(np.argmax(scores))]
    if guess not in ids:
        raise AssertionError('Ranker selected candidate outside Lean')
    return guess, len(ids), invalid


def rewind_one(cache, previous_size):
    """Keep F_(j-1), remove proposed P_j; K/V overwrite at the removed slot."""
    if int(cache.get_pos()) != previous_size + 2:
        raise AssertionError('KV cache is not at the expected verified-pair position')
    cache.cache_seqlens.fill_(previous_size + 1)
    cache.token_cache[:, previous_size + 1:previous_size + 2].fill_(-1)
    if int(cache.get_pos()) != previous_size + 1:
        raise AssertionError('KV rewind failed')


def make_rng(seed, device):
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    return rng


def one_token(token, device):
    return torch.tensor([[int(token)]], dtype=torch.long, device=device)


@torch.inference_mode()
def vanilla(transformer, make_cache, *, eos, temp, top_p, seed, max_tokens, device, p_vocab, ranks):
    cache = make_cache()
    rng = make_rng(seed, device)
    results = []
    stats = Counter()
    logits = transformer(one_token(eos, device), kv_cache=cache, logit_types=['position'])[0, 0]
    stats['forward_single'] += 1
    for i in range(max_tokens // 2):
        pos = sample(logits, rng, temp, top_p)
        if pos == eos:
            stats['eos'] += 1
            break
        if not 0 <= pos < p_vocab:
            raise AssertionError(f'Illegal position-type ID={pos}')
        if results and int(ranks[pos]) <= int(ranks[results[-2]]):
            raise AssertionError('Vanilla Position rank not monotonic')
        logits_f = transformer(one_token(pos, device), kv_cache=cache, logit_types=['feature'])[0, 0]
        stats['forward_single'] += 1
        f = sample(logits_f, rng, temp, top_p)
        if not (p_vocab <= f < eos):
            raise AssertionError(f'Illegal Feature ID={f}')
        results.extend([pos, f])
        if i + 1 < max_tokens // 2:
            logits = transformer(one_token(f, device), kv_cache=cache, logit_types=['position'])[0, 0]
            stats['forward_single'] += 1
    return results, stats


@torch.inference_mode()
def speculative(transformer, make_cache, ranker, *, eos, temp, top_p, seed, max_tokens, device, p_vocab, ranks):
    cache = make_cache()
    rng = make_rng(seed, device)
    stats = Counter()
    results = []
    # Bootstrap: after BOS, sample P0 and F0 in vanilla fashion.
    logits_p0 = transformer(one_token(eos, device), kv_cache=cache, logit_types=['position'])[0, 0]
    stats['forward_single'] += 1
    p0 = sample(logits_p0, rng, temp, top_p)
    if p0 == eos:
        stats['eos'] += 1
        return results, stats
    if not 0 <= p0 < p_vocab:
        raise AssertionError(f'Invalid P0={p0}')
    logits_f0 = transformer(one_token(p0, device), kv_cache=cache, logit_types=['feature'])[0, 0]
    stats['forward_single'] += 1
    f0 = sample(logits_f0, rng, temp, top_p)
    if not p_vocab <= f0 < eos:
        raise AssertionError(f'Invalid F0={f0}')
    results.extend([p0, f0])
    if max_tokens == 2:
        return results, stats

    online = CausalRanker(rank_map=ranks, p_vocab=p_vocab)
    pending_f, prev = f0, p0
    for j in range(1, max_tokens // 2):
        before = int(cache.get_pos())
        # When no rank-legal candidate exists, revert to the ordinary Target.
        guess = choose_position(ranker, online, prev)
        if guess is None:
            logits_pos = transformer(one_token(pending_f, device), kv_cache=cache, logit_types=['position'])[0, 0]
            stats['forward_single'] += 1
            pos = sample(logits_pos, rng, temp, top_p)
            stats['fallback'] += 1
        else:
            proposal, branches, invalid = guess
            assert branches >= 1
            stats['proposal'] += 1
            stats['branches_sum'] += branches
            stats['invalid_sum'] += invalid
            pair = torch.tensor([[pending_f, proposal]], device=device, dtype=torch.long)
            two = transformer(pair, kv_cache=cache, logit_types=['position', 'feature'])
            stats['forward_pair'] += 1
            if int(cache.get_pos()) != before + 2:
                raise AssertionError('Verified-pair cache advance is incorrect')
            p_dist = target_policy(two[0, 0], temp=temp, top_p=top_p)
            alpha = float(p_dist[proposal].item())
            stats['proposal_mass_sum'] += alpha
            # q is a point mass at proposal. Accept with p(proposal).
            u = float(torch.rand((), device=device, generator=rng).item())
            if u < alpha:
                pos = proposal
                feat_logits = two[0, 1]
                stats['accepted'] += 1
            else:
                # r(x)=(p(x)-min(p(x),q(x))) / (1-sum min(p,q)).
                # With deterministic q, remove proposal and renormalize.
                r = p_dist.clone()
                r[proposal] = 0.0
                mass = r.sum()
                if not bool(mass > 0):
                    raise AssertionError('Zero residual mass on a rejection path')
                r /= mass
                pos = int(torch.multinomial(r, 1, generator=rng).item())
                rewind_one(cache, before)
                stats['rejected'] += 1
                if pos == eos:
                    online.drop_eos()
                    stats['eos'] += 1
                    break
                feat_logits = transformer(one_token(pos, device), kv_cache=cache, logit_types=['feature'])[0, 0]
                stats['forward_single'] += 1
        if pos == eos:
            # Only fallback sampling can reach this branch, since proposal is Position.
            online.drop_eos()
            stats['eos'] += 1
            break
        if not 0 <= pos < p_vocab or int(ranks[pos]) <= int(ranks[prev]):
            raise AssertionError(f'Illegal / nonmonotonic finalized Position {prev}->{pos}')
        # At this point, the pair path already supplied logits conditional on P_j.
        # The fallback path needs to process newly sampled P_j.
        if guess is None:
            feat_logits = transformer(one_token(pos, device), kv_cache=cache, logit_types=['feature'])[0, 0]
            stats['forward_single'] += 1
        f = sample(feat_logits, rng, temp, top_p)
        if not p_vocab <= f < eos:
            raise AssertionError(f'Invalid Feature token={f}')
        results.extend([pos, f])
        online.finalize(pos)   # update only after Target has finalized the Position
        prev, pending_f = pos, f
    return results, stats


def validate_online_features(corpus_path, ranks, p_vocab, max_seq=2):
    """Compare streaming features against original CPU training extraction."""
    from eval_lean_ranker_cpu import build_examples
    data = torch.load(corpus_path, map_location='cpu', weights_only=True)
    examples, _, _ = build_examples(corpus_path)
    by_seq = defaultdict(list)
    for e in examples:
        if e['sid'] < max_seq:
            by_seq[e['sid']].append(e)
    tested = 0
    for sid in range(min(max_seq, len(data['val_sequences']))):
        seq = data['val_sequences'][sid].flatten().tolist()
        assert len(seq) == 1024 and len(seq) % 2 == 0, 'Feature test requires fixed 1024-token corpus'
        positions = seq[::2]
        s = CausalRanker(rank_map=ranks, p_vocab=p_vocab)
        for e in by_seq[sid]:
            assert e['prev'] == positions[e['i']]
            out = s.propose(e['prev'])
            assert out is not None
            ids, x, hist, invalid = out
            if ids != e['ids'] or hist != e['hist_id'] or not np.array_equal(x, e['features']):
                raise AssertionError(f'Feature parity failure at sid={sid}, i={e["i"]}: ids={ids} vs {e["ids"]}')
            s.finalize(e['actual'])
            tested += 1
    return tested


def selftest_residual():
    for vals in ([.1, .9], [.25, .25, .5], [0.0, 1.0, 0.0], [0.15, 0.0, 0.2, 0.65]):
        p = torch.tensor(vals, dtype=torch.float64)
        for proposed in range(len(vals)):
            alpha = p[proposed]
            if float(alpha) == 1.0:
                continue
            r = p.clone()
            r[proposed] = 0
            r /= r.sum()
            mix = (1. - alpha) * r
            mix[proposed] += alpha
            torch.testing.assert_close(mix, p, atol=1e-12, rtol=0)
    print('Residual point-mass identity: PASS', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ranker', type=Path, default=Path('profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib'))
    ap.add_argument('--corpus', type=Path, default=Path('profiling/target_eval/results/ranker_holdout_81001_81016/target_16x1024.pt'))
    ap.add_argument('--gpt', type=Path, default=Path('checkpoints/gpt_vfront.ckpt'))
    ap.add_argument('--vqvae', type=Path, default=Path('checkpoints/vqvae_vfront.ckpt'))
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--seed-start', type=int, default=91001)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--output-dir', type=Path, default=Path('profiling/target_eval/results/faithful_position_spec_smoke'))
    ap.add_argument('--skip-feature-parity', action='store_true')
    args = ap.parse_args()
    if args.max_tokens < 2 or args.max_tokens > 1024 or args.max_tokens % 2:
        ap.error('--max-tokens must be even and in [2, 1024]')
    if args.seeds < 1:
        ap.error('--seeds must be >= 1')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required for GaussianGPT Target')
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda')
    selftest_residual()

    print('Loading frozen pretrained ranker...', flush=True)
    ranker = load(args.ranker)
    print('Loading GaussianGPT FP32 Target...', flush=True)
    from model.gaussian_gpt import GaussianGPT
    from model.gaussian_vqvae import GaussianVQVAE
    from profiling.s3.s3_onpolicy_feature_distill_train import make_kv_cache
    vq = GaussianVQVAE.load_from_checkpoint(str(args.vqvae), map_location='cpu').eval()
    teacher = GaussianGPT.load_from_checkpoint(str(args.gpt), map_location='cpu', vqvae=vq).eval()
    teacher.gpt.to(device).eval()
    base = getattr(teacher.gpt, '_orig_mod', teacher.gpt)
    transformer = base.transformer
    p_vocab = int(base.position_vocab_size)
    eos = int(base.eos_token_id)
    ranks = transformer.sparse_position_rank_map.detach().cpu().long()
    if int(ranks.numel()) != p_vocab:
        raise AssertionError('Unexpected rank map')

    if not args.skip_feature_parity:
        print('Checking online Ranker feature parity against fixed corpus...', flush=True)
        n = validate_online_features(args.corpus, ranks, p_vocab, max_seq=2)
        print(f'Exact online/offline feature parity: PASS ({n} contexts)', flush=True)

    temp, nucleus = 0.9, 0.9
    cap = 1 + args.max_tokens + 8
    def make_cache():
        return make_kv_cache(transformer, cap, device)

    print('\n========== END-TO-END GENERATION ==========')
    rows, traces = [], []
    with torch.inference_mode(), threadpool_limits(limits=1, user_api='openmp'):
        # Warm both actual generation paths (including sparse 2-row projection).
        _ = vanilla(transformer, make_cache, eos=eos, temp=temp, top_p=nucleus,
                    seed=55001, max_tokens=8, device=device, p_vocab=p_vocab, ranks=ranks)
        _ = speculative(transformer, make_cache, ranker, eos=eos, temp=temp,
                        top_p=nucleus, seed=55002, max_tokens=8, device=device,
                        p_vocab=p_vocab, ranks=ranks)
        torch.cuda.synchronize()

        for i in range(args.seeds):
            seed = args.seed_start + i
            for name in ('vanilla', 'spec'):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if name == 'vanilla':
                    seq, stats = vanilla(transformer, make_cache, eos=eos, temp=temp, top_p=nucleus,
                                         seed=seed, max_tokens=args.max_tokens, device=device,
                                         p_vocab=p_vocab, ranks=ranks)
                else:
                    seq, stats = speculative(transformer, make_cache, ranker, eos=eos, temp=temp,
                                            top_p=nucleus, seed=seed, max_tokens=args.max_tokens,
                                            device=device, p_vocab=p_vocab, ranks=ranks)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
                positions = seq[::2]
                assert len(seq) % 2 == 0
                assert all(int(ranks[b]) > int(ranks[a]) for a, b in zip(positions[:-1], positions[1:]))
                row = {
                    'seed': seed, 'method': name, 'tokens': len(seq), 'seconds': seconds,
                    'tokens_per_s': len(seq)/seconds if seconds else 0.0,
                    **{k: float(v) for k, v in stats.items()},
                }
                rows.append(row)
                traces.append({'seed': seed, 'method': name, 'tokens': seq})
                acceptance = stats['accepted'] / stats['proposal'] if stats['proposal'] else 0
                expected_accept = stats['proposal_mass_sum'] / stats['proposal'] if stats['proposal'] else 0
                print(f"seed={seed} {name:<7} tokens={len(seq):4d} wall={seconds:.3f}s "
                      f"tok/s={row['tokens_per_s']:.2f} "
                      f"pair={stats['forward_pair']} single={stats['forward_single']} "
                      f"accept={acceptance:.2%} expected={expected_accept:.2%} "
                      f"reject={stats['rejected']} eos={stats['eos']}", flush=True)

    result = {'model': 'GaussianGPT FP32', 'max_tokens': args.max_tokens,
              'temperature': temp, 'top_p': nucleus, 'ranker': str(args.ranker),
              'seeds': args.seeds, 'runs': rows}
    for name in ('vanilla', 'spec'):
        r = [x for x in rows if x['method'] == name]
        result[name + '_aggregate'] = {
            'tokens': sum(x['tokens'] for x in r),
            'seconds': sum(x['seconds'] for x in r),
            'tokens_per_s': sum(x['tokens'] for x in r) / sum(x['seconds'] for x in r),
        }
    v = result['vanilla_aggregate']['tokens_per_s']
    s = result['spec_aggregate']['tokens_per_s']
    result['measured_speed_ratio'] = s/v if v else 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir/'summary.json', 'w') as f:
        json.dump(result, f, indent=2)
    torch.save({'runs': traces, 'max_tokens': args.max_tokens, 'temperature': temp,
                'top_p': nucleus}, args.output_dir/'samples.pt')
    print('\n========== AGGREGATE ==========')
    print(f"Vanilla: {v:.2f} tok/s")
    print(f"Spec:    {s:.2f} tok/s")
    print(f"Measured speed ratio (separate stochastic paths): {result['measured_speed_ratio']:.3f}x")
    print('NOTE: This is a correct-algorithm smoke/latency experiment, not yet a statistically powered')
    print('distribution-equivalence test. Sampling paths differ even with the same RNG seed.')
    print('Saved:', args.output_dir/'summary.json', args.output_dir/'samples.pt')


if __name__ == '__main__':
    main()
