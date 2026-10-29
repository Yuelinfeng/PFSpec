#!/usr/bin/env python3
"""GaussianGPT K=2 P/F speculative decoding with COW ranker shadow.

Frozen Lean + Numba HGB drafts deterministic Position; a trained feature-state
predictor drafts stochastic Feature. All proposals are strictly verified by
GaussianGPT Target using standard speculative accept/reject + residual laws.

This is an end-to-end algorithm benchmark, NOT a distribution-level statistical
proof. Short smoke before a longer multi-seed run is mandatory for a new block
KV/rejection implementation.

Repo-root dependencies (existing): gpt_position_spec_decode_smoke.py,
bench_fast_hgb.py, train_feature_state_distill.py,
collect_feature_teacher_from_s3.py, eval_lean_candidate_to_draft.py.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from joblib import load
from threadpoolctl import threadpool_limits

import gpt_position_spec_decode_smoke as dec
from bench_fast_hgb import FastHistRanker
from collect_feature_teacher_from_s3 import policy_batch
from train_feature_state_distill import FeatureStateDraft


class CaptureLastBlock:
    """Hook last *block* state, before GaussianGPT's final norm/lm_head."""
    def __init__(self, gpt):
        self.output = None
        self.handle = gpt.transformer.h[-1].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        x = output[0] if isinstance(output, (list, tuple)) else output
        if not torch.is_tensor(x) or x.ndim != 3 or x.size(0) != 1:
            raise RuntimeError(f'Unexpected final-block shape: {getattr(x, "shape", None)}')
        self.output = x.detach()

    def run(self, gpt, cache, tokens, kinds, device):
        if len(tokens) != len(kinds):
            raise AssertionError('logit_types must match token count')
        self.output = None
        idx = torch.as_tensor([tokens], device=device, dtype=torch.long)
        logits = gpt(idx, kv_cache=cache, logit_types=kinds)
        if self.output is None or self.output.size(1) != len(tokens):
            raise AssertionError('No matching final-block state captured')
        return logits[0], self.output[0]

    def close(self):
        self.handle.remove()


def truncate(cache, keep_pos, old_pos):
    """Keep exactly first keep_pos processed tokens; future KV slots overwritten."""
    if not (0 <= keep_pos <= old_pos) or int(cache.get_pos()) != old_pos:
        raise AssertionError(f'Bad cache truncation {keep_pos=} {old_pos=} actual={cache.get_pos()}')
    cache.cache_seqlens.fill_(keep_pos)
    cache.token_cache[:, keep_pos:old_pos].fill_(-1)
    if int(cache.get_pos()) != keep_pos:
        raise AssertionError('KV cache truncated incorrectly')


def residual_sample(p, q, rng, *, atol=1e-8):
    """Exact speculative rejection residual: (p-q)_+ normalized."""
    r = torch.clamp(p.float() - q.float(), min=0)
    mass = float(r.sum().item())
    if mass <= atol:
        raise RuntimeError('Rejected a draft with zero residual mass; check policy alignment')
    return int(torch.multinomial(r / mass, 1, generator=rng).item())


def residual_point(p, proposed, rng):
    r = p.clone()
    r[proposed] = 0
    mass = r.sum()
    if not bool(mass > 0):
        raise AssertionError('Impossible deterministic Position rejection')
    return int(torch.multinomial(r / mass, 1, generator=rng).item())


def decide(alpha, rng, device):
    # Separate uniform draw exactly matches existing Position-only decoder.
    return bool(torch.rand((), device=device, generator=rng).item() < alpha)


def check_position(p, previous, ranks, p_vocab):
    if not (0 <= p < p_vocab and int(ranks[p]) > int(ranks[previous])):
        raise AssertionError(f'Illegal Position {previous}->{p}')


def sample_target(logits, rng, temp, top_p):
    return dec.sample(logits, rng, temp, top_p)


def fork_position_state(source: dec.CausalRanker) -> dec.CausalRanker:
    """Fork state for ONE previewed Position finalization.

    Both the original online ranker and preview may be used after the fork,
    but only one survives the accept/reject decision.  All mappings get
    independent dict storage; the few counters touched by finalize(p1) use
    independent Counter storage. Read-only counters are shared until this
    fork is either committed or discarded.
    """
    if not isinstance(source, dec.CausalRanker):
        raise TypeError(f'Expected CausalRanker, got {type(source)}')
    if source.pending is None:
        raise AssertionError('Fork requires a pending P1 proposal')

    shadow = copy.copy(source)
    shadow.predictor = copy.copy(source.predictor)
    shadow.last_seen = source.last_seen.copy()
    shadow.pending = source.pending  # tuple of immutable small values

    shadow.m3 = defaultdict(Counter, source.m3)
    shadow.m3_phase = defaultdict(Counter, source.m3_phase)
    predictor = shadow.predictor
    old_pred = source.predictor
    predictor.gaps = old_pred.gaps.copy()
    predictor.m2 = defaultdict(Counter, old_pred.m2)
    predictor.phase_m2 = defaultdict(Counter, old_pred.phase_m2)
    predictor.phase_only = defaultdict(Counter, old_pred.phase_only)

    _, state, phase, tri, _ = source.pending
    # In CausalRanker.finalize, the following two keys receive +1.
    if tri is not None:
        if tri in shadow.m3:
            shadow.m3[tri] = shadow.m3[tri].copy()
        key3 = (tri, phase)
        if key3 in shadow.m3_phase:
            shadow.m3_phase[key3] = shadow.m3_phase[key3].copy()
    # In FrozenLean.observe, the following 2+1 keys receive +1.
    if state is not None:
        if state in predictor.m2:
            predictor.m2[state] = predictor.m2[state].copy()
        key2 = (state, phase)
        if key2 in predictor.phase_m2:
            predictor.phase_m2[key2] = predictor.phase_m2[key2].copy()
    if phase in predictor.phase_only:
        predictor.phase_only[phase] = predictor.phase_only[phase].copy()

    return shadow


@torch.inference_mode()
def block2_decode(target, make_cache, ranker, feature_draft, *, eos, temp,
                  top_p, seed, max_tokens, device, p_vocab, ranks,
                  draft_temp, draft_top_p):
    """K=2: propose P_i, F_i, P_(i+1), verify 4 Target input slots.

    At loop entry, [BOS,...,P_(i-1)] is processed in KV; accepted
    F_(i-1) is NOT yet processed. h_prev is the saved state at P_(i-1).
    """
    assert max_tokens % 2 == 0 and max_tokens >= 2
    cache = make_cache()
    capture = CaptureLastBlock(target)
    rng = dec.make_rng(seed, device)
    stats = Counter()
    results = []
    try:
        # Bootstrapping: identical to current Position-only algorithm.
        logits, _ = capture.run(target, cache, [eos], ['position'], device)
        stats['forward_single'] += 1
        p0 = sample_target(logits[0], rng, temp, top_p)
        if p0 == eos:
            stats['eos'] += 1
            return results, stats
        if not 0 <= p0 < p_vocab:
            raise AssertionError(f'Unexpected P0={p0}')
        logits, h = capture.run(target, cache, [p0], ['feature'], device)
        stats['forward_single'] += 1
        f0 = sample_target(logits[0], rng, temp, top_p)
        if not p_vocab <= f0 < eos:
            raise AssertionError(f'Unexpected F0={f0}')
        results.extend([p0, f0])
        if max_tokens == 2:
            return results, stats
        h_prev = h[0].detach().clone()
        prev_p, pending_f = p0, f0
        online = dec.CausalRanker(rank_map=ranks, p_vocab=p_vocab)
        n_pairs = max_tokens // 2
        pair_index = 1
        while pair_index < n_pairs:
            before = int(cache.get_pos())
            proposal_1 = dec.choose_position(ranker, online, prev_p)
            can_block2 = proposal_1 is not None and pair_index + 1 < n_pairs
            shadow = None
            proposal_2 = None
            if can_block2:
                p1, branches1, invalid1 = proposal_1
                # Critical: preview the future Position using the *accepted*
                # speculative first Position, without altering live ranker.
                shadow = fork_position_state(online)
                shadow.finalize(p1)
                proposal_2 = dec.choose_position(ranker, shadow, p1)
                can_block2 = proposal_2 is not None

            if not can_block2:
                # Exact original Position-only K=1 fallback, also used when
                # near EOS/sequence boundary or second legal guess absent.
                stats['k1_steps'] += 1
                if proposal_1 is None:
                    logits, _ = capture.run(target, cache, [pending_f], ['position'], device)
                    stats['forward_single'] += 1
                    p1 = sample_target(logits[0], rng, temp, top_p)
                    stats['position_fallback'] += 1
                    if p1 == eos:
                        online.drop_eos()
                        stats['eos'] += 1
                        break
                    check_position(p1, prev_p, ranks, p_vocab)
                    logits, h = capture.run(target, cache, [p1], ['feature'], device)
                    stats['forward_single'] += 1
                    h_new = h[0].detach().clone()
                    f1 = sample_target(logits[0], rng, temp, top_p)
                else:
                    p1, _, _ = proposal_1
                    stats['position_1_proposals'] += 1
                    logits, h = capture.run(target, cache, [pending_f, p1],
                                            ['position', 'feature'], device)
                    stats['forward_pair'] += 1
                    if int(cache.get_pos()) != before + 2:
                        raise AssertionError('K1 KV length incorrect')
                    p_dist = dec.target_policy(logits[0], temp=temp, top_p=top_p)
                    alpha = float(p_dist[p1].item())
                    stats['position_1_mass_sum'] += alpha
                    if decide(alpha, rng, device):
                        stats['position_1_accepted'] += 1
                        h_new = h[1].detach().clone()
                        f1 = sample_target(logits[1], rng, temp, top_p)
                    else:
                        p1 = residual_point(p_dist, p1, rng)
                        truncate(cache, before + 1, before + 2)
                        stats['position_1_rejected'] += 1
                        if p1 == eos:
                            online.drop_eos()
                            stats['eos'] += 1
                            break
                        check_position(p1, prev_p, ranks, p_vocab)
                        logits, h = capture.run(target, cache, [p1], ['feature'], device)
                        stats['forward_single'] += 1
                        h_new = h[0].detach().clone()
                        f1 = sample_target(logits[0], rng, temp, top_p)
                if not (p_vocab <= f1 < eos):
                    raise AssertionError(f'Illegal F1={f1}')
                results.extend([p1, f1])
                online.finalize(p1)
                prev_p, pending_f, h_prev = p1, f1, h_new
                pair_index += 1
                continue

            # K=2: precompute F_i from EAGLE-inspired state draft.
            p1, _, _ = proposal_1
            p2, _, _ = proposal_2
            stats['k2_blocks'] += 1
            stats['position_1_proposals'] += 1
            stats['position_2_proposals'] += 1
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=device.type == 'cuda'):
                _, f_logits = feature_draft(
                    h_prev.unsqueeze(0),
                    torch.tensor([pending_f - p_vocab], device=device),
                    torch.tensor([p1], device=device),
                )
            # q_F is exactly the distribution from which the Feature draft
            # token is sampled AND used in acceptance/residual correction.
            q = policy_batch(f_logits.float(), draft_temp, draft_top_p)[0]
            f1_local = int(torch.multinomial(q, 1, generator=rng).item())
            f1_draft = p_vocab + f1_local
            stats['feature_proposals'] += 1

            logits, h = capture.run(target, cache,
                                    [pending_f, p1, f1_draft, p2],
                                    ['position', 'feature', 'position', 'feature'], device)
            stats['forward_block4'] += 1
            if int(cache.get_pos()) != before + 4:
                raise AssertionError('K2 KV length incorrect')

            # Position P_i: point-mass draft.
            p1_dist = dec.target_policy(logits[0], temp=temp, top_p=top_p)
            a1 = float(p1_dist[p1].item())
            stats['position_1_mass_sum'] += a1
            if not decide(a1, rng, device):
                corrected = residual_point(p1_dist, p1, rng)
                stats['position_1_rejected'] += 1
                truncate(cache, before + 1, before + 4)
                if corrected == eos:
                    online.drop_eos()
                    stats['eos'] += 1
                    break
                check_position(corrected, prev_p, ranks, p_vocab)
                logits_corr, h_corr = capture.run(target, cache, [corrected], ['feature'], device)
                stats['forward_single'] += 1
                f_corr = sample_target(logits_corr[0], rng, temp, top_p)
                h_new = h_corr[0].detach().clone()
                results.extend([corrected, f_corr])
                online.finalize(corrected)
                prev_p, pending_f, h_prev = corrected, f_corr, h_new
                pair_index += 1
                continue

            # Position P_i accepted. Previewed ranker has already finalized P_i
            # and contains a pending proposal for P_(i+1).
            stats['position_1_accepted'] += 1
            online = shadow
            h_p1 = h[1].detach().clone()
            pf = dec.target_policy(logits[1], temp=temp, top_p=top_p)[p_vocab:eos]
            a_f = float(min(1.0, (pf[f1_local] / q[f1_local].clamp_min(1e-30)).item()))
            stats['feature_alpha_sum'] += float(torch.minimum(pf, q).sum().item())
            stats['feature_checked'] += 1
            if not decide(a_f, rng, device):
                corrected_local = residual_sample(pf, q, rng)
                corrected_f = p_vocab + corrected_local
                truncate(cache, before + 2, before + 4)
                stats['feature_rejected'] += 1
                results.extend([p1, corrected_f])
                online.drop_eos()  # discard previewed next-Position proposal
                prev_p, pending_f, h_prev = p1, corrected_f, h_p1
                pair_index += 1
                continue

            stats['feature_accepted'] += 1
            results.extend([p1, f1_draft])

            # Position P_(i+1): point mass conditional on P_i and accepted F_i.
            p2_dist = dec.target_policy(logits[2], temp=temp, top_p=top_p)
            a2 = float(p2_dist[p2].item())
            stats['position_2_mass_sum'] += a2
            stats['position_2_checked'] += 1
            if decide(a2, rng, device):
                stats['position_2_accepted'] += 1
                final_p2 = p2
                h_p2 = h[3].detach().clone()
                final_f2 = sample_target(logits[3], rng, temp, top_p)
            else:
                stats['position_2_rejected'] += 1
                final_p2 = residual_point(p2_dist, p2, rng)
                truncate(cache, before + 3, before + 4)
                if final_p2 == eos:
                    online.drop_eos()
                    stats['eos'] += 1
                    break
                check_position(final_p2, p1, ranks, p_vocab)
                l_fix, h_fix = capture.run(target, cache, [final_p2], ['feature'], device)
                stats['forward_single'] += 1
                h_p2 = h_fix[0].detach().clone()
                final_f2 = sample_target(l_fix[0], rng, temp, top_p)

            check_position(final_p2, p1, ranks, p_vocab)
            if not p_vocab <= final_f2 < eos:
                raise AssertionError(f'Illegal F2={final_f2}')
            results.extend([final_p2, final_f2])
            online.finalize(final_p2)
            prev_p, pending_f, h_prev = final_p2, final_f2, h_p2
            pair_index += 2

        if len(results) % 2 or len(results) > max_tokens:
            raise AssertionError('Bad generated P/F length')
        if results:
            ps = results[::2]
            for x, y in zip(ps, ps[1:]):
                check_position(y, x, ranks, p_vocab)
        return results, stats
    finally:
        capture.close()


def load_feature_model(path, frozen_head, target, device, p_vocab, eos):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    head = torch.load(frozen_head, map_location='cpu', weights_only=True)
    config = ckpt['model_config']
    if (int(config['position_vocab']) != p_vocab or
        int(config['feature_vocab']) != eos - p_vocab or
        int(config['hidden_dim']) != target.lm_head.weight.size(1)):
        raise RuntimeError('Feature Draft config not compatible with current Target')
    weight = head['weight'].float()
    target_w = target.lm_head.weight[p_vocab:eos].detach().float().cpu()
    if not torch.allclose(weight, target_w, atol=0, rtol=0):
        raise RuntimeError('Frozen Feature Head differs from Target (wrong checkpoint?)')
    feat = FeatureStateDraft(
        weight, position_vocab=p_vocab, feature_vocab=eos - p_vocab,
        d_model=int(config['d_model']), emb_dim=int(config['emb_dim']),
        mlp_dim=int(config['mlp_dim']), target_softcap=float(config['target_softcap']),
    )
    feat.load_state_dict(ckpt['model_state'], strict=True)
    feat.to(device).eval().requires_grad_(False)
    return feat, float(ckpt['student_temperature']), float(ckpt['student_top_p'])


def math_selftest():
    # Simple independent correctness check of both speculative residual laws.
    g = torch.Generator(device='cpu').manual_seed(111)
    for p in (torch.tensor([0.2, 0.3, 0.5]), torch.tensor([0.05, 0.95, 0.] )):
        for c in range(3):
            a = float(p[c])
            if a == 1.:
                continue
            r = p.clone()
            r[c] = 0
            if float(r.sum()) > 0:
                r /= r.sum()
                mix = a * torch.nn.functional.one_hot(torch.tensor(c), 3).float() + (1 - a) * r
                torch.testing.assert_close(mix, p, atol=1e-6, rtol=0)
        for q in (torch.tensor([.3,.5,.2]), torch.tensor([.0,.7,.3])):
            accept = torch.minimum(p, q).sum()
            residual = torch.clamp(p - q, min=0)
            residual /= residual.sum()
            mixed = torch.minimum(p,q) + (1-accept)*residual
            torch.testing.assert_close(mixed,p,atol=1e-6,rtol=0)
    print('PASS: point-mass and stochastic-feature residual identities',flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranker',type=Path,default=Path(
        'profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib'))
    ap.add_argument('--gpt',type=Path,default=Path('checkpoints/gpt_vfront.ckpt'))
    ap.add_argument('--vqvae',type=Path,default=Path('checkpoints/vqvae_vfront.ckpt'))
    ap.add_argument('--feature-checkpoint',type=Path,default=Path(
        'profiling/feature_draft/state_distill_256_continue/feature_state_draft_best.pt'))
    ap.add_argument('--feature-head',type=Path,default=Path(
        'profiling/feature_draft/state_generation_256/states/frozen_feature_head.pt'))
    ap.add_argument('--seed-start',type=int,default=99001)
    ap.add_argument('--seeds',type=int,default=2)
    ap.add_argument('--draft-temperature',type=float,default=None,
                    help='Override Feature Draft temperature; Target policy unchanged')
    ap.add_argument('--draft-top-p',type=float,default=None,
                    help='Override Feature Draft top-p; Target policy unchanged')
    ap.add_argument('--max-tokens',type=int,default=128)
    ap.add_argument('--output-dir',type=Path,default=Path('profiling/feature_draft/pf_block2_smoke'))
    ap.add_argument('--selftest-only',action='store_true')
    ap.add_argument('--include-vanilla',action='store_true',
                    help='Benchmark all three: vanilla, Position-only, PF K=2')
    args = ap.parse_args()
    math_selftest()
    if args.selftest_only:
        return
    if args.max_tokens < 4 or args.max_tokens % 2 or args.max_tokens > 1024:
        ap.error('--max-tokens must be even and 4..1024')
    if args.seeds < 1 or not torch.cuda.is_available():
        ap.error('At least one seed and CUDA are required')
    for x in (args.ranker,args.gpt,args.vqvae,args.feature_checkpoint,args.feature_head):
        if not x.is_file():
            ap.error(f'Missing input: {x}')
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda')
    from model.gaussian_gpt import GaussianGPT
    from model.gaussian_vqvae import GaussianVQVAE
    from profiling.s3.s3_onpolicy_feature_distill_train import make_kv_cache
    print('Loading frozen Target + Numba ranker + Feature Draft...',flush=True)
    vq = GaussianVQVAE.load_from_checkpoint(str(args.vqvae),map_location='cpu').eval()
    teacher = GaussianGPT.load_from_checkpoint(str(args.gpt),map_location='cpu',vqvae=vq).eval()
    teacher.gpt.to(device).eval().requires_grad_(False)
    base = getattr(teacher.gpt,'_orig_mod',teacher.gpt)
    target = base.transformer
    eos,p_vocab = int(base.eos_token_id),int(base.position_vocab_size)
    ranks = target.sparse_position_rank_map.detach().cpu().long()
    original = load(args.ranker)
    ranker = FastHistRanker(original)
    feature, ft, ftp = load_feature_model(args.feature_checkpoint,args.feature_head,
                                          target,device,p_vocab,eos)
    if args.draft_temperature is not None:
        if args.draft_temperature <= 0:
            ap.error('--draft-temperature must be >0')
        ft = args.draft_temperature
    if args.draft_top_p is not None:
        if not (0 < args.draft_top_p <= 1):
            ap.error('--draft-top-p must be within (0,1]')
        ftp = args.draft_top_p
    print(f'Effective Feature Draft policy: temperature={ft:g} top_p={ftp:g}; '
          'Target policy: temperature=0.9 top_p=0.9', flush=True)
    capacity = 1 + args.max_tokens + 8
    def make_cache():
        return make_kv_cache(target,capacity,device)
    common = dict(eos=eos,temp=.9,top_p=.9,device=device,
                  p_vocab=p_vocab,ranks=ranks,max_tokens=args.max_tokens)
    records = []
    with torch.inference_mode(),threadpool_limits(limits=1,user_api='openmp'):
        # CUDA/Numba kernels warmed before timing; short warmup also exercises
        # the target's 4-input cached attention path.
        dec.speculative(target,make_cache,ranker,seed=88001,**dict(common,max_tokens=32))
        block2_decode(target,make_cache,ranker,feature,seed=88002,
                      draft_temp=ft,draft_top_p=ftp,**dict(common,max_tokens=32))
        torch.cuda.synchronize()
        for i in range(args.seeds):
            seed = args.seed_start+i
            if args.include_vanilla:
                methods = ('vanilla','position_only','pf_block2') if i%2==0 else ('pf_block2','position_only','vanilla')
            else:
                methods = ('position_only','pf_block2') if i%2==0 else ('pf_block2','position_only')
            for m in methods:
                torch.cuda.synchronize()
                t0=time.perf_counter()
                if m=='vanilla':
                    seq,st=dec.vanilla(target,make_cache,seed=seed,**common)
                elif m=='position_only':
                    seq,st=dec.speculative(target,make_cache,ranker,seed=seed,**common)
                else:
                    seq,st=block2_decode(target,make_cache,ranker,feature,seed=seed,
                                         draft_temp=ft,draft_top_p=ftp,**common)
                torch.cuda.synchronize()
                seconds=time.perf_counter()-t0
                ps=seq[::2]
                if len(seq)%2 or any(int(ranks[y])<=int(ranks[x]) for x,y in zip(ps,ps[1:])):
                    raise AssertionError('Generated illegal P/F sequence')
                fps=len(seq)/seconds if seconds else 0
                r=dict(seed=seed,method=m,tokens=len(seq),wall_s=seconds,tok_per_s=fps,
                       stats={k:float(v) for k,v in st.items()})
                records.append(r)
                feature_accept=st['feature_accepted']/st['feature_checked'] if st['feature_checked'] else 0.
                feature_ealpha=st['feature_alpha_sum']/st['feature_checked'] if st['feature_checked'] else 0.
                pos1_accept=st['position_1_accepted']/st['position_1_proposals'] if st['position_1_proposals'] else 0.
                print(f'seed={seed} {m:13s} tokens={len(seq):4d} t={seconds:.3f}s '
                      f'tok/s={fps:.2f} block4={st["forward_block4"]} '
                      f'pair={st["forward_pair"]} single={st["forward_single"]} '
                      f'pos1_accept={pos1_accept:.2%} F_accept={feature_accept:.2%} '
                      f'F_Ealpha={feature_ealpha:.2%}',flush=True)
    agg={}
    for name in (('vanilla','position_only','pf_block2') if args.include_vanilla
                 else ('position_only','pf_block2')):
        subset=[r for r in records if r['method']==name]
        agg[name]=dict(tokens=sum(r['tokens'] for r in subset),seconds=sum(r['wall_s'] for r in subset))
        agg[name]['tok_per_s']=agg[name]['tokens']/agg[name]['seconds']
    ratio=agg['pf_block2']['tok_per_s']/agg['position_only']['tok_per_s']
    print(f'AGGREGATE: position_only={agg["position_only"]["tok_per_s"]:.2f} tok/s; '
          f'pf_block2={agg["pf_block2"]["tok_per_s"]:.2f} tok/s; '
          f'pf_block2/position_only={ratio:.4f}x',flush=True)
    if args.include_vanilla:
        vanilla_tps=agg['vanilla']['tok_per_s']
        print(f'THREE_WAY: vanilla={vanilla_tps:.2f} tok/s, '
              f'position_only={agg["position_only"]["tok_per_s"]:.2f} tok/s, '
              f'pf_block2={agg["pf_block2"]["tok_per_s"]:.2f} tok/s; '
              f'position/vanilla={agg["position_only"]["tok_per_s"]/vanilla_tps:.4f}x; '
              f'pf/vanilla={agg["pf_block2"]["tok_per_s"]/vanilla_tps:.4f}x',flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'summary.json'
    path.write_text(json.dumps(dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                                     runs=records,aggregate=agg,pf_over_position=ratio,
                                     pf_over_vanilla=(agg['pf_block2']['tok_per_s']/agg['vanilla']['tok_per_s'])
                                     if args.include_vanilla else None),indent=2))
    print('Saved:',path,flush=True)

if __name__=='__main__':
    main()
