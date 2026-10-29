#!/usr/bin/env python3
"""Evaluate frozen Lean-Hybrid position proposals and executable draft q.

Runs full-FP32 teacher forcing on on-policy GaussianGPT P/F trajectories. Each
position draft uses only the prefix's previous position, online gap counts and
spatial phase. Target logits are used ONLY for evaluation and non-deployable
oracle diagnostics, never to construct the proposed candidates or q.

This is not an end-to-end speculative decoder or a speed benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


def target_policy(logits: torch.Tensor, temp: float, top_p: float) -> torch.Tensor:
    if temp <= 0 or not (0 < top_p <= 1):
        raise ValueError('Invalid target temperature or top-p')
    z = logits.float() / temp
    if top_p < 1:
        vals, inds = torch.sort(z, descending=True)
        prob = vals.softmax(dim=-1)
        remove = prob.cumsum(dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        vals = vals.masked_fill(remove, float('-inf'))
        z = torch.full_like(z, float('-inf'))
        z.scatter_(0, inds, vals)
    return z.softmax(dim=-1)


def topk(counter, k):
    return [g for g, c in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0] != 1, kv[0]))[:k]]


def mode(counter, fallback=1):
    return topk(counter, 1)[0] if sum(counter.values()) >= 3 else fallback


class FrozenLean:
    def __init__(self):
        self.m2 = defaultdict(Counter)
        self.phase_m2 = defaultdict(Counter)
        self.phase_only = defaultdict(Counter)
        self.gaps = []

    def propose(self, prev):
        phase = prev % 20
        state = tuple(self.gaps[-2:]) if len(self.gaps) >= 2 else None
        mh = self.m2[state] if state is not None else Counter()
        zh = self.phase_m2[(state, phase)] if state is not None else Counter()
        ph = self.phase_only[phase]
        base = mode(mh)
        zp = mode(zh, base)
        core = [zp, 1, base] + topk(zh, 2)
        risk = (phase == 1 or (state == (1, 1) and phase in (12, 14)))
        sparse = sum(zh.values()) < 3
        gs = core + (topk(zh, 3) if risk else []) + (topk(ph, 2) if sparse else [])
        return gs, zp, state, phase, mh, zh, ph

    def observe(self, prev, gap):
        phase = prev % 20
        state = tuple(self.gaps[-2:]) if len(self.gaps) >= 2 else None
        if state is not None:
            self.m2[state][gap] += 1
            self.phase_m2[(state, phase)][gap] += 1
        self.phase_only[phase][gap] += 1
        self.gaps.append(gap)


def ids_from_gaps(prev, gaps, p_vocab, rank_map):
    ids = set()
    invalid = 0
    for gap in set(gaps):
        c = prev + gap
        if 0 <= c < p_vocab and int(rank_map[c]) > int(rank_map[prev]):
            ids.add(c)
        else:
            invalid += 1
    return sorted(ids), invalid


def draft_weights(ids, prev, zp, mh, zh, ph, strength=3.0, floor=0.02):
    """Causal hierarchical categorical q on Lean's candidate set.

    These fixed constants are *exploratory*, not trained and not Target-tuned.
    Backoff is a convex combination of first-order two-gap and phase-only
    empirical distributions; with no evidence favor gap=+1. The fine-state
    counts are combined with three pseudo-observations from the backoff.
    """
    nm, np = sum(mh.values()), sum(ph.values())
    nz = sum(zh.values())
    out = {}
    for c in ids:
        g = c - prev
        m_prior = mh[g] / nm if nm else float(g == 1)
        ph_prior = ph[g] / np if np else float(g == 1)
        prior = 0.5 * (m_prior + ph_prior)
        out[c] = zh[g] + strength * prior + floor
    denom = sum(out.values())
    assert denom > 0
    return {c: value / denom for c, value in out.items()}


def sharp_weights(q, exponent=4.0):
    w = {c: x**exponent for c, x in q.items()}
    den = sum(w.values())
    return {c: x / den for c, x in w.items()}


def overlap(p, q):
    return sum(min(float(p[c]), w) for c, w in q.items())


def combine_stats(rows):
    if not rows:
        return {}
    n = len(rows)
    keys = (
        'z_hit', 'det_hist_hit', 'lean_sample_hit', 'lean_oracle_recall',
        'z_accept', 'det_hist_accept', 'q_hist_accept', 'q_hist_sharp_accept',
        'lean_coverage', 'best_in_lean', 'full_pos_oracle',
        'lean_branches', 'lean_invalid', 'q_entropy', 'eos_mass',
    )
    return {'steps': n, **{k: sum(float(r[k]) for r in rows)/n for k in keys}}


def print_stats(label, stats):
    print(
        f'{label:<11} n={stats["steps"]:<5} '
        f'z={stats["z_accept"]:7.2%} '
        f'detHist={stats["det_hist_accept"]:7.2%} '
        f'qHist={stats["q_hist_accept"]:7.2%} '
        f'qSharp={stats["q_hist_sharp_accept"]:7.2%} '
        f'Coverage={stats["lean_coverage"]:7.2%} '
        f'BestInSet={stats["best_in_lean"]:7.2%} '
        f'Orcl={stats["full_pos_oracle"]:7.2%} '
        f'B={stats["lean_branches"]:.3f}',
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpt', type=Path, default=Path('checkpoints/gpt_vfront.ckpt'))
    ap.add_argument('--vqvae', type=Path, default=Path('checkpoints/vqvae_vfront.ckpt'))
    ap.add_argument('--corpus', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--max-sequences', type=int, default=0, help='0 evaluates all sequences; 1 for smoke test')
    args = ap.parse_args()

    dataset = torch.load(args.corpus, map_location='cpu', weights_only=True)
    assert dataset['format'] == 'teacher_generated_sparse_token_sequences_without_bos_or_eos'
    temp = float(dataset['generation_temperature'])
    nucleus = float(dataset['generation_top_p'])
    torch.set_float32_matmul_precision('highest')

    from model.gaussian_vqvae import GaussianVQVAE
    from model.gaussian_gpt import GaussianGPT

    print('Loading official FP32 Target checkpoints...', flush=True)
    vqvae = GaussianVQVAE.load_from_checkpoint(str(args.vqvae), map_location='cpu').eval()
    teacher = GaussianGPT.load_from_checkpoint(str(args.gpt), map_location='cpu', vqvae=vqvae).eval()
    teacher.gpt.to('cuda').eval()
    base = getattr(teacher.gpt, '_orig_mod', teacher.gpt)
    model = base.transformer
    p_vocab = int(base.position_vocab_size)
    eos = int(base.eos_token_id)
    rank_map = model.sparse_position_rank_map.detach().cpu().long()
    assert rank_map.numel() == p_vocab

    all_rows = []
    trimmed = 0
    sequence_list = dataset['val_sequences'][:args.max_sequences] if args.max_sequences > 0 else dataset['val_sequences']
    for sid, original in enumerate(sequence_list):
        seq = original.cpu().long().flatten()
        eos_locations = (seq == eos).nonzero(as_tuple=True)[0]
        if len(eos_locations):
            seq = seq[:int(eos_locations[0])]
            trimmed += 1
        if seq.numel() % 2:
            seq = seq[:-1]
            trimmed += 1
        if seq.numel() < 4:
            raise ValueError(f'Too few P/F pairs in sequence {sid}')
        positions = seq[::2].tolist()
        assert all(0 <= x < p_vocab for x in positions)
        assert all(int(rank_map[b]) > int(rank_map[a]) for a,b in zip(positions[:-1], positions[1:]))

        inp = torch.cat([torch.tensor([eos]), seq[:-1]]).unsqueeze(0).to('cuda')
        with torch.inference_mode():
            logits = model(inp)[0].float().cpu()
        assert logits.shape[0] == seq.numel()
        predictor = FrozenLean()
        local = []

        for i in range(len(positions)-1):
            prev, actual = positions[i], positions[i+1]
            gs, zp, state, phase, mh, zh, ph = predictor.propose(prev)
            ids, invalid = ids_from_gaps(prev, gs, p_vocab, rank_map)
            if not ids:
                raise RuntimeError(f'No valid Lean candidate in seq={sid}, step={i}')
            zid = prev + zp
            if zid not in ids:
                raise RuntimeError(f'Invalid phase-Z candidate in seq={sid}, step={i}')
            qhist = draft_weights(ids, prev, zp, mh, zh, ph)
            assert abs(sum(qhist.values())-1) < 1e-6
            qsharp = sharp_weights(qhist)
            # Deterministic argmax of cheap q, without any target access.
            det_id = max(ids, key=lambda c: (qhist[c], c == prev+1, -c))

            t = 2*(i+1)
            z = logits[t].clone()
            z[:p_vocab][rank_map <= rank_map[prev]] = -torch.inf
            p = target_policy(z, temp, nucleus)
            if not torch.isfinite(p).all() or abs(float(p.sum())-1) > 1e-4:
                raise RuntimeError(f'Invalid target probs at sid={sid}, t={t}')
            best_mass, best_id = p[:p_vocab].max(dim=0)

            import math
            entropy = -sum(qv*math.log(qv) for qv in qhist.values() if qv > 0)
            row = {
                'sid': sid, 'j': i+1, 'prev': prev, 'actual': actual,
                'phase': phase, 'state': str(state),
                'z_id': zid, 'det_hist_id': det_id, 'lean_size': len(ids),
                'z_hit': int(zid == actual),
                'det_hist_hit': int(det_id == actual),
                'lean_sample_hit': int(actual in ids),
                'lean_oracle_recall': int(int(best_id) in ids),
                'z_accept': float(p[zid]),
                'det_hist_accept': float(p[det_id]),
                'q_hist_accept': overlap(p, qhist),
                'q_hist_sharp_accept': overlap(p, qsharp),
                'lean_coverage': float(p[ids].sum()),
                'best_in_lean': float(p[ids].max()),
                'full_pos_oracle': float(best_mass),
                'lean_branches': len(ids),
                'lean_invalid': invalid,
                'q_entropy': entropy,
                'eos_mass': float(p[eos]),
            }
            local.append(row)
            all_rows.append(row)
            predictor.observe(prev, actual - prev)

        print_stats(f'seq={sid:02d}', combine_stats(local))

    result = {
        'corpus': str(args.corpus),
        'target_temperature': temp,
        'target_top_p': nucleus,
        'trimmed_sequences': trimmed,
        'note': 'FP32 target p teacher forcing; draft q is causal and Target-blind. No actual rejection-path rollout or speedup measured.',
        'draft_q': 'q_hist: scores zh[g] + 3*(0.5*m2_freq[g]+0.5*phase_only_freq[g]) + 0.02; normalized over legal Lean set, fallback +1. q_sharp: q_hist^4 normalized.',
        'overall': combine_stats(all_rows),
        'per_sequence': {str(i): combine_stats([r for r in all_rows if r['sid']==i]) for i in range(len(sequence_list))},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dest = args.output_dir / 'draft_steps.csv'
    with dest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    path = args.output_dir / 'summary.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print('\n========== FROZEN LEAN: TARGET-MATCHED ACCEPTANCE ==========', flush=True)
    for k,v in result['overall'].items():
        print(f'{k:<29} {v:.4%}' if isinstance(v,float) and 'branches' not in k and 'entropy' not in k and 'invalid' not in k else f'{k:<29} {v:.5f}' if isinstance(v,float) else f'{k:<29} {v}', flush=True)
    print('Wrote:', dest, path, flush=True)


if __name__ == '__main__':
    main()
