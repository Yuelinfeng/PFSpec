#!/usr/bin/env python3
"""CPU-only exploratory candidate ranker, trained on GaussianGPT Position samples.

Keep FrozenLean candidate generation unchanged. Training input: Target-sampled
on-policy Position/Feature token sequences. A candidate feature vector is built
from history STRICTLY BEFORE the sampled next Position is revealed.

This is a sampled-hit experiment, not exact Draft/Target overlap or speed.
Numerical position-bound screening only; validation reproduces the previously
rank-map-checked candidate size / baseline ID from draft_steps.csv.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch

from eval_lean_candidate_to_draft import FrozenLean, draft_weights, mode, topk


def build_examples(corpus: Path):
    data = torch.load(corpus, map_location='cpu', weights_only=True)
    assert data['format'] == 'teacher_generated_sparse_token_sequences_without_bos_or_eos'
    examples = []  # one context: features, candidates, actual, hist_choice, metadata
    multi_X = []
    multi_y = []
    for sid, tensor in enumerate(data['val_sequences']):
        tokens = tensor.long().flatten().tolist()
        if 12096 in tokens:
            tokens = tokens[:tokens.index(12096)]
        if len(tokens) % 2:
            tokens = tokens[:-1]
        positions = tokens[::2]
        predictor = FrozenLean()
        m3_phase = defaultdict(Counter)
        m3 = defaultdict(Counter)
        last_seen = {}
        for i, (prev, actual) in enumerate(zip(positions, positions[1:])):
            gs, zp, state, phase, mh, zh, ph = predictor.propose(prev)
            ids = sorted({prev + g for g in gs if 0 <= prev + g < 8000 and g > 0})
            assert ids, (sid, i, prev)
            q = draft_weights(ids, prev, zp, mh, zh, ph)
            hist_choice = max(ids, key=lambda c: (q[c], c == prev+1, -c))
            ng = sum(mh.values())
            nz = sum(zh.values())
            np_ = sum(ph.values())
            past = predictor.gaps
            last4 = list(past[-4:])
            last4 = [0]*(4-len(last4)) + last4
            tri = tuple(past[-3:]) if len(past) >= 3 else None
            t3 = m3[tri] if tri is not None else Counter()
            t3ph = m3_phase[(tri, phase)] if tri is not None else Counter()
            n3, n3ph = sum(t3.values()), sum(t3ph.values())
            qm = max(q.values())
            sorted_zh = topk(zh, 3)
            sorted_ph = topk(ph, 2)
            features = []
            for c in ids:
                g = c-prev
                seen = last_seen.get((state, phase, g))
                age = (i-seen) if seen is not None else 1024
                features.append([
                    g, phase, (prev//20)%20, prev//400, prev/8000,
                    i/max(1,len(positions)-1), *last4,
                    len(ids), int(g==1), int(g==zp),
                    int(g==mode(mh)), int(g in sorted_zh), int(g in sorted_ph),
                    q[c], qm-q[c],
                    zh[g], mh[g], ph[g],
                    np.log1p(nz), np.log1p(ng), np.log1p(np_),
                    zh[g]/nz if nz else 0.,
                    mh[g]/ng if ng else 0.,
                    ph[g]/np_ if np_ else 0.,
                    t3[g]/n3 if n3 else 0.,
                    t3ph[g]/n3ph if n3ph else 0.,
                    np.log1p(n3ph), np.log1p(n3),
                    min(age,1024)/1024,
                ])
            x = np.asarray(features, dtype=np.float32)
            if len(ids) > 1:
                multi_X.extend(features)
                multi_y.extend(int(c==actual) for c in ids)
            examples.append({
                'sid': sid, 'i': i, 'prev': prev, 'actual': actual,
                'phase': phase, 'state': str(state), 'ids': ids,
                'hist_id': hist_choice, 'features': x,
            })
            # Current sample is only revealed to the predictor AFTER construction.
            actual_gap = actual - prev
            if tri is not None:
                m3[tri][actual_gap] += 1
                m3_phase[(tri, phase)][actual_gap] += 1
            last_seen[(state, phase, actual_gap)] = i
            predictor.observe(prev, actual_gap)
    return examples, np.asarray(multi_X, dtype=np.float32), np.asarray(multi_y, dtype=np.int64)


def assess(examples, model, check_csv=None):
    expected = None
    if check_csv is not None:
        with open(check_csv, newline='') as f:
            expected = {(int(r['sid']), int(r['j'])-1): r for r in csv.DictReader(f)}
    total = Counter()
    by_seq = defaultdict(Counter)
    by_subset = defaultdict(Counter)
    for item in examples:
        ids, actual, hist = item['ids'], item['actual'], item['hist_id']
        if expected is not None:
            r = expected[(item['sid'],item['i'])]
            assert len(ids) == int(r['lean_size']), (item['sid'],item['i'],'candidate mismatch')
            assert hist == int(r['det_hist_id']), (item['sid'],item['i'],'baseline mismatch')
        if len(ids) == 1:
            chosen = ids[0]
        else:
            scores = model.predict_proba(item['features'])[:,1]
            chosen = ids[int(np.argmax(scores))]
        correct_hist = int(hist == actual)
        correct_model = int(chosen == actual)
        oracle_hit = int(actual in ids)
        for s in (total, by_seq[item['sid']]):
            s['n'] += 1
            s['hist'] += correct_hist
            s['model'] += correct_model
            s['coverage'] += oracle_hit
        subset = 'phase1_m2_6_1' if item['phase']==1 and item['state']=='(6, 1)' else 'phase1_m2_5_1' if item['phase']==1 and item['state']=='(5, 1)' else 'others'
        s = by_subset[subset]
        s['n'] += 1
        s['hist'] += correct_hist
        s['model'] += correct_model
        s['coverage'] += oracle_hit
    return total, by_seq, by_subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=Path, default=Path('profiling/target_eval/results/phase_z_holdout/target_16x1024.pt'))
    ap.add_argument('--test', type=Path, default=Path('profiling/target_eval/results/lean_hybrid_holdout/target_16x1024.pt'))
    ap.add_argument('--check-csv', type=Path, default=Path('profiling/target_eval/results/lean_draft_71001_71016/draft_steps.csv'))
    args = ap.parse_args()
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as e:
        raise SystemExit('Install scikit-learn in your gaussiangpt environment: pip install scikit-learn') from e
    print('Building training examples (strictly causal)...', flush=True)
    tr, X, y = build_examples(args.train)
    print('Train contexts:',len(tr),'ambiguous candidate rows:',len(y),'positive fraction:',round(float(y.mean()),4), flush=True)
    print('Building test examples...', flush=True)
    te, _, _ = build_examples(args.test)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=150, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=10.,
        early_stopping=False, random_state=0,
    )
    model.fit(X,y)
    all_, seq, subset = assess(te,model,args.check_csv if args.check_csv.exists() else None)
    assert all_['n']==8176, f'Expected 8176 validation steps; got {all_["n"]}'
    print('\n========== CPU LEAN RANKER EVALUATION ==========')
    for label,s in [('Baseline Hist',all_)]:
        print('Steps:',s['n'])
        print('History hit:',f"{s['hist']/s['n']:.4%}")
        print('Learned hit:',f"{s['model']/s['n']:.4%}")
        print('Set coverage:',f"{s['coverage']/s['n']:.4%}")
        print('Improvement:',f"{(s['model']-s['hist'])/s['n']:+.4%}")
    print('\n========== HIGH-REGRET SUBSETS ==========')
    print(f"{'Subset':<19} {'N':>6} {'Hist':>10} {'Learned':>10} {'Coverage':>10}")
    for key,s in sorted(subset.items()):
        print(f"{key:<19} {s['n']:>6} {s['hist']/s['n']:>10.2%} {s['model']/s['n']:>10.2%} {s['coverage']/s['n']:>10.2%}")
    print('\n========== PER-TRAJECTORY ==========')
    for sid,s in sorted(seq.items()):
        print(f"{71001+sid}: hist={s['hist']/s['n']:.2%} learned={s['model']/s['n']:.2%} gain={(s['model']-s['hist'])/s['n']:+.2%}")
    print('\nExploratory only: test seeds were previously inspected. Use fresh seeds for final evaluation.')
    print('This measures sampled prediction hit, NOT exact target probability/acceptance or speedup.')

if __name__=='__main__':
    main()
