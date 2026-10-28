#!/usr/bin/env python3
"""Summarize seed-level uncertainty in recorded acceptance counts."""
import argparse
import json
from pathlib import Path
import numpy as np


def counts(row, kind):
    s = row['stats']
    first = s.get('feature_1_checked', 0) > 0
    if kind == 'f1_accept':
        return (s.get('feature_1_accepted' if first else 'feature_accepted', 0),
                s.get('feature_1_checked' if first else 'feature_checked', 0))
    if kind == 'f1_ealpha':
        return (s.get('feature_1_alpha_sum' if first else 'feature_alpha_sum', 0),
                s.get('feature_1_checked' if first else 'feature_checked', 0))
    pairs = {'f2_accept': ('feature_2_accepted', 'feature_2_checked'),
             'f2_ealpha': ('feature_2_alpha_sum', 'feature_2_checked'),
             'f2_reach': ('feature_2_checked', 'feature_1_checked'),
             'joint_success': ('feature_2_accepted', 'feature_1_checked')}
    num, den = pairs[kind]
    return s.get(num, 0), s.get(den, 0)


def paired_counts(old, new, kind):
    if [r['seed'] for r in old] != [r['seed'] for r in new]:
        raise ValueError('Different seed sets')
    oc = np.array([counts(r, kind) for r in old], dtype=float)
    nc = np.array([counts(r, kind) for r in new], dtype=float)
    draw = np.random.default_rng(164039).integers(0, len(old), size=(5000, len(old)))
    ob, nb = oc[draw].sum(1), nc[draw].sum(1)
    keep = (ob[:, 1] > 0) & (nb[:, 1] > 0)
    delta = nb[keep, 0] / nb[keep, 1] - ob[keep, 0] / ob[keep, 1]
    old_rate = oc[:, 0].sum() / oc[:, 1].sum()
    new_rate = nc[:, 0].sum() / nc[:, 1].sum()
    return {'parent': float(old_rate), 'joint': float(new_rate), 'delta': float(new_rate - old_rate),
            'seed_boot_ci95_delta': np.quantile(delta, [.025, .975]).tolist(),
            'bootstrap_repetitions_used': int(keep.sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result_dir', type=Path)
    a = p.parse_args()
    data = json.loads((a.result_dir / 'summary.json').read_text())
    records = data['records']
    output = {'comparisons': {}, 'lengths': {}, 'unit': 'Seed-level paired resampling; rate equals pooled numerator / denominator.'}
    for method in data['aggregates']:
        rows = [r for r in records if r['method'] == method]
        output['lengths'][method] = {'seeds': len(rows), 'tokens': sum(r['tokens'] for r in rows),
                                     'min_tokens': min(r['tokens'] for r in rows),
                                     'max_tokens': max(r['tokens'] for r in rows),
                                     'eos_count': sum(int(r['stats'].get('eos', 0) > 0) for r in rows),
                                     'at_length_limit': sum(r['tokens'] == data['config']['max_tokens'] for r in rows)}
    for k in ['k2', 'k3']:
        old = sorted([r for r in records if r['method'] == 'parent_' + k], key=lambda r: r['seed'])
        new = sorted([r for r in records if r['method'] == 'joint_' + k], key=lambda r: r['seed'])
        kinds = ['f1_accept', 'f1_ealpha'] + (['f2_accept', 'f2_ealpha', 'f2_reach', 'joint_success'] if k == 'k3' else [])
        output['comparisons'][k] = {kind: paired_counts(old, new, kind) for kind in kinds}
    target = a.result_dir / 'acceptance_uncertainty.json'
    target.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__': main()
