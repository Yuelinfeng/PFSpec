"""Summarize G3 online runs with source-level paired intervals."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'docs/project/artifacts/gap_selection_20261017'


def load_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def geomean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.log(values).mean()))


def paired_interval(rows, seed=20261017, repeats=10000):
    source_method = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for row in rows:
        pair = source_method[row['scene']][row['method']]
        pair[0] += row['actual_new_tokens']
        pair[1] += row['wall_seconds']
    speeds = {
        source: {method: tokens / seconds for method, (tokens, seconds) in methods.items()}
        for source, methods in source_method.items()
    }
    source_ratios = {
        source: methods['gap'] / methods['pf']
        for source, methods in speeds.items()
        if 'gap' in methods and 'pf' in methods
    }
    sources = sorted(source_ratios)
    values = np.asarray([source_ratios[source] for source in sources], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(repeats, len(values)))
    estimates = np.exp(np.log(values[draws]).mean(axis=1))
    overall = {}
    methods_present = [method for method in ('target', 'p_only', 'pf', 'gap')
                       if any(row['method'] == method for row in rows)]
    for method in methods_present:
        tokens = sum(row['actual_new_tokens'] for row in rows if row['method'] == method)
        seconds = sum(row['wall_seconds'] for row in rows if row['method'] == method)
        method_sources = {
            source: value[0] / value[1]
            for source, methods in source_method.items()
            if method in methods for value in [methods[method]]
        }
        method_ratios = {
            source: method_sources[source] / speeds[source]['pf']
            for source in method_sources if 'pf' in speeds[source]
        }
        overall[method] = {
            'generated_tokens': tokens,
            'wall_seconds': seconds,
            'micro_tokens_per_second': tokens / seconds,
            'source_geomean_over_pf': geomean(list(method_ratios.values())),
            'source_wins_over_pf': int(sum(value > 1 for value in method_ratios.values())),
        }
    route_counts = Counter()
    branch_counts = Counter()
    gap_counts = Counter()
    feature_proposed = feature_verified = feature_accepted = 0
    for row in rows:
        if row['method'] != 'gap':
            continue
        selection = row['stats'].get('selection', {})
        route_counts.update(selection.get('counts', {}))
        branch_counts.update(selection.get('branches', {}))
        feature_proposed += row['stats'].get('feature_proposed', 0) or 0
        feature_verified += row['stats'].get('feature_verified', 0) or 0
        feature_accepted += row['stats'].get('feature_accepted', 0) or 0
        counts = selection.get('counts', {})
        gap_counts['gap_one'] += counts.get('unit_gap', 0)
        gap_counts['gap_above_one'] += counts.get('other_gap', 0)
    ratio = geomean(values)
    ci = [float(x) for x in np.quantile(estimates, [0.025, 0.975])]
    if ratio >= 1.02 and ci[0] > 1:
        decision = 'go_to_G4_G5'
    elif ratio > 1:
        decision = 'inconclusive_positive_point'
    else:
        decision = 'stop_after_G3'
    return {
        'run_count': len(rows),
        'source_count': len(source_method),
        'context_count': len({row['context_id'] for row in rows}),
        'method_totals': overall,
        'gap_over_pf': {
            'source_geomean': ratio,
            'ci95_source_bootstrap': ci,
            'source_wins': int(sum(value > 1 for value in values)),
            'source_count': len(sources),
            'source_ratios': source_ratios,
        },
        'gap_over_pf_micro_ratio': overall['gap']['micro_tokens_per_second'] / overall['pf']['micro_tokens_per_second'],
        'gap_over_p_only_micro_ratio': overall['gap']['micro_tokens_per_second'] / overall['p_only']['micro_tokens_per_second'],
        'gap_route_counts': dict(route_counts),
        'gap_branch_counts': dict(branch_counts),
        'gap_category_counts': dict(gap_counts),
        'gap_feature_acceptance_rate': feature_accepted / feature_verified if feature_verified else None,
        'gap_feature_proposals': feature_proposed,
        'gap_feature_verified': feature_verified,
        'gap_feature_accepted': feature_accepted,
        'eos': {
            method: {
                'runs': sum(row['method'] == method for row in rows),
                'terminations': sum(row['method'] == method and row['terminated_on_eos'] for row in rows),
                'mean_new_tokens': float(np.mean([
                    row['actual_new_tokens'] for row in rows if row['method'] == method
                ])),
            }
            for method in methods_present
        },
        'decision': decision,
    }


def validate_rows(rows, expected_count, allowed_methods, expected_budget):
    assert len(rows) == expected_count, (len(rows), expected_count)
    keys = set()
    for row in rows:
        assert row['method'] in allowed_methods
        key = (row['context_id'], row['seed_base'], row['method'])
        assert key not in keys, key
        keys.add(key)
        assert row['requested_new_tokens'] == expected_budget
        assert row['actual_new_tokens'] == len(row['output'])
        assert 0 < row['actual_new_tokens'] <= expected_budget
        assert row['stats']['generated_tokens'] == row['actual_new_tokens']
        assert row['stats']['eos'] == row['terminated_on_eos']
        assert row['wall_seconds'] > 0


def main():
    raw = ARTIFACTS / 'g3_raw'
    online = load_jsonl(raw / 'online_512.jsonl')
    long_rows = load_jsonl(raw / 'long_2048.jsonl')
    validate_rows(online, 672, {'target', 'p_only', 'pf', 'gap'}, 512)
    validate_rows(long_rows, 90, {'p_only', 'pf', 'gap'}, 2048)
    assert {row['context_id'] for row in online} == set(range(56))
    assert len({row['scene'] for row in online}) == 10
    assert len({row['context_id'] for row in long_rows}) == 10
    assert len({row['scene'] for row in long_rows}) == 10
    summary = {
        'status': 'complete',
        'primary_metric': 'gap-over-PF, source-cluster bootstrap on token throughput',
        'online_512': paired_interval(online),
        'long_2048': paired_interval(long_rows),
        'limitations': [
            'The 56 development prefixes share 10 sources; intervals use the source as the resampling unit.',
            'Different routes consume random numbers differently; sequence equality across methods is not expected.',
            'EOS and generated lengths are reported because early termination changes work and elapsed time.',
            'These are token-generation measurements, not complete 3D output measurements.',
        ],
        'raw_online': 'g3_raw/online_512.jsonl',
        'raw_long': 'g3_raw/long_2048.jsonl',
    }
    (ARTIFACTS / 'G3_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({
        'status': summary['status'],
        'online_gap_over_pf': summary['online_512']['gap_over_pf']['source_geomean'],
        'online_ci95': summary['online_512']['gap_over_pf']['ci95_source_bootstrap'],
        'online_decision': summary['online_512']['decision'],
        'long_gap_over_pf': summary['long_2048']['gap_over_pf']['source_geomean'],
        'long_decision': summary['long_2048']['decision'],
    }, indent=2))


if __name__ == '__main__':
    main()
