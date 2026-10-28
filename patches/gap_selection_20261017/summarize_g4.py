"""Summarize G4 ablations against the paired completed G3 measurements."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'docs/project/artifacts/gap_selection_20261017'


def load_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def throughput_by_source(rows, method):
    totals = defaultdict(lambda: [0, 0.0])
    for row in rows:
        if row['method'] == method:
            totals[row['scene']][0] += row['actual_new_tokens']
            totals[row['scene']][1] += row['wall_seconds']
    return {scene: tokens / seconds for scene, (tokens, seconds) in totals.items()}


def compare(left_rows, left_method, right_rows, right_method, seed):
    left = throughput_by_source(left_rows, left_method)
    right = throughput_by_source(right_rows, right_method)
    assert set(left) == set(right), (left_method, right_method, set(left), set(right))
    sources = sorted(left)
    ratios = np.asarray([left[source] / right[source] for source in sources], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ratios), size=(10000, len(ratios)))
    estimates = np.exp(np.log(ratios[draws]).mean(axis=1))
    return {
        'source_geomean': float(np.exp(np.log(ratios).mean())),
        'ci95_source_bootstrap': [float(value) for value in np.quantile(estimates, [0.025, 0.975])],
        'source_wins': int(sum(value > 1 for value in ratios)),
        'source_count': len(sources),
        'source_ratios': {source: float(left[source] / right[source]) for source in sources},
    }


def validate_rows(rows, expected_count, expected_budget, expected_methods):
    assert len(rows) == expected_count, (len(rows), expected_count)
    keys = set()
    for row in rows:
        assert row['method'] in expected_methods
        key = (row['context_id'], row['seed_base'], row['method'])
        assert key not in keys, key
        keys.add(key)
        assert row['requested_new_tokens'] == expected_budget
        assert row['actual_new_tokens'] == len(row['output'])
        assert 0 < row['actual_new_tokens'] <= expected_budget
        assert row['stats']['generated_tokens'] == row['actual_new_tokens']
        assert row['stats']['eos'] == row['terminated_on_eos']
        assert row['wall_seconds'] > 0
        assert row['stats']['selection']['mode'] == row['method']


def verify_overhead_matches_pf(overhead_rows, pf_rows):
    fixed = {
        (row['context_id'], row['seed_base']): row
        for row in pf_rows if row['method'] == 'pf'
    }
    for row in overhead_rows:
        reference = fixed[(row['context_id'], row['seed_base'])]
        assert row['output'] == reference['output']
        keys = set(row['stats']) - {'selection', 'seconds', 'tokens_per_second'}
        assert {key: row['stats'][key] for key in keys} == {
            key: reference['stats'][key] for key in keys
        }


def verify_pf_matches_pf(pf_rows, reference_rows):
    reference = {
        (row['context_id'], row['seed_base']): row
        for row in reference_rows if row['method'] == 'pf'
    }
    for row in pf_rows:
        paired = reference[(row['context_id'], row['seed_base'])]
        assert row['output'] == paired['output']
        keys = set(row['stats']) - {'selection', 'seconds', 'tokens_per_second'}
        assert {key: row['stats'][key] for key in keys} == {
            key: paired['stats'][key] for key in keys
        }


def route_totals(rows, method):
    counts = Counter()
    for row in rows:
        if row['method'] == method:
            counts.update(row['stats']['selection']['counts'])
    pf = counts['executed_pf']
    p_only = counts['executed_p_only']
    return {
        'counts': dict(counts),
        'actual_pf_fraction': pf / (pf + p_only) if pf + p_only else None,
        'feature_acceptance_rate': sum(
            row['stats'].get('feature_accepted', 0) or 0
            for row in rows if row['method'] == method
        ) / max(1, sum(
            row['stats'].get('feature_verified', 0) or 0
            for row in rows if row['method'] == method
        )),
    }


def method_totals(rows, method):
    selected = [row for row in rows if row['method'] == method]
    tokens = sum(row['actual_new_tokens'] for row in selected)
    seconds = sum(row['wall_seconds'] for row in selected)
    return {
        'runs': len(selected),
        'generated_tokens': tokens,
        'wall_seconds': seconds,
        'micro_tokens_per_second': tokens / seconds,
        'mean_actual_tokens': float(np.mean([row['actual_new_tokens'] for row in selected])),
        'eos_terminations': sum(row['terminated_on_eos'] for row in selected),
    }


def summarize_window(g4_rows, g3_rows, budget, expected_contexts, expected_sources,
                    expected_runs, p_dev, seed):
    methods = {'pf', 'random', 'reverse_gap', 'overhead'}
    validate_rows(g4_rows, expected_runs, budget, methods)
    assert len({row['context_id'] for row in g4_rows}) == expected_contexts
    assert len({row['scene'] for row in g4_rows}) == expected_sources
    assert {row['method'] for row in g4_rows} == methods
    by_method = {method: method_totals(g4_rows, method) for method in methods}
    routes = {method: route_totals(g4_rows, method) for method in methods}
    random_probabilities = {row['probability'] for row in g4_rows if row['method'] == 'random'}
    assert random_probabilities == {p_dev}

    g3_by_method = {method: [row for row in g3_rows if row['method'] == method]
                    for method in {'gap', 'p_only'}}
    pf_rows = [row for row in g4_rows if row['method'] == 'pf']
    overhead_rows = [row for row in g4_rows if row['method'] == 'overhead']
    g3_pf_rows = [row for row in g3_rows if row['method'] == 'pf']
    verify_overhead_matches_pf(overhead_rows, pf_rows)
    verify_pf_matches_pf(pf_rows, g3_pf_rows)

    comparisons = {
        'gap_over_pf': compare(g3_by_method['gap'], 'gap', g4_rows, 'pf', seed),
        'gap_over_random': compare(g3_by_method['gap'], 'gap', g4_rows, 'random', seed + 1),
        'random_over_pf': compare(g4_rows, 'random', g4_rows, 'pf', seed + 2),
        'reverse_over_pf': compare(g4_rows, 'reverse_gap', g4_rows, 'pf', seed + 3),
        'overhead_over_pf': compare(g4_rows, 'overhead', g4_rows, 'pf', seed + 4),
        'reverse_over_gap': compare(g4_rows, 'reverse_gap', g3_by_method['gap'], 'gap', seed + 5),
    }
    target_pf = routes['random']['actual_pf_fraction']
    gap_pf = route_totals(g3_rows, 'gap')['actual_pf_fraction']
    return {
        'budget_tokens': budget,
        'context_count': expected_contexts,
        'source_count': expected_sources,
        'run_count': expected_runs,
        'p_dev': p_dev,
        'methods': by_method,
        'route_use': routes,
        'random_minus_gap_pf_fraction': target_pf - gap_pf,
        'comparisons': comparisons,
        'overhead_output_and_token_counts_match_fixed_pf': True,
    }


def main():
    raw = ARTIFACTS / 'g4_raw'
    g3raw = ARTIFACTS / 'g3_raw'
    short = load_jsonl(raw / 'ablation_512.jsonl')
    long = load_jsonl(raw / 'ablation_2048.jsonl')
    g3short = load_jsonl(g3raw / 'online_512.jsonl')
    g3long = load_jsonl(g3raw / 'long_2048.jsonl')
    g3summary = json.loads((ARTIFACTS / 'G3_summary.json').read_text())
    summary = {
        'status': 'complete',
        'purpose': 'attribute the gap rule gain to spatial selection, random Feature use, reverse assignment, or route decision cost',
        'online_512': summarize_window(
            short, g3short, 512, 56, 10, 672,
            g3summary['online_512']['gap_route_counts']['executed_pf'] /
            (g3summary['online_512']['gap_route_counts']['executed_pf'] +
             g3summary['online_512']['gap_route_counts']['executed_p_only']),
            20261018,
        ),
        'long_2048': summarize_window(
            long, g3long, 2048, 10, 10, 120,
            g3summary['long_2048']['gap_route_counts']['executed_pf'] /
            (g3summary['long_2048']['gap_route_counts']['executed_pf'] +
             g3summary['long_2048']['gap_route_counts']['executed_p_only']),
            20261019,
        ),
        'limitations': [
            'These ablations reuse D0 sources and explain the development signal; they are not a new-source confirmation.',
            'Random probability is set from the completed G3 frequency before G4 and is not adjusted from G4 or T outcomes.',
        ],
        'raw_512': 'g4_raw/ablation_512.jsonl',
        'raw_2048': 'g4_raw/ablation_2048.jsonl',
    }
    path = ARTIFACTS / 'G4_summary.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({
        'status': summary['status'],
        '512_gap_over_random': summary['online_512']['comparisons']['gap_over_random'],
        '512_pf_rate_gap': summary['online_512']['random_minus_gap_pf_fraction'],
        '2048_gap_over_random': summary['long_2048']['comparisons']['gap_over_random'],
        '2048_pf_rate_gap': summary['long_2048']['random_minus_gap_pf_fraction'],
    }, indent=2))


if __name__ == '__main__':
    main()
