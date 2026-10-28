"""Summarize the T-set comparisons with source-cluster intervals."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'docs/project/artifacts/gap_selection_20261017'


def load_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def validate_rows(rows, expected_count, expected_contexts, expected_sources,
                  budget, methods, prefixes):
    assert len(rows) == expected_count, (len(rows), expected_count)
    assert len({row['context_id'] for row in rows}) == expected_contexts
    assert len({row['scene'] for row in rows}) == expected_sources
    assert {row['method'] for row in rows} == set(methods)
    seen = set()
    for row in rows:
        key = (row['context_id'], row['seed_base'], row['method'])
        assert key not in seen, key
        seen.add(key)
        assert row['requested_new_tokens'] == budget
        assert row['prefix'] in prefixes
        assert row['actual_new_tokens'] == len(row['output'])
        assert 0 < row['actual_new_tokens'] <= budget
        assert row['stats']['generated_tokens'] == row['actual_new_tokens']
        assert row['stats']['eos'] == row['terminated_on_eos']
        assert row['wall_seconds'] > 0
        if row['method'] != 'target':
            assert row['stats']['selection']['mode'] == row['method']


def source_speed(rows, method):
    totals = defaultdict(lambda: [0, 0.0])
    for row in rows:
        if row['method'] == method:
            totals[row['scene']][0] += row['actual_new_tokens']
            totals[row['scene']][1] += row['wall_seconds']
    return {scene: tokens / seconds for scene, (tokens, seconds) in totals.items()}


def compare(rows, numerator, denominator, seed):
    left, right = source_speed(rows, numerator), source_speed(rows, denominator)
    assert set(left) == set(right), (numerator, denominator, set(left), set(right))
    sources = sorted(left)
    ratios = np.asarray([left[source] / right[source] for source in sources], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ratios), size=(10000, len(ratios)))
    estimates = np.exp(np.log(ratios[draws]).mean(axis=1))
    return {
        'numerator': numerator,
        'denominator': denominator,
        'source_geomean': float(np.exp(np.log(ratios).mean())),
        'ci95_source_bootstrap': [float(x) for x in np.quantile(estimates, [0.025, 0.975])],
        'source_wins': int(np.sum(ratios > 1.0)),
        'source_count': len(sources),
        'source_ratios': {source: float(value) for source, value in zip(sources, ratios)},
    }


def method_totals(rows):
    result = {}
    for method in sorted({row['method'] for row in rows}):
        selected = [row for row in rows if row['method'] == method]
        tokens = sum(row['actual_new_tokens'] for row in selected)
        seconds = sum(row['wall_seconds'] for row in selected)
        result[method] = {
            'runs': len(selected),
            'generated_tokens': tokens,
            'wall_seconds': seconds,
            'micro_tokens_per_second': tokens / seconds,
            'mean_actual_tokens': float(np.mean([row['actual_new_tokens'] for row in selected])),
            'eos_terminations': sum(row['terminated_on_eos'] for row in selected),
        }
    return result


def route_totals(rows, method):
    counts = Counter()
    proposed = verified = accepted = 0
    for row in rows:
        if row['method'] != method:
            continue
        selection = row['stats'].get('selection', {})
        counts.update(selection.get('counts', {}))
        proposed += row['stats'].get('feature_proposed', 0) or 0
        verified += row['stats'].get('feature_verified', 0) or 0
        accepted += row['stats'].get('feature_accepted', 0) or 0
    pf = counts['executed_pf']
    p_only = counts['executed_p_only']
    return {
        'counts': dict(counts),
        'actual_pf_fraction': pf / (pf + p_only) if pf + p_only else None,
        'feature_proposed': proposed,
        'feature_verified': verified,
        'feature_accepted': accepted,
        'feature_acceptance_rate': accepted / verified if verified else None,
    }


def summarize_window(rows, budget, expected_count, expected_contexts,
                     expected_sources, prefixes, seed):
    methods = ('target', 'p_only', 'pf', 'gap', 'random')
    validate_rows(rows, expected_count, expected_contexts, expected_sources,
                  budget, methods, set(prefixes))
    comparisons = {
        'gap_over_pf': compare(rows, 'gap', 'pf', seed),
        'gap_over_random': compare(rows, 'gap', 'random', seed + 1),
        'gap_over_p_only': compare(rows, 'gap', 'p_only', seed + 2),
        'gap_over_target': compare(rows, 'gap', 'target', seed + 3),
        'random_over_pf': compare(rows, 'random', 'pf', seed + 4),
    }
    by_prefix = {}
    for prefix in prefixes:
        selected = [row for row in rows if row['prefix'] == prefix]
        by_prefix[str(prefix)] = {
            'context_count': len({row['context_id'] for row in selected}),
            'gap_over_pf': compare(selected, 'gap', 'pf', seed + prefix),
            'gap_over_random': compare(selected, 'gap', 'random', seed + prefix + 1),
        }
    return {
        'budget_tokens': budget,
        'run_count': len(rows),
        'context_count': expected_contexts,
        'source_count': expected_sources,
        'method_totals': method_totals(rows),
        'comparisons': comparisons,
        'by_prefix': by_prefix,
        'gap_route_use': route_totals(rows, 'gap'),
        'random_route_use': route_totals(rows, 'random'),
    }


def main():
    raw = ARTIFACTS / 'g6_raw'
    short = load_jsonl(raw / 'online_512.jsonl')
    long = load_jsonl(raw / 'long_2048.jsonl')
    short_summary = summarize_window(short, 512, 900, 60, 30, (64, 256), 20261022)
    long_summary = summarize_window(long, 2048, 450, 30, 30, (256,), 20261023)
    primary = short_summary['comparisons']['gap_over_pf']
    confirmed = primary['source_geomean'] > 1.0 and primary['ci95_source_bootstrap'][0] > 1.0
    meaningful = primary['source_geomean'] >= 1.02 and primary['ci95_source_bootstrap'][0] > 1.0
    summary = {
        'status': 'complete',
        'primary_outcome': '512-token Gap/PF source-geomean with source-cluster bootstrap interval',
        'online_512': short_summary,
        'long_2048': long_summary,
        'decision': {
            'supports_positive_gain': confirmed,
            'reaches_two_percent_point_estimate': meaningful,
            'rule': 'positive gain requires a point estimate above 1 and source-bootstrap 95% lower bound above 1; a point estimate of at least 1.02 marks the prespecified practical scale',
        },
        'limitations': [
            'Target pretraining records do not include a complete source ID list, so pretraining overlap cannot be audited.',
            'The 2048-token result is secondary; the 512-token outcome remains primary.',
            'These measurements cover decoder time and do not include VQ decode, Gaussian export, or rendering.',
        ],
        'raw_512': 'g6_raw/online_512.jsonl',
        'raw_2048': 'g6_raw/long_2048.jsonl',
    }
    path = ARTIFACTS / 'G6_summary.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({
        'decision': summary['decision'],
        '512_gap_over_pf': primary,
        '512_gap_over_random': short_summary['comparisons']['gap_over_random'],
        '2048_gap_over_pf': long_summary['comparisons']['gap_over_pf'],
        '2048_gap_over_random': long_summary['comparisons']['gap_over_random'],
    }, indent=2))


if __name__ == '__main__':
    main()
