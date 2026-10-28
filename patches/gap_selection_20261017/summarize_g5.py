"""Summarize D1 comparisons with each source as the resampling unit."""
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
        assert row['stats']['selection']['mode'] == row['method']
        assert row['wall_seconds'] > 0


def source_speed(rows, method):
    totals = defaultdict(lambda: [0, 0.0])
    for row in rows:
        if row['method'] == method:
            totals[row['scene']][0] += row['actual_new_tokens']
            totals[row['scene']][1] += row['wall_seconds']
    return {scene: tokens / seconds for scene, (tokens, seconds) in totals.items()}


def compare(rows, numerator, denominator, seed):
    left, right = source_speed(rows, numerator), source_speed(rows, denominator)
    assert set(left) == set(right), (set(left), set(right))
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
            'eos_terminations': sum(row['terminated_on_eos'] for row in selected),
            'mean_actual_tokens': float(np.mean([row['actual_new_tokens'] for row in selected])),
        }
    return result


def route_totals(rows):
    counts = Counter()
    proposed = verified = accepted = 0
    for row in rows:
        if row['method'] != 'gap':
            continue
        selection = row['stats']['selection']
        counts.update(selection['counts'])
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
    methods = ('p_only', 'pf', 'gap')
    validate_rows(rows, expected_count, expected_contexts, expected_sources,
                  budget, methods, set(prefixes))
    comparisons = {
        'gap_over_pf': compare(rows, 'gap', 'pf', seed),
        'gap_over_p_only': compare(rows, 'gap', 'p_only', seed + 1),
        'pf_over_p_only': compare(rows, 'pf', 'p_only', seed + 2),
    }
    by_prefix = {}
    for prefix in prefixes:
        selected = [row for row in rows if row['prefix'] == prefix]
        if selected:
            by_prefix[str(prefix)] = {
                'context_count': len({row['context_id'] for row in selected}),
                'gap_over_pf': compare(selected, 'gap', 'pf', seed + prefix),
                'gap_over_p_only': compare(selected, 'gap', 'p_only', seed + prefix + 1),
            }
    return {
        'budget_tokens': budget,
        'run_count': len(rows),
        'context_count': expected_contexts,
        'source_count': expected_sources,
        'method_totals': method_totals(rows),
        'comparisons': comparisons,
        'by_prefix': by_prefix,
        'gap_route_use': route_totals(rows),
    }


def main():
    raw = ARTIFACTS / 'g5_raw'
    short = load_jsonl(raw / 'online_512.jsonl')
    long = load_jsonl(raw / 'long_2048.jsonl')
    short_summary = summarize_window(short, 512, 180, 20, 10, (64, 256), 20261020)
    long_summary = summarize_window(long, 2048, 90, 10, 10, (256,), 20261021)
    primary = long_summary['comparisons']['gap_over_pf']
    proceed = primary['source_geomean'] >= 1.02 and primary['ci95_source_bootstrap'][0] > 1.0
    summary = {
        'status': 'complete',
        'primary_outcome': 'D1 2048-token Gap/PF source-geomean; threshold and interval rule were written to the run protocol before measurement',
        'online_512': short_summary,
        'long_2048': long_summary,
        'decision': {
            'status': 'go_to_G6' if proceed else 'stop_before_T',
            'rule': 'go when 2048-token Gap/PF source-geomean is at least 1.02 and the source-bootstrap 95% lower bound exceeds 1.00',
            'criterion_met': proceed,
        },
        'limitations': [
            'D1 is new development data and does not serve as independent confirmation.',
            'The Target pretraining manifest does not permit a complete source-overlap audit.',
            'The 512-token and 2048-token outcomes are reported separately because the D0 signal varied by window length.',
        ],
        'raw_512': 'g5_raw/online_512.jsonl',
        'raw_2048': 'g5_raw/long_2048.jsonl',
    }
    path = ARTIFACTS / 'G5_summary.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({
        'decision': summary['decision'],
        '512_gap_over_pf': short_summary['comparisons']['gap_over_pf'],
        '2048_gap_over_pf': primary,
    }, indent=2))


if __name__ == '__main__':
    main()
