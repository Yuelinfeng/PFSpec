"""Summarize G2 paired rounds using scene-source clusters."""
import json
from collections import defaultdict
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


def cluster_ci(source_ratios, seed=20261017, repeats=10000):
    sources = sorted(source_ratios)
    values = np.asarray([source_ratios[source] for source in sources], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(repeats, len(values)))
    estimates = np.exp(np.log(values[draws]).mean(axis=1))
    return {
        'point_geomean': geomean(values),
        'ci95_source_bootstrap': [float(x) for x in np.quantile(estimates, [0.025, 0.975])],
        'source_wins': int((values > 1).sum()),
        'source_count': len(sources),
        'source_ratios': {source: float(source_ratios[source]) for source in sources},
    }


def summarize_group(rows, group_name):
    source_method = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for row in rows:
        item = source_method[row['scene']][row['method']]
        item[0] += row['generated_tokens']
        item[1] += row['decoder_seconds']
    throughputs = {
        source: {
            method: totals[0] / totals[1]
            for method, totals in methods.items()
        }
        for source, methods in source_method.items()
    }
    ratios = {
        name: {
            source: throughputs[source]['gap'] / throughputs[source][denominator]
            for source in throughputs if denominator in throughputs[source]
        }
        for name, denominator in (('gap_over_pf', 'pf'), ('gap_over_p_only', 'p_only'))
    }
    return {
        'context_count': len({row['context_id'] for row in rows}),
        'source_count': len(source_method),
        'round_count_by_method': dict(__import__('collections').Counter(row['method'] for row in rows)),
        'method_totals': {
            method: {
                'generated_tokens': sum(row['generated_tokens'] for row in rows if row['method'] == method),
                'wall_seconds': sum(row['decoder_seconds'] for row in rows if row['method'] == method),
                'tokens_per_second': (
                    sum(row['generated_tokens'] for row in rows if row['method'] == method) /
                    sum(row['decoder_seconds'] for row in rows if row['method'] == method)
                ),
                'mean_tokens_per_round': float(np.mean([
                    row['generated_tokens'] for row in rows if row['method'] == method
                ])),
                'mean_seconds_per_round': float(np.mean([
                    row['decoder_seconds'] for row in rows if row['method'] == method
                ])),
            }
            for method in ('p_only', 'pf', 'gap')
        },
        'gap_over_pf': cluster_ci(ratios['gap_over_pf']),
        'gap_over_p_only': cluster_ci(ratios['gap_over_p_only']),
        'group': group_name,
    }


def main():
    run_dir = ARTIFACTS / 'g2_raw'
    rows = load_jsonl(run_dir / 'g2_matched_rounds.jsonl')
    profile_rows = load_jsonl(run_dir / 'g2_profile_subset.jsonl')
    micro = json.loads((run_dir / 'g2_gate_microbenchmark.json').read_text())
    assert len(rows) == 56 * 8 * 3
    keyed = {(row['context_id'], row['trial'], row['method']): row for row in rows}
    route_match = 0
    for context_id in range(56):
        for trial in range(8):
            gap_row = keyed[(context_id, trial, 'gap')]
            expected_method = 'pf' if gap_row['gap'] == 1 else 'p_only'
            endpoint = keyed[(context_id, trial, expected_method)]
            assert gap_row['route'] == expected_method
            assert gap_row['output'] == endpoint['output']
            assert gap_row['stats'] == endpoint['stats']
            route_match += 1

    gap_by_context = {
        context_id: keyed[(context_id, 0, 'gap')]['gap']
        for context_id in range(56)
    }
    eligible_gap = [value for value in gap_by_context.values() if value is not None]
    strata = {
        'gap_eq_1': [row for row in rows if gap_by_context[row['context_id']] == 1],
        'gap_gt_1': [row for row in rows if gap_by_context[row['context_id']] > 1],
    }
    profile_stage_summary = {}
    for method in ('p_only', 'pf', 'gap'):
        samples = [row for row in profile_rows if row['method'] == method]
        profile_stage_summary[method] = {
            'sample_count': len(samples),
            'stage_names': sorted({
                stage for row in samples for stage in row['stage_profile']
            }),
            'raw_profile_path': 'g2_profile_subset.jsonl',
        }

    selector = {method: summary for method, summary in micro.items()}
    output = {
        'status': 'complete',
        'data_role': 'development diagnostic; same 56 prefixes as B1',
        'prefixes': 56,
        'sources': len({row['scene'] for row in rows}),
        'seeds_per_prefix': 8,
        'unprofiled_runs': len(rows),
        'profiled_runs': len(profile_rows),
        'gap_one_prefixes': sum(value == 1 for value in eligible_gap),
        'gap_above_one_prefixes': sum(value > 1 for value in eligible_gap),
        'observed_pf_route_fraction': float(sum(value == 1 for value in eligible_gap) / len(eligible_gap)),
        'route_matches_expected_endpoint': route_match,
        'route_match_failures': 0,
        'overall': summarize_group(rows, 'all prefixes'),
        'strata': {
            name: summarize_group(group, name)
            for name, group in strata.items()
        },
        'route_microbenchmark': selector,
        'profile_subset': profile_stage_summary,
        'limitations': [
            'Single-round results diagnose routing and overhead; they do not predict the long online trajectory.',
            'The 56 prefixes share only 10 source scenes and are not independent confirmation data.',
            'Profiled stage timings are from a separate instrumented subset and are not used for the main wall-time comparison.',
        ],
        'raw_runs': 'g2_raw/g2_matched_rounds.jsonl',
    }
    (ARTIFACTS / 'G2_summary.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({
        'status': output['status'],
        'route_matches': route_match,
        'pf_fraction': output['observed_pf_route_fraction'],
        'gap_over_pf': output['overall']['gap_over_pf']['point_geomean'],
        'gap_over_ponly': output['overall']['gap_over_p_only']['point_geomean'],
    }, indent=2))


if __name__ == '__main__':
    main()
