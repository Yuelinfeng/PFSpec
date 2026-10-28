"""Compare saved execution variants without fitting or selecting parameters."""
from pathlib import Path
import argparse
import collections
import json
import math
import statistics


SEMANTIC = (
    'accepted', 'proposed', 'verified', 'position_accepted',
    'feature_accepted', 'eos_accepted', 'accepted_lengths',
    'round_generated_lengths',
)
EXECUTION = (
    'target_calls', 'target_input_tokens', 'recovery_calls',
    'tree_verify_calls', 'target_tree_nodes',
)
EQUAL_PAIRS = (
    ('w1', 'same0'), ('fixed1', 'same1'),
    ('old_value', 'value_reference'), ('old_value', 'value'),
    ('value_reference', 'value'),
)
SPEED_PAIRS = EQUAL_PAIRS + (
    ('same0', 'same1'), ('same1', 'value'), ('fixed1', 'value'),
)


def analyze(paths):
    groups = collections.defaultdict(dict)
    plans = []
    errors = []
    for path in paths:
        plan = json.loads((path / 'plan.json').read_text())
        progress = json.loads((path / 'progress.json').read_text())
        records = [json.loads(line) for line in
                   (path / 'records.jsonl').read_text().splitlines() if line]
        if progress['status'] != 'complete' or len(records) != progress['expected']:
            raise ValueError(f'Incomplete measurement: {path}')
        plans.append(plan)
        for row in records:
            key = (row['repeat_index'], row['source'], row['prefix'], row['seed'])
            if row['method'] in groups[key]:
                raise ValueError(f'Duplicate method condition: {key}')
            groups[key][row['method']] = row
    expected_methods = set(plans[0]['methods'])
    model = plans[0]['model']
    for plan in plans:
        if set(plan['methods']) != expected_methods or plan['model'] != model:
            raise ValueError('Execution variants must use the same model and methods')
    comparisons = collections.Counter()
    totals = collections.defaultdict(lambda: collections.defaultdict(float))
    coverage = collections.defaultdict(lambda: collections.defaultdict(int))
    for key, group in groups.items():
        if set(group) != expected_methods:
            raise ValueError(f'Missing method condition: {key}')
        reference = group['w1']
        for method, row in group.items():
            comparisons['output_rows'] += 1
            if (row['output_tokens'] != reference['output_tokens'] or
                row['eos'] != reference['eos'] or
                row['generated_tokens'] != reference['generated_tokens']):
                errors.append([key, method, 'output'])
            if any(row['stats'][name] != reference['stats'][name] for name in SEMANTIC):
                errors.append([key, method, 'semantic'])
            hits = row['stats'].get('position_rescue_hits', 0)
            for name in ('target_calls', 'recovery_calls'):
                if reference['stats'][name] - row['stats'][name] != hits:
                    errors.append([key, method, name, 'call savings'])
            if row['stats'].get('round_observer_enabled', False):
                errors.append([key, method, 'observer enabled in timing'])
            if row['stats'].get('diagnostic_cost_probe_rounds', 0):
                errors.append([key, method, 'cost probe enabled in timing'])
            totals[method]['wall_seconds'] += row['outer_wall_seconds']
            for name in EXECUTION + ('position_rescue_hits', 'position_backup_nodes'):
                totals[method][name] += row['stats'].get(name, 0)
            totals[method]['selector_wall_ms'] += row['stats'].get('backup_selector_wall_ms', 0)
            for name in ('position_accepted', 'position_verified', 'feature_accepted',
                         'feature_verified', 'eos_accepted', 'k1_steps'):
                coverage[method][name] += row['stats'].get(name, 0)
            coverage[method]['eos_requests'] += int(row['eos'])
        for left, right in EQUAL_PAIRS:
            if left not in group or right not in group:
                continue
            comparisons[f'{left}_vs_{right}'] += 1
            for name in EXECUTION:
                if group[left]['stats'][name] != group[right]['stats'][name]:
                    errors.append([key, left, right, name])
            if left != 'w1':
                for name in ('position_rescue_hits', 'position_backup_nodes'):
                    if group[left]['stats'].get(name, 0) != group[right]['stats'].get(name, 0):
                        errors.append([key, left, right, name])
            if left in ('old_value', 'value_reference'):
                for name in ('backup_selector_calls', 'backup_selector_zero_actions',
                             'backup_selector_nodes_selected'):
                    def comparable(row):
                        value = row['stats'].get(name, 0)
                        if name != 'backup_selector_nodes_selected':
                            value += row['stats'].get('backup_selector_empty_candidate_skips', 0)
                        return value
                    if comparable(group[left]) != comparable(group[right]):
                        errors.append([key, left, right, name])
    speeds = {}
    for reference, candidate in SPEED_PAIRS:
        if reference not in expected_methods or candidate not in expected_methods:
            continue
        by_process_source = collections.defaultdict(list)
        by_source_condition = collections.defaultdict(list)
        for key, group in groups.items():
            log_ratio = math.log(group[reference]['outer_wall_seconds'] /
                                 group[candidate]['outer_wall_seconds'])
            by_process_source[(key[0], key[1])].append(log_ratio)
            by_source_condition[key[1:]].append(log_ratio)
        by_source = collections.defaultdict(list)
        for key, values in by_source_condition.items():
            by_source[key[0]].append(statistics.fmean(values))
        by_process = collections.defaultdict(list)
        for (process, source), values in by_process_source.items():
            by_process[process].append(statistics.fmean(values))
        speeds[f'{candidate}_over_{reference}'] = {
            'source_equal_ratio': math.exp(statistics.fmean(
                statistics.fmean(values) for values in by_source.values())),
            'per_process_source_equal_ratio': {
                str(process): math.exp(statistics.fmean(values))
                for process, values in by_process.items()},
            'cumulative_time_ratio': totals[reference]['wall_seconds'] /
                                     totals[candidate]['wall_seconds'],
        }
    return dict(passed=not errors, errors=errors, comparisons=dict(comparisons),
                condition_groups=len(groups), totals=dict(totals),
                branch_coverage=dict(coverage), speed_comparisons=speeds,
                scope='Implementation review on reused development sources; no new-source method claim')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = analyze(args.paths)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
