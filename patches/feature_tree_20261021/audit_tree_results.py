#!/usr/bin/env python3
"""Independently recompute tree timing ratios and token paths from row records."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics

ARMS = ('original_target:target', 'proposal_graph:pf_mlp',
        'tree_graph:tree_w1', 'tree_graph:tree_w2')


def load(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values, p):
    values = sorted(values)
    position = (len(values)-1)*p
    left = math.floor(position)
    right = min(left+1, len(values)-1)
    return values[left]+(values[right]-values[left])*(position-left)


def ratio(rows, numerator, denominator):
    per_source = defaultdict(list)
    for pair, arms in rows.items():
        a, b = arms[numerator], arms[denominator]
        left = len(a['output_tokens'])/a['outer_wall_seconds']
        right = len(b['output_tokens'])/b['outer_wall_seconds']
        per_source[a['source']].append(math.log(left/right))
    logs = {key: statistics.mean(values) for key, values in sorted(per_source.items())}
    sources = sorted(logs)
    point = math.exp(statistics.mean(logs.values()))
    rng = random.Random(402101)
    samples = [math.exp(statistics.mean(logs[rng.choice(sources)] for _ in sources))
               for _ in range(10000)]
    return dict(value=point, ci95=[percentile(samples, .025), percentile(samples, .975)],
                per_source={k: math.exp(v) for k, v in logs.items()})


def audit(run_dir, previous=None):
    plan = load(run_dir/'plan.json')
    prompts = {r['pair_id']: r for r in load(run_dir/'prompts.json')['contexts']}
    data = records(run_dir/'records.jsonl')
    summary = load(run_dir/'summary.json')
    groups = defaultdict(dict)
    errors = []
    ranks = plan['position_rank_map']
    eos = plan['eos_id']
    if len(data) != plan['expected_runs']:
        errors.append('row_count')
    totals = {arm: Counter() for arm in ARMS}
    valid_sequences = full_budget = 0
    for row in data:
        pair, arm = row['pair_id'], row['arm']
        if pair not in prompts or arm not in ARMS or arm in groups[pair]:
            errors.append('unexpected_or_duplicate_row')
            continue
        groups[pair][arm] = row
        context = prompts[pair]
        output = row['output_tokens']
        if any(row[k] != context[k] for k in ['source', 'prefix', 'seed']):
            errors.append('row_context')
        if len(output) != row['generated_tokens'] or len(output) > plan['budget']:
            errors.append('output_count')
        if row['eos'] != bool(output and output[-1] == eos):
            errors.append('eos')
        if not math.isclose(row['tokens_per_second'], len(output)/row['outer_wall_seconds'], rel_tol=1e-12):
            errors.append('throughput')
        joined = context['prompt'][1:]+output
        previous_position = None
        valid = True
        for index, token in enumerate(joined):
            if token == eos:
                valid &= index % 2 == 0 and index == len(joined)-1
                continue
            if index % 2:
                valid &= 8000 <= token < eos
            else:
                valid &= 0 <= token < 8000
                if valid and previous_position is not None:
                    valid &= ranks[token] > ranks[previous_position]
                previous_position = token
        valid_sequences += int(valid)
        full_budget += len(output) == plan['budget']
        if not valid:
            errors.append('typed_sequence_or_position_order')
        for key, value in row['counters'].items():
            if type(value) is int:
                totals[arm][key] += value
        totals[arm]['output_tokens'] += len(output)
    if set(groups) != set(prompts) or any(set(g) != set(ARMS) for g in groups.values()):
        errors.append('pair_coverage')
    output_equal = count_equal = 0
    for arms in groups.values():
        a, b = arms[ARMS[1]], arms[ARMS[2]]
        output_equal += a['output_tokens'] == b['output_tokens']
        keys = (set(a['counters']) & set(b['counters']))-{'tree_runtime_stats', 'tree_width', 'proposal_width', 'variant'}
        count_equal += all(a['counters'][k] == b['counters'][k] for k in keys)
    if output_equal != len(groups) or count_equal != len(groups):
        errors.append('w1_mlp_parity')
    comparison_specs = {
        'proposal_graph_pf_mlp_over_original_target': (ARMS[1], ARMS[0]),
        'tree_graph_tree_w1_over_original_target': (ARMS[2], ARMS[0]),
        'tree_graph_tree_w2_over_original_target': (ARMS[3], ARMS[0]),
        'tree_w2_over_pf_mlp': (ARMS[3], ARMS[1]),
        'tree_w2_over_tree_w1': (ARMS[3], ARMS[2]),
        'tree_w1_over_pf_mlp': (ARMS[2], ARMS[1]),
    }
    comparisons = {}
    for label, (a, b) in comparison_specs.items():
        value = ratio(groups, a, b)
        comparisons[label] = value
        if label in summary['comparisons']:
            expected = summary['comparisons'][label]['all_output_lengths']
            values = [expected['source_equal_geometric_mean']]+expected['source_bootstrap_95_ci']
            actual = [value['value']]+value['ci95']
            if not all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12) for x, y in zip(actual, values)):
                errors.append('ratio_or_interval_'+label)
    mlp, w1, w2 = [totals[a] for a in ARMS[1:]]
    acceptance = dict(mlp=mlp['feature_accepted']/mlp['feature_verified'],
                      w1=w1['feature_branch_hits']/(w1['feature_branch_hits']+w1['feature_all_rejected']),
                      w2=w2['feature_branch_hits']/(w2['feature_branch_hits']+w2['feature_all_rejected']))
    cross = None
    if previous:
        old = {(r['pair_id'], r['arm']): r for r in records(previous/'records.jsonl')}
        cross = dict(rows=len(data), outputs_equal=0, counters_equal=0)
        for row in data:
            other = old[(row['pair_id'], row['arm'])]
            cross['outputs_equal'] += row['output_tokens'] == other['output_tokens']
            left = {k:v for k,v in row['counters'].items() if k != 'tree_runtime_stats'}
            right = {k:v for k,v in other['counters'].items() if k != 'tree_runtime_stats'}
            cross['counters_equal'] += left == right
        if cross['outputs_equal'] != len(data) or cross['counters_equal'] != len(data):
            errors.append('cross_run_parity')
    return dict(status='passed' if not errors else 'failed', run_dir=str(run_dir),
                rows=len(data), pairs=len(groups), sources=len({r['source'] for r in data}),
                valid_sequences=valid_sequences, full_budget_rows=full_budget,
                w1_mlp_outputs_equal=output_equal, w1_mlp_shared_counts_equal=count_equal,
                comparisons=comparisons, totals=totals, conditional_feature_any_acceptance=acceptance,
                cross_run_parity=cross, errors=errors,
                scope='Fresh standard-library calculation from records, prompts and rank map; no GPU or inference.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--previous-run-dir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Output already exists')
    result = audit(args.run_dir, args.previous_run_dir)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['status','rows','pairs','sources','errors']}))
    if result['status'] != 'passed':
        raise ArithmeticError('Result audit failed')


if __name__ == '__main__':
    main()
