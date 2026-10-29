#!/usr/bin/env python3
"""Summarize paired token throughput and complete-request elapsed time."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import tarfile


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def cohort(row):
    if row['cohort'] == 'fixed512':
        return 'fixed512'
    return 'natural_bos' if row['prefix'] == 0 else 'natural_completion'


def paired(rows, numerator, denominator, metric, seed, iterations):
    groups = defaultdict(dict)
    for row in rows:
        key = (row['pair_id'], row['source'])
        if row['method'] in groups[key]:
            raise ValueError(f"duplicate method for {key}: {row['method']}")
        groups[key][row['method']] = row
    sources = defaultdict(list)
    for (_, source), group in groups.items():
        if numerator not in group or denominator not in group:
            continue
        left, right = group[numerator], group[denominator]
        if metric == 'request_time':
            value = right['total_seconds'] / left['total_seconds']
        else:
            time_key = 'generation' if metric == 'generation_throughput' else None
            lt = left['stages'][time_key] if time_key else left['total_seconds']
            rt = right['stages'][time_key] if time_key else right['total_seconds']
            value = (left['generated_tokens'] / lt) / (right['generated_tokens'] / rt)
        sources[source].append(math.log(value))
    if not sources:
        return None
    means = [statistics.mean(values) for values in sources.values()]
    rng = random.Random(seed)
    draws = [math.exp(statistics.mean(rng.choices(means, k=len(means))))
             for _ in range(iterations)]
    return {
        'sources': len(means),
        'pairs': sum(map(len, sources.values())),
        'ratio': math.exp(statistics.mean(means)),
        'interval95': [percentile(draws, .025), percentile(draws, .975)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--iterations', type=int, default=5000)
    args = parser.parse_args()
    comparisons = {
        'A': [('opt_target', 'target'), ('ranker_only', 'opt_target'),
              ('w1', 'target'), ('w1', 'opt_target'), ('w1', 'ranker_only'),
              ('rescue', 'w1')],
        'B': [('w2', 'w1')],
    }
    members = {'A': 'run_A_merged_01/records.jsonl', 'B': 'run_B/records.jsonl'}
    result = {
        'archive': str(args.archive), 'seed': args.seed, 'iterations': args.iterations,
        'request_time_definition': 'denominator total_seconds / numerator total_seconds',
        'throughput_definition': '(numerator tokens / numerator seconds) / (denominator tokens / denominator seconds)',
        'resampling_unit': 'scene for fixed and completion; seed for BOS',
        'experiments': {},
    }
    with tarfile.open(args.archive, 'r:gz') as archive:
        for name, member in members.items():
            rows = [json.loads(line) for line in archive.extractfile(member) if line.strip()]
            result['experiments'][name] = {}
            for group in ('fixed512', 'natural_bos', 'natural_completion'):
                subset = [row for row in rows if cohort(row) == group]
                if not subset:
                    continue
                output = {}
                for numerator, denominator in comparisons[name]:
                    output[f'{numerator}/{denominator}'] = {
                        metric: paired(subset, numerator, denominator, metric, args.seed, args.iterations)
                        for metric in ('generation_throughput', 'end_to_end_throughput', 'request_time')
                    }
                result['experiments'][name][group] = output
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
