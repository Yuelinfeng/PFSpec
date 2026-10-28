#!/usr/bin/env python3
"""Summarize acceptance counts and within-request typed token repetition."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(run):
    rows = [json.loads(line) for line in (run / 'records.jsonl').read_text().splitlines()]
    eos = json.loads((run / 'plan.json').read_text())['eos_id']
    result = {'source_run': str(run), 'methods': {}}
    for method in sorted({row['method'] for row in rows if row['execution'] == 'proposal_graph'}):
        selected = [row for row in rows if row['method'] == method and row['execution'] == 'proposal_graph']
        counts = Counter()
        windows = {length: Counter() for length in (2, 4, 8)}
        f_windows = {length: Counter() for length in (1, 2, 4)}
        for row in selected:
            counts.update({key: value for key, value in row['counters'].items() if type(value) is int})
            tokens = row['output_tokens'][:]
            if tokens and tokens[-1] == eos:
                tokens.pop()
            typed = [(index % 2, token) for index, token in enumerate(tokens)]
            for length, bucket in windows.items():
                seen = set()
                for index in range(len(typed) - length + 1):
                    key = tuple(typed[index:index + length])
                    bucket['total'] += 1
                    bucket['previously_seen'] += int(key in seen)
                    seen.add(key)
            features = tokens[1::2]
            for length, bucket in f_windows.items():
                seen = set()
                for index in range(len(features) - length + 1):
                    key = tuple(features[index:index + length])
                    bucket['total'] += 1
                    bucket['previously_seen'] += int(key in seen)
                    seen.add(key)
        result['methods'][method] = {
            'runs': len(selected), 'tokens': sum(row['generated_tokens'] for row in selected),
            'counts': dict(counts),
            'position_acceptance': counts['position_accepted'] / counts['position_verified'] if counts['position_verified'] else None,
            'feature_acceptance': counts['feature_accepted'] / counts['feature_verified'] if counts['feature_verified'] else None,
            'medusa_p1_rejection': counts['p1_rejected'] / counts['p1_checked'] if counts['p1_checked'] else None,
            'typed_windows': {str(key): dict(value) for key, value in windows.items()},
            'feature_only_windows': {str(key): dict(value) for key, value in f_windows.items()},
        }
    result['limits'] = ['Counts include generated suffixes only, without supplied prompts.',
                        'Repeated feature tokens do not establish a useful conditional proposal or speed gain.',
                        'No external corpus retrieval or online decoding was measured.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(summarize(args.run), ensure_ascii=False, indent=2) + '\n')
