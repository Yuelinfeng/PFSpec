#!/usr/bin/env python3
"""Prepare fixed-length and natural-EOS prompts from explicitly selected scene sources."""
import argparse
import json
from pathlib import Path

import torch


def source_id(record):
    if record.get('scene') or record.get('source'):
        return str(record.get('scene') or record.get('source'))
    return Path(str(record['path'])).parents[1].name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crops', type=Path, required=True)
    parser.add_argument('--source-list', type=Path, required=True,
                        help='One exact, unused scene ID per line; select before execution')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--eos', type=int, default=12096)
    args = parser.parse_args()

    sources = [s.strip() for s in args.source_list.read_text().splitlines()
               if s.strip() and not s.lstrip().startswith('#')]
    if len(sources) != 30 or len(set(sources)) != 30:
        raise ValueError('Provide exactly 30 unique evaluation scene IDs')
    if args.out.exists():
        raise FileExistsError(f'Output directory already exists: {args.out}')

    rows = torch.load(args.crops, map_location='cpu', weights_only=True)
    catalog = {}
    for row in rows:
        sid = source_id(row)
        if sid not in catalog:
            catalog[sid] = row
    unknown = sorted(set(sources) - set(catalog))
    if unknown:
        raise ValueError(f'Unknown scene IDs: {unknown}')

    fixed, natural = [], []
    for index, sid in enumerate(sources):
        record = catalog[sid]
        tokens = record['tokens']
        tokens = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
        tokens = [int(t) for t in tokens]
        if len(tokens) < 260 or len(tokens) % 2:
            raise ValueError(f'{sid} has invalid or insufficient token length: {len(tokens)}')
        for prefix in (64, 256):
            for seed in (68101, 68102):
                fixed.append(dict(pair_id=f'{index:02d}-{prefix}-{seed}', source=sid,
                                  prefix=prefix, seed=seed,
                                  prompt=[args.eos] + tokens[:prefix]))
        for seed in (68101, 68102):
            natural.append(dict(pair_id=f'completion-{index:02d}-{seed}', source=sid,
                                prefix=256, seed=seed,
                                prompt=[args.eos] + tokens[:256]))

    for index in range(30):
        seed = 68201 + index
        natural.append(dict(pair_id=f'bos-{seed}', source=f'bos-{seed}',
                            prefix=0, seed=seed, prompt=[args.eos]))

    args.out.mkdir(parents=True)
    for filename, contexts in (('fixed_prompts.json', fixed),
                               ('natural_prompts.json', natural)):
        (args.out / filename).write_text(json.dumps({'contexts': contexts}, indent=2) + '\n')
    (args.out / 'source_ids.txt').write_text('\n'.join(sources) + '\n')
    print(f'Created {len(fixed)} fixed prompts and {len(natural)} natural-EOS prompts in {args.out}')


if __name__ == '__main__':
    main()
