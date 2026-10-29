#!/usr/bin/env python3
"""Resume incomplete paired request groups without changing existing records."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys


def load_original(directory):
    return json.loads((directory/'plan.json').read_text())


def attempts(directory):
    return [directory] + sorted(p for p in directory.parent.glob(directory.name+'_resume_[0-9][0-9]') if p.is_dir())


def planned(plan):
    return {(c['name'], prompt['pair_id']): (c, i, prompt)
            for c in plan['cohorts'] for i, prompt in enumerate(c['prompts'])}


def complete_groups(directory, plan):
    wanted = planned(plan)
    selected = {}
    for attempt in attempts(directory):
        path = attempt/'records.jsonl'
        if not path.exists():
            continue
        grouped = defaultdict(list)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = row['cohort'], row['pair_id']
            if key not in wanted:
                raise ValueError(f'unplanned record in {attempt}: {key}')
            expected_prompt = wanted[key][2]
            if any(row[k] != expected_prompt[k] for k in ('source', 'seed', 'prefix')):
                raise ValueError(f'prompt mismatch in {attempt}: {key}')
            grouped[key].append(row)
        for key, rows in grouped.items():
            if len(rows) == len(plan['methods']) and {r['method'] for r in rows} == set(plan['methods']):
                selected.setdefault(key, (attempt, rows))
    return selected


def next_name(directory, suffix):
    for index in range(1, 100):
        path = directory.parent/f'{directory.name}_{suffix}_{index:02d}'
        if not path.exists():
            return path
    raise RuntimeError('No unused output name')


def resume(directory, plan, selected, python):
    missing = {k: v for k, v in planned(plan).items() if k not in selected}
    if not missing:
        print('All request groups already complete')
        return
    output = next_name(directory, 'resume')
    input_dir = output.with_name(output.name+'_inputs')
    input_dir.mkdir()
    inputs = {'fixed512': [], 'natural_eos': []}
    for (cohort, _), (_, index, prompt) in missing.items():
        inputs[cohort].append(dict(prompt, order_index=index))
    for name, contexts in inputs.items():
        (input_dir/(name+'.json')).write_text(json.dumps({'contexts': contexts}, indent=2)+'\n')
    cohorts = {c['name']: c for c in plan['cohorts']}
    fixed = cohorts.get('fixed512', {'budget': 512, 'capacity': 2048})
    natural = cohorts.get('natural_eos', {'budget': 16000, 'capacity': 16384})
    cmd = [python, str(Path(__file__).with_name('run_e2e.py')), '--project', plan['project'],
           '--source', plan['source'], '--fixed-prompts', str(input_dir/'fixed512.json'),
           '--warm-tokens', str(directory.parent/'warm_tokens.pt'), '--methods', ','.join(plan['methods']),
           '--cache-capacity', str(fixed['capacity']), '--fixed-budget', str(fixed['budget']),
           '--natural-budget', str(natural['budget']), '--natural-capacity', str(natural['capacity']),
           '--max-wall-seconds', str(plan['max_wall_seconds']), '--out', str(output)]
    if plan.get('candidate_source'):
        cmd.extend(('--candidate-source', plan['candidate_source']))
    if 'natural_eos' in cohorts:
        cmd.extend(('--natural-prompts', str(input_dir/'natural_eos.json')))
    else:
        cmd.append('--fixed-only')
    print(f'Repeating {len(missing)} incomplete groups in {output}', flush=True)
    subprocess.run(cmd, check=True)


def merge(directory, plan, selected):
    wanted = planned(plan)
    output = next_name(directory, 'merged')
    output.mkdir()
    rows = []
    origins = {}
    for key in wanted:
        if key in selected:
            attempt, group = selected[key]
            rows.extend(group)
            origins['/'.join(key)] = str(attempt)
    missing = [list(k) for k in wanted if k not in selected]
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (output/'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    progress = {'status': 'complete' if not missing else 'partial', 'completed': len(rows),
                'expected': plan['expected'], 'missing_groups': missing,
                'source_attempts': [str(p) for p in attempts(directory)]}
    (output/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
    (output/'merge_sources.json').write_text(json.dumps(origins, indent=2)+'\n')
    print(f'{output}: {len(rows)}/{plan["expected"]} records, {len(missing)} incomplete groups')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('resume', 'merge'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    directory = args.directory.resolve()
    plan = load_original(directory)
    selected = complete_groups(directory, plan)
    if args.action == 'resume':
        resume(directory, plan, selected, args.python)
    else:
        merge(directory, plan, selected)


if __name__ == '__main__':
    main()
