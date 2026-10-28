#!/usr/bin/env python3
"""Run the four predeclared decoding repeats in separate sequential processes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    state_path = root / 'suite.json'
    if state_path.exists():
        raise FileExistsError(state_path)
    state = dict(status='running', started_at=time.time(), repeats=[])
    save(state_path, state)
    wrapper = Path(__file__).resolve().with_name('run_repeat.py')
    common = [
        '--candidate-source', '/root/autodl-tmp/gaussiangpt_spec_next_20261022/source_run02',
        '--project', '/root/workspace/GaussianGPT',
        '--source', '/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04',
        '--crops', '/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/final_test_crops.pt',
        '--sources', '30', '--prefixes', '64,256', '--seeds', '2',
        '--seed-start', '52101', '--budget', '512', '--cache-capacity', '2048',
        '--source-scope', 'independent',
    ]
    for index, offset in enumerate((0, 8, 15, 23)):
        name = f'repeat_{index + 1:02d}'
        method_order = 'w1,rescue' if index % 2 == 0 else 'rescue,w1'
        command = [sys.executable, str(wrapper), '--repeat-index', str(index),
                   '--source-offset', str(offset), *common,
                   '--methods', method_order, '--out', str(root / name)]
        row = dict(name=name, repeat_index=index, source_offset=offset,
                   method_order=method_order, command=command,
                   status='running', started_at=time.time())
        state['repeats'].append(row)
        save(state_path, state)
        with (root / (name + '.log')).open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            row['process_id'] = child.pid
            save(state_path, state)
            returncode = child.wait()
        row.update(returncode=returncode, finished_at=time.time(),
                   status='complete' if returncode == 0 else 'failed')
        save(state_path, state)
    state.update(status=('complete' if all(row['returncode'] == 0 for row in state['repeats'])
                         else 'failed'), finished_at=time.time())
    save(state_path, state)
    return 0 if state['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
