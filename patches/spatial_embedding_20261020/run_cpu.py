#!/usr/bin/env python3
"""Run the bounded N2 CPU experiment in its own result directory."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume-after-transfer', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'run01'
    if args.resume_after_transfer:
        state = json.loads((out/'pipeline.json').read_text())
        progress = json.loads((out/'experiment/progress.json').read_text())
        if state['status'] != 'failed' or progress['completed_updates'] != 0:
            raise ValueError('Transfer recovery requires a failed run with zero updates')
        started = state['started_unix']
        deadline = state['absolute_deadline_unix']
        state['previous_failure'] = state.pop('error')
        state['status'] = 'running'
        state['recovery_reason'] = 'Source transfer finished after the first launch; no model update occurred'
    else:
        out.mkdir(exist_ok=False)
        started = time.time()
        deadline = started + 1190.0
        state = dict(status='running', started_unix=started,
                     started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     absolute_deadline_unix=deadline, gpu_budget_seconds=0,
                     runtime_limit_seconds=1190, prior_readonly_preparation_seconds=8.7,
                     phases=[])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    source = root / 'source'
    write_json(out / 'pipeline.json', state)
    experiment = out/('experiment02' if args.resume_after_transfer else 'experiment')
    commands = [
        ('prepare', [sys.executable, str(source/'prepare_data.py'),
                     '--checkpoint', '/root/workspace/GaussianGPT/checkpoints/gpt_vfront.ckpt',
                     '--collection', '/root/autodl-tmp/gaussiangpt_pf2x_round01_20261020/run01/collection',
                     '--output', str(out/'prepared')]),
        ('fit', [sys.executable, str(source/'run_experiment.py'),
                 '--data', str(out/'prepared'), '--manifest', str(source/'source_manifest.json'),
                 '--output', str(experiment), '--deadline-unix', str(deadline),
                 '--training-soft-seconds', '720']),
        ('validate', [sys.executable, str(source/'validate_experiment.py'),
                      '--run', str(experiment), '--data', str(out/'prepared'),
                      '--manifest', str(source/'source_manifest.json')]),
    ]
    if args.resume_after_transfer:
        commands = commands[1:]
    state['result_directory'] = str(experiment)
    try:
        for name, command in commands:
            phase = dict(name=name, command=command, started_unix=time.time())
            state['phases'].append(phase)
            write_json(out/'pipeline.json', state)
            with (out/(name+'.log')).open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=True)
                try:
                    code = process.wait(timeout=max(.01, deadline-time.time()))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError('N2 CPU runtime limit reached in '+name)
            phase.update(returncode=code, seconds=time.time()-phase['started_unix'])
            write_json(out/'pipeline.json', state)
            if code:
                raise RuntimeError(name+' returned '+str(code))
        state['status'] = 'complete'
    except Exception as error:
        state.update(status='failed', error=str(error))
    state.update(finished_unix=time.time(), seconds=time.time()-started)
    write_json(out/'pipeline.json', state)
    print(json.dumps(state, ensure_ascii=False))
    return 0 if state['status']=='complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
