#!/usr/bin/env python3
"""Continue a successful two-scene pilot through completion and BOS outputs."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run_adaptation_pipeline import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--pilot-pid', type=int, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    project = args.project_root.resolve()
    out = root / 'stage_f_full3d'
    status_path = out / 'pipeline.json'
    status = dict(status='running', phase='waiting_pilot', pid=os.getpid(), pilot_pid=args.pilot_pid,
                  stages=[], full_completion_samples=270, bos_samples=90)
    started = time.monotonic()
    env = dict(os.environ)
    env['PYTHONPATH'] = ':'.join(str(project / p) for p in ('.','draft_a','draft_a/pf_multiscene','profiling/target_eval'))
    source = project / 'draft_a/pf_multiscene'
    try:
        while True:
            path = Path(f'/proc/{args.pilot_pid}/cmdline')
            running = path.exists() and b'full_3d_eval.py' in path.read_bytes()
            progress_path = out / 'development_pilot/progress.json'
            if progress_path.exists():
                status['pilot'] = read_json(progress_path)
            write_json(status_path, status)
            if not running:
                break
            time.sleep(10)
        pilot = read_json(out / 'development_pilot/quality_summary.json')
        if pilot['status'] != 'complete' or pilot['success'] != 6 or pilot['failed']:
            raise RuntimeError('The complete six-output pilot must succeed before the full run')
        for name, records, extra in (
            ('test_completion', root / 'stage_e_final/final_test_crops.pt', []),
            ('bos_generation', root / 'stage_e_final/final_test_crops.pt',
             ['--kind','bos','--seeds',*map(str, range(51801,51831))]),
        ):
            destination = out / name
            summary_path = destination / 'quality_summary.json'
            if summary_path.exists() and read_json(summary_path).get('status') == 'complete':
                status['stages'].append(dict(name=name,status='reused',summary=str(summary_path)))
                continue
            command = [sys.executable, str(source / 'full_3d_eval.py'), '--root',str(destination),
                       '--records',str(records), *extra]
            if destination.exists():
                command.append('--resume')
            log = out / f'{name}.log'
            status.update(phase=name,log=str(log))
            with log.open('a') as handle:
                process = subprocess.Popen(command,cwd=project,env=env,stdout=handle,stderr=subprocess.STDOUT)
                status['stage_pid']=process.pid
                write_json(status_path,status)
                code=process.wait()
            status['stages'].append(dict(name=name,exit_code=code,summary=str(summary_path)))
            if code or not summary_path.exists():
                raise RuntimeError(f'{name} failed; see {log}')
            result=read_json(summary_path)
            if result['failed']:
                status.update(status='needs_output_review',phase=name,failed=result['failed'])
                write_json(status_path,status)
                return
        status.update(status='complete',phase='ready_for_quality_and_timing_analysis',
                      elapsed_seconds=time.monotonic()-started)
        status.pop('stage_pid',None)
        write_json(status_path,status)
    except Exception:
        status.update(status='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        write_json(status_path,status)
        raise


if __name__ == '__main__':
    main()
