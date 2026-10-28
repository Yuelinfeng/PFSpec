"""Run online validation and measurements after the matched training finishes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
PREVIOUS = Path('/root/autodl-tmp/pfspec_repro_4090d_20261028')
PROJECT = PREVIOUS / 'project'
TRAINING = ROOT / 'train_01'


def status(**values):
    temporary = ROOT / 'pipeline.json.tmp'
    temporary.write_text(json.dumps(values, indent=2) + '\n')
    temporary.replace(ROOT / 'pipeline.json')


def run(phase, command):
    status(status='running', phase=phase, command=command, pid=os.getpid())
    with (ROOT / (phase + '.log')).open('w') as log:
        subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    try:
        status(status='waiting', phase='training', pid=os.getpid())
        while True:
            path = TRAINING / 'progress.json'
            progress = json.loads(path.read_text()) if path.exists() else {}
            if progress.get('status') == 'failed':
                raise RuntimeError('Training failed; see train_01/progress.json')
            if progress.get('status') == 'complete':
                break
            time.sleep(5)
        common = [sys.executable, str(ROOT/'online.py'), str(PREVIOUS/'compare/compare.py'),
                  str(TRAINING), '--project', str(PROJECT), '--tree-source', str(PREVIOUS/'tree'),
                  '--candidate-source', str(PREVIOUS/'candidate'),
                  '--crops', str(PREVIOUS/'data/final_test_crops.pt'),
                  '--methods', 'pf_w1,blocks0,blocks3,blocks5', '--prefixes', '64,256',
                  '--seeds', '57101', '--cache-capacity', '2048']
        run('online_validation', common + ['--out', str(ROOT/'online_validation'),
                                           '--sources', '1', '--budget', '32', '--validate-only'])
        run('online_main', common + ['--out', str(ROOT/'online_main'), '--sources', '30',
                                     '--budget', '512', '--max-seconds', '1800',
                                     '--validated-run', str(ROOT/'online_validation')])
        progress = json.loads((ROOT/'online_main/progress.json').read_text())
        if progress['status'] != 'complete' or progress['completed_runs'] != 240:
            raise RuntimeError('Online comparison is incomplete')
        sample = next((ROOT/'training_data/data/val').glob('*.pt'))
        run('latency', [sys.executable, str(ROOT/'measure.py'), 'latency',
                        '--training-root', str(TRAINING), '--sample', str(sample),
                        '--original', str(PROJECT/'draft_a/runs/feature_target_continue_20261014/selected.pt'),
                        '--output', str(ROOT/'latency.json')])
        run('analysis', [sys.executable, str(ROOT/'measure.py'), 'summarize',
                         '--training-root', str(TRAINING), '--online', str(ROOT/'online_main'),
                         '--compare', str(PREVIOUS/'compare/compare.py'), '--output', str(ROOT/'analysis.json')])
        status(status='complete', phase='complete', runs=240)
    except Exception:
        status(status='failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
