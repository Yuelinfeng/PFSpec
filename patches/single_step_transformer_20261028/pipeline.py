"""Run the validated seven-method comparison after matched training finishes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
OLD=Path('/root/autodl-tmp/pfspec_repro_4090d_20261028')
PROJECT=OLD/'project'


def status(**value):
    tmp=ROOT/'pipeline.json.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(ROOT/'pipeline.json')


def run(phase,command):
    status(status='running',phase=phase,pid=os.getpid(),command=command)
    with (ROOT/(phase+'.log')).open('w') as log:
        subprocess.run(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    try:
        status(status='waiting',phase='training',pid=os.getpid())
        while True:
            path=ROOT/'train_01/progress.json'
            progress=json.loads(path.read_text()) if path.exists() else {}
            if progress.get('status')=='failed':raise RuntimeError('Training failed')
            if progress.get('status')=='complete':break
            time.sleep(5)
        run('selected_validation',[sys.executable,str(ROOT/'validate_models.py')])
        methods=['pf_w1']+[f'{kind}:s{seed}' for seed in (20261018,20261019,20261020)
                          for kind in ('mlp','transformer')]
        common=[sys.executable,str(ROOT/'online.py'),str(OLD/'compare/compare.py'),
                str(ROOT/'train_01'),str(ROOT/'fixed_assets.pt'),
                '--project',str(PROJECT),'--tree-source',str(OLD/'tree'),
                '--candidate-source',str(OLD/'candidate'),
                '--crops',str(OLD/'data/final_test_crops.pt'),
                '--methods',','.join(methods),'--prefixes','64,256',
                '--seeds','57101','--cache-capacity','2048']
        run('online_validation',common+['--out',str(ROOT/'online_validation'),
            '--sources','1','--budget','32','--validate-only'])
        run('online_main',common+['--out',str(ROOT/'online_main'),'--sources','30',
            '--budget','512','--max-seconds','2400',
            '--validated-run',str(ROOT/'online_validation')])
        progress=json.loads((ROOT/'online_main/progress.json').read_text())
        if progress['status']!='complete' or progress['completed_runs']!=420:
            raise RuntimeError('Online comparison incomplete')
        run('latency',[sys.executable,str(ROOT/'analyze.py'),'latency','--root',str(ROOT),
            '--original',str(PROJECT/'draft_a/runs/feature_target_continue_20261014/selected.pt')])
        run('analysis',[sys.executable,str(ROOT/'analyze.py'),'analyze','--root',str(ROOT),
            '--compare',str(OLD/'compare/compare.py')])
        status(status='complete',phase='complete',runs=420)
    except Exception:
        status(status='failed',error=traceback.format_exc());raise


if __name__=='__main__':main()
