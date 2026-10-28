"""Run cost measurements and the P1 condition test after timed comparison."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/root/autodl-tmp/gaussiangpt_round2_20261017')
PYTHON='/root/miniconda3/envs/gaussiangpt/bin/python'


def status(stage,state,**extra):
    (ROOT/'progress.json').write_text(json.dumps(dict(stage=stage,status=state,unix_time=time.time(),**extra)))


def main():
    deadline=time.time()+5400
    while not (ROOT/'b1_online/20261018/summary.json').exists():
        try:os.kill(518472,0)
        except ProcessLookupError:
            status('B1_online','failed',reason='Online process ended without a summary');return
        if time.time()>deadline:
            status('B1_online','waiting',reason='Expected completion window exceeded');return
        time.sleep(10)
    for stage,command,log in [
        ('B1_profile',[PYTHON,str(ROOT/'b1_online.py'),'--profile'],'b1_profile.log'),
        ('A2_train',[PYTHON,str(ROOT/'a2_pending_feature.py')],'a2_train.log'),
        ('A2_evaluate',[PYTHON,str(ROOT/'a2_evaluate.py')],'a2_evaluate.log')]:
        status(stage,'running')
        with (ROOT/log).open('w') as stream:
            run=subprocess.run(command,cwd='/root/workspace/GaussianGPT',stdout=stream,stderr=subprocess.STDOUT)
        if run.returncode:
            status(stage,'failed',returncode=run.returncode);return
    status('A0_A1_E0_B1_A2_screen','complete')


if __name__=='__main__':main()
