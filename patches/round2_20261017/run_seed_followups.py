"""Add training-seed evidence after the first online and cost measurements."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/root/autodl-tmp/gaussiangpt_round2_20261017')
PYTHON='/root/miniconda3/envs/gaussiangpt/bin/python'


def main():
    deadline=time.time()+3600
    while not (ROOT/'A2_small_online.json').exists():
        state=json.loads((ROOT/'progress.json').read_text())
        if state.get('status')=='failed':return
        if time.time()>deadline:return
        time.sleep(10)
    commands=[('B1_training_seed_'+str(seed),[PYTHON,str(ROOT/'b1_train.py'),'--seed',str(seed)]) for seed in (20261019,20261020)]
    commands.append(('B1_fixed_contexts',[PYTHON,str(ROOT/'b1_fixed_contexts.py')]))
    for stage,command in commands:
        (ROOT/'progress.json').write_text(json.dumps(dict(stage=stage,status='running',unix_time=time.time())))
        with (ROOT/(stage+'.log')).open('w') as stream:
            run=subprocess.run(command,cwd='/root/workspace/GaussianGPT',stdout=stream,stderr=subprocess.STDOUT)
        if run.returncode:
            (ROOT/'progress.json').write_text(json.dumps(dict(stage=stage,status='failed',returncode=run.returncode)));return
    (ROOT/'progress.json').write_text(json.dumps(dict(stage='A0_A1_E0_B1_A2_screen',status='complete',
        online_training_seeds=[20261018],offline_training_seeds=[20261018,20261019,20261020],unix_time=time.time())))


if __name__=='__main__':main()
