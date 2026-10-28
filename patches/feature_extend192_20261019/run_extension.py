"""Continue exact step48000 weights to step96000 and step192000."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

PROJECT = Path('/root/workspace/GaussianGPT')
ROOT = Path('/root/autodl-tmp/gaussiangpt_feature_extend192_20261019')
START = Path('/root/autodl-tmp/gaussiangpt_feature_extend_20261019/48000.pt')
DATA = PROJECT / 'draft_a/runs/feature_target_continue_20261014'
sys.path.insert(0, str(PROJECT))


def main():
    import torch
    from draft_a.feature_target_v2 import train, save, self_test
    os.chdir(PROJECT)
    if (ROOT / 'pipeline.json').exists():
        raise FileExistsError('An experiment already exists at this output location')
    value = torch.load(START, map_location='cpu', weights_only=True)
    if value['step'] != 48000:
        raise ValueError('Expected the exact step48000 weights')
    data = json.loads((DATA / 'data_plan.json').read_text())
    if data['status'] != 'complete':
        raise ValueError('Original training data is incomplete')
    for split in ('train', 'val'):
        if not all(Path(row['path']).is_file() for row in data['splits'][split]['records']):
            raise FileNotFoundError('Original data files are missing')
    ROOT.mkdir(parents=True, exist_ok=True)
    save(ROOT / 'experiment_plan.json', dict(
        initial_checkpoint=str(START), initial_step=48000, final_step=192000,
        requested_steps=[96000, 192000], additional_steps=144000,
        data_root=str(DATA), objective='original raw Target CE + 2 * policy TV',
        learning_rate_start=1e-6, learning_rate_end=1e-7,
        schedule='one cosine schedule over 144000 additional updates; no restart at step96000',
        optimizer='fresh AdamW once; step48000 checkpoint has model weights only',
        evaluation_interval=1200, training_seed=24690,
        precision='BF16 training, FP32 distribution evaluation and decoding',
        training_data_unchanged=True, target_and_ranker_unchanged=True,
        evaluation='31 online curve nodes plus exact steps12000/24000/48000/96000/192000',
        process_id=os.getpid()))
    save(ROOT / 'pipeline.json', dict(status='running', phase='train', pid=os.getpid()))
    try:
        self_test()
        train(ROOT / 'training', chunk=256, initial_path=START, data_root=DATA,
              phase_plan=[dict(name='long_extended_policy_tv', lr=1e-6, weight=2.,
                               seed=24690, steps=144000)],
              eval_every=1200, train_probe=16)
        for step in (96000, 192000):
            p = ROOT / 'training' / f'step_{step:04d}.pt'
            ckpt = torch.load(p, map_location='cpu', weights_only=True)
            if ckpt['step'] != step:
                raise ValueError('Requested training step is missing')
            shutil.copyfile(p, ROOT / f'{step}.pt')
        save(ROOT / 'pipeline.json', dict(status='trained', phase='evaluation_pending',
                                          pid=os.getpid(), exact_steps=[96000, 192000]))
        with (ROOT / 'evaluation.log').open('a') as log:
            subprocess.run([sys.executable, '-u', str(ROOT / 'evaluate_extension.py')],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    except Exception:
        phase = json.loads((ROOT / 'pipeline.json').read_text()).get('phase')
        save(ROOT / 'pipeline.json', dict(status='failed', phase=phase,
                                          error=traceback.format_exc(), pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()
