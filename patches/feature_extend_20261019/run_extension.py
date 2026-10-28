"""Continue the existing step12000 Feature MLP to exact step24000/48000."""

import json
import os
from pathlib import Path
import sys
import traceback

PROJECT = Path('/root/workspace/GaussianGPT')
ROOT = Path('/root/autodl-tmp/gaussiangpt_feature_extend_20261019')
START = PROJECT / 'draft_a/runs/feature_target_continue_20261014/selected.pt'
DATA = PROJECT / 'draft_a/runs/feature_target_continue_20261014'
sys.path.insert(0, str(PROJECT))


def main():
    import torch
    from draft_a.feature_target_v2 import train, save, self_test
    os.chdir(PROJECT)
    if (ROOT / 'pipeline.json').exists():
        raise FileExistsError('An experiment already exists at this output location')
    value = torch.load(START, map_location='cpu', weights_only=True)
    if value['step'] != 12000:
        raise ValueError('Expected the original exact step12000 weights')
    data = json.loads((DATA / 'data_plan.json').read_text())
    if data['status'] != 'complete':
        raise ValueError('Original training data is incomplete')
    for split in ('train', 'val'):
        if not all(Path(row['path']).is_file() for row in data['splits'][split]['records']):
            raise FileNotFoundError('Original data files are missing')
    ROOT.mkdir(parents=True, exist_ok=True)
    save(ROOT / 'experiment_plan.json', dict(
        initial_checkpoint=str(START), initial_step=12000, final_step=48000,
        requested_steps=[24000, 48000], additional_steps=36000,
        data_root=str(DATA), objective='original raw Target CE + 2 * policy TV',
        learning_rate_start=1e-5, learning_rate_end=1e-6,
        schedule='one cosine schedule over 36000 additional updates',
        optimizer='fresh AdamW once; original checkpoint has model weights only',
        evaluation_interval=1200, training_seed=24689,
        precision='BF16 training, FP32 distribution evaluation and decoding',
        training_data_unchanged=True, target_and_ranker_unchanged=True,
        evaluation='compare exact requested weights; best validation selection reported separately',
        process_id=os.getpid()))
    save(ROOT / 'pipeline.json', dict(status='running', phase='train', pid=os.getpid()))
    try:
        self_test()
        train(ROOT / 'training', chunk=256, initial_path=START, data_root=DATA,
              phase_plan=[dict(name='extended_policy_tv', lr=1e-5, weight=2.,
                               seed=24689, steps=36000)],
              eval_every=1200, train_probe=16)
        for step in (24000, 48000):
            p = ROOT / 'training' / f'step_{step:04d}.pt'
            ckpt = torch.load(p, map_location='cpu', weights_only=True)
            if ckpt['step'] != step:
                raise ValueError('Requested training step is missing')
            # User-facing exact-step names; do not substitute best-validation weights.
            import shutil
            shutil.copyfile(p, ROOT / f'{step}.pt')
        save(ROOT / 'pipeline.json', dict(status='trained', phase='evaluation_pending',
                                          pid=os.getpid(), exact_steps=[24000, 48000]))
    except Exception:
        save(ROOT / 'pipeline.json', dict(status='failed', error=traceback.format_exc(), pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()
