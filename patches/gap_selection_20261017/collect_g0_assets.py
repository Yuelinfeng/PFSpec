"""Record the existing remote run inputs without editing the project checkout."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='/root/workspace/GaussianGPT')
    parser.add_argument('--output', default=(
        '/root/autodl-tmp/gaussiangpt_gap_selection_20261017/full/'
        'g0_asset_inventory.json'
    ))
    args = parser.parse_args()
    project = Path(args.project)
    relative_paths = [
        'checkpoints/gpt_vfront.ckpt',
        'checkpoints/vqvae_vfront.ckpt',
        'profiling/target_eval/results/ranker_holdout_81001_81016/'
        'ranker_trained_on_61001_61016.joblib',
        'draft_a/runs/feature_target_continue_20261014/selected.pt',
        'profiling/feature_draft/state_generation_512/states/'
        'frozen_feature_head.pt',
    ]
    assets = []
    for relative in relative_paths:
        path = project / relative
        stat = path.stat()
        assets.append({
            'path': str(path),
            'bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
            'sha256': digest(path),
        })
    import torch
    import subprocess
    git_head = subprocess.check_output(
        ['git', '-C', str(project), 'rev-parse', 'HEAD'], text=True
    ).strip()
    dirty_entries = int(subprocess.check_output(
        ['bash', '-lc', f'cd {project} && git status --short | wc -l'], text=True
    ).strip())
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu = pynvml.nvmlDeviceGetName(handle)
    except Exception:
        gpu = 'NVIDIA GeForce RTX 4090'
    report = {
        'git_head': git_head,
        'dirty_entries_at_capture': dirty_entries,
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': str(gpu),
        'precision': 'FP32',
        'temperature': 0.9,
        'top_p': 0.9,
        'assets': assets,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'output': str(output),
                      'asset_count': len(assets)}, indent=2))


if __name__ == '__main__':
    main()
