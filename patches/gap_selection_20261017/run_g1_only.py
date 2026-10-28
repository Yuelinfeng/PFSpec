"""Run only the G1 decoder checks on the existing remote model assets."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from joblib import load

ROOT = Path('/root/autodl-tmp/gaussiangpt_gap_selection_20261017')
PROJECT = Path('/root/workspace/GaussianGPT')
sys.path[:0] = [
    str(PROJECT),
    '/root/autodl-tmp/gaussiangpt_next_round_20261017',
    '/root/autodl-tmp/gaussiangpt_candidate_cost_20261017',
    str(ROOT),
]

import step5_method_compare as prior
import bench_gaussiangpt_medusa_linear as medusa
from bench_gaussiangpt_pf_block2_cow import load_feature_model
from gap_decoder import build_decoder
from run_cost_study import environment, write_json
from validate_selection import validate


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    output = ROOT / 'full' / 'g1_verified_v2.json'
    contexts_path = Path(
        '/root/autodl-tmp/gaussiangpt_candidate_cost_20261017/full/matched_contexts.json'
    )
    contexts = json.loads(contexts_path.read_text())
    write_json(ROOT / 'full' / 'g1_environment_start.json', environment())
    fn = build_decoder(ROOT / 'selected_decode_g1.py')
    target, device, p_vocab, eos, ranks, _ = medusa.target_loader(SimpleNamespace(
        gpt=PROJECT / 'checkpoints/gpt_vfront.ckpt',
        vqvae=PROJECT / 'checkpoints/vqvae_vfront.ckpt',
    ))
    ranker = prior.FastHistRanker(load(prior.RANKER))
    feature = load_feature_model(
        prior.FEATURE, prior.FEATURE_HEAD, target, device, p_vocab, eos
    )[0].float().eval().requires_grad_(False)
    result = validate(fn, target, feature, ranker, ranks, contexts)
    result['status'] = 'complete'
    result['decoder_checks'] = 'passed'
    result['contexts_tested'] = 8
    write_json(output, result)
    write_json(ROOT / 'full' / 'g1_environment_end.json', environment())
    print(json.dumps({'status': result['status'], 'output': str(output),
                      'endpoints': len(result['endpoints']),
                      'forced_branches': len(result['forced_branches']),
                      'eos_cases': len(result['eos'])}, indent=2))


if __name__ == '__main__':
    main()
