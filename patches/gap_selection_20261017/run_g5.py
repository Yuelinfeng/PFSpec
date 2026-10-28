"""Run the preselected D1 online comparison on fresh scene sources."""
import argparse
import json
from pathlib import Path
import sys
import time
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
from draft_a import pf_feature_unified as original
from gap_decoder import build_decoder, run
from run_cost_study import append, environment, write_json


METHODS = ('p_only', 'pf', 'gap')
SEED_BASES = (88111, 88112, 88113)
PREFIXES = (64, 256)
SHORT_BUDGET = 512
LONG_BUDGET = 2048


def make_contexts(crops):
    contexts = []
    source_ids = []
    for record_index, record in enumerate(crops):
        scene = record['scene']
        if scene not in source_ids:
            source_ids.append(scene)
        tokens = record['tokens'].long().cpu().tolist()
        for prefix in PREFIXES:
            if prefix > len(tokens):
                raise ValueError(f'{scene} crop has {len(tokens)} tokens, needs {prefix}')
            contexts.append({
                'scene': scene,
                'crop_seed': int(record['crop_seed']),
                'crop_record': record_index,
                'prefix': prefix,
                'origin': 'initial',
                'prompt': [original.EOS] + tokens[:prefix],
                'body_tokens': len(tokens),
            })
    if len(source_ids) != len(crops):
        raise ValueError('Expected exactly one crop record per selected source')
    if len(contexts) != 20 or len(source_ids) != 10:
        raise ValueError((len(contexts), len(source_ids)))
    return contexts


def row_stats(stats):
    keys = (
        'generated_tokens', 'tokens_per_second', 'target_calls', 'target_input_tokens',
        'draft_calls', 'draft_input_tokens', 'ranker_calls', 'rounds',
        'position_proposed', 'position_verified', 'position_accepted',
        'feature_proposed', 'feature_verified', 'feature_accepted',
        'position_fallbacks', 'seconds', 'eos',
    )
    result = {key: stats.get(key) for key in keys}
    selection = stats.get('selection')
    if selection is not None:
        result['selection'] = {
            'mode': selection['mode'],
            'probability': selection['probability'],
            'counts': selection['counts'],
            'branches': selection['branches'],
        }
    return result


def run_timed(fn, target, feature, ranker, ranks, item, budget, seed, method):
    torch.cuda.synchronize()
    started = time.perf_counter()
    stats, output = run(
        fn, target, feature, ranker, ranks, item['prompt'], budget, seed,
        mode=method, trace=False,
    )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    return stats, output, wall_seconds


def make_row(context_id, item, seed_base, seed, method, budget,
             stats, output, wall_seconds, phase):
    return {
        'phase': phase,
        'context_id': context_id,
        'scene': item['scene'],
        'crop_seed': item['crop_seed'],
        'crop_record': item['crop_record'],
        'prefix': item['prefix'],
        'origin': item['origin'],
        'seed_base': seed_base,
        'seed': seed,
        'method': method,
        'requested_new_tokens': budget,
        'actual_new_tokens': len(output),
        'terminated_on_eos': bool(output and output[-1] == original.EOS),
        'wall_seconds': wall_seconds,
        'tokens_per_second': len(output) / wall_seconds if wall_seconds else None,
        'stats': row_stats(stats),
        'output': output,
    }


def run_matrix(fn, target, feature, ranker, ranks, contexts, budget,
               output_path, phase, prefix_filter=None):
    selected = [
        (index, item) for index, item in enumerate(contexts)
        if prefix_filter is None or item['prefix'] == prefix_filter
    ]
    output_path.write_text('')
    total = len(selected) * len(SEED_BASES) * len(METHODS)
    completed = 0
    for local_index, (context_id, item) in enumerate(selected):
        for seed_index, seed_base in enumerate(SEED_BASES):
            offset = 500000 if budget == LONG_BUDGET else 0
            seed = seed_base + context_id * 1009 + offset
            order = list(METHODS)
            rotation = (local_index + seed_index) % len(order)
            order = order[rotation:] + order[:rotation]
            if (local_index + seed_index) % 2:
                order.reverse()
            for method in order:
                stats, output, seconds = run_timed(
                    fn, target, feature, ranker, ranks, item, budget, seed, method
                )
                append(output_path, make_row(
                    context_id, item, seed_base, seed, method, budget,
                    stats, output, seconds, phase,
                ))
                completed += 1
        if (local_index + 1) % 2 == 0:
            write_json(output_path.parent / 'progress.json', {
                'stage': phase,
                'contexts_done': local_index + 1,
                'contexts_total': len(selected),
                'runs_done': completed,
                'runs_total': total,
            })
            print(phase, local_index + 1, len(selected), completed, total, flush=True)
    return completed


def run_smoke(fn, target, feature, ranker, ranks, contexts, output_dir):
    selected = list(enumerate(contexts[:4]))
    path = output_dir / 'smoke.jsonl'
    path.write_text('')
    completed = 0
    for context_id, item in selected:
        for seed_base in SEED_BASES[:2]:
            seed = seed_base + context_id * 1009
            for method in METHODS:
                stats, output, seconds = run_timed(
                    fn, target, feature, ranker, ranks, item, 128, seed, method
                )
                append(path, make_row(
                    context_id, item, seed_base, seed, method, 128,
                    stats, output, seconds, 'smoke',
                ))
                completed += 1
    write_json(output_dir / 'smoke_summary.json', {
        'status': 'passed',
        'runs': completed,
        'contexts': [index for index, _ in selected],
        'source_count': len({item['scene'] for _, item in selected}),
    })
    return completed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=ROOT / 'g5_d1_full_assets')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'g5_d1_full_assets_final')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    run_root = args.run_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'environment_start.json', environment())

    crop_path = run_root / 'stage_c' / 'd1_crops.pt'
    crops = torch.load(crop_path, map_location='cpu', weights_only=False)
    contexts = make_contexts(crops)
    write_json(output_dir / 'contexts.json', [
        {key: value for key, value in item.items() if key != 'prompt'}
        for item in contexts
    ])

    target, device, p_vocab, eos, ranks, _ = medusa.target_loader(SimpleNamespace(
        gpt=PROJECT / 'checkpoints/gpt_vfront.ckpt',
        vqvae=PROJECT / 'checkpoints/vqvae_vfront.ckpt',
    ))
    ranker = prior.FastHistRanker(load(prior.RANKER))
    feature = load_feature_model(
        prior.FEATURE, prior.FEATURE_HEAD, target, device, p_vocab, eos
    )[0].float().eval().requires_grad_(False)
    fn = build_decoder(ROOT / 'selected_decode_g5.py')

    protocol = {
        'stage': 'G5 D1',
        'data_role': 'new development sources selected before performance runs',
        'source_manifest': str(ROOT / 'G5_source_manifest.json'),
        'source_ids': sorted({item['scene'] for item in contexts}),
        'methods': list(METHODS),
        'seed_bases': list(SEED_BASES),
        'prefixes_512': list(PREFIXES),
        'window_tokens': SHORT_BUDGET,
        'long_prefix': 256,
        'long_window_tokens': LONG_BUDGET,
        'short_run_count': len(contexts) * len(SEED_BASES) * len(METHODS),
        'long_run_count': 10 * len(SEED_BASES) * len(METHODS),
        'precision': 'FP32',
        'temperature': 0.9,
        'top_p': 0.9,
        'timing_boundary': 'decoder call including target cache allocation and prefix prefill; model and crop loading excluded',
        'decision_rule': 'Advance to G6 if the D1 2048-token Gap/PF source-geomean is at least 1.02 and its source-bootstrap 95% interval lower bound exceeds 1.00; treat 512-token and per-prefix results as scope diagnostics.',
        'no_new_model_training': True,
    }
    write_json(output_dir / 'protocol.json', protocol)
    smoke_runs = run_smoke(fn, target, feature, ranker, ranks, contexts, output_dir)

    warm_item = contexts[0]
    for method in METHODS:
        run_timed(fn, target, feature, ranker, ranks, warm_item, 32, 630001, method)

    short_runs = run_matrix(
        fn, target, feature, ranker, ranks, contexts, SHORT_BUDGET,
        output_dir / 'online_512.jsonl', 'online_512',
    )
    long_runs = run_matrix(
        fn, target, feature, ranker, ranks, contexts, LONG_BUDGET,
        output_dir / 'long_2048.jsonl', 'long_2048', prefix_filter=256,
    )
    write_json(output_dir / 'environment_end.json', environment())
    write_json(output_dir / 'progress.json', {
        'status': 'complete',
        'smoke_runs': smoke_runs,
        'online_512_runs': short_runs,
        'long_2048_runs': long_runs,
    })
    print(json.dumps({
        'status': 'complete',
        'online_512_runs': short_runs,
        'long_2048_runs': long_runs,
    }, indent=2))


if __name__ == '__main__':
    main()
