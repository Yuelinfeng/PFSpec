"""Run the preselected T-set fixed-window comparisons."""
import argparse
import inspect
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
from draft_a.pf_ranker_eval import decode as position_decode
from gap_decoder import build_decoder, run
from run_cost_study import append, environment, write_json


METHODS = ('target', 'p_only', 'pf', 'gap', 'random')
SEED_BASES = (88201, 88202, 88203)
PREFIXES = (64, 256)
SHORT_BUDGET = 512
LONG_BUDGET = 2048
P_DEV_512 = None
P_DEV_2048 = None


def load_development_probabilities():
    global P_DEV_512, P_DEV_2048
    summary_path = ROOT / 'G5_summary.json'
    if not summary_path.is_file():
        raise FileNotFoundError(
            'G5 development summary is required before the T-set run'
        )
    summary = json.loads(summary_path.read_text())
    if summary['decision']['status'] != 'go_to_G6':
        raise RuntimeError('G5 development decision does not authorize a T-set run')
    P_DEV_512 = summary['online_512']['gap_route_use']['actual_pf_fraction']
    P_DEV_2048 = summary['long_2048']['gap_route_use']['actual_pf_fraction']
    if not (0.0 <= P_DEV_512 <= 1.0 and 0.0 <= P_DEV_2048 <= 1.0):
        raise ValueError((P_DEV_512, P_DEV_2048))


def build_target_decode():
    source = inspect.getsource(position_decode)
    assert source.count('def decode(') == 1
    assert source.count('return _add_rates(stats)') == 1
    source = source.replace('def decode(', 'def decode_with_output(', 1)
    source = source.replace(
        'return _add_rates(stats)', 'return _add_rates(stats), output', 1
    )
    namespace = dict(position_decode.__globals__)
    exec(compile(source, position_decode.__code__.co_filename, 'exec'), namespace)
    return namespace['decode_with_output']


def make_contexts(crops):
    contexts = []
    scene_ids = []
    for record_index, record in enumerate(crops):
        scene = record['scene']
        if scene not in scene_ids:
            scene_ids.append(scene)
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
            })
    if len(scene_ids) != 30 or len(contexts) != 60 or len(crops) != 30:
        raise ValueError((len(scene_ids), len(contexts), len(crops)))
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


def call_decoder(fn, target_decode, target, feature, ranker, ranks,
                 item, budget, seed, method):
    if method == 'target':
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            return target_decode(
                target, None, ranker, ranks, item['prompt'], budget, 0, seed
            )
    probability = P_DEV_2048 if budget >= LONG_BUDGET else P_DEV_512
    return run(
        fn, target, feature, ranker, ranks, item['prompt'], budget, seed,
        mode=method, probability=probability if method == 'random' else 1.0,
        trace=False,
    )


def run_timed(fn, target_decode, target, feature, ranker, ranks,
              item, budget, seed, method):
    torch.cuda.synchronize()
    started = time.perf_counter()
    stats, output = call_decoder(
        fn, target_decode, target, feature, ranker, ranks,
        item, budget, seed, method,
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


def run_matrix(fn, target_decode, target, feature, ranker, ranks, contexts,
               budget, path, phase, prefix_filter=None):
    selected = [
        (index, item) for index, item in enumerate(contexts)
        if prefix_filter is None or item['prefix'] == prefix_filter
    ]
    path.write_text('')
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
                    fn, target_decode, target, feature, ranker, ranks,
                    item, budget, seed, method,
                )
                append(path, make_row(
                    context_id, item, seed_base, seed, method, budget,
                    stats, output, seconds, phase,
                ))
                completed += 1
        if (local_index + 1) % 2 == 0:
            write_json(path.parent / 'progress.json', {
                'stage': phase,
                'contexts_done': local_index + 1,
                'contexts_total': len(selected),
                'runs_done': completed,
                'runs_total': total,
            })
            print(phase, local_index + 1, len(selected), completed, total, flush=True)
    return completed


def run_smoke(fn, target_decode, target, feature, ranker, ranks, contexts, output_dir):
    selected = list(enumerate(contexts[:4]))
    path = output_dir / 'smoke.jsonl'
    path.write_text('')
    completed = 0
    for context_id, item in selected:
        for seed_base in SEED_BASES[:2]:
            seed = seed_base + context_id * 1009
            for method in METHODS:
                stats, output, seconds = run_timed(
                    fn, target_decode, target, feature, ranker, ranks,
                    item, 128, seed, method,
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
    parser.add_argument('--run-root', type=Path, default=ROOT / 'g6_t_sources')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'g6_t_final')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    load_development_probabilities()
    run_root, output_dir = args.run_root, args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'environment_start.json', environment())

    crop_path = run_root / 'stage_c' / 't_crops.pt'
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
    target_decode = build_target_decode()
    ranker = prior.FastHistRanker(load(prior.RANKER))
    feature = load_feature_model(
        prior.FEATURE, prior.FEATURE_HEAD, target, device, p_vocab, eos
    )[0].float().eval().requires_grad_(False)
    fn = build_decoder(ROOT / 'selected_decode_g6.py')

    protocol = {
        'stage': 'G6 T confirmation',
        'data_role': '30 preselected sources disjoint from D0, D1, and prior Stage E/F sources',
        'source_manifest': str(ROOT / 'G5_source_manifest.json'),
        'source_ids': sorted({item['scene'] for item in contexts}),
        'methods': list(METHODS),
        'seed_bases': list(SEED_BASES),
        'prefixes_512': list(PREFIXES),
        'window_tokens': SHORT_BUDGET,
        'short_run_count': len(contexts) * len(SEED_BASES) * len(METHODS),
        'random_probability_512': P_DEV_512,
        'random_probability_source': 'G5 D1 observed PF route fractions, loaded before the first T-set measurement',
        'long_prefix': 256,
        'long_window_tokens': LONG_BUDGET,
        'long_run_count': 30 * len(SEED_BASES) * len(METHODS),
        'random_probability_2048': P_DEV_2048,
        'primary_outcome': '512-token Gap/PF source-geomean with source-cluster 95% bootstrap interval',
        'decision_rule': 'Positive support requires a 512-token Gap/PF point estimate above 1 and a source-bootstrap 95% lower bound above 1; a point estimate of at least 1.02 marks the prespecified practical scale.',
        'precision': 'FP32',
        'temperature': 0.9,
        'top_p': 0.9,
        'timing_boundary': 'decoder call including target cache allocation and prefix prefill; model and crop loading excluded',
        'no_new_model_training': True,
    }
    write_json(output_dir / 'protocol.json', protocol)
    smoke_runs = run_smoke(
        fn, target_decode, target, feature, ranker, ranks, contexts, output_dir
    )

    for method in METHODS:
        run_timed(fn, target_decode, target, feature, ranker, ranks,
                  contexts[0], 32, 640001, method)
    short_runs = run_matrix(
        fn, target_decode, target, feature, ranker, ranks, contexts,
        SHORT_BUDGET, output_dir / 'online_512.jsonl', 'online_512',
    )
    long_runs = run_matrix(
        fn, target_decode, target, feature, ranker, ranks, contexts,
        LONG_BUDGET, output_dir / 'long_2048.jsonl', 'long_2048',
        prefix_filter=256,
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
