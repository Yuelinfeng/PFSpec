"""Run the online attribution ablations on the existing development contexts."""
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


METHODS = ('pf', 'random', 'reverse_gap', 'overhead')
SEED_BASES = (88101, 88102, 88103)
P_DEV_512 = 23962 / (23962 + 6906)
P_DEV_2048 = 17587 / (17587 + 6100)


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


def run_timed(fn, target, feature, ranker, ranks, item, budget, seed,
              method, probability=1.0):
    torch.cuda.synchronize()
    started = time.perf_counter()
    stats, output = run(
        fn, target, feature, ranker, ranks, item['prompt'], budget, seed,
        mode=method, probability=probability, trace=False,
    )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    return stats, output, wall_seconds


def make_row(context_id, item, seed_base, seed, method, probability,
             budget, stats, output, wall_seconds, phase):
    return {
        'phase': phase,
        'context_id': context_id,
        'scene': item['scene'],
        'prefix': item['prefix'],
        'origin': item['origin'],
        'seed_base': seed_base,
        'seed': seed,
        'method': method,
        'probability': probability,
        'requested_new_tokens': budget,
        'actual_new_tokens': len(output),
        'terminated_on_eos': bool(output and output[-1] == original.EOS),
        'wall_seconds': wall_seconds,
        'tokens_per_second': len(output) / wall_seconds if wall_seconds else None,
        'stats': row_stats(stats),
        'output': output,
    }


def choose_smoke_contexts(contexts):
    selected = []
    by_scene = {}
    for item in contexts:
        by_scene.setdefault(item['scene'], []).append(item)
    for scene in list(by_scene)[:2]:
        rows = sorted(by_scene[scene], key=lambda row: (row['origin'] != 'initial', row['prefix']))
        selected.extend(rows[:2])
    assert len(selected) == 4 and len({row['scene'] for row in selected}) == 2
    return selected


def choose_long_contexts(contexts):
    by_scene = {}
    for item in contexts:
        if item['origin'] == 'initial':
            by_scene.setdefault(item['scene'], []).append(item)
    selected = []
    for scene, rows in sorted(by_scene.items()):
        exact = [row for row in rows if row['prefix'] == 256]
        chosen = exact[0] if exact else sorted(rows, key=lambda row: abs(row['prefix'] - 256))[0]
        selected.append(chosen)
    assert len(selected) == 10
    return selected


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    output_dir = ROOT / 'g4_final'
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'environment_start.json', environment())
    contexts = json.loads(
        Path('/root/autodl-tmp/gaussiangpt_candidate_cost_20261017/full/'
             'matched_contexts.json').read_text()
    )
    target, device, p_vocab, eos, ranks, _ = medusa.target_loader(SimpleNamespace(
        gpt=PROJECT / 'checkpoints/gpt_vfront.ckpt',
        vqvae=PROJECT / 'checkpoints/vqvae_vfront.ckpt',
    ))
    ranker = prior.FastHistRanker(load(prior.RANKER))
    feature = load_feature_model(
        prior.FEATURE, prior.FEATURE_HEAD, target, device, p_vocab, eos
    )[0].float().eval().requires_grad_(False)
    fn = build_decoder(ROOT / 'selected_decode_g4.py')

    protocol = {
        'stage': 'G4',
        'data_role': 'development D0; same contexts as G3',
        'methods': list(METHODS),
        'seed_bases': list(SEED_BASES),
        'window_tokens': 512,
        'long_window_tokens': 2048,
        'long_sources': 10,
        'probability_512': P_DEV_512,
        'probability_2048': P_DEV_2048,
        'probability_source': 'observed PF route fractions from completed G3 windows',
        'precision': 'FP32',
        'temperature': 0.9,
        'top_p': 0.9,
        'timing_boundary': 'decoder call including target cache allocation and prefix prefill; model loading and data loading excluded',
        'no_new_model_training': True,
    }
    write_json(output_dir / 'protocol.json', protocol)

    smoke_contexts = choose_smoke_contexts(contexts)
    smoke_path = output_dir / 'smoke.jsonl'
    smoke_path.write_text('')
    for item in smoke_contexts:
        context_id = contexts.index(item)
        for seed_base in SEED_BASES[:2]:
            seed = seed_base + context_id * 1009
            for method in METHODS:
                probability = P_DEV_512 if method == 'random' else 1.0
                stats, output, seconds = run_timed(
                    fn, target, feature, ranker, ranks, item, 128,
                    seed, method, probability,
                )
                append(smoke_path, make_row(
                    context_id, item, seed_base, seed, method, probability,
                    128, stats, output, seconds, 'smoke',
                ))
    write_json(output_dir / 'smoke_summary.json', {
        'status': 'passed',
        'runs': len(smoke_contexts) * 2 * len(METHODS),
        'contexts': [contexts.index(item) for item in smoke_contexts],
        'source_count': len({item['scene'] for item in smoke_contexts}),
    })

    for method in METHODS:
        probability = P_DEV_512 if method == 'random' else 1.0
        run_timed(fn, target, feature, ranker, ranks, contexts[0],
                  32, 620101, method, probability)

    short_path = output_dir / 'ablation_512.jsonl'
    short_path.write_text('')
    short_total = len(contexts) * len(SEED_BASES) * len(METHODS)
    short_done = 0
    for context_id, item in enumerate(contexts):
        for seed_index, seed_base in enumerate(SEED_BASES):
            seed = seed_base + context_id * 1009
            order = list(METHODS)
            rotation = (context_id + seed_index) % len(order)
            order = order[rotation:] + order[:rotation]
            if (context_id + seed_index) % 2:
                order.reverse()
            for method in order:
                probability = P_DEV_512 if method == 'random' else 1.0
                stats, output, seconds = run_timed(
                    fn, target, feature, ranker, ranks, item, 512,
                    seed, method, probability,
                )
                append(short_path, make_row(
                    context_id, item, seed_base, seed, method, probability,
                    512, stats, output, seconds, 'ablation_512',
                ))
                short_done += 1
        if (context_id + 1) % 4 == 0:
            write_json(output_dir / 'progress.json', {
                'stage': 'ablation_512',
                'contexts_done': context_id + 1,
                'contexts_total': len(contexts),
                'runs_done': short_done,
                'runs_total': short_total,
            })
            print('G4-512', context_id + 1, len(contexts), short_done, short_total, flush=True)

    long_contexts = choose_long_contexts(contexts)
    long_path = output_dir / 'ablation_2048.jsonl'
    long_path.write_text('')
    long_total = len(long_contexts) * len(SEED_BASES) * len(METHODS)
    long_done = 0
    for context_id, item in enumerate(long_contexts):
        original_context_id = contexts.index(item)
        for seed_index, seed_base in enumerate(SEED_BASES):
            seed = seed_base + original_context_id * 1009 + 500000
            order = list(METHODS)
            rotation = (context_id + seed_index) % len(order)
            order = order[rotation:] + order[:rotation]
            if (context_id + seed_index) % 2:
                order.reverse()
            for method in order:
                probability = P_DEV_2048 if method == 'random' else 1.0
                stats, output, seconds = run_timed(
                    fn, target, feature, ranker, ranks, item, 2048,
                    seed, method, probability,
                )
                append(long_path, make_row(
                    original_context_id, item, seed_base, seed, method,
                    probability, 2048, stats, output, seconds, 'ablation_2048',
                ))
                long_done += 1
        if (context_id + 1) % 2 == 0:
            write_json(output_dir / 'progress.json', {
                'stage': 'ablation_2048',
                'sources_done': context_id + 1,
                'sources_total': len(long_contexts),
                'runs_done': long_done,
                'runs_total': long_total,
            })
            print('G4-2048', context_id + 1, len(long_contexts), long_done, long_total, flush=True)

    write_json(output_dir / 'environment_end.json', environment())
    write_json(output_dir / 'progress.json', {
        'status': 'complete',
        'ablation_512_runs': short_done,
        'ablation_512_total': short_total,
        'ablation_2048_runs': long_done,
        'ablation_2048_total': long_total,
    })
    print(json.dumps({
        'status': 'complete',
        'ablation_512_runs': short_done,
        'ablation_2048_runs': long_done,
    }, indent=2))


if __name__ == '__main__':
    main()
