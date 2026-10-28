"""Run the G3 online development matrix and its long-window subset."""
import json
import inspect
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


METHODS = ('target', 'p_only', 'pf', 'gap')
SEED_BASES = (88101, 88102, 88103)


def call_decoder(fn, target_decode, target, feature, ranker, ranks,
                 prompt, budget, seed, method):
    if method == 'target':
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            return target_decode(target, None, ranker, ranks, prompt, budget, 0, seed)
    route = {'p_only': 'p_only', 'pf': 'pf', 'gap': 'gap'}[method]
    return run(fn, target, feature, ranker, ranks, prompt, budget, seed,
               mode=route, trace=False)


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
            'counts': selection['counts'],
            'branches': selection['branches'],
        }
    return result


def run_timed(fn, target_decode, target, feature, ranker, ranks,
              item, budget, seed, method):
    torch.cuda.synchronize()
    started = time.perf_counter()
    stats, output = call_decoder(
        fn, target_decode, target, feature, ranker, ranks,
        item['prompt'], budget, seed, method
    )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    return stats, output, wall_seconds


def build_target_decode():
    """Keep the original Target decoder and expose its already-built output."""
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


def make_row(context_id, item, seed_base, seed, method, budget,
             stats, output, wall_seconds, phase):
    return {
        'phase': phase,
        'context_id': context_id,
        'scene': item['scene'],
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
        chosen = exact[0] if exact else sorted(rows, key=lambda row: abs(row['prefix']-256))[0]
        selected.append(chosen)
    assert len(selected) == 10
    return selected


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    output_dir = ROOT / 'g3_final'
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
    target_decode = build_target_decode()
    ranker = prior.FastHistRanker(load(prior.RANKER))
    feature = load_feature_model(
        prior.FEATURE, prior.FEATURE_HEAD, target, device, p_vocab, eos
    )[0].float().eval().requires_grad_(False)
    fn = build_decoder(ROOT / 'selected_decode_g3.py')

    protocol = {
        'stage': 'G3',
        'data_role': 'development D0; same 56 prefixes used by B1',
        'methods': list(METHODS),
        'seed_bases': list(SEED_BASES),
        'window_tokens': 512,
        'full_run_count': len(contexts) * len(SEED_BASES) * len(METHODS),
        'long_sources': len({row['scene'] for row in choose_long_contexts(contexts)}),
        'long_window_tokens': 2048,
        'long_methods': ['p_only', 'pf', 'gap'],
        'precision': 'FP32',
        'temperature': 0.9,
        'top_p': 0.9,
        'timing_boundary': 'decoder call including target cache allocation and prefix prefill; model loading and data loading excluded',
        'source': '/root/autodl-tmp/gaussiangpt_candidate_cost_20261017/full/matched_contexts.json',
        'no_new_model_training': True,
    }
    write_json(output_dir / 'protocol.json', protocol)

    smoke_contexts = choose_smoke_contexts(contexts)
    smoke_path = output_dir / 'smoke.jsonl'
    smoke_path.write_text('')
    for context_id, item in enumerate(smoke_contexts):
        actual_context = contexts.index(item)
        for seed_base in SEED_BASES[:2]:
            seed = seed_base + actual_context * 1009
            for method in METHODS:
                stats, output, seconds = run_timed(
                    fn, target_decode, target, feature, ranker, ranks,
                    item, 128, seed, method
                )
                row = make_row(actual_context, item, seed_base, seed, method, 128,
                               stats, output, seconds, 'smoke')
                append(smoke_path, row)
    assert smoke_path.stat().st_size > 0
    write_json(output_dir / 'smoke_summary.json', {
        'status': 'passed',
        'runs': len(smoke_contexts) * 2 * len(METHODS),
        'contexts': [contexts.index(item) for item in smoke_contexts],
        'source_count': len({item['scene'] for item in smoke_contexts}),
    })

    warm_item = contexts[0]
    for method in METHODS:
        run_timed(fn, target_decode, target, feature, ranker, ranks,
                  warm_item, 32, 620001, method)

    full_path = output_dir / 'online_512.jsonl'
    full_path.write_text('')
    total = protocol['full_run_count']
    completed = 0
    for context_id, item in enumerate(contexts):
        for seed_index, seed_base in enumerate(SEED_BASES):
            seed = seed_base + context_id * 1009
            order = list(METHODS)
            rotation = (context_id + seed_index) % len(order)
            order = order[rotation:] + order[:rotation]
            if (context_id + seed_index) % 2:
                order.reverse()
            for method in order:
                stats, output, seconds = run_timed(
                    fn, target_decode, target, feature, ranker, ranks,
                    item, 512, seed, method
                )
                append(full_path, make_row(
                    context_id, item, seed_base, seed, method, 512,
                    stats, output, seconds, 'online_512',
                ))
                completed += 1
        if (context_id + 1) % 4 == 0:
            write_json(output_dir / 'progress.json', {
                'stage': 'online_512',
                'contexts_done': context_id + 1,
                'contexts_total': len(contexts),
                'runs_done': completed,
                'runs_total': total,
            })
            print('G3-512', context_id + 1, len(contexts), completed, total, flush=True)

    long_contexts = choose_long_contexts(contexts)
    long_path = output_dir / 'long_2048.jsonl'
    long_path.write_text('')
    long_methods = ('p_only', 'pf', 'gap')
    long_total = len(long_contexts) * len(SEED_BASES) * len(long_methods)
    long_completed = 0
    for context_id, item in enumerate(long_contexts):
        original_context_id = contexts.index(item)
        for seed_index, seed_base in enumerate(SEED_BASES):
            seed = seed_base + original_context_id * 1009 + 500000
            order = list(long_methods)
            rotation = (context_id + seed_index) % len(order)
            order = order[rotation:] + order[:rotation]
            if (context_id + seed_index) % 2:
                order.reverse()
            for method in order:
                stats, output, seconds = run_timed(
                    fn, target_decode, target, feature, ranker, ranks,
                    item, 2048, seed, method
                )
                append(long_path, make_row(
                    original_context_id, item, seed_base, seed, method, 2048,
                    stats, output, seconds, 'long_2048',
                ))
                long_completed += 1
        if (context_id + 1) % 2 == 0:
            write_json(output_dir / 'progress.json', {
                'stage': 'long_2048',
                'sources_done': context_id + 1,
                'sources_total': len(long_contexts),
                'runs_done': long_completed,
                'runs_total': long_total,
            })
            print('G3-2048', context_id + 1, len(long_contexts), long_completed, long_total, flush=True)

    write_json(output_dir / 'environment_end.json', environment())
    write_json(output_dir / 'progress.json', {
        'status': 'complete',
        'online_512_runs': completed,
        'online_512_total': total,
        'long_2048_runs': long_completed,
        'long_2048_total': long_total,
    })
    print(json.dumps({
        'status': 'complete',
        'online_512_runs': completed,
        'long_2048_runs': long_completed,
    }, indent=2))


if __name__ == '__main__':
    main()
