"""Run paired one-round measurements on the 56 B1 development prefixes."""
import argparse
import json
from collections import defaultdict
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
from gap_decoder import Selector, build_decoder, run
from gap_gate import PF_K2, P_ONLY, choose_gap_route
from run_cost_study import append, environment, write_json
from validate_selection import position_with_output


def load_seed_map():
    source = Path(
        '/root/autodl-tmp/gaussiangpt_candidate_cost_20261017/full/matched_rounds.jsonl'
    )
    seeds = defaultdict(dict)
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['mode'] == 'pf':
                seeds[row['context_id']][row['trial']] = row['seed']
    return seeds


def load_gap_map(contexts):
    source = Path(
        '/root/autodl-tmp/gaussiangpt_feature_value_20261017/full/pairs.jsonl'
    )
    gaps = {}
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            context_id = row['context_id']
            assert contexts[context_id]['scene'] == row['scene']
            assert contexts[context_id]['prefix'] == row['prefix']
            assert contexts[context_id]['origin'] == row['origin']
            previous = int(row['rounds'][0]['pre_features']['previous_position'])
            assert previous == int(contexts[context_id]['prompt'][-2])
            gap = int(row['rounds'][0]['pre_features']['p1_gap'])
            if context_id in gaps:
                assert gaps[context_id] == gap
            else:
                gaps[context_id] = gap
    assert len(gaps) == len(contexts)
    return gaps


def route_microbenchmark():
    count = 200000
    results = {}
    for name, function in (
        ('pure_route', lambda: choose_gap_route(101, 100, 256)),
        ('selector_path', lambda: selector.choose((101, 4, 0), 100, 256)),
    ):
        samples = []
        for _ in range(5):
            selector = Selector('gap', 1.0, 901)
            start = time.perf_counter()
            for _ in range(count):
                function()
            samples.append((time.perf_counter() - start) / count * 1e6)
        results[name] = {
            'repeats': 5,
            'calls_per_repeat': count,
            'median_microseconds_per_call': sorted(samples)[len(samples)//2],
            'all_microseconds_per_call': samples,
        }
    return results


def comparable_stats(stats):
    keys = (
        'target_calls', 'generated_tokens', 'rounds', 'position_accepted',
        'position_verified', 'feature_accepted', 'feature_verified',
        'target_input_tokens', 'draft_input_tokens', 'position_fallbacks',
    )
    return {key: stats[key] for key in keys}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-subdir', default='g2_retry')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    output_dir = ROOT / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'g2_environment_start.json', environment())
    contexts = json.loads(
        Path('/root/autodl-tmp/gaussiangpt_candidate_cost_20261017/full/'
             'matched_contexts.json').read_text()
    )
    seeds = load_seed_map()
    gaps = load_gap_map(contexts)
    assert len(contexts) == 56 and len(seeds) == 56 and len(gaps) == 56
    fn = build_decoder(ROOT / 'selected_decode_g2.py')
    target, device, p_vocab, eos, ranks, _ = medusa.target_loader(SimpleNamespace(
        gpt=PROJECT / 'checkpoints/gpt_vfront.ckpt',
        vqvae=PROJECT / 'checkpoints/vqvae_vfront.ckpt',
    ))
    ranker = prior.FastHistRanker(load(prior.RANKER))
    feature = load_feature_model(
        prior.FEATURE, prior.FEATURE_HEAD, target, device, p_vocab, eos
    )[0].float().eval().requires_grad_(False)

    methods = ('p_only', 'pf', 'gap')
    budget = 4
    prompt = contexts[0]['prompt']
    for mode in methods:
        run(fn, target, feature, ranker, ranks, prompt, budget, 610001,
            mode=mode, trace=True, max_rounds=1)

    protocol = {
        'stage': 'G2',
        'context_count': len(contexts),
        'source_count': len({row['scene'] for row in contexts}),
        'seeds_per_context': 8,
        'runs': len(contexts) * 8 * len(methods),
        'methods': list(methods),
        'budget_tokens': budget,
        'max_rounds': 1,
        'seed_source': 'A1-A3 matched_rounds.jsonl, PF rows, trial 0..7',
        'gap_source': 'B1 first-round pre_features.p1_gap',
        'timing': 'unprofiled decoder wall time with CUDA synchronization around each run',
    }
    write_json(output_dir / 'g2_protocol.json', protocol)
    run_path = output_dir / 'g2_matched_rounds.jsonl'
    run_path.write_text('')
    profile_path = output_dir / 'g2_profile_subset.jsonl'
    profile_path.write_text('')
    profile_contexts = ([i for i in range(56) if gaps[i] == 1][:8] +
                        [i for i in range(56) if gaps[i] > 1][:8])
    assert len(profile_contexts) == 16

    totals = 0
    for context_id, item in enumerate(contexts):
        context_seed_map = seeds[context_id]
        assert set(context_seed_map) == set(range(8))
        for trial in range(8):
            seed = context_seed_map[trial]
            order = list(methods)
            rotate = (context_id + trial) % len(order)
            order = order[rotate:] + order[:rotate]
            if (context_id + trial) % 2:
                order.reverse()
            round_rows = {}
            for method in order:
                torch.cuda.synchronize()
                started = time.perf_counter()
                stats, output = run(
                    fn, target, feature, ranker, ranks, item['prompt'], budget,
                    seed, mode=method, trace=True, max_rounds=1,
                )
                torch.cuda.synchronize()
                wall_seconds = time.perf_counter() - started
                trace = stats['selection']['trace']
                assert len(trace) == 1 and stats['rounds'] == 1
                event = trace[0]
                if method == 'gap':
                    expected_route = choose_gap_route(
                        event['candidate_position'], item['prompt'][-2], budget
                    ).route
                    expected = 'pf' if expected_route == PF_K2 else 'p_only'
                    assert event['path'] == expected
                    assert event['gap'] == gaps[context_id]
                    assert event['candidate_position'] - item['prompt'][-2] == event['gap']
                row = {
                    'context_id': context_id,
                    'scene': item['scene'],
                    'prefix': item['prefix'],
                    'origin': item['origin'],
                    'trial': trial,
                    'seed': seed,
                    'method': method,
                    'gap': event['gap'],
                    'route': event['path'],
                    'route_reason': event.get('reason'),
                    'candidate_position': event['candidate_position'],
                    'branch': event['branch'],
                    'generated_tokens': stats['generated_tokens'],
                    'decoder_seconds': stats['seconds'],
                    'outer_wall_seconds': wall_seconds,
                    'tokens_per_second': stats['tokens_per_second'],
                    'stats': comparable_stats(stats),
                    'output': output,
                }
                round_rows[method] = (row, output, comparable_stats(stats))
                append(run_path, row)
                totals += 1

            gap = round_rows['gap'][0]['gap']
            endpoint = 'pf' if gap == 1 else 'p_only'
            assert round_rows['gap'][1] == round_rows[endpoint][1]
            assert round_rows['gap'][2] == round_rows[endpoint][2]

            if trial == 0 and context_id in profile_contexts:
                for method in methods:
                    stats, output = run(
                        fn, target, feature, ranker, ranks, item['prompt'], budget,
                        seed, mode=method, trace=True, max_rounds=1, profile=True,
                    )
                    profile_row = {
                        'context_id': context_id,
                        'scene': item['scene'],
                        'prefix': item['prefix'],
                        'seed': seed,
                        'method': method,
                        'gap': stats['selection']['trace'][0]['gap'],
                        'route': stats['selection']['trace'][0]['path'],
                        'elapsed_seconds_instrumented': stats['seconds'],
                        'stage_profile': stats.get('stage_profile', {}),
                    }
                    append(profile_path, profile_row)
        if (context_id + 1) % 8 == 0:
            write_json(output_dir / 'g2_progress.json', {
                'contexts_done': context_id + 1,
                'contexts_total': len(contexts),
                'runs_done': totals,
                'runs_total': protocol['runs'],
            })
            print('G2', context_id + 1, len(contexts), totals, flush=True)

    write_json(output_dir / 'g2_gate_microbenchmark.json', route_microbenchmark())
    write_json(output_dir / 'g2_environment_end.json', environment())
    write_json(output_dir / 'g2_progress.json', {
        'status': 'complete',
        'contexts_done': len(contexts),
        'runs_done': totals,
        'runs_total': protocol['runs'],
        'profile_runs': len(profile_contexts) * len(methods),
    })
    print(json.dumps({'status': 'complete', 'runs': totals,
                      'profile_runs': len(profile_contexts) * len(methods)}, indent=2))


if __name__ == '__main__':
    main()
