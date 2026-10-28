#!/usr/bin/env python3
"""Run existing multi-column scene generation with selectable sampling backends."""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path('/root/workspace/GaussianGPT')
TREE = Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
RESCUE = Path('/root/autodl-tmp/gaussiangpt_spec_next_20261022/source_run02')
COMPARISON = Path('/root/autodl-tmp/gaussiangpt_pf_comparison_20261025/source')
sys.path[:0] = [str(COMPARISON), str(TREE), str(RESCUE), str(TREE / 'reference'),
                str(TREE / 'reference/runtime'), str(PROJECT)]
os.chdir(PROJECT)

import torch
from joblib import load
from threadpoolctl import threadpool_limits
import generate_scene as scene
import method_adapters as api
import compare
from bench_fast_hgb import FastHistRanker
from model.gaussian_gpt import GaussianGPT
from model.gaussian_vqvae import GaussianVQVAE

EOS = 12096


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def load_models():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vq = GaussianVQVAE.load_from_checkpoint('checkpoints/vqvae_vfront.ckpt', map_location='cpu').eval()
    gpt = GaussianGPT.load_from_checkpoint('checkpoints/gpt_vfront.ckpt', vqvae=vq, map_location='cpu')
    gpt = gpt.cuda().float().eval().requires_grad_(False)
    model = gpt.gpt._orig_mod if hasattr(gpt.gpt, '_orig_mod') else gpt.gpt
    core = model.transformer
    ranker = FastHistRanker(load(api.RANKER_PATH))
    ranks_tensor = core.sparse_position_rank_map.detach().cpu().long()
    context = dict(project=PROJECT, device=torch.device('cuda'), target_model=model,
                   target=core, ranker=ranker, ranks=ranks_tensor.tolist(),
                   ranks_tensor=ranks_tensor, eos=EOS, p_vocab=8000, feature_vocab=4096,
                   cache_for=api.importlib.import_module('draft_a.feature_acceptance_eval').cache_for,
                   target_decode=api._decode_with_output(api.importlib.import_module('draft_a.pf_ranker_eval').decode))
    return gpt, model, context


def scene_args(gpt, size, seed):
    _, cx, cy, z = scene._validate_and_derive_geometry(gpt)
    return dict(scene_cols_x=size, scene_cols_y=size, context_chunks_x=cx, context_chunks_y=cy,
                z=z, outpainting_temperature=.9, outpainting_top_k=None, outpainting_top_p=.9,
                bootstrap_temperature=.9, bootstrap_top_k=None, bootstrap_top_p=.9,
                bootstrap2_temperature=.9, bootstrap2_top_k=None, bootstrap2_top_p=.9,
                feature_temperature=None, feature_top_k=None, feature_top_p=None,
                x_offset=5, y_offset=5, y_range=5, no_empty_columns=False,
                resample_empty_columns=True, enable_bootstrap=True,
                early_stop_min_occupancy=None, max_empty_column_retries=5,
                base_seed=seed, rank=0, scene_idx=0,
                pos_token_to_coord=scene._build_pos_token_to_coord_lut(gpt),
                debug_column_context=False)


def arm(name, context, adapter, optimized):
    from types import SimpleNamespace
    return compare.make_arm(name, adapter, context, SimpleNamespace(cache_capacity=16384), api,
                            optimized=optimized)


class Sampler:
    def __init__(self, model, context, active_arm):
        self.model, self.context, self.arm = model, context, active_arm
        self.records = []
        self.saved_prompt = model.sample_sequence_with_prompt
        self.saved_bos = model.sample_sequence

    def run(self, prompt, budget, seed):
        stats, suffix = compare.run_arm(self.arm, self.context, [EOS] + list(prompt),
                                        budget, seed, api)
        self.records.append(dict(prompt_tokens=len(prompt), budget=budget, seed=seed,
                                 generated_tokens=len(suffix), eos=bool(suffix and suffix[-1] == EOS),
                                 rounds=stats.get('rounds'), target_calls=stats.get('target_calls'),
                                 rescue_hits=stats.get('position_rescue_hits', 0)))
        return suffix

    def with_prompt(self, prompt_tokens, max_new_tokens, num_samples=1, temperature=1., top_k=None,
                    top_p=None, stop_on_eos=True, seed=None, logits_processor=None,
                    sampling_params_resolver=None):
        if (num_samples != 1 or temperature != .9 or top_k is not None or top_p != .9
                or not stop_on_eos or logits_processor is not None or seed is None):
            raise ValueError('Unsupported scene sampling parameters')
        if sampling_params_resolver is not None:
            for step in (0, 1):
                if sampling_params_resolver(step) != (.9, None, .9):
                    raise ValueError('Unsupported per-token sampling parameters')
        prompt = prompt_tokens.tolist() if torch.is_tensor(prompt_tokens) else list(prompt_tokens)
        suffix = self.run(prompt, int(max_new_tokens), int(seed))
        if suffix and suffix[-1] == EOS:
            suffix = suffix[:-1]
        return torch.tensor([prompt + suffix], dtype=torch.long, device='cuda')

    def bos(self, seq_len, batch_size=1, temperature=1., top_k=None, top_p=None,
            partials=None, stop_on_eos=True, seed=None, sampling_params_resolver=None):
        if (batch_size != 1 or temperature != .9 or top_k is not None or top_p != .9
                or partials is not None or not stop_on_eos or seed is None):
            raise ValueError('Unsupported bootstrap sampling parameters')
        if sampling_params_resolver is not None:
            for step in (0, 1):
                if sampling_params_resolver(step) != (.9, None, .9):
                    raise ValueError('Unsupported bootstrap token parameters')
        suffix = self.run([], int(seq_len), int(seed))
        return torch.tensor([suffix], dtype=torch.long, device='cuda')

    def install(self):
        self.model.sample_sequence_with_prompt = self.with_prompt
        self.model.sample_sequence = self.bos

    def remove(self):
        self.model.sample_sequence_with_prompt = self.saved_prompt
        self.model.sample_sequence = self.saved_bos


def validate(gpt, model, context, adapter, out):
    calls = []
    old = model.sample_sequence_with_prompt
    def record(prompt_tokens, *args, **kwargs):
        prompt = prompt_tokens.tolist() if torch.is_tensor(prompt_tokens) else list(prompt_tokens)
        calls.append(dict(prompt=prompt, budget=int(kwargs['max_new_tokens']), seed=int(kwargs['seed'])))
        return old(prompt_tokens, *args, **kwargs)
    model.sample_sequence_with_prompt = record
    try:
        tokens, lengths, grid = scene._generate_sparse_scene(gpt, **scene_args(gpt, 4, 0))
    finally:
        model.sample_sequence_with_prompt = old
    nonempty = [c for c in calls if c['prompt']]
    cases = [dict(prompt=[], budget=32, seed=1001)]
    for c in sorted(nonempty, key=lambda item: len(item['prompt'])):
        if len(cases) == 1 or len(c['prompt']) > len(cases[-1]['prompt']):
            cases.append(dict(prompt=c['prompt'], budget=min(32, c['budget']), seed=c['seed']))
        if len(cases) == 4:
            break
    if len(cases) < 2:
        raise RuntimeError('Scene probe did not produce a nonempty context')
    checks = []
    for case in cases:
        prompt, budget, seed = case['prompt'], case['budget'], case['seed']
        native = old(torch.tensor(prompt, dtype=torch.long), max_new_tokens=budget, num_samples=1,
                     temperature=.9, top_k=None, top_p=.9, stop_on_eos=True, seed=seed)[0].tolist()[len(prompt):]
        rows = {}
        for name in ('opt_target', 'pf_w1', 'pf_rescue'):
            selected = api.load_method('target', context) if name == 'opt_target' else adapter
            results = []
            for optimized in (False, True):
                active = arm(name, context, selected, optimized)
                try:
                    stats, suffix = compare.run_arm(active, context, [EOS] + prompt, budget, seed, api)
                    runtime = compare.arm_stats(active)
                    results.append(dict(suffix=suffix, counts=compare.count_values(stats), runtime=runtime))
                finally:
                    compare.close_arm(active)
            if results[0]['suffix'] != results[1]['suffix'] or results[0]['counts'] != results[1]['counts']:
                raise AssertionError(f'ordinary/graph mismatch: {name}, prefix {len(prompt)}')
            rows[name] = results[1]
        generated = rows['opt_target']['suffix']
        if generated and generated[-1] == EOS:
            generated = generated[:-1]
        if native != generated:
            raise AssertionError(f'original/optimized target output differs: prefix {len(prompt)}')
        if rows['pf_w1']['suffix'] != rows['pf_rescue']['suffix']:
            raise AssertionError(f'W1/rescue output differs: prefix {len(prompt)}')
        checks.append(dict(prompt_tokens=len(prompt), budget=budget, seed=seed,
                           native_target_matches=True, ordinary_graph_matches=True,
                           w1_rescue_matches=True, methods=rows))
        print(json.dumps(dict(event='validated', prefix=len(prompt), budget=budget)), flush=True)
    write_json(out / 'validation.json', dict(passed=True, size=4, captured_calls=len(calls),
                                              nonempty_columns=int((lengths > 0).sum()), cases=checks))


def execute(gpt, model, context, adapter, out, name, size, seed):
    destination = out / name
    destination.mkdir()
    active = None
    sampler = None
    torch.cuda.synchronize()
    started = time.perf_counter()
    if name != 'target':
        active = arm(name, context, adapter, True)
        sampler = Sampler(model, context, active)
        sampler.install()
    try:
        args = scene_args(gpt, size, seed)
        _, _, _, z = scene._validate_and_derive_geometry(gpt)
        tokens, lengths, grid = scene._generate_sparse_scene(gpt, **args)
        torch.cuda.synchronize()
        generated = time.perf_counter()
        torch.save(dict(tokens_sequence=tokens.cpu(), column_token_lengths=lengths.cpu()), destination / 'tokens.pt')
        decoded = scene._decode_sparse_rows_grid(gpt, grid, z=z)
        torch.cuda.synchronize()
        decoded_at = time.perf_counter()
        if not decoded['coords'].numel() or not all(bool(torch.isfinite(v).all()) for v in decoded.values() if torch.is_tensor(v)):
            raise ValueError('Scene Gaussian output is empty or nonfinite')
        scene._render_topdown_png(decoded, output_path=destination / 'topdown.png',
                                  quantile=65., resolution=1024, background_color='white')
        torch.save(scene._to_cpu_payload(decoded), destination / 'gaussians.pt')
        torch.cuda.synchronize()
        end = time.perf_counter()
        result = dict(method=name, seed=seed, scene_cols=[size, size], tokens=int(tokens.numel()),
                      nonempty_columns=int((lengths > 0).sum()), total_columns=size*size,
                      gaussians=int(decoded['coords'].shape[0]), sampling_s=generated-started,
                      decode_and_token_save_s=decoded_at-generated, render_and_save_s=end-decoded_at,
                      total_s=end-started, sampler_calls=len(sampler.records) if sampler else None,
                      rescue_hits=sum(r['rescue_hits'] for r in sampler.records) if sampler else None)
        if sampler:
            write_json(destination / 'sampler_calls.json', sampler.records)
            write_json(destination / 'runtime.json', compare.arm_stats(active))
        write_json(destination / 'result.json', result)
        print(json.dumps(dict(event='complete', **result)), flush=True)
    finally:
        if sampler:
            sampler.remove()
        if active:
            compare.close_arm(active)
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('validate', 'run'), required=True)
    parser.add_argument('--size', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--methods', default='target,pf_w1,pf_rescue')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    write_json(args.out / 'progress.json', dict(status='loading', mode=args.mode))
    try:
        with torch.inference_mode(), threadpool_limits(limits=1):
            gpt, model, context = load_models()
            adapter = api.load_method('pf_mlp', context)
            if args.mode == 'validate':
                validate(gpt, model, context, adapter, args.out)
            else:
                methods = args.methods.split(',')
                write_json(args.out / 'plan.json', dict(size=args.size, seed=args.seed, methods=methods,
                           temperature=.9, top_p=.9, bootstrap=True, resample_empty_columns=True,
                           x_offset=5, y_offset=5, y_range=5, retries=5, graph_capacity=16384,
                           timing='generation, token save, VQ decode, topdown render and Gaussian save; model load excluded',
                           gpu=torch.cuda.get_device_name(), torch=torch.__version__))
                for name in methods:
                    write_json(args.out / 'progress.json', dict(status='running', method=name))
                    execute(gpt, model, context, adapter, args.out, name, args.size, args.seed)
        write_json(args.out / 'progress.json', dict(status='complete', mode=args.mode))
    except BaseException as exc:
        write_json(args.out / 'progress.json', dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
