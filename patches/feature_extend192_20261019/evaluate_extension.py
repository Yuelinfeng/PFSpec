"""Measure exact continuation weights using existing PF decoding and development crops."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECT = Path('/root/workspace/GaussianGPT')
ROOT = Path('/root/autodl-tmp/gaussiangpt_feature_extend192_20261019')
CROPS = Path('/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_c/dev_crops.pt')
ORIGINAL = PROJECT / 'draft_a/runs/feature_target_continue_20261014/selected.pt'
PRIOR = Path('/root/autodl-tmp/gaussiangpt_feature_extend_20261019')
CURVE_STEPS = list(range(48000, 192001, 4800))
EXACT_STEPS = [12000, 24000, 48000, 96000, 192000]
sys.path.insert(0, str(PROJECT))


def aggregate(rows):
    keys = ['generated_tokens', 'seconds', 'feature_accepted', 'feature_verified',
            'feature_proposed', 'position_accepted', 'position_verified', 'target_calls', 'rounds']
    sums = {k: sum(r.get(k, 0) for r in rows) for k in keys}
    sums['runs'] = len(rows)
    sums['tokens_per_second'] = sums['generated_tokens'] / sums['seconds']
    sums['feature_acceptance'] = (sums['feature_accepted'] / sums['feature_verified']
                                  if sums['feature_verified'] else None)
    sums['position_acceptance'] = (sums['position_accepted'] / sums['position_verified']
                                   if sums['position_verified'] else None)
    return sums


def main():
    import numpy as np
    import torch
    from joblib import load
    from threadpoolctl import threadpool_limits
    from bench_fast_hgb import FastHistRanker
    from bench_gaussiangpt_pf_block2_cow import load_feature_model
    from draft_a.feature_target_v2 import HEAD, RANKER, save
    from draft_a.corpus_v2 import EOS, load_target, validate
    from draft_a.learning_eval import _fp32_runtime
    from draft_a.pf_feature_unified import decode, MLPAdapter

    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.chdir(PROJECT)
    out = ROOT / ('evaluation_smoke' if args.smoke else 'evaluation')
    if out.exists():
        raise FileExistsError(out)
    if not args.smoke:
        phase = json.loads((ROOT / 'pipeline.json').read_text())
        if phase['status'] != 'trained':
            raise ValueError('Finish training before uncontended timing')
    out.mkdir()
    torch.set_num_threads(6)
    raw = torch.load(CROPS, map_location='cpu', weights_only=True)
    by_scene = {}
    for r in raw:
        validate(r)
        scene = r.get('scene') or Path(r['path']).parents[1].name
        if len(r['tokens']) >= 256 and scene not in by_scene:
            by_scene[scene] = dict(r, scene=scene)
    records = [by_scene[s] for s in sorted(by_scene)]
    if len(records) != 10:
        raise ValueError(f'Expected ten existing development sources, got {len(records)}')
    plan = dict(crops=str(CROPS), source_count=len(records),
                selected_crops=[dict(scene=r['scene'], crop_seed=r['crop_seed'], path=r['path']) for r in records],
                precision='FP32', temperature=.9, top_p=.9, decoder='unchanged static PF-K2',
                curve=dict(steps=CURVE_STEPS, sources=3, prefixes=[64, 256],
                           seeds=[34701], budget=128, purpose='exploratory online trajectory'),
                final=dict(steps=EXACT_STEPS, sources=10, prefixes=[64, 256],
                           seeds=[34701, 34702, 34703], budget=256),
                timing='synchronized outer wall, includes cache/prefill/proposal/verification/recovery/sampling; excludes loading and warmup',
                uncertainty='previously used development sources; one training seed; no new final-test evaluation',
                exact_requested_weights=True)
    save(out / 'plan.json', plan)
    paths = {step: ROOT / 'training' / f'step_{step:04d}.pt' for step in CURVE_STEPS}
    paths.update({12000: ORIGINAL, 24000: PRIOR / '24000.pt', 48000: PRIOR / '48000.pt'})
    if args.smoke:
        paths = {48000: PRIOR / '48000.pt'}
    all_rows = []
    with torch.inference_mode(), _fp32_runtime(torch), threadpool_limits(limits=1, user_api='openmp'):
        target = load_target().float().eval().requires_grad_(False)
        ranker = FastHistRanker(load(RANKER))
        ranks = target.transformer.sparse_position_rank_map.cpu().tolist()
        adapters = {}

        def adapter(step):
            if step not in adapters:
                model, _, _ = load_feature_model(paths[step], HEAD, target.transformer,
                                                 torch.device('cuda'), 8000, 12096)
                adapters[step] = MLPAdapter(model)
                prompt = torch.cat([torch.tensor([EOS]), records[0]['tokens'][:64]]).tolist()
                decode(target.transformer, adapters[step], ranker, ranks, prompt, 16, 34700)
            return adapters[step]

        def run(stage, step, record, prefix, seed_base, budget):
            a = adapter(step)
            prompt = torch.cat([torch.tensor([EOS]), record['tokens'][:prefix]]).tolist()
            seed = seed_base + record['crop_seed'] * 1009 + prefix * 17
            torch.cuda.synchronize()
            start = time.perf_counter()
            row, tokens = decode(target.transformer, a, ranker, ranks, prompt, budget, seed)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            row.update(stage=stage, step=step, scene=record['scene'], crop_seed=record['crop_seed'],
                       prefix=prefix, seed_base=seed_base, decoder_seconds=row['seconds'],
                       seconds=elapsed, tokens_per_second=row['generated_tokens'] / elapsed,
                       output_tokens=tokens)
            all_rows.append(row)
            with (out / 'records.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            save(out / 'progress.json', dict(status='running', stage=stage, completed=len(all_rows),
                                              step=step, scene=record['scene']))
            print(json.dumps(dict(stage=stage, step=step, scene=record['scene'], prefix=prefix,
                                  F=f"{row['feature_accepted']}/{row['feature_verified']}",
                                  tps=row['tokens_per_second'])), flush=True)

        # Timing instrumentation must preserve tokens and counts on every exact requested model.
        verification = []
        for step in ([48000] if args.smoke else EXACT_STEPS):
            a = adapter(step)
            prompt = torch.cat([torch.tensor([EOS]), records[0]['tokens'][:64]]).tolist()
            ordinary, tokens = decode(target.transformer, a, ranker, ranks, prompt, 32, 34700)
            profiled, tokens2 = decode(target.transformer, a, ranker, ranks, prompt, 32, 34700, profile=True)
            if tokens != tokens2 or any(ordinary[k] != profiled[k] for k in ['accepted', 'verified', 'proposed', 'rounds']):
                raise ValueError('Timing instrumentation changed output')
            verification.append(dict(step=step, timing_replay_equal=True))
        save(out / 'verification.json', verification)
        if args.smoke:
            run('smoke', 48000, records[0], 64, 34701, 32)
            save(out / 'progress.json', dict(status='complete', runs=1))
            return

        # Rotate the checkpoint order across contexts; no training runs concurrently.
        steps = CURVE_STEPS
        for i, record in enumerate(records[:3]):
            for pi, prefix in enumerate([64, 256]):
                shift = (i * 2 + pi) * 5 % len(steps)
                order = steps[shift:] + steps[:shift]
                for step in order:
                    run('curve', step, record, prefix, 34701, 128)
        # Release unneeded curve models before the larger exact-step comparison.
        for step in list(adapters):
            if step not in EXACT_STEPS:
                del adapters[step]
        gc.collect()
        torch.cuda.empty_cache()
        for i, record in enumerate(records):
            for pi, prefix in enumerate([64, 256]):
                for repeat, seed in enumerate([34701, 34702, 34703]):
                    order = list(EXACT_STEPS)
                    shift = (i + pi + repeat) % len(order)
                    for step in order[shift:] + order[:shift]:
                        run('final', step, record, prefix, seed, 256)

    curve = {str(s): aggregate([r for r in all_rows if r['stage'] == 'curve' and r['step'] == s])
             for s in CURVE_STEPS}
    final_rows = [r for r in all_rows if r['stage'] == 'final']
    if sum(r['stage'] == 'curve' for r in all_rows) != 186 or len(final_rows) != 300:
        raise ValueError('The planned evaluation matrix is incomplete')
    final = {str(s): aggregate([r for r in final_rows if r['step'] == s]) for s in EXACT_STEPS}
    scenes = sorted(by_scene)
    groups = {s: {step: [r for r in final_rows if r['scene'] == s and r['step'] == step]
                  for step in EXACT_STEPS} for s in scenes}
    rng = np.random.default_rng(192000)

    def comparisons(reference, steps):
        result = {}
        for step in steps:
            per_source = {s: aggregate(groups[s][step])['tokens_per_second'] /
                             aggregate(groups[s][reference])['tokens_per_second'] for s in scenes}
            speed, delta = [], []
            for _ in range(10000):
                chosen = rng.choice(scenes, len(scenes), replace=True)
                speed.append(math.exp(sum(math.log(per_source[s]) for s in chosen) / len(chosen)))
                newer = aggregate([r for s in chosen for r in groups[s][step]])
                older = aggregate([r for s in chosen for r in groups[s][reference]])
                delta.append(newer['feature_acceptance'] - older['feature_acceptance'])
            result[str(step)] = dict(
                speed_source_geomean=math.exp(sum(math.log(v) for v in per_source.values()) / len(scenes)),
                speed_ci95=np.quantile(speed, [.025, .975]).tolist(),
                feature_delta=final[str(step)]['feature_acceptance'] - final[str(reference)]['feature_acceptance'],
                feature_delta_ci95=np.quantile(delta, [.025, .975]).tolist(), per_source_speed=per_source)
        return result

    save(out / 'summary.json', dict(curve=curve, final=final,
                                    relative_to_12000=comparisons(12000, [24000, 48000, 96000, 192000]),
                                    relative_to_48000=comparisons(48000, [96000, 192000]),
                                    expected_curve_runs=186, expected_final_runs=300,
                                    observed_curve_runs=sum(r['stage'] == 'curve' for r in all_rows),
                                    observed_final_runs=len(final_rows)))
    save(out / 'progress.json', dict(status='complete', runs=len(all_rows)))
    import subprocess
    subprocess.run([sys.executable, str(ROOT / 'summarize_extension.py')], check=True)
    save(ROOT / 'pipeline.json', dict(status='complete', phase='analysis_complete', exact_steps=[96000, 192000]))


if __name__ == '__main__':
    main()
