#!/usr/bin/env python3
"""Measure complete single-chunk decoding, Gaussian export, and common views."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from draft_a.corpus_v2 import EOS, load_target
from draft_a.feature_target_v2 import HEAD, RANKER, save
from draft_a.learning_eval import _fp32_runtime
from draft_a import pf_ranker_eval as ordinary
from draft_a.pf_feature_unified import MLPAdapter, decode as pf_decode

MODES = ('target_only', 'p_only', 'step12000')


def ordinary_output(core, ranker, ranks, prompt, budget, k, seed):
    import ast
    import inspect
    if not hasattr(ordinary_output, 'implementation'):
        tree = ast.parse(inspect.getsource(ordinary.decode))
        function = tree.body[0]
        if not isinstance(function.body[-1], ast.Return):
            raise ValueError('Ordinary decoder must end with its result return')
        final = function.body[-1]
        final.value = ast.Tuple(elts=[final.value, ast.Name(id='output', ctx=ast.Load())], ctx=ast.Load())
        function.name = 'decode_with_output'
        ast.fix_missing_locations(tree)
        namespace = dict(ordinary.__dict__)
        exec(compile(tree, inspect.getsourcefile(ordinary.decode), 'exec'), namespace)
        ordinary_output.implementation = namespace['decode_with_output']
    stats, output = ordinary_output.implementation(core, None, ranker, ranks, prompt, budget, k, seed)
    if len(output) != stats['generated_tokens']:
        raise ValueError('Committed ordinary samples differ from output count')
    ordinary.validate_sequence(prompt, output, ranks)
    return stats, output


def camera_views(side, count=6, size=384):
    center = np.full(3, side / 2, dtype=np.float64)
    matrices = []
    for angle in np.linspace(0, 2 * math.pi, count, endpoint=False):
        eye = center + side * np.array([1.65 * math.cos(angle), 1.65 * math.sin(angle), 1.05])
        forward = (center - eye) / np.linalg.norm(center - eye)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])
        view = np.eye(4)
        view[:3, :3] = rotation
        view[:3, 3] = -rotation @ eye
        matrices.append(view)
    return dict(side_m=float(side), width=size, height=size,
                viewmats=np.asarray(matrices).tolist(),
                K=[[size * 1.1, 0, size / 2], [0, size * 1.1, size / 2], [0, 0, 1]])


def gaussian_tensors(vqvae, body):
    from utils.pos_tokens import pos_tokens_to_centered_coords
    positions = torch.tensor(body[::2], dtype=torch.long, device='cuda')[:, None]
    features = torch.tensor(body[1::2], dtype=torch.long, device='cuda')[:, None] - 8000
    coords = pos_tokens_to_centered_coords(positions, 1, 20).to(torch.int32)
    points = vqvae.decode(coords, features)
    xyz = points['coords'].float()
    raw_scales = points['scales'].float().exp()
    scales = raw_scales.clamp(1e-4, 0.05)
    opacity = points['opacities'].float().reshape(-1).sigmoid()
    quats = torch.nn.functional.normalize(points['quats'].float(), dim=-1, eps=1e-8)
    colors = (points['sh0'].float() * 0.28209479177387814 + 0.5).clamp(0, 1)
    tensors = dict(means=xyz, scales=scales, opacities=opacity, quats=quats, colors=colors)
    if not len(xyz) or not all(bool(torch.isfinite(v).all()) for v in tensors.values()):
        raise ValueError('Empty or nonfinite Gaussian output')
    quality = dict(gaussians=len(xyz), scale_clipped_fraction=float((raw_scales > .05).float().mean()),
                   xyz_min=xyz.min(0).values.tolist(), xyz_max=xyz.max(0).values.tolist(),
                   mean_opacity=float(opacity.mean()))
    return tensors, quality


def render_views(tensors, cameras):
    from gsplat import rasterization
    rgb, alpha, _ = rasterization(**tensors,
        viewmats=torch.tensor(cameras['viewmats'], dtype=torch.float32, device='cuda'),
        Ks=torch.tensor(cameras['K'], dtype=torch.float32, device='cuda')[None].repeat(6, 1, 1),
        width=cameras['width'], height=cameras['height'], packed=True,
        backgrounds=torch.zeros((6, 3), dtype=torch.float32, device='cuda'), render_mode='RGB')
    if not bool(torch.isfinite(rgb).all() and torch.isfinite(alpha).all()):
        raise ValueError('Nonfinite render output')
    return rgb.clamp(0, 1), alpha


def save_views(rgb, destination):
    for i, array in enumerate(rgb.mul(255).byte().cpu().numpy()):
        Image.fromarray(array).save(destination / f'view_{i:02d}.png')


def sheet(rows, root):
    width, height = 384, 424
    canvas = Image.new('RGB', (len(MODES) * width, len(rows) // len(MODES) * height), '#eeeeee')
    draw = ImageDraw.Draw(canvas)
    groups = list(dict.fromkeys(r['sample'] for r in rows))
    for row in rows:
        x = MODES.index(row['variant']) * width
        y = groups.index(row['sample']) * height
        if row.get('status') == 'complete':
            canvas.paste(Image.open(Path(row['output_dir']) / 'view_00.png'), (x, y + 40))
        draw.text((x + 5, y + 3), f"{row['variant']}  {row.get('total_seconds', 0):.2f}s", fill='black')
        draw.text((x + 5, y + 20), row['sample'][:45], fill='black')
    canvas.save(root / 'all_samples.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--scene-ids', nargs='*')
    parser.add_argument('--seeds', nargs='+', type=int, default=[51701, 51702, 51703])
    parser.add_argument('--kind', choices=['completion', 'bos'], default='completion')
    parser.add_argument('--body-limit', type=int, default=16000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.body_limit != 16000:
        raise ValueError('The full-output plan uses the common 16000 body-token limit')
    if args.root.exists() and not args.resume:
        raise FileExistsError(args.root)
    args.root.mkdir(parents=True, exist_ok=True)
    cached = torch.load(args.records, map_location='cpu', weights_only=True)
    first = {}
    for row in cached:
        first.setdefault(row['scene'], row)
    scene_ids = args.scene_ids or list(first)
    if args.kind == 'completion':
        records = [first[s] for s in scene_ids]
        if any(len(r['tokens']) < 256 for r in records):
            raise ValueError('Every requested source must support the selected 256-token prefix')
    else:
        records = [dict(scene='bos', crop_seed=0, tokens=torch.empty(0, dtype=torch.long))]
    plan = dict(kind=args.kind, scenes=[r['scene'] for r in records], seeds=args.seeds,
                modes=list(MODES), prefix_tokens=256 if args.kind == 'completion' else 0,
                body_limit=args.body_limit, temperature=.9, top_p=.9, precision='FP32',
                target='checkpoints/gpt_vfront.ckpt', vqvae='checkpoints/vqvae_vfront.ckpt',
                feature='draft_a/runs/feature_target_continue_20261014/selected.pt',
                scope='one 20x20x20 latent chunk in local coordinates; no multi-chunk extension',
                timing='synchronized outer wall time from ready tokens to all six PNGs and Gaussian NPZ; model loading and reference rendering separate',
                views='six fixed orbit views of the common 4-metre local chunk; same cameras for all samples',
                quality_scope='finite outputs and descriptive full-reference chunk metrics; sampled suffix has multiple valid outcomes')
    if args.resume and (args.root / 'plan.json').exists():
        if json.loads((args.root / 'plan.json').read_text()) != plan:
            raise ValueError('Existing full-output settings differ')
    save(args.root / 'plan.json', plan)
    torch.set_num_threads(6)
    rows = []
    with torch.inference_mode(), _fp32_runtime(torch):
        from joblib import load
        from threadpoolctl import threadpool_limits
        from bench_fast_hgb import FastHistRanker
        from bench_gaussiangpt_pf_block2_cow import load_feature_model
        from model.gaussian_vqvae import GaussianVQVAE
        start = time.perf_counter()
        target = load_target().float().eval()
        feature = load_feature_model(Path(plan['feature']), HEAD, target.transformer,
                                     torch.device('cuda'), 8000, 12096)[0]
        ranker = FastHistRanker(load(RANKER))
        ranks = target.transformer.sparse_position_rank_map.cpu().tolist()
        vqvae = GaussianVQVAE.load_from_checkpoint(plan['vqvae'], map_location='cpu').cuda().float().eval()
        vqvae.requires_grad_(False)
        torch.cuda.synchronize()
        save(args.root / 'loading.json', dict(seconds=time.perf_counter() - start))
        side = 20 * vqvae.autoencoder.stride * vqvae.autoencoder.default_grid_size
        cameras = camera_views(side)
        save(args.root / 'cameras.json', cameras)
        with threadpool_limits(limits=1, user_api='openmp'):
            for ri, record in enumerate(records):
                source = record['tokens'].tolist()
                prompt = [EOS] + source[:plan['prefix_tokens']]
                budget = args.body_limit - plan['prefix_tokens']
                reference = None
                reference_path = args.root / 'references' / record['scene']
                if source:
                    reference_path.mkdir(parents=True, exist_ok=True)
                    before = time.perf_counter()
                    tensors, quality = gaussian_tensors(vqvae, source)
                    reference, _ = render_views(tensors, cameras)
                    reference = reference.cpu()
                    save_views(reference, reference_path)
                    save(reference_path / 'metrics.json', dict(**quality, seconds=time.perf_counter()-before))
                    del tensors
                for mode in MODES:
                    warm_prompt = prompt if len(prompt) > 1 else [EOS]
                    if mode == 'step12000':
                        pf_decode(target.transformer, MLPAdapter(feature), ranker, ranks, warm_prompt, 32, 51699)
                    else:
                        ordinary_output(target.transformer, ranker, ranks, warm_prompt, 32, int(mode=='p_only'), 51699)
                for si, seed in enumerate(args.seeds):
                    sample = f"{record['scene']}_seed_{seed}"
                    modes = list(MODES)
                    shift = (ri + si) % len(modes)
                    for mode in modes[shift:] + modes[:shift]:
                        out = args.root / 'samples' / sample / mode
                        metrics_path = out / 'metrics.json'
                        if args.resume and metrics_path.exists():
                            previous = json.loads(metrics_path.read_text())
                            rows.append(previous)
                            continue
                        out.mkdir(parents=True, exist_ok=True)
                        result = dict(sample=sample, scene=record['scene'], crop_seed=record['crop_seed'],
                                      seed=seed, variant=mode, output_dir=str(out), status='running')
                        save(args.root / 'progress.json', dict(status='running', completed=len(rows),
                             expected=len(records)*len(args.seeds)*len(MODES), current=result))
                        try:
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            if mode == 'step12000':
                                stats, suffix = pf_decode(target.transformer, MLPAdapter(feature), ranker, ranks,
                                                          prompt, budget, seed)
                            else:
                                stats, suffix = ordinary_output(target.transformer, ranker, ranks,
                                                                prompt, budget, int(mode=='p_only'), seed)
                            torch.cuda.synchronize()
                            ar_end = time.perf_counter()
                            ordinary.validate_sequence(prompt, suffix, ranks)
                            body = prompt[1:] + suffix
                            eos = bool(body and body[-1] == EOS)
                            body = body[:-1] if eos else body
                            if len(body) % 2:
                                raise ValueError('Complete output contains an unfinished P/F pair')
                            torch.save(dict(tokens=torch.tensor(body), suffix=torch.tensor(suffix),
                                            has_eos=eos, prefix_tokens=plan['prefix_tokens']), out / 'tokens.pt')
                            if not body:
                                raise ValueError('Natural EOS produced an empty chunk')
                            tensors, quality = gaussian_tensors(vqvae, body)
                            torch.cuda.synchronize()
                            decode_end = time.perf_counter()
                            np.savez_compressed(out / 'gaussians.npz',
                                                **{k:v.cpu().numpy() for k,v in tensors.items()})
                            export_end = time.perf_counter()
                            rgb, alpha = render_views(tensors, cameras)
                            torch.cuda.synchronize()
                            render_end = time.perf_counter()
                            save_views(rgb, out)
                            total_end = time.perf_counter()
                            result.update(status='complete', eos=eos, capped=not eos, body_tokens=len(body),
                                generated_tokens=len(suffix), ar_seconds=ar_end-started,
                                decode_seconds=decode_end-ar_end, export_seconds=export_end-decode_end,
                                render_seconds=render_end-export_end, png_seconds=total_end-render_end,
                                total_seconds=total_end-started, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                                alpha_coverage=float((alpha>.01).float().mean()), quality=quality,
                                acceptance={k:stats[k] for k in stats if 'acceptance' in k or k.endswith('_calls')})
                            if reference is not None:
                                mse = float((rgb.cpu()-reference).square().mean())
                                result['full_reference_psnr_db'] = -10*math.log10(max(mse,1e-12))
                                from skimage.metrics import structural_similarity
                                result['full_reference_ssim'] = float(np.mean([
                                    structural_similarity(a,b,channel_axis=-1,data_range=1.)
                                    for a,b in zip(rgb.cpu().numpy(),reference.numpy())]))
                            del tensors, rgb, alpha
                        except Exception:
                            import traceback
                            result.update(status='failed', error=traceback.format_exc())
                            torch.cuda.empty_cache()
                        save(metrics_path, result)
                        rows.append(result)
                        with (args.root / 'e2e_records.jsonl').open('a') as handle:
                            handle.write(json.dumps(result)+'\n')
                        save(args.root / 'progress.json', dict(status='running',completed=len(rows),
                             expected=len(records)*len(args.seeds)*len(MODES)))
                        print(json.dumps({k:result.get(k) for k in ('sample','variant','status','eos','body_tokens','ar_seconds','total_seconds','error')}),flush=True)
        sheet(rows, args.root)
    summary = dict(status='complete', expected=len(records)*len(args.seeds)*len(MODES),
                   completed=len(rows), success=sum(r['status']=='complete' for r in rows),
                   failed=sum(r['status']!='complete' for r in rows), rows=rows)
    save(args.root / 'quality_summary.json', summary)
    save(args.root / 'progress.json', {k:v for k,v in summary.items() if k!='rows'})


if __name__ == '__main__':
    main()
