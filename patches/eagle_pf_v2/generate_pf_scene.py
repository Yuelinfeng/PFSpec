"""Generate a complete single chunk and export Gaussian arrays and six views."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval_pf_draft import load_draft
from speculative.pf_engine import PositionRanker, decode, synchronize
from train_pf_draft import load_core, save_json


@torch.inference_mode()
def export_scene(vqvae, body, destination):
    from utils.pos_tokens import pos_tokens_to_centered_coords
    from gsplat import rasterization
    positions = torch.tensor(body[::2], device="cuda")[:, None]
    features = torch.tensor(body[1::2], device="cuda")[:, None]-8000
    coords = pos_tokens_to_centered_coords(positions, 1, 20).to(torch.int32)
    points = vqvae.decode(coords, features)
    arrays = dict(means=points["coords"].float(),
                  scales=points["scales"].float().exp().clamp(1e-4, 0.05),
                  opacities=points["opacities"].float().reshape(-1).sigmoid(),
                  quats=torch.nn.functional.normalize(points["quats"].float(), dim=-1, eps=1e-8),
                  colors=(points["sh0"].float()*0.28209479177387814+0.5).clamp(0, 1))
    if not len(arrays["means"]) or not all(bool(torch.isfinite(x).all()) for x in arrays.values()):
        raise ArithmeticError("empty or nonfinite Gaussian scene")
    np.savez_compressed(destination/"gaussians.npz", **{k: v.cpu().numpy() for k, v in arrays.items()})
    side = float(20*vqvae.autoencoder.stride*vqvae.autoencoder.default_grid_size)
    center = np.full(3, side/2)
    views = []
    for angle in np.linspace(0, 2*math.pi, 6, endpoint=False):
        eye = center+side*np.array([1.65*math.cos(angle), 1.65*math.sin(angle), 1.05])
        forward = center-eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1.])
        right /= np.linalg.norm(right)
        rotation = np.stack((right, np.cross(forward, right), forward))
        view = np.eye(4)
        view[:3, :3], view[:3, 3] = rotation, -rotation@eye
        views.append(view)
    size = 384
    camera = torch.tensor([[size*1.1, 0, size/2], [0, size*1.1, size/2], [0, 0, 1.]], device="cuda")
    rgb, alpha, _ = rasterization(**arrays,
        viewmats=torch.tensor(np.asarray(views), dtype=torch.float32, device="cuda"),
        Ks=camera[None].repeat(6, 1, 1), width=size, height=size, packed=True,
        backgrounds=torch.zeros(6, 3, device="cuda"), render_mode="RGB")
    if not torch.isfinite(rgb).all():
        raise ArithmeticError("nonfinite scene render")
    for i, image in enumerate(rgb.clamp(0, 1).mul(255).byte().cpu().numpy()):
        Image.fromarray(image).save(destination/f"view_{i:02d}.png")
    return dict(gaussians=len(arrays["means"]), side_m=side, views=6,
                display_scale_clip=[1e-4, 0.05], finite=True,
                note="same display scale clipping as existing single-chunk evaluator")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crops", type=Path, default=Path("/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_c/dev_crops.pt"))
    parser.add_argument("--prefix", type=int, default=256)
    parser.add_argument("--modes", nargs="+", choices=["target", "joint", "feature", "feature_g5", "feature_v2", "old_pf"], default=["target", "feature", "joint"])
    parser.add_argument("--feature-checkpoint", type=Path)
    parser.add_argument("--reference-feature", type=Path)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--seed", type=int, default=51701)
    parser.add_argument("--body-limit", type=int, default=16000)
    args = parser.parse_args()
    if args.prefix % 2 or args.prefix < 0 or args.body_limit % 2 or args.body_limit <= args.prefix:
        parser.error("prefix and body limit must leave a complete P/F continuation")
    wrapper, core = load_core()
    records = torch.load(args.crops, map_location="cpu", weights_only=False)
    unique, seen = [], set()
    for row in records:
        if row["scene"] not in seen and len(row["tokens"]) >= args.prefix:
            seen.add(row["scene"])
            unique.append(row)
    source = unique[args.scene_index]
    prompt = [core.sparse_eos_token_id]+source["tokens"][:args.prefix].tolist()
    from joblib import load
    from bench_fast_hgb import FastHistRanker
    from draft_a.feature_target_v2 import RANKER
    from model.gaussian_vqvae import GaussianVQVAE
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    ranker = FastHistRanker(load(RANKER))
    ranks = core.sparse_position_rank_map.cpu().tolist()
    vqvae = GaussianVQVAE.load_from_checkpoint("checkpoints/vqvae_vfront.ckpt", map_location="cpu").cuda().float().eval()
    vqvae.requires_grad_(False)
    drafts = {}
    for mode in args.modes:
        if mode in ("target", "old_pf"):
            continue
        path = (args.reference_feature if mode == "feature_v2" else args.feature_checkpoint if mode in ("feature", "feature_g5") and args.feature_checkpoint else args.training/("feature" if mode == "feature_g5" else mode)/"selected.pt")
        if path is None:
            parser.error("feature_v2 needs --reference-feature")
        drafts[mode] = load_draft(core, path)
    old_adapter = None
    if "old_pf" in args.modes:
        from bench_gaussiangpt_pf_block2_cow import load_feature_model
        from draft_a.feature_target_v2 import HEAD
        from draft_a.pf_feature_unified import MLPAdapter
        old, _, _ = load_feature_model(Path("draft_a/runs/feature_target_continue_20261014/selected.pt"), HEAD, core, "cuda", 8000, 12096)
        old_adapter = MLPAdapter(old)
    rows = []
    for mode in args.modes:
        run_gamma = 3 if mode == "old_pf" else 5 if mode == "feature_g5" else args.gamma
        destination = args.output/mode
        destination.mkdir(parents=True, exist_ok=True)
        synchronize(torch.device("cuda"))
        started = time.perf_counter()
        if mode == "old_pf":
            from draft_a.pf_feature_unified import decode as old_decode
            stats, suffix = old_decode(core, old_adapter, ranker, ranks, prompt, args.body_limit-args.prefix, args.seed)
        else:
            stats, suffix = decode(core, drafts.get(mode), prompt, args.body_limit-args.prefix,
                                   mode="feature" if mode.startswith("feature") else mode,
                                   gamma=run_gamma, ranker=PositionRanker(ranker, ranks), seed=args.seed)
        ar_finished = time.perf_counter()
        body = prompt[1:]+suffix
        eos = bool(body and body[-1] == core.sparse_eos_token_id)
        if eos:
            body = body[:-1]
        if len(body) % 2 or not body:
            raise ValueError("generated chunk has no complete P/F representation")
        torch.save(dict(tokens=torch.tensor(body), suffix=torch.tensor(suffix), has_eos=eos,
                        prefix=args.prefix), destination/"tokens.pt")
        quality = export_scene(vqvae, body, destination)
        synchronize(torch.device("cuda"))
        result = dict(mode=mode, gamma=run_gamma, scene=source["scene"], prefix=args.prefix,
                      seed=args.seed, body_tokens=len(body), eos=eos, ar_seconds=ar_finished-started,
                      total_seconds=time.perf_counter()-started, decoder=stats, output=str(destination),
                      quality=quality, scope="ready tokens through Gaussian NPZ and six saved PNGs; loading excluded")
        rows.append(result)
        save_json(args.output/"results.json", rows)
        print(result, flush=True)


if __name__ == "__main__":
    main()
