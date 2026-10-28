#!/usr/bin/env python3
"""Decode VFront tokens and render the recovered Gaussians with gsplat."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization

from data.vfront import load_pth_sparse_gaussian
from model.gaussian_vqvae import GaussianVQVAE

VOXEL_SIZE = 0.025
SH_C0 = 0.28209479177387814


def save_rgb(image: torch.Tensor, path: Path) -> None:
    array = image[0].detach().clamp(0, 1).mul(255).byte().cpu().numpy()
    Image.fromarray(array).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()

    scene_root = args.scene_root.resolve()
    device = torch.device("cuda")
    token_path = scene_root / "gaussian_init" / "vfront_pilot_tokens.pt"
    token_data = torch.load(token_path, map_location="cpu", weights_only=False)
    coords = token_data["coords"].to(device=device, dtype=torch.int32)
    feature_ids = token_data["feature_ids"].to(device=device, dtype=torch.long)

    model = GaussianVQVAE.load_from_checkpoint(
        str(args.vqvae_checkpoint), map_location="cpu"
    ).to(device).eval()
    model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        points = model.decode(coords, feature_ids)

    xyz = points["coords"].detach().float()
    scale_unclamped = points["scales"].detach().float().exp()
    scales = scale_unclamped.clamp(min=1e-4, max=2 * VOXEL_SIZE)
    opacity_logits = points["opacities"].detach().float().reshape(-1)
    opacities = torch.sigmoid(opacity_logits)
    quats = F.normalize(points["quats"].detach().float(), dim=-1, eps=1e-8)
    sh0 = points["sh0"].detach().float()
    colors = (sh0 * SH_C0 + 0.5).clamp(0, 1)

    tensors = {"xyz": xyz, "scales": scales, "opacities": opacities, "quats": quats, "sh0": sh0, "colors": colors}
    for name, value in tensors.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite decoded field: {name}")
    count = int(xyz.shape[0])
    if count == 0:
        raise RuntimeError("VQ-VAE decoder returned no Gaussians")

    out = scene_root / "gaussian_init" / "decoded_render"
    out.mkdir(parents=True, exist_ok=True)
    xyz_np = xyz.cpu().numpy().astype(np.float32)
    scale_np = scales.cpu().numpy().astype(np.float32)
    opacity_np = opacity_logits.cpu().numpy().astype(np.float32)[:, None]
    quat_np = quats.cpu().numpy().astype(np.float32)
    sh0_np = sh0.cpu().numpy().astype(np.float32)
    np.savez_compressed(
        out / "decoded_gaussians.npz",
        xyz=xyz_np,
        scales=scale_np,
        opacity_logits=opacity_np[:, 0],
        quats=quat_np,
        sh0=sh0_np,
    )

    # Export the decoder output in the repository's sparse-Gaussian checkpoint format.
    anchors = np.floor(xyz_np.astype(np.float64) / VOXEL_SIZE) * VOXEL_SIZE
    offset_ratio = (xyz_np.astype(np.float64) - anchors) / (1.5 * VOXEL_SIZE)
    raw_offset = np.arctanh(np.clip(offset_ratio, -0.999, 0.999)).astype(np.float32)
    scale_fraction = np.clip(scale_np / (2 * VOXEL_SIZE), 1e-4, 1 - 1e-4)
    raw_scale = np.log(scale_fraction / (1 - scale_fraction)).astype(np.float32)
    checkpoint = {
        "anchor": anchors.astype(np.float32),
        "offset": raw_offset,
        "scale": raw_scale,
        "opacity": opacity_np,
        "rotation": quat_np,
        "f_dc": sh0_np,
        "f_rest": np.zeros((count, 45), dtype=np.float32),
    }
    decoded_ckpt = out / "decoded_ckpt.pth"
    torch.save(checkpoint, decoded_ckpt)
    loaded = load_pth_sparse_gaussian(str(decoded_ckpt), sh_degree=0, voxel_size=VOXEL_SIZE)
    recovered = loaded["anchor"] + loaded["offset"]
    max_position_error = float(np.max(np.abs(recovered - xyz_np)))
    if max_position_error > 1e-5:
        raise RuntimeError(f"Decoded checkpoint position error {max_position_error}")

    camera_cfg = json.loads((scene_root / "depth_pilot" / "cameras.json").read_text())
    width, height = int(camera_cfg["width"]), int(camera_cfg["height"])
    K_np = np.array(
        [[camera_cfg["fx"], 0, camera_cfg["cx"]],
         [0, camera_cfg["fy"], camera_cfg["cy"]],
         [0, 0, 1]],
        dtype=np.float32,
    )
    Ks = torch.tensor(K_np, dtype=torch.float32, device=device).unsqueeze(0)
    backgrounds = torch.zeros((1, 3), dtype=torch.float32, device=device)
    per_view = []
    for frame in camera_cfg["frames"]:
        name = frame["name"]
        c2w = np.asarray(frame["camera_to_world"], dtype=np.float32)
        c2w[:3, 1:3] *= -1
        viewmat = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device).unsqueeze(0)
        target_np = np.asarray(
            Image.open(scene_root / "rgb_pilot" / f"{name}_rgb.png").convert("RGB")
        )
        target = torch.tensor(target_np.copy(), dtype=torch.float32, device=device).div(255).unsqueeze(0)
        rendered, _, _ = rasterization(
            means=xyz,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=Ks,
            width=width,
            height=height,
            packed=True,
            backgrounds=backgrounds,
            render_mode="RGB",
        )
        mae = float(F.l1_loss(rendered, target).item())
        mse = float(F.mse_loss(rendered, target).item())
        psnr = float(-10 * math.log10(max(mse, 1e-12)))
        save_rgb(rendered, out / f"{name}_decoded.png")
        Image.fromarray(target_np).save(out / f"{name}_target.png")
        per_view.append({"name": name, "mae": mae, "mse": mse, "psnr_db": psnr})
        print(f"{name}: MAE={mae:.6f} PSNR={psnr:.2f}dB", flush=True)

    metrics = {
        "scene_id": args.scene_id,
        "input_latents": int(coords.shape[0]),
        "decoded_gaussians": count,
        "scale_clip_min_m": 1e-4,
        "scale_clip_max_m": 2 * VOXEL_SIZE,
        "scale_clipped_fraction": float((scale_unclamped > 2 * VOXEL_SIZE).float().mean().item()),
        "decoded_checkpoint": str(decoded_ckpt),
        "position_roundtrip_max_error_m": max_position_error,
        "views": per_view,
        "mean_mae": float(np.mean([x["mae"] for x in per_view])),
        "mean_psnr_db": float(np.mean([x["psnr_db"] for x in per_view])),
        "peak_gpu_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print("VQ DECODE AND GSPLAT RENDER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
