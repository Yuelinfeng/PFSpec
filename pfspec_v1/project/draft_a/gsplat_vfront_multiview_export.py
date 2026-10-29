import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization

torch.manual_seed(2026)
torch.set_float32_matmul_precision("high")

DEVICE = "cuda"
VOXEL = 0.025
N = 30000
STEPS = 90

scene_id = Path(
    "draft_a/vfront_pilot_scene.txt"
).read_text().strip().removesuffix(".json")

root = Path("/root/autodl-tmp/3D-FRONT-pilot") / scene_id
out = root / "gaussian_init/gsplat_multiview"
out.mkdir(parents=True, exist_ok=True)

with (root / "depth_pilot/cameras.json").open() as f:
    cfg = json.load(f)

W, H = int(cfg["width"]), int(cfg["height"])

K = np.array([
    [cfg["fx"], 0, cfg["cx"]],
    [0, cfg["fy"], cfg["cy"]],
    [0, 0, 1],
], dtype=np.float32)

Ks = torch.tensor(
    K, device=DEVICE
).unsqueeze(0)

cameras = []

for frame in cfg["frames"]:
    name = frame["name"]
    c2w = np.array(
        frame["camera_to_world"], dtype=np.float32
    )
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w).astype(np.float32)

    rgb = np.asarray(
        Image.open(
            root / "rgb_pilot" / f"{name}_rgb.png"
        ).convert("RGB")
    )
    assert rgb.shape == (H, W, 3)

    cameras.append({
        "name": name,
        "viewmat": torch.tensor(
            w2c, device=DEVICE
        ).unsqueeze(0),
        "target": torch.tensor(
            rgb.copy(),
            dtype=torch.float32,
            device=DEVICE,
        ).div(255).unsqueeze(0),
    })

assert len(cameras) == 3

with np.load(
    root / "gaussian_init/voxel_init_025m.npz",
    allow_pickle=False,
) as d:
    xyz = d["xyz"].copy()
    rgb = d["rgb"].copy()

rng = np.random.default_rng(2026)
ids = rng.choice(len(xyz), size=N, replace=False)
xyz = xyz[ids]
rgb = rgb[ids]

anchors_np = np.floor(
    xyz.astype(np.float64) / VOXEL
) * VOXEL

offset_np = xyz.astype(np.float64) - anchors_np

anchors = torch.tensor(
    anchors_np,
    dtype=torch.float32,
    device=DEVICE,
)

raw_offsets = torch.nn.Parameter(
    torch.tensor(
        np.arctanh(
            np.clip(offset_np / (1.5 * VOXEL), -0.95, 0.95)
        ).astype(np.float32),
        device=DEVICE,
    )
)

initial_scale = 0.020
frac = initial_scale / (2 * VOXEL)
scale_logit = math.log(frac / (1 - frac))

raw_scales = torch.nn.Parameter(
    torch.full((N, 3), scale_logit, device=DEVICE)
)

raw_opacity = torch.nn.Parameter(
    torch.zeros(N, device=DEVICE)
)

color = torch.tensor(
    rgb.astype(np.float32) / 255,
    device=DEVICE,
).clamp(0.01, 0.99)

raw_colors = torch.nn.Parameter(torch.logit(color))

quats = torch.zeros(N, 4, device=DEVICE)
quats[:, 0] = 1.0

optimizer = torch.optim.Adam([
    {"params": [raw_offsets], "lr": 0.002},
    {"params": [raw_scales], "lr": 0.005},
    {"params": [raw_opacity], "lr": 0.01},
    {"params": [raw_colors], "lr": 0.03},
])

def render(cam):
    return rasterization(
        means=anchors + 1.5 * VOXEL * torch.tanh(raw_offsets),
        quats=quats,
        scales=2 * VOXEL * torch.sigmoid(raw_scales),
        opacities=torch.sigmoid(raw_opacity),
        colors=torch.sigmoid(raw_colors),
        viewmats=cam["viewmat"],
        Ks=Ks,
        width=W,
        height=H,
        packed=True,
        backgrounds=torch.zeros(1, 3, device=DEVICE),
        render_mode="RGB",
    )

def save_image(image, path):
    arr = (
        image[0].detach().clamp(0, 1)
        .mul(255).byte().cpu().numpy()
    )
    Image.fromarray(arr).save(out / path)

@torch.no_grad()
def evaluate(tag):
    losses = []
    for cam in cameras:
        image, _, _ = render(cam)
        loss = F.l1_loss(image, cam["target"]).item()
        losses.append(loss)

        save_image(
            image,
            f'{cam["name"]}_{tag}.png',
        )

        if tag == "initial":
            save_image(
                cam["target"],
                f'{cam["name"]}_target.png',
            )

        print(
            f'{tag} {cam["name"]} L1={loss:.6f}',
            flush=True,
        )

    mean = float(np.mean(losses))
    print(f"{tag} MEAN L1={mean:.6f}", flush=True)
    return mean

print("===== MULTIVIEW SMOKE =====", flush=True)
print("Gaussians:", N)
print("Cameras:", [c["name"] for c in cameras])
print("Image:", W, H)

torch.cuda.reset_peak_memory_stats()

initial_mean = evaluate("initial")

for step in range(STEPS):
    cam = cameras[step % len(cameras)]

    optimizer.zero_grad(set_to_none=True)

    image, alpha, _ = render(cam)
    loss = F.l1_loss(image, cam["target"])

    assert torch.isfinite(loss).item()
    loss.backward()

    if step == 0:
        for name, param in [
            ("offsets", raw_offsets),
            ("scales", raw_scales),
            ("opacity", raw_opacity),
            ("colors", raw_colors),
        ]:
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all().item(), name
            print(name, "grad_norm", param.grad.norm().item())

    optimizer.step()

    if (step + 1) % 15 == 0:
        print(
            f'Step {step+1}/{STEPS} '
            f'camera={cam["name"]} '
            f'L1={loss.item():.6f}',
            flush=True,
        )

final_mean = evaluate("optimized")

# Export raw parameter representation expected by
# GaussianGPT data/vfront.py.
SH_C0 = 0.28209479177387814

with torch.no_grad():
    colors_rgb = torch.sigmoid(raw_colors)

    ckpt = {
        "anchor": anchors.cpu().numpy().astype(np.float32),
        "offset": raw_offsets.cpu().numpy().astype(np.float32),
        "scale": raw_scales.cpu().numpy().astype(np.float32),
        "opacity": raw_opacity[:, None].cpu().numpy().astype(np.float32),
        "rotation": F.normalize(
            quats, dim=-1
        ).cpu().numpy().astype(np.float32),
        "f_dc": (
            (colors_rgb - 0.5) / SH_C0
        ).cpu().numpy().astype(np.float32),
        "f_rest": np.zeros((N, 45), dtype=np.float32),
    }

    for key, value in ckpt.items():
        assert np.isfinite(value).all(), key
        print("Export field:", key, value.shape)

    path = out / "ckpt.pth"
    torch.save(ckpt, path)
    print("Checkpoint saved:", path)

# Verify the official GaussianGPT loader.
from data.vfront import load_pth_sparse_gaussian

loaded = load_pth_sparse_gaussian(
    str(path),
    sh_degree=0,
    voxel_size=VOXEL,
)

recovered_xyz = loaded["anchor"] + loaded["offset"]

expected_xyz = (
    anchors + 1.5 * VOXEL * torch.tanh(raw_offsets)
).detach().cpu().numpy()

position_error = float(
    np.max(np.abs(recovered_xyz - expected_xyz))
)

recovered_rgb = (
    loaded["rgb"] * SH_C0 + 0.5
)

color_error = float(
    np.max(
        np.abs(
            recovered_rgb
            - colors_rgb.detach().cpu().numpy()
        )
    )
)

print("Position reconstruction max error:", position_error)
print("Color reconstruction max error:", color_error)

assert position_error < 1e-5
assert color_error < 1e-5

from data.vfront_dataset import VFrontGaussianDataset

dataset = VFrontGaussianDataset(
    paths=[str(path)],
    split="tokenize",
    transforms_root=None,
    voxel_size=VOXEL,
    n_images=0,
    sh_degree=0,
)

sample = dataset.load_gaussians(str(path))

for key, value in sample.items():
    print("Dataset field:", key, tuple(value.shape))
    assert torch.isfinite(value).all(), key

print("OFFICIAL GAUSSIAN LOADER CHECK PASSED", flush=True)


print(
    "Mean L1 improvement:",
    round((initial_mean - final_mean) / initial_mean * 100, 2),
    "%",
)

print(
    "Peak PyTorch allocated GPU GiB:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 3),
)

print("Outputs:", out)
print("MULTIVIEW SMOKE COMPLETE", flush=True)
