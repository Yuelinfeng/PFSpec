import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization

DEVICE = "cuda"
VOXEL = 0.025
NUM_GAUSSIANS = 30000
STEPS = 20

torch.manual_seed(2026)
torch.set_float32_matmul_precision("high")

scene_id = Path(
    "draft_a/vfront_pilot_scene.txt"
).read_text().strip().removesuffix(".json")

root = Path("/root/autodl-tmp/3D-FRONT-pilot") / scene_id
out = root / "gaussian_init/gsplat_smoke"
out.mkdir(parents=True, exist_ok=True)

with (root / "depth_pilot/cameras.json").open() as f:
    cfg = json.load(f)

frame = cfg["frames"][0]
W, H = cfg["width"], cfg["height"]

# OpenGL camera-to-world -> OpenCV camera-to-world.
c2w = np.asarray(
    frame["camera_to_world"], dtype=np.float32
)
c2w_cv = c2w.copy()
c2w_cv[:3, 1:3] *= -1

w2c = np.linalg.inv(c2w_cv).astype(np.float32)

K = np.array([
    [cfg["fx"], 0, cfg["cx"]],
    [0, cfg["fy"], cfg["cy"]],
    [0, 0, 1],
], dtype=np.float32)

viewmats = torch.tensor(
    w2c, device=DEVICE
).unsqueeze(0)

Ks = torch.tensor(
    K, device=DEVICE
).unsqueeze(0)

target_image = np.asarray(
    Image.open(
        root / "rgb_pilot" / "living_0_rgb.png"
    ).convert("RGB")
)

assert target_image.shape == (H, W, 3)

target = (
    torch.from_numpy(target_image.copy())
    .to(DEVICE, dtype=torch.float32)
    .div(255.0)
    .unsqueeze(0)
)

with np.load(
    root / "gaussian_init/voxel_init_025m.npz",
    allow_pickle=False,
) as data:
    xyz = data["xyz"].copy()
    rgb = data["rgb"].copy()

rng = np.random.default_rng(2026)
chosen = rng.choice(
    len(xyz),
    size=min(NUM_GAUSSIANS, len(xyz)),
    replace=False,
)

xyz = xyz[chosen]
rgb = rgb[chosen]

# Initial Gaussian positions are represented
# as voxel anchors plus bounded offsets.
anchors_np = np.floor(
    xyz.astype(np.float64) / VOXEL
) * VOXEL

offset_np = xyz.astype(np.float64) - anchors_np

offset_init = np.arctanh(
    np.clip(
        offset_np / (1.5 * VOXEL),
        -0.95,
        0.95,
    )
).astype(np.float32)

anchors = torch.tensor(
    anchors_np, dtype=torch.float32, device=DEVICE
)

raw_offsets = torch.nn.Parameter(
    torch.tensor(offset_init, device=DEVICE)
)

# Official GaussianGPT VFront scale activation:
# scales = 2 * voxel_size * sigmoid(raw_scale).
initial_scale = 0.020
scale_fraction = initial_scale / (2 * VOXEL)
scale_logit = math.log(
    scale_fraction / (1 - scale_fraction)
)

raw_scales = torch.nn.Parameter(
    torch.full(
        (len(xyz), 3),
        scale_logit,
        device=DEVICE,
    )
)

opacity_logits = torch.nn.Parameter(
    torch.zeros(len(xyz), device=DEVICE)
)

initial_colors = (
    torch.tensor(
        rgb.astype(np.float32) / 255.0,
        device=DEVICE,
    )
    .clamp(0.01, 0.99)
)

color_logits = torch.nn.Parameter(
    torch.logit(initial_colors)
)

quats = torch.zeros(
    len(xyz), 4, device=DEVICE
)
quats[:, 0] = 1.0  # wxyz identity quaternion

optimizer = torch.optim.Adam([
    {"params": [raw_offsets], "lr": 0.002},
    {"params": [raw_scales], "lr": 0.005},
    {"params": [opacity_logits], "lr": 0.01},
    {"params": [color_logits], "lr": 0.03},
])

def render():
    means = (
        anchors
        + 1.5 * VOXEL * torch.tanh(raw_offsets)
    )
    scales = (
        2 * VOXEL * torch.sigmoid(raw_scales)
    )
    opacities = torch.sigmoid(opacity_logits)
    colors = torch.sigmoid(color_logits)

    image, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        packed=True,
        backgrounds=torch.zeros(
            1, 3, device=DEVICE
        ),
        render_mode="RGB",
    )

    return image, alpha, info

def save_image(image, filename):
    array = (
        image[0]
        .detach()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .cpu()
        .numpy()
    )
    Image.fromarray(array).save(out / filename)

print("===== GSPLAT SMOKE =====", flush=True)
print("GPU:", torch.cuda.get_device_name(0))
print("Gaussians:", len(xyz))
print("Image:", W, H)
print("Camera:", frame["name"])

# Check positive camera-space depth.
camera_xyz = (
    xyz @ w2c[:3, :3].T + w2c[:3, 3]
)
print(
    "Positive camera Z ratio:",
    round(float((camera_xyz[:, 2] > 0).mean()), 6),
    flush=True,
)

torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    initial_render, _, _ = render()
    save_image(initial_render, "initial.png")
    save_image(target, "target.png")

print("Initial render saved", flush=True)

for step in range(STEPS):
    optimizer.zero_grad(set_to_none=True)

    image, alpha, info = render()
    loss = F.l1_loss(image, target)

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss at step {step}"
        )

    loss.backward()

    if step == 0:
        for name, p in [
            ("offsets", raw_offsets),
            ("scales", raw_scales),
            ("opacities", opacity_logits),
            ("colors", color_logits),
        ]:
            if p.grad is None:
                raise RuntimeError(
                    f"Missing gradient: {name}"
                )
            if not torch.isfinite(p.grad).all():
                raise RuntimeError(
                    f"Non-finite gradient: {name}"
                )
            print(
                f"Grad {name}: "
                f"{p.grad.norm().item():.8f}",
                flush=True,
            )

    optimizer.step()

    if step in (0, 1, 4, 9, 19):
        torch.cuda.synchronize()
        print(
            f"Step {step+1:02d}/{STEPS} "
            f"L1={loss.item():.6f} "
            f"alpha_mean={alpha.mean().item():.6f}",
            flush=True,
        )

with torch.no_grad():
    final_render, _, _ = render()
    final_loss = F.l1_loss(
        final_render, target
    ).item()
    save_image(final_render, "optimized.png")

print("Final L1:", round(final_loss, 6))
print(
    "Peak GPU GiB:",
    round(
        torch.cuda.max_memory_allocated() / 1024**3,
        3,
    ),
)
print("Outputs:", out)
print("GSPLAT SMOKE COMPLETE", flush=True)
