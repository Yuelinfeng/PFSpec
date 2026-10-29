from pathlib import Path

import torch

from data.common import collate_fn
from data.vfront_dataset import VFrontGaussianDataset
from model.gaussian_vqvae import GaussianVQVAE
from utils.gaussian_vqvae_utils import split_batch_dict

torch.set_float32_matmul_precision("high")

scene_id = (
    Path("draft_a/vfront_pilot_scene.txt")
    .read_text().strip().removesuffix(".json")
)

root = (
    Path("/root/autodl-tmp/3D-FRONT-pilot")
    / scene_id
)

checkpoint = (
    root / "gaussian_init"
    / "gsplat_multiview/ckpt.pth"
)

vqvae_checkpoint = Path(
    "checkpoints/vqvae_vfront.ckpt"
)

output = (
    root / "gaussian_init"
    / "vfront_pilot_tokens.pt"
)

assert checkpoint.is_file(), checkpoint
assert vqvae_checkpoint.is_file(), vqvae_checkpoint
assert torch.cuda.is_available()

device = torch.device("cuda")

print("===== LOAD GAUSSIANS =====", flush=True)

dataset = VFrontGaussianDataset(
    paths=[str(checkpoint)],
    split="tokenize",
    transforms_root=None,
    voxel_size=0.025,
    n_images=0,
    sh_degree=0,
)

sample = dataset.load_gaussians(str(checkpoint))
print("Input Gaussians:", len(sample["coords"]))

batch = collate_fn([sample])
points, _ = split_batch_dict(batch, device=device)

print("\n===== LOAD VQ-VAE =====", flush=True)

vqvae = GaussianVQVAE.load_from_checkpoint(
    str(vqvae_checkpoint),
    map_location="cpu",
)

vqvae = vqvae.to(device).eval()
vqvae.requires_grad_(False)

codebook_size = int(vqvae.autoencoder.vq.codebook_size)

print("VQ Codebook size:", codebook_size)
print("VQ tokens per latent:", vqvae.autoencoder.vq.num_tokens)

print("\n===== TOKENIZE =====", flush=True)

with torch.inference_mode():
    result = vqvae.tokenize(
        dict(points),
        sort_latents="z",
    )

coords = result["coords"].detach().cpu().long()
feature_ids = result["feature_ids"].detach().cpu().long()

print("Coords shape:", tuple(coords.shape))
print("Feature IDs shape:", tuple(feature_ids.shape))

assert coords.ndim == 2 and coords.shape[1] == 3
assert feature_ids.ndim == 2
assert feature_ids.shape[0] == coords.shape[0]
assert feature_ids.shape[1] == 1
assert coords.shape[0] > 0

min_id = int(feature_ids.min())
max_id = int(feature_ids.max())
unique_ids = int(torch.unique(feature_ids).numel())

assert min_id >= 0
assert max_id < codebook_size

coord_min = coords.min(dim=0).values
coord_max = coords.max(dim=0).values

print("\n===== TOKEN STATISTICS =====")
print("Number of latents:", len(coords))
print("Coord min:", coord_min.tolist())
print("Coord max:", coord_max.tolist())
print("Feature ID min:", min_id)
print("Feature ID max:", max_id)
print("Unique feature IDs:", unique_ids)
print("Codebook usage:", round(unique_ids / codebook_size, 6))

print("\nFirst 5 latents:")
for i in range(min(5, len(coords))):
    print(coords[i].tolist(), feature_ids[i].tolist())

output.parent.mkdir(parents=True, exist_ok=True)

torch.save({
    "coords": coords,
    "feature_ids": feature_ids,
}, output)

print("\nSaved:", output)

print(
    "Peak allocated GPU GiB:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 3),
)

print("\nVFRONT TOKENIZATION PILOT COMPLETE")
