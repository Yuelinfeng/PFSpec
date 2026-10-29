from pathlib import Path

import torch
import torch.nn.functional as F

from data.common import PreprocessedDataset
from model.gaussian_vqvae import GaussianVQVAE
from model.gaussian_gpt import GaussianGPT

torch.set_float32_matmul_precision("high")
torch.manual_seed(0)

device = "cuda"

scene = "04396802-8d89-4377-8b28-4cadc9eb07c6"
path = (
    Path("/root/autodl-tmp/3D-FRONT-pilot")
    / scene
    / "gaussian_init/vfront_pilot_tokens.pt"
)

dataset = PreprocessedDataset(
    file_list=[str(path)],
    num_position_tokens=1,
    position_vocab_size=8000,
    codebook_size=4096,
    chunk_shape=[20, 20, 20],
    dense_chunks=False,
    chunk_order="xyz",
    min_chunk_occupancy=0.3,
    max_chunk_attempts=10,
    chunk_origin=[None, None, -1],
    load_augmented_tokens=False,
    shared=False,
)

body = dataset[0].long()
positions = body[0::2]

assert body.numel() > 0
assert body.numel() % 2 == 0
assert torch.all(positions[1:] > positions[:-1])

print("===== REAL VFRONT SAMPLE =====", flush=True)
print("Number of latents:", len(positions))
print("Body tokens:", len(body))

print("\n===== LOAD FROZEN TARGET =====", flush=True)

vqvae = GaussianVQVAE.load_from_checkpoint(
    "checkpoints/vqvae_vfront.ckpt",
    map_location="cpu",
).eval()

target = GaussianGPT.load_from_checkpoint(
    "checkpoints/gpt_vfront.ckpt",
    vqvae=vqvae,
    map_location="cpu",
).eval()

gpt = target.gpt
if hasattr(gpt, "_orig_mod"):
    gpt = gpt._orig_mod

gpt = gpt.to(device).eval()
gpt.requires_grad_(False)

eos = int(gpt.eos_token_id)
n_ctx = int(gpt.gpt_config.sequence_len)

print("EOS/BOS:", eos)
print("Target context:", n_ctx)

assert eos == 12096
assert len(body) <= n_ctx

# BOS predicts the first Position token.
# No artificial EOS is added: this is a cropped scene.
sequence = torch.cat([
    torch.tensor([eos], dtype=torch.long),
    body,
]).unsqueeze(0).to(device)

def per_token_nll(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )

print("\n===== TARGET FORWARD =====", flush=True)

with torch.inference_mode():
    logits = gpt(sequence[:, :-1])
    gold = sequence[:, 1:]

    assert logits.shape[:2] == gold.shape
    assert logits.shape[-1] >= 12097

    raw_nll = per_token_nll(logits, gold)

    # Match the monotonic Position rule:
    # Position_i must be greater than Position_(i-1).
    n = len(positions)
    prev_positions = positions[:-1].to(device)

    pos_ids = torch.arange(
        8000, device=device
    ).unsqueeze(0)

    invalid = pos_ids <= prev_positions.unsqueeze(1)

    # Position predictions occur at indices 0,2,4,...
    # Index 0 has no previous Position to constrain.
    logits[:, 2::2, :8000].masked_fill_(
        invalid.unsqueeze(0),
        -torch.inf,
    )

    constrained_nll = per_token_nll(logits, gold)

    assert torch.isfinite(raw_nll).all()
    assert torch.isfinite(constrained_nll).all()

def report(name, values):
    position = values[0::2]
    feature = values[1::2]

    print(f"\n{name}")
    print("Overall NLL:", round(values.mean().item(), 6))
    print("Position NLL:", round(position.mean().item(), 6))
    print("Feature NLL:", round(feature.mean().item(), 6))
    print(
        "Position P90 NLL:",
        round(torch.quantile(position, 0.9).item(), 6),
    )
    print(
        "Feature P90 NLL:",
        round(torch.quantile(feature, 0.9).item(), 6),
    )

print("\n===== RESULTS =====")
report("Raw slot-masked Target", raw_nll)
report("Monotonic-constrained Target", constrained_nll)

print(
    "\nPeak GPU allocated GiB:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 3),
)

print("\nTARGET REAL VFRONT SCORING COMPLETE")
