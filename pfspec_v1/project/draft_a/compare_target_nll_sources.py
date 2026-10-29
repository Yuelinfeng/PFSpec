from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from data.common import PreprocessedDataset
from model.gaussian_vqvae import GaussianVQVAE
from model.gaussian_gpt import GaussianGPT

torch.manual_seed(0)
torch.set_float32_matmul_precision("high")

DEVICE = "cuda"
EOS = 12096
POS_VOCAB = 8000
NUM_TEACHER = 5

scene_id = "04396802-8d89-4377-8b28-4cadc9eb07c6"
real_path = (
    Path("/root/autodl-tmp/3D-FRONT-pilot")
    / scene_id
    / "gaussian_init/vfront_pilot_tokens.pt"
)

dataset = PreprocessedDataset(
    file_list=[str(real_path)],
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

real_body = dataset[0].long()
TOKEN_COUNT = len(real_body)

print("===== DATA =====", flush=True)
print("Real VFront body length:", TOKEN_COUNT)

assert TOKEN_COUNT > 0
assert TOKEN_COUNT % 2 == 0

teacher_root = Path("draft_a/data/teacher_sequences")
files = sorted(teacher_root.glob("*.pt"))

teacher_samples = []

for path in files:
    payload = torch.load(
        path, map_location="cpu", weights_only=True
    )
    if not isinstance(payload, dict):
        continue
    if "tokens" not in payload:
        continue

    body = torch.as_tensor(
        payload["tokens"]
    ).reshape(-1).long()

    if body.numel() < TOKEN_COUNT:
        continue

    body = body[:TOKEN_COUNT]
    positions = body[0::2]
    features = body[1::2]

    valid = (
        bool(((positions >= 0) & (positions < 8000)).all())
        and bool(((features >= 8000) & (features < 12096)).all())
        and bool((positions[1:] > positions[:-1]).all())
    )

    if not valid:
        continue

    teacher_samples.append((path.name, body))
    if len(teacher_samples) == NUM_TEACHER:
        break

print("Teacher files:", len(files))
print("Selected matched-length samples:", len(teacher_samples))

if not teacher_samples:
    raise RuntimeError(
        "No valid Teacher sequence long enough for comparison."
    )

for name, _ in teacher_samples:
    print("Selected:", name)

print("\n===== LOAD TARGET =====", flush=True)

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

gpt = gpt.to(DEVICE).eval()
gpt.requires_grad_(False)

assert int(gpt.eos_token_id) == EOS
assert TOKEN_COUNT <= int(gpt.gpt_config.sequence_len)

def score(body):
    body = body.to(DEVICE)

    sequence = torch.cat([
        torch.tensor([EOS], device=DEVICE),
        body,
    ]).unsqueeze(0)

    with torch.inference_mode():
        logits = gpt(sequence[:, :-1])
        gold = sequence[:, 1:]

        raw_nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            gold.reshape(-1),
            reduction="none",
        )

        # The previous Position constrains the next
        # Position. Feature slots are left unchanged.
        previous_positions = body[0::2][:-1]

        invalid = (
            torch.arange(
                POS_VOCAB, device=DEVICE
            ).unsqueeze(0)
            <= previous_positions.unsqueeze(1)
        )

        logits[:, 2::2, :POS_VOCAB].masked_fill_(
            invalid.unsqueeze(0),
            float("-inf"),
        )

        constrained_nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            gold.reshape(-1),
            reduction="none",
        )

    def summarize(nll):
        assert torch.isfinite(nll).all()

        return np.array([
            nll.mean().item(),
            nll[0::2].mean().item(),
            nll[1::2].mean().item(),
        ])

    return summarize(raw_nll), summarize(constrained_nll)

def display(label, values):
    print(
        f"{label}: "
        f"Overall={values[0]:.6f}, "
        f"Position={values[1]:.6f}, "
        f"Feature={values[2]:.6f}",
        flush=True,
    )

print("\n===== REAL VFRONT =====", flush=True)

real_raw, real_masked = score(real_body)
display("Real raw", real_raw)
display("Real constrained", real_masked)

print("\n===== TEACHER ROLLOUTS =====", flush=True)

teacher_raw = []
teacher_masked = []

for name, body in teacher_samples:
    raw, masked = score(body)
    teacher_raw.append(raw)
    teacher_masked.append(masked)

    print("\nFile:", name)
    display("Raw", raw)
    display("Constrained", masked)

teacher_raw = np.stack(teacher_raw)
teacher_masked = np.stack(teacher_masked)

print("\n===== SUMMARY =====", flush=True)

display("Real raw", real_raw)
display("Teacher raw mean", teacher_raw.mean(axis=0))
display("Real constrained", real_masked)
display(
    "Teacher constrained mean",
    teacher_masked.mean(axis=0),
)

print(
    "Teacher constrained std:",
    np.round(teacher_masked.std(axis=0), 6).tolist(),
)

print(
    "Real minus Teacher constrained:",
    np.round(
        real_masked - teacher_masked.mean(axis=0), 6
    ).tolist(),
)

print(
    "\nTARGET DATA SOURCE COMPARISON COMPLETE",
    flush=True,
)
