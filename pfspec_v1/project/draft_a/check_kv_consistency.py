import torch
from omegaconf import OmegaConf

from model.gaussian_gpt import GaussianGPT
from model.gaussian_vqvae import GaussianVQVAE
from model.gpt.gpt import NanoGPTGaussianModel
from model.gpt.engine import KVCache
from draft_a.eval_overlap import sampling_probs

EOS = 12096
DEVICE = "cuda"

torch.set_float32_matmul_precision("high")

ckpt = torch.load(
    "draft_a/checkpoints/ce_43_v2/best.pt",
    map_location="cpu",
    weights_only=True,
)

cfg = OmegaConf.create(ckpt["config"])

draft = NanoGPTGaussianModel(
    config=cfg,
    codebook_size=4096,
    tokens_per_latent=2,
    num_feature_tokens=1,
).to(DEVICE)

draft.load_state_dict(ckpt["model"], strict=True)
draft.eval().requires_grad_(False)

vqvae = GaussianVQVAE.load_from_checkpoint(
    "checkpoints/vqvae_vfront.ckpt",
    map_location="cpu",
).eval()

target = GaussianGPT.load_from_checkpoint(
    "checkpoints/gpt_vfront.ckpt",
    vqvae=vqvae,
    map_location="cpu",
).eval()

target.gpt = target.gpt.to(DEVICE)
target.requires_grad_(False)

# Use the original module rather than invoking torch.compile
# to keep this correctness test independent of compilation.
target_wrapper = (
    target.gpt._orig_mod
    if hasattr(target.gpt, "_orig_mod")
    else target.gpt
)

models = {
    "target": target_wrapper.transformer,
    "draft": draft.transformer,
}

name = ckpt["val_files"][0]

record = torch.load(
    "draft_a/data/teacher_sequences/" + name,
    map_location="cpu",
    weights_only=True,
)

body = record["tokens"].long()
seq = torch.cat([torch.tensor([EOS]), body])

test_positions = [0, 1, 256, 257, 1022, 1023]
test_positions = [
    t for t in test_positions if t < len(seq)
]

print("Validation file:", name, flush=True)

def make_cache(core, sequence_length):
    config = core.config
    return KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=sequence_length,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
        device=DEVICE,
        dtype=next(core.parameters()).dtype,
    )

def evaluate_one(core, t):
    x = seq[:t + 1].unsqueeze(0).to(DEVICE)

    # Offline forward: slot mask is applied internally,
    # but Position monotonicity must be added manually.
    logits_full = core(x)[0, -1]

    previous_position = None
    if t % 2 == 0 and t >= 2:
        previous_position = int(seq[t - 1].item())

    p_full = sampling_probs(
        logits_full,
        previous_position,
        temperature=0.9,
        top_p=0.9,
    )

    # Cached forward: both slot mask and Position
    # monotonicity are applied inside GaussianGPT.
    cache = make_cache(core, x.shape[1])
    logits_cached = core(
        x, kv_cache=cache
    )[0, -1]

    p_cached = sampling_probs(
        logits_cached,
        None,
        temperature=0.9,
        top_p=0.9,
    )

    tv = 0.5 * (
        p_full - p_cached
    ).abs().sum().item()

    return p_full, p_cached, tv


with torch.inference_mode():
    for t in test_positions:
        print(
            f"\n===== t={t} "
            f"type={'Position' if t % 2 == 0 else 'Feature'} =====",
            flush=True,
        )

        probabilities = {}

        for name, core in models.items():
            p_full, p_cached, tv = evaluate_one(
                core, t
            )

            probabilities[name] = p_cached

            print(
                f"{name}: "
                f"offline_vs_KV_TV={tv:.8f} "
                f"max_prob={p_cached.max().item():.6f} "
                f"support={(p_cached > 0).sum().item()}",
                flush=True,
            )

            if tv > 0.05:
                print(
                    "WARNING: large offline/KV difference!",
                    flush=True,
                )

        target_p = probabilities["target"]
        draft_q = probabilities["draft"]

        overlap = torch.minimum(
            target_p, draft_q
        ).sum().item()

        print(
            f"Target-Draft KV overlap={overlap:.6f}",
            flush=True,
        )

        if t % 2 == 0:
            print(
                "Target EOS probability:",
                target_p[EOS].item(),
                flush=True,
            )
            print(
                "Draft EOS probability:",
                draft_q[EOS].item(),
                flush=True,
            )

print("\nKV CONSISTENCY CHECK COMPLETE", flush=True)


# Compare one-shot KV prefill with true
# token-by-token incremental KV decoding.
print("\n===== INCREMENTAL KV CHECK =====", flush=True)

check_positions = [0, 1, 256, 257]
check_positions = [
    t for t in check_positions if t < len(seq)
]

max_t = max(check_positions)
failed = False

with torch.inference_mode():
    for model_name, core in models.items():
        print(
            f"\nModel: {model_name}",
            flush=True,
        )

        # One persistent cache, updated token by token.
        incremental_cache = make_cache(
            core, max_t + 1
        )

        for t in range(max_t + 1):
            token = seq[t:t + 1].reshape(1, 1).to(DEVICE)

            incremental_logits = core(
                token,
                kv_cache=incremental_cache,
            )[0, -1]

            if t not in check_positions:
                continue

            # Reference: write the full prefix into
            # a fresh KV cache in a single forward.
            full_x = (
                seq[:t + 1]
                .reshape(1, -1)
                .to(DEVICE)
            )

            reference_cache = make_cache(
                core, t + 1
            )

            reference_logits = core(
                full_x,
                kv_cache=reference_cache,
            )[0, -1]

            p_inc = sampling_probs(
                incremental_logits,
                None,
                temperature=0.9,
                top_p=0.9,
            )

            p_ref = sampling_probs(
                reference_logits,
                None,
                temperature=0.9,
                top_p=0.9,
            )

            tv = (
                0.5 * (p_inc - p_ref).abs().sum()
            ).item()

            print(
                f"t={t:04d} "
                f"incremental_vs_prefill_TV={tv:.8f}",
                flush=True,
            )

            if tv > 0.01:
                print(
                    "WARNING: KV discrepancy exceeds 0.01",
                    flush=True,
                )
                failed = True

            del reference_cache

print("\n===== INCREMENTAL SUMMARY =====", flush=True)

if failed:
    print("INCREMENTAL KV CHECK: NEEDS INVESTIGATION")
else:
    print("INCREMENTAL KV CHECK: PASSED")
