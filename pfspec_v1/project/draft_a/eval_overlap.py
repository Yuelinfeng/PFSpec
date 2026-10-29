import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from model.gaussian_gpt import GaussianGPT
from model.gaussian_vqvae import GaussianVQVAE
from model.gpt.gpt import NanoGPTGaussianModel

EOS = 12096


def sampling_probs(raw_logits, prev_pos, temperature, top_p):
    logits = raw_logits.float().clone()

    # Position monotonicity. The model's normal full-sequence
    # forward pass does not apply the KV-cache-only mask.
    if prev_pos is not None:
        assert 0 <= prev_pos < 8000
        logits[:prev_pos + 1] = -torch.inf

    if not torch.isfinite(logits).any():
        raise RuntimeError("No legal sampling tokens")

    logits = logits / temperature

    # Match GaussianGPT's nucleus sampling implementation.
    sorted_logits, indices = torch.sort(
        logits, descending=True
    )
    sorted_probs = torch.softmax(sorted_logits, dim=0)

    remove = torch.cumsum(sorted_probs, dim=0) > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False

    sorted_logits[remove] = -torch.inf

    filtered = torch.full_like(logits, -torch.inf)
    filtered.scatter_(0, indices, sorted_logits)

    probs = torch.softmax(filtered, dim=0)

    assert torch.isfinite(probs).all()
    assert abs(probs.sum().item() - 1.0) < 1e-4

    return probs


def choose_slots(length, count):
    result = []

    for parity in (0, 1):
        slots = list(range(parity, length, 2))
        if not slots:
            continue

        n = min(len(slots), count)

        selected = {
            round(i * (len(slots) - 1) / max(n - 1, 1))
            for i in range(n)
        }
        result.extend(slots[j] for j in sorted(selected))

    return sorted(result)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max-scenes", type=int, default=3)
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--samples-per-type", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--checkpoint",
        default="draft_a/checkpoints/ce_43_v2/best.pt",
    )

    args = parser.parse_args()

    assert torch.cuda.is_available()
    assert args.temperature > 0
    assert 0 < args.top_p <= 1

    torch.set_float32_matmul_precision("high")

    print("===== LOAD DRAFT =====", flush=True)

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    cfg = OmegaConf.create(ckpt["config"])

    draft = NanoGPTGaussianModel(
        config=cfg,
        codebook_size=4096,
        tokens_per_latent=2,
        num_feature_tokens=1,
    ).cuda()

    draft.load_state_dict(ckpt["model"], strict=True)
    draft.eval().requires_grad_(False)

    print("Draft checkpoint step:", ckpt["step"], flush=True)

    print("===== LOAD TARGET =====", flush=True)

    vqvae = GaussianVQVAE.load_from_checkpoint(
        "checkpoints/vqvae_vfront.ckpt",
        map_location="cpu",
    ).eval()

    target = GaussianGPT.load_from_checkpoint(
        "checkpoints/gpt_vfront.ckpt",
        vqvae=vqvae,
        map_location="cpu",
    ).eval()

    # Only move the GPT to GPU.
    # VQ-VAE decoding is not needed for this experiment.
    target.gpt = target.gpt.cuda()
    target.requires_grad_(False)

    assert draft.eos_token_id == target.gpt.eos_token_id == EOS
    assert cfg.sparse_position_hard_constraints
    assert target.model_config.sparse_position_hard_constraints
    assert str(cfg.chunk_order) == "xyz"
    assert str(target.model_config.chunk_order) == "xyz"

    bf16 = torch.cuda.is_bf16_supported()

    metrics = {
        "position": [],
        "feature": [],
        "eos": [],
    }

    files = ckpt["val_files"][:args.max_scenes]

    print("Evaluation files:", files, flush=True)

    with torch.inference_mode():
        for name in files:
            record = torch.load(
                Path("draft_a/data/teacher_sequences") / name,
                map_location="cpu",
                weights_only=True,
            )

            body = record["tokens"].long()

            pieces = [
                torch.tensor([EOS]),
                body,
            ]

            if record["has_eos"]:
                pieces.append(torch.tensor([EOS]))

            seq = torch.cat(pieces)

            length = min(
                len(seq) - 1,
                args.max_context,
            )

            # seq[0] is BOS.
            # Input at t predicts seq[t+1].
            x = seq[:length].unsqueeze(0).cuda()

            slots = choose_slots(
                length, args.samples_per_type
            )

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=bf16,
            ):
                target_logits = target.gpt(
                    input_ids=x
                )[0, slots].float().cpu()

                draft_logits = draft(
                    input_ids=x
                )[0, slots].float().cpu()

            print(
                f"\n{name}: "
                f"context={length}, "
                f"positions_tested={len(slots)}",
                flush=True,
            )

            for j, t in enumerate(slots):
                is_position = t % 2 == 0

                prev_pos = (
                    int(seq[t - 1])
                    if is_position and t >= 2
                    else None
                )

                p = sampling_probs(
                    target_logits[j],
                    prev_pos,
                    args.temperature,
                    args.top_p,
                )

                q = sampling_probs(
                    draft_logits[j],
                    prev_pos,
                    args.temperature,
                    args.top_p,
                )

                overlap = torch.minimum(
                    p, q
                ).sum().item()

                kind = (
                    "position" if is_position else "feature"
                )
                metrics[kind].append(overlap)

                if int(seq[t + 1]) == EOS:
                    metrics["eos"].append({
                        "overlap": overlap,
                        "target_eos": p[EOS].item(),
                        "draft_eos": q[EOS].item(),
                    })

            del x, target_logits, draft_logits

    print("\n===== RESULTS =====", flush=True)

    for kind in ("position", "feature"):
        values = metrics[kind]

        if values:
            print(
                f"{kind}: "
                f"n={len(values)} "
                f"mean_overlap={sum(values)/len(values):.5f}",
                flush=True,
            )

    all_values = (
        metrics["position"] + metrics["feature"]
    )

    print(
        "Overall theoretical token acceptance:",
        sum(all_values) / len(all_values),
        flush=True,
    )

    print(
        "Terminal EOS states evaluated:",
        len(metrics["eos"]),
        flush=True,
    )

    for entry in metrics["eos"]:
        print("EOS:", entry, flush=True)

    print("\nOVERLAP SMOKE TEST COMPLETE", flush=True)


if __name__ == "__main__":
    main()
