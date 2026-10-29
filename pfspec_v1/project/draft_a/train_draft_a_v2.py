import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from model.gpt.gpt import NanoGPTGaussianModel
from draft_a.train_draft_a import load_corpus, split_corpus

EOS = 12096


@torch.inference_mode()
def evaluate(model, records, bf16):
    model.eval()

    totals = {
        "all": [0.0, 0],
        "position": [0.0, 0],
        "feature": [0.0, 0],
        "eos": [0.0, 0],
    }

    for record in records:
        seq = record["seq"].unsqueeze(0).cuda()
        x = seq[:, :-1]
        y = seq[:, 1:].reshape(-1)

        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=bf16
        ):
            logits = model(input_ids=x)

        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y,
            reduction="none",
        )

        masks = {
            "all": torch.ones_like(y, dtype=torch.bool),
            "position": (y >= 0) & (y < 8000),
            "feature": (y >= 8000) & (y < EOS),
            "eos": y == EOS,
        }

        for key, mask in masks.items():
            count = int(mask.sum())
            if count:
                totals[key][0] += (
                    per_token_loss[mask].sum().item()
                )
                totals[key][1] += count

    model.train()

    result = {}
    for key, (loss_sum, count) in totals.items():
        result[key] = (
            loss_sum / count if count else float("nan")
        )
        result[key + "_count"] = count

    return result


def save_checkpoint(path, model, cfg, step, metrics,
                    train, val, seed):
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model.state_dict(),
        "config": OmegaConf.to_container(
            cfg, resolve=True
        ),
        "step": step,
        "seed": seed,
        "metrics": metrics,
        "train_files": [r["name"] for r in train],
        "val_files": [r["name"] for r in val],
        "source": "synthetic_vfront_target",
    }

    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument(
        "--output-dir",
        default="draft_a/checkpoints/ce_43_v2",
    )
    args = parser.parse_args()

    assert args.steps > 0
    assert args.eval_every > 0
    assert torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory exists: {output_dir}"
        )

    records = load_corpus(
        "draft_a/data/teacher_sequences"
    )
    train, val = split_corpus(records, args.seed)

    cfg = OmegaConf.load(
        "conf/model/gpt_draft_a.yaml"
    )

    assert cfg.gpt_size.n_layer == 8
    assert cfg.gpt_size.n_embd == 512
    assert cfg.gpt_size.n_head == 8
    assert cfg.position_vocab_size == 8000
    assert cfg.rope_basis == "position"
    assert cfg.sparse_position_hard_constraints

    for record in records:
        assert len(record["seq"]) - 1 <= cfg.n_ctx

    print("===== DATASET =====", flush=True)
    print("Total sequences:", len(records), flush=True)
    print("Train sequences:", len(train), flush=True)
    print("Val sequences:", len(val), flush=True)
    print(
        "Train natural EOS:",
        sum(r["has_eos"] for r in train),
        flush=True,
    )
    print(
        "Val natural EOS:",
        sum(r["has_eos"] for r in val),
        flush=True,
    )

    print("\n===== FRESH DRAFT =====", flush=True)

    # Always initialize new random weights.
    # Never load the previous smoke/pilot checkpoint.
    model = NanoGPTGaussianModel(
        config=cfg,
        codebook_size=4096,
        tokens_per_latent=2,
        num_feature_tokens=1,
    ).cuda()

    assert model.eos_token_id == EOS

    model.train()
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    bf16 = torch.cuda.is_bf16_supported()

    baseline = evaluate(model, val, bf16)
    print("Initial validation:", baseline, flush=True)

    rng = random.Random(args.seed + 1)
    order = []
    cursor = 0
    best_val = float("inf")

    output_dir.mkdir(parents=True, exist_ok=False)

    for step in range(1, args.steps + 1):
        if cursor >= len(order):
            order = list(train)
            rng.shuffle(order)
            cursor = 0

        record = order[cursor]
        cursor += 1

        seq = record["seq"].unsqueeze(0).cuda()
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=bf16
        ):
            loss = model(input_ids=seq, labels=seq)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Nonfinite training loss at step {step}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        if not torch.isfinite(grad_norm):
            raise RuntimeError("Nonfinite gradient")

        optimizer.step()

        print(
            f"step={step:04d} "
            f"train_loss={loss.item():.5f} "
            f"tokens={seq.shape[1]-1} "
            f"eos={record['has_eos']} "
            f"grad={grad_norm.item():.4f}",
            flush=True,
        )

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val, bf16)

            print(
                f"===== VALIDATION step={step} =====",
                flush=True,
            )
            print(metrics, flush=True)

            if metrics["all"] < best_val:
                best_val = metrics["all"]

                save_checkpoint(
                    output_dir / "best.pt",
                    model, cfg, step, metrics,
                    train, val, args.seed,
                )
                print("Saved new best checkpoint", flush=True)

    final_metrics = evaluate(model, val, bf16)

    save_checkpoint(
        output_dir / "last.pt",
        model, cfg, args.steps,
        final_metrics, train, val, args.seed,
    )

    print("\n===== FINAL SUMMARY =====", flush=True)
    print("Initial validation:", baseline, flush=True)
    print("Final validation:", final_metrics, flush=True)
    print("Best validation CE:", best_val, flush=True)
    print(
        "Peak GPU GiB:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        flush=True,
    )

    print("Output:", output_dir, flush=True)
    print("DRAFT-A V2 TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
