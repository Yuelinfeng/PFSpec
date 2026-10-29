import argparse
import random
from pathlib import Path

import torch
from omegaconf import OmegaConf

from model.gpt.gpt import NanoGPTGaussianModel

EOS = 12096


def load_corpus(directory):
    records = []

    for path in sorted(Path(directory).glob("*.pt")):
        item = torch.load(
            path, map_location="cpu", weights_only=True
        )

        body = item["tokens"].long()
        ended = bool(item["has_eos"])

        assert body.ndim == 1
        assert len(body) > 0 and len(body) % 2 == 0

        pos = body[0::2]
        feat = body[1::2]

        assert ((pos >= 0) & (pos < 8000)).all()
        assert ((feat >= 8000) & (feat < 12096)).all()
        assert (pos[1:] > pos[:-1]).all()

        pieces = [
            torch.tensor([EOS], dtype=torch.long),
            body,
        ]

        if ended:
            pieces.append(
                torch.tensor([EOS], dtype=torch.long)
            )

        seq = torch.cat(pieces)

        records.append({
            "name": path.name,
            "seq": seq,
            "has_eos": ended,
        })

    if len(records) < 4:
        raise RuntimeError("Need at least 4 sequences")

    return records


def split_corpus(records, seed, val_fraction=0.2):
    rng = random.Random(seed)

    train = []
    val = []

    for ended in [False, True]:
        group = [
            r for r in records if r["has_eos"] == ended
        ]
        rng.shuffle(group)

        if len(group) >= 2:
            n_val = max(
                1, round(len(group) * val_fraction)
            )
            n_val = min(n_val, len(group) - 1)
        else:
            n_val = 0

        val.extend(group[:n_val])
        train.extend(group[n_val:])

    if not train or not val:
        raise RuntimeError("Train/val split is empty")

    return train, val


@torch.no_grad()
def evaluate(model, records, use_bf16):
    model.eval()

    loss_sum = 0.0
    token_count = 0

    for record in records:
        seq = record["seq"].unsqueeze(0).cuda()

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = model(
                input_ids=seq,
                labels=seq,
            )

        n = seq.shape[1] - 1
        loss_sum += float(loss.item()) * n
        token_count += n

    model.train()
    return loss_sum / token_count


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument(
        "--corpus",
        default="draft_a/data/teacher_sequences",
    )
    parser.add_argument(
        "--output",
        default="draft_a/checkpoints/draft_a_ce_pilot.pt",
    )

    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    assert torch.cuda.is_available()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint: {output}"
        )

    records = load_corpus(args.corpus)
    train, val = split_corpus(records, args.seed)

    print("===== DATASET =====", flush=True)
    print("Total:", len(records), flush=True)
    print("Train:", len(train), flush=True)
    print("Validation:", len(val), flush=True)

    for name, subset in [("train", train), ("val", val)]:
        print(f"\n{name.upper()} FILES:", flush=True)
        for r in subset:
            print(
                r["name"],
                "supervised_tokens=", len(r["seq"]) - 1,
                "has_eos=", r["has_eos"],
                flush=True,
            )

    cfg = OmegaConf.load(
        "conf/model/gpt_draft_a.yaml"
    )

    assert cfg.gpt_size.n_layer == 8
    assert cfg.gpt_size.n_embd == 512
    assert cfg.gpt_size.n_head == 8
    assert cfg.position_vocab_size == 8000
    assert cfg.rope_basis == "position"
    assert cfg.sparse_position_hard_constraints
    assert not cfg.shared_vocab

    for r in records:
        if len(r["seq"]) - 1 > cfg.n_ctx:
            raise RuntimeError(
                f"Sequence exceeds context: {r['name']}"
            )

    print("\n===== INITIALIZE NEW DRAFT =====", flush=True)

    # No checkpoint loading. Always random initialization.
    model = NanoGPTGaussianModel(
        config=cfg,
        codebook_size=4096,
        tokens_per_latent=2,
        num_feature_tokens=1,
    ).cuda()

    model.train()

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
        flush=True,
    )

    assert model.eos_token_id == EOS
    assert model.vocab_size == 12098

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    use_bf16 = torch.cuda.is_bf16_supported()
    print("BF16:", use_bf16, flush=True)

    initial_val = evaluate(model, val, use_bf16)
    print(
        "\nInitial validation loss:",
        round(initial_val, 5),
        flush=True,
    )

    rng = random.Random(args.seed + 1)
    order = []
    cursor = 0

    print("\n===== TRAIN =====", flush=True)

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
            "cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = model(
                input_ids=seq,
                labels=seq,
            )

        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        if not torch.isfinite(grad_norm):
            raise RuntimeError("Nonfinite gradients")

        optimizer.step()

        print(
            f"step={step:03d} "
            f"loss={loss.item():.5f} "
            f"grad={grad_norm.item():.4f} "
            f"tokens={seq.shape[1]-1} "
            f"eos={record['has_eos']} "
            f"sample={record['name']}",
            flush=True,
        )

    final_val = evaluate(model, val, use_bf16)

    print("\n===== SUMMARY =====", flush=True)
    print("Initial validation loss:", initial_val, flush=True)
    print("Final validation loss:", final_val, flush=True)
    print(
        "Peak GPU GiB:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        flush=True,
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": OmegaConf.to_container(
            cfg, resolve=True
        ),
        "steps": args.steps,
        "seed": args.seed,
        "train_files": [r["name"] for r in train],
        "val_files": [r["name"] for r in val],
        "validation_loss": final_val,
        "data_source": "synthetic_vfront_target",
    }

    # Preserve existing artifacts, including previous experiments.
    torch.save(checkpoint, output)

    print("Saved:", output, flush=True)
    print("\nDRAFT-A TRAINING PILOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
