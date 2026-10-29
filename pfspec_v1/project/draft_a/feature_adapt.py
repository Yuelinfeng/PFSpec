"""Finite full-distribution Feature fine tuning of an existing trained Draft."""

import argparse
import gc
import json
import math
from pathlib import Path
import random
import time

import torch

from draft_a.feature_reuse import make_model, evaluate
from draft_a.feature_distill import _read_training, load_training_labels, feature_loss
from draft_a.learning_train import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--label-root", type=Path, default=Path("draft_a/runs/pf_feature_distill_20261014"))
    ap.add_argument("--evaluation-root", type=Path, default=Path("draft_a/runs/pf_learning_budget_20261013"))
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=24684)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "progress.json").exists():
        raise FileExistsError("This finite run already has progress")
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    plan = json.loads((args.label_root / "plan.json").read_text())
    records = _read_training(plan)
    probabilities = load_training_labels(args.label_root, records)
    initial = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = initial["config"]
    model = make_model(config)
    model.load_state_dict(initial["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    del initial
    order, cursor, trained_tokens, feature_tokens = [], 0, 0, 0
    started = time.monotonic()
    history = []
    best = -1.0
    milestones = sorted(set([0, min(200, args.steps), min(600, args.steps), args.steps]))
    metadata = dict(initial_checkpoint=str(args.checkpoint), steps=args.steps, lr=args.lr,
                    schedule="cosine to 0.1 times initial lr", seed=args.seed,
                    label_root=str(args.label_root), source_root=plan["source_root"],
                    train_files=[r["name"] for r in records],
                    objective="all F Target distributions, P and EOS hard labels",
                    feature_temperature=1.0, train_precision="BF16", evaluation_precision="FP32",
                    trajectories_per_update=2, parameters=sum(p.numel() for p in model.parameters()))
    write_json(args.output / "plan.json", metadata)

    def measure(step):
        nonlocal best
        path = args.output / f"step_{step:04d}.pt"
        torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        config=config, step=step, trained_tokens=trained_tokens,
                        trained_feature_tokens=feature_tokens, **metadata), path)
        result_path = args.output / f"step_{step:04d}_metrics.json"
        evaluate(path, Path(plan["source_root"]), args.evaluation_root, result_path, None)
        result = json.loads(result_path.read_text())
        f = result["splits"]["val"]["feature"]["pooled_overlap"]
        p = result["splits"]["val"]["position"]["pooled_overlap"]
        row = dict(step=step, feature=f, position=p, trained_tokens=trained_tokens,
                   trained_feature_tokens=feature_tokens, seconds=time.monotonic() - started,
                   checkpoint=str(path))
        history.append(row)
        if p >= 0.90 and f > best:
            best = f
            write_json(args.output / "selected.json", row)
        write_json(args.output / "history.json", history)
        print(json.dumps(dict(event="measurement", **row)), flush=True)
        model.train()

    measure(0)
    model.train()
    for step in range(1, args.steps + 1):
        batch = []
        for _ in range(2):
            if cursor == len(order):
                order = list(range(len(records)))
                rng.shuffle(order)
                cursor = 0
            batch.append(records[order[cursor]])
            cursor += 1
        tokens = sum(len(r["seq"]) - 1 for r in batch)
        rate = args.lr * (0.1 + 0.9 * (1 + math.cos(math.pi * (step - 1) / args.steps)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for record in batch:
            seq = record["seq"].cuda().unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=seq[:, :-1])[0]
                loss = feature_loss(logits, seq[0, 1:], probabilities[record["name"]].cuda())
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            weight = (len(record["seq"]) - 1) / tokens
            (loss * weight).backward()
            total += float(loss.detach()) * weight
            del seq, logits, loss
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite training gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        trained_tokens += tokens
        feature_tokens += sum(len(r["tokens"]) // 2 for r in batch)
        if step == 1 or step % 50 == 0:
            progress = dict(status="training", step=step, loss=total, lr=rate,
                            trained_tokens=trained_tokens, trained_feature_tokens=feature_tokens,
                            seconds=time.monotonic() - started)
            write_json(args.output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        if step in milestones:
            measure(step)
    write_json(args.output / "progress.json", dict(status="complete", step=args.steps,
               trained_tokens=trained_tokens, trained_feature_tokens=feature_tokens,
               seconds=time.monotonic() - started, selected=json.loads((args.output / "selected.json").read_text())))
    del model, optimizer, probabilities
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
