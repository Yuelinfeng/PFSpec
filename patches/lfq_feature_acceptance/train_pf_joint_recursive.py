#!/usr/bin/env python3
"""Matched CPU experiments for gradients through the F1-to-F2 latent state."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from collect_lfq_onpolicy_sequences import fingerprint
from lfq_scratch_model import LFQScratchDraft, make_bits
from pf3_depth2_model import PF3Depth2, load_base, load_depth2
from train_lfq_heads import sampler, sequence_ids
from train_lfq_scratch_draft import load_data, loss_metrics
from train_pf3_joint_refresh import sources


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def evaluate(base, depth2, data, ids, batch_size, second=False):
    base.eval(); depth2.eval()
    values = []
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
            if second:
                h, fprev, p1, f1, p2, teacher = batch
                x1, _ = base(h.float(), fprev, p1)
                _, logits = depth2(x1, f1, p2)
            else:
                h, fprev, p1, teacher = batch
                _, logits = base(h.float(), fprev, p1)
            _, metrics = loss_metrics(logits, teacher, make_bits("cpu"))
            values.append(metrics["overlap"].cpu())
    overlaps = torch.cat(values)
    ids = torch.as_tensor(ids)
    per = {str(int(sid)): float(overlaps[ids == sid].double().mean()) for sid in ids.unique()}
    return {"rows": len(data), "row_overlap": float(overlaps.double().mean()),
            "sequence_overlap": sum(per.values()) / len(per), "per_sequence": per}


def validate(base, depth2, f1_val, f1_ids, f2_val, args):
    first = evaluate(base, depth2, f1_val, f1_ids, args.batch_size)
    second = [{"path": str(s["path"]), **evaluate(base, depth2, s["dataset"], s["ids"],
                                                args.batch_size, second=True)} for s in f2_val]
    f2_mean = sum(s["sequence_overlap"] for s in second) / len(second)
    return {"f1": first, "f2": second, "f2_sequence_mean": f2_mean,
            "selection_score": first["sequence_overlap"] + args.f2_weight * f2_mean}


def train_epoch(base, depth2, first_loader, second_loader, opt, mode, args):
    base.train(); depth2.train()
    codes = make_bits("cpu")
    parameters = [p for model in (base, depth2) for p in model.parameters() if p.requires_grad]
    count = 0; sums = {"f1_overlap": 0., "f2_overlap": 0., "loss": 0.}
    if len(first_loader) != len(second_loader):
        raise ValueError("Matched loaders require the same number of batches")
    for first, second in zip(first_loader, second_loader):
        opt.zero_grad(set_to_none=True)
        h, fprev, p1, teacher = first
        _, z1 = base(h.float(), fprev, p1)
        loss1, m1 = loss_metrics(z1, teacher, codes)
        h, fprev, p1, f1_recorded, p2, teacher = second
        # Preserve the recorded prefix associated with this Teacher distribution.
        x1, _ = base(h.float(), fprev, p1)
        _, z2 = depth2(x1.detach() if mode == "detached" else x1, f1_recorded, p2)
        loss2, m2 = loss_metrics(z2, teacher, codes)
        loss = loss1 + args.f2_weight * loss2
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite joint objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        opt.step()
        n = len(h); count += n
        sums["f1_overlap"] += float(m1["overlap"].double().sum())
        sums["f2_overlap"] += float(m2["overlap"].double().sum())
        sums["loss"] += float(loss.detach()) * n
    return {"rows_per_depth": count, **{k: v / count for k, v in sums.items()}}


def save_pair(directory, base, depth2, args, number, validation):
    first = directory / "lfq_scratch_conditional_best.pt"
    second = directory / "pf3_depth2_best.pt"
    common = {"training_method": "pf_joint_recursive", "selected_epoch": number,
              "validation": validation, "mode": directory.name,
              "training_config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    torch.save({**common, "source": "new_training_from_random_initialization",
                "model_config": base.config, "student_temperature": .9, "student_top_p": .9,
                "parent_checkpoint": str(args.base_checkpoint.resolve()),
                "model_state": {k: v.detach().cpu() for k, v in base.state_dict().items()}}, first)
    torch.save({**common, "kind": "pf3_depth2_conditional_v1", "best_epoch": number,
                "policy": {"temperature": .9, "top_p": .9},
                "base_checkpoint": str(first.resolve()), "base_sha256": fingerprint(first),
                "parent_checkpoint": str(args.depth2_checkpoint.resolve()),
                "val_overlap": validation["f2_sequence_mean"],
                "depth2_state": {k: v.detach().cpu() for k, v in depth2.state_dict().items()}}, second)


def load_pair(directory):
    first = directory / "lfq_scratch_conditional_best.pt"
    base = load_base(first, torch.device("cpu"))
    depth2 = load_depth2(directory / "pf3_depth2_best.pt", base, torch.device("cpu"), fingerprint(first))
    return base, depth2


def difference(current, reference, seed):
    keys = sorted(current)
    if set(keys) != set(reference):
        raise ValueError("Paired validation requires identical trajectories")
    delta = np.array([current[k] - reference[k] for k in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), size=(3000, len(delta)))].mean(1)
    return {"sequences": len(delta), "mean_delta": float(delta.mean()),
            "improved": int((delta > 0).sum()), "descriptive_ci95": np.quantile(boot, [.025, .975]).tolist()}


def compare(current, reference, seed):
    return {"f1": difference(current["f1"]["per_sequence"], reference["f1"]["per_sequence"], seed),
            "f2": [{"path": c["path"], **difference(c["per_sequence"], r["per_sequence"], seed + i + 1)}
                   for i, (c, r) in enumerate(zip(current["f2"], reference["f2"]))],
            "f1_delta": current["f1"]["sequence_overlap"] - reference["f1"]["sequence_overlap"],
            "f2_delta": current["f2_sequence_mean"] - reference["f2_sequence_mean"]}


def selftest():
    torch.manual_seed(81)
    base = LFQScratchDraft(mode="conditional", hidden_dim=32, width=16, token_emb=8, dropout=0)
    depth2 = PF3Depth2(base)
    h = torch.randn(3, 32); f = torch.tensor([1, 70, 2048]); p = torch.tensor([15, 207, 7050])
    teacher = torch.softmax(torch.randn(3, 4096), -1)
    for mode in ("detached", "joint"):
        base.zero_grad(set_to_none=True); depth2.zero_grad(set_to_none=True)
        x1, _ = base(h, f, p)
        _, z = depth2(x1.detach() if mode == "detached" else x1, f, p + 1)
        loss, _ = loss_metrics(z, teacher, make_bits("cpu")); loss.backward()
        grads = [v.grad for v in base.parameters() if v.grad is not None]
        assert (not grads) if mode == "detached" else (sum(float(g.abs().sum()) for g in grads) > 0)
        assert all(v.grad is None for v in depth2.feat_embed.parameters())
        assert all(v.grad is None for v in depth2.pos_embed.parameters())
        assert base.head_lo.weight.grad is None and base.head_hi_given_lo.weight.grad is None
    print("PASS: F2 reaches F1 latent only in joint mode; copied embeddings remain fixed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest-only", action="store_true")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("profiling/feature_draft/lfq_new_onpolicy_warmstart/lfq_scratch_conditional_best.pt"))
    parser.add_argument("--depth2-checkpoint", type=Path, default=Path("profiling/feature_draft/pf3_joint_refresh_train/pf3_depth2_best.pt"))
    parser.add_argument("--offline-root", type=Path, default=Path("profiling/feature_draft/state_generation_512"))
    parser.add_argument("--online-root", type=Path, default=Path("profiling/feature_draft/lfq_new_onpolicy"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--f2-weight", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=164039)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.selftest_only:
        selftest(); return
    if args.out_dir is None or args.out_dir.exists():
        parser.error("A new output directory is required")
    if min(args.epochs, args.samples_per_epoch, args.batch_size) < 1 or min(args.lr, args.f2_weight) <= 0:
        parser.error("Training sizes and objective weights must be positive")
    collection_sha = fingerprint(args.base_checkpoint)
    f1_train = [load_data(root, "train") for root in (args.offline_root, args.online_root)]
    f1_train_ids = [sequence_ids(root, "train", len(data)) for root, data in zip((args.offline_root, args.online_root), f1_train)]
    f1_val = load_data(args.online_root, "val")
    f1_val_ids = sequence_ids(args.online_root, "val", len(f1_val))
    if f1_val_ids is None or f1_train_ids[1] is None:
        raise ValueError("Online data must include trajectory IDs")
    f2_paths = [Path("profiling/feature_draft") / name for name in
                ("pf3_depth2_data", "pf3_refresh_stale", "pf3_refresh_trained")]
    f2_train = sources(f2_paths, "train", collection_sha)
    f2_val = sources([f2_paths[0], f2_paths[2]], "val", collection_sha)
    train_ids = set().union(*(s["unique_seeds"] for s in f2_train), *(set(ids.tolist()) for ids in f1_train_ids if ids is not None))
    val_ids = set().union(*(s["unique_seeds"] for s in f2_val), set(f1_val_ids.tolist()))
    if train_ids & val_ids:
        raise ValueError(f"Train/val trajectory overlap: {sorted(train_ids & val_ids)[:12]}")
    f2_train_ids = [torch.as_tensor(s["ids"]) for s in f2_train]
    datasets2 = [s["dataset"] for s in f2_train]
    args.out_dir.mkdir(parents=True)
    comparison = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "test_read": False, "device": "cpu", "threads": 1,
                  "f1_source_fractions": [.4, .6], "f2_source_fractions": [.15, .45, .4],
                  "f2_validation_source_weights": [.5, .5],
                  "selection": "F1 sequence overlap + 0.25 F2 source-equal sequence overlap at default weight; not online joint acceptance",
                  "caution": "Recorded-prefix development data; confidence intervals are descriptive after epoch selection. Offline F1 trajectory IDs may be unavailable.",
                  "arms": {}}
    parent = None
    for mode in ("detached", "joint"):
        random.seed(args.seed); torch.manual_seed(args.seed)
        base = load_base(args.base_checkpoint, torch.device("cpu")).requires_grad_(True)
        depth2 = load_depth2(args.depth2_checkpoint, base, torch.device("cpu"), collection_sha).requires_grad_(True)
        depth2.feat_embed.requires_grad_(False); depth2.pos_embed.requires_grad_(False)
        directory = args.out_dir / mode; directory.mkdir()
        initial = validate(base, depth2, f1_val, f1_val_ids, f2_val, args)
        if parent is None:
            parent = initial; comparison["parent"] = parent
        elif initial != parent:
            raise ValueError("Both arms require identical initial validation")
        best = initial; best_epoch = 0; history = []
        save_pair(directory, base, depth2, args, 0, initial)
        opt = torch.optim.AdamW([v for m in (base, depth2) for v in m.parameters() if v.requires_grad], lr=args.lr, weight_decay=.01)
        for number in range(1, args.epochs + 1):
            start = time.perf_counter()
            draw1, notes1 = sampler(f1_train, f1_train_ids, [.4, .6], "trajectories", args.samples_per_epoch, args.seed + number)
            draw2, notes2 = sampler(datasets2, f2_train_ids, [.15, .45, .4], "trajectories", args.samples_per_epoch, args.seed + 1000 + number)
            first_loader = DataLoader(ConcatDataset(f1_train), batch_size=args.batch_size, sampler=draw1,
                                      generator=torch.Generator().manual_seed(args.seed + number))
            second_loader = DataLoader(ConcatDataset(datasets2), batch_size=args.batch_size, sampler=draw2,
                                       generator=torch.Generator().manual_seed(args.seed + 1000 + number))
            train = train_epoch(base, depth2, first_loader, second_loader, opt, mode, args)
            validation = validate(base, depth2, f1_val, f1_val_ids, f2_val, args)
            if validation["selection_score"] > best["selection_score"] + 1e-6:
                best = validation; best_epoch = number
                save_pair(directory, base, depth2, args, number, best)
            entry = {"epoch": number, "train": train, "validation": validation, "wall_seconds": time.perf_counter() - start}
            history.append(entry)
            write_json(directory / "summary.json", {"mode": mode, "parent": initial, "best_epoch": best_epoch,
                                                      "selected": best, "history": history, "sampling_notes": [notes1, notes2]})
            print("EPOCH", mode, number, "F1", validation["f1"]["sequence_overlap"],
                  "F2", validation["f2_sequence_mean"], "BEST", best_epoch, "SECONDS", entry["wall_seconds"], flush=True)
        # Confirm the actual saved pair with the existing inference loaders.
        restored_base, restored_depth2 = load_pair(directory)
        reloaded = validate(restored_base, restored_depth2, f1_val, f1_val_ids, f2_val, args)
        if reloaded != best:
            raise ValueError("Saved pair and selected validation differ")
        comparison["arms"][mode] = {"best_epoch": best_epoch, "selected": best,
                                    "vs_parent": compare(best, parent, args.seed), "pair_reload_equal": True,
                                    "rows_per_depth": args.epochs * args.samples_per_epoch}
        write_json(args.out_dir / "comparison.json", comparison)
    comparison["joint_vs_detached"] = compare(comparison["arms"]["joint"]["selected"],
                                               comparison["arms"]["detached"]["selected"], args.seed)
    write_json(args.out_dir / "comparison.json", comparison)
    print("COMPLETE", str(args.out_dir), json.dumps(comparison["joint_vs_detached"]), flush=True)


if __name__ == "__main__":
    main()
