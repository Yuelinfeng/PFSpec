#!/usr/bin/env python3
"""Warm-start LFQ with separate low6 and conditional high6 overlap objectives."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from lfq_scratch_model import LFQScratchDraft, make_bits
from train_lfq_scratch_draft import load_data, loss_metrics


def head_metrics(logits, teacher):
    p = teacher.float().clamp_min(0)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-15)
    q = torch.softmax(logits.float() / 0.9, -1)
    pt = p.reshape(-1, 64, 64).transpose(1, 2)
    qt = q.reshape(-1, 64, 64).transpose(1, 2)
    pl, ql = pt.sum(-1), qt.sum(-1)
    pc = pt / pl.unsqueeze(-1).clamp_min(1e-15)
    qc = qt / ql.unsqueeze(-1).clamp_min(1e-15)
    low = torch.minimum(pl, ql).sum(-1)
    conditional = (pl * torch.minimum(pc, qc).sum(-1)).sum(-1)
    return low, conditional


def objective(logits, teacher, bits, low_weight=0.5, conditional_weight=0.5):
    original, metrics = loss_metrics(logits, teacher, bits, overlap_weight=0.5, bit_weight=0.1)
    low, conditional = head_metrics(logits, teacher)
    loss = original + low_weight * (1 - low.mean()) + conditional_weight * (1 - conditional.mean())
    return loss, {"policy_overlap": metrics["overlap"], "low6_overlap": low,
                  "conditional_overlap": conditional, "ce": metrics["ce"]}


def restore(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("source") != "new_training_from_random_initialization":
        raise ValueError("Expected weights from the existing LFQ training lineage")
    config = saved["model_config"]
    names = ("mode", "hidden_dim", "width", "token_emb", "dropout", "temperature")
    if config["mode"] != "conditional" or float(config["temperature"]) != 0.9:
        raise ValueError("Expected conditional LFQ weights with temperature 0.9")
    model = LFQScratchDraft(**{key: config[key] for key in names})
    model.load_state_dict(saved["model_state"])
    return model, config


def sequence_ids(root, split, rows):
    saved = torch.load(root / f"states/{split}_state_pairs.pt", map_location="cpu", weights_only=True)
    ids = saved.get("sequence_id")
    if ids is None:
        return None
    if len(ids) != rows:
        raise ValueError("Sequence IDs and data rows differ")
    return ids.long()


def sampler(datasets, sequences, source_fractions, mode, samples, seed):
    parts, notes = [], []
    for dataset, ids, fraction in zip(datasets, sequences, source_fractions):
        if mode == "trajectories" and ids is not None:
            _, inverse, counts = torch.unique(ids, return_inverse=True, return_counts=True)
            weights = 1.0 / (len(counts) * counts[inverse].double())
            notes.append("equal probability per sequence")
        else:
            weights = torch.full((len(dataset),), 1.0 / len(dataset), dtype=torch.double)
            notes.append("equal probability per row" if mode == "rows" else "row sampling: sequence IDs unavailable")
        parts.append(weights * fraction)
    result = WeightedRandomSampler(torch.cat(parts), samples, replacement=True,
                                   generator=torch.Generator().manual_seed(seed))
    return result, notes


def epoch(model, loader, device, args, optimizer=None, sequences=None):
    model.train(optimizer is not None)
    codes = make_bits(device)
    sums, per_row, rows = {}, [], 0
    with torch.set_grad_enabled(optimizer is not None):
        for hidden, feature, position, teacher in loader:
            hidden, feature, position, teacher = [value.to(device) for value in (hidden, feature, position, teacher)]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, logits = model(hidden, feature, position)
            loss, metrics = objective(logits, teacher, codes, args.low_overlap_weight, args.conditional_overlap_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training objective")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().double().sum())
            if optimizer is None:
                per_row.append(metrics["policy_overlap"].detach().cpu())
            rows += len(hidden)
    result = {key: value / rows for key, value in sums.items()}
    result["rows"] = rows
    if optimizer is None:
        values = torch.cat(per_row)
        result["sequence_overlap"] = (sum(float(values[sequences == sid].mean()) for sid in torch.unique(sequences))
                                      / len(torch.unique(sequences)) if sequences is not None else float(values.mean()))
    return result


def save_weights(path, model, config, args, selected_epoch, validation, optimizer=None):
    saved = {"model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
             "source": "new_training_from_random_initialization",
             "training_method": "lfq_head_overlap_continuation",
             "model_config": config, "student_temperature": 0.9, "student_top_p": 0.9,
             "parent_checkpoint": str(args.from_checkpoint.resolve()), "selected_epoch": selected_epoch,
             "validation": validation, "training_config": vars_to_json(args),
             "optimizer_semantics": "New AdamW for each invocation; no automatic continuation."}
    if optimizer is not None:
        saved["optimizer_state"] = optimizer.state_dict()
        saved["cpu_rng_state"] = torch.get_rng_state()
        saved["python_rng_state"] = random.getstate()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(saved, temporary)
    os.replace(temporary, path)


def vars_to_json(args):
    return {key: str(value) if isinstance(value, Path) else [str(x) for x in value] if isinstance(value, list) else value
            for key, value in vars(args).items()}


def selftest():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    logits = torch.randn(3, 4096, requires_grad=True)
    p = torch.softmax(logits.detach() / 0.9, -1)
    low, conditional = head_metrics(logits, p)
    torch.testing.assert_close(low, torch.ones_like(low), atol=2e-6, rtol=0)
    torch.testing.assert_close(conditional, torch.ones_like(conditional), atol=2e-6, rtol=0)
    q = torch.softmax(logits / 0.9, -1).view(-1, 64, 64).transpose(1, 2)
    pt = p.view(-1, 64, 64).transpose(1, 2)
    pl, ql = pt.sum(-1), q.sum(-1)
    joint_ce = -(p * torch.log_softmax(logits / 0.9, -1)).sum(-1)
    split_ce = -(pl * ql.log()).sum(-1) - (pt * (q / ql.unsqueeze(-1)).log()).sum((1, 2))
    torch.testing.assert_close(joint_ce, split_ce, atol=2e-6, rtol=1e-6)
    sparse = torch.zeros_like(p)
    sparse[:, 0] = 1
    original, _ = objective(logits, sparse, make_bits("cpu"), 0, 0)
    expected, _ = loss_metrics(logits, sparse, make_bits("cpu"), overlap_weight=0.5, bit_weight=0.1)
    torch.testing.assert_close(original, expected)
    low, conditional = head_metrics(logits, sparse)
    (2 - low.mean() - conditional.mean()).backward()
    assert torch.isfinite(logits.grad).all() and float(logits.grad.abs().sum()) > 0
    datasets = [list(range(4))]
    ids = [torch.tensor([0, 1, 1, 1])]
    draw, _ = sampler(datasets, ids, [1.0], "trajectories", 10, 17)
    torch.testing.assert_close(draw.weights, torch.tensor([0.5, 1/6, 1/6, 1/6], dtype=torch.double))
    model = LFQScratchDraft(hidden_dim=32, width=16, token_emb=8, dropout=0)
    hidden, feat, pos = torch.randn(3, 32), torch.tensor([0, 2, 5]), torch.tensor([0, 17, 400])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _, z = model(hidden, feat, pos)
    value, _ = objective(z, sparse, make_bits("cpu"))
    value.backward()
    for head in (model.head_lo, model.head_hi_given_lo):
        assert torch.isfinite(head.weight.grad).all() and float(head.weight.grad.abs().sum()) > 0
    optimizer.step()
    restored = LFQScratchDraft(hidden_dim=32, width=16, token_emb=8, dropout=0)
    restored.load_state_dict(model.state_dict())
    model.eval(); restored.eval()
    torch.testing.assert_close(model(hidden, feat, pos)[1], restored(hidden, feat, pos)[1])
    print("PASS: head objectives, CE decomposition, zero-mass classes, gradients, sequence sampling, update and reload", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest-only", action="store_true")
    parser.add_argument("--from-checkpoint", type=Path, default=Path("profiling/feature_draft/lfq_new_onpolicy_warmstart/lfq_scratch_conditional_best.pt"))
    parser.add_argument("--offline-root", type=Path, default=Path("profiling/feature_draft/state_generation_512"))
    parser.add_argument("--online-dirs", type=Path, nargs="+", default=[Path("profiling/feature_draft/lfq_new_onpolicy")])
    parser.add_argument("--no-offline", action="store_true")
    parser.add_argument("--offline-fraction", type=float, default=0.4)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=164039)
    parser.add_argument("--low-overlap-weight", type=float, default=0.5)
    parser.add_argument("--conditional-overlap-weight", type=float, default=0.5)
    parser.add_argument("--sampling", choices=("rows", "trajectories"), default="rows")
    args = parser.parse_args()
    if args.selftest_only:
        selftest(); return
    if args.out_dir is None:
        parser.error("An output directory is required")
    if min(args.epochs, args.patience, args.samples_per_epoch, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error("Training sizes and learning rate must be positive")
    if min(args.low_overlap_weight, args.conditional_overlap_weight) < 0:
        parser.error("Objective weights must be nonnegative")
    if not args.no_offline and not 0 < args.offline_fraction < 1:
        parser.error("Expected offline fraction between zero and one")
    if args.out_dir.resolve() == args.from_checkpoint.parent.resolve() or args.out_dir.exists():
        parser.error("Use a new candidate directory separate from the input weights")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("Requested CUDA device is unavailable")
    model, config = restore(args.from_checkpoint)
    model.to(device)
    roots = ([] if args.no_offline else [args.offline_root]) + args.online_dirs
    datasets = [load_data(root, "train") for root in roots]
    seqs = [sequence_ids(root, "train", len(data)) for root, data in zip(roots, datasets)]
    online_rows = sum(len(ds) for ds in datasets[(0 if args.no_offline else 1):])
    fractions = ([len(ds)/online_rows for ds in datasets] if args.no_offline else
                 [args.offline_fraction] + [(1-args.offline_fraction)*len(ds)/online_rows for ds in datasets[1:]])
    validation_data = [load_data(root, "val") for root in args.online_dirs]
    val_seqs = [sequence_ids(root, "val", len(data)) for root, data in zip(args.online_dirs, validation_data)]
    val_loaders = [DataLoader(data, batch_size=args.batch_size, shuffle=False) for data in validation_data]
    initial = [epoch(model, loader, device, args, sequences=seq) for loader, seq in zip(val_loaders, val_seqs)]
    best = sum(value["sequence_overlap"] for value in initial) / len(initial)
    args.out_dir.mkdir(parents=True)
    best_epoch, history, wait = 0, [], 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    save_weights(args.out_dir/"lfq_scratch_conditional_best.pt", model, config, args, 0, initial)
    print("PARENT", json.dumps(initial), flush=True)
    notes = []
    for number in range(1, args.epochs + 1):
        started = time.perf_counter()
        draw, notes = sampler(datasets, seqs, fractions, args.sampling, args.samples_per_epoch, args.seed + number)
        loader = DataLoader(ConcatDataset(datasets), batch_size=args.batch_size, sampler=draw)
        train = epoch(model, loader, device, args, optimizer)
        validation = [epoch(model, loader, device, args, sequences=seq) for loader, seq in zip(val_loaders, val_seqs)]
        score = sum(value["sequence_overlap"] for value in validation) / len(validation)
        history.append({"epoch": number, "train": train, "validation": validation,
                        "sequence_overlap": score, "wall_seconds": time.perf_counter() - started})
        if score > best + 1e-6:
            best, best_epoch, wait = score, number, 0
            save_weights(args.out_dir/"lfq_scratch_conditional_best.pt", model, config, args, number, validation)
        else:
            wait += 1
        save_weights(args.out_dir/"last.pt", model, config, args, number, validation, optimizer)
        summary = {"args": vars_to_json(args), "parent_validation": initial, "best_epoch": best_epoch,
                   "best_sequence_overlap": best, "history": history, "sampling_notes": notes,
                   "source_fractions": fractions, "test_read": False,
                   "note": "Validation is development data. Selected epoch zero retains the parent weights."}
        temporary = args.out_dir/"summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2)+"\n")
        os.replace(temporary, args.out_dir/"summary.json")
        print("EPOCH", number, json.dumps(history[-1]), "BEST", best_epoch, flush=True)
        if wait >= args.patience:
            break
    print("COMPLETE", str(args.out_dir), "selected_epoch", best_epoch, flush=True)


if __name__ == "__main__":
    main()
