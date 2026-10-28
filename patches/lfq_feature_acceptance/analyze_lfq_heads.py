#!/usr/bin/env python3
"""Describe LFQ head errors on saved train or validation rows using one CPU thread."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from lfq_scratch_model import LFQScratchDraft, policy_batch


def restore(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved["model_config"]
    names = ("mode", "hidden_dim", "width", "token_emb", "dropout", "temperature")
    model = LFQScratchDraft(**{key: config[key] for key in names})
    if config["mode"] != "conditional":
        raise ValueError("Expected the low6/high6 conditional model")
    model.load_state_dict(saved["model_state"])
    return model.eval()


def summarize(metrics, mask=None):
    return {key: float(value.mean() if mask is None else value[mask].mean())
            for key, value in metrics.items()}


def analyze(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root = args.data_dir
    states = torch.load(root / f"states/{args.split}_state_pairs.pt", map_location="cpu", weights_only=True)
    labels = torch.load(root / f"labels/{args.split}.pt", map_location="cpu", weights_only=True)
    if (float(labels["temperature"]), float(labels["top_p"])) != (0.9, 0.9):
        raise ValueError("Expected saved Target policy 0.9/0.9")
    ids = states["row_index"].long()
    seq = states["sequence_id"].long()
    rows = len(ids)
    if rows == 0 or len(seq) != rows:
        raise ValueError("Expected nonempty rows and sequence IDs")
    if not torch.equal(states["current_position"].long(), labels["input_ids"][ids, -1].long()):
        raise ValueError("Position input and teacher row differ")
    model = restore(args.checkpoint)
    outputs = {}
    with torch.inference_mode():
        for start in range(0, rows, args.batch_size):
            end = min(rows, start + args.batch_size)
            p = labels["teacher_probs"][ids[start:end]].float().clamp_min(0)
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-15)
            _, z = model(states["prev_state"][start:end].float(),
                         states["prev_feature"][start:end], states["current_position"][start:end])
            qraw = torch.softmax(z / 0.9, -1)
            q = policy_batch(z, 0.9, 0.9)
            pt = p.view(-1, 64, 64).transpose(1, 2)
            qt = qraw.view(-1, 64, 64).transpose(1, 2)
            pl, ql = pt.sum(-1), qt.sum(-1)
            pc = pt / pl.unsqueeze(-1).clamp_min(1e-15)
            qc = qt / ql.unsqueeze(-1).clamp_min(1e-15)
            pc_fill = torch.where(pl.unsqueeze(-1) > 0, pc, torch.full_like(pc, 1 / 64))
            ideal_low = (pl.unsqueeze(-1) * qc).transpose(1, 2).reshape(-1, 4096)
            ideal_high = (ql.unsqueeze(-1) * pc_fill).transpose(1, 2).reshape(-1, 4096)
            metrics = {
                "policy_overlap": torch.minimum(p, q).sum(-1),
                "raw_overlap": torch.minimum(p, qraw).sum(-1),
                "low6_overlap": torch.minimum(pl, ql).sum(-1),
                "conditional_overlap": (pl * torch.minimum(pc, qc).sum(-1)).sum(-1),
                "target_mass_in_draft_support": (p * (q > 0)).sum(-1),
                "target_entropy": -torch.xlogy(p, p).sum(-1),
                "draft_entropy": -torch.xlogy(q, q).sum(-1),
                "draft_support_size": (q > 0).float().sum(-1),
            }
            for name, distribution in (("teacher_low6_replacement", ideal_low),
                                       ("teacher_conditional_replacement", ideal_high)):
                replacement = policy_batch(0.9 * distribution.clamp_min(1e-30).log(), 0.9, 0.9)
                metrics[name] = torch.minimum(p, replacement).sum(-1)
            for top_p in (0.95, 1.0):
                metrics[f"overlap_top_p_{top_p}"] = torch.minimum(p, policy_batch(z, 0.9, top_p)).sum(-1)
            for key, value in metrics.items():
                outputs.setdefault(key, []).append(value)
            if end % 1024 < args.batch_size or end == rows:
                print(f"CPU rows={end}/{rows}", flush=True)
    metrics = {key: torch.cat(parts) for key, parts in outputs.items()}
    sequence_reports = []
    progress = torch.empty(rows)
    for sid in torch.unique(seq):
        selected = torch.where(seq == sid)[0]
        progress[selected] = torch.arange(len(selected)).float() / max(len(selected), 1)
        sequence_reports.append({"sequence_id": int(sid), "rows": len(selected),
                                 **summarize(metrics, seq == sid)})
    progress_reports = []
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        mask = (progress >= lo) & (progress < hi)
        progress_reports.append({"observed_row_progress": [lo, hi], "rows": int(mask.sum()),
                                 **summarize(metrics, mask)})
    cutoff = torch.quantile(metrics["policy_overlap"], 0.25)
    hard = metrics["policy_overlap"] <= cutoff
    return {
        "checkpoint": str(args.checkpoint.resolve()), "data_dir": str(root.resolve()),
        "split": args.split, "device": "cpu", "threads": args.threads, "rows": rows,
        "sequences": len(sequence_reports), "target_temperature": 0.9, "target_top_p": 0.9,
        "draft_temperature": 0.9, "draft_top_p": 0.9,
        "row_mean": summarize(metrics),
        "sequence_mean": {key: sum(item[key] for item in sequence_reports) / len(sequence_reports)
                          for key in metrics},
        "by_sequence": sequence_reports, "by_observed_progress": progress_reports,
        "lowest_overlap_quartile": {"cutoff": float(cutoff), "rows": int(hard.sum()),
                                    **summarize(metrics, hard)},
        "limitations": [
            "Validation was previously used for model selection; test data was not read.",
            "Observed row progress is ordinal within each sequence, not an exact token depth.",
            "Teacher replacements describe errors; those probabilities are unavailable to the draft at inference.",
            "Overlap is expected acceptance at these saved contexts, not measured online acceptance or throughput.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("profiling/feature_draft/lfq_new_onpolicy_warmstart/lfq_scratch_conditional_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("profiling/feature_draft/lfq_new_onpolicy"))
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("Batch size and thread count must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose the intended new result path")
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({key: result[key] for key in ("rows", "sequences", "row_mean", "sequence_mean")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
