#!/usr/bin/env python3
"""Cache full Target Feature probabilities and causal states for adaptation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from draft_a.corpus_v2 import EOS, load_target
from draft_a.learning_eval import _fp32_runtime


FEATURE_START = 8000
FEATURE_END = 12096
LABEL_POLICY = {
    "temperature": 1.0,
    "top_p": None,
    "dtype": "FP32",
    "support": "all 4096 allowed Feature ids",
    "slots": "all Feature prediction slots with a previous P/F pair",
}


def read_records(rollout_root: Path) -> list[dict]:
    plan = json.loads((rollout_root / "rollout_plan.json").read_text())
    if plan["status"] != "complete":
        raise ValueError("Target rollout generation must complete first")
    records = []
    for entry in plan["trajectories"]:
        row = torch.load(entry["path"], map_location="cpu", weights_only=True)
        if (row["name"] != entry["name"] or row["split"] != "adaptation_pool"
                or row["teacher"] != "checkpoints/gpt_vfront.ckpt"):
            raise ValueError(f"Rollout metadata differs from its plan: {entry['path']}")
        row["label_split"] = "adaptation"
        records.append(row)
    return records


def read_development(crops_path: Path) -> list[dict]:
    crops = torch.load(crops_path, map_location="cpu", weights_only=True)
    records = []
    for crop in crops:
        records.append(dict(
            name=f"{crop['scene']}_crop_{crop['crop_seed']}",
            split="development",
            label_split="development",
            scene=crop["scene"],
            crop_seed=crop["crop_seed"],
            real_prefix_tokens=len(crop["tokens"]),
            seed_base=None,
            tokens=crop["tokens"].long().cpu(),
            has_eos=False,
            source_crop=crop["path"],
            teacher="checkpoints/gpt_vfront.ckpt",
        ))
    return records


def label_record(target, capture, record: dict, output: Path) -> dict:
    tokens = record["tokens"].long().cpu()
    pairs = len(tokens) // 2
    if pairs < 2:
        raise ValueError(f"Need at least two P/F pairs: {record['name']}")
    depth = torch.arange(1, pairs)
    eos = torch.tensor([EOS], dtype=torch.long) if record["has_eos"] else torch.empty(0, dtype=torch.long)
    seq = torch.cat([torch.tensor([EOS], dtype=torch.long), tokens, eos])
    capture.output = None
    logits = target(input_ids=seq[:-1].unsqueeze(0).cuda())[0]
    if capture.output is None:
        raise RuntimeError(f"Target state hook did not run for {record['name']}")
    states = capture.output[0]
    indices = (2 * depth + 1).cuda()
    feature_logits = logits.index_select(0, indices)[:, FEATURE_START:FEATURE_END].float()
    if (not bool(torch.isfinite(feature_logits).all())
            or not bool(torch.isneginf(logits.index_select(0, indices)[:, :FEATURE_START]).all())
            or not bool(torch.isneginf(logits.index_select(0, indices)[:, FEATURE_END:]).all())):
        raise ValueError(f"Target Feature support is invalid for {record['name']}")
    probabilities = feature_logits.softmax(-1).cpu()
    previous_state = states.index_select(0, indices - 2).float().cpu()
    count = len(depth)
    value = dict(
        name=record["name"],
        split=record["label_split"],
        scene=record["scene"],
        crop_seed=record["crop_seed"],
        real_prefix_tokens=record["real_prefix_tokens"],
        seed_base=record.get("seed_base"),
        tokens=tokens,
        has_eos=bool(record["has_eos"]),
        teacher="checkpoints/gpt_vfront.ckpt",
        label_policy=LABEL_POLICY,
        depth=depth,
        previous_state=previous_state,
        previous_feature=seq[2 * depth] - FEATURE_START,
        position=seq[2 * depth + 1],
        sampled_feature=seq[2 * depth + 2] - FEATURE_START,
        target=probabilities,
        rows=count,
        source_crop=record.get("source_crop"),
        rollout_tokens=max(0, len(tokens) - record["real_prefix_tokens"]),
    )
    if (probabilities.shape != (count, FEATURE_END - FEATURE_START)
            or not bool(torch.isfinite(probabilities).all())
            or not torch.allclose(probabilities.sum(-1), torch.ones(count), atol=1e-6, rtol=1e-6)
            or not torch.equal(value["position"], tokens[2::2][:count])
            or not torch.equal(value["previous_feature"], tokens[1:-1:2][:count] - FEATURE_START)):
        raise ValueError(f"Target state/probability rows do not align for {record['name']}")
    real_pairs = record["real_prefix_tokens"] // 2
    value["real_prefix_rows"] = min(count, max(0, real_pairs - 1))
    value["target_continuation_rows"] = count - value["real_prefix_rows"]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, output)
    capture.output = None
    del logits, states, indices, feature_logits, probabilities, previous_state, value
    return dict(
        name=record["name"],
        scene=record["scene"],
        rows=count,
        real_prefix_rows=min(count, max(0, real_pairs - 1)),
        target_continuation_rows=count - min(count, max(0, real_pairs - 1)),
        path=str(output),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--development-crops", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.out}")
    records = read_records(args.rollouts) + read_development(args.development_crops)
    args.out.mkdir(parents=True, exist_ok=True)
    adaptation = [row for row in records if row["label_split"] == "adaptation"]
    development = [row for row in records if row["label_split"] == "development"]
    plan = dict(
        status="running",
        teacher="checkpoints/gpt_vfront.ckpt",
        initial_draft="draft_a/runs/feature_target_continue_20261014/selected.pt",
        label_policy=LABEL_POLICY,
        state_precision="FP32",
        split_counts={
            "adaptation_trajectories": len(adaptation),
            "development_crops": len(development),
            "adaptation_scenes": len({row["scene"] for row in adaptation}),
            "development_scenes": len({row["scene"] for row in development}),
        },
        splits={"adaptation": [], "development": []},
    )
    started = time.monotonic()
    with torch.inference_mode(), _fp32_runtime(torch):
        target = load_target().float().eval().requires_grad_(False)
        from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock
        capture = CaptureLastBlock(target.transformer)
        try:
            total = len(records)
            for index, record in enumerate(records, 1):
                path = args.out / record["label_split"] / f"{record['name']}.pt"
                entry = label_record(target, capture, record, path)
                plan["splits"][record["label_split"]].append(entry)
                plan["completed"] = index
                plan["expected"] = total
                plan["elapsed_seconds"] = time.monotonic() - started
                if index % 10 == 0 or index == total:
                    (args.out / "label_plan.json").write_text(json.dumps(plan, indent=2))
                    print(json.dumps({
                        "completed": index,
                        "expected": total,
                        "split": record["label_split"],
                        "scene": record["scene"],
                        "rows": entry["rows"],
                    }), flush=True)
        finally:
            capture.close()
    plan.update(status="complete", elapsed_seconds=time.monotonic() - started)
    (args.out / "label_plan.json").write_text(json.dumps(plan, indent=2))
    print(json.dumps({
        "status": plan["status"],
        "split_counts": plan["split_counts"],
        "adaptation_rows": sum(row["rows"] for row in plan["splits"]["adaptation"]),
        "adaptation_real_prefix_rows": sum(row["real_prefix_rows"] for row in plan["splits"]["adaptation"]),
        "adaptation_target_continuation_rows": sum(row["target_continuation_rows"] for row in plan["splits"]["adaptation"]),
        "development_rows": sum(row["rows"] for row in plan["splits"]["development"]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
