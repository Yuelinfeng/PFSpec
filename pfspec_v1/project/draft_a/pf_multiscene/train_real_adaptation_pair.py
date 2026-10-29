#!/usr/bin/env python3
"""Train equal-budget old-only and mixed-source FeatureStateDraft controls."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from draft_a.feature_adapt_continue import feature_policy_tv
from draft_a.feature_target_v2 import HEAD, model_from, save


FIELDS = ("previous_state", "previous_feature", "position", "sampled_feature", "target")
MIXED_NEW_COUNTS = (614, 614, 615, 614, 615)


def load_old_pool(plan_path: Path) -> list[dict]:
    plan = json.loads(plan_path.read_text())
    if plan["status"] != "complete" or plan["teacher"] != "checkpoints/gpt_vfront.ckpt":
        raise ValueError("Completed original-Target training data is required")
    rows = []
    for item in plan["splits"]["train"]["records"]:
        data = torch.load(item["path"], map_location="cpu", weights_only=True)
        if data["name"] != item["name"] or len(data["position"]) != item["rows"]:
            raise ValueError(f"Old training data differs from its plan: {item['path']}")
        rows.append(data)
    return rows


def load_label_pool(label_plan_path: Path, split: str) -> list[dict]:
    plan = json.loads(label_plan_path.read_text())
    if plan["status"] != "complete" or plan["teacher"] != "checkpoints/gpt_vfront.ckpt":
        raise ValueError("Completed original-Target Feature labels are required")
    rows = []
    for item in plan["splits"][split]:
        data = torch.load(item["path"], map_location="cpu", weights_only=True)
        if (data["name"] != item["name"] or data["split"] != split
                or len(data["position"]) != item["rows"]):
            raise ValueError(f"Feature labels differ from their plan: {item['path']}")
        rows.append(data)
    return rows


def pool_summary(pool: list[dict]) -> dict:
    return dict(
        trajectories=len(pool),
        scenes=len({row.get("scene", row["name"]) for row in pool}),
        feature_rows=sum(len(row["position"]) for row in pool),
    )


def select_rows(pool: list[dict], count: int, rng: np.random.Generator,
                *, balance_scenes: bool) -> dict[str, torch.Tensor]:
    if count <= 0:
        return {key: torch.empty((0, *(pool[0][key].shape[1:])), dtype=pool[0][key].dtype)
                for key in FIELDS}
    if balance_scenes:
        by_scene = defaultdict(list)
        for index, row in enumerate(pool):
            by_scene[row.get("scene", row["name"])].append(index)
        scenes = sorted(by_scene)
        selected_scenes = rng.integers(0, len(scenes), size=count)
        row_ids = np.empty(count, dtype=np.int64)
        record_ids = np.empty(count, dtype=np.int64)
        for scene_index, scene in enumerate(scenes):
            slots = np.flatnonzero(selected_scenes == scene_index)
            scene_records = by_scene[scene]
            chosen = rng.integers(0, len(scene_records), size=len(slots))
            for local_index, record_id in enumerate(scene_records):
                positions = slots[chosen == local_index]
                if len(positions):
                    record_ids[positions] = record_id
                    row_ids[positions] = rng.integers(
                        0, len(pool[record_id]["position"]), size=len(positions)
                    )
    else:
        record_ids = rng.integers(0, len(pool), size=count)
        row_ids = np.empty(count, dtype=np.int64)
        for record_id in np.unique(record_ids):
            slots = np.flatnonzero(record_ids == record_id)
            row_ids[slots] = rng.integers(
                0, len(pool[int(record_id)]["position"]), size=len(slots)
            )

    batch = {
        key: torch.empty((count, *pool[0][key].shape[1:]), dtype=pool[0][key].dtype)
        for key in FIELDS
    }
    for record_id in np.unique(record_ids):
        slots = np.flatnonzero(record_ids == record_id)
        indices = torch.from_numpy(row_ids[slots])
        output_slots = torch.from_numpy(slots)
        record = pool[int(record_id)]
        for key in FIELDS:
            batch[key][output_slots] = record[key].index_select(0, indices)
    return batch


def objective(logits: torch.Tensor, target: torch.Tensor, weight: float):
    probabilities = target.float()
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    ce = -(probabilities * logits.float().log_softmax(-1)).sum(-1)
    tv = feature_policy_tv(logits, probabilities) if weight else torch.zeros_like(ce)
    return ce + weight * tv, ce, tv


def evaluate(model, records: list[dict], *, batch_size: int = 256) -> dict:
    model.eval()
    per_record = []
    with torch.inference_mode():
        for record in records:
            count = len(record["position"])
            overlap_sum = ce_sum = 0.0
            for start in range(0, count, batch_size):
                end = min(count, start + batch_size)
                batch = {key: record[key][start:end].cuda() for key in FIELDS}
                _, logits = model(
                    batch["previous_state"],
                    batch["previous_feature"],
                    batch["position"],
                )
                values, ce, _ = objective(logits, batch["target"], 0.0)
                del values
                target_policy = batch["target"] / batch["target"].sum(-1, keepdim=True)
                target_policy = target_policy.log()
                # The real acceptance policy uses nucleus truncation on both sides.
                from draft_a.feature_adapt_continue import policy_probs
                p = policy_probs(target_policy, 0.9, 0.9)
                q = policy_probs(logits, 0.9, 0.9)
                overlap_sum += float(torch.minimum(p, q).sum())
                ce_sum += float(ce.sum())
            rows = max(count, 1)
            per_record.append(dict(
                name=record["name"],
                scene=record.get("scene", record["name"]),
                rows=count,
                pooled_overlap=overlap_sum / rows,
                raw_ce=ce_sum / rows,
            ))
    by_scene = defaultdict(list)
    for row in per_record:
        by_scene[row["scene"]].append(row)
    scene_values = {
        scene: sum(item["pooled_overlap"] for item in rows) / len(rows)
        for scene, rows in by_scene.items()
    }
    total_rows = sum(row["rows"] for row in per_record)
    return dict(
        records=len(per_record),
        scenes=len(by_scene),
        rows=total_rows,
        pooled_overlap=(sum(row["pooled_overlap"] * row["rows"] for row in per_record)
                        / total_rows if total_rows else None),
        equal_scene_overlap=(sum(scene_values.values()) / len(scene_values)
                             if scene_values else None),
        pooled_raw_ce=(sum(row["raw_ce"] * row["rows"] for row in per_record)
                       / total_rows if total_rows else None),
        per_scene=scene_values,
        by_record=per_record,
    )


def save_checkpoint(path: Path, model, initial: dict, global_step: int,
                    exposure: int, group: str, plan: dict, metrics: dict):
    torch.save(dict(
        model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
        model_config=initial["model_config"],
        student_temperature=0.9,
        student_top_p=0.9,
        teacher="checkpoints/gpt_vfront.ckpt",
        step=global_step,
        trained_feature_rows=exposure,
        phase=plan["phase"],
        adaptation_group=group,
        original_checkpoint=plan["initial_checkpoint"],
        validation=metrics,
    ), path)


def train_group(group: str, out: Path, initial_path: Path, old_pool: list[dict],
                real_pool: list[dict], dev_pool: list[dict], steps_per_phase: int,
                chunk_size: int, seed: int, resume_complete: bool = False) -> dict:
    import gc

    if out.exists() and any(out.iterdir()):
        summary_path = out / "group_summary.json"
        checkpoint_path = out / "selected.pt"
        if not resume_complete or not summary_path.exists() or not checkpoint_path.exists():
            raise FileExistsError(f"Output directory is not a completed group: {out}")
        summary = json.loads(summary_path.read_text())
        plan = json.loads((out / "training_plan.json").read_text())
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        expected_step = int(initial.get("step", 0)) + 3 * steps_per_phase
        expected_exposure = int(initial.get("trained_feature_rows", 0)) + 3 * steps_per_phase * chunk_size
        if (summary.get("status") != "complete" or summary.get("group") != group
                or plan["initial_checkpoint"] != str(initial_path) or plan["seed"] != seed
                or plan["steps_per_phase"] != steps_per_phase or plan["features_per_update"] != chunk_size
                or summary["final_step"] != expected_step or state["step"] != expected_step
                or state["trained_feature_rows"] != expected_exposure
                or state["adaptation_group"] != group or state["model_config"] != initial["model_config"]):
            raise ValueError(f"Completed group does not match requested training: {out}")
        model = model_from(state)
        del model, state, initial
        torch.cuda.empty_cache()
        save(out / "progress.json", summary)
        print(json.dumps(dict(event="reuse_complete_group", **summary)), flush=True)
        return summary
    out.mkdir(parents=True, exist_ok=True)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    model = model_from(initial)
    model.train()
    phase_plan = [
        dict(name="full_ce", steps=steps_per_phase, lr=3e-5, tv_weight=0.0),
        dict(name="low_lr_ce", steps=steps_per_phase, lr=1e-5, tv_weight=0.0),
        dict(name="policy_tv", steps=steps_per_phase, lr=1e-5, tv_weight=2.0),
    ]
    mixture = ({"old_rollout": 1.0, "adaptation": 0.0}
               if group == "A_old_only" else {"old_rollout": 0.7, "adaptation": 0.3})
    plan = dict(
        group=group,
        initial_checkpoint=str(initial_path),
        start_step=int(initial.get("step", 0)),
        teacher="checkpoints/gpt_vfront.ckpt",
        feature_head=str(HEAD),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        phases=phase_plan,
        steps_per_phase=steps_per_phase,
        total_steps=3 * steps_per_phase,
        features_per_update=chunk_size,
        total_feature_exposure=3 * steps_per_phase * chunk_size,
        mixture=mixture,
        mixture_count_pattern=list(MIXED_NEW_COUNTS),
        old_data=pool_summary(old_pool),
        adaptation_data=pool_summary(real_pool),
        dev_data=pool_summary(dev_pool),
        seed=seed,
        optimizer="AdamW; weight decay 0.01; reinitialized at each phase",
        gradient_clip=1.0,
        train_precision="BF16 autocast",
        evaluation_precision="FP32",
        objective="Target raw T=1 Feature CE; final phase adds 2 times TV at T=0.9/top_p=0.9",
        evaluation="all 10 development scenes; full real-prefix Target distributions",
        model_selection="preselected final step 3600; step 1200 and 2400 retained for learning curves",
        test_used=False,
    )
    save(out / "training_plan.json", plan)
    history = []
    global_step = int(initial.get("step", 0))
    exposure = int(initial.get("trained_feature_rows", 0))
    started = time.monotonic()
    phase_index = 0
    for phase in phase_plan:
        phase_index += 1
        optimizer = torch.optim.AdamW(model.parameters(), lr=phase["lr"], weight_decay=0.01)
        for local_step in range(1, steps_per_phase + 1):
            step_seed = seed + (phase_index - 1) * 1_000_000 + local_step * 10_007
            old_rng = np.random.default_rng(step_seed)
            old_all = select_rows(old_pool, chunk_size, old_rng, balance_scenes=False)
            if group == "B_real_mixed":
                real_count = MIXED_NEW_COUNTS[(global_step - int(initial.get("step", 0))) % len(MIXED_NEW_COUNTS)]
                old_count = chunk_size - real_count
                old_batch = {key: value[:old_count] for key, value in old_all.items()}
                real_rng = np.random.default_rng(step_seed + 1)
                real_batch = select_rows(real_pool, real_count, real_rng, balance_scenes=True)
                batch = {key: torch.cat([old_batch[key], real_batch[key]], dim=0)
                         for key in FIELDS}
                order_rng = random.Random(step_seed + 2)
                order = list(range(chunk_size))
                order_rng.shuffle(order)
                order_tensor = torch.tensor(order, dtype=torch.long)
                batch = {key: value.index_select(0, order_tensor) for key, value in batch.items()}
            else:
                real_count = 0
                batch = old_all

            learning_rate = phase["lr"] * (
                0.1 + 0.9 * (1 + math.cos(math.pi * (local_step - 1) / steps_per_phase)) / 2
            )
            for group_values in optimizer.param_groups:
                group_values["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            total_loss = total_ce = total_tv = 0.0
            training_batch = {key: value.cuda() for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, logits = model(
                    training_batch["previous_state"],
                    training_batch["previous_feature"],
                    training_batch["position"],
                )
                losses, ce, tv = objective(
                    logits, training_batch["target"], phase["tv_weight"]
                )
                loss = losses.mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite adaptation loss")
            loss.backward()
            total_loss = float(loss.detach())
            total_ce = float(ce.detach().mean())
            total_tv = float(tv.detach().mean())
            del training_batch, logits, losses, ce, tv, loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise ValueError("Nonfinite adaptation gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            exposure += chunk_size
            if local_step == 1 or local_step % 50 == 0:
                progress = dict(
                    status="training",
                    group=group,
                    phase=phase["name"],
                    step=global_step,
                    phase_step=local_step,
                    loss=total_loss,
                    ce=total_ce,
                    tv=total_tv,
                    real_rows=real_count,
                    learning_rate=learning_rate,
                    trained_feature_rows=exposure,
                    elapsed_seconds=time.monotonic() - started,
                )
                save(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)

        del optimizer
        model.eval()
        from draft_a.learning_eval import _fp32_runtime
        with _fp32_runtime(torch):
            metrics = evaluate(model, dev_pool)
        checkpoint_step = global_step
        ckpt_path = out / f"step_{checkpoint_step:05d}.pt"
        phase_state = dict(plan, phase=phase["name"])
        save_checkpoint(ckpt_path, model, initial, global_step, exposure,
                        group, phase_state, metrics)
        row = dict(
            group=group,
            phase=phase["name"],
            step=global_step,
            trained_feature_rows=exposure,
            elapsed_seconds=time.monotonic() - started,
            learning_rate=phase["lr"],
            dev_pooled_overlap=metrics["pooled_overlap"],
            dev_equal_scene_overlap=metrics["equal_scene_overlap"],
            dev_raw_ce=metrics["pooled_raw_ce"],
            checkpoint=str(ckpt_path),
        )
        history.append(row)
        save(out / "history.json", history)
        save(out / f"step_{checkpoint_step:05d}_dev_metrics.json", metrics)
        print(json.dumps(dict(event="phase_end", **row)), flush=True)
        model.train()

    final_path = out / "selected.pt"
    final_metrics = json.loads((out / f"step_{global_step:05d}_dev_metrics.json").read_text())
    summary = dict(
        status="complete",
        group=group,
        final_step=global_step,
        trained_feature_rows=exposure,
        checkpoint=str(final_path),
        final_dev_pooled_overlap=final_metrics["pooled_overlap"],
        final_dev_equal_scene_overlap=final_metrics["equal_scene_overlap"],
        final_dev_raw_ce=final_metrics["pooled_raw_ce"],
        elapsed_seconds=time.monotonic() - started,
    )
    # The predetermined final checkpoint is copied to a stable group path.
    torch.save(torch.load(out / f"step_{global_step:05d}.pt", map_location="cpu", weights_only=True), final_path)
    save(out / "group_summary.json", summary)
    save(out / "progress.json", summary)
    del model, initial
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-data-plan", type=Path, required=True)
    parser.add_argument("--label-plan", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--steps-per-phase", type=int, default=1200)
    parser.add_argument("--features-per-update", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42001)
    parser.add_argument("--resume-complete-groups", action="store_true")
    args = parser.parse_args()
    if args.steps_per_phase <= 0 or args.features_per_update != 2048:
        raise ValueError("This plan uses positive updates with exactly 2048 Feature rows each")
    if args.out_root.exists() and any(args.out_root.iterdir()) and not args.resume_complete_groups:
        raise FileExistsError(f"Output root is not empty: {args.out_root}")
    old_pool = load_old_pool(args.old_data_plan)
    real_pool = load_label_pool(args.label_plan, "adaptation")
    dev_pool = load_label_pool(args.label_plan, "development")
    if len({row["scene"] for row in real_pool}) < 10 or len(dev_pool) == 0:
        raise ValueError("The adaptation pool or held-out development data is too small")
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    args.out_root.mkdir(parents=True, exist_ok=True)
    pair_plan = dict(
        status="running",
        initial_checkpoint=str(args.initial_checkpoint),
        seed=args.seed,
        steps_per_group=3 * args.steps_per_phase,
        steps_per_phase=args.steps_per_phase,
        features_per_update=args.features_per_update,
        feature_exposure_per_group=3 * args.steps_per_phase * args.features_per_update,
        groups={
            "A_old_only": str(args.out_root / "A_old_only"),
            "B_real_mixed": str(args.out_root / "B_real_mixed"),
        },
        comparison="same initialization, update count, effective Feature rows, and seed",
        mixture_pattern={"old_rollout_rows": "2048 - real_rows", "real_rows_per_5_updates": list(MIXED_NEW_COUNTS)},
        real_scene_sampling="uniform scene, then uniform trajectory within scene, then uniform Feature row",
        old_sampling="uniform trajectory, then uniform Feature row",
        optimizer="AdamW; weight decay 0.01; gradient clip 1.0",
        phases=[
            dict(name="full_ce", updates=args.steps_per_phase, lr=3e-5, tv_weight=0.0),
            dict(name="low_lr_ce", updates=args.steps_per_phase, lr=1e-5, tv_weight=0.0),
            dict(name="policy_tv", updates=args.steps_per_phase, lr=1e-5, tv_weight=2.0),
        ],
        train_precision="BF16 autocast",
        evaluation_precision="FP32",
        heldout_development_scenes=len({row["scene"] for row in dev_pool}),
        test_scenes_used=False,
    )
    pair_plan_path = args.out_root / "training_pair_plan.json"
    if args.resume_complete_groups and pair_plan_path.exists():
        if json.loads(pair_plan_path.read_text()) != pair_plan:
            raise ValueError("Existing pair settings differ from requested training")
    save(pair_plan_path, pair_plan)
    from draft_a.learning_eval import _fp32_runtime
    with _fp32_runtime(torch):
        initial_model = model_from(torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True))
        initial_dev = evaluate(initial_model, dev_pool)
        save(args.out_root / "initial_dev_metrics.json", initial_dev)
        del initial_model
        torch.cuda.empty_cache()
    summaries = []
    for group in ("A_old_only", "B_real_mixed"):
        summary = train_group(
            group,
            Path(pair_plan["groups"][group]),
            args.initial_checkpoint,
            old_pool,
            real_pool,
            dev_pool,
            args.steps_per_phase,
            args.features_per_update,
            args.seed,
            args.resume_complete_groups,
        )
        summaries.append(summary)
        save(args.out_root / "pair_progress.json", dict(
            status="running",
            completed_groups=len(summaries),
            expected_groups=2,
            summaries=summaries,
        ))
    result = dict(
        status="complete",
        initial_dev_pooled_overlap=initial_dev["pooled_overlap"],
        initial_dev_equal_scene_overlap=initial_dev["equal_scene_overlap"],
        groups=summaries,
        mixed_minus_old_only_dev_overlap=(
            summaries[1]["final_dev_equal_scene_overlap"]
            - summaries[0]["final_dev_equal_scene_overlap"]
        ),
    )
    save(args.out_root / "pair_summary.json", result)
    save(args.out_root / "pair_progress.json", dict(status="complete", summaries=summaries))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
