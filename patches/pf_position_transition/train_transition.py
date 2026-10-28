#!/usr/bin/env python3
"""Train the four fixed Direct/TwoStep configurations under one shared deadline."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transition_model import (
    EOS_ID,
    FEATURE_VOCAB,
    POSITION_VOCAB,
    TEMPERATURE,
    TOP_P,
    TransitionModel,
    config_parameter_table,
    fit_hidden_map,
    parameter_table,
    policy_batch,
    standard_library_self_test,
    torch_cpu_self_test,
)


ARRAY_FIELDS = (
    "h0", "hF", "hP", "previous_feature_id", "previous_position_id",
    "candidate_position_id", "old_logits_feature", "target_logits_feature",
    "p_f_policy", "q_old", "p_p1", "p1_accepted",
)
G2_MAX_SECONDS = 2400.0


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for model training") from exc
    return torch


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_torch_save(torch, path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _string_rows(value, size: int, name: str) -> list[str]:
    if isinstance(value, str):
        raise ValueError(f"{name} must have one value per row")
    result = [str(item) for item in value]
    if len(result) != size:
        raise ValueError(f"{name} row count mismatch")
    return result


def load_teacher(torch, path: Path) -> tuple[dict, dict]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("teacher.pt must contain a tensor dictionary")
    missing = [key for key in ARRAY_FIELDS if key not in raw]
    if missing:
        raise ValueError(f"teacher.pt is missing fields: {missing}")
    data = {key: raw[key] for key in ARRAY_FIELDS}
    metadata = raw.get("row_metadata")
    if "source_id" in raw and "split" in raw:
        data["source_id"] = raw["source_id"]
        data["split"] = raw["split"]
        data["scene_id"] = raw.get("scene_id", raw["source_id"])
        data["row_id"] = raw.get("row_id", list(range(len(raw["source_id"]))))
    elif isinstance(metadata, list) and metadata and all(isinstance(row, dict) for row in metadata):
        data["source_id"] = [row["source_id"] for row in metadata]
        data["split"] = [row["split"] for row in metadata]
        data["scene_id"] = [row.get("scene_id", row["source_id"]) for row in metadata]
        data["row_id"] = [row.get("row_id", index) for index, row in enumerate(metadata)]
    else:
        raise ValueError("teacher.pt needs row_metadata or top-level source_id/split arrays")
    n = len(data["source_id"])
    if n == 0:
        raise ValueError("teacher.pt has no rows")
    for key in ARRAY_FIELDS:
        tensor = torch.as_tensor(data[key])
        if tensor.ndim == 0 or tensor.size(0) != n:
            raise ValueError(f"{key} must have {n} rows")
        data[key] = tensor.detach().cpu()
    if data["h0"].ndim != 2 or data["hF"].shape != data["h0"].shape or data["hP"].shape != data["h0"].shape:
        raise ValueError("h0/hF/hP must share shape [N,H]")
    for key in ("old_logits_feature", "target_logits_feature", "p_f_policy", "q_old"):
        if data[key].shape != (n, FEATURE_VOCAB):
            raise ValueError(f"{key} must have shape [N,{FEATURE_VOCAB}]")
    for key in ("previous_feature_id", "previous_position_id", "candidate_position_id", "p_p1", "p1_accepted"):
        if data[key].numel() != n:
            raise ValueError(f"{key} must have one value per row")
        data[key] = data[key].reshape(n)
    data["source_id"] = _string_rows(data["source_id"], n, "source_id")
    data["scene_id"] = _string_rows(data["scene_id"], n, "scene_id")
    data["split"] = _string_rows(data["split"], n, "split")
    data["row_id"] = list(data["row_id"])
    if len(data["row_id"]) != n:
        raise ValueError("row_id count mismatch")

    split_by_source: dict[str, str] = {}
    for source, split in zip(data["source_id"], data["split"]):
        if split not in {"train", "val", "dev", "confirmation"}:
            raise ValueError(f"Unknown split {split!r}")
        previous = split_by_source.setdefault(source, split)
        if previous != split:
            raise ValueError(f"Source appears in multiple splits: {source}")
    if len(set(data["scene_id"])) != len(set(data["source_id"])):
        source_scenes: dict[str, set[str]] = {}
        scene_sources: dict[str, set[str]] = {}
        for source, scene in zip(data["source_id"], data["scene_id"]):
            source_scenes.setdefault(source, set()).add(scene)
            scene_sources.setdefault(scene, set()).add(source)
        if any(len(scenes) != 1 for scenes in source_scenes.values()) or any(len(sources) != 1 for sources in scene_sources.values()):
            raise ValueError("source_id and scene_id do not form a one-to-one grouping")

    fids = data["previous_feature_id"].long()
    pids = data["previous_position_id"].long()
    cids = data["candidate_position_id"].long()
    if bool(((fids < POSITION_VOCAB) | (fids >= EOS_ID)).any()):
        raise ValueError("previous_feature_id must use global Feature IDs")
    if bool(((pids < 0) | (pids >= POSITION_VOCAB) | (cids < 0) | (cids >= POSITION_VOCAB)).any()):
        raise ValueError("Position IDs must be in [0,8000)")
    for key in ("p_f_policy", "q_old"):
        probs = data[key].float()
        if not torch.isfinite(probs).all() or bool((probs < 0).any()):
            raise ValueError(f"{key} contains invalid probabilities")
        torch.testing.assert_close(probs.sum(-1), torch.ones(n), atol=2e-3, rtol=0)
    weights = data["p_p1"].float()
    if not torch.isfinite(weights).all() or bool(((weights < 0) | (weights > 1)).any()):
        raise ValueError("p_p1 must be finite probabilities")
    if torch.isnan(data["target_logits_feature"].float()).any():
        raise ValueError("target_logits_feature contains NaN")
    old_replay = policy_batch(data["old_logits_feature"].float(), TEMPERATURE, TOP_P)
    target_replay = policy_batch(data["target_logits_feature"].float(), TEMPERATURE, TOP_P)
    torch.testing.assert_close(old_replay, data["q_old"].float(), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(target_replay, data["p_f_policy"].float(), atol=2e-3, rtol=2e-3)
    summary = {
        "rows": n,
        "source_counts": {split: len({s for s, sp in zip(data["source_id"], data["split"]) if sp == split})
                          for split in ("train", "val", "dev", "confirmation")},
        "row_counts": {split: data["split"].count(split) for split in ("train", "val", "dev", "confirmation")},
        "hidden_width": int(data["h0"].size(1)),
        "rejected_p1_rows_kept": int((~data["p1_accepted"].bool()).sum().item()),
    }
    return data, summary


def load_token_table(torch, path: Path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(value):
        raise ValueError("token_embeddings.pt must be the original Target wte Tensor")
    if value.ndim != 2 or value.size(0) < EOS_ID or value.size(1) < 64:
        raise ValueError(f"Target wte must have shape at least [{EOS_ID},64]")
    return value.detach().float().cpu()


def _row_batch(torch, data: dict, indices, device: str) -> dict:
    indices = torch.as_tensor(indices, dtype=torch.long, device="cpu")
    keys = (
        "h0", "hF", "hP", "previous_feature_id", "previous_position_id",
        "candidate_position_id", "old_logits_feature", "target_logits_feature",
        "p_f_policy", "p_p1",
    )
    return {key: data[key].index_select(0, indices).to(device=device) for key in keys}


def _loss(torch, model, batch: dict):
    import torch.nn.functional as functional

    predicted, aux = model(
        batch["old_logits_feature"], batch["h0"], batch["previous_feature_id"],
        batch["previous_position_id"], batch["candidate_position_id"], return_aux=True,
    )
    target_logits = batch["target_logits_feature"].float() / TEMPERATURE
    allowed = torch.isfinite(target_logits)
    target_log = torch.log_softmax(target_logits, dim=-1).masked_fill(~allowed, 0.0)
    target_prob = torch.softmax(target_logits, dim=-1).masked_fill(~allowed, 0.0)
    student_logits = (predicted.float() / TEMPERATURE).masked_fill(~allowed, float("-inf"))
    student_log = torch.log_softmax(student_logits, dim=-1).masked_fill(~allowed, 0.0)
    kl = (target_prob * (target_log - student_log)).sum(dim=-1)
    kl_loss = kl.mean()
    if model.method == "two_step":
        target_f = model.hidden_state(batch["hF"].float()).detach()
        target_p = model.hidden_state(batch["hP"].float()).detach()
        f_loss = functional.smooth_l1_loss(aux["uF"], target_f, reduction="none").mean(dim=-1)
        p_loss = functional.smooth_l1_loss(aux["uP"], target_p, reduction="none").mean(dim=-1)
        f_loss = f_loss.mean()
        p_loss = p_loss.mean()
    else:
        f_loss = kl_loss.new_zeros(())
        p_loss = kl_loss.new_zeros(())
    residual_loss = aux["delta"].square().mean(dim=-1).mean()
    total = kl_loss + 0.05 * f_loss + 0.05 * p_loss + 1e-4 * residual_loss
    return total, {
        "kl": float(kl_loss.detach().cpu()),
        "state_f": float(f_loss.detach().cpu()),
        "state_p": float(p_loss.detach().cpu()),
        "residual_l2": float(residual_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def _validation(torch, model, data: dict, val_indices, device: str, batch_size: int = 128) -> dict:
    model.eval()
    source_rows: dict[str, list[tuple[float, float]]] = {}
    with torch.inference_mode():
        for start in range(0, len(val_indices), batch_size):
            ids = torch.tensor(val_indices[start:start + batch_size], dtype=torch.long)
            batch = _row_batch(torch, data, ids, device)
            logits = model(
                batch["old_logits_feature"], batch["h0"], batch["previous_feature_id"],
                batch["previous_position_id"], batch["candidate_position_id"],
            )
            probs = policy_batch(logits, TEMPERATURE, TOP_P)
            teacher = batch["p_f_policy"].float()
            overlap = torch.minimum(teacher, probs).sum(dim=-1).cpu().tolist()
            weights = batch["p_p1"].float().cpu().tolist()
            for local, global_id in enumerate(ids.tolist()):
                source = data["source_id"][global_id]
                source_rows.setdefault(source, []).append((float(overlap[local]), float(weights[local])))
    per_source = {}
    for source, values in source_rows.items():
        numerator = sum(overlap * weight for overlap, weight in values)
        denominator = sum(weight for _, weight in values)
        if denominator <= 0:
            raise ValueError(f"Validation source has zero total P1 probability: {source}")
        per_source[source] = {
            "p1_weighted_overlap": numerator / denominator,
            "mean_overlap": sum(overlap for overlap, _ in values) / len(values),
            "rows": len(values),
        }
    if not per_source:
        raise ValueError("No validation rows found")
    model.train()
    return {
        "source_count": len(per_source),
        "source_equal_p1_weighted_overlap": sum(v["p1_weighted_overlap"] for v in per_source.values()) / len(per_source),
        "source_equal_overlap": sum(v["mean_overlap"] for v in per_source.values()) / len(per_source),
        "per_source": per_source,
    }


def _save_run(torch, model, output: Path, method: str, size: str, seed: int, step: int,
              status: str, best: dict, deadline_unix: float, loss_row: dict | None):
    output.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "pf_position_transition_v1",
        "method": method,
        "size": size,
        "seed": int(seed),
        "update": int(step),
        "status": status,
        "deadline_unix": float(deadline_unix),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "best_validation": best,
        "latest_loss": loss_row,
        "trainable_parameters": model.trainable_parameter_count(),
    }
    _atomic_torch_save(torch, output / "last.pt", state)
    _atomic_json(output / "progress.json", {
        "method": method, "size": size, "seed": seed, "update": step,
        "status": status, "deadline_unix": deadline_unix,
        "best_validation": best, "latest_loss": loss_row,
    })


def train_one(torch, data, token_table, output: Path, method: str, size: str, seed: int,
              updates: int, batch_size: int, learning_rate: float, device: str,
              deadline_unix: float, reserve_seconds: float = 20.0) -> dict:
    train_indices = [i for i, split in enumerate(data["split"]) if split == "train"]
    val_indices = [i for i, split in enumerate(data["split"]) if split == "val"]
    if not train_indices or not val_indices:
        raise ValueError("teacher.pt must contain train and val rows")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(seed)
    train_ids = torch.tensor(train_indices, dtype=torch.long)
    train_hidden = torch.cat((data["h0"][train_ids], data["hF"][train_ids], data["hP"][train_ids]), dim=0)
    width = int(config_parameter_table(size)["d"])
    hidden_map = fit_hidden_map(train_hidden, width=width)
    model = TransitionModel(method, token_table, hidden_map, size=size).to(device)
    if model.trainable_parameter_count() != config_parameter_table(size)[f"{method}_trainable"]:
        raise RuntimeError("Actual trainable parameter count differs from the printed table")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best = {"score": float("-inf"), "update": 0, "validation": None}
    latest_loss = None
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "progress.json", {
        "method": method, "size": size, "seed": seed, "update": 0,
        "status": "running", "deadline_unix": deadline_unix,
    })
    training_log = output / "training.jsonl"
    validation_log = output / "validation.jsonl"
    status = "complete"
    completed = 0
    for step in range(1, updates + 1):
        if time.time() >= deadline_unix - reserve_seconds:
            status = "partial"
            break
        model.train()
        chosen = torch.randint(len(train_indices), (batch_size,), generator=generator)
        batch_ids = train_ids.index_select(0, chosen)
        batch = _row_batch(torch, data, batch_ids, device)
        optimizer.zero_grad(set_to_none=True)
        total, latest_loss = _loss(torch, model, batch)
        if not torch.isfinite(total):
            raise RuntimeError(f"Non-finite loss at update {step}")
        total.backward()
        optimizer.step()
        completed = step
        _append_jsonl(training_log, {
            "method": method, "size": size, "seed": seed, "update": step,
            "time_unix": time.time(), "loss": latest_loss,
        })

        if step % 100 == 0 or step == updates:
            if time.time() >= deadline_unix - reserve_seconds:
                status = "partial"
                break
            validation = _validation(torch, model, data, val_indices, device)
            score = float(validation["source_equal_overlap"])
            _append_jsonl(validation_log, {
                "method": method, "size": size, "seed": seed, "update": step,
                "time_unix": time.time(), "selection_score": score,
                "validation": validation,
            })
            if score > best["score"]:
                best = {"score": score, "update": step, "validation": validation}
                _save_run(torch, model, output, method, size, seed, step, "complete", best, deadline_unix, latest_loss)
                best_path = output / "best.pt"
                source = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
                _atomic_torch_save(torch, best_path, source)
            _save_run(torch, model, output, method, size, seed, step, "running", best, deadline_unix, latest_loss)

    if time.time() >= deadline_unix:
        status = "partial"
    if completed == updates and status == "complete":
        final_validation = _validation(torch, model, data, val_indices, device)
        score = float(final_validation["source_equal_overlap"])
        if score > best["score"]:
            best = {"score": score, "update": completed, "validation": final_validation}
            source = {
                "format": "pf_position_transition_v1", "method": method, "size": size,
                "seed": int(seed), "update": int(completed), "status": "complete",
                "deadline_unix": float(deadline_unix), "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "best_validation": best, "latest_loss": latest_loss,
                "trainable_parameters": model.trainable_parameter_count(),
            }
            _atomic_torch_save(torch, output / "best.pt", source)
    if time.time() > deadline_unix and status == "complete":
        status = "budget_overrun"
    _save_run(torch, model, output, method, size, seed, completed, status, best, deadline_unix, latest_loss)
    return {"method": method, "size": size, "seed": seed, "update": completed, "status": status, "best_validation": best}


def run_g2(torch, teacher_path: Path, token_path: Path, output: Path, size: str,
           updates: int, batch_size: int, learning_rate: float, device: str,
           deadline_unix: float) -> dict:
    started = time.time()
    if started >= deadline_unix:
        raise TimeoutError("Shared training deadline has already passed")
    if deadline_unix - started > G2_MAX_SECONDS:
        raise ValueError("G2 absolute deadline must be within 2400 seconds of this shared start")
    output.mkdir(parents=True, exist_ok=True)
    data, data_summary = load_teacher(torch, teacher_path)
    token_table = load_token_table(torch, token_path)
    manifest = {
        "status": "running", "started_unix": started, "deadline_unix": deadline_unix,
        "remaining_seconds": max(0.0, deadline_unix - time.time()),
        "teacher_path": str(teacher_path), "token_embeddings_path": str(token_path),
        "data": data_summary, "size": size, "updates_per_run_max": updates,
        "batch_size": batch_size, "learning_rate": learning_rate, "device": device,
        "runs": [],
    }
    _atomic_json(output / "pipeline.json", manifest)
    results = []
    for method, seed in (("direct", 17), ("direct", 29), ("two_step", 17), ("two_step", 29)):
        if time.time() >= deadline_unix:
            results.append({"method": method, "seed": seed, "status": "not_started"})
            continue
        run_output = output / f"{method}_seed{seed}"
        try:
            result = train_one(
                torch, data, token_table, run_output, method, size, seed, updates,
                batch_size, learning_rate, device, deadline_unix,
            )
        except Exception as exc:
            result = {"method": method, "seed": seed, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            manifest["runs"] = results
            manifest["status"] = "failed"
            manifest["remaining_seconds"] = max(0.0, deadline_unix - time.time())
            _atomic_json(output / "pipeline.json", manifest)
            raise
        results.append(result)
        manifest["runs"] = results
        manifest["remaining_seconds"] = max(0.0, deadline_unix - time.time())
        _atomic_json(output / "pipeline.json", manifest)
        if result["status"] in {"partial", "budget_overrun"}:
            for next_method, next_seed in (("direct", 17), ("direct", 29), ("two_step", 17), ("two_step", 29)):
                if (next_method, next_seed) not in {(row.get("method"), row.get("seed")) for row in results}:
                    results.append({"method": next_method, "seed": next_seed, "status": "not_started"})
            break
    manifest["runs"] = results
    manifest["remaining_seconds"] = max(0.0, deadline_unix - time.time())
    manifest["finished_unix"] = time.time()
    if time.time() > deadline_unix:
        manifest["status"] = "budget_overrun"
    else:
        manifest["status"] = "complete" if len(results) == 4 and all(row.get("status") == "complete" for row in results) else "partial"
    _atomic_json(output / "pipeline.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-parameters", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-g2", action="store_true", help="Run Direct/TwoStep with seeds 17 and 29")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--token-embeddings", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--size", choices=("main", "narrow"), default="main")
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deadline-unix", type=float)
    args = parser.parse_args(argv)

    if args.show_parameters:
        print(json.dumps(parameter_table(), indent=2))
        return 0
    if args.self_test:
        standard = standard_library_self_test()
        result = {"stdlib": "passed", "parameters": standard}
        try:
            result["torch_cpu"] = torch_cpu_self_test()
        except RuntimeError as exc:
            if "PyTorch is required" not in str(exc):
                raise
            result["torch_cpu"] = "not available in this Python environment"
        print(json.dumps(result, indent=2))
        return 0
    if not args.run_g2:
        parser.error("Select --show-parameters, --self-test, or --run-g2")
    if not all((args.teacher, args.token_embeddings, args.out, args.deadline_unix is not None)):
        parser.error("--run-g2 needs --teacher, --token-embeddings, --out, and one shared --deadline-unix")
    if not args.teacher.is_file() or not args.token_embeddings.is_file():
        parser.error("Teacher and token embedding files must exist")
    if not 1 <= args.updates <= 1200 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("Invalid training limits")
    torch = _torch()
    run_g2(torch, args.teacher, args.token_embeddings, args.out, args.size,
           args.updates, args.batch_size, args.learning_rate, args.device, args.deadline_unix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
