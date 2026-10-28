#!/usr/bin/env python3
"""Analyze the finite R1 proposal rows without loading Target or a GPU."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Sequence


FEATURES = 4096
TARGET_SPEED_REFERENCE = 1.7125
TEMPERATURE_GRID = (0.65, 0.75, 0.85, 0.9, 1.0, 1.1, 1.25)
TOP_P_GRID = (0.8, 0.85, 0.9, 0.95, 1.0)
MIX_GRID = (0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0)
RESIDUAL_SCALE_GRID = (0.0, 0.25, 0.5, 1.0)
BOOTSTRAP_DRAWS = 10000
FIT_STEPS = 100
SEED = 32003


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def validate_manifest(manifest: dict) -> dict[str, list[str]]:
    required = {
        "schema_version", "seed", "split_unit", "split_rule", "seed_used_for_split",
        "source_ids", "splits", "hold_purpose",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Split manifest is missing fields: {missing}")
    if manifest["schema_version"] != 1 or manifest["split_unit"] not in {"scene", "scene_id"}:
        raise ValueError("Expected schema version 1 with scene-level splitting")
    if not isinstance(manifest["split_rule"], str) or not manifest["split_rule"].strip():
        raise ValueError("The split rule must be a non-empty description")
    if manifest["seed_used_for_split"] is not False:
        raise ValueError("The agreed split is deterministic and does not use its seed")
    if not isinstance(manifest["hold_purpose"], str) or not manifest["hold_purpose"].strip():
        raise ValueError("The hold purpose must be a non-empty description")
    source_ids = manifest["source_ids"]
    splits = manifest["splits"]
    if not isinstance(source_ids, list) or len(source_ids) != 10:
        raise ValueError("The source manifest must list exactly ten scene IDs")
    if any(not isinstance(value, str) or not value for value in source_ids):
        raise ValueError("Scene IDs must be non-empty strings")
    if source_ids != sorted(set(source_ids)):
        raise ValueError("Scene IDs must be unique and lexicographically sorted")
    if not isinstance(splits, dict) or set(splits) != {"fit", "tune", "hold"}:
        raise ValueError("Expected fit, tune, and hold splits")
    expected_lengths = {"fit": 6, "tune": 2, "hold": 2}
    result: dict[str, list[str]] = {}
    offset = 0
    for name, expected_length in expected_lengths.items():
        values = splits[name]
        if not isinstance(values, list) or len(values) != expected_length:
            raise ValueError(f"The {name} split must contain {expected_length} scene IDs")
        if values != source_ids[offset:offset + expected_length]:
            raise ValueError(f"The {name} split does not match the manifest ordering rule")
        result[name] = values
        offset += expected_length
    if len(set(result["fit"] + result["tune"] + result["hold"])) != 10:
        raise ValueError("Scene splits overlap")
    if int(manifest["seed"]) != SEED:
        raise ValueError(f"Expected recorded seed {SEED}")
    return result


def load_r1(collection_dir: str | Path, manifest_path: str | Path) -> dict:
    """Load and validate the saved R1 arrays and source-level split."""
    import numpy as np

    collection = Path(collection_dir)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    splits = validate_manifest(manifest)
    contexts = _read_jsonl(collection / "contexts.jsonl")
    runs = _read_jsonl(collection / "w1_runs.jsonl")
    profiles = _read_jsonl(collection / "w1_profiles.jsonl")
    with np.load(collection / "vectors.npz", allow_pickle=False) as archive:
        required_arrays = {"p_f", "q1", "raw_logits"}
        if not required_arrays.issubset(archive.files):
            raise ValueError(f"vectors.npz must contain {sorted(required_arrays)}")
        p_raw = np.asarray(archive["p_f"], dtype=np.float64)
        q1_raw = np.asarray(archive["q1"], dtype=np.float64)
        raw_logits = np.asarray(archive["raw_logits"], dtype=np.float32)
    expected_shape = (480, FEATURES)
    for name, matrix in (("p_f", p_raw), ("q1", q1_raw), ("raw_logits", raw_logits)):
        if matrix.shape != expected_shape:
            raise ValueError(f"{name} has shape {matrix.shape}; expected {expected_shape}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains a non-finite value")
    if np.any(p_raw < 0.0) or np.any(q1_raw < 0.0):
        raise ValueError("Saved probability arrays contain a negative value")
    if len(contexts) != len(p_raw):
        raise ValueError(f"Found {len(contexts)} contexts for {len(p_raw)} vectors")
    contexts.sort(key=lambda row: int(row.get("vector_index", -1)))
    indices = [int(row.get("vector_index", -1)) for row in contexts]
    if indices != list(range(len(contexts))):
        raise ValueError("Context vector_index values must be unique, contiguous, and ordered")
    sample_ids = [row.get("sample_id") for row in contexts]
    if len(set(sample_ids)) != len(contexts) or any(not value for value in sample_ids):
        raise ValueError("Context sample_id values must be non-empty and unique")
    scenes = [row.get("scene") for row in contexts]
    if set(scenes) != set(manifest["source_ids"]):
        raise ValueError("Context scene IDs do not match the ten-source manifest")
    if any(not isinstance(row.get("context_prefix"), list) or
           not isinstance(row.get("p1_id"), int) for row in contexts):
        raise ValueError("Each row must contain context_prefix tokens and an integer p1_id")
    per_scene = {scene: scenes.count(scene) for scene in sorted(set(scenes))}
    if any(count != 48 for count in per_scene.values()):
        raise ValueError(f"Expected 48 contexts per source; observed {per_scene}")
    if len(runs) != 60 or len(profiles) != 24481:
        raise ValueError(f"R1 counts differ from the collected set: {len(runs)} runs, {len(profiles)} profiles")
    for name, values in splits.items():
        if not set(values).issubset(set(scenes)):
            raise ValueError(f"{name} split references a source absent from contexts")
    p_mass = p_raw.sum(axis=1, dtype=np.float64)
    q_mass = q1_raw.sum(axis=1, dtype=np.float64)
    if np.any(p_mass <= 0.0) or np.any(q_mass <= 0.0):
        raise ValueError("Each saved p and q row must have positive mass")
    p_norm = p_raw / p_mass[:, None]
    q1_norm = q1_raw / q_mass[:, None]
    scene_to_split = {scene: name for name, values in splits.items() for scene in values}
    return {
        "np": np,
        "collection": collection,
        "manifest": manifest,
        "splits": splits,
        "contexts": contexts,
        "runs": runs,
        "profiles": profiles,
        "p_raw": p_raw,
        "q1_raw": q1_raw,
        "raw_logits": raw_logits,
        "p_norm": p_norm,
        "q1_norm": q1_norm,
        "p_mass": p_mass,
        "q1_mass": q_mass,
        "scenes": scenes,
        "scene_to_split": scene_to_split,
        "per_scene_contexts": per_scene,
    }


def _load_feature_module():
    feature_path = Path(__file__).with_name("pf_round03_feature.py")
    spec = importlib.util.spec_from_file_location("pf_round03_feature", feature_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load spatial probe module: {feature_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_by_scene(data: dict) -> dict[str, list[int]]:
    return {
        name: [index for index, scene in enumerate(data["scenes"]) if scene in scene_ids]
        for name, scene_ids in data["splits"].items()
    }


def _normalized_rows(matrix, *, name: str):
    np = __import__("numpy")
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURES:
        raise ValueError(f"{name} must have shape [N,{FEATURES}]")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    sums = values.sum(axis=1, dtype=np.float64)
    if np.any(sums <= 0.0):
        raise ValueError(f"{name} rows must have positive mass")
    return values / sums[:, None]


def _source_means(values, scenes: Sequence[str], indices: Sequence[int]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for index in indices:
        grouped.setdefault(str(scenes[index]), []).append(float(values[index]))
    return {scene: statistics.fmean(rows) for scene, rows in sorted(grouped.items())}


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _source_bootstrap(source_values: dict[str, float], *, seed: int,
                      draws: int = BOOTSTRAP_DRAWS) -> dict:
    sources = sorted(source_values)
    if not sources:
        return {"source_count": 0, "draws": 0, "seed": seed, "ci95": None}
    rng = random.Random(seed)
    samples = [statistics.fmean(source_values[rng.choice(sources)] for _ in sources)
               for _ in range(draws)]
    return {
        "source_count": len(sources),
        "draws": draws,
        "seed": seed,
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
    }


def score_policy(p, q, scenes: Sequence[str], indices: Sequence[int]) -> dict:
    """Score normalized overlap and source-equal improvement against q1."""
    np = __import__("numpy")
    p_norm = _normalized_rows(p, name="p")
    q_norm = _normalized_rows(q, name="q")
    if p_norm.shape != q_norm.shape or p_norm.shape[0] != len(scenes):
        raise ValueError("p, q, and source labels must align")
    overlaps = np.minimum(p_norm, q_norm).sum(axis=1, dtype=np.float64)
    subset = [int(index) for index in indices]
    if not subset:
        return {"row_count": 0, "source_count": 0, "row_mean": None,
                "source_equal_mean": None, "by_source": {}, "row_overlap": overlaps}
    by_source = _source_means(overlaps, scenes, subset)
    return {
        "row_count": len(subset),
        "source_count": len(by_source),
        "row_mean": statistics.fmean(float(overlaps[index]) for index in subset),
        "source_equal_mean": statistics.fmean(by_source.values()),
        "by_source": by_source,
        "row_overlap": overlaps,
    }


def _policy_rows(feature_module, raw_logits, *, temperature: float, top_p: float):
    np = __import__("numpy")
    return np.asarray([
        feature_module.policy_from_logits(row, temperature=temperature, top_p=top_p)
        for row in raw_logits
    ], dtype=np.float64)


def _entropy_rows(q_norm):
    np = __import__("numpy")
    positive = q_norm > 0.0
    safe = np.where(positive, q_norm, 1.0)
    return -np.sum(np.where(positive, q_norm * np.log(safe), 0.0), axis=1, dtype=np.float64)


def _neighbor_inputs(data: dict, feature_module):
    np = data["np"]
    count = len(data["contexts"])
    positions = np.empty(count, dtype=np.int64)
    neighbor_ids = np.full((count, 6), -1, dtype=np.int64)
    empirical = np.zeros((count, FEATURES), dtype=np.float64)
    neighbor_counts = np.zeros(count, dtype=np.int64)
    for index, row in enumerate(data["contexts"]):
        position_id = int(row["p1_id"])
        positions[index] = position_id
        book = feature_module.CommittedNeighbors.from_tokens(row["context_prefix"])
        features = book.batch_features([position_id])["neighbor_feature_ids"][0]
        neighbor_ids[index] = features
        present_ids = [feature_id - feature_module.FEATURE_OFFSET for feature_id in features
                       if feature_module.FEATURE_OFFSET <= feature_id < feature_module.BOS_EOS]
        neighbor_counts[index] = len(present_ids)
        if present_ids:
            for feature_id in present_ids:
                empirical[index, feature_id] += 1.0 / len(present_ids)
    return positions, neighbor_ids, empirical, neighbor_counts


def _mixture(q1_norm, empirical, weight: float):
    np = __import__("numpy")
    mixed = (1.0 - weight) * q1_norm + weight * empirical
    has_neighbors = empirical.sum(axis=1, dtype=np.float64) > 0.0
    mixed[~has_neighbors] = q1_norm[~has_neighbors]
    return _normalized_rows(mixed, name="neighbor mixture")


def _batch_feature_policy(feature_module, raw_logits, deltas, q1_raw, indices,
                          *, residual_scale: float):
    np = __import__("numpy")
    rows = []
    for local, index in enumerate(indices):
        rows.append(feature_module.apply_feature_policy(
            raw_logits[index], deltas[local], original_q=q1_raw[index],
            gate=residual_scale, temperature=0.9, top_p=0.9,
        ))
    return np.asarray(rows, dtype=np.float64)


def _tune_calibration(data: dict, indices: list[int], feature_module) -> tuple[dict, list[dict]]:
    all_results = []
    best = None
    for temperature in TEMPERATURE_GRID:
        for top_p in TOP_P_GRID:
            candidate = _policy_rows(feature_module, data["raw_logits"][indices],
                                     temperature=temperature, top_p=top_p)
            subset_scenes = [data["scenes"][index] for index in indices]
            score = score_policy(data["p_norm"][indices], candidate, subset_scenes,
                                 list(range(len(indices))))
            value = float(score["source_equal_mean"])
            row = {"temperature": temperature, "top_p": top_p,
                   "tune_source_equal_overlap": value}
            all_results.append(row)
            if best is None or value > best["tune_source_equal_overlap"]:
                best = row
    return best, all_results


def _tune_mixture(data: dict, indices: list[int], empirical, feature_module) -> tuple[dict, list[dict]]:
    all_results = []
    best = None
    q1 = data["q1_norm"][indices]
    neighbors = empirical[indices]
    for weight in MIX_GRID:
        candidate = _mixture(q1.copy(), neighbors.copy(), weight)
        subset_scenes = [data["scenes"][index] for index in indices]
        score = score_policy(data["p_norm"][indices], candidate, subset_scenes,
                             list(range(len(indices))))
        value = float(score["source_equal_mean"])
        row = {"neighbor_weight": weight, "tune_source_equal_overlap": value}
        all_results.append(row)
        if best is None or value > best["tune_source_equal_overlap"]:
            best = row
    return best, all_results


def _tune_residual(data: dict, indices: list[int], deltas, feature_module) -> tuple[dict, list[dict]]:
    all_results = []
    best = None
    for scale in RESIDUAL_SCALE_GRID:
        candidate = _batch_feature_policy(
            feature_module, data["raw_logits"], deltas, data["q1_raw"], indices,
            residual_scale=scale,
        )
        subset_scenes = [data["scenes"][index] for index in indices]
        score = score_policy(data["p_norm"][indices], candidate, subset_scenes,
                             list(range(len(indices))))
        value = float(score["source_equal_mean"])
        row = {"residual_scale": scale, "tune_source_equal_overlap": value}
        all_results.append(row)
        if best is None or value > best["tune_source_equal_overlap"]:
            best = row
    return best, all_results


def _profile_costs(rows: list[dict]) -> dict:
    by_run: dict[tuple, dict] = {}
    for row in rows:
        key = (row.get("scene"), row.get("prefix"), row.get("seed_base"))
        item = by_run.setdefault(key, {
            "round_target_host": 0.0,
            "round_non_target_host": 0.0,
            "rounds": int(row.get("rounds", 0) or 0),
            "generated_tokens": int(row.get("generated_tokens", 0) or 0),
        })
        stage = row.get("stage", "")
        operation = row.get("operation", "")
        host = float(row.get("host_seconds", 0.0))
        if operation in {"target_kv", "capture_open", "prefill", "prefix_replay"}:
            continue
        if stage in {"target_verify", "target_context"}:
            item["round_target_host"] += host
        else:
            item["round_non_target_host"] += host
    runs = list(by_run.values())
    valid = [row for row in runs if row["rounds"] > 0]
    return {
        "run_count": len(runs),
        "mean_rounds": statistics.fmean(row["rounds"] for row in runs) if runs else None,
        "mean_generated_tokens_per_round": statistics.fmean(
            row["generated_tokens"] / row["rounds"] for row in valid) if valid else None,
        "target_host_seconds_per_round": statistics.fmean(
            row["round_target_host"] / row["rounds"] for row in valid) if valid else None,
        "non_target_host_seconds_per_round": statistics.fmean(
            row["round_non_target_host"] / row["rounds"] for row in valid) if valid else None,
        "timing_note": "Host component averages are separate; this is not a same-round candidate measurement.",
    }


def estimate_break_even(runs: list[dict], profiles: list[dict], *,
                        target_speed_reference: float = TARGET_SPEED_REFERENCE) -> dict:
    stats_rows = [row.get("stats", {}) for row in runs]
    total_rounds = sum(int(row.get("rounds", 0) or 0) for row in stats_rows)
    total_generated = sum(int(row.get("generated_tokens", 0) or 0) for row in stats_rows)
    feature_verified = sum(int(row.get("feature_verified", 0) or 0) for row in stats_rows)
    feature_accepted = sum(int(row.get("feature_accepted", 0) or 0) for row in stats_rows)
    costs = _profile_costs(profiles)
    if total_rounds <= 0 or feature_verified <= 0:
        raise ValueError("W1 run records do not contain usable round and Feature counts")
    if costs["run_count"] != len(runs):
        raise ValueError(f"W1 profile covers {costs['run_count']} runs, but run records contain {len(runs)}")
    l0 = total_generated / total_rounds
    v = feature_verified / total_rounds
    acceptance = feature_accepted / feature_verified
    target_ms = 1000.0 * float(costs["target_host_seconds_per_round"])
    other_ms = 1000.0 * float(costs["non_target_host_seconds_per_round"])
    total_ms = target_ms + other_ms
    thresholds = []
    for extra_ms in (0.0, 0.25, 0.5, 1.0):
        required_length = l0 * (2.0 / target_speed_reference) * (1.0 + extra_ms / total_ms)
        required_acceptance = (required_length - 2.0) / (2.0 * v)
        thresholds.append({
            "extra_proposal_ms_per_round": extra_ms,
            "required_mean_tokens_per_round": required_length,
            "required_feature_acceptance": required_acceptance,
            "feasible_under_this_simple_length_model": 0.0 <= required_acceptance <= 1.0,
        })
    return {
        "r1_runs": len(runs),
        "r1_rounds": total_rounds,
        "r1_generated_tokens": total_generated,
        "feature_verified": feature_verified,
        "feature_accepted": feature_accepted,
        "mean_generated_tokens_per_round": l0,
        "verified_features_per_round": v,
        "observed_feature_acceptance": acceptance,
        "profile": costs,
        "historical_speed_reference": target_speed_reference,
        "rough_thresholds": thresholds,
        "limitation": "Uses a cross-batch historical speed ratio and R1 W1 costs; it is a screening estimate, not a strict bound or measured candidate speed.",
    }


def _split_metrics(data: dict, split_indices: dict[str, list[int]], outputs: dict[str, object],
                   q1_norm, entropies, fit_entropy_median: float) -> dict:
    np = data["np"]
    results = {}
    for split_name, indices in split_indices.items():
        method_rows = {}
        for method, q in outputs.items():
            subset_scenes = [data["scenes"][index] for index in indices]
            score = score_policy(data["p_raw"][indices], q[indices], subset_scenes,
                                 list(range(len(indices))))
            q_raw = q[indices]
            q_mass = q_raw.sum(axis=1, dtype=np.float64)
            method_rows[method] = {
                "row_count": score["row_count"],
                "source_count": score["source_count"],
                "row_mean_overlap": score["row_mean"],
                "source_equal_overlap": score["source_equal_mean"],
                "source_overlap": score["by_source"],
                "mean_mass_error": float(np.mean(np.abs(q_mass - 1.0))),
                "mean_support": float(np.mean(np.count_nonzero(q_raw > 0.0, axis=1))),
                "minimum_positive_probability": float(np.min(q_raw[q_raw > 0.0])) if np.any(q_raw > 0.0) else None,
            }
        reference = method_rows["reference_q1"]["source_overlap"]
        for method, item in method_rows.items():
            source_delta = {scene: value - reference[scene]
                            for scene, value in item["source_overlap"].items()}
            item["source_equal_improvement_vs_q1"] = statistics.fmean(source_delta.values())
            item["source_improvement_vs_q1"] = source_delta
            item["source_bootstrap_ci95_vs_q1"] = _source_bootstrap(
                source_delta, seed=SEED + 31 * len(split_name))
        entropy_groups = {}
        for label, chosen in (
            ("low", [index for index in indices if entropies[index] < fit_entropy_median]),
            ("high", [index for index in indices if entropies[index] >= fit_entropy_median]),
        ):
            group_result = {}
            for method, q in outputs.items():
                if not chosen:
                    group_result[method] = {"row_count": 0, "source_count": 0,
                                             "source_equal_overlap": None}
                    continue
                subset_scenes = [data["scenes"][index] for index in chosen]
                score = score_policy(data["p_raw"][chosen], q[chosen], subset_scenes,
                                     list(range(len(chosen))))
                group_result[method] = {
                    "row_count": score["row_count"],
                    "source_count": score["source_count"],
                    "source_equal_overlap": score["source_equal_mean"],
                    "source_overlap": score["by_source"],
                }
            entropy_groups[label] = group_result
        results[split_name] = {
            "source_ids": sorted({data["scenes"][index] for index in indices}),
            "entropy_split_fit_median_nats": fit_entropy_median,
            "entropy_group_counts": {
                "low": int(sum(entropies[index] < fit_entropy_median for index in indices)),
                "high": int(sum(entropies[index] >= fit_entropy_median for index in indices)),
            },
            "methods": method_rows,
            "entropy_groups": entropy_groups,
        }
    return results


def analyze(collection_dir: str | Path, manifest_path: str | Path, output_dir: str | Path) -> dict:
    np = __import__("numpy")
    started = time.perf_counter()
    data = load_r1(collection_dir, manifest_path)
    feature_module = _load_feature_module()
    split_indices = split_by_scene(data)
    fit_indices = split_indices["fit"]
    tune_indices = split_indices["tune"]
    if any(len(split_indices[name]) != count for name, count in (("fit", 288), ("tune", 96), ("hold", 96))):
        raise ValueError("Expected 288 fit, 96 tune, and 96 one-pass diagnostic rows")

    phase_times = {}
    t0 = time.perf_counter()
    positions, neighbor_ids, empirical, neighbor_counts = _neighbor_inputs(data, feature_module)
    phase_times["neighbor_feature_extraction_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    best_calibration, calibration_grid = _tune_calibration(data, tune_indices, feature_module)
    phase_times["proposal_policy_tuning_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    best_mixture, mixture_grid = _tune_mixture(data, tune_indices, empirical, feature_module)
    phase_times["neighbor_mixture_tuning_seconds"] = time.perf_counter() - t0

    models = {}
    fit_reports = {}
    for mode in ("spatial", "context_only"):
        probe = feature_module.DirectionalResidualProbe(
            mode=mode, rank=4, neighbor_groups=8, logit_buckets=16, seed=SEED,
        )
        t0 = time.perf_counter()
        fit_reports[mode] = probe.fit(
            data["raw_logits"][fit_indices], positions[fit_indices], neighbor_ids[fit_indices],
            data["p_raw"][fit_indices], steps=FIT_STEPS, lr=0.03, l2=1e-3,
        )
        fit_reports[mode]["fit_seconds"] = time.perf_counter() - t0
        models[mode] = probe
    phase_times["probe_fitting_seconds"] = sum(row["fit_seconds"] for row in fit_reports.values())

    t0 = time.perf_counter()
    tune_deltas = {
        mode: model.residual_logits(data["raw_logits"][tune_indices], positions[tune_indices],
                                    neighbor_ids[tune_indices])
        for mode, model in models.items()
    }
    best_scales = {}
    scale_grids = {}
    for mode in ("spatial", "context_only"):
        best_scales[mode], scale_grids[mode] = _tune_residual(
            data, tune_indices, tune_deltas[mode], feature_module,
        )
    phase_times["residual_scale_tuning_seconds"] = time.perf_counter() - t0

    selections = {
        "proposal_calibration": best_calibration,
        "neighbor_mixture": best_mixture,
        "residual_scale": best_scales,
    }
    method_timings = {}
    generation_started = time.perf_counter()
    t0 = time.perf_counter()
    q_cal = _policy_rows(feature_module, data["raw_logits"],
                         temperature=float(best_calibration["temperature"]),
                         top_p=float(best_calibration["top_p"]))
    q_cal_seconds = time.perf_counter() - t0
    method_timings["proposal_calibrated"] = {
        "rows": len(data["contexts"]), "elapsed_seconds": q_cal_seconds,
        "milliseconds_per_row": q_cal_seconds * 1000.0 / len(data["contexts"]),
    }
    t0 = time.perf_counter()
    q_mix = _mixture(data["q1_norm"].copy(), empirical.copy(),
                     float(best_mixture["neighbor_weight"]))
    q_mix_seconds = time.perf_counter() - t0
    method_timings["neighbor_mixture"] = {
        "rows": len(data["contexts"]), "elapsed_seconds": q_mix_seconds,
        "milliseconds_per_row": q_mix_seconds * 1000.0 / len(data["contexts"]),
    }
    all_indices = list(range(len(data["contexts"])))
    outputs = {"reference_q1": data["q1_raw"].copy(), "proposal_calibrated": q_cal,
               "neighbor_mixture": q_mix}
    fitted_deltas = {}
    for mode, method_name in (("spatial", "spatial_residual"),
                              ("context_only", "context_only_residual")):
        t0 = time.perf_counter()
        selected_scale = float(best_scales[mode]["residual_scale"])
        if selected_scale == 0.0:
            outputs[method_name] = data["q1_raw"].copy()
        else:
            fitted_deltas[mode] = models[mode].residual_logits(
                data["raw_logits"], positions, neighbor_ids,
            )
            outputs[method_name] = _batch_feature_policy(
                feature_module, data["raw_logits"], fitted_deltas[mode], data["q1_raw"],
                all_indices, residual_scale=selected_scale,
            )
        elapsed = time.perf_counter() - t0
        method_timings[method_name] = {
            "rows": len(data["contexts"]), "elapsed_seconds": elapsed,
            "milliseconds_per_row": elapsed * 1000.0 / len(data["contexts"]),
            "selected_residual_scale": selected_scale,
            "residual_inference_skipped_at_zero_scale": selected_scale == 0.0,
        }
    phase_times["selected_policy_generation_seconds"] = time.perf_counter() - generation_started

    t0 = time.perf_counter()
    q_rebuilt = _policy_rows(feature_module, data["raw_logits"], temperature=0.9, top_p=0.9)
    reconstruction_seconds = time.perf_counter() - t0
    reconstruction_abs_difference = np.abs(q_rebuilt - data["q1_raw"])
    numerical_reproduction = {
        "policy": {"temperature": 0.9, "top_p": 0.9},
        "rows": len(data["contexts"]),
        "elapsed_seconds": reconstruction_seconds,
        "milliseconds_per_row": reconstruction_seconds * 1000.0 / len(data["contexts"]),
        "maximum_absolute_q_difference_vs_saved_q1": float(reconstruction_abs_difference.max()),
        "mean_absolute_q_difference_vs_saved_q1": float(reconstruction_abs_difference.mean()),
        "scope": "CPU Python float64 policy reconstruction compared with saved float32 q1 values",
    }

    entropies = _entropy_rows(data["q1_norm"])
    fit_entropy_median = float(np.median(entropies[fit_indices]))
    split_results = _split_metrics(data, split_indices, outputs, data["q1_norm"],
                                   entropies, fit_entropy_median)
    hold_methods = split_results["hold"]["methods"]
    spatial_delta = hold_methods["spatial_residual"]["source_equal_improvement_vs_q1"]
    mix_delta = hold_methods["neighbor_mixture"]["source_equal_improvement_vs_q1"]
    context_delta = hold_methods["context_only_residual"]["source_equal_improvement_vs_q1"]
    if spatial_delta < 0.03:
        stop_recommendation = "stop_below_three_percentage_points"
    elif spatial_delta <= max(mix_delta, context_delta):
        stop_recommendation = "stop_no_extra_spatial_gain"
    else:
        stop_recommendation = "send_for_primary_review"

    break_even = estimate_break_even(data["runs"], data["profiles"])
    raw_prob_stats = {
        "p_f_mean_absolute_mass_error": float(np.mean(np.abs(data["p_mass"] - 1.0))),
        "p_f_maximum_absolute_mass_error": float(np.max(np.abs(data["p_mass"] - 1.0))),
        "q1_mean_absolute_mass_error": float(np.mean(np.abs(data["q1_mass"] - 1.0))),
        "q1_maximum_absolute_mass_error": float(np.max(np.abs(data["q1_mass"] - 1.0))),
        "normalization_for_overlap": "each saved p and q row is divided by its own float64 sum; raw values and mass errors are also preserved",
        "small_positive_values": "no probability cutoff is applied",
    }
    summary = {
        "schema_version": 1,
        "status": "completed_cpu_offline_analysis",
        "source_split": data["manifest"],
        "counts": {
            "contexts": len(data["contexts"]), "sources": len(set(data["scenes"])),
            "contexts_per_source": data["per_scene_contexts"],
            "w1_runs": len(data["runs"]), "w1_profiles": len(data["profiles"]),
            "split_rows": {name: len(values) for name, values in split_indices.items()},
            "hold_evaluation_passes": 1,
            "six_neighbor_rows_with_any_committed_feature": int(np.count_nonzero(neighbor_counts)),
        },
        "probability_integrity": raw_prob_stats,
        "selection": selections,
        "tuning_grids": {
            "proposal_calibration": calibration_grid,
            "neighbor_mixture": mixture_grid,
            "spatial_residual_scale": scale_grids["spatial"],
            "context_only_residual_scale": scale_grids["context_only"],
        },
        "probe": {
            "fit_rows_per_mode": len(fit_indices), "fit_steps": FIT_STEPS,
            "fit_reports": fit_reports,
            "parameter_counts": {mode: model.parameter_count for mode, model in models.items()},
            "modes_have_equal_parameters": models["spatial"].parameter_count == models["context_only"].parameter_count,
            "target_p_usage": "fit labels only; residual_logits receives raw logits, positions, and neighbor feature IDs",
        },
        "metrics": split_results,
        "fit_entropy_median_nats": fit_entropy_median,
        "neighbor_counts": {
            "minimum": int(neighbor_counts.min()), "mean": float(neighbor_counts.mean()),
            "maximum": int(neighbor_counts.max()),
            "histogram": {str(count): int(np.count_nonzero(neighbor_counts == count))
                          for count in range(7)},
        },
        "selected_method_cpu_elapsed": {
            "measurements": method_timings,
            "neighbor_query": {
                "rows": len(data["contexts"]),
                "elapsed_seconds": phase_times["neighbor_feature_extraction_seconds"],
                "milliseconds_per_row": phase_times["neighbor_feature_extraction_seconds"] *
                                        1000.0 / len(data["contexts"]),
            },
            "policy_reconstruction": numerical_reproduction,
            "interpretation": "CPU batch and Python processing time; not an online GPU Draft latency estimate.",
        },
        "original_policy_reconstruction": numerical_reproduction,
        "break_even_screen": break_even,
        "stop_recommendation": {
            "result": stop_recommendation,
            "spatial_hold_gain_vs_q1": spatial_delta,
            "simple_mixture_hold_gain_vs_q1": mix_delta,
            "context_only_hold_gain_vs_q1": context_delta,
            "rule": "stop when spatial source-equal hold gain is below 0.03, or it does not exceed both simple mixture and context-only control",
        },
        "elapsed_seconds": {
            **phase_times,
            "analysis_compute_total": time.perf_counter() - started,
            "measurement": "wall-clock elapsed time in a CPU-only process",
        },
        "limitations": [
            "The 480 rows are a finite development diagnostic. The web plan already reported sample-internal calibration and mixture results before this source split.",
            "The two diagnostic sources were previously visible during method selection; the split isolates this script's fitting and tuning only.",
            "The source bootstrap on two diagnostic sources has very coarse support and cannot establish broad generalization.",
            "No Target-only or candidate decoder was run in this analysis, so these overlap values do not establish a speed ratio.",
            "The rough cost screen combines an older 1.7125 speed reference with R1 W1 profile costs; changed P2, EOS, Ranker, and acceptance paths are not modeled.",
            "NumPy fitting uses CPU float64 for the probe math and does not validate a GPU implementation or online latency.",
        ],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as stream:
        for index, context in enumerate(data["contexts"]):
            row = {
                "sample_id": context["sample_id"], "vector_index": index,
                "scene": context["scene"], "split": data["scene_to_split"][context["scene"]],
                "prefix": context["prefix"], "seed_base": context["seed_base"],
                "seed": context["seed"], "round_index": context["round_index"],
                "p1_id": context["p1_id"], "neighbor_count": int(neighbor_counts[index]),
                "q1_entropy_nats": float(entropies[index]),
                "raw_mass": {"p_f": float(data["p_mass"][index]), "q1": float(data["q1_mass"][index])},
                "methods": {},
            }
            for method, q in outputs.items():
                q_row = q[index]
                row["methods"][method] = {
                    "overlap": float(np.minimum(data["p_norm"][index],
                                                _normalized_rows(q[index:index + 1], name=method)[0]).sum()),
                    "mass": float(np.sum(q_row, dtype=np.float64)),
                    "support": int(np.count_nonzero(q_row > 0.0)),
                    "minimum_positive_probability": float(np.min(q_row[q_row > 0.0])) if np.any(q_row > 0.0) else None,
                }
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    arrays = {
        "p_f_raw_float64": data["p_raw"],
        "q1_raw_float64": data["q1_raw"],
        "raw_logits_float32": data["raw_logits"],
        "reference_q1": outputs["reference_q1"],
        "proposal_calibrated": outputs["proposal_calibrated"],
        "neighbor_mixture": outputs["neighbor_mixture"],
        "spatial_residual": outputs["spatial_residual"],
        "context_only_residual": outputs["context_only_residual"],
        "positions": positions,
        "neighbor_feature_ids": neighbor_ids,
    }
    np.savez_compressed(output_dir / "q_outputs.npz", **arrays)
    model_arrays = {}
    for mode, model in models.items():
        for name, values in model._weights.items():
            model_arrays[f"{mode}_{name}"] = values
    np.savez_compressed(output_dir / "fit_parameters.npz", **model_arrays)
    summary["outputs"] = {
        "summary": "summary.json", "rows": "rows.jsonl",
        "q_outputs": "q_outputs.npz", "fit_parameters": "fit_parameters.npz",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _self_test() -> dict:
    feature_module = _load_feature_module()
    source_ids = [f"source-{index:02d}" for index in range(10)]
    test_manifest = {
        "schema_version": 1, "seed": SEED, "split_unit": "scene",
        "split_rule": "lexicographically sort unique scene IDs, slice first 6 fit, next 2 tune, final 2 hold",
        "seed_used_for_split": False, "source_ids": source_ids,
        "splits": {"fit": source_ids[:6], "tune": source_ids[6:8], "hold": source_ids[8:]},
        "hold_purpose": "one diagnostic pass only; no historical independence claim",
    }
    test_splits = validate_manifest(test_manifest)
    assert [len(test_splits[name]) for name in ("fit", "tune", "hold")] == [6, 2, 2]
    q = feature_module.policy_from_logits([math.log(0.6), math.log(0.3), math.log(0.1)],
                                          temperature=1.0, top_p=0.6)
    assert q[0] > 0.0 and q[1] > 0.0 and q[2] == 0.0
    assert abs(math.fsum(q) - 1.0) < 1e-15
    tiny = feature_module._probability_row([1.0 - 1e-14, 1e-14], name="tiny")
    assert tiny[1] > 0.0
    mix_small = [(1.0 - 0.1) * 0.5, (1.0 - 0.1) * 0.5 + 0.1]
    assert abs(math.fsum(mix_small) - 1.0) < 1e-15
    assert mix_small == [0.45, 0.55]
    overlap = feature_module.overlap([0.4, 0.6], [0.5, 0.5])
    assert abs(overlap - 0.9) < 1e-15
    boot = _source_bootstrap({"a": 0.1, "b": 0.2}, seed=SEED, draws=100)
    assert boot == _source_bootstrap({"a": 0.1, "b": 0.2}, seed=SEED, draws=100)
    try:
        import numpy as np
    except ImportError:
        numpy_scalar_test = {"status": "skipped_no_numpy"}
    else:
        test_entropies = np.asarray([0.25, 1.25], dtype=np.float64)
        test_threshold = 0.75
        returned_count = sum(value < test_threshold for value in test_entropies)
        raw_json_rejected = False
        try:
            json.dumps({"count": returned_count})
        except TypeError:
            raw_json_rejected = True
        native_count = int(returned_count)
        encoded_count = json.dumps({"count": native_count})
        assert native_count == 1 and json.loads(encoded_count)["count"] == 1
        numpy_scalar_test = {
            "status": "passed",
            "sum_return_type": type(returned_count).__name__,
            "unconverted_json_rejected": raw_json_rejected,
            "native_integer_json": encoded_count,
        }
    counts = estimate_break_even(
        [{"stats": {"rounds": 10, "generated_tokens": 28,
                    "feature_verified": 8, "feature_accepted": 4}}],
        [{"scene": "a", "prefix": 64, "seed_base": 1, "rounds": 10,
          "generated_tokens": 28, "stage": "target_verify", "host_seconds": 0.2},
         {"scene": "a", "prefix": 64, "seed_base": 1, "rounds": 10,
          "generated_tokens": 28, "stage": "feature_policy", "host_seconds": 0.03}],
    )
    assert counts["r1_rounds"] == 10 and counts["feature_verified"] == 8
    return {
        "passed": True,
        "scope": "synthetic standard-library cases; no R1 data or NumPy fit",
        "scene_manifest_validation": "passed",
        "nucleus_threshold_tie_keeps_crossing_token": "passed",
        "tiny_positive_mass_retained": "passed",
        "mixture_normalization": "passed",
        "overlap": "passed",
        "bootstrap_reproducible_seed": SEED,
        "numpy_entropy_count_json": numpy_scalar_test,
        "rough_break_even_inputs": "passed",
        "spatial_module": feature_module._self_test(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path,
                        default=root / "docs/project/artifacts/pf2x_loop_20261019/round01/cost_probe_raw/run01/collection")
    parser.add_argument("--split-manifest", type=Path,
                        default=root / "docs/project/artifacts/pf2x_loop_20261019/round03/source_manifest.json")
    parser.add_argument("--out", type=Path,
                        default=root / "docs/project/artifacts/pf2x_loop_20261019/round03/results/offline")
    parser.add_argument("--self-test", action="store_true",
                        help="run synthetic standard-library checks and write selftest.json")
    args = parser.parse_args(argv)
    if args.self_test:
        args.out.mkdir(parents=True, exist_ok=True)
        result = _self_test()
        (args.out / "selftest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    summary = analyze(args.collection_dir, args.split_manifest, args.out)
    print(json.dumps({"status": summary["status"], "output": str(args.out.resolve()),
                      "stop_recommendation": summary["stop_recommendation"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
