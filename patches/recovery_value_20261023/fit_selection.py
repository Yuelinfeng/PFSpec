"""Fit and source-holdout-check a small CPU recovery-value selector.

Input is JSONL with one pre-validation round context and a separate ``labels``
object. Target probabilities, realized recovery, and timings are labels only.
No network, GPU, third-party installation, or neural training is used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from selection import (
    CANDIDATE_FEATURES,
    MAX_BACKUPS,
    RECOVERY_COST_FEATURES,
    SCHEMA_VERSION,
    SHAPE_COST_FEATURES,
    _candidate_rows,
    candidate_feature_values,
    choose_backups,
    make_selector,
    predict_candidate,
    predict_shape_cost,
    recovery_cost_feature_values,
    save_model,
    shape_cost_feature_values,
)


def _number(value: Any, name: str, *, default: float | None = None) -> float:
    if value is None and default is not None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _token_key_map(value: Any) -> dict[int, float]:
    result: dict[int, float] = {}
    if isinstance(value, Mapping):
        iterator = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        iterator = (
            (
                item.get("token_id"),
                item.get("value", item.get("probability", item.get("target_mass", item.get("observed_value_ms")))),
            )
            for item in value if isinstance(item, Mapping)
        )
    else:
        return result
    for raw_token, raw_value in iterator:
        if raw_token is None or raw_value is None:
            continue
        try:
            token = int(raw_token)
            number = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(number):
            result[token] = number
    return result


def _candidate_label_map(labels: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    raw = labels.get("candidate_labels", labels.get("candidates", ()))
    result: dict[int, dict[str, Any]] = {}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if not isinstance(item, Mapping) or item.get("token_id") is None:
                continue
            result[int(item["token_id"])] = dict(item)
    return result


def _get_context(row: Mapping[str, Any]) -> Mapping[str, Any]:
    context = row.get("round_context", row.get("features"))
    if isinstance(context, Mapping):
        return context
    return row


def _get_labels(row: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("labels", "observer_event", "event"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _get_identity(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    context = _get_context(row)
    source = row.get("source_id", context.get("source_id"))
    if source is None or str(source) == "":
        raise ValueError(f"row {index} has no source_id; source-group holdout is required")
    context_id = row.get("context_id", context.get("context_id"))
    if context_id is None:
        raise ValueError(f"row {index} has no context_id")
    seed = row.get("seed", row.get("decode_seed", context.get("seed", 0)))
    round_id = row.get("round_id", row.get("round_index", context.get("round_id", context.get("round_index", index))))
    scene_id = row.get("scene_id", context.get("scene_id", source))
    return {
        "source_id": str(source),
        "scene_id": str(scene_id),
        "context_id": str(context_id),
        "seed": int(seed),
        "round_id": str(round_id),
    }


def _records_from_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {index}: {error}") from error
            if not isinstance(row, Mapping):
                raise ValueError(f"line {index} must contain a JSON object")
            scope = row.get("scope", "single_round")
            if scope not in ("single_round", "shape_probe", "value_probe", "round"):
                continue
            identity = _get_identity(row, index)
            context = dict(_get_context(row))
            if "candidates" not in context and "candidate_pool" not in context:
                event_candidates = row.get("candidates")
                if isinstance(event_candidates, Sequence) and not isinstance(event_candidates, (str, bytes)):
                    context["candidates"] = list(event_candidates)
            labels = dict(_get_labels(row))
            rows.append({
                "identity": identity,
                "round_context": context,
                "labels": labels,
                "probe_kind": row.get("probe_kind", labels.get("probe_kind", scope)),
                "selected_backup_ids": list(row.get("selected_backup_ids", ())),
                "diagnostic_only": bool(row.get("diagnostic_only", False)),
                "row_number": index,
            })
    if not rows:
        raise ValueError("input JSONL contains no usable single-round records")
    return rows


def _same_context_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    ident = record["identity"]
    ctx = record["round_context"]
    return (
        ident["source_id"],
        ident["context_id"],
        str(ident["seed"]),
        str(ident["round_id"]),
        str(ctx.get("prefix_len", "")),
        str(ctx.get("cache_pos", ctx.get("cache_pos_before", ctx.get("logical_cache_pos", "")))),
        str(ctx.get("read_bucket", ctx.get("tree_read_bucket", ""))),
        str(bool(ctx.get("graph_exists", ctx.get("graph_present", False)))),
        str(bool(ctx.get("graph_cold", ctx.get("cold_graph", ctx.get("runtime_created", False))))),
        str(ctx.get("remaining_budget", "")),
        str(len(ctx.get("candidates", ctx.get("candidate_pool", ())))),
    )


def _shape_context_key(
    identity: Mapping[str, Any], context: Mapping[str, Any]
) -> tuple[Any, ...]:
    feature_signature = tuple(
        tuple(
            float(values[name])
            for name in SHAPE_COST_FEATURES
        )
        for backup_count in range(MAX_BACKUPS + 1)
        for values in (shape_cost_feature_values(context, backup_count),)
    )
    return (
        identity["source_id"],
        identity["context_id"],
        str(identity["seed"]),
        str(identity["round_id"]),
        feature_signature,
    )


def _shape_measurement_context(record: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    base_context = dict(record["round_context"])
    labels = record["labels"]
    raw_probe_context = labels.get("cost_probe_feature_context")
    if not isinstance(raw_probe_context, Mapping):
        return base_context, "legacy_round_context"

    probe_context: Mapping[str, Any] = raw_probe_context
    nested = probe_context.get("round_context")
    if isinstance(nested, Mapping):
        probe_context = nested
    merged = dict(base_context)
    merged.update(probe_context)
    for name in ("candidates", "candidate_pool"):
        if name in base_context:
            merged[name] = base_context[name]
    return merged, "cost_probe_feature_context"


def _mean_values(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _aggregate_records(rows: Sequence[Mapping[str, Any]], *, position_vocab_size: int, eos_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Collapse timer repeats while retaining distinct seeds as value samples."""
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if "identity" in row and "round_context" in row and "labels" in row:
            normalized_rows.append(dict(row))
            continue
        context = dict(_get_context(row))
        if "candidates" not in context and "candidate_pool" not in context:
            event_candidates = row.get("candidates")
            if isinstance(event_candidates, Sequence) and not isinstance(event_candidates, (str, bytes)):
                context["candidates"] = list(event_candidates)
        normalized_rows.append({
            "identity": _get_identity(row, index),
            "round_context": context,
            "labels": dict(_get_labels(row)),
            "probe_kind": row.get("probe_kind", row.get("scope", "single_round")),
            "selected_backup_ids": list(row.get("selected_backup_ids", ())),
            "diagnostic_only": bool(row.get("diagnostic_only", False)),
            "row_number": index,
        })
    rows = normalized_rows
    context_stub = {
        "config": {
            "position_vocab_size": position_vocab_size,
            "eos_id": eos_id,
            "max_enum_candidates": 8,
            "max_backups": 4,
            "max_measured_backups": 4,
        }
    }
    value_groups: dict[tuple[str, str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    shape_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    shape_probe_context_rows = 0
    shape_probe_postwarm_context_rows = 0
    shape_probe_legacy_context_rows = 0
    for record in rows:
        labels = record["labels"]
        probe = str(record.get("probe_kind", record.get("scope", "single_round")))
        if any(labels.get(name) is not None for name in (
            "extra_shape_cost_ms_by_k", "shape_cost_ms_by_k", "shape_costs"
        )) or probe == "shape_probe":
            shape_probe_context_rows += 1
            shape_context, shape_context_source = _shape_measurement_context(record)
            if shape_context_source == "cost_probe_feature_context":
                shape_probe_postwarm_context_rows += 1
            else:
                shape_probe_legacy_context_rows += 1
            shape_normalized, _, _ = _candidate_rows(
                {"round_context": shape_context}, context_stub
            )
            shape_record = dict(record)
            shape_record["shape_round_context"] = shape_context
            shape_record["shape_context_source"] = shape_context_source
            shape_key = _shape_context_key(record["identity"], shape_normalized)
            shape_groups[shape_key].append(shape_record)
        context = record["round_context"]
        has_candidates = context.get("candidates", context.get("candidate_pool")) is not None
        has_candidate_observation = any(labels.get(name) is not None for name in (
            "target_mass", "target_mass_by_token", "target_policy_by_token",
            "residual_mass_by_token", "candidate_labels",
            "recovery_value_observed_ms_by_candidate",
        ))
        if has_candidates and has_candidate_observation and probe != "shape_probe":
            ident = record["identity"]
            key = (
                ident["source_id"], ident["context_id"], ident["scene_id"],
                ident["seed"], ident["round_id"],
            )
            value_groups[key].append(record)

    value_samples: list[dict[str, Any]] = []
    for key, group in value_groups.items():
        first = group[0]
        raw_context = first["round_context"]
        normalized_context, candidates, _ = _candidate_rows(
            {"round_context": raw_context}, context_stub
        )
        if not candidates:
            continue
        candidate_mass_values: dict[int, list[float]] = defaultdict(list)
        candidate_value_values: dict[int, list[float]] = defaultdict(list)
        recovery_cost_values: list[float] = []
        recovery_cost_weights: list[float] = []
        p_main_values: list[float] = []
        p_eos_values: list[float] = []

        for record in group:
            labels = record["labels"]
            label_rows = _candidate_label_map(labels)
            target_map = _token_key_map(labels.get(
                "target_mass_by_token",
                labels.get("target_policy_by_token", labels.get("target_mass")),
            ))
            residual_map = _token_key_map(labels.get("residual_mass_by_token"))
            actual_y_value = labels.get("actual_residual_position", labels.get("actual_y", labels.get("residual_position")))
            w1_valid_value = labels.get("recovery_value_label_valid_for_w1")
            if w1_valid_value is None:
                w1_valid = not record.get("selected_backup_ids", ())
            else:
                w1_valid = bool(w1_valid_value)
            diagnostic_only = bool(record.get("diagnostic_only", False))
            if isinstance(actual_y_value, str) and actual_y_value.strip().lower() in {"eos", "terminal"}:
                actual_y = eos_id
            else:
                actual_y = None if actual_y_value is None else int(actual_y_value)
            recovery_called = bool(labels.get("recovery_call_occurred", labels.get("recovery_occurred", False)))
            stage_value = labels.get("recovery_stage")
            if stage_value is None:
                # Older event rows can identify P1 recovery by the residual P1
                # Position. P2 recovery has no P1 residual label.
                recovery_stage = "p1" if actual_y is not None else ("p2" if recovery_called else "none")
            else:
                recovery_stage = str(stage_value).strip().lower()
            recovery_flag = recovery_called and recovery_stage in {
                "p1", "position_p1", "p1_position", "p1_recovery"
            }
            recovery_ms_value = labels.get("recovery_ms", labels.get("recovery_cost_ms"))
            recovery_ms = None if recovery_ms_value is None else _number(recovery_ms_value, "labels.recovery_ms")

            p_main = labels.get("p_main", labels.get("target_main_mass"))
            p_eos = labels.get("p_eos", labels.get("target_eos_mass"))
            if p_main is not None:
                p_main_values.append(_number(p_main, "labels.p_main"))
            if p_eos is not None:
                p_eos_values.append(_number(p_eos, "labels.p_eos"))

            observation_probability = labels.get("recovery_observation_probability")
            if observation_probability is None and p_main is not None and p_eos is not None:
                observation_probability = 1.0 - float(p_main) - float(p_eos)
            if recovery_flag and recovery_ms is not None and observation_probability is not None:
                p_obs = _number(observation_probability, "labels.recovery_observation_probability")
                if p_obs > 0.0:
                    recovery_cost_values.append(recovery_ms)
                    recovery_cost_weights.append(1.0 / p_obs)

            observed_value_map = _token_key_map(labels.get("recovery_value_observed_ms_by_candidate"))
            valid_value_observation = w1_valid and not diagnostic_only
            for candidate in candidates:
                token = int(candidate["token_id"])
                row_label = label_rows.get(token, {})
                mass = residual_map.get(token)
                if mass is None:
                    mass = row_label.get("residual_mass")
                if mass is None:
                    mass = target_map.get(token, row_label.get(
                        "target_mass", row_label.get("probability")
                    ))
                if mass is not None:
                    mass = _number(mass, f"candidate {token} mass")
                    if mass < -1e-8 or mass > 1.0 + 1e-6:
                        raise ValueError(f"candidate {token} mass outside [0,1]")
                    candidate_mass_values[token].append(min(1.0, max(0.0, mass)))

                if token in observed_value_map:
                    value = observed_value_map[token]
                elif "realized_recovery_value_ms" in row_label:
                    value = _number(row_label["realized_recovery_value_ms"], f"candidate {token} realized_recovery_value_ms")
                    if value < 0.0:
                        raise ValueError("realized_recovery_value_ms must be nonnegative")
                elif "recovery_value_ms" in row_label:
                    value = _number(row_label["recovery_value_ms"], f"candidate {token} recovery_value_ms")
                    if value < 0.0:
                        raise ValueError("recovery_value_ms must be nonnegative")
                elif recovery_flag and actual_y == token and recovery_ms is not None:
                    value = recovery_ms
                else:
                    value = 0.0
                observation_present = (
                    token in observed_value_map
                    or "realized_recovery_value_ms" in row_label
                    or actual_y is not None
                    or "main_accepted" in labels
                    or "recovery_call_occurred" in labels
                )
                if valid_value_observation and observation_present:
                    candidate_value_values[token].append(value)

        # Timer repeats for one context/seed/round represent one stochastic sample.
        value_sample = {
            "identity": first["identity"],
            "context": normalized_context,
            "raw_context": dict(raw_context),
            "shape_join_key": _shape_context_key(first["identity"], normalized_context),
            "candidates": candidates,
            "mass_labels": {token: _mean_values(values) for token, values in candidate_mass_values.items()},
            "value_labels": {token: _mean_values(values) for token, values in candidate_value_values.items()},
            "cost_labels": {},
            "recovery_cost_ms": _mean_values(recovery_cost_values) if recovery_cost_values else None,
            "recovery_cost_weight": _mean_values(recovery_cost_weights) if recovery_cost_weights else None,
            "p_main": _mean_values(p_main_values) if p_main_values else None,
            "p_eos": _mean_values(p_eos_values) if p_eos_values else None,
            "shape_costs": {},
            "row_count": len(group),
        }
        value_samples.append(value_sample)

    shape_samples: list[dict[str, Any]] = []
    for key, group in shape_groups.items():
        first = group[0]
        shape_raw_context = first["shape_round_context"]
        normalized_shape_context, candidates, _ = _candidate_rows(
            {"round_context": shape_raw_context}, context_stub
        )
        by_k: dict[int, list[float]] = defaultdict(list)
        for record in group:
            labels = record["labels"]
            cost_map = labels.get("extra_shape_cost_ms_by_k")
            if not isinstance(cost_map, Mapping):
                cost_map = labels.get("shape_cost_ms_by_k", {})
                if isinstance(cost_map, Mapping) and "0" in cost_map:
                    base = _number(cost_map["0"], "shape_cost_ms_by_k[0]")
                    cost_map = {
                        str(raw_k): _number(raw_cost, f"shape_cost_ms_by_k[{raw_k}]") - base
                        for raw_k, raw_cost in cost_map.items()
                    }
                elif isinstance(cost_map, Mapping):
                    # Older synthetic rows supplied added-cost values directly.
                    cost_map = dict(cost_map)
            if isinstance(cost_map, Mapping):
                for raw_k, raw_cost in cost_map.items():
                    k = int(raw_k)
                    cost = _number(raw_cost, f"extra_shape_cost_ms_by_k[{k}]")
                    if 0 <= k <= MAX_BACKUPS:
                        by_k[k].append(cost)
            subset_rows = record["labels"].get("shape_costs", ())
            if isinstance(subset_rows, Sequence) and not isinstance(subset_rows, (str, bytes)):
                for item in subset_rows:
                    if not isinstance(item, Mapping):
                        continue
                    k = _number(item.get("backup_count", item.get("k")), "shape_costs.k")
                    cost = _number(item.get("extra_shape_cost_ms"), "shape_costs.extra_shape_cost_ms")
                    if k.is_integer() and 0 <= int(k) <= MAX_BACKUPS:
                        by_k[int(k)].append(cost)
        costs = {k: _mean_values(values) for k, values in by_k.items()}
        if not costs:
            continue
        shape_samples.append({
            "identity": first["identity"],
            "context": normalized_shape_context,
            "raw_context": dict(shape_raw_context),
            "shape_join_key": key,
            "candidates": candidates,
            "shape_costs": costs,
            "repeat_rows": len(group),
            "measurement_context_source": first["shape_context_source"],
        })

    # Attach repeated shape probes to matching stochastic context samples.
    shape_by_context: dict[tuple[Any, ...], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sample in shape_samples:
        key = sample["shape_join_key"]
        for k, cost in sample["shape_costs"].items():
            shape_by_context[key][k].append(cost)
    for sample in value_samples:
        key = sample["shape_join_key"]
        sample["shape_costs"] = {k: _mean_values(values) for k, values in shape_by_context.get(key, {}).items()}

    stats = {
        "input_rows": len(rows),
        "value_context_seed_rounds": len(value_samples),
        "shape_probe_contexts": len(shape_samples),
        "shape_probe_cost_context_rows": shape_probe_context_rows,
        "shape_probe_postwarm_context_rows": shape_probe_postwarm_context_rows,
        "shape_probe_legacy_context_rows": shape_probe_legacy_context_rows,
        "shape_probe_context_note": (
            "cost_probe_feature_context is used when present; legacy rows use round_context, "
            "whose graph state may predate shape-probe warmup"
        ),
        "source_count": len({sample["identity"]["source_id"] for sample in value_samples + shape_samples}),
        "value_sources": sorted({sample["identity"]["source_id"] for sample in value_samples}),
        "shape_sources": sorted({sample["identity"]["source_id"] for sample in shape_samples}),
        "shape_probe_context_sources": {
            "cost_probe_feature_context": shape_probe_postwarm_context_rows,
            "legacy_round_context": shape_probe_legacy_context_rows,
        },
        "value_label_semantics": "W1-only realized per-candidate observed_value_ms; regression estimates its conditional mean",
        "recovery_cost_weighting": "inverse observation probability when p_main and p_eos are supplied",
        "diagnostic_only_value_rows_excluded": sum(bool(record.get("diagnostic_only")) for record in rows),
        "p2_recovery_rows_excluded_from_p1_cost": sum(
            str(record["labels"].get("recovery_stage", "")).lower().startswith("p2")
            for record in rows
        ),
    }
    return value_samples, shape_samples, stats


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    work = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-12:
            raise ValueError("ridge system is singular")
        work[column], work[pivot] = work[pivot], work[column]
        divisor = work[column][column]
        for j in range(column, n + 1):
            work[column][j] /= divisor
        for row in range(n):
            if row == column:
                continue
            factor = work[row][column]
            if factor == 0.0:
                continue
            for j in range(column, n + 1):
                work[row][j] -= factor * work[column][j]
    return [work[index][n] for index in range(n)]


def _fit_ridge(
    rows: Sequence[Mapping[str, float]],
    targets: Sequence[float],
    feature_names: Sequence[str],
    *,
    alpha: float,
    sample_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    if not rows or len(rows) != len(targets):
        raise ValueError("ridge needs paired feature and target rows")
    width = len(feature_names)
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in feature_names:
        values = [float(row.get(name, 0.0)) for row in rows]
        mean = statistics.fmean(values)
        variance = statistics.fmean((value - mean) ** 2 for value in values)
        scale = math.sqrt(variance) if variance > 1e-12 else 1.0
        if name == "bias":
            mean, scale = 0.0, 1.0
        means[name] = mean
        scales[name] = scale
    weights = [1.0] * len(rows) if sample_weights is None else [float(value) for value in sample_weights]
    if len(weights) != len(rows) or any(value <= 0.0 or not math.isfinite(value) for value in weights):
        raise ValueError("sample weights must be positive finite values")

    xtx = [[0.0] * width for _ in range(width)]
    xty = [0.0] * width
    for row, target, row_weight in zip(rows, targets, weights):
        vector = [(float(row.get(name, 0.0)) - means[name]) / scales[name] for name in feature_names]
        for i in range(width):
            xty[i] += row_weight * vector[i] * float(target)
            for j in range(width):
                xtx[i][j] += row_weight * vector[i] * vector[j]
    for i in range(width):
        if feature_names[i] != "bias":
            xtx[i][i] += float(alpha)
    coefficients = _solve_linear(xtx, xty)
    return {
        "feature_names": list(feature_names),
        "means": means,
        "scales": scales,
        "weights": {name: coefficients[index] for index, name in enumerate(feature_names)},
        "train_rows": len(rows),
        "ridge_alpha": float(alpha),
        "target_mean": statistics.fmean(float(value) for value in targets),
        "target_min": min(float(value) for value in targets),
        "target_max": max(float(value) for value in targets),
    }


def _source_split(sources: Sequence[str], validation_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    unique = sorted(set(str(source) for source in sources))
    if len(unique) < 2:
        raise ValueError("at least two distinct sources are needed for source-held-out validation")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    shuffled = unique[:]
    random.Random(int(seed)).shuffle(shuffled)
    n_validation = min(len(unique) - 1, max(1, int(round(len(unique) * validation_fraction))))
    validation = sorted(shuffled[:n_validation])
    train = sorted(shuffled[n_validation:])
    if set(train) & set(validation):
        raise AssertionError("source split overlap")
    return train, validation


def _training_rows(samples: Sequence[Mapping[str, Any]], train_sources: set[str], alpha: float) -> tuple[dict[str, Any], dict[str, Any]]:
    mass_features: list[dict[str, float]] = []
    mass_targets: list[float] = []
    value_features: list[dict[str, float]] = []
    value_targets: list[float] = []
    recovery_cost_features: list[dict[str, float]] = []
    recovery_cost_targets: list[float] = []
    recovery_cost_weights: list[float] = []
    shape_features: list[dict[str, float]] = []
    shape_targets: list[float] = []
    shape_k_seen: set[int] = set()

    for sample in samples:
        if sample["identity"]["source_id"] not in train_sources:
            continue
        ctx = sample["context"]
        for candidate in sample["candidates"]:
            token = int(candidate["token_id"])
            features = candidate_feature_values(ctx, candidate)
            if token in sample["mass_labels"]:
                mass_features.append(features)
                mass_targets.append(float(sample["mass_labels"][token]))
            if token in sample["value_labels"]:
                value_features.append(features)
                value_targets.append(float(sample["value_labels"][token]))
        if sample.get("recovery_cost_ms") is not None and sample.get("recovery_cost_weight") is not None:
            recovery_cost_features.append(recovery_cost_feature_values(ctx))
            recovery_cost_targets.append(float(sample["recovery_cost_ms"]))
            recovery_cost_weights.append(float(sample["recovery_cost_weight"]))
        for k, cost in sample.get("shape_costs", {}).items():
            k = int(k)
            if k < 0:
                continue
            shape_features.append(shape_cost_feature_values(ctx, k))
            shape_targets.append(float(cost))
            shape_k_seen.add(k)

    models = {
        "residual_mass": _fit_ridge(mass_features, mass_targets, CANDIDATE_FEATURES, alpha=alpha) if mass_features else None,
        "recovery_value_ms": _fit_ridge(value_features, value_targets, CANDIDATE_FEATURES, alpha=alpha) if value_features else None,
        "recovery_cost_ms": _fit_ridge(recovery_cost_features, recovery_cost_targets, RECOVERY_COST_FEATURES, alpha=alpha, sample_weights=recovery_cost_weights) if recovery_cost_features else None,
        "shape_cost_ms": _fit_ridge(shape_features, shape_targets, SHAPE_COST_FEATURES, alpha=alpha) if shape_features else None,
    }
    weight_sum = sum(recovery_cost_weights)
    weight_square_sum = sum(weight * weight for weight in recovery_cost_weights)
    return models, {
        "supported_k": sorted(shape_k_seen),
        "training_candidate_rows": len(mass_features),
        "training_value_rows": len(value_features),
        "training_recovery_cost_rows": len(recovery_cost_features),
        "recovery_cost_effective_sample_size": (weight_sum * weight_sum / weight_square_sum) if weight_square_sum else 0.0,
        "training_shape_rows": len(shape_features),
    }


def _mae(predicted: Sequence[float], actual: Sequence[float]) -> float | None:
    if not predicted:
        return None
    return statistics.fmean(abs(float(p) - float(a)) for p, a in zip(predicted, actual))


def _source_mean(values_by_source: Mapping[str, Sequence[float]]) -> float | None:
    source_means = [statistics.fmean(values) for values in values_by_source.values() if values]
    return statistics.fmean(source_means) if source_means else None


def _label_coverage(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mass_rows = sum(len(sample.get("mass_labels", {})) for sample in samples)
    value_rows = sum(len(sample.get("value_labels", {})) for sample in samples)
    positive_rows = 0
    positive_contexts = 0
    positive_by_rank: dict[str, int] = defaultdict(int)
    for sample in samples:
        labels = sample.get("value_labels", {})
        context_hit = False
        for candidate in sample.get("candidates", ()):
            token = int(candidate["token_id"])
            if token in labels and float(labels[token]) > 0.0:
                positive_rows += 1
                context_hit = True
                positive_by_rank[str(int(candidate["ranker_rank"]))] += 1
        positive_contexts += int(context_hit)
    return {
        "contexts": len(samples),
        "candidate_mass_labels": mass_rows,
        "realized_candidate_value_labels": value_rows,
        "positive_realized_candidate_value_labels": positive_rows,
        "contexts_with_positive_realized_candidate_value": positive_contexts,
        "positive_label_fraction": positive_rows / value_rows if value_rows else None,
        "positive_labels_by_ranker_rank": dict(sorted(positive_by_rank.items(), key=lambda item: int(item[0]))),
        "p1_recovery_cost_contexts": sum(sample.get("recovery_cost_ms") is not None for sample in samples),
        "positive_label_semantics": "W1 realized candidate hit times P1 recovery latency; positive labels are sample outcomes, not acceptance probabilities",
    }


def _actual_net(sample: Mapping[str, Any], selected: Sequence[int], *, predicted_shape: float | None = None) -> float | None:
    k = len(selected)
    if k == 0:
        cost = 0.0
    elif k in sample.get("shape_costs", {}):
        cost = float(sample["shape_costs"][k])
    elif predicted_shape is not None:
        cost = predicted_shape
    else:
        return None
    values = sample.get("value_labels", {})
    if any(token not in values for token in selected):
        return None
    return sum(float(values[token]) for token in selected) - cost


def _ranker_ids(sample: Mapping[str, Any], k: int) -> tuple[int, ...]:
    return tuple(int(row["token_id"]) for row in sample["candidates"][:k])


def _rank_by(sample: Mapping[str, Any], model: Mapping[str, Any], key: str) -> list[int]:
    scored = []
    for candidate in sample["candidates"]:
        prediction = predict_candidate(model, sample["context"], candidate)
        if key == "mass_times_cost_ms":
            score = prediction["mass_times_cost_ms"]
        else:
            score = prediction[key]
        scored.append((float(score), int(candidate["ranker_rank"]), int(candidate["token_id"])))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [token for _, _, token in scored]


def _evaluate_validation(samples: Sequence[Mapping[str, Any]], model: Mapping[str, Any], max_backups: int) -> dict[str, Any]:
    policies = [
        "fixed_ranker_1",
        "fixed_ranker_k",
        "ranker_same_k",
        "predicted_mass_same_k",
        "factorized_mass_cost",
        "direct_recovery_value",
        "offline_oracle",
    ]
    net_by_policy_source: dict[str, dict[str, list[float]]] = {policy: defaultdict(list) for policy in policies}
    chosen_k_by_policy: dict[str, list[int]] = {policy: [] for policy in policies}
    coverage = {policy: 0 for policy in policies}
    cost_usage = {
        policy: {"zero_backup": 0, "measured_shape_cost": 0, "predicted_shape_cost": 0, "unavailable": 0}
        for policy in policies
    }
    measured_shape_contexts_by_k: dict[int, int] = defaultdict(int)
    oracle_eligible = 0
    selection_comparisons: dict[str, dict[str, Any]] = {}
    comparison_pairs = (
        ("direct_recovery_value", "ranker_same_k"),
        ("direct_recovery_value", "predicted_mass_same_k"),
        ("direct_recovery_value", "factorized_mass_cost"),
        ("direct_recovery_value", "fixed_ranker_1"),
    )
    for left, right in comparison_pairs:
        selection_comparisons[f"{left}_vs_{right}"] = {
            "contexts_compared": 0,
            "same_candidate_ids": 0,
            "different_candidate_ids": 0,
            "different_backup_count": 0,
            "examples": [],
        }
    total = 0
    mass_predicted: list[float] = []
    mass_actual: list[float] = []
    value_predicted: list[float] = []
    value_actual: list[float] = []
    cost_predicted: list[float] = []
    cost_actual: list[float] = []
    for sample in samples:
        total += 1
        source = sample["identity"]["source_id"]
        context = sample["context"]
        for k in sample.get("shape_costs", {}):
            measured_shape_contexts_by_k[int(k)] += 1
        if sample.get("recovery_cost_ms") is not None and sample["candidates"]:
            cost_predicted.append(
                predict_candidate(model, context, sample["candidates"][0])["recovery_cost_ms"]
            )
            cost_actual.append(float(sample["recovery_cost_ms"]))
        for candidate in sample["candidates"]:
            token = int(candidate["token_id"])
            prediction = predict_candidate(model, context, candidate)
            if token in sample["mass_labels"]:
                mass_predicted.append(prediction["residual_mass"])
                mass_actual.append(float(sample["mass_labels"][token]))
            if token in sample["value_labels"]:
                value_predicted.append(prediction["recovery_value_ms"])
                value_actual.append(float(sample["value_labels"][token]))
        direct = choose_backups(model, {"round_context": sample["raw_context"]}, mode="net_value", max_backups=max_backups)
        factorized = choose_backups(model, {"round_context": sample["raw_context"]}, mode="mass_times_cost", max_backups=max_backups)

        selected_by_policy: dict[str, tuple[int, ...]] = {
            "fixed_ranker_1": _ranker_ids(sample, min(1, max_backups)),
            "fixed_ranker_k": _ranker_ids(sample, max_backups),
            "ranker_same_k": _ranker_ids(sample, len(direct.backup_ids)),
            "predicted_mass_same_k": tuple(_rank_by(sample, model, "residual_mass")[:len(direct.backup_ids)]),
            "factorized_mass_cost": factorized.backup_ids,
            "direct_recovery_value": direct.backup_ids,
            "offline_oracle": (),
        }
        # This hindsight reference is scored only when the same context has both
        # realized W1 candidate labels and measured costs for every allowed k.
        best_oracle_net = 0.0
        best_oracle_ids: tuple[int, ...] = ()
        allowed_k = list(range(1, min(max_backups, len(sample["candidates"])) + 1))
        all_candidate_labels_present = all(
            int(candidate["token_id"]) in sample["value_labels"]
            for candidate in sample["candidates"]
        )
        oracle_has_common_cost = all(k in sample["shape_costs"] for k in allowed_k)
        if all_candidate_labels_present and oracle_has_common_cost:
            oracle_eligible += 1
        for k in allowed_k if all_candidate_labels_present and oracle_has_common_cost else ():
            for subset in combinations(sample["candidates"], k):
                ids = tuple(int(row["token_id"]) for row in subset)
                net = sum(float(sample["value_labels"][token]) for token in ids) - float(sample["shape_costs"][k])
                if net > best_oracle_net:
                    best_oracle_net = net
                    best_oracle_ids = ids
        selected_by_policy["offline_oracle"] = best_oracle_ids

        for left, right in comparison_pairs:
            key = f"{left}_vs_{right}"
            comparison = selection_comparisons[key]
            left_ids = tuple(selected_by_policy[left])
            right_ids = tuple(selected_by_policy[right])
            comparison["contexts_compared"] += 1
            if set(left_ids) == set(right_ids):
                comparison["same_candidate_ids"] += 1
            else:
                comparison["different_candidate_ids"] += 1
                if len(left_ids) != len(right_ids):
                    comparison["different_backup_count"] += 1
                if len(comparison["examples"]) < 3:
                    comparison["examples"].append({
                        "context_id": sample["identity"]["context_id"],
                        "source_id": source,
                        left: list(left_ids),
                        right: list(right_ids),
                    })

        for policy, selected in selected_by_policy.items():
            if policy in ("fixed_ranker_k",) and max_backups > len(sample["candidates"]):
                continue
            predicted_shape = None
            if policy == "offline_oracle" and not (all_candidate_labels_present and oracle_has_common_cost):
                cost_usage[policy]["unavailable"] += 1
                continue
            if not selected:
                cost_usage[policy]["zero_backup"] += 1
            elif len(selected) in sample["shape_costs"]:
                cost_usage[policy]["measured_shape_cost"] += 1
            elif policy == "offline_oracle":
                cost_usage[policy]["unavailable"] += 1
                continue
            else:
                predicted_shape = predict_shape_cost(model, context, len(selected))
                cost_usage[policy]["predicted_shape_cost"] += 1
            actual = _actual_net(sample, selected, predicted_shape=predicted_shape)
            if actual is None:
                cost_usage[policy]["unavailable"] += 1
                continue
            net_by_policy_source[policy][source].append(actual)
            chosen_k_by_policy[policy].append(len(selected))
            coverage[policy] += 1

    policy_report: dict[str, Any] = {}
    for policy in policies:
        values = net_by_policy_source[policy]
        report = {
            "contexts_scored": coverage[policy],
            "mean_selected_backups": statistics.fmean(chosen_k_by_policy[policy]) if chosen_k_by_policy[policy] else None,
            "source_count": len(values),
            "shape_cost_usage_contexts": cost_usage[policy],
        }
        if policy == "offline_oracle":
            report.update({
                "status": "measured_hindsight_reference" if oracle_eligible else "not_estimable",
                "source_equal_realized_label_hindsight_net_ms_measured_cost_only": _source_mean(values) if oracle_eligible else None,
                "eligible_common_measured_contexts": oracle_eligible,
            })
        else:
            report.update({
                "source_equal_realized_label_net_ms_using_measured_or_predicted_shape_cost": _source_mean(values),
                "interpretation": "source-equal offline estimate from W1 realized candidate labels and same-context measured or ridge-predicted added shape cost; not an observed online result",
            })
        policy_report[policy] = report
    selector = model.get("runtime_selector", {})
    selector_contexts = samples[: min(128, len(samples))]
    if selector_contexts:
        timings: list[float] = []
        for _ in range(3):
            started = __import__("time").perf_counter_ns()
            for sample in selector_contexts:
                choose_backups(model, {"round_context": sample["raw_context"]}, mode="net_value", max_backups=max_backups)
            timings.append((__import__("time").perf_counter_ns() - started) / max(1, len(selector_contexts)) / 1000.0)
        selector_cpu_us = statistics.median(timings)
    else:
        selector_cpu_us = None
    return {
        "validation_contexts": total,
        "mass_mae": _mae(mass_predicted, mass_actual),
        "realized_recovery_value_mae_ms": _mae(value_predicted, value_actual),
        "recovery_cost_mae_ms": _mae(cost_predicted, cost_actual),
        "policies": policy_report,
        "measured_shape_cost_contexts_by_k": {str(k): measured_shape_contexts_by_k.get(k, 0) for k in range(1, max_backups + 1)},
        "measured_only_oracle_coverage": {
            "contexts_with_all_candidate_labels_and_same_context_cost_for_each_allowed_k": oracle_eligible,
            "eligible_value_contexts": total,
            "fraction": oracle_eligible / total if total else None,
            "status": "not_estimable" if oracle_eligible == 0 else "measured_hindsight_reference",
        },
        "candidate_selection_comparisons": selection_comparisons,
        "selector_cpu_us_per_context_median": selector_cpu_us,
        "selector_cpu_timing_note": "CPU callback time only; no GPU synchronization; not subtracted from offline estimates",
        "offline_oracle_note": "realized-label hindsight reference; requires same-context measured added shape costs for every considered nonempty action; unavailable contexts are excluded and never assigned a zero net value",
        "direct_value_label_note": "held-out W1 realized indicator(actual_Y==candidate and P1 recovery call)*recovery_ms; ridge estimates a conditional mean, while validation policy sums use the held-out realized labels",
        "mixed_cost_note": "policy net estimates use same-context measured shape costs where available and otherwise ridge-predicted costs; these are not jointly measured action costs or online gains",
    }


def _raw_input_action_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, dict[str, Any]] = {
        "all_rows": {"rows": 0, "rows_with_nonempty_selected_backup_ids": 0, "candidate_value_label_rows": 0, "positive_candidate_value_labels": 0},
        "non_diagnostic_rows": {"rows": 0, "rows_with_nonempty_selected_backup_ids": 0, "candidate_value_label_rows": 0, "positive_candidate_value_labels": 0},
        "diagnostic_only_rows": {"rows": 0, "rows_with_nonempty_selected_backup_ids": 0, "candidate_value_label_rows": 0, "positive_candidate_value_labels": 0},
    }
    backup_count_distribution: dict[str, dict[str, int]] = {
        "all_rows": defaultdict(int),
        "non_diagnostic_rows": defaultdict(int),
        "diagnostic_only_rows": defaultdict(int),
    }
    for row in rows:
        diagnostic = bool(row.get("diagnostic_only", False))
        group_name = "diagnostic_only_rows" if diagnostic else "non_diagnostic_rows"
        labels = row.get("labels", {})
        selected = row.get("selected_backup_ids", ())
        value_rows = labels.get("recovery_value_observed_ms_by_candidate", ()) if isinstance(labels, Mapping) else ()
        count = len(selected) if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)) else 0
        backup_count_distribution["all_rows"][str(count)] += 1
        backup_count_distribution[group_name][str(count)] += 1
        for name in ("all_rows", group_name):
            result[name]["rows"] += 1
            result[name]["rows_with_nonempty_selected_backup_ids"] += int(count > 0)
            if isinstance(value_rows, Sequence) and not isinstance(value_rows, (str, bytes)):
                for item in value_rows:
                    if not isinstance(item, Mapping):
                        continue
                    result[name]["candidate_value_label_rows"] += 1
                    try:
                        positive = float(item.get("observed_value_ms", 0.0)) > 0.0
                    except (TypeError, ValueError, OverflowError):
                        positive = False
                    result[name]["positive_candidate_value_labels"] += int(positive)
    for group_name, counts in backup_count_distribution.items():
        result[group_name]["selected_backup_count_distribution"] = dict(sorted(counts.items(), key=lambda item: int(item[0])))
    return result


def fit_from_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float = 0.2,
    split_seed: int = 20261023,
    ridge_alpha: float = 1.0,
    max_backups: int = 1,
    position_vocab_size: int = 8000,
    eos_id: int = 12096,
) -> tuple[dict[str, Any], dict[str, Any]]:
    samples, shape_samples, input_stats = _aggregate_records(rows, position_vocab_size=position_vocab_size, eos_id=eos_id)
    if not samples:
        raise ValueError("no rows with candidates were found")
    sources = sorted({sample["identity"]["source_id"] for sample in samples})
    train_sources, validation_sources = _source_split(sources, validation_fraction, split_seed)

    # Shape probes may be separate from value observations; add them as cost-only samples.
    combined_samples = list(samples)
    sample_keys = {
        (s["identity"]["source_id"], s["identity"]["context_id"], s["identity"]["seed"], s["identity"]["round_id"])
        for s in samples
    }
    for sample in shape_samples:
        ident = sample["identity"]
        key = (ident["source_id"], ident["context_id"], ident["seed"], ident["round_id"])
        if key in sample_keys:
            continue
        sample_keys.add(key)
        combined_samples.append({
            "identity": ident,
            "context": sample["context"],
            "raw_context": sample["raw_context"],
            "candidates": sample["candidates"],
            "mass_labels": {},
            "value_labels": {},
            "cost_labels": {},
            "recovery_cost_ms": None,
            "recovery_cost_weight": None,
            "p_main": None,
            "p_eos": None,
            "shape_costs": sample["shape_costs"],
            "row_count": sample["repeat_rows"],
        })
    # Preserve the original raw context for inference and validation replay.
    for sample in combined_samples:
        if "raw_context" not in sample:
            matches = next((row for row in rows if row["identity"]["context_id"] == sample["identity"]["context_id"] and row["identity"]["source_id"] == sample["identity"]["source_id"]), None)
            sample["raw_context"] = dict(matches["round_context"]) if matches else {}

    train_set = set(train_sources)
    model_fits, fit_stats = _training_rows(combined_samples, train_set, ridge_alpha)
    supported_k = fit_stats["supported_k"]
    if not any(int(k) > 0 for k in supported_k):
        raise ValueError("no measured shape_cost_ms_by_k labels; shape costs may not be guessed")
    trained_max = min(4, max(supported_k))
    enabled_max = min(max(0, int(max_backups)), trained_max)
    if enabled_max < 1:
        enabled_max = 1 if 1 in supported_k else 0
    model = {
        "schema_version": SCHEMA_VERSION,
        "kind": "gaussiangpt_recovery_value_cpu_model",
        "config": {
            "position_vocab_size": int(position_vocab_size),
            "eos_id": int(eos_id),
            "max_enum_candidates": 8,
            "max_backups": enabled_max,
            "max_measured_backups": trained_max,
            "default_max_backups": 1,
            "supported_k": supported_k,
            "feature_source": "round_context only; labels are excluded",
            "position_proposal": "point",
            "cost_label": "measured added shape cost excluding avoided recovery forward",
        },
        "split": {
            "kind": "source_group_holdout",
            "seed": int(split_seed),
            "train_sources": train_sources,
            "validation_sources": validation_sources,
        },
        "models": model_fits,
    }

    validation = [sample for sample in combined_samples if sample["identity"]["source_id"] in set(validation_sources)]
    validation_value = [sample for sample in validation if sample.get("value_labels")]
    evaluation = _evaluate_validation(validation_value, model, enabled_max)
    validation_shape_only = [
        sample for sample in validation
        if not sample.get("value_labels") and sample.get("shape_costs")
    ]
    evaluation["shape_probe_coverage"] = {
        "unjoined_validation_shape_probe_contexts": len(validation_shape_only),
        "validation_value_contexts_with_same_context_measured_cost": sum(
            bool(sample.get("shape_costs")) for sample in validation_value
        ),
        "interpretation": "separate cost-probe rows are not treated as same-context costs unless the aggregation keys match",
    }
    limitations = [
        "Source-held-out development validation is not an external test result.",
        "Direct recovery-value labels are stochastic realized indicators; ridge estimates their conditional mean.",
        "Factorized mass-times-cost is an ablation, not an identity after conditioning on compressed features.",
        "Only actions with measured shape costs are enabled; default max_backups remains one.",
        "Policy net values combine realized W1 labels with measured or predicted added shape costs; no policy net is a measured online gain.",
        "The measured-cost hindsight reference is not estimable when same-context costs are absent; it is never filled with zero or predicted cost.",
    ]
    if input_stats["shape_probe_legacy_context_rows"]:
        limitations.append(
            "Legacy shape probes without cost_probe_feature_context use round_context; its graph state may predate probe warmup."
        )
    diagnostics = {
        "fit_config": {
            "ridge_alpha": float(ridge_alpha),
            "max_backups": int(max_backups),
            "validation_fraction": float(validation_fraction),
            "source_split_seed": int(split_seed),
            "position_vocab_size": int(position_vocab_size),
            "eos_id": int(eos_id),
        },
        "input": input_stats,
        "fit": fit_stats,
        "split": model["split"],
        "validation": evaluation,
        "sample_coverage": {
            "training_value_contexts": _label_coverage([
                sample for sample in samples if sample["identity"]["source_id"] in train_set
            ]),
            "validation_value_contexts": _label_coverage(validation_value),
            "training_measured_shape_cost_labels": sum(
                len(sample.get("shape_costs", {}))
                for sample in combined_samples
                if sample["identity"]["source_id"] in train_set
            ),
            "validation_measured_shape_cost_labels": sum(
                len(sample.get("shape_costs", {})) for sample in validation
            ),
        },
        "raw_input_action_coverage": _raw_input_action_coverage(rows),
        "limitations": limitations,
    }
    return model, diagnostics


def _synthetic_record(source: str, context_index: int, *, score_shift: float = 0.0) -> dict[str, Any]:
    prefix = 64 if context_index % 2 == 0 else 256
    candidates = [
        {"token_id": 9, "ranker_rank": 0, "ranker_score": 0.95, "position_rank": 95, "rank_gap": 0},
        {"token_id": 10, "ranker_rank": 1, "ranker_score": 0.9 + score_shift, "position_rank": 100, "rank_gap": 5},
        {"token_id": 11, "ranker_rank": 2, "ranker_score": 0.8 + score_shift, "position_rank": 105, "rank_gap": 10},
        {"token_id": 12, "ranker_rank": 3, "ranker_score": 0.2, "position_rank": 112, "rank_gap": 17},
    ]
    actual_y = None if context_index % 4 == 0 else (10 if context_index % 3 == 0 else 11)
    recovery_ms = 4.0 + prefix / 128.0 + context_index / 20.0
    recovery_called = actual_y is not None
    return {
        "schema_version": 1,
        "source_id": source,
        "scene_id": source,
        "context_id": f"{source}-ctx-{context_index}",
        "seed": context_index,
        "round_id": 0,
        "scope": "single_round",
        "round_context": {
            "prefix_len": prefix,
            "remaining_budget": 512,
            "cache_pos": prefix - 1,
            "read_bucket": 512,
            "graph_exists": bool(context_index % 2),
            "graph_cold": bool(context_index % 3 == 0),
            "main_position": 9,
            "main_score": 0.95,
            "main_position_rank": 95,
            "previous_position_rank": 90,
            "candidates": candidates,
        },
        "labels": {
            "p_main": 0.65,
            "p_eos": 0.01,
            "recovery_observation_probability": 0.34,
            "target_mass": [
                {"token_id": 9, "probability": 0.65},
                {"token_id": 10, "probability": 0.18},
                {"token_id": 11, "probability": 0.12},
                {"token_id": 12, "probability": 0.04},
            ],
            "actual_residual_position": actual_y,
            "main_accepted": actual_y is None,
            "recovery_stage": "p1" if recovery_called else "none",
            "recovery_call_occurred": recovery_called,
            "recovery_ms": recovery_ms if recovery_called else None,
            "recovery_value_label_valid_for_w1": True,
            "recovery_value_observed_ms_by_candidate": [
                {"token_id": token, "observed_value_ms": recovery_ms if actual_y == token else 0.0}
                for token in (10, 11, 12)
            ],
            "shape_cost_ms_by_k": {"0": 5.0, "1": 5.5, "2": 7.5},
            "extra_shape_cost_ms_by_k": {"1": 0.5, "2": 2.5},
        },
    }


def _self_test() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source_index in range(8):
        source = f"source-{source_index:02d}"
        for context_index in range(8):
            record = _synthetic_record(source, context_index)
            row = {
                "source_id": record["source_id"],
                "scene_id": record["scene_id"],
                "context_id": record["context_id"],
                "seed": record["seed"],
                "round_id": record["round_id"],
                "features": record["round_context"],
                "candidates": record["round_context"]["candidates"],
                "labels": record["labels"],
                "selected_backup_ids": [],
                "diagnostic_only": False,
            }
            rows.append(row)
            # Timer repetitions must not duplicate stochastic training labels.
            repeat = json.loads(json.dumps(row))
            repeat["timing_repeat"] = 1
            repeat["labels"]["shape_cost_ms_by_k"] = {"1": 0.6, "2": 3.1}
            rows.append(repeat)
    model, report = fit_from_records(
        rows,
        validation_fraction=0.25,
        split_seed=42,
        ridge_alpha=1e-4,
        max_backups=1,
    )
    if set(model["split"]["train_sources"]) & set(model["split"]["validation_sources"]):
        raise AssertionError("source split leaked")
    sample = _synthetic_record("fresh-source", 3)["round_context"]
    decision = choose_backups(model, sample, mode="net_value", max_backups=1)
    if len(decision.backup_ids) > 1:
        raise AssertionError("default selector exceeded one backup")
    selector_value = make_selector(model, "value")
    selector_mass = make_selector(model, "mass")
    selector_ranker = make_selector(model, "ranker_same_k")
    selected_value = selector_value(sample)
    selected_mass = selector_mass(sample)
    selected_ranker = selector_ranker(sample)
    if len(selected_value) > 1 or len(selected_mass) > 1 or len(selected_ranker) > 1:
        raise AssertionError("runtime selectors exceeded their configured action count")
    if selector_ranker.last_decision is None or selector_ranker.last_decision.mode != "ranker_same_k":
        raise AssertionError("ranker_same_k did not report its control action")
    if selector_value.stats()["backup_count_counts"] == {}:
        raise AssertionError("selector action-count accounting is empty")
    no_measured_shape = dict(next(s for s in _aggregate_records(rows, position_vocab_size=8000, eos_id=12096)[0] if s["identity"]["source_id"] in model["split"]["validation_sources"]))
    no_measured_shape["shape_costs"] = {}
    no_cost_evaluation = _evaluate_validation([no_measured_shape], model, max_backups=1)
    oracle = no_cost_evaluation["policies"]["offline_oracle"]
    if oracle["status"] != "not_estimable" or oracle["source_equal_realized_label_hindsight_net_ms_measured_cost_only"] is not None:
        raise AssertionError("measured-cost hindsight reference must stay unavailable without same-context cost")
    if no_cost_evaluation["policies"]["direct_recovery_value"]["shape_cost_usage_contexts"]["predicted_shape_cost"] != 1:
        raise AssertionError("ordinary policy report must identify predicted shape cost use")
    altered = dict(sample)
    altered["labels"] = {"target_mass_by_token": {"11": 1.0}, "accepted": True}
    if choose_backups(model, altered, mode="net_value", max_backups=1).backup_ids != decision.backup_ids:
        raise AssertionError("labels changed online selection")
    if report["validation"]["validation_contexts"] <= 0:
        raise AssertionError("source-held-out report is empty")
    stats = {
        "passed": True,
        "default_decision": decision.as_dict(),
        "selectors": {
            "value": selected_value,
            "mass": selected_mass,
            "ranker_same_k": selected_ranker,
            "value_stats": selector_value.stats(),
        },
        "split": model["split"],
        "validation": report["validation"],
    }
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="single-round JSONL with separated round_context and labels")
    parser.add_argument("--out", help="serialized CPU model JSON")
    parser.add_argument("--report", help="diagnostics JSON; defaults to OUT.report.json")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20261023)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--max-backups", type=int, default=1)
    parser.add_argument("--position-vocab-size", type=int, default=8000)
    parser.add_argument("--eos-id", type=int, default=12096)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        print(json.dumps(_self_test(), ensure_ascii=False, indent=2, sort_keys=True))
        if not args.input:
            return 0
    if not args.input or not args.out:
        parser.error("--input and --out are required unless --self-test is used alone")
    raw_rows = _records_from_jsonl(args.input)
    model, report = fit_from_records(
        raw_rows,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        ridge_alpha=args.ridge_alpha,
        max_backups=args.max_backups,
        position_vocab_size=args.position_vocab_size,
        eos_id=args.eos_id,
    )
    save_model(model, args.out)
    report_path = args.report or f"{args.out}.report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"model": args.out, "report": report_path, "summary": report}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

