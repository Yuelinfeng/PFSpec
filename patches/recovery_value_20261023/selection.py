"""CPU-only recovery-branch selection from pre-validation inputs.

The callable interface returns backup token IDs for the decoder. Training labels
are read only by ``fit_selection.py`` and never enter this module's feature path.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import math
import time
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
DEFAULT_POSITION_VOCAB = 8000
DEFAULT_EOS_ID = 12096
MAX_ENUM_CANDIDATES = 8
MAX_BACKUPS = 4

CANDIDATE_FEATURES = (
    "bias",
    "log_prefix_len",
    "log_remaining_budget",
    "log_cache_pos",
    "log_read_bucket",
    "graph_exists",
    "graph_cold",
    "log_candidate_count",
    "ranker_rank_fraction",
    "ranker_score",
    "ranker_score_margin",
    "position_rank_fraction",
    "position_rank_missing",
    "log_previous_position_gap",
    "position_gap_missing",
    "log_main_position_gap",
    "main_position_gap_missing",
)

SHAPE_COST_FEATURES = (
    "bias",
    "log_prefix_len",
    "log_cache_pos",
    "log_read_bucket",
    "graph_exists",
    "graph_cold",
    "log_candidate_count",
    "backup_count",
    "backup_count_sq",
    "backup_count_x_log_bucket",
    "backup_count_x_cold",
)

RECOVERY_COST_FEATURES = (
    "bias",
    "log_prefix_len",
    "log_remaining_budget",
    "log_cache_pos",
    "log_read_bucket",
    "graph_exists",
    "graph_cold",
    "log_candidate_count",
)


class SelectionInputError(ValueError):
    pass


@dataclass(frozen=True)
class SelectionDecision:
    backup_ids: tuple[int, ...]
    mode: str
    predicted_mass: float
    predicted_recovery_value_ms: float
    predicted_shape_cost_ms: float
    predicted_net_value_ms: float
    evaluated_subsets: int
    candidate_truncation: int
    ranker_prefix_matches: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "backup_ids": list(self.backup_ids),
            "mode": self.mode,
            "predicted_mass": self.predicted_mass,
            "predicted_recovery_value_ms": self.predicted_recovery_value_ms,
            "predicted_shape_cost_ms": self.predicted_shape_cost_ms,
            "predicted_net_value_ms": self.predicted_net_value_ms,
            "evaluated_subsets": self.evaluated_subsets,
            "candidate_truncation": self.candidate_truncation,
            "ranker_prefix_matches": self.ranker_prefix_matches,
        }


def _finite(value: Any, field: str, *, default: float | None = None) -> float:
    if value is None and default is not None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SelectionInputError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise SelectionInputError(f"{field} must be finite")
    return result


def _nonnegative_int(value: Any, field: str, *, default: int = 0) -> int:
    if value is None:
        return default
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SelectionInputError(f"{field} must be an integer") from error
    if isinstance(value, bool) or result != value or result < 0:
        raise SelectionInputError(f"{field} must be a nonnegative integer")
    return result


def _context(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise SelectionInputError("round_context must be a mapping")
    nested = raw.get("round_context")
    if isinstance(nested, Mapping):
        return nested
    nested = raw.get("features")
    if isinstance(nested, Mapping):
        return nested
    return raw


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _log1p(value: Any, field: str, *, default: float = 0.0) -> float:
    number = _finite(value, field, default=default)
    if number < 0:
        raise SelectionInputError(f"{field} must be nonnegative")
    return math.log1p(number)


def _candidate_rows(raw: Mapping[str, Any], model: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    ctx = _context(raw)
    raw_candidates = ctx.get("candidates", ctx.get("candidate_pool"))
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        raise SelectionInputError("round_context.candidates must be a sequence")

    config = model.get("config", {})
    position_vocab = _nonnegative_int(
        ctx.get("position_vocab_size", config.get("position_vocab_size", DEFAULT_POSITION_VOCAB)),
        "position_vocab_size",
        default=DEFAULT_POSITION_VOCAB,
    )
    eos_id = _nonnegative_int(
        ctx.get("eos_id", config.get("eos_id", DEFAULT_EOS_ID)), "eos_id", default=DEFAULT_EOS_ID
    )
    main_value = ctx.get("main_position", ctx.get("main_token_id"))
    main_id = None if main_value is None else _nonnegative_int(main_value, "main_position")
    previous_rank_value = ctx.get("previous_position_rank")
    previous_rank = None if previous_rank_value is None else _finite(previous_rank_value, "previous_position_rank")
    main_score_value = ctx.get("main_score")
    main_rank_value = ctx.get("main_position_rank")

    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, Mapping):
            raise SelectionInputError("each candidate must be a mapping")
        token_value = item.get("token_id", item.get("position_id"))
        if token_value is None:
            raise SelectionInputError("each candidate needs token_id")
        token_id = _nonnegative_int(token_value, "candidate.token_id")
        if token_id in seen:
            raise SelectionInputError("candidate token IDs must be unique")
        seen.add(token_id)
        rank_value = item.get("ranker_rank", item.get("rank", index))
        rank = _nonnegative_int(rank_value, "candidate.ranker_rank", default=index)
        score = _finite(item.get("ranker_score", item.get("score")), "candidate.ranker_score", default=0.0)
        pos_rank_value = item.get("position_rank", item.get("rank_value"))
        pos_rank = None if pos_rank_value is None else _finite(pos_rank_value, "candidate.position_rank")
        if token_id == main_id:
            if main_score_value is None:
                main_score_value = score
            if main_rank_value is None and pos_rank is not None:
                main_rank_value = pos_rank
        if token_id == main_id or token_id == eos_id or token_id >= position_vocab:
            continue
        if item.get("legal") is False or item.get("is_eos") is True:
            continue
        gap_value = item.get("rank_gap")
        if gap_value is None and pos_rank is not None and previous_rank is not None:
            gap_value = pos_rank - previous_rank
        rank_gap = None if gap_value is None else _finite(gap_value, "candidate.rank_gap")
        rows.append({
            "token_id": token_id,
            "ranker_rank": rank,
            "ranker_score": score,
            "position_rank": pos_rank,
            "rank_gap": rank_gap,
            "raw": item,
        })

    rows.sort(key=lambda row: (row["ranker_rank"], row["token_id"]))
    cap = min(MAX_ENUM_CANDIDATES, _nonnegative_int(config.get("max_enum_candidates", MAX_ENUM_CANDIDATES), "max_enum_candidates", default=MAX_ENUM_CANDIDATES))
    cap = max(1, cap)
    truncated = max(0, len(rows) - cap)
    rows = rows[:cap]
    graph_state = ctx.get("graph_state", {})
    if not isinstance(graph_state, Mapping):
        graph_state = {}
    base_graph = graph_state.get("0", {})
    if not isinstance(base_graph, Mapping):
        base_graph = {}
    cache_pos_value = ctx.get(
        "cache_pos", ctx.get("cache_pos_before", ctx.get("logical_cache_pos"))
    )
    read_bucket_value = ctx.get("read_bucket", ctx.get("tree_read_bucket", base_graph.get("read_bucket")))
    graph_exists_value = ctx.get("graph_exists", ctx.get("graph_present", base_graph.get("graph_exists", False)))
    graph_cold_value = ctx.get(
        "graph_cold", ctx.get("cold_graph", ctx.get("runtime_created", not bool(graph_exists_value)))
    )
    normalized_context = {
        "prefix_len": _nonnegative_int(ctx.get("prefix_len"), "prefix_len"),
        "remaining_budget": _nonnegative_int(ctx.get("remaining_budget"), "remaining_budget"),
        "cache_pos": _nonnegative_int(cache_pos_value, "cache_pos"),
        "read_bucket": _nonnegative_int(read_bucket_value, "read_bucket"),
        "graph_exists": bool(graph_exists_value),
        "graph_cold": bool(graph_cold_value),
        "graph_state": {str(key): dict(value) for key, value in graph_state.items() if isinstance(value, Mapping)},
        "previous_position_rank": previous_rank,
        "main_position": main_id,
        "main_position_rank": None if main_rank_value is None else _finite(main_rank_value, "main_position_rank"),
        "main_score": 0.0 if main_score_value is None else _finite(main_score_value, "main_score"),
        "candidate_count": len(raw_candidates),
        "position_vocab_size": max(1, position_vocab),
    }
    return normalized_context, rows, truncated


def candidate_feature_values(ctx: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    candidates = int(ctx["candidate_count"])
    rank = int(candidate["ranker_rank"])
    score = float(candidate["ranker_score"])
    main_score = float(ctx.get("main_score", score))
    pos_rank = candidate.get("position_rank")
    prev_rank = ctx.get("previous_position_rank")
    rank_gap = candidate.get("rank_gap")
    main_rank_value = ctx.get("main_position_rank")
    main_gap_missing = pos_rank is None or ctx.get("main_position") is None or main_rank_value is None
    main_gap = 0.0 if main_gap_missing else abs(float(pos_rank) - float(main_rank_value))
    return {
        "bias": 1.0,
        "log_prefix_len": _log1p(ctx["prefix_len"], "prefix_len"),
        "log_remaining_budget": _log1p(ctx["remaining_budget"], "remaining_budget"),
        "log_cache_pos": _log1p(ctx["cache_pos"], "cache_pos"),
        "log_read_bucket": _log1p(ctx["read_bucket"], "read_bucket"),
        "graph_exists": _bool(ctx["graph_exists"]),
        "graph_cold": _bool(ctx["graph_cold"]),
        "log_candidate_count": math.log1p(max(0, candidates)),
        "ranker_rank_fraction": rank / max(1, candidates - 1),
        "ranker_score": score,
        "ranker_score_margin": max(0.0, main_score - score),
        "position_rank_fraction": 0.0 if pos_rank is None else float(pos_rank) / ctx["position_vocab_size"],
        "position_rank_missing": _bool(pos_rank is None),
        "log_previous_position_gap": 0.0 if rank_gap is None else math.log1p(max(0.0, float(rank_gap))),
        "position_gap_missing": _bool(rank_gap is None),
        "log_main_position_gap": math.log1p(main_gap),
        "main_position_gap_missing": _bool(main_gap_missing),
    }


def shape_cost_feature_values(ctx: Mapping[str, Any], backup_count: int) -> dict[str, float]:
    count = int(backup_count)
    action_context = _context_for_backup_count(ctx, count)
    log_bucket = _log1p(action_context["read_bucket"], "read_bucket")
    return {
        "bias": 1.0,
        "log_prefix_len": _log1p(action_context["prefix_len"], "prefix_len"),
        "log_cache_pos": _log1p(action_context["cache_pos"], "cache_pos"),
        "log_read_bucket": log_bucket,
        "graph_exists": _bool(action_context["graph_exists"]),
        "graph_cold": _bool(action_context["graph_cold"]),
        "log_candidate_count": math.log1p(max(0, int(action_context["candidate_count"]))),
        "backup_count": float(count),
        "backup_count_sq": float(count * count),
        "backup_count_x_log_bucket": count * log_bucket,
        "backup_count_x_cold": count * _bool(action_context["graph_cold"]),
    }


def _context_for_backup_count(ctx: Mapping[str, Any], backup_count: int) -> dict[str, Any]:
    """Use the pre-observed graph shape for this action when available."""
    result = dict(ctx)
    graph_state = ctx.get("graph_state", {})
    entry = graph_state.get(str(int(backup_count)), {}) if isinstance(graph_state, Mapping) else {}
    if isinstance(entry, Mapping):
        if entry.get("read_bucket") is not None:
            result["read_bucket"] = int(entry["read_bucket"])
        if entry.get("graph_exists") is not None:
            result["graph_exists"] = bool(entry["graph_exists"])
            result["graph_cold"] = not bool(entry["graph_exists"])
    return result


def recovery_cost_feature_values(ctx: Mapping[str, Any]) -> dict[str, float]:
    return {
        "bias": 1.0,
        "log_prefix_len": _log1p(ctx["prefix_len"], "prefix_len"),
        "log_remaining_budget": _log1p(ctx["remaining_budget"], "remaining_budget"),
        "log_cache_pos": _log1p(ctx["cache_pos"], "cache_pos"),
        "log_read_bucket": _log1p(ctx["read_bucket"], "read_bucket"),
        "graph_exists": _bool(ctx["graph_exists"]),
        "graph_cold": _bool(ctx["graph_cold"]),
        "log_candidate_count": math.log1p(max(0, int(ctx["candidate_count"]))),
    }


def _predict(model: Mapping[str, Any] | None, values: Mapping[str, float], *, default: float = 0.0) -> float:
    if not model:
        return float(default)
    names = model.get("feature_names", ())
    means = model.get("means", {})
    scales = model.get("scales", {})
    weights = model.get("weights", {})
    result = 0.0
    for name in names:
        scale = float(scales.get(name, 1.0))
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0
        x = (float(values.get(name, 0.0)) - float(means.get(name, 0.0))) / scale
        result += float(weights.get(name, 0.0)) * x
    return result


def predict_candidate(model: Mapping[str, Any], context: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    values = candidate_feature_values(context, candidate)
    models = model.get("models", {})
    mass = _predict(models.get("residual_mass"), values)
    recovery_value = _predict(models.get("recovery_value_ms"), values)
    recovery_cost = _predict(models.get("recovery_cost_ms"), recovery_cost_feature_values(context))
    return {
        "residual_mass": min(1.0, max(0.0, mass)),
        "recovery_value_ms": max(0.0, recovery_value),
        "recovery_cost_ms": max(0.0, recovery_cost),
        "mass_times_cost_ms": max(0.0, mass) * max(0.0, recovery_cost),
    }


def predict_shape_cost(model: Mapping[str, Any], context: Mapping[str, Any], backup_count: int) -> float:
    if backup_count <= 0:
        return 0.0
    cost_model = model.get("models", {}).get("shape_cost_ms")
    predicted = _predict(cost_model, shape_cost_feature_values(context, backup_count))
    return predicted


@dataclass(frozen=True)
class _CompiledLinear:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    feature_indices: tuple[int, ...]

    @classmethod
    def build(
        cls,
        model: Mapping[str, Any] | None,
        feature_names: Sequence[str],
    ) -> "_CompiledLinear":
        if not model:
            return cls((), (), (), ())
        names = tuple(model.get("feature_names", ()))
        means = model.get("means", {})
        scales = model.get("scales", {})
        weights = model.get("weights", {})
        indexes = {name: index for index, name in enumerate(feature_names)}
        result_means = []
        result_scales = []
        result_weights = []
        result_indexes = []
        for name in names:
            scale = float(scales.get(name, 1.0))
            if not math.isfinite(scale) or scale <= 0:
                scale = 1.0
            result_means.append(float(means.get(name, 0.0)))
            result_scales.append(scale)
            result_weights.append(float(weights.get(name, 0.0)))
            result_indexes.append(indexes.get(name, -1))
        return cls(
            tuple(result_means), tuple(result_scales), tuple(result_weights),
            tuple(result_indexes),
        )

    def predict(self, values: Sequence[float]) -> float:
        result = 0.0
        for mean, scale, weight, index in zip(
            self.means, self.scales, self.weights, self.feature_indices
        ):
            value = float(values[index]) if index >= 0 else 0.0
            x = (value - mean) / scale
            result += weight * x
        return result


@dataclass(frozen=True)
class _CompiledModels:
    residual_mass: _CompiledLinear
    recovery_value: _CompiledLinear
    recovery_cost: _CompiledLinear
    shape_cost: _CompiledLinear

    @classmethod
    def build(cls, model: Mapping[str, Any]) -> "_CompiledModels":
        models = model.get("models", {})
        return cls(
            _CompiledLinear.build(models.get("residual_mass"), CANDIDATE_FEATURES),
            _CompiledLinear.build(models.get("recovery_value_ms"), CANDIDATE_FEATURES),
            _CompiledLinear.build(models.get("recovery_cost_ms"), RECOVERY_COST_FEATURES),
            _CompiledLinear.build(models.get("shape_cost_ms"), SHAPE_COST_FEATURES),
        )


def _candidate_feature_tuple(
    context: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[float, ...]:
    candidate_count = int(context["candidate_count"])
    rank = int(candidate["ranker_rank"])
    score = float(candidate["ranker_score"])
    main_score = float(context.get("main_score", score))
    position_rank = candidate.get("position_rank")
    previous_rank = context.get("previous_position_rank")
    rank_gap = candidate.get("rank_gap")
    main_rank = context.get("main_position_rank")
    main_gap_missing = (
        position_rank is None or context.get("main_position") is None or main_rank is None
    )
    main_gap = 0.0 if main_gap_missing else abs(float(position_rank) - float(main_rank))
    return (
        1.0,
        _log1p(context["prefix_len"], "prefix_len"),
        _log1p(context["remaining_budget"], "remaining_budget"),
        _log1p(context["cache_pos"], "cache_pos"),
        _log1p(context["read_bucket"], "read_bucket"),
        _bool(context["graph_exists"]),
        _bool(context["graph_cold"]),
        math.log1p(max(0, candidate_count)),
        rank / max(1, candidate_count - 1),
        score,
        max(0.0, main_score - score),
        0.0 if position_rank is None else float(position_rank) / context["position_vocab_size"],
        _bool(position_rank is None),
        0.0 if rank_gap is None else math.log1p(max(0.0, float(rank_gap))),
        _bool(rank_gap is None),
        math.log1p(main_gap),
        _bool(main_gap_missing),
    )


def _recovery_cost_feature_tuple(context: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        1.0,
        _log1p(context["prefix_len"], "prefix_len"),
        _log1p(context["remaining_budget"], "remaining_budget"),
        _log1p(context["cache_pos"], "cache_pos"),
        _log1p(context["read_bucket"], "read_bucket"),
        _bool(context["graph_exists"]),
        _bool(context["graph_cold"]),
        math.log1p(max(0, int(context["candidate_count"]))),
    )


def _shape_cost_feature_tuple(
    context: Mapping[str, Any], backup_count: int
) -> tuple[float, ...]:
    count = int(backup_count)
    graph_state = context.get("graph_state", {})
    entry = graph_state.get(str(count), {}) if isinstance(graph_state, Mapping) else {}
    read_bucket = context["read_bucket"]
    graph_exists = context["graph_exists"]
    graph_cold = context["graph_cold"]
    if isinstance(entry, Mapping):
        if entry.get("read_bucket") is not None:
            read_bucket = int(entry["read_bucket"])
        if entry.get("graph_exists") is not None:
            graph_exists = bool(entry["graph_exists"])
            graph_cold = not graph_exists
    log_bucket = _log1p(read_bucket, "read_bucket")
    return (
        1.0,
        _log1p(context["prefix_len"], "prefix_len"),
        _log1p(context["cache_pos"], "cache_pos"),
        log_bucket,
        _bool(graph_exists),
        _bool(graph_cold),
        math.log1p(max(0, int(context["candidate_count"]))),
        float(count),
        float(count * count),
        count * log_bucket,
        count * _bool(graph_cold),
    )


def choose_backups_compiled(
    model: Mapping[str, Any],
    round_context: Mapping[str, Any],
    *,
    mode: str = "net_value",
    max_backups: int | None = None,
    fixed_k: int | None = None,
    compiled_models: _CompiledModels | None = None,
) -> SelectionDecision:
    """Compiled CPU path; ``choose_backups`` remains the reference path."""
    if mode not in {"net_value", "mass_times_cost", "mass_topk", "value_topk", "ranker_topk", "ranker_same_k"}:
        raise ValueError(f"unsupported selection mode: {mode}")
    context, candidates, truncated = _candidate_rows(round_context, model)
    config = model.get("config", {})
    trained_max = _nonnegative_int(config.get("max_backups", 1), "config.max_backups", default=1)
    measured_max = _nonnegative_int(
        config.get("max_measured_backups", trained_max),
        "config.max_measured_backups",
        default=trained_max,
    )
    cap = min(MAX_BACKUPS, trained_max, measured_max)
    if max_backups is not None:
        cap = min(cap, _nonnegative_int(max_backups, "max_backups"))
    cap = min(cap, len(candidates))
    supported_k = config.get("supported_k", ())
    if isinstance(supported_k, Sequence) and not isinstance(supported_k, (str, bytes)):
        supported = {int(value) for value in supported_k if int(value) >= 0}
    else:
        supported = set(range(cap + 1))
    supported.add(0)
    graph_state = context.get("graph_state", {})
    available_k = {
        k for k in range(cap + 1)
        if k in supported and (
            not isinstance(graph_state, Mapping)
            or str(k) not in graph_state
            or bool(graph_state[str(k)].get("candidate_available", True))
        )
    }
    cap = max(available_k, default=0)
    if fixed_k is not None:
        fixed_k = _nonnegative_int(fixed_k, "fixed_k")
        if fixed_k > cap or fixed_k not in available_k:
            raise ValueError("fixed_k exceeds the enabled or measured backup count")

    if not candidates:
        return SelectionDecision((), mode, 0.0, 0.0, 0.0, 0.0, 1, truncated, True)

    models = compiled_models or _CompiledModels.build(model)
    candidate_predictions: list[tuple[int, int, float, float, float]] = []
    recovery_cost_values = max(0.0, models.recovery_cost.predict(_recovery_cost_feature_tuple(context)))
    for row in candidates:
        features = _candidate_feature_tuple(context, row)
        raw_mass = models.residual_mass.predict(features)
        mass = min(1.0, max(0.0, raw_mass))
        value = max(0.0, models.recovery_value.predict(features))
        mass_times_cost = max(0.0, raw_mass) * recovery_cost_values
        candidate_predictions.append((
            int(row["token_id"]), int(row["ranker_rank"]), mass, value, mass_times_cost
        ))

    shape_costs = {
        count: models.shape_cost.predict(_shape_cost_feature_tuple(context, count))
        for count in available_k if count > 0
    }
    ranker_order = list(range(len(candidates)))

    def best_subset(selection_mode: str) -> tuple[tuple[float, int, tuple[int, ...], float, float, float], int]:
        best: tuple[float, int, tuple[int, ...], float, float, float] = (0.0, 0, (), 0.0, 0.0, 0.0)
        evaluated = 1
        for count in range(1, cap + 1):
            if count not in available_k:
                continue
            cost = shape_costs[count]
            for subset in combinations(ranker_order, count):
                evaluated += 1
                mass_sum = sum(candidate_predictions[index][2] for index in subset)
                if selection_mode == "mass_times_cost":
                    value_sum = sum(candidate_predictions[index][4] for index in subset)
                else:
                    value_sum = sum(candidate_predictions[index][3] for index in subset)
                net = value_sum - cost
                subset_rank_key = tuple(candidate_predictions[index][1] for index in subset)
                best_rank_key = tuple(candidate_predictions[index][1] for index in best[2])
                if net > best[0] + 1e-12 or (
                    abs(net - best[0]) <= 1e-12
                    and (count < best[1] or (count == best[1] and subset_rank_key < best_rank_key))
                ):
                    best = (net, count, subset, mass_sum, value_sum, cost)
        return best, evaluated

    if mode == "ranker_same_k":
        direct_best, evaluated = best_subset("net_value")
        selected_indices = tuple(ranker_order[:direct_best[1]])
        selected = tuple(candidate_predictions[index][0] for index in selected_indices)
        mass_sum = sum(candidate_predictions[index][2] for index in selected_indices)
        value_sum = sum(candidate_predictions[index][3] for index in selected_indices)
        cost = 0.0 if not selected_indices else shape_costs[len(selected_indices)]
        return SelectionDecision(
            selected, mode, mass_sum, value_sum, cost, value_sum - cost,
            evaluated, truncated, True,
        )

    if mode == "ranker_topk":
        count = cap if fixed_k is None else fixed_k
        selected_indices = tuple(ranker_order[:count])
        selected = tuple(candidate_predictions[index][0] for index in selected_indices)
        mass_sum = sum(candidate_predictions[index][2] for index in selected_indices)
        value_sum = sum(candidate_predictions[index][3] for index in selected_indices)
        cost = 0.0 if not selected_indices else shape_costs[len(selected_indices)]
        return SelectionDecision(
            selected, mode, mass_sum, value_sum, cost, value_sum - cost,
            1, truncated, True,
        )

    if mode in {"mass_topk", "value_topk"}:
        count = cap if fixed_k is None else fixed_k
        score_index = 2 if mode == "mass_topk" else 3
        ordered = sorted(
            ranker_order,
            key=lambda index: (
                -candidate_predictions[index][score_index],
                candidate_predictions[index][1],
                candidate_predictions[index][0],
            ),
        )
        selected_indices = tuple(ordered[:count])
        selected = tuple(candidate_predictions[index][0] for index in selected_indices)
        mass_sum = sum(candidate_predictions[index][2] for index in selected_indices)
        value_sum = sum(candidate_predictions[index][3] for index in selected_indices)
        cost = 0.0 if not selected_indices else shape_costs[len(selected_indices)]
        ranker_prefix = tuple(candidate_predictions[index][0] for index in ranker_order[:len(selected_indices)])
        same_as_ranker = set(selected) == set(ranker_prefix)
        return SelectionDecision(
            selected, mode, mass_sum, value_sum, cost, value_sum - cost,
            1, truncated, same_as_ranker,
        )

    best, evaluated = best_subset(mode)
    selected = tuple(candidate_predictions[index][0] for index in best[2])
    ranker_prefix = tuple(candidate_predictions[index][0] for index in ranker_order[:len(best[2])])
    same_as_ranker = set(selected) == set(ranker_prefix)
    return SelectionDecision(
        selected, mode, best[3], best[4], best[5], best[0],
        evaluated, truncated, same_as_ranker,
    )


def choose_backups(
    model: Mapping[str, Any],
    round_context: Mapping[str, Any],
    *,
    mode: str = "net_value",
    max_backups: int | None = None,
    fixed_k: int | None = None,
) -> SelectionDecision:
    """Choose legal backup IDs using only round_context and the saved CPU model.

    Modes: ``net_value`` directly predicts E[1_hit * recovery_ms];
    ``mass_times_cost`` predicts residual mass and recovery cost separately;
    ``mass_topk`` chooses by predicted residual mass without cost gating;
    ``value_topk`` chooses by direct recovery value without cost gating;
    ``ranker_topk`` is a fixed-k Ranker action; ``ranker_same_k`` uses the
    direct-value policy to choose k, then returns the Ranker prefix of that size.
    """
    if mode not in {"net_value", "mass_times_cost", "mass_topk", "value_topk", "ranker_topk", "ranker_same_k"}:
        raise ValueError(f"unsupported selection mode: {mode}")
    context, candidates, truncated = _candidate_rows(round_context, model)
    config = model.get("config", {})
    trained_max = _nonnegative_int(config.get("max_backups", 1), "config.max_backups", default=1)
    measured_max = _nonnegative_int(config.get("max_measured_backups", trained_max), "config.max_measured_backups", default=trained_max)
    cap = min(MAX_BACKUPS, trained_max, measured_max)
    if max_backups is not None:
        cap = min(cap, _nonnegative_int(max_backups, "max_backups"))
    cap = min(cap, len(candidates))
    supported_k = config.get("supported_k", ())
    if isinstance(supported_k, Sequence) and not isinstance(supported_k, (str, bytes)):
        supported = {int(value) for value in supported_k if int(value) >= 0}
    else:
        supported = set(range(cap + 1))
    supported.add(0)
    graph_state = context.get("graph_state", {})
    available_k = {
        k for k in range(cap + 1)
        if k in supported and (
            not isinstance(graph_state, Mapping)
            or str(k) not in graph_state
            or bool(graph_state[str(k)].get("candidate_available", True))
        )
    }
    cap = max(available_k, default=0)
    if fixed_k is not None:
        fixed_k = _nonnegative_int(fixed_k, "fixed_k")
        if fixed_k > cap or fixed_k not in available_k:
            raise ValueError("fixed_k exceeds the enabled or measured backup count")

    if "main_score" not in context:
        raw_ctx = _context(round_context)
        main_score = raw_ctx.get("main_score")
        if main_score is None:
            main_id = context.get("main_position")
            match = next((row for row in candidates if row["token_id"] == main_id), None)
            main_score = match["ranker_score"] if match else (candidates[0]["ranker_score"] if candidates else 0.0)
        context["main_score"] = _finite(main_score, "main_score", default=0.0)

    predictions = {
        row["token_id"]: predict_candidate(model, context, row)
        for row in candidates
    }
    ranker_order = list(candidates)

    def best_subset(selection_mode: str) -> tuple[tuple[float, int, tuple[int, ...], float, float, float], int]:
        best: tuple[float, int, tuple[int, ...], float, float, float] = (0.0, 0, (), 0.0, 0.0, 0.0)
        evaluated = 1
        rank_by_token = {row["token_id"]: row["ranker_rank"] for row in candidates}
        for count in range(1, cap + 1):
            if count not in available_k:
                continue
            for subset in combinations(candidates, count):
                evaluated += 1
                ids = tuple(row["token_id"] for row in subset)
                candidate_predictions = [predictions[token] for token in ids]
                mass_sum = sum(item["residual_mass"] for item in candidate_predictions)
                if selection_mode == "mass_times_cost":
                    value_sum = sum(item["mass_times_cost_ms"] for item in candidate_predictions)
                else:
                    value_sum = sum(item["recovery_value_ms"] for item in candidate_predictions)
                cost = predict_shape_cost(model, context, count)
                net = value_sum - cost
                subset_rank_key = tuple(rank_by_token[token] for token in ids)
                best_rank_key = tuple(rank_by_token[token] for token in best[2])
                if net > best[0] + 1e-12 or (
                    abs(net - best[0]) <= 1e-12
                    and (count < best[1] or (count == best[1] and subset_rank_key < best_rank_key))
                ):
                    best = (net, count, ids, mass_sum, value_sum, cost)
        return best, evaluated

    if mode == "ranker_same_k":
        direct_best, evaluated = best_subset("net_value")
        selected = tuple(row["token_id"] for row in ranker_order[:direct_best[1]])
        candidate_predictions = [predictions[token] for token in selected]
        mass_sum = sum(item["residual_mass"] for item in candidate_predictions)
        value_sum = sum(item["recovery_value_ms"] for item in candidate_predictions)
        cost = predict_shape_cost(model, context, len(selected))
        return SelectionDecision(
            selected, mode, mass_sum, value_sum, cost, value_sum - cost,
            evaluated, truncated, True,
        )
    if mode == "ranker_topk":
        k = cap if fixed_k is None else fixed_k
        chosen = ranker_order[:k]
        selected = tuple(row["token_id"] for row in chosen)
        chosen_prediction = [predictions[token] for token in selected]
        mass_sum = sum(item["residual_mass"] for item in chosen_prediction)
        value_sum = sum(item["recovery_value_ms"] for item in chosen_prediction)
        cost = predict_shape_cost(model, context, len(selected))
        return SelectionDecision(selected, mode, mass_sum, value_sum, cost, value_sum - cost, 1, truncated, True)

    metric = {
        "mass_topk": "residual_mass",
        "value_topk": "recovery_value_ms",
        "mass_times_cost": "mass_times_cost_ms",
        "net_value": "recovery_value_ms",
    }[mode]
    if mode in {"mass_topk", "value_topk"}:
        k = cap if fixed_k is None else fixed_k
        ordered = sorted(candidates, key=lambda row: (-predictions[row["token_id"]][metric], row["ranker_rank"], row["token_id"]))
        selected = tuple(row["token_id"] for row in ordered[:k])
        chosen_prediction = [predictions[token] for token in selected]
        mass_sum = sum(item["residual_mass"] for item in chosen_prediction)
        value_sum = sum(item["recovery_value_ms"] for item in chosen_prediction)
        cost = predict_shape_cost(model, context, len(selected))
        ranker_prefix = tuple(row["token_id"] for row in ranker_order[:len(selected)])
        same_as_ranker = set(selected) == set(ranker_prefix)
        return SelectionDecision(selected, mode, mass_sum, value_sum, cost, value_sum - cost, 1, truncated, same_as_ranker)

    best, evaluated = best_subset(mode)

    ranker_prefix = tuple(row["token_id"] for row in ranker_order[:len(best[2])])
    same_as_ranker = set(best[2]) == set(ranker_prefix)
    return SelectionDecision(best[2], mode, best[3], best[4], best[5], best[0], evaluated, truncated, same_as_ranker)


def make_backup_selector(
    model: Mapping[str, Any],
    *,
    mode: str = "net_value",
    max_backups: int = 1,
    fixed_k: int | None = None,
    implementation: str = "reference",
) -> "TimedBackupSelector":
    return TimedBackupSelector(
        model,
        mode=mode,
        max_backups=max_backups,
        fixed_k=fixed_k,
        implementation=implementation,
    )


def make_selector(
    metadata_dict: Mapping[str, Any] | None,
    method: str,
    *,
    implementation: str = "compiled",
) -> "TimedBackupSelector":
    """Build the callback expected by ``run_experiment.py``.

    ``value`` fits candidate subsets with the direct realized-value model;
    ``mass`` uses the separate mass-times-recovery-cost model; and
    ``ranker_same_k`` keeps the direct-value choice of k while returning the
    Ranker-ordered candidate prefix as a control.
    """
    if not isinstance(metadata_dict, Mapping):
        raise ValueError("a serialized selection model is required")
    model = metadata_dict.get("selection_model", metadata_dict)
    if not isinstance(model, Mapping) or model.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("metadata_dict does not contain a supported selection model")
    method_to_mode = {
        "value": "net_value",
        "mass": "mass_times_cost",
        "ranker_same_k": "ranker_same_k",
    }
    try:
        mode = method_to_mode[str(method)]
    except KeyError as error:
        raise ValueError(f"unsupported selector method: {method}") from error
    config = model.get("config", {})
    default_count = _nonnegative_int(
        config.get("default_max_backups", config.get("max_backups", 1)),
        "config.default_max_backups",
        default=1,
    )
    if implementation not in {"compiled", "reference"}:
        raise ValueError("implementation must be 'compiled' or 'reference'")
    return TimedBackupSelector(
        model,
        mode=mode,
        max_backups=default_count,
        implementation=implementation,
    )


class TimedBackupSelector:
    """Callback adapter; the caller's normal round timer includes its CPU work."""

    def __init__(
        self,
        model: Mapping[str, Any],
        *,
        mode: str,
        max_backups: int,
        fixed_k: int | None = None,
        implementation: str = "reference",
    ):
        if implementation not in {"compiled", "reference"}:
            raise ValueError("implementation must be 'compiled' or 'reference'")
        self.model = model
        self.mode = mode
        self.max_backups = max_backups
        self.fixed_k = fixed_k
        self.implementation = implementation
        self._compiled_models = _CompiledModels.build(model) if implementation == "compiled" else None
        self.last_decision: SelectionDecision | None = None
        self.calls = 0
        self.elapsed_ns = 0
        self.backup_count_counts: dict[str, int] = {}
        self.different_from_ranker_same_k_calls = 0

    def __call__(self, round_context: Mapping[str, Any]) -> list[int]:
        started = time.perf_counter_ns()
        try:
            chooser = choose_backups_compiled if self.implementation == "compiled" else choose_backups
            chooser_options = {
                "mode": self.mode,
                "max_backups": self.max_backups,
                "fixed_k": self.fixed_k,
            }
            if self.implementation == "compiled":
                chooser_options["compiled_models"] = self._compiled_models
            self.last_decision = chooser(
                self.model,
                round_context,
                **chooser_options,
            )
            selected = list(self.last_decision.backup_ids)
            k = len(selected)
            self.backup_count_counts[str(k)] = self.backup_count_counts.get(str(k), 0) + 1
            if not self.last_decision.ranker_prefix_matches:
                self.different_from_ranker_same_k_calls += 1
            return selected
        finally:
            self.elapsed_ns += time.perf_counter_ns() - started
            self.calls += 1

    def stats(self) -> dict[str, Any]:
        return {
            "selector_calls": self.calls,
            "selector_cpu_ms": self.elapsed_ns / 1_000_000.0,
            "selector_mean_cpu_us": (self.elapsed_ns / self.calls / 1000.0) if self.calls else 0.0,
            "backup_count_counts": dict(self.backup_count_counts),
            "different_from_ranker_same_k_calls": self.different_from_ranker_same_k_calls,
            "last_decision": None if self.last_decision is None else self.last_decision.as_dict(),
        }


def load_model(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported selection model schema_version")
    if model.get("kind") != "gaussiangpt_recovery_value_cpu_model":
        raise ValueError("file is not a recovery-value selection model")
    return model


def save_model(model: Mapping[str, Any], path: str) -> None:
    data = dict(model)
    data["schema_version"] = SCHEMA_VERSION
    data["kind"] = "gaussiangpt_recovery_value_cpu_model"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

