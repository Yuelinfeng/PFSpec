"""Portable probability estimates from information available before verification."""
from __future__ import annotations

import math


FEATURES = (
    "score", "main_score", "score_margin", "rank", "candidate_count",
    "previous_gap", "main_gap", "position", "cache_pos", "remaining",
)


def feature_vector(context, candidate):
    score = float(candidate.get("ranker_score") or 0.0)
    main_score = float(context.get("main_score") or 0.0)
    position = candidate.get("position_rank")
    previous = context.get("previous_position_rank")
    main_position = context.get("main_position_rank")
    gap = candidate.get("rank_gap")
    if gap is None and position is not None and previous is not None:
        gap = float(position) - float(previous)
    values = [
        score, main_score, main_score - score,
        float(candidate.get("ranker_rank", 0)),
        float(len(context.get("candidates", context.get("candidate_pool", ())))),
        math.log1p(max(0.0, float(gap or 0.0))),
        math.log1p(abs(float(position or 0.0) - float(main_position or 0.0))),
        float(position or 0.0) / 8000.0,
        float(context.get("cache_pos", context.get("logical_cache_pos", 0))),
        float(context.get("remaining_budget", 0)),
    ]
    if any(not math.isfinite(x) for x in values):
        raise ValueError("Probability inputs must be finite")
    return values


def predict_raw(model, values):
    result = float(model["initial"])
    rate = float(model["learning_rate"])
    for tree in model["trees"]:
        node = 0
        while tree["left"][node] != -1:
            node = (tree["left"][node] if values[tree["feature"][node]] <=
                    tree["threshold"][node] else tree["right"][node])
        result += rate * tree["value"][node]
    return result


def predict(model, context, candidate):
    return min(1.0, max(0.0, predict_raw(model, feature_vector(context, candidate))))
