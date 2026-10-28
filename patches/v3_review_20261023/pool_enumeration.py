#!/usr/bin/env python3
"""Enumerate backup probability mass on the saved Position contexts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


P_VOCAB = 8000
EOS = 12096
WIDTHS = (1, 2, 3, 4)
MASS_EPS = 1e-12


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def choose_main(ids, scores):
    if not ids or len(ids) != len(scores):
        raise ValueError("ids and scores must be nonempty and aligned")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("Ranker scores must be finite")
    # max preserves the first input index when scores tie.
    return max(range(len(ids)), key=lambda index: float(scores[index]))


def legal_pool(ids, main_index):
    if len(ids) != len(set(int(token) for token in ids)):
        raise ValueError("candidate IDs must be unique")
    main_id = int(ids[main_index])
    if not 0 <= main_id < P_VOCAB or main_id == EOS:
        raise ValueError("Ranker main candidate is not a legal Position")
    return [
        index for index, token in enumerate(ids)
        if index != main_index and 0 <= int(token) < P_VOCAB and int(token) != EOS
    ]


def mass_for(policy, ids, indices):
    return math.fsum(float(policy[int(ids[index])]) for index in indices)


def enumerate_context(row, offline_row):
    ids = [int(value) for value in row["ids"]]
    scores = [float(value) for value in row["scores"]]
    policy = [float(value) for value in row["target_policy"]]
    if any(not math.isfinite(value) or value < 0.0 for value in policy):
        raise ValueError("target_policy must contain finite nonnegative values")
    if any(token < 0 or token >= len(policy) for token in ids):
        raise ValueError("candidate ID is outside target_policy")

    main_index = choose_main(ids, scores)
    pool = legal_pool(ids, main_index)
    main_id = ids[main_index]
    main_probability = float(policy[main_id])
    rejection_mass = max(0.0, 1.0 - main_probability)
    ranker_order = sorted(pool, key=lambda index: (-scores[index], index))
    probability_order = sorted(pool, key=lambda index: (-policy[ids[index]], index))

    if int(row["point"]) != main_id:
        raise ValueError("saved point differs from the original Ranker argmax")
    if offline_row is None or int(offline_row["point"]) != main_id:
        raise ValueError("offline_mass row is missing or has a different main Position")

    context = {
        "source": row["source"],
        "prefix": int(row["prefix"]),
        "crop_path": row.get("crop_path"),
        "crop_seed": row.get("crop_seed"),
        "candidate_count": len(ids),
        "legal_backup_count": len(pool),
        "main_position": main_id,
        "main_ranker_score": scores[main_index],
        "main_target_probability": main_probability,
        "main_rejection_mass": rejection_mass,
        "policy_sum": math.fsum(policy),
        "candidate_pool": [
            {
                "position": ids[index],
                "ranker_score": scores[index],
                "target_probability": float(policy[ids[index]]),
                "ranker_backup_order": ranker_order.index(index) + 1,
                "target_probability_order": probability_order.index(index) + 1,
            }
            for index in pool
        ],
        "by_width": {},
    }

    for width in WIDTHS:
        ranker_selected = ranker_order[:width]
        probability_selected = probability_order[:width]
        ranker_ids = [ids[index] for index in ranker_selected]
        probability_ids = [ids[index] for index in probability_selected]
        ranker_mass = mass_for(policy, ids, ranker_selected)
        probability_mass = mass_for(policy, ids, probability_selected)
        rejection_denominator = rejection_mass
        ranker_conditional = ranker_mass / rejection_denominator if rejection_denominator > 0.0 else 0.0
        probability_conditional = probability_mass / rejection_denominator if rejection_denominator > 0.0 else 0.0

        old_width = offline_row["widths"][str(width)]
        old_ids = [int(token) for token in old_width["backup_ids"]]
        old_mass = float(old_width["mass"])
        old_conditional = float(old_width["conditional_hit"])
        if ranker_ids != old_ids:
            raise ValueError(f"Ranker-selected IDs differ from offline_mass at width {width}")
        if len(ranker_ids) != int(old_width["available"]):
            raise ValueError(f"available count differs from offline_mass at width {width}")
        if not math.isclose(ranker_mass, old_mass, rel_tol=0.0, abs_tol=2e-7):
            raise ValueError(f"Ranker mass differs from offline_mass at width {width}")
        if not math.isclose(ranker_conditional, old_conditional, rel_tol=0.0, abs_tol=2e-7):
            raise ValueError(f"conditional mass differs from offline_mass at width {width}")

        delta = probability_mass - ranker_mass
        context["by_width"][str(width)] = {
            "requested_backup_count": width,
            "available_backup_count": len(ranker_ids),
            "candidate_shortfall": len(ranker_ids) < width,
            "ranker_backup_ids": ranker_ids,
            "target_probability_backup_ids": probability_ids,
            "selected_set_differs": set(ranker_ids) != set(probability_ids),
            "selected_order_differs": ranker_ids != probability_ids,
            "ranker_probability_mass": ranker_mass,
            "target_probability_sorted_mass": probability_mass,
            "mass_gain": delta,
            "ranker_conditional_hit_given_main_reject": ranker_conditional,
            "target_probability_sorted_conditional_hit": probability_conditional,
            "offline_mass_ids_match": True,
            "offline_mass_abs_error": abs(ranker_mass - old_mass),
            "offline_conditional_abs_error": abs(ranker_conditional - old_conditional),
        }
    return context


def summarize_contexts(contexts):
    source_names = sorted({row["source"] for row in contexts})
    by_source = {}
    for source in source_names:
        selected = [row for row in contexts if row["source"] == source]
        source_widths = {}
        for width in WIDTHS:
            values = [row["by_width"][str(width)] for row in selected]
            gains = [value["mass_gain"] for value in values]
            source_widths[str(width)] = {
                "contexts": len(values),
                "candidate_shortfall_contexts": sum(value["candidate_shortfall"] for value in values),
                "different_selected_set_contexts": sum(value["selected_set_differs"] for value in values),
                "different_selected_order_contexts": sum(value["selected_order_differs"] for value in values),
                "mass_gain_positive_contexts": sum(gain > MASS_EPS for gain in gains),
                "mass_gain_zero_contexts": sum(abs(gain) <= MASS_EPS for gain in gains),
                "mass_gain_negative_contexts": sum(gain < -MASS_EPS for gain in gains),
                "ranker_mass_mean": math.fsum(value["ranker_probability_mass"] for value in values) / len(values),
                "target_probability_sorted_mass_mean": math.fsum(value["target_probability_sorted_mass"] for value in values) / len(values),
                "mass_gain_mean": math.fsum(gains) / len(gains),
                "ranker_conditional_hit_mean": math.fsum(value["ranker_conditional_hit_given_main_reject"] for value in values) / len(values),
                "target_probability_sorted_conditional_hit_mean": math.fsum(value["target_probability_sorted_conditional_hit"] for value in values) / len(values),
            }
        by_source[source] = source_widths

    by_width = {}
    for width in WIDTHS:
        values = [row["by_width"][str(width)] for row in contexts]
        source_means = [by_source[source][str(width)] for source in source_names]
        gains = [value["mass_gain"] for value in values]
        by_width[str(width)] = {
            "contexts": len(values),
            "candidate_shortfall_contexts": sum(value["candidate_shortfall"] for value in values),
            "available_backup_count_total": sum(value["available_backup_count"] for value in values),
            "different_selected_set_contexts": sum(value["selected_set_differs"] for value in values),
            "different_selected_order_contexts": sum(value["selected_order_differs"] for value in values),
            "mass_gain_positive_contexts": sum(gain > MASS_EPS for gain in gains),
            "mass_gain_zero_contexts": sum(abs(gain) <= MASS_EPS for gain in gains),
            "mass_gain_negative_contexts": sum(gain < -MASS_EPS for gain in gains),
            "ranker_mass_context_mean": math.fsum(value["ranker_probability_mass"] for value in values) / len(values),
            "target_probability_sorted_mass_context_mean": math.fsum(value["target_probability_sorted_mass"] for value in values) / len(values),
            "mass_gain_context_mean": math.fsum(gains) / len(gains),
            "ranker_mass_source_equal_mean": math.fsum(value["ranker_mass_mean"] for value in source_means) / len(source_means),
            "target_probability_sorted_mass_source_equal_mean": math.fsum(value["target_probability_sorted_mass_mean"] for value in source_means) / len(source_means),
            "mass_gain_source_equal_mean": math.fsum(value["mass_gain_mean"] for value in source_means) / len(source_means),
            "ranker_conditional_hit_context_mean": math.fsum(value["ranker_conditional_hit_given_main_reject"] for value in values) / len(values),
            "target_probability_sorted_conditional_hit_context_mean": math.fsum(value["target_probability_sorted_conditional_hit"] for value in values) / len(values),
            "ranker_conditional_hit_source_equal_mean": math.fsum(value["ranker_conditional_hit_mean"] for value in source_means) / len(source_means),
            "target_probability_sorted_conditional_hit_source_equal_mean": math.fsum(value["target_probability_sorted_conditional_hit_mean"] for value in source_means) / len(source_means),
        }
    return by_width, by_source


def enumerate_pools(input_path: Path, offline_path: Path):
    data = load_json(input_path)
    offline = load_json(offline_path)
    source_rows = data["rows"]
    offline_rows = {(row["source"], int(row["prefix"])): row for row in offline["rows"]}
    contexts = []
    sum_values = []
    raw_candidate_count = 0
    removed_illegal_count = 0
    eos_candidate_count = 0
    duplicate_count = 0
    score_tie_contexts = 0
    probability_tie_comparisons = 0

    keys = set()
    for row in source_rows:
        key = (row["source"], int(row["prefix"]))
        if key in keys:
            raise ValueError(f"duplicate context key: {key}")
        keys.add(key)
        policy = [float(value) for value in row["target_policy"]]
        policy_sum = math.fsum(policy)
        sum_values.append(policy_sum)
        ids = [int(value) for value in row["ids"]]
        scores = [float(value) for value in row["scores"]]
        raw_candidate_count += len(ids)
        eos_candidate_count += sum(token == EOS for token in ids)
        removed_illegal_count += sum(token < 0 or token >= P_VOCAB or token == EOS for token in ids)
        duplicate_count += len(ids) - len(set(ids))
        if scores and sum(float(score) == max(scores) for score in scores) > 1:
            score_tie_contexts += 1
        context = enumerate_context(row, offline_rows.get(key))
        # A probability tie is counted as tied adjacent alternatives in the legal pool.
        probs = sorted(float(policy[int(row["ids"][i])]) for i in range(len(row["ids"])) if int(row["ids"][i]) != context["main_position"] and 0 <= int(row["ids"][i]) < P_VOCAB)
        probability_tie_comparisons += sum(left == right for left, right in zip(probs, probs[1:]))
        contexts.append(context)

    if len(contexts) != int(offline.get("contexts", -1)):
        raise ValueError("context count differs from offline_mass")
    by_width, by_source = summarize_contexts(contexts)
    policy_max_abs_error = max(abs(value - 1.0) for value in sum_values)
    if policy_max_abs_error > 1e-5:
        raise ValueError("saved target policies are not normalized within tolerance")

    return {
        "scope": "Fixed saved prefixes only; target-probability ordering is an oracle upper bound over the same legal backup pool.",
        "inputs": {
            "position_overlap": str(input_path),
            "offline_mass": str(offline_path),
            "contexts": len(contexts),
            "sources": len(by_source),
            "prefixes": sorted({row["prefix"] for row in contexts}),
            "contexts_by_prefix": {
                str(prefix): sum(row["prefix"] == prefix for row in contexts)
                for prefix in sorted({row["prefix"] for row in contexts})
            },
            "candidate_count_distribution": {
                str(count): sum(row["candidate_count"] == count for row in contexts)
                for count in sorted({row["candidate_count"] for row in contexts})
            },
            "legal_backup_count_distribution": {
                str(count): sum(row["legal_backup_count"] == count for row in contexts)
                for count in sorted({row["legal_backup_count"] for row in contexts})
            },
        },
        "selection_rules": {
            "main": "First argmax of the saved Ranker scores; exact ties keep input order.",
            "backup_pool": "Saved candidate IDs filtered to Position IDs in [0, 8000), with the main Position and EOS 12096 excluded.",
            "ranker_order": "Descending saved score, stable by original candidate index.",
            "target_probability_order": "Descending saved target_policy probability for the same candidate IDs, stable by original candidate index.",
            "quality_metric": "Sum of target_policy probabilities over the selected backups. The p-sorted value is the best mass available from that fixed pool for k items; it is not an online outcome or speed estimate.",
            "probability_handling": "Saved probabilities are used as written; no renormalization is applied.",
            "mass_comparison_epsilon": MASS_EPS,
        },
        "validation": {
            "p_vocab": P_VOCAB,
            "eos_id": EOS,
            "raw_candidate_ids": raw_candidate_count,
            "illegal_or_eos_ids_removed": removed_illegal_count,
            "eos_ids_seen": eos_candidate_count,
            "duplicate_candidate_ids": duplicate_count,
            "all_saved_points_match_ranker_argmax": True,
            "score_tie_contexts": score_tie_contexts,
            "adjacent_equal_probability_pairs_in_legal_pools": probability_tie_comparisons,
            "policy_sum_min": min(sum_values),
            "policy_sum_max": max(sum_values),
            "policy_sum_max_abs_error_from_one": policy_max_abs_error,
            "offline_mass_contexts": len(offline_rows),
            "offline_width_rows_compared": len(contexts) * len(WIDTHS),
            "offline_ranker_backup_ids_match": True,
            "offline_max_abs_mass_error": max(
                row["by_width"][str(width)]["offline_mass_abs_error"]
                for row in contexts for width in WIDTHS
            ),
            "offline_max_abs_conditional_error": max(
                row["by_width"][str(width)]["offline_conditional_abs_error"]
                for row in contexts for width in WIDTHS
            ),
        },
        "by_width": by_width,
        "by_source": by_source,
        "contexts_detail": contexts,
    }


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repo / "docs/project/artifacts/spec_next_20261022/raw/position_overlap_run01.json",
    )
    parser.add_argument(
        "--offline-mass",
        type=Path,
        default=repo / "docs/project/artifacts/position_widths_20261022/offline_mass.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo / "docs/project/artifacts/v3_review_20261023/POOL_ENUMERATION.json",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = enumerate_pools(args.input, args.offline_mass)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"contexts": result["inputs"]["contexts"], "sources": result["inputs"]["sources"], "by_width": result["by_width"], "validation": result["validation"]}, indent=2))


if __name__ == "__main__":
    main()
