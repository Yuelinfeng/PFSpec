"""CPU audits for B1 gap labels, route boundaries, and exact toy distributions."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from gap_gate import PF_K2, P_ONLY, choose_gap_route


ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "docs/project/artifacts/feature_value_20261017/full/paths.jsonl"


def audit_labels():
    rows = []
    with LABELS.open() as stream:
        for row in stream:
            item = json.loads(row)
            for round_index, round_data in enumerate(item["rounds"]):
                features = round_data["pre_features"]
                candidate = int(round_data["p1"])
                previous = int(features["previous_position"])
                recorded = int(features["p1_gap"])
                recomputed = candidate - previous
                rows.append((item["row_id"], round_index, recorded, recomputed))
    assert len(rows) == 5122, len(rows)
    assert all(recorded == recomputed for _, _, recorded, recomputed in rows)
    return {
        "status": "passed",
        "source": str(LABELS.relative_to(ROOT)),
        "candidate_gaps_checked": len(rows),
        "labels_matching": len(rows),
        "first_100_matching": sum(a[2] == a[3] for a in rows[:100]),
        "gap_counts": dict(sorted(Counter(row[2] for row in rows).items())),
        "negative_or_zero_gaps": sum(gap <= 0 for _, _, gap, _ in rows),
        "sample": [
            {"row_id": row, "round_index": index, "recorded_gap": recorded,
             "recomputed_gap": recomputed}
            for row, index, recorded, recomputed in rows[:100]
        ],
    }


def audit_routes():
    cases = [
        (2410, 2409, 4, PF_K2, "unit_gap"),
        (2411, 2409, 4, P_ONLY, "larger_gap"),
        (2410, 2409, 2, P_ONLY, "short_budget"),
        (None, 2409, 4, P_ONLY, "no_candidate"),
        (8000, 7999, 4, P_ONLY, "invalid_position"),
        (2409, 2409, 4, P_ONLY, "nonpositive_gap"),
        (2408, 2409, 4, P_ONLY, "nonpositive_gap"),
    ]
    results = []
    for candidate, previous, remaining, expected_route, expected_reason in cases:
        decision = choose_gap_route(candidate, previous, remaining)
        assert (decision.route, decision.reason) == (expected_route, expected_reason)
        results.append({
            "candidate_position": candidate,
            "previous_position": previous,
            "remaining_tokens": remaining,
            "route": decision.route,
            "gap": decision.gap,
            "reason": decision.reason,
            "passed": True,
        })
    return {"status": "passed", "cases": results}


def exact_token_law(target, proposal):
    target = np.asarray(target, dtype=np.float64)
    proposal = np.asarray(proposal, dtype=np.float64)
    accepted = np.minimum(target, proposal)
    residual = np.maximum(target - proposal, 0.0)
    rejected_mass = 1.0 - float(accepted.sum())
    if rejected_mass <= 1e-15:
        return accepted
    return accepted + rejected_mass * residual / float(residual.sum())


def exact_position_law(target, candidate):
    point = np.zeros_like(target, dtype=np.float64)
    point[candidate] = 1.0
    return exact_token_law(target, point)


def sequence_distribution(strategy):
    p_ids = (10, 11, 13, 16)
    f_ids = (100, 101, 102)
    position_rows = {
        10: np.array([0.12, 0.53, 0.25, 0.10]),
        11: np.array([0.08, 0.17, 0.61, 0.14]),
        13: np.array([0.21, 0.11, 0.18, 0.50]),
        16: np.array([0.45, 0.20, 0.23, 0.12]),
    }
    feature_rows = {
        (position, candidate): np.asarray(
            [0.15 + 0.08 * (i % 2), 0.55 - 0.06 * (i % 2), 0.30 - 0.02 * (i % 2)],
            dtype=np.float64,
        )
        for i, position in enumerate(p_ids)
        for candidate in p_ids
    }
    draft_rows = {
        key: np.roll(np.asarray([0.58, 0.27, 0.15], dtype=np.float64), i % 3)
        for i, key in enumerate(feature_rows)
    }
    ranker_choice = {10: 11, 11: 16, 13: 16, 16: 10}
    result = defaultdict(float)
    partial = [((), 1.0, 10)]
    for _ in range(2):
        next_partial = []
        for prefix, prefix_probability, previous in partial:
            candidate = ranker_choice[previous]
            if strategy == "target":
                route = P_ONLY
            elif strategy == "pf_k2":
                route = PF_K2
            elif strategy == "p_only":
                route = P_ONLY
            else:
                route = choose_gap_route(candidate, previous, 4).route
            p_target = position_rows[previous]
            p_output = (p_target if strategy == "target" else
                        exact_position_law(p_target, p_ids.index(candidate)))
            for pi, p_probability in enumerate(p_output):
                position = p_ids[pi]
                f_target = feature_rows[(previous, position)]
                if strategy == "target":
                    f_output = f_target
                elif route == PF_K2:
                    f_output = exact_token_law(f_target, draft_rows[(previous, position)])
                else:
                    f_output = f_target
                for fi, f_probability in enumerate(f_output):
                    mass = prefix_probability * float(p_probability) * float(f_probability)
                    next_partial.append((prefix + (position, f_ids[fi]), mass, position))
        partial = next_partial
    for sequence, probability, _ in partial:
        result[sequence] += probability
    return dict(result)


def audit_exact_distribution():
    target = sequence_distribution("target")
    comparisons = {}
    for strategy in ("p_only", "pf_k2", "gap_gated"):
        candidate = sequence_distribution(strategy)
        all_sequences = set(target) | set(candidate)
        max_error = max(abs(target.get(key, 0.0) - candidate.get(key, 0.0))
                        for key in all_sequences)
        total_error = sum(abs(target.get(key, 0.0) - candidate.get(key, 0.0))
                          for key in all_sequences) / 2.0
        assert len(all_sequences) == 16 * 9
        assert max_error < 1e-12 and total_error < 1e-12
        comparisons[strategy] = {
            "sequences_enumerated": len(all_sequences),
            "maximum_probability_error": max_error,
            "total_variation": total_error,
            "passed": True,
        }
    routes = {
        choose_gap_route(ranker_choice, previous, 4).route
        for previous, ranker_choice in {10: 11, 11: 16, 13: 16, 16: 10}.items()
    }
    assert routes == {PF_K2, P_ONLY}
    return {
        "status": "passed",
        "position_support": 4,
        "feature_support": 3,
        "pairs_enumerated": 2,
        "route_types_reached": sorted(routes),
        "comparisons": comparisons,
        "scope": "Exact two-pair autoregressive enumeration with deterministic ranker proposals and corrected P/F token kernels.",
    }


def main():
    report = {
        "gap_labels": audit_labels(),
        "route_boundaries": audit_routes(),
        "exact_distribution": audit_exact_distribution(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
