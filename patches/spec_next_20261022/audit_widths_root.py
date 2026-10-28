"""Independent arithmetic and output audit for completed six-arm decode groups."""
import argparse
import collections
import json
import math
from pathlib import Path
import random
import statistics


ARMS = ("target", "w1", "rescue1", "rescue2", "rescue3", "rescue4")
SEMANTIC = ("rounds", "proposed", "verified", "accepted", "accepted_lengths",
            "round_generated_lengths", "position_fallbacks", "k1_steps",
            "position_proposed", "position_verified", "position_accepted",
            "feature_proposed", "feature_verified", "feature_accepted",
            "feature_attempts", "feature_branch_hits", "feature_all_rejected",
            "feature_branch_index_counts", "eos_proposed", "eos_verified", "eos_accepted")


def comparison(groups, numerator, denominator):
    per_source = collections.defaultdict(list)
    for arms in groups:
        left, right = arms[numerator], arms[denominator]
        ratio = (left["generated_tokens"] / left["outer_wall_seconds"]) / (right["generated_tokens"] / right["outer_wall_seconds"])
        per_source[left["source"]].append(math.log(ratio))
    values = [statistics.fmean(items) for items in per_source.values()]
    if not values:
        return None
    rng = random.Random(104212)
    draws = sorted(math.exp(statistics.fmean(rng.choices(values, k=len(values)))) for _ in range(10000))
    return dict(ratio=math.exp(statistics.fmean(values)), ci95=[draws[249], draws[9749]],
                pairs=sum(len(items) for items in per_source.values()), sources=len(values),
                source_ratios={key: math.exp(statistics.fmean(items)) for key, items in per_source.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    grouped = collections.defaultdict(dict)
    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    for row in rows:
        key = (row["source"], row["prefix"], row["seed"], row["budget"])
        if row["method"] in grouped[key]:
            raise ValueError(f"duplicate arm for {key}")
        grouped[key][row["method"]] = row
    groups = [arms for arms in grouped.values() if set(arms) == set(ARMS)]
    errors = []
    counts = {}
    actual = {}
    for arm in ARMS:
        counts[arm] = {field: sum(group[arm]["stats"].get(field, 0) for group in groups)
                       for field in ("target_calls", "target_input_tokens", "target_tree_nodes",
                                     "recovery_calls", "position_backup_nodes", "position_rescue_hits")}
        if arm.startswith("rescue"):
            width = int(arm[-1])
            present = {rank: sum(group[arm]["stats"].get("position_backup_applicable_rounds_by_rank", {}).get(str(rank), 0) for group in groups)
                       for rank in range(1, width + 2)}
            actual[arm] = {str(rank): present[rank] - present[rank + 1] for rank in range(1, width + 1)}
    for group in groups:
        original = group["w1"]
        for arm in ARMS[2:]:
            row = group[arm]
            differences = [key for key in SEMANTIC if row["stats"].get(key) != original["stats"].get(key)]
            if differences or row["output_tokens"] != original["output_tokens"]:
                errors.append(dict(pair=row["pair_id"], arm=arm, output_equal=row["output_tokens"] == original["output_tokens"], semantic_differences=differences))
            saved = original["stats"]["recovery_calls"] - row["stats"]["recovery_calls"]
            hits = row["stats"]["position_rescue_hits"]
            if saved != hits or original["stats"]["target_calls"] - row["stats"]["target_calls"] != hits:
                errors.append(dict(pair=row["pair_id"], arm=arm, recovery_saved=saved, hits=hits))
    pairs = [(arm, reference) for arm in ARMS[1:] for reference in ("target", "w1", "rescue1") if arm != reference]
    pairs += [(f"rescue{k}", f"rescue{k-1}") for k in range(2, 5)]
    result = dict(records=str(args.records), rows=len(rows), complete_groups=len(groups),
                  incomplete_groups=len(grouped) - len(groups), errors=errors, counts=counts,
                  actual_backup_round_distribution=actual,
                  comparisons={f"{a}/{b}": comparison(groups, a, b) for a, b in pairs},
                  full512={f"{a}/{b}": comparison([g for g in groups if all(r["generated_tokens"] == 512 for r in g.values())], a, b) for a, b in pairs})
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("rows", "complete_groups", "incomplete_groups", "errors")}))


if __name__ == "__main__":
    main()
