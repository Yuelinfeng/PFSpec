#!/usr/bin/env python3
"""Summarize decoder counters from a completed Target Methods run."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


GENERIC_METHODS = {"target", "ranker_only", "pf_mlp", "d8_pf", "d8_joint"}


def number(stats: dict, key: str) -> int:
    return int(stats.get(key, 0) or 0)


def rate(accepted: int, denominator: int):
    return accepted / denominator if denominator else None


def add_counts(left: dict, right: dict) -> dict:
    return {key: int(left.get(key, 0)) + int(right.get(key, 0))
            for key in ("proposed", "verified", "accepted")}


def classify_record(row: dict) -> dict:
    method = row["method"]
    stats = row.get("stats") or {}

    if method in GENERIC_METHODS:
        stages = {
            kind: {field: number(stats, f"{kind}_{field}")
                   for field in ("proposed", "verified", "accepted")}
            for kind in ("position", "feature", "eos")
        }
        totals = {field: number(stats, field)
                  for field in ("proposed", "verified", "accepted")}
        target_calls = number(stats, "target_calls")
        target_slots = stats.get("target_input_tokens")
        target_slots_basis = "stats.target_input_tokens"
    elif method.startswith("eagle_"):
        stages = {
            kind: {field: number(stats, f"{kind}_{field}")
                   for field in ("proposed", "verified", "accepted")}
            for kind in ("position", "feature")
        }
        totals = {field: sum(values[field] for values in stages.values())
                  for field in ("proposed", "verified", "accepted")}
        target_calls = number(stats, "target_calls")
        verify_calls = number(stats, "verify_calls")
        recovery_calls = number(stats, "recovery_calls")
        prefill_slots = int(row["prefix"]) + 1
        verify_slots = totals["proposed"] + max(verify_calls - 1, 0)
        target_slots = prefill_slots + verify_slots + recovery_calls
        target_slots_basis = (
            "prefix+1 prefill + typed proposals + verify_calls-1 pending slots "
            "+ one input slot per recovery_call"
        )
    elif method in {"medusa_lowrank", "hydra_conditional"}:
        p1 = {"proposed": number(stats, "p1_checked"),
              "verified": number(stats, "p1_checked"),
              "accepted": number(stats, "p1_accepted")}
        f1 = {"proposed": number(stats, "f1_proposed"),
              "verified": number(stats, "f1_checked"),
              "accepted": number(stats, "f1_accepted")}
        p2 = {"proposed": number(stats, "p2_proposed"),
              "verified": number(stats, "p2_checked"),
              "accepted": number(stats, "p2_accepted")}
        stages = {"p1": p1, "f1": f1, "p2": p2}
        position = add_counts(p1, p2)
        feature = f1.copy()
        totals = add_counts(position, feature)
        target_calls = sum(number(stats, key) for key in (
            "forward_prefill", "forward_single", "forward_pair",
            "forward_block3", "forward_block4",
        ))
        prefix_slots = int(row["prefix"]) * number(stats, "forward_prefill")
        target_slots = prefix_slots + number(stats, "forward_single")
        target_slots += 2 * number(stats, "forward_pair")
        target_slots += 3 * number(stats, "forward_block3")
        target_slots += 4 * number(stats, "forward_block4")
        target_slots_basis = "prefix*forward_prefill + widths 1/2/3/4 by forward type"
    elif method == "lfq_scratch_pf_bos":
        p1 = {"proposed": number(stats, "position_1_proposals"),
              "verified": number(stats, "position_1_proposals"),
              "accepted": number(stats, "position_1_accepted")}
        f1 = {"proposed": number(stats, "feature_proposals"),
              "verified": number(stats, "feature_checked"),
              "accepted": number(stats, "feature_accepted")}
        p2 = {"proposed": number(stats, "position_2_proposals"),
              "verified": number(stats, "position_2_checked"),
              "accepted": number(stats, "position_2_accepted")}
        stages = {"position_1": p1, "feature": f1, "position_2": p2}
        totals = add_counts(add_counts(p1, f1), p2)
        target_calls = sum(number(stats, key) for key in (
            "forward_single", "forward_pair", "forward_block4",
        ))
        target_slots = (number(stats, "forward_single")
                        + 2 * number(stats, "forward_pair")
                        + 4 * number(stats, "forward_block4"))
        target_slots_basis = "widths 1/2/4 by forward type"
    else:
        raise ValueError(f"Unsupported method: {method}")

    if method in {"medusa_lowrank", "hydra_conditional"}:
        types = {"position": add_counts(stages["p1"], stages["p2"]),
                 "feature": stages["f1"].copy()}
    elif method == "lfq_scratch_pf_bos":
        types = {"position": add_counts(stages["position_1"], stages["position_2"]),
                 "feature": stages["feature"].copy()}
    else:
        types = stages

    if "draft_calls" in stats:
        draft_calls = number(stats, "draft_calls")
        draft_calls_basis = "stats.draft_calls"
        head_forward = None
    elif "head_forward" in stats:
        draft_calls = None
        draft_calls_basis = None
        head_forward = number(stats, "head_forward")
    elif method == "lfq_scratch_pf_bos":
        draft_calls = number(stats, "k2_blocks")
        draft_calls_basis = "one Feature Draft call per k2_block"
        head_forward = None
    else:
        draft_calls = None
        draft_calls_basis = None
        head_forward = None

    draft_input_slots = stats.get("draft_input_tokens")
    if draft_input_slots is not None:
        draft_input_slots = int(draft_input_slots)
        draft_input_basis = "stats.draft_input_tokens"
    else:
        draft_input_basis = None

    generated = int(row["generated_tokens"])
    eos = bool(row.get("eos", False))
    return {
        "method": method,
        "execution": row["execution"],
        "pair_id": row["pair_id"],
        "source": row["source"],
        "prefix": int(row["prefix"]),
        "generated_tokens": generated,
        "eos": eos,
        "target_calls": target_calls,
        "target_input_slots": None if target_slots is None else int(target_slots),
        "target_input_slots_basis": target_slots_basis,
        "candidate_counts": totals,
        "counts_by_type": types,
        "counts_by_stage": stages if method in {"medusa_lowrank", "hydra_conditional", "lfq_scratch_pf_bos"} else None,
        "draft_calls": draft_calls,
        "draft_calls_basis": draft_calls_basis,
        "head_forward_calls": head_forward,
        "draft_input_slots": draft_input_slots,
        "draft_input_slots_basis": draft_input_basis,
        "ranker_calls": (number(stats, "ranker_calls") if "ranker_calls" in stats else None),
    }


def summarize_arm(method: str, execution: str, classified: list[dict]) -> dict:
    candidate_totals = {field: sum(row["candidate_counts"][field] for row in classified)
                        for field in ("proposed", "verified", "accepted")}
    all_types = sorted({kind for row in classified for kind in row["counts_by_type"]})
    by_type = {}
    for kind in all_types:
        values = {field: sum(row["counts_by_type"].get(kind, {}).get(field, 0)
                             for row in classified)
                  for field in ("proposed", "verified", "accepted")}
        by_type[kind] = {
            **values,
            "acceptance_verified": rate(values["accepted"], values["verified"]),
            "acceptance_proposed": rate(values["accepted"], values["proposed"]),
        }

    generated = sum(row["generated_tokens"] for row in classified)
    target_calls = sum(row["target_calls"] for row in classified)
    target_slots_values = [row["target_input_slots"] for row in classified]
    target_slots = (sum(target_slots_values) if all(value is not None for value in target_slots_values)
                    else None)
    draft_call_values = [row["draft_calls"] for row in classified]
    draft_calls = (sum(draft_call_values) if all(value is not None for value in draft_call_values)
                   else None)
    head_forward_values = [row["head_forward_calls"] for row in classified]
    head_forward = (sum(head_forward_values) if all(value is not None for value in head_forward_values)
                    else None)
    draft_slot_values = [row["draft_input_slots"] for row in classified]
    draft_slots = (sum(draft_slot_values) if all(value is not None for value in draft_slot_values)
                   else None)
    return {
        "records": len(classified),
        "generated_tokens": generated,
        "natural_eos_records": sum(row["eos"] for row in classified),
        "candidate_counts": {
            **candidate_totals,
            "acceptance_verified": rate(candidate_totals["accepted"], candidate_totals["verified"]),
            "acceptance_proposed": rate(candidate_totals["accepted"], candidate_totals["proposed"]),
            "accepted_per_target_call": rate(candidate_totals["accepted"], target_calls),
        },
        "counts_by_type": by_type,
        "target_calls": target_calls,
        "generated_tokens_per_target_call": rate(generated, target_calls),
        "target_input_slots": target_slots,
        "target_input_slots_per_generated_token": rate(target_slots, generated) if target_slots is not None else None,
        "target_input_slots_per_target_call": rate(target_slots, target_calls) if target_slots is not None else None,
        "target_input_slots_basis": sorted({row["target_input_slots_basis"] for row in classified}),
        "draft_calls": draft_calls,
        "draft_calls_per_generated_token": rate(draft_calls, generated) if draft_calls is not None else None,
        "draft_calls_basis": sorted({row["draft_calls_basis"] for row in classified if row["draft_calls_basis"]}),
        "head_forward_calls": head_forward,
        "head_forward_calls_per_generated_token": rate(head_forward, generated) if head_forward is not None else None,
        "draft_input_slots": draft_slots,
        "draft_input_slots_per_generated_token": rate(draft_slots, generated) if draft_slots is not None else None,
        "draft_input_slots_basis": sorted({row["draft_input_slots_basis"] for row in classified if row["draft_input_slots_basis"]}),
        "ranker_calls": (sum(row["ranker_calls"] for row in classified)
                         if all(row["ranker_calls"] is not None for row in classified) else None),
        "ranker_calls_per_generated_token": (rate(sum(row["ranker_calls"] for row in classified), generated)
                                               if all(row["ranker_calls"] is not None for row in classified) else None),
    }


def build_summary(run_dir: Path) -> dict:
    plan = json.loads((run_dir / "plan.json").read_text())
    audit = json.loads((run_dir / "audit.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines() if line.strip()]
    if audit.get("status") != "passed":
        raise ValueError("Run audit status is not passed")
    if len(rows) != int(plan["expected_runs"]):
        raise ValueError(f"Expected {plan['expected_runs']} rows, read {len(rows)}")

    methods = list(plan["methods"])
    executions = list(plan["executions"])
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["execution"])].append(classify_record(row))
    method_results = {}
    for method in methods:
        method_results[method] = {
            execution: summarize_arm(method, execution, grouped[(method, execution)])
            for execution in executions
        }

    pair_groups = defaultdict(dict)
    for row in rows:
        pair_groups[(row["method"], row["pair_id"])][row["execution"]] = row
    parity = defaultdict(lambda: {"pairs": 0, "output_equal": 0, "counters_equal": 0})
    for (method, _), pair in pair_groups.items():
        if {"original", "optimized"} <= set(pair):
            parity[method]["pairs"] += 1
            parity[method]["output_equal"] += pair["original"].get("output_tokens") == pair["optimized"].get("output_tokens")
            parity[method]["counters_equal"] += pair["original"].get("counters") == pair["optimized"].get("counters")

    speed_context = {}
    for method in methods:
        if method == "target":
            continue
        speed_context[method] = {}
        for execution in executions:
            key = f"algorithm_gain:{execution}:{method}"
            value = summary.get("contrasts", {}).get(key)
            if value is None:
                continue
            ratio = value["all_output_lengths"]
            speed_context[method][execution] = {
                "source_equal_geometric_mean": ratio["source_equal_geometric_mean"],
                "source_bootstrap_95_ci": ratio["source_bootstrap_95_ci"],
                "pair_count": ratio["pair_count"],
            }

    return {
        "run_dir": str(run_dir.resolve()),
        "source_scope": plan.get("source_scope"),
        "status": summary.get("status"),
        "audit_status": audit.get("status"),
        "expected_runs": plan["expected_runs"],
        "records": len(rows),
        "methods": methods,
        "executions": executions,
        "prompt_count": plan.get("prompt_count"),
        "sources": plan.get("sources"),
        "prefixes": plan.get("prefixes"),
        "budget": plan.get("budget"),
        "aggregation": "pooled counters over rows within each method/execution; rates divide summed counts",
        "paired_execution_parity_applicable": {"original", "optimized"} <= set(executions),
        "pair_parity": {method: values for method, values in sorted(parity.items())},
        "methods_summary": method_results,
        "measured_algorithm_gain_context": speed_context,
        "limits": [
            "LFQ BOS is not included in this run.",
            "Draft input slots are absent for D8-Joint and Medusa/Hydra; head calls are listed separately.",
            "Target input slots for EAGLE are derived from original counter fields and match optimized runtime query-width totals.",
            "Counter ratios describe work and acceptance, not elapsed CPU or GPU time shares.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve() if args.output else run_dir / "counter_summary.json"
    result = build_summary(run_dir)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "output": str(output), "records": result["records"]}))


if __name__ == "__main__":
    main()
