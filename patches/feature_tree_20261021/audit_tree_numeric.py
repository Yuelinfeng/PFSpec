#!/usr/bin/env python3
"""Recompute Target and Feature proposal probability checks from saved CPU tensors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def finite_all(torch, *values):
    return all(bool(torch.isfinite(value).all()) for value in values)


def support_equal(torch, left, right):
    if tuple(left.shape) != tuple(right.shape):
        return False
    return bool(torch.equal(left > 0, right > 0))


def total_variation(left, right):
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"Probability shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}")
    return float((left.double() - right.double()).abs().sum().item() / 2.0)


def mass_error(left, right):
    return max(
        abs(float(left.double().sum().item()) - 1.0),
        abs(float(right.double().sum().item()) - 1.0),
    )


def sampling_probs(torch, raw_logits, temperature, top_p):
    """Match draft_a.eval_overlap.sampling_probs for an unmasked Position row."""
    logits = raw_logits.float().clone() / temperature
    sorted_logits, indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=0)
    remove = torch.cumsum(sorted_probs, dim=0) > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_logits[remove] = -torch.inf
    filtered = torch.full_like(logits, -torch.inf)
    filtered.scatter_(0, indices, sorted_logits)
    return torch.softmax(filtered, dim=0)


def summarize(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"count": 0}

    def quantile(probability):
        index = min(len(values) - 1, max(0, math.ceil(probability * len(values)) - 1))
        return values[index]

    return {
        "count": len(values),
        "min": values[0],
        "mean": sum(values) / len(values),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "max": values[-1],
    }


def target_summary(torch, rows):
    return {
        "count": len(rows),
        "tv_from_saved_policy_distributions": summarize(row["tv_saved"] for row in rows),
        "tv_from_cpu_recomputed_logits": summarize(row["tv_cpu"] for row in rows),
        "policy_distribution_nonfinite_rows": sum(not row["distributions_finite"] for row in rows),
        "support_mismatches_saved": sum(not row["support_saved"] for row in rows),
        "support_mismatches_cpu_recomputed": sum(not row["support_cpu"] for row in rows),
        "max_saved_mass_error": max((row["mass_saved"] for row in rows), default=0.0),
        "max_cpu_mass_error": max((row["mass_cpu"] for row in rows), default=0.0),
        "rows_over_mass_tolerance_1e-4": sum(
            max(row["mass_saved"], row["mass_cpu"]) > 1e-4 for row in rows
        ),
        "max_cpu_vs_saved_probability_abs": max(
            (row["cpu_saved_max_abs"] for row in rows), default=0.0
        ),
        "cpu_vs_saved_support_mismatches": sum(
            row["cpu_saved_support_mismatches"] for row in rows
        ),
        "max_json_tv_delta_saved": max(
            (row["json_tv_delta_saved"] for row in rows), default=0.0
        ),
        "max_json_tv_delta_cpu": max(
            (row["json_tv_delta_cpu"] for row in rows), default=0.0
        ),
        "json_support_disagreements_saved": sum(
            not row["json_support_saved"] for row in rows
        ),
        "json_support_disagreements_cpu": sum(
            not row["json_support_cpu"] for row in rows
        ),
        "json_finite_mask_disagreements": sum(
            not row["json_finite_mask"] for row in rows
        ),
        "raw_logit_nan_entries": sum(row["raw_logit_nans"] for row in rows),
        "raw_logit_negative_inf_entries": sum(row["raw_logit_negative_infs"] for row in rows),
        "raw_logit_positive_inf_entries": sum(row["raw_logit_positive_infs"] for row in rows),
        "json_rows_passed": sum(row["json_passed"] for row in rows),
    }


def mlp_summary(rows):
    return {
        "count": len(rows),
        "tv_from_saved_q_distributions": summarize(row["tv"] for row in rows),
        "nonfinite_rows": sum(not row["finite"] for row in rows),
        "support_mismatch_rows": sum(not row["support"] for row in rows),
        "max_mass_error": max((row["mass"] for row in rows), default=0.0),
        "rows_over_mass_tolerance_1e-5": sum(row["mass"] > 1e-5 for row in rows),
        "max_json_tv_delta": max((row["json_tv_delta"] for row in rows), default=0.0),
        "json_support_disagreements": sum(not row["json_support"] for row in rows),
        "json_finite_disagreements": sum(not row["json_finite"] for row in rows),
        "max_json_mass_delta": max((row["json_mass_delta"] for row in rows), default=0.0),
        "json_rows_passed": sum(row["json_passed"] for row in rows),
    }


def make_audit(json_path, raw_path, temperature=0.9, top_p=0.9):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to read the raw tensor file") from error

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    expected = json.loads(json_path.read_text(encoding="utf-8"))
    raw = torch.load(raw_path, map_location="cpu")
    if not isinstance(raw, list):
        raise ValueError("Raw tensor file must contain a list of records")
    if not all(
        tensor.device.type == "cpu"
        for item in raw if isinstance(item, dict)
        for tensor in item.values() if torch.is_tensor(tensor)
    ):
        raise ValueError("A raw tensor was not loaded on CPU")

    expected_rows = [row for row in expected.get("rows", []) if "policy_tv" in row]
    if len(raw) != len(expected_rows):
        raise ValueError(
            f"Raw/JSON probability row counts differ: {len(raw)} != {len(expected_rows)}"
        )

    target_rows = []
    mlp_rows = []
    target_subclasses = {}
    label_mismatches = []
    for index, (item, row) in enumerate(zip(raw, expected_rows)):
        label = item.get("label")
        if label != row.get("label"):
            label_mismatches.append({"index": index, "raw": label, "json": row.get("label")})

        if str(label).startswith("mlp_q_"):
            left, right = item["p"], item["q"]
            tv = total_variation(left, right)
            normalized = mass_error(left, right)
            finite = finite_all(torch, left, right)
            support = support_equal(torch, left, right)
            mlp_rows.append({
                "tv": tv,
                "mass": normalized,
                "finite": finite,
                "support": support,
                "json_tv_delta": abs(tv - float(row["policy_tv"])),
                "json_mass_delta": abs(normalized - float(row.get("mass_error", 0.0))),
                "json_support": support == bool(row["policy_support_equal"]),
                "json_finite": finite == bool(row.get("all_finite", False)),
                "json_passed": bool(row.get("passed")),
            })
            continue

        logits_left, logits_right = item["left"], item["right"]
        saved_left, saved_right = item["p"], item["q"]
        cpu_left = sampling_probs(torch, logits_left, temperature, top_p)
        cpu_right = sampling_probs(torch, logits_right, temperature, top_p)
        tv_saved = total_variation(saved_left, saved_right)
        tv_cpu = total_variation(cpu_left, cpu_right)
        subtype = (
            "after_commit_target" if label == "after_commit_logit"
            else "bmm_target" if str(label).startswith("bmm_p")
            else "tree_target"
        )
        row_result = {
            "tv_saved": tv_saved,
            "tv_cpu": tv_cpu,
            "distributions_finite": finite_all(torch, saved_left, saved_right, cpu_left, cpu_right),
            "support_saved": support_equal(torch, saved_left, saved_right),
            "support_cpu": support_equal(torch, cpu_left, cpu_right),
            "mass_saved": mass_error(saved_left, saved_right),
            "mass_cpu": mass_error(cpu_left, cpu_right),
            "cpu_saved_max_abs": max(
                float((cpu_left.double() - saved_left.double()).abs().max().item()),
                float((cpu_right.double() - saved_right.double()).abs().max().item()),
            ),
            "cpu_saved_support_mismatches": int(not torch.equal(cpu_left > 0, saved_left > 0))
            + int(not torch.equal(cpu_right > 0, saved_right > 0)),
            "json_tv_delta_saved": abs(tv_saved - float(row["policy_tv"])),
            "json_tv_delta_cpu": abs(tv_cpu - float(row["policy_tv"])),
            "json_support_saved": support_equal(torch, saved_left, saved_right)
            == bool(row["policy_support_equal"]),
            "json_support_cpu": support_equal(torch, cpu_left, cpu_right)
            == bool(row["policy_support_equal"]),
            "json_finite_mask": bool(torch.equal(torch.isfinite(logits_left), torch.isfinite(logits_right)))
            == bool(row["finite_mask_equal"]),
            "raw_logit_nans": int(torch.isnan(logits_left).sum().item()
                                  + torch.isnan(logits_right).sum().item()),
            "raw_logit_negative_infs": int(torch.isneginf(logits_left).sum().item()
                                           + torch.isneginf(logits_right).sum().item()),
            "raw_logit_positive_infs": int(torch.isposinf(logits_left).sum().item()
                                           + torch.isposinf(logits_right).sum().item()),
            "json_passed": bool(row.get("passed")),
        }
        target_rows.append(row_result)
        target_subclasses.setdefault(subtype, []).append(row_result)

    json_tv_max = max((float(row["policy_tv"]) for row in expected_rows), default=0.0)
    json_summary = {
        key: expected.get(key) for key in (
            "status", "contexts", "comparisons", "passed", "max_policy_tv",
            "policy_support_differences", "elementwise_allclose_failures", "max_abs",
            "max_relative_l2", "max_relative_max",
        )
    }
    target = target_summary(torch, target_rows)
    mlp = mlp_summary(mlp_rows)
    checks = {
        "raw_json_label_order_matches": not label_mismatches,
        "saved_target_tv_matches_json": target["max_json_tv_delta_saved"] <= 1e-12,
        "cpu_target_tv_within_1e-6_of_json": target["max_json_tv_delta_cpu"] <= 1e-6,
        "target_distributions_finite": target["policy_distribution_nonfinite_rows"] == 0,
        "target_supports_match": target["support_mismatches_saved"] == 0
        and target["support_mismatches_cpu_recomputed"] == 0,
        "target_normalization_within_1e-4": target["rows_over_mass_tolerance_1e-4"] == 0,
        "mlp_q_distributions_finite": mlp["nonfinite_rows"] == 0,
        "mlp_q_supports_match": mlp["support_mismatch_rows"] == 0,
        "mlp_q_normalization_within_1e-5": mlp["rows_over_mass_tolerance_1e-5"] == 0,
        "all_probability_json_rows_passed": target["json_rows_passed"] == len(target_rows)
        and mlp["json_rows_passed"] == len(mlp_rows),
        "json_summary_max_tv_matches_rows": math.isclose(
            float(expected.get("max_policy_tv", -1)), json_tv_max, rel_tol=0, abs_tol=1e-12
        ),
    }
    checks["json_summary_reports_complete"] = (
        expected.get("status") == "complete"
        and expected.get("comparisons") == expected.get("passed")
    )
    return {
        "audit_version": 1,
        "result": "passed" if all(checks.values()) else "failed",
        "input_json": str(json_path),
        "raw_tensor_path": str(raw_path),
        "raw_tensor_loaded_with_map_location": "cpu",
        "torch_device_for_recalculation": "cpu",
        "policy_recalculation": {
            "function": "draft_a.eval_overlap.sampling_probs equivalent",
            "temperature": temperature,
            "top_p": top_p,
            "prev_pos": None,
        },
        "scope": "Recomputed Target sampling distributions from saved logits and MLP q TV, support, and normalization from saved policy tensors. Hidden/KV relative metrics were not independently recalculated.",
        "json_run_summary": json_summary,
        "raw_probability_pairs": len(raw),
        "json_probability_rows": len(expected_rows),
        "label_order_mismatch_count": len(label_mismatches),
        "first_label_order_mismatches": label_mismatches[:10],
        "target_logits": target,
        "target_subclasses": {
            key: target_summary(torch, value) for key, value in target_subclasses.items()
        },
        "mlp_q": mlp,
        "checks": checks,
        "interpretation": "Target logits may contain matching negative-infinity masks; the audited sampling distributions must be finite and normalized. The reported elementwise allclose failures remain visible in the source JSON.",
        "limitations": "The MLP q raw artifact contains final probability vectors but not their pre-policy logits, so its TV/support/mass checks use those saved vectors directly.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", type=Path, required=True,
                        help="Verification JSON containing comparison rows")
    parser.add_argument("--raw", dest="raw_path", type=Path, required=True,
                        help="Raw policy tensor file from the same run")
    parser.add_argument("--out", required=True,
                        help="New output JSON path, or - to print the result")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args(argv)
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be positive and top-p must be in (0, 1]")
    if not args.json_path.is_file() or not args.raw_path.is_file():
        parser.error("JSON and raw tensor inputs must be existing files")
    result = make_audit(args.json_path, args.raw_path, args.temperature, args.top_p)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out == "-":
        sys.stdout.write(rendered)
    else:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    return 0 if result["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
