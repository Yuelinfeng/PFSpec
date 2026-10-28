#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


FEATURES = 4096
POLICY = {"temperature": 0.9, "top_p": 0.9}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _near(actual: float, expected: float, *, relative: float = 1e-12,
          absolute: float = 0.0) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=relative, abs_tol=absolute)


def _distribution(np, entries: dict[int, float]):
    values = np.zeros(FEATURES, dtype=np.float64)
    for token, mass in entries.items():
        values[token] = mass
    return values


def _raw_logits(q1, np):
    logits = np.full(FEATURES, -100.0, dtype=np.float64)
    positive = np.flatnonzero(q1 > 0.0)
    logits[positive] = np.log(q1[positive])
    return logits


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
                    encoding="utf-8")


def _build_collection(collection: Path, np, checkpoint: Path) -> tuple[float, list[dict]]:
    collection.mkdir(parents=True)
    tiny = 2.0 ** -60
    distributions = [
        (_distribution(np, {0: 0.4, 1: 0.6}),
         _distribution(np, {0: 0.4, 1: 0.6})),
        (_distribution(np, {0: 0.5, 1: 0.5}),
         _distribution(np, {0: 1.0})),
        (_distribution(np, {0: 2.0 ** -10, 1: 2.0 ** -10,
                            2: 1.0 - 2.0 ** -9}),
         _distribution(np, {0: 2.0 ** -10 + tiny, 1: 2.0 ** -10 - tiny,
                            2: 1.0 - 2.0 ** -9})),
        (_distribution(np, {0: 0.5, 1: 0.5}),
         _distribution(np, {0: 1.0})),
    ]
    p_vectors = np.stack([pair[0] for pair in distributions])
    q_vectors = np.stack([pair[1] for pair in distributions])
    logits = np.stack([_raw_logits(q1, np) for _, q1 in distributions])
    np.savez_compressed(collection / "vectors.npz", p_f=p_vectors,
                        q1=q_vectors, raw_logits=logits)

    contexts = [
        dict(sample_id="a64-zero", scene="source-a", prefix=64, vector_index=0,
             p1_accept_probability=0.8, p1_accepted_actual=True,
             f1_accepted_actual=True, policy=dict(POLICY)),
        dict(sample_id="a256-unsupported", scene="source-a", prefix=256,
             vector_index=1, p1_accept_probability=0.7,
             p1_accepted_actual=True, f1_accepted_actual=True, policy=dict(POLICY)),
        dict(sample_id="b64-tiny-legal", scene="source-b", prefix=64,
             vector_index=2, p1_accept_probability=0.6,
             p1_accepted_actual=True, f1_accepted_actual=True, policy=dict(POLICY)),
        dict(sample_id="b64-unsupported", scene="source-b", prefix=64,
             vector_index=3, p1_accept_probability=0.6,
             p1_accepted_actual=True, f1_accepted_actual=True, policy=dict(POLICY)),
    ]
    _write_jsonl(collection / "contexts.jsonl", contexts)
    cell_counts = [
        dict(scene="source-a", prefix=64, p1_proposals=10,
             p1_probability_mean=0.8, p1_acceptance_actual_rate=0.7,
             p1_accepted=7, p1_rejected=3, f1_accepted=6, f1_rejected=1,
             source_rows=1),
        dict(scene="source-a", prefix=256, p1_proposals=10,
             p1_probability_mean=0.7, p1_acceptance_actual_rate=0.6,
             p1_accepted=6, p1_rejected=4, f1_accepted=5, f1_rejected=1,
             source_rows=1),
        dict(scene="source-b", prefix=64, p1_proposals=12,
             p1_probability_mean=0.6, p1_acceptance_actual_rate=0.6,
             p1_accepted=7, p1_rejected=5, f1_accepted=5, f1_rejected=2,
             source_rows=2),
        dict(scene="source-b", prefix=256, p1_proposals=8,
             p1_probability_mean=0.4, p1_acceptance_actual_rate=0.5,
             p1_accepted=4, p1_rejected=4, f1_accepted=3, f1_rejected=1,
             source_rows=0),
    ]
    metadata = dict(feature_step="synthetic-step-106000",
                    feature_checkpoint=str(checkpoint.resolve()),
                    selected_sources=["source-a", "source-b"],
                    by_source_prefix=cell_counts,
                    sample_shortfalls=[dict(scene="source-b", prefix=256,
                                            requested=1, collected=0)])
    (collection / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (collection / "w1_profiles.jsonl").write_text("", encoding="utf-8")
    return tiny, contexts


def _build_measurement(measurement: Path) -> None:
    measurement.mkdir(parents=True)
    metadata = dict(
        selected_contexts=4, unsupported_q2_contexts=2, maximum_contexts=24,
        q2_mode="narrow", paired_contexts_with_p3=1,
        paired_contexts_skipped_no_p3=0, all_diagnostic_target_calls=8,
        target_call_counts=dict(prefill=1, original_shape=2, warmup=2, paired=4),
        kv_truncation_counts=dict(shape_pair_setup=1, shape_prewarm=2,
                                  shape_pair_forward=4))
    (measurement / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    branch_rows = [dict(sample_id="b64-tiny-legal", first_reject_local_id=0,
                        second_candidate_local_id=1, q2_mode="narrow",
                        p2_accept_probability=0.25)]
    _write_jsonl(measurement / "branch_rows.jsonl", branch_rows)

    timings = [
        dict(record_type="target_forward", sample_id="b64-tiny-legal",
             scene="source-b", prefix=64, shape="short_2", input_tokens=2,
             cuda_seconds=0.003, wall_seconds=0.01),
        dict(record_type="target_forward", sample_id="b64-tiny-legal",
             scene="source-b", prefix=64, shape="next_4", input_tokens=4,
             cuda_seconds=0.009, wall_seconds=0.03),
        dict(record_type="shape_prewarm", sample_id="b64-tiny-legal",
             shape="short_2", cuda_seconds=0.004, host_wall_seconds=0.05),
        dict(record_type="shape_prewarm", sample_id="b64-tiny-legal",
             shape="next_4", cuda_seconds=0.008, host_wall_seconds=0.07),
    ]
    sequences = {"short_2": [8001, 9001],
                 "next_4": [8001, 9001, 7001, 7002]}
    paired_values = [
        (0, "short_2", 0.20, 0.02),
        (0, "next_4", 0.40, 0.04),
        (1, "next_4", 0.60, 0.06),
        (1, "short_2", 0.30, 0.03),
    ]
    for pair_index, shape, wall_seconds, cuda_seconds in paired_values:
        pair_order = 0 if (pair_index == 0 and shape == "short_2" or
                           pair_index == 1 and shape == "next_4") else 1
        pair_start = "short_2" if pair_index == 0 else "next_4"
        input_ids = sequences[shape]
        before = 7
        timings.append(dict(
            record_type="shape_pair_forward", sample_id="b64-tiny-legal",
            scene="source-b", prefix=64, pair_id=f"b64-tiny-legal:pair:{pair_index}",
            pair_index=pair_index, pair_order=pair_order, order=pair_order,
            pair_start_shape=pair_start, shape=shape, input_ids=input_ids,
            input_tokens=len(input_ids), kv_position_before=before,
            kv_position_after=before + len(input_ids),
            kv_position_after_restore=before, device="synthetic-cpu",
            cuda_seconds=cuda_seconds, host_wall_seconds=wall_seconds,
            restore_cuda_seconds=0.001, restore_host_wall_seconds=0.003))
    for timing_kind, calls in (("shape_pair_setup", 1),
                               ("shape_prewarm", 2),
                               ("shape_pair_forward", 4)):
        for call_index in range(calls):
            timings.append(dict(
                record_type="kv_operation", sample_id="b64-tiny-legal",
                operation="truncate", timing_kind=timing_kind,
                wall_seconds=0.003, cuda_seconds=0.001,
                call_index=call_index))
    _write_jsonl(measurement / "shape_timings.jsonl", timings)


def _assert_result(result: dict, tiny: float) -> dict:
    _require(result["sample_count"] == 4, "analysis sample count differs from NPZ rows")
    _require(result["branch_sample_count"] == 1, "branch row was not read")
    _require(set(result["by_mode"]) == {"narrow", "wide"}, "analysis modes are incomplete")

    narrow = result["by_mode"]["narrow"]
    cells = narrow["by_source_prefix"]
    zero = cells["source-a|64"]
    unsupported = cells["source-a|256"]
    mixed = cells["source-b|64"]
    _require(zero["r1_mean_given_accepted_p1"] == 0.0 and
             zero["beta_mean_condition_weighted"] is None and
             zero["p1_and_legal_f1_reject_probability"] == 0.0,
             "zero-R1 context did not produce a zero legal-event contribution")
    _require(unsupported["q2_invalid_rejection_mass_mean"] == 1.0 and
             unsupported["q2_invalid_rejection_event_fraction_r1_weighted"] == 1.0 and
             unsupported["p1_and_legal_f1_reject_probability"] == 0.0,
             "unsupported q2 context was not excluded from legal events")
    _require(_near(mixed["q2_invalid_rejection_mass_mean"], 0.5),
             "unweighted invalid-q2 context mean differs")
    _require(mixed["q2_invalid_rejection_event_fraction_r1_weighted"] > 0.999999999999,
             "invalid-q2 rejection fraction did not use R1 weights")
    _require(_near(mixed["legal_q2_r1_weighted_mean"], tiny / 2.0),
             "legal trigger did not use the context-wise R1 by legal-mass product")

    trigger = mixed["p1_and_legal_f1_reject_probability"]
    expected_trigger = 0.6 * tiny / 2.0
    _require(_near(trigger, expected_trigger, relative=1e-12),
             "legal event trigger differs from the R1-weighted cross moment")
    _require(trigger < 1e-16,
             "legal event trigger resembles the product of separate R1 and legal-mass means")

    branch = narrow["p2_after_second_accept"]
    branch_cell = branch["by_source_prefix"]["source-b|64"]
    expected_h = expected_trigger * 0.25
    _require(branch_cell["status"] == "estimated_from_sample" and
             _near(branch_cell["favorable_event_per_round"], expected_h, relative=1e-12),
             "branch H estimate did not multiply the legal trigger by sampled min(r1,q2)*P2")
    _require(_near(branch_cell["mean_second_accept_probability"], 1.0) and
             _near(branch_cell["mean_second_accept_and_p2_probability"], 0.25),
             "branch acceptance and joint P2 sample weights differ")
    _require(branch["source_prefix_coverage_complete"] is False and
             "source-b|256" in branch["missing_source_prefix_cells"],
             "branch output did not retain the missing source/prefix cell")

    _require(narrow["source_prefix_cells"] == 4 and
             narrow["source_prefix_cells_with_contexts"] == 3 and
             narrow["trigger_cell_coverage_complete"] is False and
             "source-b|256" in narrow["trigger_missing_cells"],
             "mixed source/prefix context coverage differs")
    _require(narrow["by_source"]["source-b"]["trigger_prefix_coverage_complete"] is False and
             narrow["by_source"]["source-b"]["trigger_missing_prefixes"] == [256],
             "source-level missing prefix was not reported")

    tiny_row = next(row for row in result["by_mode"]["narrow"]["by_source_prefix"].values()
                    if row["scene"] == "source-b" and row["prefix"] == 64)
    _require(tiny_row["r1_mean_given_accepted_p1"] > 0.0,
             "positive 2^-60 residual disappeared during NPZ analysis")

    shapes = result["target_shapes"]
    _require(shapes["short_2"]["runs"] == 1 and
             shapes["short_2"]["input_tokens"] == 2 and
             _near(shapes["short_2"]["host_wall_seconds_mean"], 0.01) and
             _near(shapes["short_2"]["cuda_seconds_mean"], 0.003),
             "single short-shape Target measurements include paired repetitions or wrong units")
    _require(shapes["next_4"]["runs"] == 1 and
             shapes["next_4"]["input_tokens"] == 4 and
             _near(shapes["next_4"]["host_wall_seconds_mean"], 0.03) and
             _near(shapes["next_4"]["cuda_seconds_mean"], 0.009),
             "single four-token Target measurements include paired repetitions or wrong units")
    paired = result["shape_pair_comparison"]
    _require(paired["planned_pair_ids"] == 2 and paired["complete_pairs"] == 2 and
             paired["paired_contexts"] == 1 and
             paired["short_then_next_pairs"] == 1 and
             paired["next_then_short_pairs"] == 1,
             "paired repetitions were not independently grouped")
    _require(_near(paired["host_next_minus_short_seconds_mean"], 0.25) and
             _near(paired["host_next_over_short_ratio_mean"], 2.0) and
             _near(paired["cuda_next_minus_short_seconds_mean"], 0.025),
             "paired forward means or seconds units differ")
    _require(result["shape_prewarm"]["short_2"]["calls"] == 1 and
             result["shape_prewarm"]["next_4"]["calls"] == 1,
             "prewarm rows were mixed with forward timing groups")
    restore = result["timing_restore_costs"]["shape_pair_forward"]
    _require(restore["calls"] == 4 and
             _near(restore["host_wall_seconds_mean"], 0.003) and
             _near(restore["cuda_seconds_mean"], 0.001),
             "paired KV restore cost or seconds units were lost")
    return dict(sample_count=result["sample_count"], branch_rows=result["branch_sample_count"],
                feature_dimensions=FEATURES, tiny_r1=tiny_row["r1_mean_given_accepted_p1"],
                legal_trigger=trigger, favorable_event_per_round=branch_cell[
                    "favorable_event_per_round"], paired_count=paired["complete_pairs"],
                short_2_seconds=shapes["short_2"]["host_wall_seconds_mean"],
                next_4_seconds=shapes["next_4"]["host_wall_seconds_mean"],
                missing_cell="source-b|256")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run analyze_cost_probe.py against synthetic NumPy CPU inputs.")
    parser.add_argument("--analyzer", type=Path,
                        default=Path(__file__).with_name("analyze_cost_probe.py"))
    args = parser.parse_args(argv)
    analyzer = args.analyzer.resolve()
    _require(analyzer.is_file(), f"analysis script not found: {analyzer}")
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="pf2x_npz_cpu_") as temporary:
        root = Path(temporary)
        collection = root / "collection"
        measurement = root / "measurement"
        output = root / "analysis.json"
        checkpoint = root / "synthetic-checkpoint-reference"
        tiny, _ = _build_collection(collection, np, checkpoint)
        _build_measurement(measurement)
        command = [
            sys.executable, str(analyzer),
            "--collection-dir", str(collection),
            "--measurement-dir", str(measurement),
            "--feature-checkpoint", str(checkpoint),
            "--feature-step", "synthetic-step-106000",
            "--output", str(output),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        completed = subprocess.run(command, check=False, capture_output=True,
                                   text=True, env=environment)
        if completed.returncode != 0:
            raise RuntimeError(
                f"analysis command failed ({completed.returncode})\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
        result = json.loads(output.read_text(encoding="utf-8"))
        summary = _assert_result(result, tiny)
    print(json.dumps(dict(passed=True, cpu_only=True, output=summary),
                     ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
