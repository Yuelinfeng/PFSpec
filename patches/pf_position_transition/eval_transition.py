#!/usr/bin/env python3
"""CPU-only planning and injected-runner interface for Position-transition evaluation."""

import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import random
import time


METHODS = ("T", "W1", "B", "C")
PREFIXES = (64, 256)
SEEDS = (39101, 39102)
EOS_ID = 12096
COUNTERS = (
    "target_calls", "target_input_tokens", "draft_calls", "draft_input_tokens",
    "ranker_calls", "rounds", "proposed", "verified", "accepted",
    "position_proposed", "position_verified", "position_accepted",
    "feature_proposed", "feature_verified", "feature_accepted",
    "position_fallbacks",
)
ZERO_COMPARE_FIELDS = (
    "output_tokens", "target_calls", "target_input_tokens", "ranker_calls",
    "draft_calls", "draft_input_tokens", "rounds", "proposed", "verified",
    "accepted", "position_proposed", "position_verified", "position_accepted",
    "feature_proposed", "feature_verified", "feature_accepted",
    "position_fallbacks", "accepted_lengths", "round_generated_lengths",
    "rewind_keep_trace",
)

PHASES = {
    "g3": dict(source_count=3, budget=256, context_count=12, run_count=48,
               wall_limit_seconds=900, stop_rule="C/W1 paired throughput < 1.05"),
    "g4": dict(source_count=10, budget=512, context_count=40, run_count=160,
               wall_limit_seconds=1500,
               stop_rule="C/W1 < 1.08 or C/T < 1.90"),
}


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Output already exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def _cyclic_method_order(index):
    shift = index % len(METHODS)
    return list(METHODS[shift:] + METHODS[:shift])


def read_dev_sources(manifest_path):
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Source manifest does not exist: " + str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise ValueError("Source manifest must contain a sources list")
    counts = Counter(str(row.get("split")) for row in data["sources"])
    expected = Counter(train=16, val=4, dev=10, confirmation=20)
    if counts != expected:
        raise ValueError("Source split counts differ from the supplied round manifest")
    ids = sorted(str(row.get("scene_id", "")).strip()
                 for row in data["sources"] if row.get("split") == "dev")
    if len(ids) != 10 or any(not value for value in ids) or len(set(ids)) != 10:
        raise ValueError("Manifest dev entries must contain ten unique scene IDs")
    return path, ids


def build_matrix_plan(phase, source_ids, source_manifest_path=None):
    if phase not in PHASES:
        raise ValueError("phase must be g3 or g4")
    config = PHASES[phase]
    sources = [str(value).strip() for value in source_ids]
    if len(sources) != config["source_count"] or any(not value for value in sources):
        raise ValueError(f"{phase} requires exactly {config['source_count']} supplied source IDs")
    if len(set(sources)) != len(sources):
        raise ValueError("Source IDs must be unique")

    contexts = []
    runs = []
    context_index = 0
    for source_index, source_id in enumerate(sources):
        for prefix in PREFIXES:
            for seed_index, seed in enumerate(SEEDS):
                pair_id = f"{source_index:02d}-{prefix:03d}-{seed}"
                order = _cyclic_method_order(context_index)
                contexts.append(dict(
                    pair_id=pair_id, source_id=source_id, source_index=source_index,
                    prefix_tokens=prefix, seed=seed, budget=config["budget"],
                    method_order=order,
                ))
                for order_index, method in enumerate(order):
                    runs.append(dict(
                        phase=phase, pair_id=pair_id, source_id=source_id,
                        source_index=source_index, prefix_tokens=prefix,
                        seed=seed, budget=config["budget"], method=method,
                        order_index=order_index, method_order=order,
                    ))
                context_index += 1

    expected_order_counts = {
        method: [config["context_count"] // len(METHODS)] * len(METHODS)
        for method in METHODS
    }
    if len(contexts) != config["context_count"] or len(runs) != config["run_count"]:
        raise RuntimeError("Planned context or run count differs from the phase definition")
    return dict(
        status="planned", phase=phase, methods=list(METHODS),
        source_ids_supplied_by_caller=sources,
        source_identity_status="IDs were selected from the dev split of the caller-supplied source manifest.",
        prefixes=list(PREFIXES), seeds=list(SEEDS), budget=config["budget"],
        expected_contexts=config["context_count"], expected_runs=config["run_count"],
        wall_limit_seconds=config["wall_limit_seconds"], stop_rule=config["stop_rule"],
        ordering=dict(
            method_position_counts=expected_order_counts,
            rule="Four cyclic rotations balance each method across the four run positions.",
            context_order="manifest dev order (ascending scene_id), then prefix, then seed",
        ),
        contexts=contexts, planned_runs=runs,
        runtime_inputs=dict(
            source_manifest=(str(source_manifest_path) if source_manifest_path else None),
            target_path=None, ranker_path=None,
            feature_checkpoint_path=None, selected_B_path=None, selected_C_path=None,
            note="Runtime paths and source membership are supplied after parent verification; this plan invents none.",
        ),
        timing=dict(
            row_wall="CUDA synchronize; perf_counter; one complete decode; CUDA synchronize; perf_counter",
            includes=["cache allocation", "prefix prefill", "Ranker", "W1 MLP", "B/C adapter",
                      "Target verification", "residual sampling", "rewind", "adapter close"],
            teacher_capture="No separate teacher collector or teacher-state capture runs in the timed path.",
            total_wall="Caller starts one monotonic deadline before imports, loading, warmup, matrix, and saves.",
        ),
    )


def build_zero_replay_plan(source_id, seed=39101, source_manifest_path=None):
    source_id = str(source_id).strip()
    if not source_id:
        raise ValueError("A caller-supplied source ID is required")
    cases = []
    for prefix in (0, 64, 256):
        for budget in (16, 64):
            pair_id = f"zero-{prefix:03d}-{budget:03d}-{seed}"
            for method in ("W1", "B_zero", "C_zero"):
                cases.append(dict(
                    pair_id=pair_id, source_id=source_id, prefix_tokens=prefix,
                    seed=int(seed), budget=budget, method=method,
                    zero_residual=(method != "W1"),
                ))
    return dict(
        status="planned", phase="zero_replay", source_id_from_manifest=source_id,
        source_manifest=(str(source_manifest_path) if source_manifest_path else None),
        prefix_tokens=[0, 64, 256], budgets=[16, 64], seed=int(seed),
        expected_runs=18, cases=cases,
        equality_fields=list(ZERO_COMPARE_FIELDS),
        note="Each prompt is W1, then B residual=0, then C residual=0 with the same seed.",
    )


def capture_rewind_trace(adapter):
    """Return a mutable keep-position trace and wrap only this adapter instance."""
    trace = []
    original = adapter.rewind

    def traced_rewind(keep):
        trace.append(int(keep))
        return original(keep)

    adapter.rewind = traced_rewind
    return trace


def expose_target_output(position_decode):
    """Copy the existing k=0 function and expose its already-built output list."""
    import inspect

    source = inspect.getsource(position_decode)
    if (source.count("def decode(") != 1
            or source.count("output = prefix[length:]") != 1
            or source.count("return _add_rates(stats)") != 1):
        raise RuntimeError("The supplied Position decoder source layout is unexpected")
    source = source.replace("def decode(", "def decode_with_output(", 1)
    source = source.replace("return _add_rates(stats)",
                             "return _add_rates(stats), output", 1)
    namespace = dict(position_decode.__globals__)
    exec(compile(source, position_decode.__code__.co_filename, "exec"), namespace)
    return namespace["decode_with_output"]


def timed_decode(method, context, prompt_tokens, call_method, cuda_synchronize):
    """Run one injected method and return a complete JSON-ready record."""
    prompt = [int(token) for token in prompt_tokens]
    if not prompt or prompt[0] != EOS_ID or len(prompt) != context["prefix_tokens"] + 1:
        raise ValueError("Prompt must be BOS plus exactly the requested complete P/F prefix")
    cuda_synchronize()
    started = time.perf_counter()
    try:
        result = call_method(method, prompt, context["budget"], context["seed"])
    finally:
        cuda_synchronize()
    outer_wall = time.perf_counter() - started
    if not isinstance(result, dict) or "stats" not in result or "output_tokens" not in result:
        raise TypeError("Decoder callback must return stats and output_tokens")
    output = [int(token) for token in result["output_tokens"]]
    if outer_wall <= 0:
        raise ValueError("Outer synchronized decode wall must be positive")
    if EOS_ID in output and (not output or output[-1] != EOS_ID or output.count(EOS_ID) != 1):
        raise ValueError("EOS must occur once and only at the end of generated output")
    stats = {str(key): value for key, value in result["stats"].items()}
    row = dict(
        phase=context["phase"], pair_id=context["pair_id"],
        source_id=context["source_id"], prefix_tokens=context["prefix_tokens"],
        prompt_tokens=prompt, seed=context["seed"], budget=context["budget"],
        method=method, method_order=context["method_order"],
        order_index=context["order_index"], output_tokens=output,
        generated_tokens=len(output), eos=bool(output and output[-1] == EOS_ID),
        last_token_id=(output[-1] if output else None),
        full_budget=(len(output) == context["budget"]),
        outer_wall_seconds=outer_wall,
        tokens_per_second=(len(output) / outer_wall if outer_wall else None),
        rewind_keep_trace=[int(value) for value in result.get("rewind_keep_trace", [])],
        cache_recoveries_derived=max(0, int(stats.get("verified", 0))
                                     - int(stats.get("accepted", 0))),
        cache_recoveries_basis="verified minus accepted; one rewind per rejected verification in the unified decoder",
        stats=stats,
    )
    json.dumps(row, ensure_ascii=False, allow_nan=False)
    return row


def compare_zero_replay(w1_row, candidate_row):
    """Compare a residual-zero B/C run with W1 at the same prompt and seed."""
    left_context = (w1_row["source_id"], w1_row["prefix_tokens"],
                    w1_row["seed"], w1_row["budget"])
    right_context = (candidate_row["source_id"], candidate_row["prefix_tokens"],
                     candidate_row["seed"], candidate_row["budget"])
    if left_context != right_context:
        raise ValueError("Replay rows do not share source, prefix, seed, and budget")
    left = dict(w1_row["stats"], output_tokens=w1_row["output_tokens"],
                rewind_keep_trace=w1_row["rewind_keep_trace"])
    right = dict(candidate_row["stats"], output_tokens=candidate_row["output_tokens"],
                 rewind_keep_trace=candidate_row["rewind_keep_trace"])
    differences = {key: dict(w1=left.get(key), candidate=right.get(key))
                   for key in ZERO_COMPARE_FIELDS if left.get(key) != right.get(key)}
    return dict(equal=not differences, differences=differences,
                candidate_method=candidate_row["method"],
                context_id=candidate_row["pair_id"])


def execute_matrix(plan, prompt_for, call_method, cuda_synchronize, records_path):
    """Execute a plan with caller-provided verified prompts, adapters, and models."""
    if plan.get("phase") not in PHASES:
        raise ValueError("execute_matrix accepts only G3 or G4 matrix plans")
    contexts = {context["pair_id"]: context for context in plan["contexts"]}
    records_path = Path(records_path)
    if records_path.exists():
        raise FileExistsError("Refusing to append to an existing run: " + str(records_path))
    rows = []
    runs_by_pair = {}
    for run in plan["planned_runs"]:
        runs_by_pair.setdefault(run["pair_id"], []).append(run)
    for pair_id, base_context in contexts.items():
        prompt = prompt_for(base_context["source_id"], base_context["prefix_tokens"])
        for run in runs_by_pair[pair_id]:
            context = dict(base_context, phase=run["phase"], order_index=run["order_index"])
            row = timed_decode(
                run["method"], context, prompt,
                lambda method, prompt_ids, budget, seed: call_method(
                    method, prompt_ids, budget, seed), cuda_synchronize,
            )
            append_jsonl(records_path, row)
            rows.append(row)
    summary = summarize_matrix(rows, plan["contexts"])
    summary.update(phase=plan["phase"], status="complete",
                   wall_limit_seconds=plan["wall_limit_seconds"])
    return summary


def execute_zero_replay(plan, prompt_for, call_method, cuda_synchronize,
                        records_path, comparison_path):
    """Run W1/B0/C0 rows through injected methods and save exact comparisons."""
    if plan.get("phase") != "zero_replay":
        raise ValueError("execute_zero_replay requires a zero-replay plan")
    records_path, comparison_path = Path(records_path), Path(comparison_path)
    if records_path.exists() or comparison_path.exists():
        raise FileExistsError("Replay output files must be new")
    comparisons = []
    grouped = {}
    for case in plan["cases"]:
        grouped.setdefault(case["pair_id"], []).append(case)
    for pair_id, cases in grouped.items():
        by_method = {case["method"]: case for case in cases}
        if set(by_method) != {"W1", "B_zero", "C_zero"}:
            raise RuntimeError("A replay comparison lacks one of its three methods")
        pair_rows = {}
        for order_index, method in enumerate(("W1", "B_zero", "C_zero")):
            case = by_method[method]
            context = dict(case, phase="zero_replay", order_index=order_index,
                           method_order=["W1", "B_zero", "C_zero"])
            prompt = prompt_for(context["source_id"], context["prefix_tokens"])
            row = timed_decode(
                method, context, prompt,
                lambda method_name, prompt_ids, budget, seed: call_method(
                    method_name, prompt_ids, budget, seed), cuda_synchronize,
            )
            append_jsonl(records_path, row)
            pair_rows[method] = row
            if method != "W1":
                comparison = compare_zero_replay(pair_rows["W1"], row)
                append_jsonl(comparison_path, comparison)
                comparisons.append(comparison)
                if not comparison["equal"]:
                    raise RuntimeError("Residual-zero replay differs from W1: " + pair_id)
    return dict(status="complete", pair_count=len(grouped),
                comparisons=comparisons, equal_all=all(row["equal"] for row in comparisons))


def _quantile(values, p):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * p
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    fraction = position - lo
    return values[lo] * (1.0 - fraction) + values[hi] * fraction


def _source_ratio_summary(source_log_means, bootstrap_seed=381012, reps=10000):
    valid = {key: value for key, value in source_log_means.items()
             if value is not None and math.isfinite(value)}
    if not valid:
        return dict(source_count=0, geometric_mean=None, ci95=None,
                    per_source_log_mean={})
    keys = sorted(valid)
    mean = math.fsum(valid[key] for key in keys) / len(keys)
    rng = random.Random(bootstrap_seed)
    draws = []
    for _ in range(reps):
        sample = [valid[rng.choice(keys)] for _ in keys]
        draws.append(math.exp(math.fsum(sample) / len(sample)))
    return dict(
        source_count=len(keys), geometric_mean=math.exp(mean),
        ci95=[_quantile(draws, .025), _quantile(draws, .975)],
        bootstrap_seed=bootstrap_seed, bootstrap_replicates=reps,
        per_source_ratio={key: math.exp(value) for key, value in valid.items()},
        per_source_log_mean=valid,
    )


def summarize_matrix(rows, planned_contexts, methods=METHODS):
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], {})[row["method"]] = row
    complete = {key: value for key, value in by_pair.items()
                if all(method in value for method in methods)}
    source_ids = sorted({context["source_id"] for context in planned_contexts})
    comparisons = {"W1_over_T": ("W1", "T"), "B_over_W1": ("B", "W1"),
                   "C_over_W1": ("C", "W1"), "C_over_T": ("C", "T")}
    by_method = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        lengths = {}
        for row in selected:
            key = str(int(row["generated_tokens"]))
            lengths[key] = lengths.get(key, 0) + 1
        total_tokens = sum(int(row["generated_tokens"]) for row in selected)
        total_wall = math.fsum(float(row["outer_wall_seconds"]) for row in selected)
        by_method[method] = dict(
            runs=len(selected), generated_tokens=total_tokens,
            outer_wall_seconds=total_wall,
            pooled_tokens_per_second=(total_tokens / total_wall if total_wall else None),
            eos_count=sum(bool(row["eos"]) for row in selected),
            output_length_counts=lengths,
            counter_sums={key: sum(int(row["stats"].get(key, 0)) for row in selected)
                          for key in COUNTERS},
        )

    ratios = {}
    for name, (numerator, denominator) in comparisons.items():
        source_values = {}
        full_source_values = {}
        for source_id in source_ids:
            logs = []
            full_logs = []
            for context in planned_contexts:
                pair = complete.get(context["pair_id"])
                if context["source_id"] != source_id or pair is None:
                    continue
                top, bottom = pair.get(numerator), pair.get(denominator)
                if top is None or bottom is None:
                    continue
                if top["outer_wall_seconds"] <= 0 or bottom["outer_wall_seconds"] <= 0:
                    continue
                top_tps = top["generated_tokens"] / top["outer_wall_seconds"]
                bottom_tps = bottom["generated_tokens"] / bottom["outer_wall_seconds"]
                if top_tps > 0 and bottom_tps > 0:
                    logs.append(math.log(top_tps / bottom_tps))
                if all(int(pair[method]["generated_tokens"])
                       == int(pair[method]["budget"]) for method in methods):
                    full_logs.append(math.log(
                        float(bottom["outer_wall_seconds"])
                        / float(top["outer_wall_seconds"])))
            source_values[source_id] = (math.fsum(logs) / len(logs)) if logs else None
            full_source_values[source_id] = (
                math.fsum(full_logs) / len(full_logs) if full_logs else None)
        top_tps = by_method[numerator]["pooled_tokens_per_second"]
        bottom_tps = by_method[denominator]["pooled_tokens_per_second"]
        ratios[name] = _source_ratio_summary(source_values)
        ratios[name]["pooled_tps_ratio_secondary"] = (
            top_tps / bottom_tps if top_tps is not None and bottom_tps else None)
        ratios[name]["common_full_budget"] = _source_ratio_summary(full_source_values)

    common_full = [pair for pair in complete.values()
                   if all(int(pair[method]["generated_tokens"]) == int(pair[method]["budget"])
                          for method in methods)]
    return dict(
        complete_contexts=len(complete), planned_contexts=len(planned_contexts),
        run_count=len(rows), method_run_count={method: sum(row["method"] == method for row in rows)
                                               for method in methods},
        by_method=by_method,
        common_full_budget_contexts=len(common_full),
        common_full_budget_sources=sorted({row["source_id"] for pair in common_full
                                           for row in pair.values()}),
        source_equal_geometric_speed_ratios=ratios,
        metric="Average paired log(tokens_per_second ratio) inside each source, then equal-source mean and exponentiate.",
        secondary="Pooled tokens and outer wall may be reported separately; EOS and output lengths remain explicit.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("g3", "g4", "zero-replay"), required=True)
    parser.add_argument("--source-manifest", type=Path, required=True,
                        help="Caller-supplied round manifest with verified split labels")
    parser.add_argument("--seed", type=int, default=39101,
                        help="Seed for the zero-residual replay plan")
    parser.add_argument("--out", type=Path, required=True,
                        help="New file path for the JSON plan; this CLI does not start models")
    args = parser.parse_args()
    manifest_path, dev_ids = read_dev_sources(args.source_manifest)
    if args.phase == "zero-replay":
        plan = build_zero_replay_plan(dev_ids[0], args.seed, manifest_path)
    else:
        source_ids = dev_ids[:3] if args.phase == "g3" else dev_ids
        plan = build_matrix_plan(args.phase, source_ids, manifest_path)
    write_json(args.out, plan)
    print(json.dumps(dict(status=plan["status"], phase=plan["phase"],
                          expected_runs=plan["expected_runs"],
                          output=str(args.out)), ensure_ascii=False))


if __name__ == "__main__":
    main()
