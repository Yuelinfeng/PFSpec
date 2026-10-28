"""Run a bounded same-context comparison of Target, old PF, and spatial S3."""

import argparse
import gc
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import sys
import time


PREFIXES = (64, 256)
SEED_BASES = (38101, 38102, 38103)
BUDGET = 512
METHODS = ("target_only", "old_pf", "spatial_s3")
EXPECTED_RUNS = 180
EXPECTED_PAIRS = 60
EXPECTED_SOURCES = 10
EXPECTED_CONTEXTS_PER_SOURCE = 6
EXPECTED_Q_ROWS = 480
COUNTER_FIELDS = (
    "target_calls", "target_input_tokens", "draft_calls", "draft_input_tokens",
    "ranker_calls", "rounds", "proposed", "verified", "accepted",
    "position_proposed", "position_verified", "position_accepted",
    "feature_proposed", "feature_verified", "feature_accepted",
    "position_fallbacks", "mlp_calls", "s3_calls", "cache_recoveries",
)
REPLAY_COMPARE_FIELDS = tuple(key for key in COUNTER_FIELDS if key != "s3_calls")
PHASE_LIMITS = {"load_and_parity": 250.0, "replay_and_warmup": 400.0,
                "matrix": 1750.0, "save": 1800.0}


class DeadlineReached(RuntimeError):
    pass


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def jsonable(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        return jsonable(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        return jsonable(value.item())
    raise TypeError("Value cannot be written as JSON: " + type(value).__name__)


def path_from_project(value, project):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def check_stop(deadline, stop_signal, stage):
    if stop_signal["name"]:
        raise InterruptedError("stop requested by " + stop_signal["name"] + " during " + stage)
    if time.monotonic() >= deadline:
        raise DeadlineReached("shared time limit reached during " + stage)


def crop_source(record):
    return record.get("scene") or Path(record["path"]).parents[1].name


def select_records(raw, validate):
    selected = {}
    for record in raw:
        validate(record)
        source = crop_source(record)
        if len(record["tokens"]) >= 256 and source not in selected:
            selected[source] = dict(record, scene=source)
    records = [selected[source] for source in sorted(selected)]
    if len(records) != EXPECTED_SOURCES:
        raise ValueError("Expected ten existing development sources with at least 256 tokens; got "
                         + str(len(records)))
    return records


def make_plan(records):
    permutations = list(itertools.permutations(METHODS))
    groups = []
    order_counts = {"|".join(order): 0 for order in permutations}
    order_index = 0
    for source_index, record in enumerate(records):
        for prefix_index, prefix in enumerate(PREFIXES):
            for seed_index, seed_base in enumerate(SEED_BASES):
                pair_id = f"{source_index:02d}-{prefix:03d}-{seed_base}"
                derived_seed = seed_base + int(record["crop_seed"]) * 1009 + prefix * 17
                method_order = permutations[order_index % len(permutations)]
                order_counts["|".join(method_order)] += 1
                groups.append(dict(
                    pair_id=pair_id, source=record["scene"], source_index=source_index,
                    crop_seed=int(record["crop_seed"]), prefix=prefix,
                    prefix_index=prefix_index, seed_index=seed_index,
                    seed_base=seed_base, derived_seed=derived_seed, budget=BUDGET,
                    method_order=list(method_order), permutation_index=order_index % 6,
                    input_path=str(record.get("path", "")),
                ))
                order_index += 1
    if len(groups) != EXPECTED_PAIRS:
        raise RuntimeError("Internal context count differs from sixty")
    if set(order_counts.values()) != {10}:
        raise RuntimeError("The six method orders are not evenly represented")
    return groups, order_counts


def locate_n2(root):
    root = Path(root).resolve()
    candidates = (root, root / "run01" / "experiment02", root / "experiment02")
    experiment = next((path for path in candidates
                       if (path / "model_S3_seed_29.npz").is_file()
                       and (path / "condition_inputs.npz").is_file()), None)
    if experiment is None:
        raise FileNotFoundError("N2 S3 seed 29 files are missing under " + str(root))
    prepared_candidates = (
        experiment / "prepared",
        experiment.parent / "prepared",
        root / "run01" / "prepared",
        root / "prepared",
    )
    prepared = next((path for path in prepared_candidates
                     if (path / "prepared_data.npz").is_file()
                     and (path / "rows.jsonl").is_file()), None)
    if prepared is None:
        raise FileNotFoundError("N2 prepared_data.npz or rows.jsonl is missing under " + str(root))
    q_outputs = experiment / "q_outputs.npz"
    if not q_outputs.is_file():
        raise FileNotFoundError("N2 q_outputs.npz is missing: " + str(q_outputs))
    return dict(
        root=str(root), experiment=experiment,
        model=experiment / "model_S3_seed_29.npz",
        condition=experiment / "condition_inputs.npz",
        q_outputs=q_outputs, prepared=prepared,
        rows=prepared / "rows.jsonl",
    )


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_target_decode(position_decode):
    """Expose the output already built by the existing k=0 decoder."""
    import inspect

    source = inspect.getsource(position_decode)
    if source.count("def decode(") != 1 or source.count("return _add_rates(stats)") != 1:
        raise RuntimeError("The current Position decoder source layout differs")
    source = source.replace("def decode(", "def decode_with_output(", 1)
    source = source.replace("return _add_rates(stats)",
                             "return _add_rates(stats), output", 1)
    namespace = dict(position_decode.__globals__)
    exec(compile(source, position_decode.__code__.co_filename, "exec"), namespace)
    return namespace["decode_with_output"]


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(rows):
    if not rows:
        return dict(
            runs=0, generated_tokens=0, outer_wall_seconds=0.0,
            pooled_tokens_per_second=None, eos_count=0, eos_rate=None,
            mean_length=None, output_length_counts={}, counters={key: 0 for key in COUNTER_FIELDS},
        )
    generated = sum(int(row["generated_tokens"]) for row in rows)
    seconds = sum(float(row["outer_wall_seconds"]) for row in rows)
    lengths = {}
    for row in rows:
        key = str(int(row["generated_tokens"]))
        lengths[key] = lengths.get(key, 0) + 1
    eos_count = sum(bool(row["eos"]) for row in rows)
    counters = {key: sum(int(row.get(key, 0)) for row in rows) for key in COUNTER_FIELDS}
    return dict(
        runs=len(rows), generated_tokens=generated, outer_wall_seconds=seconds,
        pooled_tokens_per_second=(generated / seconds if seconds else None),
        eos_count=eos_count, eos_rate=eos_count / len(rows),
        mean_length=generated / len(rows), output_length_counts=lengths,
        counters=counters,
    )


def ratio_summary(source_log_means):
    valid = {source: value for source, value in source_log_means.items()
             if value is not None and math.isfinite(value)}
    if not valid:
        return dict(source_count=0, expected_sources=EXPECTED_SOURCES,
                    source_equal_geometric_mean=None, source_bootstrap_95_ci=None,
                    source_bootstrap_replicates=0, per_source={})
    names = sorted(valid)
    geo = math.exp(math.fsum(valid[name] for name in names) / len(names))
    samples = []
    if len(names) >= 2:
        rng = random.Random(381012)
        for _ in range(10000):
            drawn = [valid[rng.choice(names)] for _ in names]
            samples.append(math.exp(math.fsum(drawn) / len(drawn)))
    return dict(
        source_count=len(names), expected_sources=EXPECTED_SOURCES,
        all_sources_complete=(len(names) == EXPECTED_SOURCES),
        source_equal_geometric_mean=geo,
        source_bootstrap_seed=381012,
        source_bootstrap_95_ci=([quantile(samples, .025), quantile(samples, .975)]
                                if samples else None),
        source_bootstrap_replicates=len(samples),
        per_source={name: math.exp(value) for name, value in valid.items()},
        per_source_mean_log_ratio=valid,
    )


def trace_rewinds(adapter):
    trace = []
    original = adapter.rewind

    def wrapped(keep):
        trace.append(int(keep))
        return original(keep)

    adapter.rewind = wrapped
    return trace


def build_summary(rows, groups, order_counts, status, started_wall, started_mono,
                  deadline_seconds, stage_seconds):
    pair_rows = {}
    for row in rows:
        pair_rows.setdefault(row["pair_id"], {})[row["method"]] = row
    complete_pairs = {
        pair_id: value for pair_id, value in pair_rows.items()
        if set(value) == set(METHODS)
    }
    group_by_id = {group["pair_id"]: group for group in groups}
    source_names = sorted({group["source"] for group in groups})
    by_method = {method: describe([row for row in rows if row["method"] == method])
                 for method in METHODS}
    by_source = {}
    source_methods = {}
    for source in source_names:
        source_groups = [group for group in groups if group["source"] == source]
        source_complete = [complete_pairs[group["pair_id"]] for group in source_groups
                           if group["pair_id"] in complete_pairs]
        source_methods[source] = {
            method: describe([pair[method] for pair in source_complete])
            for method in METHODS
        }
        by_source[source] = dict(
            complete_pairs=len(source_complete), expected_pairs=EXPECTED_CONTEXTS_PER_SOURCE,
            by_method=source_methods[source],
        )

    contrasts = {
        "spatial_s3_over_old_pf": ("spatial_s3", "old_pf"),
        "old_pf_over_target_only": ("old_pf", "target_only"),
        "spatial_s3_over_target_only": ("spatial_s3", "target_only"),
    }
    ratios = {}
    for name, (numerator, denominator) in contrasts.items():
        source_log_means = {}
        pooled_source_ratios = {}
        for source in source_names:
            values = []
            source_complete_pairs = [complete_pairs[group["pair_id"]]
                                     for group in groups
                                     if group["source"] == source
                                     and group["pair_id"] in complete_pairs]
            for pair in source_complete_pairs:
                up_tokens = int(pair[numerator]["generated_tokens"])
                down_tokens = int(pair[denominator]["generated_tokens"])
                up_wall = float(pair[numerator]["outer_wall_seconds"])
                down_wall = float(pair[denominator]["outer_wall_seconds"])
                if up_tokens > 0 and down_tokens > 0 and up_wall > 0 and down_wall > 0:
                    values.append(math.log((up_tokens / up_wall) / (down_tokens / down_wall)))
            source_log_means[source] = (math.fsum(values) / len(values) if values else None)
            up = source_methods[source][numerator]["pooled_tokens_per_second"]
            down = source_methods[source][denominator]["pooled_tokens_per_second"]
            pooled_source_ratios[source] = up / down if up is not None and down else None
        primary = ratio_summary(source_log_means)
        pooled_valid = {source: value for source, value in pooled_source_ratios.items()
                        if value is not None and value > 0}
        pooled_geo = (math.exp(math.fsum(math.log(value) for value in pooled_valid.values())
                               / len(pooled_valid)) if pooled_valid else None)
        paired_up = describe([pair[numerator] for pair in complete_pairs.values()])
        paired_down = describe([pair[denominator] for pair in complete_pairs.values()])
        overall_up = paired_up["pooled_tokens_per_second"]
        overall_down = paired_down["pooled_tokens_per_second"]
        ratios[name] = dict(
            primary_pair_log=primary,
            pooled_secondary=dict(
                pooled_tps_ratio=(overall_up / overall_down
                                  if overall_up is not None and overall_down else None),
                source_equal_geometric_mean=pooled_geo,
                per_source=pooled_valid,
                basis="ratio of summed generated tokens per summed synchronized outer wall across complete pairs; secondary only",
            ),
        )

    full_pair_ids = [pair_id for pair_id, pair in complete_pairs.items()
                     if all(int(pair[method]["generated_tokens"]) == BUDGET
                            for method in METHODS)]
    full_pairs = [complete_pairs[pair_id] for pair_id in full_pair_ids]
    common_rows = {method: [pair[method] for pair in full_pairs] for method in METHODS}
    completed_orders = {key: 0 for key in order_counts}
    for pair_id in complete_pairs:
        group = group_by_id[pair_id]
        completed_orders["|".join(group["method_order"])] += 1
    elapsed = time.monotonic() - started_mono
    full_ratio_logs = {}
    for name, (numerator, denominator) in contrasts.items():
        values = {}
        for source in source_names:
            logs = []
            for group in groups:
                pair = complete_pairs.get(group["pair_id"])
                if group["source"] != source or pair is None:
                    continue
                if not all(int(pair[method]["generated_tokens"]) == BUDGET for method in METHODS):
                    continue
                up_wall = float(pair[numerator]["outer_wall_seconds"])
                down_wall = float(pair[denominator]["outer_wall_seconds"])
                if up_wall > 0 and down_wall > 0:
                    logs.append(math.log(down_wall / up_wall))
            values[source] = math.fsum(logs) / len(logs) if logs else None
        full_ratio_logs[name] = ratio_summary(values)

    return dict(
        status=status, started_at_unix=started_wall, elapsed_seconds=elapsed,
        deadline_seconds=deadline_seconds, within_deadline=(elapsed <= deadline_seconds),
        run_count=len(rows), expected_run_count=EXPECTED_RUNS,
        completed_contexts=len(complete_pairs), expected_contexts=EXPECTED_PAIRS,
        source_scope=("Ten existing development scenes shared with the N2 source list; only matrix seeds and timing are new. This is not an independent-source assessment."),
        method_order_counts={"planned": order_counts, "complete_contexts": completed_orders},
        by_method=by_method, by_source=by_source,
        source_equal_geometric_speed_ratios=ratios,
        common_full_512=dict(
            contexts=len(full_pairs),
            source_coverage=sorted({group_by_id[pair_id]["source"] for pair_id in full_pair_ids}),
            by_method={method: describe(common_rows[method]) for method in METHODS},
            source_equal_geometric_wall_speed_ratios=full_ratio_logs,
            condition="all three methods returned exactly 512 tokens; EOS at token 512 is reported separately",
        ),
        stage_seconds=stage_seconds,
        timing_scope=("Synchronized outer wall per decode, including cache creation, prefill, proposal, Target verification, recovery, sampling, adapter preparation and close. Setup, 480-row CUDA consistency, replay, warmup, result writing, and cleanup share the run deadline but are reported separately."),
        seed_rule="derived_seed = seed_base + crop_seed * 1009 + prefix * 17; methods in one context share the same prompt and seed",
        capacity_note="Target k=0 retains its original length+budget+2 cache capacity; PF-K2 retains its existing length+budget+6 capacity.",
        phase_limits_seconds=PHASE_LIMITS,
        primary_metric=("For each complete pair, take log(tps numerator / tps denominator); average six pair logs within each source, then take the equal-source mean and exponentiate. Bootstrap resamples ten sources 10,000 times with seed 381012."),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="GaussianGPT project directory")
    parser.add_argument("--crops", required=True, help="Existing development crops file")
    parser.add_argument("--feature-checkpoint", required=True, help="Existing old PF Feature MLP weights")
    parser.add_argument("--n2-artifact-root", required=True, help="N2 run artifact directory")
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--deadline-seconds", type=float, default=1800,
                        help="One shared time limit for imports, validation, replay, warmup, matrix, and saves")
    args = parser.parse_args()
    if args.deadline_seconds <= 0 or args.deadline_seconds > PHASE_LIMITS["save"]:
        parser.error("--deadline-seconds must be in (0, 1800]")

    started_wall = time.time()
    started_mono = time.monotonic()
    deadline = started_mono + args.deadline_seconds
    project = Path(args.project).expanduser().resolve()
    crops = path_from_project(args.crops, project)
    feature_checkpoint = path_from_project(args.feature_checkpoint, project)
    n2_root = path_from_project(args.n2_artifact_root, project)
    out = path_from_project(args.out, project)
    if out.exists():
        parser.error("--out must name a new directory: " + str(out))
    out.mkdir(parents=True)

    stop_signal = {"name": None}
    previous_handlers = {}

    def request_stop(signum, _frame):
        stop_signal["name"] = signal.Signals(signum).name

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, request_stop)

    rows = []
    groups = []
    order_counts = {}
    stage_seconds = {}
    pipeline = dict(
        status="running", current_stage="startup", project=str(project), crops=str(crops),
        feature_checkpoint=str(feature_checkpoint), n2_artifact_root=str(n2_root),
        output=str(out), started_at_unix=started_wall,
        deadline_unix=started_wall + args.deadline_seconds,
        deadline_seconds=args.deadline_seconds, completed_runs=0,
        expected_runs=EXPECTED_RUNS,
    )

    def update(stage, status="running"):
        elapsed = time.monotonic() - started_mono
        pipeline.update(
            status=status, current_stage=stage, completed_runs=len(rows),
            elapsed_seconds=elapsed,
            remaining_seconds=max(0.0, args.deadline_seconds - elapsed),
            stage_seconds=stage_seconds,
        )
        write_json(out / "pipeline.json", pipeline)
        write_json(out / "progress.json", dict(
            status=status, stage=stage, completed_runs=len(rows),
            expected_runs=EXPECTED_RUNS, elapsed_seconds=elapsed,
            remaining_seconds=max(0.0, args.deadline_seconds - elapsed),
        ))

    def check_phase_limit(limit, name):
        elapsed = time.monotonic() - started_mono
        if elapsed >= min(float(limit), args.deadline_seconds):
            raise DeadlineReached("cumulative " + name + " limit reached at "
                                  + f"{elapsed:.3f}s")

    def stage(name, fn, phase_limit=None, phase_name=None):
        check_stop(deadline, stop_signal, name + " start")
        if phase_limit is not None:
            check_phase_limit(phase_limit, phase_name or name)
        update(name)
        begin = time.monotonic()
        try:
            result = fn()
        finally:
            stage_seconds[name] = stage_seconds.get(name, 0.0) + (time.monotonic() - begin)
            update(name)
        if phase_limit is not None:
            check_phase_limit(phase_limit, phase_name or name)
        return result

    def save_summary(summary_status):
        summary = build_summary(rows, plan_contexts, order_counts, summary_status,
                                started_wall, started_mono, args.deadline_seconds, stage_seconds)
        begin = time.monotonic()
        write_json(out / "summary.json", summary)
        duration = time.monotonic() - begin
        stage_seconds["summary_write"] = duration
        summary["stage_seconds"] = dict(stage_seconds)
        summary["elapsed_seconds"] = time.monotonic() - started_mono
        summary["within_deadline"] = summary["elapsed_seconds"] <= args.deadline_seconds
        second_begin = time.monotonic()
        write_json(out / "summary.json", summary)
        stage_seconds["summary_write"] = duration + time.monotonic() - second_begin
        summary["stage_seconds"] = dict(stage_seconds)
        summary["elapsed_seconds"] = time.monotonic() - started_mono
        summary["within_deadline"] = summary["elapsed_seconds"] <= args.deadline_seconds
        write_json(out / "summary.json", summary)

    status = "failed"
    plan_contexts = []
    prompts = []
    verification = dict(saved_distribution=None, zero_residual_replays=[],
                        spatial_replays=[], warmups=[])

    try:
        check_stop(deadline, stop_signal, "imports")
        sys.path.insert(0, str(project))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        os.chdir(project)

        def import_runtime():
            import torch
            from joblib import load
            from threadpoolctl import threadpool_limits
            from bench_fast_hgb import FastHistRanker
            from bench_gaussiangpt_pf_block2_cow import load_feature_model
            from draft_a import feature_target_v2
            from draft_a.corpus_v2 import EOS, load_target, validate
            from draft_a.learning_eval import _fp32_runtime
            from draft_a.pf_feature_unified import MLPAdapter, decode
            from draft_a.pf_ranker_eval import decode as position_decode
            from spatial_online_adapter import SpatialAdapter, validate_saved_distribution

            return dict(
                torch=torch, load=load, threadpool_limits=threadpool_limits,
                FastHistRanker=FastHistRanker, load_feature_model=load_feature_model,
                feature_target_v2=feature_target_v2, EOS=EOS, load_target=load_target,
                validate=validate, fp32_runtime=_fp32_runtime,
                MLPAdapter=MLPAdapter, pf_decode=decode, position_decode=position_decode,
                SpatialAdapter=SpatialAdapter,
                validate_saved_distribution=validate_saved_distribution,
            )

        runtime = stage("imports", import_runtime,
                        phase_limit=PHASE_LIMITS["load_and_parity"],
                        phase_name="load_and_parity")
        torch = runtime["torch"]
        torch.set_num_threads(1)

        def load_cpu_inputs():
            if not project.is_dir() or not crops.is_file():
                raise FileNotFoundError("Project directory or crop file is missing")
            if not feature_checkpoint.is_file():
                raise FileNotFoundError("Old PF Feature checkpoint is missing: " + str(feature_checkpoint))
            checkpoint = torch.load(feature_checkpoint, map_location="cpu", weights_only=True)
            feature_step = int(checkpoint.get("step", checkpoint.get("global_step", -1)))
            if feature_step != 12000:
                raise ValueError("Expected the existing step-12000 Feature MLP; found "
                                 + str(feature_step))
            del checkpoint
            n2 = locate_n2(n2_root)
            saved_rows = read_jsonl(n2["rows"])
            if len(saved_rows) != EXPECTED_Q_ROWS:
                raise ValueError("Expected 480 prepared N2 rows: " + str(n2["rows"]))
            raw = torch.load(crops, map_location="cpu", weights_only=True)
            records = select_records(raw, runtime["validate"])
            crop_sources = {record["scene"] for record in records}
            saved_sources = {str(row["source_id"]) for row in saved_rows}
            if saved_sources != crop_sources:
                raise ValueError("The 480 N2 rows and selected crop sources differ")
            groups, counts = make_plan(records)
            context_map = {}
            for source_index, record in enumerate(records):
                for prefix_index, prefix in enumerate(PREFIXES):
                    tokens = record["tokens"][:prefix]
                    prompt = torch.cat((torch.tensor([runtime["EOS"]]), tokens)).tolist()
                    context_id = f"{source_index:02d}-{prefix:03d}"
                    context_map[(record["scene"], prefix)] = dict(
                        context_id=context_id, source=record["scene"], source_index=source_index,
                        crop_seed=int(record["crop_seed"]), prefix=prefix,
                        input_path=str(record.get("path", "")), prompt=[int(token) for token in prompt],
                    )
            return records, groups, counts, n2, context_map, feature_step, saved_sources

        records, groups, order_counts, n2, context_map, feature_step, n2_sources = stage(
            "load_cpu_inputs", load_cpu_inputs,
            phase_limit=PHASE_LIMITS["load_and_parity"], phase_name="load_and_parity")
        for group in groups:
            context = context_map[(group["source"], group["prefix"])]
            group["prompt_context_id"] = context["context_id"]
        plan_contexts = groups
        prompts = sorted(context_map.values(), key=lambda item: (item["source_index"], item["prefix"]))
        write_json(out / "plan.json", dict(
            project=str(project), crops=str(crops), feature_checkpoint=str(feature_checkpoint),
            n2_artifacts={key: str(value) for key, value in n2.items()},
            sources=[dict(scene=record["scene"], crop_seed=int(record["crop_seed"]),
                          path=str(record.get("path", "")), token_count=len(record["tokens"]))
                     for record in records],
            prefixes=list(PREFIXES), seed_bases=list(SEED_BASES), budget=BUDGET,
            methods=list(METHODS), planned_runs=EXPECTED_RUNS, context_count=EXPECTED_PAIRS,
            feature_checkpoint_step=feature_step,
            n2_source_ids=sorted(n2_sources),
            order_counts=order_counts,
            order_rule="Each of the six method permutations is used ten times, cycling by context order.",
            order_plan_note="The webpage plan listed three rotations by seed index. This runner uses the separately directed six-permutation balance; prompts, methods, seeds, and pair membership are unchanged.",
            derived_seed_rule="seed_base + crop_seed * 1009 + prefix * 17",
            source_scope="Ten existing development scenes shared with the N2 source list; only matrix seeds and timing are new.",
            timing_scope="All setup, saved-distribution evaluation, replay, warmup, 180 matrix calls, output writes, and cleanup share the deadline.",
            parity_limits=dict(max_abs=1e-5, total_variation=1e-4),
            phase_limits_seconds=PHASE_LIMITS,
            cache_recoveries_basis="Derived as verified minus accepted; each rejected verification causes one rewind in the current decoder.",
            neighbor_lookup_timing="Not separately synchronized; lookup cost remains inside outer wall time.",
            target_mode="Existing draft_a.pf_ranker_eval.decode with k=0; only its already-built output is exposed.",
            capacity_note="Target k=0 retains length+budget+2. PF-K2 retains its existing length+budget+6.",
        ))
        write_json(out / "prompts.json", dict(
            bos_id=runtime["EOS"],
            contexts=prompts,
        ))

        torch_mod = runtime["torch"]
        device_name = None
        if not torch_mod.cuda.is_available():
            raise RuntimeError("CUDA is required for this online comparison")

        fp32_context = runtime["fp32_runtime"](torch_mod)
        with torch_mod.inference_mode(), fp32_context, runtime["threadpool_limits"](
                limits=1, user_api="openmp"):
            if (torch_mod.get_float32_matmul_precision() != "highest"
                    or torch_mod.backends.cuda.matmul.allow_tf32
                    or torch_mod.backends.cudnn.allow_tf32):
                raise RuntimeError("FP32 highest precision with TF32 disabled was not established")
            device_name = torch_mod.cuda.get_device_name(0)
            pipeline["hardware_software"] = dict(
                python=sys.version, platform=platform.platform(), torch=torch_mod.__version__,
                cuda_runtime=torch_mod.version.cuda, cuda_device=device_name,
                visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                matmul_precision=torch_mod.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=torch_mod.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch_mod.backends.cudnn.allow_tf32,
                target_precision="FP32", old_pf_precision="FP32", spatial_s3_precision="FP32",
                temperature=0.9, top_p=0.9,
            )

            def evaluate_saved_q():
                result = runtime["validate_saved_distribution"](
                    n2["model"], n2["condition"], n2["prepared"],
                    n2["q_outputs"], n2["rows"], device="cuda",
                    max_abs_limit=1e-5, tv_limit=1e-4,
                )
                return jsonable(result)

            verification["saved_distribution"] = stage(
                "n2_cuda_consistency_480", evaluate_saved_q,
                phase_limit=PHASE_LIMITS["load_and_parity"], phase_name="load_and_parity")
            write_json(out / "verification.json", jsonable(verification))

            def load_online_models():
                target_wrapper = runtime["load_target"]().float().eval().requires_grad_(False)
                target = target_wrapper.transformer
                ranks = target.sparse_position_rank_map.detach().cpu().tolist()
                if len(ranks) != 8000:
                    raise ValueError("Expected 8000 sparse Position ranks")
                head = path_from_project(runtime["feature_target_v2"].HEAD, project)
                if not head.is_file():
                    raise FileNotFoundError("Frozen Feature head is missing: " + str(head))
                feature, temperature, top_p = runtime["load_feature_model"](
                    feature_checkpoint, head, target, torch_mod.device("cuda"), 8000, 12096,
                )
                if (temperature, top_p) != (0.9, 0.9):
                    raise ValueError("Old PF checkpoint policy must be temperature=.9 and top_p=.9")
                ranker_path = path_from_project(runtime["feature_target_v2"].RANKER, project)
                if not ranker_path.is_file():
                    raise FileNotFoundError("Position ranker is missing: " + str(ranker_path))
                ranker = runtime["FastHistRanker"](runtime["load"](ranker_path))
                return target_wrapper, target, feature, ranker, ranks, head, ranker_path

            (target_wrapper, target, feature, ranker, ranks,
             head_path, ranker_path) = stage(
                "load_target_ranker_old_pf", load_online_models,
                phase_limit=PHASE_LIMITS["load_and_parity"], phase_name="load_and_parity")
            write_json(out / "rank_map.json", dict(
                source="Target.transformer.sparse_position_rank_map",
                values=[int(value) for value in ranks],
            ))
            pipeline["head_path"] = str(head_path)
            pipeline["ranker_path"] = str(ranker_path)
            pipeline["target_core"] = type(target).__name__
            update("load_target_ranker_old_pf")

            def make_prompt(group):
                return context_map[(group["source"], group["prefix"])]["prompt"]

            def ensure_fp32_runtime():
                if (torch_mod.get_float32_matmul_precision() != "highest"
                        or torch_mod.backends.cuda.matmul.allow_tf32
                        or torch_mod.backends.cudnn.allow_tf32):
                    raise RuntimeError("FP32 highest precision with TF32 disabled changed")

            def decode_once(method, prompt, budget, seed, adapter=None, check_before=True):
                if check_before:
                    check_stop(deadline, stop_signal, "decode " + method)
                ensure_fp32_runtime()
                torch_mod.cuda.synchronize()
                begin = time.perf_counter()
                if method == "target_only":
                    with torch_mod.inference_mode(), torch_mod.autocast(device_type="cuda", enabled=False):
                        stats, output = runtime["target_decode"](
                            target, None, ranker, ranks, prompt, budget, 0, seed,
                        )
                else:
                    stats, output = runtime["pf_decode"](
                        target, adapter, ranker, ranks, prompt, budget, seed,
                    )
                torch_mod.cuda.synchronize()
                wall = time.perf_counter() - begin
                tokens = [int(token) for token in output]
                stats = dict(stats)
                row = {key: int(stats.get(key, 0)) for key in COUNTER_FIELDS}
                if adapter is not None:
                    row["mlp_calls"] = int(getattr(adapter, "forward_calls",
                                                    stats.get("draft_calls", 0)))
                    row["s3_calls"] = int(getattr(adapter, "s3_calls", 0))
                row["cache_recoveries"] = max(
                    0, int(stats.get("verified", 0)) - int(stats.get("accepted", 0)))
                row["cache_recoveries_basis"] = (
                    "derived: verified minus accepted; one rewind for each rejected verification")
                ended = bool(tokens and tokens[-1] == runtime["EOS"])
                row.update(
                    generated_tokens=len(tokens), eos=ended,
                    terminated_by_eos=ended,
                    last_token_id=(tokens[-1] if tokens else None),
                    full_budget=(len(tokens) == budget),
                    decoder_seconds=float(stats.get("seconds", 0.0)),
                    decoder_tokens_per_second=stats.get("tokens_per_second"),
                    mean_generated_per_round=stats.get("mean_generated_per_round"),
                    variant=stats.get("variant", method),
                    feature_precision=stats.get("feature_precision", "not_applicable"),
                    neighbor_lookup_ms=None,
                    neighbor_lookup_timing="included in outer wall; no separate synchronization",
                    output_tokens=tokens,
                )
                row["outer_wall_seconds"] = wall
                row["tokens_per_second"] = len(tokens) / wall if wall else None
                return row

            def add_context(row, group, method, order_index, phase="matrix"):
                row.update(
                    phase=phase, pair_id=group["pair_id"], source=group["source"],
                    source_id=group["source"],
                    source_index=group["source_index"], crop_seed=group["crop_seed"],
                    prefix=group["prefix"], prefix_index=group["prefix_index"],
                    seed_base=group["seed_base"], derived_seed=group["derived_seed"],
                    seed_index=group["seed_index"], budget=group["budget"],
                    requested_new_tokens=group["budget"], method=method,
                    method_order=group["method_order"], order_index=order_index,
                    prompt_context_id=group["prompt_context_id"],
                    temperature=0.9, top_p=0.9, target_precision="FP32",
                )
                return row

            def replay_zero_and_spatial():
                zero_rows = []
                spatial_rows = []
                zero_adapter = runtime["SpatialAdapter"](
                    feature, n2["model"], n2["condition"], n2["prepared"],
                    zero_residual=True, bf16=False, device="cuda",
                )
                zero_rewind = zero_adapter.rewind
                try:
                    for source_index, prefix in ((0, 64), (1, 256)):
                        record = records[source_index]
                        ctx = context_map[(record["scene"], prefix)]
                        seed = 38101 + 900000 + int(record["crop_seed"]) * 1009 + prefix * 17
                        prompt = ctx["prompt"]
                        old_adapter = runtime["MLPAdapter"](feature)
                        old_trace = trace_rewinds(old_adapter)
                        zero_trace = trace_rewinds(zero_adapter)
                        old_row = decode_once("old_pf", prompt, 16, seed, adapter=old_adapter)
                        old_row["rewind_keep_trace"] = list(old_trace)
                        zero_row = decode_once("spatial_zero_residual", prompt, 16, seed,
                                               adapter=zero_adapter)
                        zero_row["rewind_keep_trace"] = list(zero_trace)
                        old_equal = (old_row["output_tokens"] == zero_row["output_tokens"]
                                     and all(old_row.get(key) == zero_row.get(key)
                                             for key in REPLAY_COMPARE_FIELDS)
                                     and old_trace == zero_trace)
                        if not old_equal:
                            raise RuntimeError("Zero-residual replay differs from old PF at "
                                               + ctx["context_id"])
                        zero_rows.append(dict(
                            context_id=ctx["context_id"], source=record["scene"], prefix=prefix,
                            seed=seed, old_pf=old_row, zero_residual=zero_row,
                            rewind_keep_trace=zero_trace, equal=True,
                        ))
                        old_adapter.close()
                        del old_adapter
                        zero_adapter.rewind = zero_rewind
                finally:
                    zero_adapter.close()
                    del zero_adapter
                    gc.collect()
                    torch_mod.cuda.empty_cache()

                spatial_adapter = runtime["SpatialAdapter"](
                    feature, n2["model"], n2["condition"], n2["prepared"],
                    zero_residual=False, bf16=False, device="cuda",
                )
                try:
                    for source_index, prefix in ((0, 64), (1, 256)):
                        record = records[source_index]
                        ctx = context_map[(record["scene"], prefix)]
                        seed = 38102 + 900000 + int(record["crop_seed"]) * 1009 + prefix * 17
                        first = decode_once("spatial_s3", ctx["prompt"], 16, seed,
                                            adapter=spatial_adapter)
                        second = decode_once("spatial_s3", ctx["prompt"], 16, seed,
                                             adapter=spatial_adapter)
                        equal = (first["output_tokens"] == second["output_tokens"]
                                 and all(first.get(key) == second.get(key)
                                         for key in COUNTER_FIELDS))
                        if not equal:
                            raise RuntimeError("Spatial S3 same-seed replay differs at "
                                               + ctx["context_id"])
                        spatial_rows.append(dict(
                            context_id=ctx["context_id"], source=record["scene"], prefix=prefix,
                            seed=seed, first=first, repeated=second, equal=True,
                        ))
                finally:
                    spatial_adapter.close()
                    del spatial_adapter
                    gc.collect()
                    torch_mod.cuda.empty_cache()
                if not any(row["rewind_keep_trace"] for row in zero_rows):
                    raise RuntimeError("The zero-residual replays did not exercise a cache recovery")
                return dict(zero_residual_replays=zero_rows, spatial_replays=spatial_rows)

            replay_result = stage(
                "zero_residual_and_same_seed_replays", replay_zero_and_spatial,
                phase_limit=PHASE_LIMITS["replay_and_warmup"], phase_name="replay_and_warmup")
            verification.update(replay_result)
            write_json(out / "verification.json", jsonable(verification))

            def make_adapters():
                old_adapter = runtime["MLPAdapter"](feature)
                spatial_adapter = runtime["SpatialAdapter"](
                    feature, n2["model"], n2["condition"], n2["prepared"],
                    zero_residual=False, bf16=False, device="cuda",
                )
                return old_adapter, spatial_adapter

            old_adapter, spatial_adapter = stage(
                "prepare_matrix_adapters", make_adapters,
                phase_limit=PHASE_LIMITS["replay_and_warmup"], phase_name="replay_and_warmup")
            adapter_by_method = {"old_pf": old_adapter, "spatial_s3": spatial_adapter}
            target_decode = build_target_decode(runtime["position_decode"])
            runtime["target_decode"] = target_decode

            def run_warmups():
                warmups = []
                warm_times = {}
                selected_prompts = [
                    context_map[(records[0]["scene"], 64)],
                    context_map[(records[1]["scene"], 256)],
                ]
                for prefix_index, context in enumerate(selected_prompts):
                    warm_seed = 38101 + 800000 + int(context["crop_seed"]) * 1009 + context["prefix"] * 17
                    for method in METHODS:
                        row = decode_once(method, context["prompt"], 32, warm_seed,
                                          adapter=adapter_by_method.get(method))
                        warm_times[(method, context["prefix"])] = row["outer_wall_seconds"]
                        warmups.append(dict(
                            method=method, warm_context=context["context_id"],
                            prefix=context["prefix"], seed=warm_seed,
                            generated_tokens=row["generated_tokens"], eos=row["eos"],
                            outer_wall_seconds=row["outer_wall_seconds"],
                            counters={key: row[key] for key in COUNTER_FIELDS},
                        ))
                pipeline["warmup_times_by_method_prefix"] = {
                    method + "_" + str(prefix): seconds
                    for (method, prefix), seconds in warm_times.items()
                }
                return warmups

            verification["warmups"] = stage(
                "warmup_32_tokens", run_warmups,
                phase_limit=PHASE_LIMITS["replay_and_warmup"], phase_name="replay_and_warmup")
            write_json(out / "verification.json", jsonable(verification))

            if len(ranks) != 8000 or any(type(value) not in (int, float) for value in ranks):
                raise ValueError("Target sparse Position rank map is invalid")

            warm_time_map = {
                (entry["method"], entry["prefix"]): entry["outer_wall_seconds"]
                for entry in verification["warmups"]
            }
            group_times = []
            matrix_begin = time.monotonic()
            stage_seconds["matrix"] = 0.0
            update("matrix")
            try:
                for group_index, group in enumerate(groups):
                    check_stop(deadline, stop_signal, "matrix before " + group["pair_id"])
                    check_phase_limit(PHASE_LIMITS["matrix"], "matrix")
                    if group_times:
                        estimate = max(group_times[-3:]) * 1.5
                    else:
                        estimate = max(
                            5.0,
                            math.fsum(warm_time_map[(method, group["prefix"])] * 16
                                      for method in METHODS) * 1.5,
                        )
                    elapsed_now = time.monotonic() - started_mono
                    remain_phase = min(PHASE_LIMITS["matrix"], args.deadline_seconds) - elapsed_now
                    remain_total = deadline - time.monotonic()
                    if remain_phase < estimate or remain_total < estimate + 50.0:
                        raise DeadlineReached("not enough time for the next complete three-method context")
                    group_begin = time.monotonic()
                    prompt = make_prompt(group)
                    for order_index, method in enumerate(group["method_order"]):
                        row = decode_once(method, prompt, BUDGET, group["derived_seed"],
                                          adapter=adapter_by_method.get(method), check_before=False)
                        row = add_context(row, group, method, order_index)
                        append_jsonl(out / "records.jsonl", jsonable(row))
                        rows.append(row)
                        update("matrix")
                        print(json.dumps(dict(
                            pair_id=group["pair_id"], source=group["source"],
                            prefix=group["prefix"], method=method,
                            generated_tokens=row["generated_tokens"],
                            outer_wall_seconds=row["outer_wall_seconds"],
                            completed_runs=len(rows),
                        ), ensure_ascii=False), flush=True)
                    group_times.append(time.monotonic() - group_begin)
                    stage_seconds["matrix"] = time.monotonic() - matrix_begin
                    update("matrix")
                    check_stop(deadline, stop_signal,
                               "matrix after complete context " + group["pair_id"])
                    if stage_seconds["matrix"] >= PHASE_LIMITS["matrix"]:
                        raise DeadlineReached("cumulative matrix limit reached after complete context "
                                              + group["pair_id"])
            finally:
                stage_seconds["matrix"] = time.monotonic() - matrix_begin
                update("matrix")

            for adapter in adapter_by_method.values():
                adapter.close()
            del adapter_by_method, old_adapter, spatial_adapter
            gc.collect()
            torch_mod.cuda.empty_cache()

        check_stop(deadline, stop_signal, "summary")
        status = "complete" if len(rows) == EXPECTED_RUNS else "partial"
        save_summary(status)
        update(status, status)
        if time.monotonic() > deadline:
            status = "budget_overrun"
            save_summary(status)
            update(status, status)
    except DeadlineReached as exc:
        if "cumulative matrix limit reached" in str(exc):
            status = ("matrix_phase_limit" if len(rows) == EXPECTED_RUNS
                      else "partial_matrix_phase_limit")
        else:
            status = "budget_overrun" if len(rows) == EXPECTED_RUNS else "partial_deadline"
        pipeline["stop_reason"] = str(exc)
        save_summary(status)
        update(status, status)
    except InterruptedError as exc:
        status = "partial_signal"
        pipeline["stop_reason"] = str(exc)
        save_summary(status)
        update(status, status)
    except Exception as exc:
        status = "failed" if not rows else "partial_failed"
        pipeline["error"] = dict(type=type(exc).__name__, message=str(exc))
        save_summary(status)
        update(status, status)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    if status != "complete":
        raise SystemExit("Comparison ended with status " + status + "; see " + str(out / "pipeline.json"))


if __name__ == "__main__":
    main()
