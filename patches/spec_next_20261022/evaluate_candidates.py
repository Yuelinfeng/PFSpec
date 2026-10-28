#!/usr/bin/env python3
"""Compare original Target, W1, and isolated speculative candidates."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


def jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if hasattr(value, "detach"):
        return jsonable(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        return jsonable(value.item())
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row), ensure_ascii=False,
                                allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def runtime_delta(before, after):
    old, new = before or {}, after or {}

    def differences(name):
        left, right = old.get(name, {}), new.get(name, {})
        return {key: int(right.get(key, 0)) - int(left.get(key, 0))
                for key in set(left) | set(right)
                if int(right.get(key, 0)) != int(left.get(key, 0))}

    return dict(
        calls=differences("calls"),
        fallbacks=differences("fallbacks"),
        graph_count_before=int(old.get("graph_count", 0)),
        graph_count_after=int(new.get("graph_count", 0)),
        new_graph_events=list(new.get("graph_events", []))[len(old.get("graph_events", [])):],
        new_cache_events=list(new.get("cache_events", []))[len(old.get("cache_events", [])):],
        copy_bytes=differences("copy_bytes"),
    )


CORE_COUNTER_KEYS = (
    "target_calls", "target_input_tokens", "draft_calls", "draft_input_tokens",
    "ranker_calls", "rounds", "proposed", "verified", "accepted",
    "accepted_lengths", "round_generated_lengths", "position_fallbacks",
    "k1_steps", "tree_verify_calls", "candidate_tree_nodes", "target_tree_nodes",
    "feature_attempts", "feature_branch_hits", "feature_all_rejected",
    "feature_branch_index_counts", "recovery_calls",
    "generated_tokens", "eos", "position_proposed", "position_verified",
    "position_accepted", "feature_proposed", "feature_verified", "feature_accepted",
    "eos_proposed", "eos_verified", "eos_accepted",
)


def core_counters(stats):
    return {key: jsonable(stats[key]) for key in CORE_COUNTER_KEYS if key in stats}


def compare_core(left, right):
    missing = sorted((set(CORE_COUNTER_KEYS) - set(left))
                     | (set(CORE_COUNTER_KEYS) - set(right)))
    keys = sorted(set(left) & set(right))
    return dict(
        equal=not missing and all(left[key] == right[key] for key in keys),
        missing=missing,
        differing=[key for key in keys if left[key] != right[key]],
    )


def _source_id(record):
    source = record.get("scene") or record.get("source") or record.get("source_id")
    if source:
        return str(source)
    path = Path(str(record.get("path", "")))
    return path.parents[1].name if len(path.parts) >= 3 else "unknown"


def _crop_path(record):
    value = record.get("path") or record.get("crop_path")
    return None if value is None else str(value)


def select_crops(records, max_prefix, source_limit, scene_names=None):
    by_source = {}
    for record in records:
        by_source.setdefault(_source_id(record), []).append(record)
    requested = None if not scene_names else {str(value) for value in scene_names}
    names = [name for name in sorted(by_source)
             if requested is None or name in requested]
    selected = []
    for name in names:
        record = next((item for item in by_source[name]
                       if len(item["tokens"]) >= int(max_prefix) + 4), None)
        if record is not None:
            selected.append((name, record))
        if len(selected) >= int(source_limit):
            break
    return selected


def make_prompt(source, record, prefix, seed, eos_id, pair_id):
    prefix = int(prefix)
    tokens = [] if prefix == 0 else [int(item) for item in record["tokens"][:prefix]]
    return dict(
        pair_id=str(pair_id),
        context_id=f"{source}:p{prefix:04d}",
        source=str(source),
        crop_path=_crop_path(record),
        crop_seed=record.get("seed", record.get("crop_seed")),
        prefix=int(prefix),
        seed=int(seed),
        prompt=[int(eos_id), *tokens],
    )


def _event_setup_ms(events):
    total = 0.0
    for event in events or []:
        setup = float(event.get("setup_ms", 0.0))
        if setup <= 0.0:
            setup = sum(float(event.get(key, 0.0)) for key in
                        ("prepare_ms", "warmup_ms", "capture_ms"))
        total += setup
    return total


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="Existing feature_tree source_run04 snapshot")
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="New output directory; existing paths are refused")
    parser.add_argument("--methods", default="target,w1,rescue,reuse",
                        help="Comma-separated arms: target,w1,prob,rescue,rescue1,rescue2,rescue3,rescue4,reuse")
    parser.add_argument("--sources", type=int, default=10)
    parser.add_argument("--scene-names", default="",
                        help="Optional comma-separated exact source ids")
    parser.add_argument("--prefixes", default="64,256")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=47101)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--max-wall-seconds", type=float, default=0.0,
                        help="Stop before another complete prompt group when time is nearly used")
    parser.add_argument("--cache-capacity", type=int, default=1024)
    parser.add_argument("--reuse-weight", type=float, default=0.25)
    parser.add_argument("--verify-only", action="store_true",
                        help="Compare W1 with point mode and repeat probabilistic mode")
    parser.add_argument("--source-scope", choices=("development", "independent"),
                        default="development")
    return parser


def _new_arm(name, context, target, args, *, DynamicTargetGraph,
             TargetTreeRuntime, adapter=None, load_method=None):
    if name == "target":
        if adapter is None:
            adapter = load_method("target", context)
        return dict(name=name, adapter=adapter,
                    runtime=None, proposal=None, tree=None)
    if adapter is None:
        adapter = load_method("pf_mlp", context)
    runtime = DynamicTargetGraph(target, cache_capacity_buckets=(args.cache_capacity,))
    from proposal_adapters import build_proposal
    proposal = build_proposal(adapter, verify=False)
    tree = TargetTreeRuntime(target, runtime, graph=True, max_width=4)
    result = dict(name=name, adapter=adapter, runtime=runtime,
                  proposal=proposal, tree=tree, reuse_weight=args.reuse_weight)
    if name in {"rescue", "rescue1", "rescue2", "rescue3", "rescue4"}:
        from position_rescue_runtime import PositionRescueRuntime
        count = int(name[-1]) if name[-1].isdigit() else 1
        result["backup_count"] = count
        result["rescue"] = PositionRescueRuntime(target, runtime, graph=True,
                                                backup_count=count)
    return result


def _close_arm(arm):
    proposal = arm.get("proposal")
    if proposal is not None:
        proposal.close()


def _reset_adapter_counters(adapter):
    for key in ("forward_calls", "input_tokens"):
        if hasattr(adapter, key):
            setattr(adapter, key, 0)


def _run_arm(arm, mode, context, prompt, budget, seed, *,
             decode_method, decode_w1, decode_probability, target_cache_for):
    import torch

    name = arm["name"]
    adapter = arm["adapter"]
    runtime = arm["runtime"]
    proposal = arm["proposal"]
    _reset_adapter_counters(getattr(adapter, "model", None))
    target = context["target"]
    if name == "target":
        stats, suffix = decode_method(adapter, target, prompt, budget, seed, runtime=None)
    else:
        cache_factory = lambda core, capacity: runtime.cache_for(
            core, capacity, factory=target_cache_for
        )
        with proposal.context(), runtime.attention_context():
            if name == "w1":
                stats, suffix = decode_w1(
                    target, adapter.model, context["ranker"], context["ranks"],
                    prompt, budget, seed, tree_runtime=arm["tree"], width=1,
                    cache_factory=cache_factory,
                )
            elif name in {"rescue", "rescue1", "rescue2", "rescue3", "rescue4"}:
                from position_rescue_decoder import decode as decode_rescue
                stats, suffix = decode_rescue(
                    target, adapter.model, context["ranker"], context["ranks"],
                    prompt, budget, seed, tree_runtime=arm["tree"],
                    rescue_runtime=arm["rescue"], width=1, cache_factory=cache_factory,
                    backup_count=arm["backup_count"],
                )
            elif name == "reuse":
                from feature_reuse_decoder import decode as decode_reuse
                stats, suffix = decode_reuse(
                    target, adapter.model, context["ranker"], context["ranks"],
                    prompt, budget, seed, tree_runtime=arm["tree"], width=1,
                    cache_factory=cache_factory, reuse_weight=arm["reuse_weight"],
                )
            else:
                stats, suffix = decode_probability(
                    target, adapter.model, context["ranker"], context["ranks"],
                    prompt, budget, seed, tree_runtime=arm["tree"], width=1,
                    cache_factory=cache_factory, position_mode=mode,
                )
    return stats, [int(token) for token in suffix]


def _run_one(arm, mode, prompt_row, budget, context, *, torch,
             decode_method, decode_w1, decode_probability, target_cache_for,
             validate_sequence, started_at=None):
    runtime = arm.get("runtime")
    proposal = arm.get("proposal")
    tree = arm.get("tree")
    runtime_before = runtime.stats() if runtime else None
    proposal_before = proposal.stats() if proposal else None
    tree_before = tree.stats() if tree else None
    rescue = arm.get("rescue")
    rescue_before = rescue.stats() if rescue else None
    if started_at is None:
        torch.cuda.synchronize()
        started_at = time.perf_counter()
    stats, output = _run_arm(
        arm, mode, context, prompt_row["prompt"], budget, prompt_row["seed"],
        decode_method=decode_method, decode_w1=decode_w1,
        decode_probability=decode_probability, target_cache_for=target_cache_for,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at
    validate_sequence(prompt_row["prompt"], output, context["ranks"])
    if len(output) > int(budget):
        raise RuntimeError(f"{arm['name']} returned more tokens than the budget")
    suffix_eos = bool(output and output[-1] == int(context["eos"]))
    stats_json = jsonable(stats)
    after_runtime = runtime.stats() if runtime else None
    after_proposal = proposal.stats() if proposal else None
    after_tree = tree.stats() if tree else None
    row = dict(
        pair_id=prompt_row["pair_id"], context_id=prompt_row["context_id"],
        source=prompt_row["source"], crop_path=prompt_row["crop_path"],
        crop_seed=prompt_row["crop_seed"], prefix=prompt_row["prefix"],
        prompt_tokens=len(prompt_row["prompt"]), seed=prompt_row["seed"],
        budget=int(budget), method=arm["name"], position_mode=mode,
        generated_tokens=len(output), output_tokens=output,
        backup_count=arm.get("backup_count", 0),
        eos=suffix_eos, outer_wall_seconds=elapsed,
        tokens_per_second=(len(output) / elapsed if elapsed > 0 else None),
        counters=core_counters(stats), stats=stats_json,
        sequence_validated=True,
        runtime_created=runtime_before is None and runtime is not None,
        runtime_delta=(runtime_delta(runtime_before, after_runtime)
                       if runtime is not None else None),
        proposal_runtime_delta=(runtime_delta(proposal_before, after_proposal)
                                if proposal is not None else None),
        tree_runtime_delta=(runtime_delta(tree_before, after_tree)
                            if tree is not None else None),
        rescue_runtime_delta=(runtime_delta(rescue_before, rescue.stats()) if rescue else None),
        graph_setup_ms=(
            _event_setup_ms((runtime_delta(rescue_before, rescue.stats()) if rescue else {}).get("new_graph_events", [])) +
            _event_setup_ms((runtime_delta(runtime_before, after_runtime) or {}).get("new_graph_events", []))
            + _event_setup_ms((runtime_delta(proposal_before, after_proposal) or {}).get("new_graph_events", []))
            + _event_setup_ms((runtime_delta(tree_before, after_tree) or {}).get("new_graph_events", []))
        ),
    )
    return row


def _prompt_rows(records, prefixes, sources, seeds, seed_start, eos_id, scene_names=None):
    maximum = max((int(prefix) for prefix in prefixes), default=0)
    selected = select_crops(records, maximum, sources, scene_names)
    needed = min(int(sources), len(set(scene_names or []))) if scene_names else int(sources)
    if len(selected) < needed:
        raise ValueError(f"Only {len(selected)} sources have a crop long enough for prefix {maximum}")
    prompts = []
    for source_index, (source, record) in enumerate(selected):
        for prefix in prefixes:
            for seed_index in range(int(seeds)):
                seed = int(seed_start) + seed_index
                pair_id = f"{source_index:02d}-{int(prefix):04d}-{seed_index:02d}"
                prompts.append(make_prompt(source, record, prefix, seed, eos_id, pair_id))
    return prompts, [name for name, _record in selected]


def _verify_prompts(records, seed_start, eos_id):
    selected = select_crops(records, 256, 1)
    if not selected:
        raise ValueError("No crop has 256 tokens for verify-only mode")
    source, record = selected[0]
    prompts = []
    for prefix in (0, 64, 256):
        for seed_offset in range(2):
            seed = int(seed_start) + seed_offset
            pair_id = f"verify-{prefix:04d}-s{seed}"
            prompts.append(make_prompt(source, record, prefix, seed, eos_id, pair_id))
    return prompts


def _summarize(rows, methods, prompts, budget):
    by_pair = {}
    by_method = {name: [] for name in methods}
    for row in rows:
        by_pair.setdefault(row["pair_id"], {})[row["method"]] = row
        by_method[row["method"]].append(row)
    result = dict(status="complete", completed_runs=len(rows), budget=int(budget),
                  method_summary={})
    for method, method_rows in by_method.items():
        pair_ratios = []
        per_source = {}
        target_rows = {row["pair_id"]: row for row in by_method.get("target", [])}
        for row in method_rows:
            reference = target_rows.get(row["pair_id"])
            if reference is None or method == "target":
                continue
            ratio = (float(row["tokens_per_second"]) /
                     float(reference["tokens_per_second"]))
            pair_ratios.append(ratio)
            per_source.setdefault(row["source"], []).append(ratio)
        source_geomeans = {
            source: math.exp(statistics.fmean(math.log(value) for value in values))
            for source, values in per_source.items() if values and all(value > 0 for value in values)
        }
        result["method_summary"][method] = dict(
            run_count=len(method_rows),
            mean_generated_tokens=statistics.fmean(row["generated_tokens"] for row in method_rows),
            median_outer_wall_seconds=statistics.median(row["outer_wall_seconds"] for row in method_rows),
            mean_tokens_per_second=statistics.fmean(row["tokens_per_second"] for row in method_rows),
            paired_ratio_count=len(pair_ratios),
            source_equal_geomean_ratio=(
                math.exp(statistics.fmean(math.log(value) for value in source_geomeans.values()))
                if source_geomeans and all(value > 0 for value in source_geomeans.values()) else None
            ),
            source_ratios=source_geomeans,
        )
    result["prompt_count"] = len(prompts)
    return result


def _verify_report(rows):
    groups = {}
    for row in rows:
        key = (row["pair_id"], int(row["budget"]))
        groups.setdefault(key, {})[row["verification_role"]] = row
    comparisons = []
    for (pair_id, budget), arms in groups.items():
        w1 = arms.get("w1")
        point = arms.get("point")
        prob_a = arms.get("prob_repeat_1")
        prob_b = arms.get("prob_repeat_2")
        if w1 is None or point is None or prob_a is None or prob_b is None:
            raise RuntimeError(
                f"verify-only output is incomplete for {pair_id}, budget {budget}: {sorted(arms)}"
            )
        w1_point_counts = compare_core(w1["counters"], point["counters"])
        prob_repeat_counts = compare_core(prob_a["counters"], prob_b["counters"])
        comparisons.append(dict(
            pair_id=pair_id,
            budget=budget,
            w1_point_output_equal=w1["output_tokens"] == point["output_tokens"],
            w1_point_core_counts=w1_point_counts,
            probability_repeat_output_equal=prob_a["output_tokens"] == prob_b["output_tokens"],
            probability_repeat_core_counts=prob_repeat_counts,
            all_sequences_validated=all(row["sequence_validated"] for row in arms.values()),
        ))
    passed = all(
        item["w1_point_output_equal"]
        and item["w1_point_core_counts"]["equal"]
        and item["probability_repeat_output_equal"]
        and item["probability_repeat_core_counts"]["equal"]
        and item["all_sequences_validated"]
        for item in comparisons
    )
    complete = len(comparisons) == 24
    passed = passed and complete
    return dict(status="passed" if passed else "failed",
                prompt_count=len(comparisons) // 4, comparison_count=len(comparisons),
                expected_prefixes=[0, 64, 256], expected_budgets=[2, 4, 16, 64],
                expected_comparison_count=24, seeds_per_prefix=2,
                complete_grid=complete, comparisons=comparisons)


def main():
    parser = _make_parser()
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    allowed = {"target", "w1", "prob", "rescue", "rescue1", "rescue2", "rescue3", "rescue4", "reuse"}
    prefixes = [int(item) for item in args.prefixes.split(",") if item.strip()]
    if (not methods or len(methods) != len(set(methods))
            or set(methods) - allowed
            or args.max_wall_seconds < 0
            or args.sources < 1 or args.seeds < 1 or args.budget < 2
            or args.budget % 2 or not prefixes
            or any(prefix < 0 or prefix % 2 for prefix in prefixes)
            or args.cache_capacity < max(prefixes or [0]) + args.budget + 8):
        parser.error("Invalid methods, prefix, budget, or cache capacity")
    project, source, crops, out = (
        args.project.resolve(), args.source.resolve(), args.crops.resolve(), args.out.resolve()
    )
    if out.exists():
        parser.error("Output must be a new directory")
    if not source.is_dir():
        parser.error(f"Source snapshot does not exist: {source}")
    if not crops.is_file():
        parser.error(f"Crop file does not exist: {crops}")

    out.mkdir(parents=True)
    started = time.time()
    pipeline = dict(status="loading", started_at_unix=started, completed_runs=0,
                    command=sys.argv, methods=methods, verify_only=bool(args.verify_only))
    write_json(out / "pipeline.json", pipeline)
    try:
        script_dir = Path(__file__).resolve().parent
        sys.path[:0] = [str(script_dir), str(source), str(source / "reference"),
                        str(source / "reference" / "runtime"), str(project)]
        os.chdir(project)
        import torch
        from threadpoolctl import threadpool_limits
        from method_adapters import build_context, load_method, decode_method
        from dynamic_target_graph import DynamicTargetGraph
        from tree_runtime import TargetTreeRuntime
        from tree_decoder import decode as decode_w1
        from prob_position_decoder import decode as decode_probability
        from draft_a.corpus_v2 import EOS, validate
        from draft_a.pf_ranker_eval import cache_for as target_cache_for, validate_sequence
        from draft_a.learning_eval import _fp32_runtime

        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")

        raw = torch.load(crops, map_location="cpu", weights_only=True)
        for record in raw:
            validate(record)
        scene_names = [item.strip() for item in args.scene_names.split(",") if item.strip()]
        if args.verify_only:
            prompts = _verify_prompts(raw, args.seed_start, int(EOS))
            run_methods = ["w1", "prob"]
            budgets = [2, 4, 16, 64]
            expected_runs = len(prompts) * len(budgets) * 4
            selected_sources = sorted({row["source"] for row in prompts})
        else:
            prompts, selected_sources = _prompt_rows(
                raw, prefixes, args.sources, args.seeds, args.seed_start, int(EOS),
                scene_names=scene_names or None,
            )
            if not prompts:
                raise ValueError("No source/prefix prompts were selected")
            run_methods = methods
            budgets = [int(args.budget)]
            expected_runs = len(prompts) * len(run_methods)

        load_started = time.perf_counter()
        context = build_context(project, device="cuda")
        target = context["target"]
        if args.verify_only:
            adapters = {
                "w1": load_method("pf_mlp", context),
                "prob": load_method("pf_mlp", context),
            }
        else:
            adapters = {
                name: load_method("target" if name == "target" else "pf_mlp", context)
                for name in run_methods
            }
        model_load_seconds = time.perf_counter() - load_started
        plan = dict(
            project=str(project), source_snapshot=str(source), crops=str(crops),
            source_scope=args.source_scope, methods=run_methods,
            execution_meaning={
                "target": "Original Target with ordinary execution",
                "w1": "Existing W1 decoder with Target, proposal, and tree graphs",
                "prob": "Normalized-score Position proposals",
                "rescue": "W1 with a fifth backup Position node; residual draws unchanged",
                "reuse": "W1 with a causal mixture of MLP q and previous Target Feature policy",
            },
            sources=selected_sources,
            prefixes=([0, 64, 256] if args.verify_only else prefixes),
            seeds=(2 if args.verify_only else args.seeds),
            seed_start=args.seed_start,
            budgets=(budgets if args.verify_only else [int(args.budget)]),
            prompt_count=len(prompts), expected_runs=expected_runs,
            cache_capacity=args.cache_capacity,
            device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
            python_version=sys.version, target_precision="FP32",
            matmul_precision="highest", tf32_enabled=False,
            position_rank_map=jsonable(context["ranks"]), eos_id=int(EOS),
            temperature=0.9, top_p=0.9, reuse_weight=args.reuse_weight, model_load_seconds=model_load_seconds,
            method_metadata={name: jsonable(adapter.metadata) for name, adapter in adapters.items()},
            verify_only=bool(args.verify_only),
            timing_scope=("Synchronized outer decode wall includes per-arm runtime and proposal graph construction, cache allocation/reset, prefill, tree verification, recovery, sampling, and graph capture. Model weights load before timing."),
            runtime_scope=("Target arm uses ordinary execution. Every speculative arm owns its persistent Target runtime, MLP proposal graph, and W1 tree runtime. The rescue arm additionally owns its five-node runtime. Warmup objects are discarded."),
        )
        for name in run_methods:
            if name.startswith("rescue") and name[-1].isdigit():
                count = int(name[-1])
                plan["execution_meaning"][name] = f"W1 with up to {count} sibling backup Positions, at most {4 + count} nodes; residual draws unchanged"
        plan["max_wall_seconds"] = args.max_wall_seconds
        plan["runtime_scope"] = "Each arm owns its persistent runtime; backup node counts use separate graphs. Formal first captures are timed and warmup objects are discarded."
        write_json(out / "plan.json", plan)
        write_json(out / "prompts.json", dict(prompts=prompts, eos_id=int(EOS)))
        pipeline.update(status="warming", expected_runs=expected_runs,
                        model_load_seconds=model_load_seconds)
        write_json(out / "pipeline.json", pipeline)

        with threadpool_limits(limits=1), torch.inference_mode(), _fp32_runtime(torch), torch.autocast("cuda", enabled=False):
            if not args.verify_only:
                warmups = []
                warm_prompt = prompts[0]
                for name in run_methods:
                    arm = _new_arm(
                        name, context, target, args,
                        DynamicTargetGraph=DynamicTargetGraph,
                        TargetTreeRuntime=TargetTreeRuntime,
                        adapter=adapters[name],
                        load_method=load_method,
                    )
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    mode = "prob" if name == "prob" else None
                    stats, output = _run_arm(
                        arm, mode, context, warm_prompt["prompt"], 16,
                        warm_prompt["seed"], decode_method=decode_method,
                        decode_w1=decode_w1, decode_probability=decode_probability,
                        target_cache_for=target_cache_for,
                    )
                    torch.cuda.synchronize()
                    warmups.append(dict(
                        method=name, position_mode=mode,
                        outer_wall_seconds=time.perf_counter() - begin,
                        generated_tokens=len(output), output_tokens=output,
                        counters=core_counters(stats),
                        runtime=arm["runtime"].stats() if arm["runtime"] else None,
                        proposal_runtime=arm["proposal"].stats() if arm["proposal"] else None,
                        tree_runtime=arm["tree"].stats() if arm["tree"] else None,
                    ))
                    write_json(out / "warmup.json", warmups)
                    _close_arm(arm)
                    del arm
                    gc.collect()
                write_json(out / "warmup.json", warmups)

            pipeline.update(status="running")
            write_json(out / "pipeline.json", pipeline)
            arms = {}
            rows = []
            comparison_inputs = []
            if args.verify_only:
                arm_specs = [("w1", None, "w1"), ("prob", "point", "point"),
                             ("prob", "prob", "prob_repeat_1"),
                             ("prob", "prob", "prob_repeat_2")]
            else:
                arm_specs = [(name, "prob" if name == "prob" else None, name)
                             for name in run_methods]

            group_durations = []
            time_limit_reached = False
            for prompt_index, prompt_row in enumerate(prompts):
                if args.max_wall_seconds and not args.verify_only:
                    reserve = max(35.0, max(group_durations[-6:], default=0.0) * 1.5)
                    if time.time() - started + reserve >= args.max_wall_seconds:
                        time_limit_reached = True
                        break
                group_begin = time.perf_counter()
                if args.verify_only:
                    prompt_budgets = budgets
                else:
                    prompt_budgets = budgets
                for budget in prompt_budgets:
                    order = list(arm_specs)
                    rotation = (prompt_index + int(budget)) % len(order)
                    order = order[rotation:] + order[:rotation]
                    for method, mode, verification_role in order:
                        key = method if not (args.verify_only and method == "prob") else "prob"
                        torch.cuda.synchronize()
                        started_at = time.perf_counter()
                        arm = arms.get(key)
                        created = False
                        if arm is None:
                            arm = _new_arm(
                                method, context, target, args,
                                DynamicTargetGraph=DynamicTargetGraph,
                                TargetTreeRuntime=TargetTreeRuntime,
                                adapter=adapters[key],
                                load_method=load_method,
                            )
                            arms[key] = arm
                            created = True
                        row = _run_one(
                            arm, mode, prompt_row, budget, context, torch=torch,
                            decode_method=decode_method, decode_w1=decode_w1,
                            decode_probability=decode_probability,
                            target_cache_for=target_cache_for,
                            validate_sequence=validate_sequence,
                            started_at=started_at,
                        )
                        row["runtime_created"] = bool(created)
                        row["action_id"] = verification_role
                        if args.verify_only:
                            row["verification_role"] = verification_role
                        append_jsonl(out / "records.jsonl", row)
                        rows.append(row)
                        comparison_inputs.append(row)
                        pipeline.update(completed_runs=len(rows), elapsed_seconds=time.time() - started,
                                        last_completed=dict(pair_id=row["pair_id"], method=method,
                                                            position_mode=row["position_mode"], budget=budget))
                        write_json(out / "pipeline.json", pipeline)
                        print(json.dumps(dict(
                            completed_runs=len(rows), expected_runs=expected_runs,
                            pair_id=row["pair_id"], method=method,
                            position_mode=row["position_mode"], budget=budget,
                            generated_tokens=row["generated_tokens"],
                            outer_wall_seconds=row["outer_wall_seconds"],
                        ), ensure_ascii=False), flush=True)
                group_durations.append(time.perf_counter() - group_begin)

            if args.verify_only:
                verification = _verify_report(rows)
                write_json(out / "verification.json", verification)
                if verification["status"] != "passed":
                    raise ArithmeticError("verify-only output or core counters differ")
                summary = dict(status="complete", completed_runs=len(rows),
                               verification=verification)
            else:
                summary = _summarize(rows, run_methods, prompts, args.budget)
            summary.update(
                elapsed_seconds=time.time() - started,
                runtime_summary={name: arm["runtime"].stats() if arm["runtime"] else None
                                 for name, arm in arms.items()},
                proposal_summary={name: arm["proposal"].stats() if arm["proposal"] else None
                                  for name, arm in arms.items()},
                rescue_summary={name: arm["rescue"].stats() if arm.get("rescue") else None
                                for name, arm in arms.items()},
                tree_summary={name: arm["tree"].stats() if arm["tree"] else None
                              for name, arm in arms.items()},
                timing_scope=plan["timing_scope"], runtime_scope=plan["runtime_scope"],
                time_limit_reached=time_limit_reached,
                expected_runs=expected_runs,
                completed_prompt_groups=len(group_durations),
            )
            write_json(out / "summary.json", summary)
            for arm in arms.values():
                _close_arm(arm)
            arms.clear()
            gc.collect()
        pipeline.update(status="complete", completed_runs=len(rows),
                        time_limit_reached=time_limit_reached,
                        elapsed_seconds=time.time() - started)
        write_json(out / "pipeline.json", pipeline)
        print(json.dumps(dict(status="complete", output=str(out),
                              completed_runs=len(rows)), ensure_ascii=False), flush=True)
    except Exception as error:
        pipeline.update(status="failed", error=f"{type(error).__name__}: {error}",
                        elapsed_seconds=time.time() - started)
        write_json(out / "pipeline.json", pipeline)
        raise


if __name__ == "__main__":
    main()
