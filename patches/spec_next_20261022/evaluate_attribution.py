#!/usr/bin/env python3
"""Compare ordinary Target, graph Target, W1 and single backup in one run."""

from __future__ import annotations

import argparse
import gc
import importlib.util
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sources", type=int, default=30)
    parser.add_argument("--prefixes", default="64,256")
    parser.add_argument("--seed", type=int, default=51101)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--cache-capacity", type=int, default=2048)
    parser.add_argument("--max-wall-seconds", type=float, default=820)
    args = parser.parse_args()
    prefixes = [int(x) for x in args.prefixes.split(",")]
    if (not prefixes or any(x < 0 or x % 2 for x in prefixes)
            or args.sources < 1 or args.budget < 2 or args.budget % 2
            or args.cache_capacity < max(prefixes) + args.budget + 8
            or args.max_wall_seconds < 0):
        parser.error("Invalid size or time argument")
    project, source, candidate, out = [x.resolve() for x in
        (args.project, args.source, args.candidate_source, args.out)]
    if out.exists():
        parser.error("Output directory already exists")
    if not (candidate / "evaluate_candidates.py").is_file():
        parser.error("Candidate evaluator is unavailable")
    sys.path[:0] = [str(candidate), str(source), str(source / "reference"),
                   str(source / "reference/runtime"), str(project)]
    spec = importlib.util.spec_from_file_location(
        "measured_candidate", candidate / "evaluate_candidates.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    import os
    os.chdir(project)
    import torch
    from threadpoolctl import threadpool_limits
    from method_adapters import build_context, load_method, decode_method
    from dynamic_target_graph import DynamicTargetGraph
    from tree_runtime import TargetTreeRuntime
    from tree_decoder import decode as decode_w1
    from prob_position_decoder import decode as decode_probability
    from draft_a.corpus_v2 import EOS, validate
    from draft_a.pf_ranker_eval import cache_for, validate_sequence
    from draft_a.learning_eval import _fp32_runtime

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    out.mkdir(parents=True)
    started = time.perf_counter()
    methods = ["target", "opt_target", "w1", "rescue"]
    pipeline = dict(status="loading", command=sys.argv, completed_runs=0)
    evaluator.write_json(out / "pipeline.json", pipeline)
    args.reuse_weight = .25
    try:
        raw = torch.load(args.crops, map_location="cpu", weights_only=True)
        for record in raw:
            validate(record)
        prompts, sources = evaluator._prompt_rows(
            raw, prefixes, args.sources, 1, args.seed, int(EOS))
        load_started = time.perf_counter()
        context = build_context(project, device="cuda")
        target = context["target"]
        adapters = {name: load_method("target" if name in
                    {"target", "opt_target"} else "pf_mlp", context)
                    for name in methods}
        torch.cuda.synchronize()
        plan = dict(project=str(project), source_snapshot=str(source),
            candidate_source=str(candidate), crops=str(args.crops.resolve()),
            methods=methods, sources=sources, prefixes=prefixes,
            seeds=1, seed_start=args.seed, budgets=[args.budget],
            cache_capacity=args.cache_capacity, prompt_count=len(prompts),
            expected_runs=len(prompts) * len(methods),
            device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
            python_version=sys.version, target_precision="FP32", tf32_enabled=False,
            temperature=.9, top_p=.9, matmul_precision="highest",
            model_load_seconds=time.perf_counter() - load_started,
            method_metadata={name: adapter.metadata for name, adapter in adapters.items()},
            max_wall_seconds=args.max_wall_seconds,
            execution_meaning=dict(target="ordinary original Target",
                opt_target="same original Target decoder with DynamicTargetGraph",
                w1="measured four-node W1 with Target and proposal graphs",
                rescue="measured single-backup five-node decoder"),
            timing_scope="Synchronized outer decode includes runtime creation, KV, prefill, graph creation, sampling, recovery and state maintenance. Weight loading, rendering and file writes excluded.",
            runtime_scope="Independent persistent runtimes per arm; 16-token warmup runtimes discarded; formal first captures timed.")
        evaluator.write_json(out / "plan.json", plan)
        evaluator.write_json(out / "prompts.json", dict(prompts=prompts, eos_id=int(EOS)))

        def make_arm(name):
            arm = evaluator._new_arm("target" if name == "opt_target" else name,
                context, target, args, DynamicTargetGraph=DynamicTargetGraph,
                TargetTreeRuntime=TargetTreeRuntime, adapter=adapters[name],
                load_method=load_method)
            if name == "opt_target":
                arm["runtime"] = DynamicTargetGraph(target,
                    cache_capacity_buckets=(args.cache_capacity,))
            return arm

        def target_decode(arm, name):
            def run(adapter, core, prompt, budget, seed, runtime=None):
                return decode_method(adapter, core, prompt, budget, seed,
                    runtime=arm["runtime"] if name == "opt_target" else runtime)
            return run

        pipeline.update(status="warming", expected_runs=plan["expected_runs"])
        evaluator.write_json(out / "pipeline.json", pipeline)
        with threadpool_limits(limits=1), torch.inference_mode(), \
                _fp32_runtime(torch), torch.autocast("cuda", enabled=False):
            warmups = []
            for name in methods:
                arm = make_arm(name)
                torch.cuda.synchronize()
                begin = time.perf_counter()
                stats, output = evaluator._run_arm(arm, None, context,
                    prompts[0]["prompt"], 16, args.seed,
                    decode_method=target_decode(arm, name), decode_w1=decode_w1,
                    decode_probability=decode_probability, target_cache_for=cache_for)
                torch.cuda.synchronize()
                warmups.append(dict(method=name, outer_wall_seconds=time.perf_counter()-begin,
                    output_tokens=output, counters=evaluator.core_counters(stats),
                    runtime=arm["runtime"].stats() if arm["runtime"] else None))
                evaluator._close_arm(arm)
                del arm
                gc.collect()
            evaluator.write_json(out / "warmup.json", warmups)
            arms, rows, durations = {}, [], []
            pipeline.update(status="running")
            evaluator.write_json(out / "pipeline.json", pipeline)
            stopped = False
            for index, prompt in enumerate(prompts):
                reserve = max(40., max(durations[-6:], default=0.) * 1.5)
                if args.max_wall_seconds and time.perf_counter()-started+reserve >= args.max_wall_seconds:
                    stopped = True
                    break
                group_begin = time.perf_counter()
                rotation = (index + args.budget) % len(methods)
                order = methods[rotation:] + methods[:rotation]
                for name in order:
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    created = name not in arms
                    if created:
                        arms[name] = make_arm(name)
                    arm = arms[name]
                    row = evaluator._run_one(arm, None, prompt, args.budget, context,
                        torch=torch, decode_method=target_decode(arm, name),
                        decode_w1=decode_w1, decode_probability=decode_probability,
                        target_cache_for=cache_for, validate_sequence=validate_sequence,
                        started_at=begin)
                    row.update(method=name, action_id=name, runtime_created=created)
                    evaluator.append_jsonl(out / "records.jsonl", row)
                    rows.append(row)
                    pipeline.update(completed_runs=len(rows),
                        elapsed_seconds=time.perf_counter()-started,
                        last_completed=dict(pair_id=prompt["pair_id"], method=name))
                    evaluator.write_json(out / "pipeline.json", pipeline)
                    print(evaluator.json.dumps({k:row[k] for k in
                        ["pair_id", "method", "generated_tokens", "outer_wall_seconds"]}
                        | dict(completed_runs=len(rows)), allow_nan=False), flush=True)
                durations.append(time.perf_counter()-group_begin)
            summary = evaluator._summarize(rows, methods, prompts, args.budget)
            summary.update(elapsed_seconds=time.perf_counter()-started,
                completed_prompt_groups=len(durations), time_limit_reached=stopped,
                expected_runs=plan["expected_runs"],
                runtime_summary={name:arm["runtime"].stats() if arm["runtime"] else None
                    for name,arm in arms.items()},
                tree_summary={name:arm["tree"].stats() if arm["tree"] else None
                    for name,arm in arms.items()},
                rescue_summary={name:arm["rescue"].stats() if arm.get("rescue") else None
                    for name,arm in arms.items()},
                proposal_summary={name:arm["proposal"].stats() if arm["proposal"] else None
                    for name,arm in arms.items()})
            evaluator.write_json(out / "summary.json", summary)
            for arm in arms.values():
                evaluator._close_arm(arm)
            arms.clear()
            gc.collect()
        pipeline.update(status="partial" if stopped else "complete",
            completed_runs=len(rows), elapsed_seconds=time.perf_counter()-started,
            time_limit_reached=stopped)
        evaluator.write_json(out / "pipeline.json", pipeline)
    except Exception as error:
        pipeline.update(status="failed", error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.perf_counter()-started)
        evaluator.write_json(out / "pipeline.json", pipeline)
        raise


if __name__ == "__main__":
    main()
