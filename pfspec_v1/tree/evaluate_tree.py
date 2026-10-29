#!/usr/bin/env python3
"""Measure Feature tree decoding and the original Target in one rotating run."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

from contextlib import nullcontext

PREVIOUS_SOURCE = Path(__file__).resolve().parent / "reference"
sys.path.insert(0, str(PREVIOUS_SOURCE))
from proposal_adapters import build_proposal


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


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def append_row(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def count_fields(stats):
    """Keep method counters and their nested per-depth counts for replay comparison."""
    def is_count(key, value):
        if any(word in key.lower() for word in ("time", "second", "rate", "throughput", "latency")):
            return False
        return isinstance(value, (int, bool, list, dict))
    return {key: jsonable(value) for key, value in stats.items() if is_count(key, value)}


def runtime_delta(before, after):
    old = before or {}
    return dict(
        calls={key: value - old.get("calls", {}).get(key, 0)
               for key, value in after.get("calls", {}).items()},
        fallbacks={key: value - old.get("fallbacks", {}).get(key, 0)
                   for key, value in after.get("fallbacks", {}).items()
                   if value != old.get("fallbacks", {}).get(key, 0)},
        graph_count_before=old.get("graph_count", 0), graph_count_after=after.get("graph_count", 0),
        new_graph_events=after.get("graph_events", [])[len(old.get("graph_events", [])):],
        new_cache_events=after.get("cache_events", [])[len(old.get("cache_events", [])):],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", required=True, help="Comma-separated adapter names, including target")
    parser.add_argument("--executions", default="proposal_graph")
    parser.add_argument("--sources", type=int, default=3)
    parser.add_argument("--prefixes", default="64,256")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=45101)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--cache-capacity", type=int, default=1024)
    parser.add_argument("--verify-proposals", action="store_true")
    parser.add_argument("--source-scope", choices=("development", "independent"), default="development")
    args = parser.parse_args()
    methods = args.methods.split(",")
    executions = args.executions.split(",")
    prefixes = [int(item) for item in args.prefixes.split(",")]
    if (len(methods) != len(set(methods)) or "target" not in methods
            or not executions or len(executions) != len(set(executions))
            or any(item not in {"target_only", "proposal_graph"} for item in executions)
            or args.sources < 1 or args.seeds < 1 or args.budget < 2
            or args.budget % 2 or any(prefix < 0 or prefix % 2 for prefix in prefixes)
            or (0 in prefixes and (prefixes != [0] or args.sources != 1))):
        parser.error("Invalid method, execution, prefix or budget setting")
    project, out = args.project.resolve(), args.out.resolve()
    if out.exists():
        parser.error("Output must be a new directory")
    os.chdir(project)
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(PREVIOUS_SOURCE / "runtime"))
    import torch
    from threadpoolctl import threadpool_limits
    from method_adapters import build_context, load_method, decode_method
    from dynamic_target_graph import DynamicTargetGraph
    from draft_a.corpus_v2 import EOS, validate
    from draft_a.pf_ranker_eval import validate_sequence
    from draft_a.learning_eval import _fp32_runtime

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    out.mkdir(parents=True)
    started = time.time()
    pipeline = dict(status="loading", started_at_unix=started, completed_runs=0,
                    command=sys.argv, methods=methods, executions=executions)
    write_json(out / "pipeline.json", pipeline)
    try:
        raw = torch.load(args.crops, map_location="cpu", weights_only=True)
        selected = {}
        for record in raw:
            validate(record)
            source = str(record.get("scene") or record.get("source") or Path(record["path"]).parents[1].name)
            if len(record["tokens"]) >= max(prefixes) + 4 and source not in selected:
                selected[source] = record
        if len(selected) < args.sources:
            raise ValueError("Too few source scenes in crop file")
        prompts = []
        for scene_index, source in enumerate(sorted(selected)[:args.sources]):
            for prefix in prefixes:
                for seed_index in range(args.seeds):
                    prompts.append(dict(pair_id=f"{scene_index:02d}-{prefix:04d}-{seed_index:02d}",
                                        source=source, prefix=prefix, seed=args.seed_start + seed_index,
                                        prompt=[EOS] + [int(token) for token in selected[source]["tokens"][:prefix]]))
        write_json(out / "prompts.json", dict(contexts=prompts, eos_id=EOS))
        load_started = time.perf_counter()
        context = build_context(project, device="cuda")
        target = context["target"]
        adapters = {method: load_method("pf_mlp" if method.startswith("tree_w") else method, context) for method in methods}
        from tree_runtime import TargetTreeRuntime
        from tree_decoder import decode as decode_tree
        def execution_for(method):
            return ["original_target"] if method == "target" else ["tree_graph" if method.startswith("tree_w") else "proposal_graph"]
        def tree_for(method, runtime):
            return TargetTreeRuntime(target, runtime, graph=True, max_width=4) if method.startswith("tree_w") else None
        def run_decode(method, runtime, tree, prompt, budget, seed):
            if tree is None:
                return decode_method(adapters[method], target, prompt, budget, seed, runtime=runtime)
            from draft_a.pf_ranker_eval import cache_for
            factory = lambda core, length: runtime.cache_for(core, length, factory=cache_for)
            with runtime.attention_context():
                return decode_tree(target, adapters[method].model, context["ranker"], context["ranks"],
                                   prompt, budget, seed, tree_runtime=tree, width=int(method[6:]), cache_factory=factory)
        torch.cuda.synchronize()
        plan = dict(project=str(project), crops=str(args.crops.resolve()), source_scope=args.source_scope,
                    methods=methods, executions=executions, sources=sorted(selected)[:args.sources],
                    prefixes=prefixes, seeds=args.seeds, seed_start=args.seed_start, budget=args.budget,
                    input_kind="BOS only; scene tokens unused" if prefixes == [0] else "real scene prefix",
                    prompt_count=len(prompts), expected_runs=len(prompts)*len(methods),
                    cache_capacity=args.cache_capacity, device=torch.cuda.get_device_name(0),
                    torch_version=torch.__version__, python_version=sys.version,
                    target_precision="FP32", matmul_precision="highest", tf32_enabled=False,
                    position_rank_map=jsonable(context["ranks"]), eos_id=int(EOS),
                    temperature=.9, top_p=.9, model_load_seconds=time.perf_counter()-load_started,
                    method_metadata={name: getattr(adapter, "metadata", {}) for name, adapter in adapters.items()},
                    proposal_verified=args.verify_proposals,
                    execution_meaning={"original_target":"Original Target with ordinary execution", "target_only":"Optimized Target with ordinary proposal execution", "proposal_graph":"Existing MLP with Target and proposal graphs", "tree_graph":"Feature tree with Target, proposal and tree graphs"},
                    timing_scope="Synchronized decode outer wall includes runtime creation, KV allocation, graph creation, prefill, proposals, verification, recovery, sampling and temporary hooks. Weight loading is separate.",
                    runtime_scope="One persistent Target and proposal runtime per arm. Warmup runtimes are discarded. MLP and tree arms each own proposal graphs; tree arms also own tree graphs.")
        write_json(out / "plan.json", plan)
        pipeline.update(status="warming", expected_runs=plan["expected_runs"])
        write_json(out / "pipeline.json", pipeline)
        warmups = []
        with threadpool_limits(limits=1), torch.inference_mode(), _fp32_runtime(torch), torch.autocast("cuda", enabled=False):
            for method in methods:
                for execution in execution_for(method):
                    runtime = (DynamicTargetGraph(target, cache_capacity_buckets=(args.cache_capacity,))
                               if execution != "original_target" else None)
                    proposal = build_proposal(adapters[method], verify=args.verify_proposals) if method != "target" else None
                    tree = tree_for(method, runtime)
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    with proposal.context() if proposal else nullcontext():
                        stats, output = run_decode(method, runtime, tree, prompts[0]["prompt"], 16, prompts[0]["seed"])
                    torch.cuda.synchronize()
                    warmups.append(dict(method=method, execution=execution, seconds=time.perf_counter()-begin,
                                        output_tokens=list(map(int, output)), counters=count_fields(stats),
                                        runtime=runtime.stats() if runtime else None, proposal_runtime=proposal.stats() if proposal else None,
                                        tree_runtime=tree.stats() if tree else None))
                    write_json(out / "warmup.json", warmups)
                    if proposal is not None:
                        proposal_stats = proposal.stats()
                        if not proposal_stats.get("graph_count", 0):
                            raise ArithmeticError(f"No proposal graph used in warmup: {method}")
                        if args.verify_proposals and proposal_stats.get("verification", {}).get("passed") is not True:
                            raise ArithmeticError(f"Proposal numeric comparison failed: {method}")
                    pipeline.update(warmup_completed=len(warmups), warmup_last=f"{execution}:{method}")
                    write_json(out / "pipeline.json", pipeline)
                    if proposal is not None:
                        proposal.close()
                    del runtime, proposal, tree
                    gc.collect()
            write_json(out / "warmup.json", warmups)
            if set(executions) == {"target_only", "proposal_graph"}:
                for method in [m for m in methods if m != "target"]:
                    old, new = [next(row for row in warmups if row["method"] == method
                                     and row["execution"] == execution)
                                for execution in ("target_only", "proposal_graph")]
                    if old["output_tokens"] != new["output_tokens"] or old["counters"] != new["counters"]:
                        raise ArithmeticError(f"Warmup output/counter replay differs for {method}")
            rows, runtimes, proposals, trees = [], {}, {}, {}
            arms = [(execution, method) for method in methods for execution in execution_for(method)]
            pipeline.update(status="running")
            for group_index, prompt in enumerate(prompts):
                order = arms[group_index % len(arms):] + arms[:group_index % len(arms)]
                if (group_index // len(arms)) % 2:
                    order = list(reversed(order))
                for execution, method in order:
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    runtime = None
                    before = None
                    created = False
                    key = f"{execution}:{method}"
                    if execution != "original_target" and key not in runtimes:
                        runtimes[key] = DynamicTargetGraph(target, cache_capacity_buckets=(args.cache_capacity,))
                        if method != "target":
                            proposals[key] = build_proposal(adapters[method], verify=args.verify_proposals)
                        if method.startswith("tree_w"):
                            trees[key] = tree_for(method, runtimes[key])
                        created = True
                    runtime = runtimes.get(key)
                    proposal = proposals.get(key)
                    tree = trees.get(key)
                    tree_before = tree.stats() if tree else None
                    before = runtime.stats() if runtime else None
                    proposal_before = proposal.stats() if proposal else None
                    with proposal.context() if proposal else nullcontext():
                        stats, output = run_decode(method, runtime, tree, prompt["prompt"], args.budget, prompt["seed"])
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - begin
                    output = [int(token) for token in output]
                    validate_sequence(prompt["prompt"], output, context["ranks"])
                    after = runtime.stats() if runtime else None
                    row = dict(pair_id=prompt["pair_id"], source=prompt["source"], prefix=prompt["prefix"],
                               seed=prompt["seed"], method=method, execution=execution, arm=f"{execution}:{method}",
                               generated_tokens=len(output), output_tokens=output,
                               eos=bool(output and output[-1] == EOS), outer_wall_seconds=seconds,
                               tokens_per_second=len(output)/seconds, counters=count_fields(stats), stats=jsonable(stats),
                               sequence_validated=True,
                               runtime_created=created, runtime_delta=runtime_delta(before, after or {}),
                               proposal_runtime_delta=runtime_delta(proposal_before, proposal.stats()) if proposal else None,
                               tree_runtime_delta=runtime_delta(tree_before, tree.stats()) if tree else None)
                    append_row(out / "records.jsonl", row)
                    rows.append(row)
                    pipeline.update(completed_runs=len(rows), elapsed_seconds=time.time()-started,
                                    last_completed=dict(pair_id=row["pair_id"], arm=row["arm"]))
                    write_json(out / "pipeline.json", pipeline)
                    print(json.dumps({key:row[key] for key in ("pair_id", "arm", "generated_tokens", "outer_wall_seconds", "tokens_per_second")}
                                     | {"completed_runs": len(rows)}, ensure_ascii=False), flush=True)
            summary = dict(status="complete", completed_runs=len(rows), elapsed_seconds=time.time()-started,
                           dynamic_runtimes={key:value.stats() for key,value in runtimes.items()},
                           proposal_runtimes={key:value.stats() for key,value in proposals.items()},
                           tree_runtimes={key:value.stats() for key,value in trees.items()},
                           proposal_verified=args.verify_proposals,
                    execution_meaning={"original_target":"Original Target with ordinary execution", "target_only":"Optimized Target with ordinary proposal execution", "proposal_graph":"Existing MLP with Target and proposal graphs", "tree_graph":"Feature tree with Target, proposal and tree graphs"},
                    timing_scope=plan["timing_scope"], runtime_scope=plan["runtime_scope"])
            write_json(out / "runtime_summary.json", summary)
            for proposal in proposals.values():
                proposal.close()
            proposals.clear()
            trees.clear()
            runtimes.clear()
        pipeline.update(status="complete", completed_runs=len(rows), elapsed_seconds=time.time()-started)
        write_json(out / "pipeline.json", pipeline)
        print(json.dumps(dict(status="complete", output=str(out), completed_runs=len(rows))), flush=True)
    except Exception as error:
        pipeline.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.time()-started)
        write_json(out / "pipeline.json", pipeline)
        raise


if __name__ == "__main__":
    main()
