#!/usr/bin/env python3
"""Run paired W1/W2/K1 single-round and complete-request probes."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import json
import math
import os
from pathlib import Path
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


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _source_id(record):
    source = record.get("scene") or record.get("source") or record.get("source_id")
    if source:
        return str(source)
    path = Path(str(record.get("path", "")))
    return path.parents[1].name if len(path.parts) >= 3 else "unknown"


def _crop_path(record):
    value = record.get("path") or record.get("crop_path")
    return None if value is None else str(value)


def select_contexts(records, prefixes, source_limit, scene_names=None, eos_id=12096):
    """Pick the first adequate crop per source and prefix, preserving its metadata."""
    by_source = {}
    for record in records:
        tokens = record.get("tokens")
        if tokens is None:
            continue
        by_source.setdefault(_source_id(record), []).append(record)
    wanted = None if scene_names is None else {str(name) for name in scene_names}
    source_ids = sorted(by_source)
    if wanted is not None:
        source_ids = [name for name in source_ids if name in wanted]
    source_ids = source_ids[:int(source_limit)]
    contexts, missing = [], []
    for source in source_ids:
        candidates = by_source[source]
        for prefix in prefixes:
            selected = next((record for record in candidates
                             if len(record["tokens"]) >= int(prefix) + 4), None)
            if selected is None:
                missing.append(dict(source_id=source, prefix_len=int(prefix)))
                continue
            context_id = f"{source}:p{int(prefix):04d}"
            tokens = [int(token) for token in selected["tokens"][:int(prefix)]]
            contexts.append(dict(
                context_id=context_id,
                source_id=source,
                prefix_len=int(prefix),
                crop_seed=selected.get("seed", selected.get("crop_seed")),
                crop_path=_crop_path(selected),
                prompt=[int(eos_id), *tokens],
            ))
    return contexts, missing, source_ids


def runtime_delta(before, after):
    old, new = before or {}, after or {}

    def difference_map(name):
        old_values = old.get(name, {})
        new_values = new.get(name, {})
        return {key: int(value) - int(old_values.get(key, 0))
                for key, value in new_values.items()
                if int(value) - int(old_values.get(key, 0)) != 0}

    return dict(
        calls=difference_map("calls"),
        fallbacks=difference_map("fallbacks"),
        graph_count_before=int(old.get("graph_count", 0)),
        graph_count_after=int(new.get("graph_count", 0)),
        new_graph_events=list(new.get("graph_events", []))[len(old.get("graph_events", [])):],
        new_cache_events=list(new.get("cache_events", []))[len(old.get("cache_events", [])):],
        copy_bytes=difference_map("copy_bytes"),
    )


def _graph_event_rows(named_deltas):
    rows = []
    total_ms = 0.0
    for owner, delta in named_deltas.items():
        for event in delta.get("new_graph_events", []):
            timing = float(event.get("setup_ms", 0.0))
            if timing <= 0.0:
                timing = sum(float(event.get(key, 0.0)) for key in
                             ("prepare_ms", "warmup_ms", "capture_ms"))
            total_ms += timing
            rows.append(dict(owner=owner, cold_setup_ms=timing, event=event))
    return total_ms, rows


def _action_order(actions, rotation, reverse=False):
    ordered = list(actions)
    if ordered:
        offset = int(rotation) % len(ordered)
        ordered = ordered[offset:] + ordered[:offset]
    if reverse:
        ordered.reverse()
    return ordered


def _first_round(stats):
    rounds = stats.get("probe", {}).get("rounds", [])
    return rounds[0] if rounds else {}


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="Isolated feature_tree source snapshot, such as source_run04")
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="New output directory; existing paths are refused")
    parser.add_argument("--sources", type=int, default=10,
                        help="Maximum number of source scenes after name filtering")
    parser.add_argument("--scene-names", default="",
                        help="Optional comma-separated exact source/scene ids")
    parser.add_argument("--prefixes", default="64,256,512,1024")
    parser.add_argument("--seeds", type=int, default=8,
                        help="Number of paired seeds, starting at --seed-start")
    parser.add_argument("--seed-start", type=int, default=46101)
    parser.add_argument("--actions", default="K1,W1,W2",
                        help="Comma-separated paired actions: K1,W1,W2")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeated single-round measurements per seed and action")
    parser.add_argument("--full-request-repeats", type=int, default=0,
                        help="Complete-request repetitions; set 0 to skip")
    parser.add_argument("--budget", type=int, default=512,
                        help="Complete-request token budget")
    parser.add_argument("--cache-capacity", type=int, default=2048)
    parser.add_argument("--diagnostics", choices=("off", "components"), default="off",
                        help="components records CUDA event spans; off is preferred for the main timing")
    parser.add_argument("--verify-proposals", action="store_true")
    parser.add_argument("--source-scope", default="development")
    return parser


def main(argv=None):
    parser = _make_parser()
    args = parser.parse_args(argv)
    prefixes = [int(item) for item in args.prefixes.split(",") if item.strip()]
    actions = [item.strip().upper() for item in args.actions.split(",") if item.strip()]
    scene_names = [item.strip() for item in args.scene_names.split(",") if item.strip()]
    if (not prefixes or any(prefix < 2 or prefix % 2 for prefix in prefixes)
            or len(prefixes) != len(set(prefixes)) or args.sources < 1
            or args.seeds < 1 or args.repeats < 1 or args.full_request_repeats < 0
            or args.budget < 2 or args.budget % 2
            or args.cache_capacity < 2048
            or not actions or len(actions) != len(set(actions))
            or any(action not in {"K1", "W1", "W2"} for action in actions)):
        parser.error("Invalid source, prefix, seed, action, repeat or capacity setting")

    project = args.project.resolve()
    source = args.source.resolve()
    crops = args.crops.resolve()
    out = args.out.resolve()
    if out.exists():
        parser.error("Output must be a new directory")
    for required in (project, source, crops):
        if not required.exists():
            parser.error(f"Required path does not exist: {required}")
    script_dir = Path(__file__).resolve().parent
    sys.path[:0] = [
        str(script_dir), str(source), str(source / "reference"),
        str(source / "reference" / "runtime"), str(project),
    ]
    os.chdir(project)

    import torch
    from threadpoolctl import threadpool_limits
    from method_adapters import build_context, load_method
    from dynamic_target_graph import DynamicTargetGraph
    from tree_runtime import TargetTreeRuntime
    from proposal_adapters import build_proposal
    from probe_decoder import decode_probe
    from draft_a.corpus_v2 import EOS, validate
    from draft_a.learning_eval import _fp32_runtime
    from draft_a.pf_ranker_eval import cache_for as target_cache_for, validate_sequence

    if int(EOS) != 12096:
        raise ValueError(f"Crop EOS id {EOS} differs from the expected P/F sentinel")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the matched probe")

    raw = torch.load(crops, map_location="cpu", weights_only=True)
    for record in raw:
        validate(record)
    contexts, missing, selected_sources = select_contexts(
        raw, prefixes, args.sources, scene_names or None, EOS,
    )
    if not contexts:
        raise ValueError("No source/prefix context has enough crop tokens")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    pipeline = dict(status="loading", started_at=time.time(), command=sys.argv,
                    source=str(source), project=str(project), crops=str(crops),
                    contexts=len(contexts), completed_rows=0)
    write_json(out / "pipeline.json", pipeline)
    prompts = []
    for context_row in contexts:
        prompts.append({key: value for key, value in context_row.items()})
    write_json(out / "prompts.json", dict(contexts=prompts, missing=missing,
                                          selected_sources=selected_sources,
                                          eos_id=int(EOS)))

    started = time.perf_counter()
    load_started = time.perf_counter()
    context = build_context(project, device="cuda")
    target = context["target"]
    adapter = load_method("pf_mlp", context)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    if len(context["ranks"]) != int(target.sparse_position_vocab_size):
        raise ValueError("Rank map and Target Position vocabulary differ")
    for row in contexts:
        validate_sequence(row["prompt"], [], context["ranks"])
    max_required = max(len(row["prompt"]) + args.budget + 6 for row in contexts)
    if max_required > args.cache_capacity:
        raise ValueError(f"cache capacity {args.cache_capacity} is below required {max_required}")

    plan = dict(
        status="running",
        source_scope=args.source_scope,
        source_snapshot=str(source),
        project=str(project),
        crop_file=str(crops),
        selected_sources=selected_sources,
        contexts=len(contexts),
        missing_contexts=missing,
        prefixes=prefixes,
        seed_count=args.seeds,
        seed_start=args.seed_start,
        actions=actions,
        single_round_repeats=args.repeats,
        full_request_repeats=args.full_request_repeats,
        full_request_budget=args.budget,
        cache_capacity=args.cache_capacity,
        diagnostics=args.diagnostics,
        expected_single_round_rows=len(contexts) * args.seeds * args.repeats * len(actions),
        expected_full_request_rows=len(contexts) * args.seeds * args.full_request_repeats * len(actions),
        target_precision="FP32",
        matmul_precision=torch.get_float32_matmul_precision(),
        tf32_enabled=bool(torch.backends.cuda.matmul.allow_tf32),
        device=torch.cuda.get_device_name(0),
        torch_version=str(torch.__version__),
        python_version=sys.version,
        model_load_ms=load_ms,
        timing_definition={
            "single_round_wall_ms": "Synchronized wall from immediately after prefill through one action round; excludes runtime construction, cache/adapter setup, and prefill.",
            "setup_ms": "Cache reset/allocation, hidden capture hook creation, and adapter preparation.",
            "prefill_ms": "Prompt Target call and online Ranker state construction; synchronized for single-round runs, enqueue time for complete requests.",
            "full_request_wall_ms": "Synchronized outer decode wall including cache/adapter setup, prefill, proposals, Target calls, recovery, commit, and final drain.",
            "components": "Host-call elapsed and CUDA event stream intervals; no component-level synchronization is added.",
        },
        action_semantics={
            "K1_single_round": "action_budget_tokens=2; existing decoder naturally uses its K1 branch.",
            "W1_single_round": "action_budget_tokens=4; one Feature candidate, one tree verification.",
            "W2_single_round": "action_budget_tokens=4; two Feature candidates, one tree verification.",
            "K1_full_request": "force_k1_each_round=True, distinct from K1 single-round budget behavior.",
        },
        proposal_diagnostics=args.verify_proposals,
    )
    write_json(out / "plan.json", plan)
    pipeline.update(status="running", expected_rows=(plan["expected_single_round_rows"]
                                                      + plan["expected_full_request_rows"]))
    write_json(out / "pipeline.json", pipeline)

    states = {}

    def state_for(action):
        state = states.get(action)
        if state is not None:
            return state, False
        width = 1 if action in {"K1", "W1"} else 2
        create_started = time.perf_counter()
        runtime = DynamicTargetGraph(target, cache_capacity_buckets=(args.cache_capacity,))
        tree = (TargetTreeRuntime(target, runtime, graph=True, max_width=width)
                if action in {"W1", "W2"} else None)
        proposal = (build_proposal(adapter, verify=args.verify_proposals)
                    if action in {"W1", "W2"} else None)
        state = dict(
            runtime=runtime,
            tree=tree,
            proposal=proposal,
            width=width,
            constructor_ms=(time.perf_counter() - create_started) * 1000.0,
            created=True,
        )
        states[action] = state
        return state, True

    def run_action(context_row, seed, action, *, scope, repeat_index):
        state, created = state_for(action)
        runtime, tree, proposal = state["runtime"], state["tree"], state["proposal"]
        target_before = runtime.stats()
        tree_before = tree.stats() if tree is not None else {}
        proposal_before = proposal.stats() if proposal is not None else {}
        budget = (2 if action == "K1" else 4) if scope == "single_round" else args.budget
        max_rounds = 1 if scope == "single_round" else None
        force_k1 = bool(scope == "full_request" and action == "K1")
        if action in {"W1", "W2"}:
            cache_factory = lambda core, capacity: runtime.cache_for(
                core, capacity, factory=target_cache_for
            )
        else:
            cache_factory = lambda core, capacity: runtime.cache_for(
                core, capacity, factory=target_cache_for
            )
        torch.cuda.synchronize()
        call_started = time.perf_counter()
        with ExitStack() as stack:
            stack.enter_context(runtime.attention_context())
            if proposal is not None:
                stack.enter_context(proposal.context())
            stats, suffix = decode_probe(
                target,
                adapter.model,
                context["ranker"],
                context["ranks"],
                context_row["prompt"],
                int(seed),
                action_id=action,
                action_budget=budget,
                tree_runtime=tree,
                cache_factory=cache_factory,
                runtime=runtime,
                max_rounds=max_rounds,
                diagnostics=args.diagnostics if scope == "single_round" else "off",
                force_k1_each_round=force_k1,
            )
        torch.cuda.synchronize()
        decode_outer_ms = (time.perf_counter() - call_started) * 1000.0
        suffix = [int(token) for token in suffix]
        validate_sequence(context_row["prompt"], suffix, context["ranks"])
        target_after = runtime.stats()
        tree_after = tree.stats() if tree is not None else {}
        proposal_after = proposal.stats() if proposal is not None else {}
        deltas = dict(
            target=runtime_delta(target_before, target_after),
            tree=runtime_delta(tree_before, tree_after) if tree is not None else {},
            proposal=runtime_delta(proposal_before, proposal_after) if proposal is not None else {},
        )
        cold_graph_ms, cold_graph_events = _graph_event_rows(deltas)
        all_rounds = stats.get("probe", {}).get("rounds", [])
        round_row = (_first_round(stats) if scope == "single_round"
                     else (all_rounds[-1] if all_rounds else {}))
        round_counters = round_row if scope == "single_round" else {}
        probe = stats.get("probe", {})
        round_host = round_row.get("round_wall_ms")
        full_node_count = 0 if action == "K1" else 2 + 2 * state["width"]
        full_read_bucket = None
        if full_node_count:
            full_required_read = len(context_row["prompt"]) - 1 + full_node_count
            full_read_bucket = max(8, 1 << (full_required_read - 1).bit_length())
        timing = dict(
            setup_ms=probe.get("setup_ms"),
            prefill_ms=probe.get("prefill_ms"),
            round_wall_ms=round_host,
            round_enqueue_ms=round_row.get("round_enqueue_ms"),
            full_request_wall_ms=decode_outer_ms if scope == "full_request" else None,
            decode_outer_ms=decode_outer_ms,
        )
        row = dict(
            context_id=context_row["context_id"],
            source_id=context_row["source_id"],
            crop_seed=jsonable(context_row.get("crop_seed")),
            crop_path=context_row.get("crop_path"),
            prefix_ref=context_row["context_id"],
            prefix_len=int(context_row["prefix_len"]),
            seed=int(seed),
            scope=scope,
            action_id=action,
            action_budget_tokens=int(budget),
            repeat_index=int(repeat_index),
            remaining_budget_before=(round_row.get("remaining_budget_before")
                                     if scope == "single_round" else int(budget)),
            R=int(len(suffix) if scope == "full_request" else round_row.get("generated_tokens", 0)),
            eos=bool(suffix and suffix[-1] == EOS) if scope == "full_request"
                else bool(round_row.get("eos", False)),
            eos_stage=round_row.get("eos_stage"),
            round_count=int(stats.get("rounds", 0)),
            round_generated_lengths=list(stats.get("round_generated_lengths", [])),
            accepted_lengths=list(stats.get("accepted_lengths", [])),
            output_tokens=suffix if scope == "full_request" else list(round_row.get("actual_tokens", suffix)),
            actual_tokens=list(round_row.get("actual_tokens", suffix)),
            proposed_p1=round_row.get("proposed_p1"),
            proposed_p2=round_row.get("proposed_p2"),
            ranker_p1_candidate_count=round_row.get("ranker_p1_candidate_count"),
            ranker_p2_candidate_count=round_row.get("ranker_p2_candidate_count"),
            candidate_features=list(round_row.get("candidate_features", [])),
            candidate_count=len(round_row.get("candidate_features", [])),
            accepted_p1=round_row.get("accepted_p1"),
            accepted_feature=round_row.get("accepted_feature"),
            accepted_p2=round_row.get("accepted_p2"),
            selected_feature_branch=round_row.get("selected_feature_branch"),
            actual_p1=round_row.get("actual_p1"),
            actual_feature=round_row.get("actual_feature"),
            actual_p2=round_row.get("actual_p2"),
            commit_path=round_row.get("commit_path"),
            cache_truncate_to=round_row.get("cache_truncate_to"),
            cache_pos_before=(round_row.get("cache_pos_before")
                              if scope == "single_round" else len(context_row["prompt"]) - 1),
            cache_pos_after=round_row.get("cache_pos_after"),
            tree_width=(round_row.get("tree_width", 0) if scope == "single_round"
                        else (state["width"] if action in {"W1", "W2"} else 0)),
            node_count=(round_row.get("node_count", 0) if scope == "single_round"
                        else full_node_count),
            tree_read_bucket=(round_row.get("tree_read_bucket") if scope == "single_round"
                              else full_read_bucket),
            target_calls=round_counters.get("target_calls", stats.get("target_calls", 0)),
            target_input_tokens=round_counters.get("target_input_tokens", stats.get("target_input_tokens", 0)),
            tree_verify_calls=round_counters.get("tree_verify_calls", 0)
                              if scope == "single_round" else stats.get("tree_verify_calls", 0),
            candidate_tree_nodes=round_counters.get("candidate_tree_nodes", 0)
                                 if scope == "single_round" else stats.get("candidate_tree_nodes", 0),
            target_tree_nodes=round_counters.get("target_tree_nodes", 0)
                              if scope == "single_round" else stats.get("target_tree_nodes", 0),
            draft_calls=round_counters.get("draft_calls", stats.get("draft_calls", 0)),
            draft_input_tokens=round_counters.get("draft_input_tokens", stats.get("draft_input_tokens", 0)),
            ranker_calls=round_counters.get("ranker_calls", stats.get("ranker_calls", 0)),
            recovery_calls=round_counters.get("recovery_calls", stats.get("recovery_calls", 0)),
            recovery_target_calls=round_counters.get("recovery_target_calls", stats.get("recovery_calls", 0)),
            recovery_target_input_tokens=round_counters.get("recovery_target_input_tokens", stats.get("recovery_target_input_tokens", 0)),
            tree_copy_bytes=(round_counters.get("tree_copy_bytes", {}) if scope == "single_round"
                             else deltas["tree"].get("copy_bytes", {})),
            component_cpu_ms=round_row.get("component_cpu_ms", {}),
            component_device_ms=round_row.get("component_device_ms", {}),
            component_measurement=("CUDA event stream intervals may include host submission gaps"
                                   if args.diagnostics == "components" and scope == "single_round"
                                   else "disabled"),
            **timing,
            arm_constructor_ms=state["constructor_ms"] if created else 0.0,
            runtime_created=bool(created),
            cold_status=("cold" if created or cold_graph_events else "warm"),
            cold_graph_ms=float(cold_graph_ms),
            new_graph_events=cold_graph_events,
            runtime_delta=deltas,
            counters={key: value for key, value in stats.items()
                      if isinstance(value, (int, float, bool, str)) or
                      (isinstance(value, list) and key not in {"accepted_lengths", "round_generated_lengths"})},
            sequence_validated=True,
            diagnostics=args.diagnostics if scope == "single_round" else "off",
        )
        row["outer_wall_including_arm_constructor_ms"] = (
            decode_outer_ms + state["constructor_ms"] if created else decode_outer_ms
        )
        row["full_request_wall_including_arm_constructor_ms"] = (
            row["outer_wall_including_arm_constructor_ms"] if scope == "full_request" else None
        )
        return row

    rows_written = 0
    try:
        with threadpool_limits(limits=1), torch.inference_mode(), _fp32_runtime(torch), \
                torch.autocast("cuda", enabled=False):
            seed_rows = [(context_row, args.seed_start + seed_index, seed_index)
                         for context_row in contexts
                         for seed_index in range(args.seeds)]
            for context_index, (context_row, seed, seed_index) in enumerate(seed_rows):
                for repeat_index in range(args.repeats):
                    order = _action_order(actions, context_index + seed_index + repeat_index,
                                          reverse=((context_index + seed_index + repeat_index) // max(1, len(actions))) % 2 == 1)
                    for action in order:
                        row = run_action(context_row, seed, action,
                                         scope="single_round", repeat_index=repeat_index)
                        append_jsonl(out / "records.jsonl", row)
                        rows_written += 1
                        pipeline.update(completed_rows=rows_written,
                                        last_completed=dict(context_id=row["context_id"],
                                                            seed=seed, action=action,
                                                            scope=row["scope"]))
                        write_json(out / "pipeline.json", pipeline)
                        print(json.dumps(dict(rows=rows_written, context_id=row["context_id"],
                                              seed=seed, action=action, R=row["R"],
                                              round_wall_ms=row["round_wall_ms"],
                                              cold_status=row["cold_status"]),
                                         ensure_ascii=False), flush=True)
            if args.full_request_repeats:
                for context_index, (context_row, seed, seed_index) in enumerate(seed_rows):
                    for repeat_index in range(args.full_request_repeats):
                        order = _action_order(actions, context_index + seed_index + repeat_index,
                                              reverse=((context_index + seed_index + repeat_index) // max(1, len(actions))) % 2 == 1)
                        for action in order:
                            row = run_action(context_row, seed, action,
                                             scope="full_request", repeat_index=repeat_index)
                            append_jsonl(out / "records.jsonl", row)
                            rows_written += 1
                            pipeline.update(completed_rows=rows_written,
                                            last_completed=dict(context_id=row["context_id"],
                                                                seed=seed, action=action,
                                                                scope=row["scope"]))
                            write_json(out / "pipeline.json", pipeline)
                            print(json.dumps(dict(rows=rows_written, context_id=row["context_id"],
                                                  seed=seed, action=action, R=row["R"],
                                                  full_request_wall_ms=row["full_request_wall_ms"],
                                                  cold_status=row["cold_status"]),
                                             ensure_ascii=False), flush=True)
        runtime_summary = {
            action: dict(
                runtime=state["runtime"].stats(),
                tree=state["tree"].stats() if state["tree"] is not None else None,
                proposal=state["proposal"].stats() if state["proposal"] is not None else None,
                arm_constructor_ms=state["constructor_ms"],
            )
            for action, state in states.items()
        }
        write_json(out / "runtime_summary.json", runtime_summary)
        summary = dict(
            status="complete",
            completed_rows=rows_written,
            expected_single_round_rows=plan["expected_single_round_rows"],
            expected_full_request_rows=plan["expected_full_request_rows"],
            elapsed_seconds=time.perf_counter() - started,
            runtime_created_by_action={action: True for action in states},
            runtime_summary=runtime_summary,
        )
        write_json(out / "summary.json", summary)
        pipeline.update(status="complete", completed_rows=rows_written,
                        elapsed_seconds=time.perf_counter() - started)
        write_json(out / "pipeline.json", pipeline)
        print(json.dumps(dict(status="complete", output=str(out), rows=rows_written)), flush=True)
    except Exception as error:
        pipeline.update(status="failed", completed_rows=rows_written,
                        error=f"{type(error).__name__}: {error}")
        write_json(out / "pipeline.json", pipeline)
        raise
    finally:
        for state in states.values():
            if state["proposal"] is not None:
                state["proposal"].close()


if __name__ == "__main__":
    main()
