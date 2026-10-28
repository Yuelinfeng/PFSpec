#!/usr/bin/env python3
"""Bounded correctness probe for combined Position rescue and Feature reuse."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _compare_maps(left, right):
    keys = sorted(set(left) | set(right))
    missing = [key for key in keys if key not in left or key not in right]
    differing = [key for key in keys
                 if key in left and key in right and left[key] != right[key]]
    return dict(equal=not missing and not differing,
                missing=missing, differing=differing)


def _semantic_counts(stats):
    keys = (
        "rounds", "proposed", "verified", "accepted", "accepted_lengths",
        "round_generated_lengths", "generated_tokens", "eos",
        "position_fallbacks", "k1_steps", "position_proposed",
        "position_verified", "position_accepted", "feature_proposed",
        "feature_verified", "feature_accepted", "feature_attempts",
        "feature_branch_hits", "feature_all_rejected",
        "feature_branch_index_counts", "eos_proposed", "eos_verified",
        "eos_accepted",
    )
    return {key: stats[key] for key in keys if key in stats}


class _LerpProbe:
    """Record FP32 torch.lerp versus the ordinary weighted-sum expression."""

    def __init__(self, torch, vocab_size: int, expected_weight: float):
        self.torch = torch
        self.vocab_size = int(vocab_size)
        self.expected_weight = float(expected_weight)
        self.rows = []
        self._original = None

    def __enter__(self):
        self._original = self.torch.lerp
        original = self._original

        def observed(start, end, weight, *args, **kwargs):
            result = original(start, end, weight, *args, **kwargs)
            if (self.torch.is_tensor(start) and self.torch.is_tensor(end)
                    and self.torch.is_tensor(result)
                    and start.dtype == self.torch.float32
                    and end.dtype == self.torch.float32
                    and result.dtype == self.torch.float32
                    and start.ndim == end.ndim == result.ndim == 1
                    and start.numel() == end.numel() == result.numel() == self.vocab_size
                    and isinstance(weight, (int, float))
                    and float(weight) == self.expected_weight):
                ordinary = ((1.0 - float(weight)) * start
                            + float(weight) * end)
                ordinary_finite = bool(self.torch.isfinite(ordinary).all().item())
                lerp_finite = bool(self.torch.isfinite(result).all().item())
                if ordinary_finite and lerp_finite:
                    difference = (ordinary - result).abs()
                    mass_ordinary = float(ordinary.sum().item())
                    mass_lerp = float(result.sum().item())
                    support_diff = int(self.torch.count_nonzero(
                        (ordinary > 0) != (result > 0)
                    ).item())
                    row = dict(
                        all_finite=True,
                        max_tv=float(difference.sum().item() * 0.5),
                        max_abs=float(difference.max().item()),
                        ordinary_mass=mass_ordinary,
                        lerp_mass=mass_lerp,
                        mass_abs_delta=abs(mass_ordinary - mass_lerp),
                        ordinary_mass_error=abs(mass_ordinary - 1.0),
                        lerp_mass_error=abs(mass_lerp - 1.0),
                        support_equal=(support_diff == 0),
                        support_different_values=support_diff,
                    )
                else:
                    row = dict(
                        all_finite=False, max_tv=None, max_abs=None,
                        ordinary_mass=None, lerp_mass=None, mass_abs_delta=None,
                        ordinary_mass_error=None, lerp_mass_error=None,
                        support_equal=False, support_different_values=None,
                    )
                self.rows.append(row)
            return result

        self.torch.lerp = observed
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.torch.lerp = self._original
        self._original = None
        return False


def _mixture_summary(rows):
    if not rows:
        return dict(calls=0, all_finite=False, all_support_equal=False)
    finite_rows = [row for row in rows if row["all_finite"]]
    return dict(
        calls=len(rows),
        all_finite=len(finite_rows) == len(rows),
        max_tv=max((row["max_tv"] for row in finite_rows), default=None),
        max_abs=max((row["max_abs"] for row in finite_rows), default=None),
        max_mass_abs_delta=max((row["mass_abs_delta"] for row in finite_rows), default=None),
        max_ordinary_mass_error=max((row["ordinary_mass_error"] for row in finite_rows), default=None),
        max_lerp_mass_error=max((row["lerp_mass_error"] for row in finite_rows), default=None),
        all_support_equal=(len(finite_rows) == len(rows)
                           and all(row["support_equal"] for row in finite_rows)),
        support_different_calls=sum(not row["support_equal"] for row in rows),
        max_support_different_values=max(
            (row["support_different_values"] or 0 for row in rows), default=0
        ),
    )


def _action_specs():
    return [
        ("w1", "w1", None),
        ("rescue", "rescue", None),
        ("combined0", "combined", 0.0),
        ("fast25", "fast_reuse", 0.25),
        ("combined25_first", "combined", 0.25),
        ("combined25_repeat", "combined", 0.25),
    ]


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--source", type=Path,
                        help="Existing feature_tree source snapshot")
    parser.add_argument("--crops", type=Path)
    parser.add_argument("--out", type=Path,
                        help="New result directory")
    parser.add_argument("--seed", type=int, default=50917,
                        help="New paired seed for this bounded probe")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--cache-capacity", type=int, default=512)
    parser.add_argument("--scene-names", default=(
        "030345aa-f00e-4922-b50f-b56055702045,"
        "2dc39940-b53e-4451-84f8-ce8cf3aa9171"
    ))
    return parser


def _arm_snapshot(arm):
    runtime = arm["runtime"]
    proposal = arm["proposal"]
    tree = arm["tree"]
    rescue = arm.get("rescue")
    return dict(
        runtime=runtime.stats(),
        proposal=proposal.stats(),
        tree=tree.stats(),
        rescue=(rescue.stats() if rescue is not None else None),
    )


def _graph_usage(arm):
    runtime = arm["runtime"].stats()
    tree = arm["tree"].stats()
    rescue = arm.get("rescue")
    rescue_stats = rescue.stats() if rescue is not None else None

    def calls(stats):
        return int((stats or {}).get("calls", {}).get("graph_forward", 0))

    return dict(
        target_runtime_graph_count=int(runtime.get("graph_count", 0)),
        target_runtime_graph_forward=calls(runtime),
        tree_graph_count=int(tree.get("graph_count", 0)),
        tree_graph_forward=calls(tree),
        rescue_graph_count=(int(rescue_stats.get("graph_count", 0))
                            if rescue_stats is not None else 0),
        rescue_graph_forward=calls(rescue_stats),
    )


def _run_action(action, arm, prompt_row, context, *, budget, seed,
                torch, runner, methods, cache_factory, validate_sequence,
                weight, vocab_size):
    arm["reuse_weight"] = float(weight or 0.0)
    before = _arm_snapshot(arm)
    torch.cuda.synchronize()
    started = time.perf_counter()
    mixture_probe = None
    if weight is not None and float(weight) > 0.0:
        mixture_probe = _LerpProbe(torch, vocab_size, float(weight))
    probe_context = mixture_probe if mixture_probe is not None else _NullContext()
    with probe_context:
        stats, output = runner._run_arm(
            arm, None, context, prompt_row["prompt"], int(budget), int(seed),
            decode_method=None, decode_w1=methods["w1"],
            decode_probability=None, target_cache_for=cache_factory,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = [int(token) for token in output]
    validate_sequence(prompt_row["prompt"], output, context["ranks"])
    if len(output) > int(budget):
        raise RuntimeError(f"{action} generated more than the requested budget")
    after = _arm_snapshot(arm)
    semantic = _semantic_counts(stats)
    core = runner.core_counters(stats)
    qmix_rows = mixture_probe.rows if mixture_probe is not None else []
    expected_mix_calls = int(stats.get("feature_reuse_mixed_proposals", 0))
    if len(qmix_rows) != expected_mix_calls:
        raise RuntimeError(
            f"{action}: captured {len(qmix_rows)} lerp operations, "
            f"decoder recorded {expected_mix_calls} mixed proposals"
        )
    runtime_delta = runner.runtime_delta(before["runtime"], after["runtime"])
    proposal_delta = runner.runtime_delta(before["proposal"], after["proposal"])
    tree_delta = runner.runtime_delta(before["tree"], after["tree"])
    rescue_delta = (runner.runtime_delta(before["rescue"], after["rescue"])
                    if before["rescue"] is not None else None)
    row = dict(
        pair_id=prompt_row["pair_id"], source=prompt_row["source"],
        crop_path=prompt_row["crop_path"], crop_seed=prompt_row["crop_seed"],
        prefix=prompt_row["prefix"], seed=int(seed), budget=int(budget),
        action=action, decoder=arm["name"], reuse_weight=weight,
        output_tokens=output, generated_tokens=len(output),
        eos=bool(stats.get("eos", False)), elapsed_seconds=elapsed,
        sequence_validated=True, semantic_counters=semantic,
        core_counters=core,
        execution_counters={key: stats.get(key) for key in (
            "target_calls", "target_input_tokens", "draft_calls", "draft_input_tokens",
            "target_tree_nodes", "candidate_tree_nodes", "recovery_calls",
            "position_backup_nodes", "position_backup_rejections", "position_rescue_hits",
            "feature_reuse_policy_updates", "feature_reuse_mixed_proposals",
        ) if key in stats},
        runtime_delta=runtime_delta, proposal_runtime_delta=proposal_delta,
        tree_runtime_delta=tree_delta, rescue_runtime_delta=rescue_delta,
        runtime_after=after,
        graph_usage=_graph_usage(arm),
        lerp_comparisons=qmix_rows,
        lerp_summary=_mixture_summary(qmix_rows),
        stats=runner.jsonable(stats),
    )
    return row


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def _pair_comparisons(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["pair_id"], {})[row["action"]] = row
    expected = {"w1", "rescue", "combined0", "fast25",
                "combined25_first", "combined25_repeat"}
    results = []
    for pair_id, arms in sorted(grouped.items()):
        if set(arms) != expected:
            raise RuntimeError(f"incomplete action set for {pair_id}: {sorted(arms)}")
        w1, rescue = arms["w1"], arms["rescue"]
        combined0 = arms["combined0"]
        fast25 = arms["fast25"]
        combined_a, combined_b = arms["combined25_first"], arms["combined25_repeat"]

        def relation(left, right):
            return dict(
                output_equal=left["output_tokens"] == right["output_tokens"],
                semantic_counters=_compare_maps(
                    left["semantic_counters"], right["semantic_counters"]
                ),
                full_core_counters=_compare_maps(left["core_counters"], right["core_counters"]),
            )

        rescue_hits = int(rescue["execution_counters"].get("position_rescue_hits", 0))
        combined0_hits = int(combined0["execution_counters"].get("position_rescue_hits", 0))
        combined_hits = int(combined_a["execution_counters"].get("position_rescue_hits", 0))
        recovery = dict(
            w1_minus_rescue=(int(w1["execution_counters"].get("recovery_calls", 0))
                             - int(rescue["execution_counters"].get("recovery_calls", 0))),
            rescue_hits=rescue_hits,
            w1_minus_combined0=(int(w1["execution_counters"].get("recovery_calls", 0))
                                - int(combined0["execution_counters"].get("recovery_calls", 0))),
            combined0_hits=combined0_hits,
            fast25_minus_combined25=(int(fast25["execution_counters"].get("recovery_calls", 0))
                                     - int(combined_a["execution_counters"].get("recovery_calls", 0))),
            combined25_hits=combined_hits,
        )
        recovery["w1_minus_rescue_equals_hits"] = recovery["w1_minus_rescue"] == rescue_hits
        recovery["w1_minus_combined0_equals_hits"] = recovery["w1_minus_combined0"] == combined0_hits
        recovery["fast25_minus_combined25_equals_hits"] = recovery["fast25_minus_combined25"] == combined_hits
        recovery["combined25_repeat_recovery_equal"] = (
            int(combined_a["execution_counters"].get("recovery_calls", 0))
            == int(combined_b["execution_counters"].get("recovery_calls", 0))
            and combined_hits
            == int(combined_b["execution_counters"].get("position_rescue_hits", 0))
        )
        comparisons = dict(
            pair_id=pair_id, source=w1["source"], prefix=w1["prefix"], seed=w1["seed"],
            w1_vs_rescue=relation(w1, rescue),
            w1_vs_combined_weight0=relation(w1, combined0),
            fast_reuse25_vs_combined25=relation(fast25, combined_a),
            combined25_repeat=relation(combined_a, combined_b),
            recovery=recovery,
            all_sequences_validated=all(arms[name]["sequence_validated"] for name in expected),
        )
        comparisons["passed"] = all((
            comparisons["w1_vs_rescue"]["output_equal"],
            comparisons["w1_vs_rescue"]["semantic_counters"]["equal"],
            comparisons["w1_vs_combined_weight0"]["output_equal"],
            comparisons["w1_vs_combined_weight0"]["semantic_counters"]["equal"],
            comparisons["fast_reuse25_vs_combined25"]["output_equal"],
            comparisons["fast_reuse25_vs_combined25"]["semantic_counters"]["equal"],
            comparisons["combined25_repeat"]["output_equal"],
            comparisons["combined25_repeat"]["semantic_counters"]["equal"],
            recovery["w1_minus_rescue_equals_hits"],
            recovery["w1_minus_combined0_equals_hits"],
            recovery["fast25_minus_combined25_equals_hits"],
            recovery["combined25_repeat_recovery_equal"],
            comparisons["all_sequences_validated"],
        ))
        results.append(comparisons)
    return results


def _mixture_summary_by_action(rows):
    grouped = {name: [] for name in ("fast25", "combined25_first", "combined25_repeat")}
    for row in rows:
        if row["action"] in grouped:
            grouped[row["action"]].extend(row["lerp_comparisons"])
    return {name: _mixture_summary(values) for name, values in grouped.items()}


def _self_test():
    assert [action for action, _decoder, _weight in _action_specs()] == [
        "w1", "rescue", "combined0", "fast25",
        "combined25_first", "combined25_repeat",
    ]
    assert _compare_maps({"a": 1}, {"a": 1})["equal"]
    assert not _compare_maps({"a": 1}, {"a": 2})["equal"]
    return {"passed": True, "action_count": len(_action_specs())}


def main():
    parser = _parser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), sort_keys=True))
        return
    if args.project is None or args.source is None or args.crops is None or args.out is None:
        parser.error("--project, --source, --crops, and --out are required")
    if (args.budget != 64 or args.cache_capacity < 256 + args.budget + 8):
        parser.error("the bounded probe requires budget 64 and capacity at least 328")
    project, source, crops, out = (
        args.project.resolve(), args.source.resolve(), args.crops.resolve(), args.out.resolve()
    )
    if out.exists():
        parser.error("--out must name a new directory")
    if not project.is_dir() or not source.is_dir() or not crops.is_file():
        parser.error("project, source, or crops path does not exist")
    out.mkdir(parents=True)
    pipeline = dict(status="starting", command=sys.argv, started_at=time.time(),
                    completed_runs=0, error=None)
    _write_json(out / "pipeline.json", pipeline)
    arms = {}
    try:
        script_dir = Path(__file__).resolve().parent
        sys.path[:0] = [str(script_dir), str(source), str(source / "reference"),
                        str(source / "reference" / "runtime"), str(project)]
        os.chdir(project)
        import torch
        from threadpoolctl import threadpool_limits
        import evaluate_combined as runner
        from method_adapters import build_context, load_method
        from dynamic_target_graph import DynamicTargetGraph
        from tree_runtime import TargetTreeRuntime
        from position_rescue_runtime import PositionRescueRuntime
        from tree_decoder import decode as decode_w1
        from position_rescue_decoder import decode as decode_rescue
        from combined_decoder import decode as decode_combined
        from feature_reuse_fast_decoder import decode as decode_fast_reuse
        from draft_a.corpus_v2 import EOS, validate
        from draft_a.pf_ranker_eval import cache_for as target_cache_for, validate_sequence
        from draft_a.learning_eval import _fp32_runtime
        from draft_a.pf_feature_unified import P_VOCAB, EOS as FEATURE_EOS

        if int(EOS) != int(FEATURE_EOS):
            raise RuntimeError("EOS definitions differ between corpus and PF runtime")
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Target graph verification")
        raw = torch.load(crops, map_location="cpu", weights_only=True)
        for record in raw:
            validate(record)
        scene_names = [item.strip() for item in args.scene_names.split(",") if item.strip()]
        selected = runner.select_crops(raw, 256, 2, scene_names)
        if len(selected) != 2:
            raise ValueError(f"expected two selected sources, found {len(selected)}")
        context = build_context(project, device="cuda")
        target = context["target"]
        shared = __import__("draft_a.pf_feature_unified", fromlist=["_shared"])._shared()
        prompts = []
        for source_index, (source_id, record) in enumerate(selected):
            for prefix in (0, 64, 256):
                pair_id = f"s{source_index:02d}-p{prefix:04d}-seed{args.seed}"
                prompt = runner.make_prompt(source_id, record, prefix, args.seed,
                                            int(EOS), pair_id)
                shared.validate_sequence(prompt["prompt"], [], context["ranks"])
                prompts.append(prompt)

        adapters = {name: load_method("pf_mlp", context)
                    for name in ("w1", "rescue", "combined", "fast_reuse")}
        arm_args = SimpleNamespace(cache_capacity=int(args.cache_capacity),
                                   reuse_weight=0.25)
        for name in ("w1", "rescue", "combined", "fast_reuse"):
            arms[name] = runner._new_arm(
                name, context, target, arm_args,
                DynamicTargetGraph=DynamicTargetGraph,
                TargetTreeRuntime=TargetTreeRuntime,
                adapter=adapters[name], load_method=load_method,
            )
            if name in {"combined", "fast_reuse"}:
                arms[name]["reuse_weight"] = 0.25

        plan = dict(
            source_snapshot=str(source), crops=str(crops),
            sources=[name for name, _record in selected],
            prefixes=[0, 64, 256], seed=int(args.seed), budget=int(args.budget),
            cache_capacity=int(args.cache_capacity), temperature=0.9, top_p=0.9,
            target_precision="FP32", matmul_precision="highest", tf32_enabled=False,
            actions=[dict(action=name, decoder=decoder, reuse_weight=weight)
                     for name, decoder, weight in _action_specs()],
            repetitions=6 * len(prompts),
            runtime_ownership=("W1, rescue, combined and fast reuse each keep separate runtimes and proposal graphs; combined weight 0 and 0.25 reuse the same combined arm; the final 0.25 call repeats with the same seed."),
            comparison_scope=("W1/rescue/combined weight 0 compare outputs and semantic counters; fast reuse/combined weight 0.25 compare paired outputs and semantic counters; recovery reduction is checked against Position rescue hits; prefix 0 exercises BOS-only decoding."),
            counter_scope=("Semantic counters cover sampling and acceptance. Full core counters remain in every row; Target token and tree node counts can differ when the backup node is evaluated."),
            mixture_scope=("During each weight 0.25 decode, observe FP32 torch.lerp(q,p,w) and compare with the ordinary expression (1-w)*q+w*p. Record half-L1, mass, support and finiteness without changing the returned lerp tensor."),
            timing_scope=("Elapsed times are diagnostic only; added mixture measurements synchronize FP32 vectors and are not performance results."),
        )
        _write_json(out / "plan.json", plan)
        _write_json(out / "prompts.json", dict(prompts=prompts, eos_id=int(EOS)))
        pipeline.update(status="running", expected_runs=plan["repetitions"])
        _write_json(out / "pipeline.json", pipeline)

        with threadpool_limits(limits=1), torch.inference_mode(), _fp32_runtime(torch), torch.autocast("cuda", enabled=False):
            rows = []
            specs = _action_specs()
            for prompt_index, prompt_row in enumerate(prompts):
                base_specs = specs[:-1]
                rotation = prompt_index % len(base_specs)
                order = base_specs[rotation:] + base_specs[:rotation]
                order.append(specs[-1])
                for action, decoder_name, weight in order:
                    arm = arms[decoder_name]
                    cache_factory = lambda core, capacity: arm["runtime"].cache_for(
                        core, capacity, factory=target_cache_for
                    )
                    methods = dict(w1=decode_w1, rescue=decode_rescue,
                                   combined=decode_combined, fast_reuse=decode_fast_reuse)
                    row = _run_action(
                        action, arm, prompt_row, context, budget=args.budget,
                        seed=args.seed, torch=torch, runner=runner, methods=methods,
                        cache_factory=cache_factory, validate_sequence=validate_sequence,
                        weight=weight, vocab_size=int(FEATURE_EOS) - int(P_VOCAB),
                    )
                    row["runtime_created"] = False
                    _append_jsonl(out / "records.jsonl", row)
                    rows.append(row)
                    pipeline["completed_runs"] = len(rows)
                    _write_json(out / "pipeline.json", pipeline)

        comparisons = _pair_comparisons(rows)
        _write_json(out / "comparisons.json", dict(comparisons=comparisons))
        mixture = _mixture_summary_by_action(rows)
        _write_json(out / "mixture_comparison.json", mixture)
        mixture_actions = {"fast25", "combined25_first", "combined25_repeat"}
        expected_mix_calls = {
            action: sum(int(row["execution_counters"].get("feature_reuse_mixed_proposals", 0))
                        for row in rows if row["action"] == action)
            for action in mixture_actions
        }
        captured_mix_calls = {
            action: sum(len(row["lerp_comparisons"])
                        for row in rows if row["action"] == action)
            for action in mixture_actions
        }
        graph_usage = {name: _graph_usage(arm) for name, arm in arms.items()}
        rescue_hit_count = sum(int(row["execution_counters"].get("position_rescue_hits", 0))
                               for row in rows if row["action"] in {
                                   "rescue", "combined0", "combined25_first", "combined25_repeat"
                               })
        summary = dict(
            status="passed",
            prompt_count=len(prompts), run_count=len(rows),
            comparison_count=len(comparisons), graph_usage=graph_usage,
            total_position_rescue_hits=rescue_hit_count,
            lerp_calls_match_decoder_counts=(expected_mix_calls == captured_mix_calls),
            mixture=mixture,
            comparisons_passed=all(row["passed"] for row in comparisons),
            bos_comparison_count=sum(1 for row in comparisons if row["prefix"] == 0),
        )
        expected_graphs = all(
            values["tree_graph_forward"] + values["rescue_graph_forward"] > 0
            for values in graph_usage.values()
        )
        expected_mixes = all(
            mixture[name]["calls"] > 0 and mixture[name]["all_finite"]
            for name in ("fast25", "combined25_first", "combined25_repeat")
        )
        support_ok = all(
            mixture[name]["all_support_equal"]
            for name in ("fast25", "combined25_first", "combined25_repeat")
        )
        summary["graphs_used_for_every_arm"] = expected_graphs
        summary["mixture_captured_for_every_positive_weight_arm"] = expected_mixes
        summary["mixture_support_equal_for_every_call"] = support_ok
        summary["passed"] = bool(
            len(prompts) == 6 and len(rows) == 36 and len(comparisons) == 6
            and
            summary["comparisons_passed"] and summary["lerp_calls_match_decoder_counts"]
            and expected_graphs and expected_mixes and support_ok
            and rescue_hit_count > 0
        )
        summary["status"] = "passed" if summary["passed"] else "failed"
        _write_json(out / "summary.json", summary)
        pipeline.update(status="complete" if summary["passed"] else "failed",
                        completed_runs=len(rows), finished_at=time.time())
        _write_json(out / "pipeline.json", pipeline)
        for arm in arms.values():
            runner._close_arm(arm)
        arms.clear()
        if not summary["passed"]:
            raise AssertionError("bounded combined verification did not meet all comparisons")
        print(json.dumps(dict(status="passed", output=str(out),
                              prompts=len(prompts), runs=len(rows),
                              rescue_hits=rescue_hit_count), ensure_ascii=False), flush=True)
    except Exception as error:
        pipeline.update(status="failed", error=f"{type(error).__name__}: {error}",
                        finished_at=time.time())
        _write_json(out / "pipeline.json", pipeline)
        for arm in arms.values():
            try:
                import evaluate_combined as runner
                runner._close_arm(arm)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
