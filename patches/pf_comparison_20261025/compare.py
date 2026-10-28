"""Compare saved proposal models on the existing Target execution backend."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
import gc
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
from types import MethodType


NATIVE = ("opt_target", "ranker_only", "medusa_lowrank", "hydra_conditional",
          "eagle_v3_g3", "eagle_v4_reuse_g3")
FEATURE = ("b0", "b1_m", "b1_h", "b1_e")
ABLATIONS = ("full", "no_candidate_position", "no_previous_feature", "no_target_state")
MAIN = ("opt_target", "ranker_only", "pf_w1", "pf_rescue", "medusa_lowrank",
        "hydra_conditional", "eagle_v3_g3", "eagle_v4_reuse_g3")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def count_values(stats):
    return {k: v for k, v in stats.items()
            if isinstance(v, (int, bool, list, dict))
            and not any(w in k.lower() for w in
                        ("seconds", "time", "rate", "profile", "precision"))}


def expand_methods(names, training_seeds):
    result = []
    for name in names:
        if name in FEATURE:
            result.extend(f"{name}:s{seed}" for seed in training_seeds)
        elif name in NATIVE or name in ("pf_w1", "pf_rescue") or name in {
                "ablate_" + mode for mode in ABLATIONS}:
            result.append(name)
        else:
            raise ValueError(f"Unknown method {name!r}")
    if not result or len(set(result)) != len(result):
        raise ValueError("Methods must be nonempty and unique")
    return result


def paired_summary(rows, methods, references, *, samples=4000, seed=251025):
    """Average log throughput ratios within source, then across sources."""
    indexed = {}
    for row in rows:
        key = (row["case_id"], row["method"])
        if key in indexed:
            raise ValueError(f"Duplicate observation {key}")
        if (row["generated_tokens"] <= 0 or row["seconds"] <= 0
                or not math.isfinite(row["seconds"])):
            raise ValueError("Nonpositive or nonfinite measurement")
        indexed[key] = row
    cases = sorted({key[0] for key in indexed})
    for case in cases:
        if {m for c, m in indexed if c == case} != set(methods):
            raise ValueError(f"Incomplete method group {case}")
        sources = {indexed[(case, m)]["source"] for m in methods}
        if len(sources) != 1:
            raise ValueError(f"Mismatched source for {case}")
    result = {}
    for name in methods:
        own = [indexed[(case, name)] for case in cases]
        counts = defaultdict(int)
        for row in own:
            for k, v in row["counts"].items():
                if isinstance(v, int) and not isinstance(v, bool):
                    counts[k] += v
        item = dict(runs=len(own), tokens=sum(r["generated_tokens"] for r in own),
                    seconds=sum(r["seconds"] for r in own), counts=dict(counts),
                    comparisons={})
        item["pooled_tokens_per_second"] = item["tokens"] / item["seconds"]
        for reference in references:
            if name == reference or reference not in methods:
                continue
            by_source = defaultdict(list)
            for case in cases:
                a, b = indexed[(case, name)], indexed[(case, reference)]
                ratio = ((a["generated_tokens"] / a["seconds"]) /
                         (b["generated_tokens"] / b["seconds"]))
                by_source[a["source"]].append(math.log(ratio))
            values = [sum(v) / len(v) for _, v in sorted(by_source.items())]
            point = math.exp(sum(values) / len(values))
            rng = random.Random(seed)
            boot = sorted(math.exp(sum(rng.choices(values, k=len(values))) / len(values))
                          for _ in range(samples))
            item["comparisons"][reference] = dict(
                source_equal_throughput_ratio=point,
                ci95=[boot[int(.025 * (samples - 1))], boot[int(.975 * (samples - 1))]],
                sources=len(values), sources_faster=sum(x > 0 for x in values),
                source_ratios={s: math.exp(sum(v) / len(v)) for s, v in sorted(by_source.items())})
        result[name] = item
    return dict(prompt_groups=len(cases), methods=result,
                interval_unit="scene source; repeated seeds are kept within source")


def _load_python(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ProposalGroup:
    def __init__(self, members):
        self.members = members

    @contextmanager
    def context(self):
        with ExitStack() as stack:
            for member in self.members:
                stack.enter_context(member.context())
            yield

    def stats(self):
        return {str(i): member.stats() for i, member in enumerate(self.members)}

    def close(self):
        for member in self.members:
            member.close()


def load_feature(name, context, args, api):
    import torch
    from draft_a.pf_feature_unified import MLPAdapter
    from bench_gaussiangpt_pf_block2_cow import load_feature_model
    from draft_a.feature_target_v2 import HEAD
    from torch import nn

    if ":s" in name:
        kind, training_seed = name.split(":s")
        path = args.feature_root / "b1_training" / training_seed / kind / "selected.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["method"] != kind or int(saved["seed"]) != int(training_seed):
            raise ValueError(f"Checkpoint identity differs: {path}")
        module = _load_python("matched_feature_models", args.feature_root / "b1_feature_models.py")
        model = module.make_model(kind, context["target"]).to(context["device"]).float()
        model.load_state_dict(saved["state"], strict=True)
        training = json.loads((path.parent / "summary.json").read_text())
        metadata = {k: training[k] for k in
                    ("seed", "parameters", "shared_fixed_parameters", "objective", "optimizer",
                     "epochs", "train_rows", "validation_rows", "train_keys", "validation_keys")}
        if kind == "b1_e":
            import torch.nn.functional as F
            original_forward = model.forward
            model.register_buffer("static_causal_mask", torch.tensor(
                [[False, True], [False, False]], device=context["device"]), persistent=False)

            def static_forward(self, state, previous_feature, position):
                ids = torch.stack((previous_feature + 8000, position), -1)
                embed = F.embedding(ids, self.token_embeddings)
                x = self.state_project(self.state_norm(state)).unsqueeze(1) + self.token_project(embed)
                x = x + self.role.weight.unsqueeze(0)
                x = self.block(x, src_mask=self.static_causal_mask)
                pred = F.rms_norm(state + self.out(x[:, -1]), (1024,))
                raw = F.linear(pred, self.feature_weight)
                return pred, 15. * torch.tanh(raw.float() / 15.)

            model.eval().requires_grad_(False)
            # Moving this constant allocation does not change the trained predictor.
            state = torch.arange(2048, device=context["device"], dtype=torch.float32).reshape(2, 1024) / 2048
            previous_feature = torch.tensor([0, 33], device=context["device"])
            position = torch.tensor([1, 203], device=context["device"])
            with torch.inference_mode():
                old = original_forward(state, previous_feature, position)
                new = static_forward(model, state, previous_feature, position)
            if not all(torch.equal(a, b) for a, b in zip(old, new)):
                raise ArithmeticError("Preallocated causal mask changed B1-E output")
            model.forward = MethodType(static_forward, model)
            metadata["execution_change"] = "same two-token causal mask allocated before forward; exact output comparison passed"
    else:
        mode = name.removeprefix("ablate_")
        path = args.ablation_root / "feature_models" / (mode + ".pt")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["ablation"] != mode:
            raise ValueError(f"Checkpoint ablation differs: {path}")
        base = load_feature_model(path, HEAD, context["target"], context["device"],
                                  context["p_vocab"], context["eos"])[0]

        original = _load_python("saved_input_ablation", args.ablation_root / "train_feature_inputs.py")
        config = {k: v for k, v in saved["model_config"].items() if k != "hidden_dim"}
        model = original.AblatedDraft(base.frozen_head, ablation=mode, **config)
        model.load_state_dict(base.state_dict(), strict=True)
        model = model.to(context["device"]).float()
        training = json.loads((args.ablation_root / "feature_models/training.json").read_text())
        metadata = {k: training[k] for k in
                    ("seed", "train_rows", "val_rows", "epochs", "warmup_epochs", "batch_size", "lr")}
        metadata.update(ablation=mode, best_epoch=saved["best_epoch"])
    model.eval().requires_grad_(False)
    metadata.update(checkpoint=str(path), trainable_model_parameters=sum(p.numel() for p in model.parameters()),
                    decoder="tree_decoder.decode(width=1)", hidden_state="pre-finalnorm",
                    candidate_tokens=3, target_verify_nodes=4)
    return api.MethodAdapter(name, metadata, model=MLPAdapter(model, bf16=False), context=context)


def proposal_for(adapter, capacity, api, *, verify=False):
    from mlp_medusa_graph import ProposalForwardGraph
    if adapter.name in ("target", "opt_target", "ranker_only"):
        return None
    if adapter.name == "hydra_conditional":
        # Token validation, probability transforms and RNG remain in the saved decoder.
        return ProposalGroup([ProposalForwardGraph(module, "forward", method_kind="mlp")
                              for module in (adapter.model.p1, adapter.model.f1, adapter.model.p2)])
    if adapter.variant == "eagle":
        from eagle_graph import build_proposal
        return build_proposal(adapter, verify=verify, max_cache_tokens=capacity)
    if adapter.name == "medusa_lowrank":
        from mlp_medusa_graph import build_proposal
        return build_proposal(adapter, verify=verify)
    return ProposalForwardGraph(adapter.model.feature, "forward", method_kind="mlp",
                                adapter=adapter, verify=verify)


def make_arm(name, adapter, context, args, api, *, optimized=True):
    from dynamic_target_graph import DynamicTargetGraph
    from tree_runtime import TargetTreeRuntime
    from position_rescue_runtime import PositionRescueRuntime
    runtime = DynamicTargetGraph(context["target"], cache_capacity_buckets=(args.cache_capacity,))
    is_feature = name not in NATIVE
    tree = TargetTreeRuntime(context["target"], runtime, graph=optimized, max_width=1) if is_feature else None
    rescue = (PositionRescueRuntime(context["target"], runtime, graph=optimized)
              if name == "pf_rescue" else None)
    proposal = proposal_for(adapter, args.cache_capacity, api) if optimized else None
    return dict(name=name, adapter=adapter, runtime=runtime, tree=tree,
                rescue=rescue, proposal=proposal, optimized=optimized)


def run_arm(arm, context, prompt, budget, seed, api):
    from tree_decoder import decode as decode_w1
    from position_rescue_decoder import decode as decode_rescue
    from draft_a.pf_ranker_eval import cache_for
    proposal = arm["proposal"]
    with proposal.context() if proposal else nullcontext():
        if arm["tree"] is None:
            return api.decode_method(arm["adapter"], context["target"], prompt, budget, seed,
                                     runtime=arm["runtime"] if arm["optimized"] else None)
        runtime = arm["runtime"]
        factory = lambda core, capacity: runtime.cache_for(core, capacity, factory=cache_for)
        with runtime.attention_context(enabled=arm["optimized"]):
            extra = dict(tree_runtime=arm["tree"], width=1, cache_factory=factory)
            if arm["rescue"] is not None:
                extra["rescue_runtime"] = arm["rescue"]
            return (decode_rescue if arm["rescue"] is not None else decode_w1)(
                context["target"], arm["adapter"].model, context["ranker"], context["ranks"],
                prompt, budget, seed, **extra)


def arm_stats(arm):
    return {k: arm[k].stats() if arm[k] is not None else None
            for k in ("runtime", "tree", "rescue", "proposal")}


def close_arm(arm):
    if arm["proposal"] is not None:
        arm["proposal"].close()
    arm.clear()
    gc.collect()


def make_prompts(records, prefixes, sources, seeds, eos):
    selected = {}
    for record in records:
        source = str(record.get("scene") or record.get("source") or Path(record["path"]).parents[1].name)
        if len(record["tokens"]) >= max(prefixes) + 4:
            selected.setdefault(source, record)
    if len(selected) < sources:
        raise ValueError(f"Requested {sources} sources, found {len(selected)}")
    rows = []
    for index, source in enumerate(sorted(selected)[:sources]):
        record = selected[source]
        for prefix in prefixes:
            for seed in seeds:
                rows.append(dict(case_id=f"{index:02d}-{prefix}-{seed}", source=source,
                                 prefix=prefix, seed=seed,
                                 crop_path=str(record.get("path", "")),
                                 prompt=[eos] + [int(t) for t in record["tokens"][:prefix]]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/root/workspace/GaussianGPT"))
    parser.add_argument("--tree-source", type=Path, default=Path("/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04"))
    parser.add_argument("--candidate-source", type=Path, default=Path("/root/autodl-tmp/gaussiangpt_spec_next_20261022/source_run02"))
    parser.add_argument("--feature-root", type=Path, default=Path("/root/autodl-tmp/gaussiangpt_round2_20261017"))
    parser.add_argument("--ablation-root", type=Path, default=Path("/root/autodl-tmp/gaussiangpt_next_round_20261017/step3_ablation"))
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(MAIN))
    parser.add_argument("--training-seeds", default="20261018,20261019,20261020")
    parser.add_argument("--sources", type=int, default=10)
    parser.add_argument("--prefixes", default="64,256")
    parser.add_argument("--seeds", default="57101")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--cache-capacity", type=int, default=2048)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validated-run", type=Path,
                        help="Completed numerical/output run for these exact methods and source paths")
    parser.add_argument("--max-seconds", type=float, default=1500)
    args = parser.parse_args()
    training_seeds = [int(x) for x in args.training_seeds.split(",")]
    methods = expand_methods(args.methods.split(","), training_seeds)
    prefixes, seeds = ([int(x) for x in v.split(",")] for v in (args.prefixes, args.seeds))
    if (not prefixes or any(p < 2 or p % 2 for p in prefixes) or len(set(prefixes)) != len(prefixes)
            or not seeds or len(set(seeds)) != len(seeds) or args.sources < 1 or args.budget < 2
            or args.budget % 2 or args.cache_capacity < max(prefixes) + args.budget + 16):
        parser.error("Invalid source, prefix, seed, budget or cache size")
    if args.out.exists():
        parser.error("Use a new output directory")
    started = time.perf_counter()
    sys.path[:0] = [str(args.tree_source), str(args.candidate_source),
                   str(args.tree_source / "reference"), str(args.tree_source / "reference/runtime"),
                   str(args.project)]
    import os
    os.chdir(args.project)
    import torch
    from threadpoolctl import threadpool_limits
    import method_adapters as api
    from draft_a.corpus_v2 import validate
    from draft_a.pf_ranker_eval import validate_sequence
    from draft_a.learning_eval import _fp32_runtime
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU results cannot replace GPU timing")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True)
    progress = dict(status="loading", methods=methods, command=sys.argv, completed_runs=0)
    write_json(args.out / "progress.json", progress)
    arms = {}
    try:
        records = torch.load(args.crops, map_location="cpu", weights_only=True)
        for record in records:
            validate(record)
        prompts = make_prompts(records, prefixes, args.sources, seeds, 12096)
        context = api.build_context(args.project, device="cuda")
        adapters = {}
        for name in methods:
            if name in NATIVE:
                adapters[name] = api.load_method("target" if name == "opt_target" else name, context)
            elif name in ("pf_w1", "pf_rescue"):
                adapters[name] = api.load_method("pf_mlp", context)
            else:
                adapters[name] = load_feature(name, context, args, api)
        for prompt in prompts:
            validate_sequence(prompt["prompt"], [], context["ranks"])
        plan = dict(methods=methods, sources=sorted({p["source"] for p in prompts}),
                    prompt_groups=len(prompts), expected_runs=len(prompts) * len(methods),
                    prefixes=prefixes, seeds=seeds, budget=args.budget, cache_capacity=args.cache_capacity,
                    source_paths={k: str(getattr(args, k).resolve()) for k in
                                  ("tree_source", "candidate_source", "feature_root", "ablation_root", "project", "crops")},
                    metadata={name: a.metadata for name, a in adapters.items()},
                    target_precision="FP32; highest matmul; TF32 off", temperature=.9, top_p=.9,
                    gpu=torch.cuda.get_device_name(), torch=torch.__version__, python=sys.version,
                    timing="synchronized outer decode, including runtime and first graph creation, cache, prefill, all proposal work, validation, recovery and state maintenance; excluding weights and files",
                    meaning="native methods retain their saved sampling rules; Feature models share the exact W1 decoder and 4-node verification; Rescue adds one node",
                    loading_seconds=time.perf_counter() - started)
        write_json(args.out / "plan.json", plan)
        write_json(args.out / "prompts.json", prompts)
        with threadpool_limits(limits=1), torch.inference_mode(), _fp32_runtime(torch), torch.autocast("cuda", enabled=False):
            if args.validated_run:
                checked_plan = json.loads((args.validated_run / "plan.json").read_text())
                checked = json.loads((args.validated_run / "validation.json").read_text())
                checked_progress = json.loads((args.validated_run / "progress.json").read_text())
                if (checked.get("passed") is not True or checked_progress.get("status") != "complete"
                        or checked_progress.get("phase") != "validation"
                        or len(checked.get("rows", [])) != len(checked_plan["methods"]) * len(checked_plan["prefixes"])
                        or any(name not in checked_plan["methods"] for name in methods)
                        or any(checked_plan["source_paths"][key] != plan["source_paths"][key]
                               for key in ("tree_source", "candidate_source", "feature_root", "ablation_root", "project"))
                        or any(checked_plan["metadata"][name] != plan["metadata"][name] for name in methods)):
                    raise ValueError("Validation does not correspond to these models and source paths")
                write_json(args.out / "validation.json", dict(passed=True, reused_from=str(args.validated_run)))
            else:
                progress.update(status="validating")
                write_json(args.out / "progress.json", progress)
                validations = []
                # Different prefix lengths cover both short and longer cache positions.
                probe_cases = [next(p for p in prompts if p["prefix"] == n) for n in prefixes]
                for name in methods:
                    for prompt in probe_cases:
                        observations = []
                        for optimized in (False, True):
                            arm = make_arm(name, adapters[name], context, args, api, optimized=optimized)
                            stats, output = run_arm(arm, context, prompt["prompt"], min(32, args.budget), prompt["seed"], api)
                            torch.cuda.synchronize()
                            observations.append(dict(output=[int(t) for t in output], counts=count_values(stats),
                                                     runtime=arm_stats(arm)))
                            close_arm(arm)
                        passed = (observations[0]["output"] == observations[1]["output"] and
                                  observations[0]["counts"] == observations[1]["counts"])
                        validations.append(dict(method=name, case_id=prompt["case_id"], passed=passed,
                                                eager=observations[0], optimized=observations[1]))
                        write_json(args.out / "validation.json", dict(
                            passed=(all(v["passed"] for v in validations) and
                                    len(validations) == len(methods) * len(probe_cases)), rows=validations))
                        if not passed:
                            raise ArithmeticError(f"Output or action differs: {name}, {prompt['case_id']}")
                        print(json.dumps(dict(event="validated", method=name, prefix=prompt["prefix"])), flush=True)
            if args.validate_only:
                progress.update(status="complete", phase="validation", elapsed_seconds=time.perf_counter() - started)
                write_json(args.out / "progress.json", progress)
                return
            # These are separate short requests; all warmup runtimes are discarded.
            for name in methods:
                arm = make_arm(name, adapters[name], context, args, api)
                run_arm(arm, context, prompts[0]["prompt"], 16, seeds[0], api)
                close_arm(arm)
            rows, group_seconds = [], []
            progress.update(status="running", expected_runs=plan["expected_runs"])
            write_json(args.out / "progress.json", progress)
            stopped = False
            for index, prompt in enumerate(prompts):
                reserve = max(45., max(group_seconds[-4:], default=0.) * 1.5)
                if args.max_seconds > 0 and time.perf_counter() - started + reserve >= args.max_seconds:
                    stopped = True
                    break
                group_start = time.perf_counter()
                order = methods[index % len(methods):] + methods[:index % len(methods)]
                if index // len(methods) % 2:
                    order = list(reversed(order))
                for name in order:
                    before = (arms[name]["runtime"].stats() if name in arms
                              else dict(graph_count=0, calls={}))
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    if name not in arms:
                        arms[name] = make_arm(name, adapters[name], context, args, api)
                    arm = arms[name]
                    stats, output = run_arm(arm, context, prompt["prompt"], args.budget, prompt["seed"], api)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - begin
                    output = [int(t) for t in output]
                    validate_sequence(prompt["prompt"], output, context["ranks"])
                    if len(output) != int(stats["generated_tokens"]):
                        raise ValueError(f"Generated-token count differs for {name}")
                    after = arm["runtime"].stats()
                    row = {k: prompt[k] for k in ("case_id", "source", "prefix", "seed", "crop_path")}
                    row.update(method=name, generated_tokens=len(output), seconds=elapsed,
                               output_tokens=output, counts=count_values(stats), eos=bool(output and output[-1] == context["eos"]),
                               target_graphs_before=before["graph_count"], target_graphs_after=after["graph_count"],
                               target_runtime_calls={k: v - before.get("calls", {}).get(k, 0)
                                                     for k, v in after.get("calls", {}).items()})
                    with (args.out / "records.jsonl").open("a") as handle:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
                    progress.update(completed_runs=len(rows), elapsed_seconds=time.perf_counter() - started,
                                    last_method=name, last_case=prompt["case_id"])
                    write_json(args.out / "progress.json", progress)
                    print(json.dumps(dict(event="decode", done=len(rows), method=name,
                                          case=prompt["case_id"], seconds=elapsed)), flush=True)
                group_seconds.append(time.perf_counter() - group_start)
            if rows:
                references = [m for m in ("opt_target", "ranker_only", "pf_w1", "pf_rescue",
                                          "ablate_full", "b0:s20261018", "b0:s20261019", "b0:s20261020") if m in methods]
                summary = paired_summary(rows, methods, references)
                summary.update(status="partial" if stopped else "complete", expected_runs=plan["expected_runs"],
                               runtime_summary={name: arm_stats(arm) for name, arm in arms.items()})
                write_json(args.out / "summary.json", summary)
            progress.update(status="partial" if stopped else "complete", completed_runs=len(rows),
                            elapsed_seconds=time.perf_counter() - started)
            write_json(args.out / "progress.json", progress)
    except Exception as error:
        progress.update(status="failed", error=f"{type(error).__name__}: {error}",
                        elapsed_seconds=time.perf_counter() - started)
        write_json(args.out / "progress.json", progress)
        raise
    finally:
        for arm in arms.values():
            close_arm(arm)


if __name__ == "__main__":
    main()
