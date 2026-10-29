#!/usr/bin/env python3
"""Measure ready-prompt to six saved views with the measured W1 implementation."""

import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import sys
import time


def jsonable(value):
    if isinstance(value, dict):
        return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "detach"):
        return jsonable(value.detach().cpu().tolist())
    return value


def runtime_delta(before, after):
    before, after = before or {}, after or {}
    return dict(calls={k:v-before.get("calls",{}).get(k,0)
        for k,v in after.get("calls",{}).items()},
        new_graph_events=after.get("graph_events",[])[len(before.get("graph_events",[])):])


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixed-prompts", type=Path, required=True)
    parser.add_argument("--natural-prompts", type=Path)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--methods", default="target,opt_target,ranker_only,w1,rescue")
    parser.add_argument("--fixed-only", action="store_true")
    parser.add_argument("--cache-capacity", type=int, default=1024)
    parser.add_argument("--fixed-budget", type=int, default=512)
    parser.add_argument("--natural-budget", type=int, default=16000)
    parser.add_argument("--natural-capacity", type=int, default=16384)
    parser.add_argument("--max-wall-seconds", type=float, default=0)
    parser.add_argument("--warm-tokens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    methods = args.methods.split(",")
    if (not methods or len(set(methods)) != len(methods)
            or set(methods)-{"target","opt_target","ranker_only","w1","w2","rescue"}
            or ("rescue" in methods and args.candidate_source is None)
            or (not args.fixed_only and args.natural_prompts is None)
            or args.cache_capacity < max(776, args.fixed_budget + 257) or args.max_wall_seconds < 0
            or args.fixed_budget < 2 or args.natural_budget < 2
            or args.natural_capacity < args.natural_budget + 257):
        parser.error("Invalid methods, inputs, capacity or time limit")
    project, source, out = args.project.resolve(), args.source.resolve(), args.out.resolve()
    if out.exists():
        parser.error("Output must name a new directory")
    out.mkdir(parents=True)
    os.chdir(project)
    sys.path[:0] = [str(source), str(source / "reference"),
                    str(source / "reference/runtime"), str(project)]
    dependencies = project.parent / "deps"
    if dependencies.is_dir():
        sys.path.insert(0, str(dependencies))
    if args.candidate_source:
        sys.path.insert(0, str(args.candidate_source.resolve()))
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from method_adapters import build_context, load_method, decode_method
    from dynamic_target_graph import DynamicTargetGraph
    from tree_runtime import TargetTreeRuntime
    from tree_decoder import decode as decode_tree
    from proposal_adapters import build_proposal
    from draft_a.pf_ranker_eval import cache_for, validate_sequence
    from draft_a.learning_eval import _fp32_runtime
    from draft_a.pf_multiscene.full_3d_eval import (
        camera_views, gaussian_tensors, render_views, save_views,
    )
    from model.gaussian_vqvae import GaussianVQVAE
    if "rescue" in methods:
        from position_rescue_runtime import PositionRescueRuntime
        from position_rescue_decoder import decode as decode_rescue

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cohorts = [dict(name="fixed512", path=args.fixed_prompts, budget=args.fixed_budget, capacity=args.cache_capacity)]
    if not args.fixed_only:
        cohorts.append(dict(name="natural_eos", path=args.natural_prompts, budget=args.natural_budget, capacity=args.natural_capacity))
    for cohort in cohorts:
        cohort["prompts"] = json.loads(cohort.pop("path").read_text())["contexts"]
    cohorts = [cohort for cohort in cohorts if cohort["prompts"]]
    if not cohorts:
        parser.error("At least one prompt is required")
    expected = sum(len(methods) * len(c["prompts"]) for c in cohorts)
    progress = dict(status="loading", completed=0, expected=expected)
    write_json(out / "progress.json", progress)
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1), torch.inference_mode(), _fp32_runtime(torch), torch.autocast("cuda", enabled=False):
            loading = time.perf_counter()
            context = build_context(project, device="cuda")
            target = context["target"]
            adapters = {m:load_method("target" if m in {"target", "opt_target"} else
                                      "ranker_only" if m == "ranker_only" else "pf_mlp", context)
                        for m in methods}
            vqvae = GaussianVQVAE.load_from_checkpoint(
                str(project / "checkpoints/vqvae_vfront.ckpt"), map_location="cpu"
            ).cuda().float().eval().requires_grad_(False)
            torch.cuda.synchronize()
            loading_seconds = time.perf_counter() - loading
            cameras = camera_views(20 * vqvae.autoencoder.stride * vqvae.autoencoder.default_grid_size)
            write_json(out / "cameras.json", cameras)
            plan = dict(project=str(project), source=str(source), cohorts=cohorts,
                        expected=expected, methods=methods,
                        candidate_source=str(args.candidate_source.resolve()) if args.candidate_source else None,
                        max_wall_seconds=args.max_wall_seconds,
                        device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                        python_version=sys.version, fp32=True, tf32=False,
                        temperature=.9, top_p=.9, model_loading_seconds=loading_seconds,
                        target_checkpoint=str(project / "checkpoints/gpt_vfront.ckpt"),
                        vq_checkpoint=str(project / "checkpoints/vqvae_vfront.ckpt"),
                        proposal_metadata={m:a.metadata for m,a in adapters.items()},
                        total_scope="Ready prompt to saved tokens, compressed Gaussian NPZ and six 384x384 PNGs. Synchronized CPU wall time. Per-request directory creation, validation, conversion, graph creation and KV allocation included. Model loading, warmup, result JSON and input acquisition excluded.",
                        runtime_scope="Independent persistent runtimes per method and cohort; warmup runtimes discarded. First measured request includes fresh runtimes and lazy graphs.",
                        natural_scope="Same real prefix and seed, separate sampling streams. Different suffix lengths and Gaussian counts are retained; request time measures actual complete outputs.")
            write_json(out / "plan.json", plan)

            def make_runtime(method, capacity):
                if method == "target":
                    return dict(runtime=None, tree=None, proposal=None, rescue=None)
                runtime = DynamicTargetGraph(target, cache_capacity_buckets=(capacity,))
                if method in {"opt_target", "ranker_only"}:
                    return dict(runtime=runtime, tree=None, proposal=None, rescue=None)
                return dict(runtime=runtime,
                    tree=TargetTreeRuntime(target, runtime, graph=True, max_width=4),
                    proposal=build_proposal(adapters[method], verify=False),
                    rescue=PositionRescueRuntime(target,runtime,graph=True) if method == "rescue" else None)

            def generate(method, prompt, budget, seed, arm):
                runtime,tree,proposal = [arm[k] for k in ("runtime","tree","proposal")]
                if method in {"target", "opt_target", "ranker_only"}:
                    return decode_method(adapters[method], target, prompt, budget, seed, runtime=runtime)
                factory = lambda core, length: runtime.cache_for(core, length, factory=cache_for)
                with proposal.context(), runtime.attention_context():
                    if method == "rescue":
                        return decode_rescue(target,adapters[method].model,context["ranker"],context["ranks"],
                            prompt,budget,seed,tree_runtime=tree,rescue_runtime=arm["rescue"],width=1,cache_factory=factory)
                    return decode_tree(target, adapters[method].model, context["ranker"], context["ranks"],
                                       prompt, budget, seed, tree_runtime=tree, width=(2 if method == "w2" else 1), cache_factory=factory)

            progress.update(status="warming")
            write_json(out / "progress.json", progress)
            warm_started = time.perf_counter()
            body = torch.load(args.warm_tokens, map_location="cpu", weights_only=True)["tokens"].tolist()
            tensors, quality = gaussian_tensors(vqvae, body)
            rgb, alpha = render_views(tensors, cameras)
            warm_dir = out / "warmup"
            warm_dir.mkdir()
            np.savez_compressed(warm_dir / "gaussians.npz", **{k:v.cpu().numpy() for k,v in tensors.items()})
            save_views(rgb, warm_dir)
            del tensors, rgb, alpha
            warmups = []
            for cohort in cohorts:
                prompt = cohort["prompts"][0]
                for method in methods:
                    arm = make_runtime(method, cohort["capacity"])
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    stats, suffix = generate(method, prompt["prompt"], 16, prompt["seed"], arm)
                    torch.cuda.synchronize()
                    warmups.append(dict(cohort=cohort["name"], method=method, seconds=time.perf_counter()-begin,
                                        generated_tokens=len(suffix)))
                    if arm["proposal"]:
                        arm["proposal"].close()
                    del arm
                    gc.collect()
            torch.cuda.synchronize()
            write_json(out / "warmup.json", dict(rows=warmups, total_seconds=time.perf_counter()-warm_started))
            rows = []
            stopped = False
            durations = []
            for cohort in cohorts:
                arms = {}
                for index, prompt in enumerate(cohort["prompts"]):
                    reserve = max(40., max(durations[-6:],default=0)*1.5)
                    if args.max_wall_seconds and time.perf_counter()-started+reserve >= args.max_wall_seconds:
                        stopped = True
                        break
                    group_begin = time.perf_counter()
                    shift = prompt.get("order_index", index) % len(methods)
                    for method in methods[shift:]+methods[:shift]:
                        progress.update(status="running", current=dict(cohort=cohort["name"], pair_id=prompt["pair_id"], method=method))
                        write_json(out / "progress.json", progress)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        begin = time.perf_counter()
                        sample_dir = out / "samples" / cohort["name"] / prompt["pair_id"] / method
                        sample_dir.mkdir(parents=True)
                        created = method != "target" and method not in arms
                        before = {k:arms[method][k].stats() if arms.get(method,{}).get(k) else None
                                  for k in ("runtime","tree","proposal","rescue")}
                        if method not in arms:
                            arms[method] = make_runtime(method,cohort["capacity"])
                        arm = arms[method]
                        stats, suffix = generate(method, prompt["prompt"], cohort["budget"], prompt["seed"], arm)
                        torch.cuda.synchronize()
                        generation_end = time.perf_counter()
                        suffix = [int(t) for t in suffix]
                        validate_sequence(prompt["prompt"], suffix, context["ranks"])
                        eos = bool(suffix and suffix[-1] == context["eos"])
                        body = prompt["prompt"][1:] + (suffix[:-1] if eos else suffix)
                        if not body or len(body) % 2:
                            raise ValueError("Output must contain nonempty complete P/F pairs")
                        torch.save(dict(tokens=torch.tensor(body), suffix=torch.tensor(suffix), has_eos=eos,
                                        prefix_tokens=prompt["prefix"]), sample_dir / "tokens.pt")
                        pack_end = time.perf_counter()
                        tensors, quality = gaussian_tensors(vqvae, body)
                        torch.cuda.synchronize()
                        vq_end = time.perf_counter()
                        np.savez_compressed(sample_dir / "gaussians.npz", **{k:v.cpu().numpy() for k,v in tensors.items()})
                        export_end = time.perf_counter()
                        rgb, alpha = render_views(tensors, cameras)
                        torch.cuda.synchronize()
                        render_end = time.perf_counter()
                        save_views(rgb, sample_dir)
                        finish = time.perf_counter()
                        if tuple(rgb.shape) != (6, 384, 384, 3) or tuple(alpha.shape) != (6, 384, 384, 1):
                            raise ValueError("Unexpected rendered dimensions")
                        paths = [sample_dir / f"view_{i:02d}.png" for i in range(6)]
                        if not all(p.is_file() and p.stat().st_size > 0 for p in paths):
                            raise ValueError("Saved views must be nonempty")
                        coverage = float((alpha > .01).float().mean().item())
                        if coverage <= 0:
                            raise ValueError("Empty rendered alpha coverage")
                        stages = dict(generation=generation_end-begin, packing=pack_end-generation_end,
                                      vq=vq_end-pack_end, gaussian_export=export_end-vq_end,
                                      render=render_end-export_end, png=finish-render_end)
                        row = dict(cohort=cohort["name"], pair_id=prompt["pair_id"], source=prompt["source"],
                                   seed=prompt["seed"], prefix=prompt["prefix"], method=method,
                                   output_tokens=suffix, generated_tokens=len(suffix), body_tokens=len(body),
                                   eos=eos, capped=not eos, runtime_created=created,
                                   stats=jsonable(stats),
                                   counters={k:jsonable(v) for k,v in stats.items()
                                       if isinstance(v,(int,bool,list,dict)) and not any(s in k for s in ("rate","seconds","time","runtime"))},
                                   runtime_deltas={k:runtime_delta(before[k],v.stats())
                                       for k,v in arm.items() if v is not None},
                                   stages=stages, total_seconds=finish-begin,
                                   compute_seconds=sum(stages[k] for k in ("generation", "packing", "vq", "render")),
                                   quality=quality, alpha_coverage=coverage,
                                   peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                                   finite_gaussians=True, finite_rgb_alpha=True, sequence_validated=True,
                                   saved_views=6, output_dir=str(sample_dir))
                        write_json(sample_dir / "metrics.json", row)
                        with (out / "records.jsonl").open("a") as handle:
                            handle.write(json.dumps(row, allow_nan=False) + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        rows.append(row)
                        progress.update(completed=len(rows), elapsed_seconds=time.perf_counter()-started)
                        write_json(out / "progress.json", progress)
                        print(json.dumps({k:row[k] for k in ("cohort", "pair_id", "method", "generated_tokens", "eos", "total_seconds")}
                                         | dict(stages=stages, completed=len(rows))), flush=True)
                        del tensors, rgb, alpha
                    durations.append(time.perf_counter()-group_begin)
                write_json(out / (cohort["name"] + "_runtime.json"), dict(
                    methods={m:{k:v.stats() for k,v in arm.items() if v is not None}
                             for m,arm in arms.items()}))
                for arm in arms.values():
                    if arm["proposal"]:
                        arm["proposal"].close()
                arms.clear()
                gc.collect()
                if stopped:
                    break
            progress.update(status="partial" if stopped else "complete", completed=len(rows),
                elapsed_seconds=time.perf_counter()-started,time_limit_reached=stopped)
            write_json(out / "progress.json", progress)
    except Exception as error:
        progress.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter()-started)
        write_json(out / "progress.json", progress)
        raise


if __name__ == "__main__":
    main()
