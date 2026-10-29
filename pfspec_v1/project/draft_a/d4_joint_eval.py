"""Select D4 on validation inputs, then compare real-prefix decoding."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import traceback

from draft_a.d4_joint_cost import D8, SOURCE, ROOT, load_model, real_inputs, measure
from draft_a.d4_joint_train import ENDS
from draft_a.feature_target_v2 import HEAD, RANKER, save, records_for
from draft_a.feature_d8_gamma_eval import total

MLP = Path("draft_a/runs/feature_target_continue_20261014/selected.pt")


def summary(rows, names):
    values = {name: total([r for r in rows if r["variant"] == name]) for name in names}
    for name in names[1:]:
        values[name]["speedup"] = values[name]["tps"]/values["target_only"]["tps"]
        if "mlp" in values:
            values[name]["relative_to_mlp"] = values[name]["tps"]/values["mlp"]["tps"]
        if "d8" in values:
            values[name]["relative_to_d8"] = values[name]["tps"]/values["d8"]["tps"]
    return values


def timed_decode(call, **metadata):
    import torch
    torch.cuda.synchronize()
    started = time.perf_counter()
    row = call()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    row.update(decoder_seconds=row["seconds"], seconds=elapsed,
               tokens_per_second=row["generated_tokens"]/elapsed, **metadata)
    return row


def select(root, target, smoke):
    import torch
    from draft_a.feature_acceptance_eval import decode
    from draft_a.pf_ranker_eval import decode as ordinary
    destination = root / "selection"
    if (destination / "summary.json").exists():
        old = json.loads((destination / "summary.json").read_text())
        if old["status"] == "complete":
            return old
    candidates = [1, 2, 3, 4] if smoke else ENDS
    models = {f"step{s}": load_model(root / f"step_{s:04d}.pt") for s in candidates}
    records = sorted([r for r in records_for("val") if len(r["tokens"]) >= 256], key=lambda r: r["name"])
    records = records[:1 if smoke else 12]
    if len(records) != (1 if smoke else 12):
        raise ValueError("Not enough eligible validation inputs")
    prefixes, budget = ([64], 32) if smoke else ([64, 256], 256)
    names = ["target_only", *models]
    ranks = target.transformer.sparse_position_rank_map.cpu().tolist()
    parameters = dict(files=[r["name"] for r in records], prefixes=prefixes, seed=24710, budget=budget,
                      gamma=4, models={name: str(root/f"step_{step:04d}.pt") for name, step in zip(models, candidates)},
                      selection="aggregate validation throughput, all candidates retained", precision="FP32")
    save(destination / "plan.json", parameters)
    prompt = records[0]["seq"][:prefixes[0]+1].tolist()
    ordinary(target.transformer, None, None, ranks, prompt, 32, 0, 24710)
    for model in models.values():
        decode(target.transformer, model.transformer, prompt, 32, 4, 24710)
    result = dict(status="running", parameters=parameters, rows=[])
    expected = len(records)*len(prefixes)*len(names)
    for ci, record in enumerate(records):
        for pi, prefix in enumerate(prefixes):
            prompt = record["seq"][:prefix+1].tolist()
            shift = (ci+pi)%len(names)
            for name in names[shift:]+names[:shift]:
                if name == "target_only":
                    call = lambda: ordinary(target.transformer, None, None, ranks, prompt, budget, 0, 24710)
                else:
                    call = lambda: decode(target.transformer, models[name].transformer, prompt, budget, 4, 24710)
                row = timed_decode(call, variant=name, name=record["name"], prefix=prefix, seed_base=24710)
                result["rows"].append(row)
                save(destination / "online.json", result)
                save(root / "evaluation_progress.json", dict(status="running", phase="selection",
                     completed=len(result["rows"]), expected=expected))
                print(json.dumps(dict(event="selection", run=len(result["rows"]), expected=expected,
                                      variant=name, tps=row["tokens_per_second"])), flush=True)
    result["status"] = "complete"
    save(destination / "online.json", result)
    values = summary(result["rows"], names)
    chosen = max(models, key=lambda name: values[name]["tps"])
    step = int(chosen.removeprefix("step"))
    report = dict(status="complete", selected=chosen, step=step, overall=values,
                  by_prefix={str(p): summary([r for r in result["rows"] if r["prefix"] == p], names) for p in prefixes},
                  limitation="one sampling seed for validation selection; minor timing differences may reorder candidates")
    shutil.copy2(root / f"step_{step:04d}.pt", root / "selected.pt")
    save(root / "selected.json", dict(step=step, checkpoint=str(root / "selected.pt"), validation=values[chosen]))
    save(destination / "summary.json", report)
    print(json.dumps(dict(event="selected", step=step, overall=values)), flush=True)
    del models
    gc.collect()
    torch.cuda.empty_cache()
    return report


def historical(root, target, models, smoke):
    import torch
    from draft_a import learning_eval as ev, train_v2 as train, eval_overlap as overlap
    from draft_a.feature_acceptance_eval import _test_summary
    records = records_for("test", 1 if smoke else None)
    train_plan = json.loads((root / "training_plan.json").read_text())
    if {r["name"] for r in records} & (set(train_plan["train_files"]) | set(train_plan["val_files"])):
        raise ValueError("Historical comparison inputs overlap training or validation")
    cached_root = D8.parent.parent / "teacher_cache_test"
    output = dict(scope="historical rollout comparison, fixed cached P/F slots", models={}, records=len(records),
                  previously_viewed=True, caches_reused=0, caches_created=0)
    rows = {name: [] for name in models}
    for record in records:
        seq = record["seq"]
        meta = ev._meta(record, seq, overlap)
        path = cached_root / record["name"]
        if path.exists():
            cache = torch.load(path, map_location="cpu", weights_only=True)
            origin = cache.get("source_root")
            if (not isinstance(origin, str) or Path(origin).resolve() != SOURCE.resolve()
                    or not torch.equal(cache["tokens"], record["tokens"])):
                raise ValueError("Existing historical cache source differs")
            output["caches_reused"] += 1
        else:
            cache = ev._teacher_labels(target, record, seq, meta, overlap, torch)
            cache.update(source_root=str(SOURCE.resolve()), tokens=record["tokens"].clone().cpu())
            ev._save_labels(root / "historical_cache" / record["name"], cache, torch)
            output["caches_created"] += 1
        ev._valid_cache(cache, meta, torch)
        for name, model in models.items():
            rows[name].extend(ev._student_sequence(model, seq, cache, overlap, torch))
    for name in models:
        output["models"][name] = dict(summary=_test_summary(rows[name], ev), per_sequence=rows[name])
    save(root / "historical_overlap.json", output)
    print(json.dumps(dict(event="historical", models={n: v["summary"] for n, v in output["models"].items()})), flush=True)


def fixed_real(root, target, models, mlp, records):
    import torch
    from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock
    from draft_a.feature_target_v2 import evaluate
    from draft_a.feature_adapt_continue import policy_probs
    rows = []
    capture = CaptureLastBlock(target.transformer)
    try:
        for record in records:
            seq = record["seq"]
            depth = torch.arange(1, len(record["tokens"])//2)
            indices = (2*depth+1).cuda()
            logits = target(input_ids=seq[:-1].unsqueeze(0).cuda())[0]
            states = capture.output[0]
            data = dict(name=record["name"], depth=depth, position=seq[2*depth+1],
                        sampled_feature=seq[2*depth+2]-8000, previous_feature=seq[2*depth]-8000,
                        previous_state=states.index_select(0, indices-2).float().cpu(),
                        target=logits.index_select(0, indices)[:, 8000:12096].float().softmax(-1).cpu())
            capture.output = None
            del logits, states
            result = dict(name=record["name"], rows=len(depth), mlp=evaluate(mlp, [data], 256)["pooled_overlap"])
            p = policy_probs(data["target"].cuda().log())
            for name, model in models.items():
                logits = model(input_ids=seq[:-1].unsqueeze(0).cuda())[0]
                q = policy_probs(logits.index_select(0, indices)[:, 8000:12096].float())
                result[name] = float(torch.minimum(p, q).sum(-1).mean())
                del logits, q
            rows.append(result)
            del data, p, indices
    finally:
        capture.close()
    count = sum(r["rows"] for r in rows)
    save(root / "fixed_real_overlap.json", dict(rows=count, crops=rows, unique_scenes=len({r["scene"] for r in records}),
         scope="identical full real Feature prefixes, expected single-step acceptance", pooled={
             name: sum(r[name]*r["rows"] for r in rows)/count for name in [*models, "mlp"]}))


class Meter:
    def __init__(self):
        self.rows, self.events = {}, []

    def run(self, name, fn, *args, **kwargs):
        import torch
        entry = self.rows.setdefault(name, dict(calls=0, host_seconds=0.0, cuda_seconds=0.0))
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        start = time.perf_counter()
        value = fn(*args, **kwargs)
        entry["host_seconds"] += time.perf_counter()-start
        entry["calls"] += 1
        b.record()
        self.events.append((name, a, b))
        return value

    def result(self):
        import torch
        torch.cuda.synchronize()
        for name, a, b in self.events:
            self.rows[name]["cuda_seconds"] += a.elapsed_time(b)/1000
        return self.rows


class TimedCore:
    def __init__(self, core, label, meter):
        self.core, self.label, self.meter, self.calls = core, label, meter, 0

    def __getattr__(self, name):
        return getattr(self.core, name)

    def __call__(self, tokens, **kwargs):
        name = self.label + ("_prefill" if self.calls == 0 and tokens.shape[1] > 1 else "_forward")
        self.calls += 1
        return self.meter.run(name, self.core, tokens, **kwargs)


def joint_profile(target, draft, prompt, budget, gamma, seed):
    from draft_a import feature_acceptance_eval as decoder
    meter = Meter()
    original = {name: getattr(decoder, name) for name in ["policy", "draw", "residual", "cache_for"]}
    for name, function in original.items():
        def wrap(*args, _name=name, _function=function, **kwargs):
            return meter.run(_name, _function, *args, **kwargs)
        setattr(decoder, name, wrap)
    try:
        row = decoder.decode(TimedCore(target, "target", meter), TimedCore(draft, "draft", meter),
                             prompt, budget, gamma, seed)
        row["stage_profile"] = meter.result()
        return row
    finally:
        for name, function in original.items():
            setattr(decoder, name, function)


def profiles(root, target, d4, d8, adapter, ranker, ranks, record, smoke):
    from draft_a.feature_acceptance_eval import decode
    from draft_a.pf_feature_unified import decode as feature_decode
    rows = []
    for prefix in ([64] if smoke else [64, 1024]):
        prompt = record["seq"][:prefix+1].tolist()
        for name, model, gamma in [("d4", d4, g) for g in ([2] if smoke else [2, 4, 6])] + [("d8", d8, 2)]:
            plain = decode(target.transformer, model.transformer, prompt, 32 if smoke else 128, gamma, 24720)
            timed = joint_profile(target.transformer, model.transformer, prompt, 32 if smoke else 128, gamma, 24720)
            keys = ["accepted", "verified", "proposed", "rounds", "generated_tokens", "eos", "target_calls", "draft_calls"]
            if any(plain[k] != timed[k] for k in keys):
                raise ValueError("Detailed timing changed joint decoding counts")
            rows.append(dict(model=name, prefix=prefix, gamma=gamma, status="passed", **timed))
        plain, output = feature_decode(target.transformer, adapter, ranker, ranks, prompt, 32 if smoke else 128, 24720)
        timed, output2 = feature_decode(target.transformer, adapter, ranker, ranks, prompt, 32 if smoke else 128, 24720, profile=True)
        if output != output2 or any(plain[k] != timed[k] for k in ["accepted", "verified", "proposed", "rounds"]):
            raise ValueError("Detailed timing changed Feature decoding")
        rows.append(dict(model="mlp", prefix=prefix, status="passed", **timed))
    save(root / "stage_profile.json", dict(rows=rows, timing="host and CUDA times are separate, not additive; instrumented runs excluded from main throughput"))


def run(args):
    import torch
    from joblib import load
    from threadpoolctl import threadpool_limits
    from bench_fast_hgb import FastHistRanker
    from bench_gaussiangpt_pf_block2_cow import load_feature_model
    from draft_a.corpus_v2 import load_target
    from draft_a.learning_eval import _fp32_runtime
    from draft_a.feature_acceptance_eval import decode, verify_blocks
    from draft_a.pf_feature_unified import MLPAdapter, decode as feature_decode
    from draft_a.pf_ranker_eval import decode as ordinary
    state = json.loads((args.root / "training_progress.json").read_text())
    if state["status"] != "complete" or state["step"] != (4 if args.smoke else 8600):
        raise ValueError("Complete the agreed training before evaluation")
    torch.set_num_threads(6)
    with _fp32_runtime(torch), threadpool_limits(limits=1, user_api="openmp"):
        target = load_target().float().eval().requires_grad_(False)
        chosen = select(args.root, target, args.smoke)
        d4, d8 = load_model(args.root / "selected.pt"), load_model(D8)
        historical(args.root, target, {"d4": d4, "d8": d8}, args.smoke)
        records = real_inputs()[:1 if args.smoke else 4]
        numerical = {name: verify_blocks(model.transformer, records[0]["seq"]) for name, model in [("target", target), ("d4", d4), ("d8", d8)]}
        mlp = load_feature_model(MLP, HEAD, target.transformer, torch.device("cuda"), 8000, 12096)[0]
        ranker = FastHistRanker(load(RANKER))
        ranks = target.transformer.sparse_position_rank_map.cpu().tolist()
        adapter = MLPAdapter(mlp)
        fixed_real(args.root, target, {"d4": d4, "d8": d8}, mlp, records)
        prefixes, seeds, budget = ([64], [34701], 32) if args.smoke else ([64, 256, 1024], [34701, 34702, 34703], 512)
        validation = []
        for prefix in prefixes:
            prompt = records[0]["seq"][:prefix+1].tolist()
            ordinary(target.transformer, None, None, ranks, prompt, 32, 0, 34700)
            decode(target.transformer, d8.transformer, prompt, 32, 2, 34700)
            for gamma in ([2] if args.smoke else [2, 4, 6]):
                a = decode(target.transformer, d4.transformer, prompt, 32, gamma, 34700)
                b = decode(target.transformer, d4.transformer, prompt, 32, gamma, 34700)
                if any(a[k] != b[k] for k in ["accepted", "verified", "proposed", "rounds", "generated_tokens"]):
                    raise ValueError("Selected D4 repeated decoding differs")
                same = decode(target.transformer, target.transformer, prompt, 32, gamma, 34700)
                if same["accepted"] != same["verified"]:
                    raise ValueError("Target self proposals differ")
                validation.append(dict(prefix=prefix, gamma=gamma, status="passed", target_self_accepted=same["accepted"],
                                       target_self_verified=same["verified"]))
            plain, suffix = feature_decode(target.transformer, adapter, ranker, ranks, prompt, 32, 34700)
            instrumented, suffix2 = feature_decode(target.transformer, adapter, ranker, ranks, prompt, 32, 34700, profile=True)
            if suffix != suffix2 or any(plain[k] != instrumented[k] for k in ["accepted", "verified", "proposed", "rounds"]):
                raise ValueError("Feature comparison output or counts differ under detailed timing")
            validation.append(dict(prefix=prefix, variant="mlp", status="passed"))
        save(args.root / "final_validation.json", dict(models=numerical, short_runs=validation))
        names = ["target_only", "d8", "mlp", "d4_gamma2", "d4_gamma4", "d4_gamma6"]
        expected = len(records)*len(prefixes)*len(seeds)*len(names)
        parameters = dict(selected_step=chosen["step"], d4=str(args.root/"selected.pt"), d8=str(D8), mlp=str(MLP),
                          target="checkpoints/gpt_vfront.ckpt", prefixes=prefixes, seeds=seeds, budget=budget,
                          gammas=[2, 4, 6], unique_scenes=len({r["scene"] for r in records}), precision="FP32",
                          temperature=.9, top_p=.9, scope="real prefixes with sampled continuation",
                          timing="common outer wall time includes validation, cache allocation/release, prefill, proposals, verification, rejection and sampling; excludes loading and data preparation",
                          crops=[{k: r[k] for k in ["name", "scene", "crop_seed", "path"]} for r in records])
        result = dict(status="running", parameters=parameters, rows=[])
        save(args.root / "real_plan.json", parameters)
        for ci, record in enumerate(records):
            for pi, prefix in enumerate(prefixes):
                prompt = record["seq"][:prefix+1].tolist()
                for repeat, base in enumerate(seeds):
                    seed = base+record["crop_seed"]*1009+prefix*17
                    shift = (ci+pi+repeat)%len(names)
                    for name in names[shift:]+names[:shift]:
                        def call():
                            if name == "target_only":
                                return ordinary(target.transformer, None, None, ranks, prompt, budget, 0, seed)
                            if name == "d8":
                                return decode(target.transformer, d8.transformer, prompt, budget, 2, seed)
                            if name == "mlp":
                                return feature_decode(target.transformer, adapter, ranker, ranks, prompt, budget, seed)[0]
                            return decode(target.transformer, d4.transformer, prompt, budget, int(name[-1]), seed)
                        row = timed_decode(call, variant=name, name=record["name"], scene=record["scene"], prefix=prefix, seed_base=base)
                        result["rows"].append(row)
                        save(args.root / "real_online.json", result)
                        save(args.root / "evaluation_progress.json", dict(status="running", phase="real_online",
                             completed=len(result["rows"]), expected=expected))
                        print(json.dumps(dict(event="real_online", run=len(result["rows"]), expected=expected,
                                              variant=name, P=row["position_acceptance_verified"], F=row["feature_acceptance_verified"],
                                              tps=row["tokens_per_second"])), flush=True)
        result["status"] = "complete"
        save(args.root / "real_online.json", result)
        report = dict(overall=summary(result["rows"], names),
                      by_prefix={str(p): summary([r for r in result["rows"] if r["prefix"] == p], names) for p in prefixes},
                      by_seed={str(s): summary([r for r in result["rows"] if r["seed_base"] == s], names) for s in seeds},
                      limitations="one real scene, FP32 token decoding; no scene generalization or full 3D pipeline timing")
        save(args.root / "real_summary.json", report)
        print(json.dumps(dict(event="real_summary", overall=report["overall"])), flush=True)
        profiles(args.root, target, d4, d8, adapter, ranker, ranks, records[0], args.smoke)
        measure({"d4": d4, "d8": d8}, args.root / "trained_cost", args.smoke)
        save(args.root / "evaluation_progress.json", dict(status="complete", phase="complete", completed=expected, expected=expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        save(args.root / "evaluation_failure.json", dict(error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
