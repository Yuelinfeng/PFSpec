"""Train the existing Feature state model directly against the original Target."""

import argparse
import gc
import json
import math
from pathlib import Path
import random
import time


ROOT = Path("draft_a/runs/feature_target_full_v2_20261014")
SOURCE = Path("draft_a/runs/rollout500_20261013")
INITIAL = Path("profiling/feature_draft/state_distill_mixed_overlap/feature_state_draft_best.pt")
HEAD = Path("profiling/feature_draft/state_generation_512/states/frozen_feature_head.pt")
RAW_LABELS = Path("draft_a/runs/pf_feature_distill_20261014/feature_labels")
RANKER = Path("profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def records_for(split, limit=None):
    from draft_a.analyze_v2 import read_records
    from draft_a.train_v2 import sequence
    plan = json.loads((SOURCE / "plan.json").read_text())
    entries = [e for e in plan["entries"] if e["split"] == split]
    if limit is not None:
        entries = entries[:limit]
    records = read_records(SOURCE, entries)
    for record in records:
        record["seq"] = sequence(record)
    return records


def collect(root, limit=None):
    import torch
    from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock
    from draft_a.corpus_v2 import load_target
    from draft_a.learning_eval import _fp32_runtime
    sets = {s: records_for(s, limit) for s in ("train", "val")}
    train_names = {r["name"] for r in sets["train"]}
    val_names = {r["name"] for r in sets["val"]}
    if train_names & val_names:
        raise ValueError("Training and validation files overlap")
    plan = dict(status="collecting", source_root=str(SOURCE), teacher="checkpoints/gpt_vfront.ckpt",
                initial_checkpoint=str(INITIAL), feature_head=str(HEAD),
                label_policy=dict(temperature=1., top_p=None, feature_classes=4096, precision="FP32"),
                scope="all F positions with an existing previous P/F pair; first F uses Target at decode time",
                test_used=False, limit=limit, files={s: [r["name"] for r in rs] for s, rs in sets.items()},
                splits={}, state_precision="FP32", raw_label_max_difference=0.)
    save(root / "data_plan.json", plan)
    started = time.monotonic()
    with torch.inference_mode(), _fp32_runtime(torch):
        target = load_target().float().eval().requires_grad_(False)
        capture = CaptureLastBlock(target.transformer)
        try:
            for split, records in sets.items():
                rows = []
                total = 0
                for n, record in enumerate(records, 1):
                    path = root / "data" / split / record["name"]
                    pairs = len(record["tokens"]) // 2
                    if pairs < 2:
                        raise ValueError("Need a previous pair for state-model supervision")
                    expected_depth = torch.arange(1, pairs)
                    if path.exists():
                        data = torch.load(path, map_location="cpu", weights_only=True)
                        if (data["name"] != record["name"] or data["split"] != split
                                or not torch.equal(data["tokens"], record["tokens"])
                                or not torch.equal(data["depth"], expected_depth)):
                            raise ValueError("Stored state data differs from the source trajectory")
                        count = len(data["position"])
                        del data
                    else:
                        seq = record["seq"]
                        capture.output = None
                        logits = target(input_ids=seq[:-1].unsqueeze(0).cuda())[0]
                        states = capture.output[0]
                        indices = (expected_depth * 2 + 1).cuda()
                        p = logits.index_select(0, indices)[:, 8000:12096].float().softmax(-1).cpu()
                        previous = states.index_select(0, indices - 2).float().cpu()
                        if not bool(torch.isfinite(p).all()) or not bool(torch.isfinite(previous).all()):
                            raise ValueError("Nonfinite Target labels or states")
                        if split == "train":
                            old = torch.load(RAW_LABELS / record["name"], map_location="cpu", weights_only=True)
                            if (old["policy"]["temperature"] != 1. or old["policy"]["top_p"] is not None
                                    or not torch.equal(old["tokens"], record["tokens"])):
                                raise ValueError("Existing original-Target labels differ")
                            difference = float((p - old["probabilities"][1:]).abs().max())
                            plan["raw_label_max_difference"] = max(plan["raw_label_max_difference"], difference)
                            if difference > 3e-6:
                                raise ValueError("Fresh original-Target probabilities differ from stored raw labels")
                            del old
                        count = len(indices)
                        data = dict(name=record["name"], split=split, tokens=record["tokens"], depth=expected_depth,
                                    previous_state=previous, previous_feature=seq[2 * expected_depth] - 8000,
                                    position=seq[2 * expected_depth + 1], sampled_feature=seq[2 * expected_depth + 2] - 8000,
                                    target=p, teacher=plan["teacher"], label_policy=plan["label_policy"])
                        if not torch.equal(data["position"], record["tokens"][2::2]):
                            raise ValueError("Current P rows are offset incorrectly")
                        if not torch.equal(data["previous_feature"], record["tokens"][1:-1:2] - 8000):
                            raise ValueError("Previous F rows are offset incorrectly")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(data, path)
                        del logits, states, indices, p, previous, data
                        capture.output = None
                    total += count
                    rows.append(dict(name=record["name"], rows=count, path=str(path)))
                    plan["splits"][split] = dict(scenes=n, rows=total, records=rows)
                    if n % 20 == 0 or n == len(records):
                        plan["seconds"] = time.monotonic() - started
                        save(root / "data_plan.json", plan)
                        print(json.dumps(dict(event="collect", split=split, scenes=n, rows=total)), flush=True)
        finally:
            capture.close()
    plan.update(status="complete", seconds=time.monotonic() - started)
    save(root / "data_plan.json", plan)
    del target
    gc.collect()
    torch.cuda.empty_cache()


def objective(logits, probabilities, weight):
    import torch
    from draft_a.feature_adapt_continue import feature_policy_tv
    p = probabilities.detach().float()
    p = p / p.sum(-1, keepdim=True)
    ce = -(p * logits.float().log_softmax(-1)).sum(-1)
    tv = feature_policy_tv(logits, p) if weight else torch.zeros_like(ce)
    return ce + weight * tv, ce, tv


def model_from(value, device="cuda"):
    import torch
    from train_feature_state_distill import FeatureStateDraft
    head = torch.load(HEAD, map_location="cpu", weights_only=True)["weight"]
    config = value["model_config"]
    model = FeatureStateDraft(head, **{k: config[k] for k in ("position_vocab", "feature_vocab", "d_model", "emb_dim", "mlp_dim", "target_softcap")})
    model.load_state_dict(value["model_state"], strict=True)
    return model.to(device).float()


def batches(data, size):
    import torch
    count = len(data["position"])
    for start in range(0, count, size):
        end = min(start + size, count)
        yield {k: data[k][start:end].cuda() for k in ("previous_state", "previous_feature", "position", "sampled_feature", "target", "depth")}


def evaluate(model, datasets, size):
    import torch
    from draft_a.feature_adapt_continue import policy_probs
    model.eval()
    reports = []
    with torch.inference_mode():
        for data in datasets:
            ce = overlap = hard = 0.
            count = 0
            for b in batches(data, size):
                _, logits = model(b["previous_state"], b["previous_feature"], b["position"])
                loss, _, _ = objective(logits, b["target"], 0.)
                p = b["target"] / b["target"].sum(-1, keepdim=True)
                p = policy_probs(p.log())
                q = policy_probs(logits)
                overlap += float(torch.minimum(p, q).sum())
                ce += float(loss.sum())
                hard += float(torch.nn.functional.cross_entropy(logits.float(), b["sampled_feature"], reduction="sum"))
                count += len(logits)
            reports.append(dict(name=data["name"], rows=count, overlap=overlap/count, raw_ce=ce/count, sampled_ce=hard/count))
    total = sum(r["rows"] for r in reports)
    return dict(rows=total, scenes=len(reports),
                pooled_overlap=sum(r["overlap"] * r["rows"] for r in reports)/total,
                scene_mean_overlap=sum(r["overlap"] for r in reports)/len(reports),
                raw_ce=sum(r["raw_ce"] * r["rows"] for r in reports)/total,
                sampled_ce=sum(r["sampled_ce"] * r["rows"] for r in reports)/total,
                by_scene=reports)


def train(root, steps=1200, chunk=256, *, initial_path=INITIAL, data_root=None,
          phase_plan=None, eval_every=None, train_probe=0):
    import torch
    from draft_a.learning_eval import _fp32_runtime
    data_plan = json.loads(((data_root or root) / "data_plan.json").read_text())
    if data_plan["status"] != "complete":
        raise ValueError("All training and validation states are required")
    datasets = {s: [torch.load(r["path"], map_location="cpu", weights_only=True)
                    for r in data_plan["splits"][s]["records"]] for s in ("train", "val")}
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    torch.manual_seed(phase_plan[0]["seed"] if phase_plan else 24684)
    torch.set_num_threads(6)
    model = model_from(initial)
    phases = phase_plan or [dict(name="full_ce", lr=3e-5, weight=0., seed=24684),
              dict(name="low_lr_ce", lr=1e-5, weight=0., seed=24686),
              dict(name="policy_tv", lr=1e-5, weight=2., seed=24688)]
    start_step = int(initial.get("step", 0))
    initial_exposure = int(initial.get("trained_feature_rows", 0))
    probe_count = min(train_probe, len(datasets["train"]))
    probe = [datasets["train"][round(i*(len(datasets["train"])-1)/max(1,probe_count-1))]
             for i in range(probe_count)]
    plan = dict(teacher="checkpoints/gpt_vfront.ckpt", initial_checkpoint=str(initial_path), feature_head=str(HEAD),
                parameters=sum(p.numel() for p in model.parameters()), phases=phases, steps_per_phase=steps,
                trajectories_per_update=2, rows_per_chunk=chunk, precision="BF16 training; FP32 evaluation",
                objective="raw T=1 full Target F distribution CE + weight * TV of both T0.9/top_p0.9 policies",
                selection="maximum pooled overlap on all validation F rows; retain initial if no improvement",
                source_root=str(SOURCE), train_files=data_plan["files"]["train"], val_files=data_plan["files"]["val"],
                train_unique_rows=data_plan["splits"]["train"]["rows"], test_used=False,
                start_step=start_step, initial_trained_feature_rows=initial_exposure,
                evaluation_interval=eval_every, training_probe_files=[d["name"] for d in probe],
                optimizer_start="new AdamW from model weights")
    save(root / "training_plan.json", plan)
    history = []
    best = -1.
    global_step, exposure = start_step, initial_exposure
    started = time.monotonic()

    def measure(phase):
        nonlocal best
        with _fp32_runtime(torch):
            metrics = evaluate(model, datasets["val"], chunk)
            training_metrics = evaluate(model, probe, chunk) if probe else None
        path = root / f"step_{global_step:04d}.pt"
        value = dict(model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                     model_config=initial["model_config"], student_temperature=.9, student_top_p=.9,
                     teacher=plan["teacher"], step=global_step, trained_feature_rows=exposure,
                     phase=phase, original_checkpoint=str(initial_path), validation=metrics)
        torch.save(value, path)
        if metrics["pooled_overlap"] > best:
            best = metrics["pooled_overlap"]
            torch.save(value, root / "selected.pt")
            save(root / "selected.json", dict(step=global_step, phase=phase, checkpoint=str(root/"selected.pt"),
                                              overlap=best, trained_feature_rows=exposure))
        row = dict(step=global_step, phase=phase, trained_feature_rows=exposure,
                   seconds=time.monotonic()-started, **{k:v for k,v in metrics.items() if k != "by_scene"})
        if training_metrics is not None:
            row.update(training_probe_overlap=training_metrics["pooled_overlap"],
                       training_probe_ce=training_metrics["raw_ce"],
                       training_probe_rows=training_metrics["rows"])
            save(root / f"step_{global_step:04d}_training_probe.json", training_metrics)
        history.append(row)
        save(root / "history.json", history)
        save(root / f"step_{global_step:04d}_validation.json", metrics)
        print(json.dumps(dict(event="validation", **row)), flush=True)
        model.train()

    measure("initial")
    for phase in phases:
        phase_steps = int(phase.get("steps", steps))
        torch.manual_seed(phase["seed"])
        rng = random.Random(phase["seed"])
        order = []
        cursor = 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=phase["lr"], weight_decay=.01)
        for local_step in range(1, phase_steps+1):
            pair = []
            for _ in range(2):
                if cursor == len(order):
                    order = list(range(len(datasets["train"])))
                    rng.shuffle(order)
                    cursor = 0
                pair.append(datasets["train"][order[cursor]])
                cursor += 1
            rows = sum(len(d["position"]) for d in pair)
            lr = phase["lr"]*(.1+.9*(1+math.cos(math.pi*(local_step-1)/phase_steps))/2)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for data in pair:
                for b in batches(data, chunk):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, logits = model(b["previous_state"], b["previous_feature"], b["position"])
                        losses, _, _ = objective(logits, b["target"], phase["weight"])
                        loss = losses.sum()/rows
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite Feature loss")
                    loss.backward()
                    total += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(torch.isfinite(norm)):
                raise ValueError("Nonfinite Feature gradient")
            optimizer.step()
            global_step += 1
            exposure += rows
            if local_step == 1 or local_step % 50 == 0:
                progress = dict(status="training", phase=phase["name"], step=global_step, local_step=local_step,
                                loss=total, lr=lr, trained_feature_rows=exposure, seconds=time.monotonic()-started)
                save(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            milestones = ({min(200, phase_steps), min(600, phase_steps), phase_steps}
                          if eval_every is None else set(range(eval_every, phase_steps+1, eval_every)) | {phase_steps})
            if local_step in milestones:
                measure(phase["name"])
        del optimizer
    save(root / "progress.json", dict(status="complete", step=global_step, seconds=time.monotonic()-started,
                                      trained_feature_rows=exposure, selected=json.loads((root/"selected.json").read_text())))
    del model, datasets
    gc.collect()
    torch.cuda.empty_cache()


def online(root, scenes=12, tokens=512, limit=None, *, reference_checkpoint=INITIAL,
           reference_name="old_mlp", candidate_name="target_full_v2"):
    import numpy as np
    import torch
    from joblib import load
    from threadpoolctl import threadpool_limits
    from bench_fast_hgb import FastHistRanker
    from bench_gaussiangpt_pf_block2_cow import load_feature_model
    from draft_a.corpus_v2 import load_target
    from draft_a.learning_eval import _fp32_runtime
    from draft_a.pf_ranker_eval import decode as ordinary, _prompts
    from draft_a.pf_feature_unified import decode, MLPAdapter
    if json.loads((root/"progress.json").read_text())["status"] != "complete":
        raise ValueError("Training must complete before test evaluation")
    records = records_for("test", scenes if limit is None else limit)
    result = dict(status="running", split="test", selected=json.loads((root/"selected.json").read_text()),
                  parameters=dict(scenes=len(records), prefixes=[0,64,256], tokens=tokens, k=2, seed=24685,
                                  temperature=.9, top_p=.9, precision="FP32"), rows=[],
                  scope="includes cache allocation, prefill, ranker, F proposal, Target verification and sampling; excludes model loading",
                  model_selection="validation only; test trajectories previously used in earlier comparisons")
    with torch.inference_mode(), _fp32_runtime(torch), threadpool_limits(limits=1, user_api="openmp"):
        target = load_target().float().eval().requires_grad_(False)
        ranker = FastHistRanker(load(RANKER))
        ranks = target.transformer.sparse_position_rank_map.cpu().tolist()
        models = {}
        for name, path in [(reference_name, reference_checkpoint), (candidate_name, root/"selected.pt")]:
            model, _, _ = load_feature_model(path, HEAD, target.transformer, torch.device("cuda"), 8000, 12096)
            models[name] = MLPAdapter(model)
        validations = []
        for prefix, prompt in _prompts(records[0]):
            for name, adapter in models.items():
                row, seq = decode(target.transformer, adapter, ranker, ranks, prompt, 32, 24685)
                prof, seq2 = decode(target.transformer, adapter, ranker, ranks, prompt, 32, 24685, profile=True)
                if seq != seq2 or any(row[k] != prof[k] for k in ["accepted", "verified", "proposed", "rounds"]):
                    raise ValueError("Stage timing changed sequence or counts")
                validations.append(dict(prefix=prefix, variant=name, status="passed"))
        result["validation"] = validations
        for index, record in enumerate(records):
            for prefix, prompt in _prompts(record):
                seed = (24685+index*1009+prefix*17) % (2**63-1)
                variants = ["target_only", *models]
                shift = (index+prefix//64) % len(variants)
                variants = variants[shift:]+variants[:shift]
                for name in variants:
                    if name == "target_only":
                        row = ordinary(target.transformer, None, ranker, ranks, prompt, tokens, 0, seed)
                    else:
                        row, _ = decode(target.transformer, models[name], ranker, ranks, prompt, tokens, seed)
                    row.update(name=record["name"], prefix=prefix, variant=name)
                    result["rows"].append(row)
                    save(root/"online.json", result)
                    print(json.dumps(dict(event="online", name=record["name"], prefix=prefix, variant=name,
                                          P=row.get("position_acceptance_verified"), F=row.get("feature_acceptance_verified"),
                                          tps=row["tokens_per_second"])), flush=True)
    result["status"] = "complete"
    save(root/"online.json", result)
    names = [r["name"] for r in records]
    variants = ["target_only", reference_name, candidate_name]
    def aggregate(rows):
        keys = ["generated_tokens", "seconds", "target_calls", "position_accepted", "position_verified",
                "feature_accepted", "feature_verified", "feature_proposed", "rounds"]
        sums = {k: sum(r.get(k,0) for r in rows) for k in keys}
        sums["tps"] = sums["generated_tokens"]/sums["seconds"]
        for kind in ("position", "feature"):
            sums[kind+"_acceptance"] = sums[kind+"_accepted"]/sums[kind+"_verified"] if sums[kind+"_verified"] else None
        return sums
    summary = {v: aggregate([r for r in result["rows"] if r["variant"] == v]) for v in variants}
    rng = np.random.default_rng(24685)
    speed = {v: [] for v in variants[1:]}
    relative = []
    delta = []
    for _ in range(2000):
        sampled = rng.choice(names, len(names), replace=True)
        values = {v: aggregate([r for n in sampled for r in result["rows"] if r["name"] == n and r["variant"] == v]) for v in variants}
        for v in speed:
            speed[v].append(values[v]["tps"]/values["target_only"]["tps"])
        relative.append(values[candidate_name]["tps"]/values[reference_name]["tps"])
        delta.append(values[candidate_name]["feature_acceptance"]-values[reference_name]["feature_acceptance"])
    for v in speed:
        summary[v]["speedup"] = summary[v]["tps"]/summary["target_only"]["tps"]
        summary[v]["speedup_ci95"] = np.quantile(speed[v],[.025,.975]).tolist()
    summary["new_over_old"] = dict(speed=summary[candidate_name]["tps"]/summary[reference_name]["tps"],
                                   speed_ci95=np.quantile(relative,[.025,.975]).tolist(),
                                   feature_delta=summary[candidate_name]["feature_acceptance"]-summary[reference_name]["feature_acceptance"],
                                   feature_delta_ci95=np.quantile(delta,[.025,.975]).tolist())
    save(root/"online_summary.json", summary)


def self_test():
    import torch
    from draft_a.feature_adapt_continue import policy_probs, feature_policy_tv
    p = torch.randn(5,4096).softmax(-1)
    logits = torch.randn(5,4096,requires_grad=True)
    zero, ce, tv = objective(logits,p,0.)
    torch.testing.assert_close(zero, -(p*logits.log_softmax(-1)).sum(-1))
    weighted, _, tv = objective(logits,p,2.)
    torch.testing.assert_close(weighted,ce+2*tv)
    torch.testing.assert_close(tv,1-torch.minimum(policy_probs(p.log()),policy_probs(logits)).sum(-1),atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(feature_policy_tv(p.log(),p),torch.zeros(5),atol=2e-6,rtol=0)
    weighted.mean().backward()
    if not bool(torch.isfinite(logits.grad).all()) or not bool((logits.grad != 0).any()):
        raise ValueError("Invalid probability gradient")
    print(json.dumps(dict(status="passed", raw_ce=True, tv_matches_overlap=True, finite_gradient=True)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "self-test"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.mode == "self-test":
        self_test()
        return
    if args.root.exists():
        raise FileExistsError("Select a new output directory; existing results are retained")
    args.root.mkdir(parents=True)
    save(args.root/"pipeline.json",dict(status="running",phase="collect",pid=__import__("os").getpid()))
    try:
        collect(args.root,2 if args.smoke else None)
        save(args.root/"pipeline.json",dict(status="running",phase="train"))
        train(args.root,2 if args.smoke else args.steps,args.chunk)
        save(args.root/"pipeline.json",dict(status="running",phase="online"))
        online(args.root,args.scenes,32 if args.smoke else args.tokens,1 if args.smoke else None)
        save(args.root/"pipeline.json",dict(status="complete",phase="complete"))
    except Exception:
        import traceback
        save(args.root/"pipeline.json",dict(status="failed",error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
