"""Finite comparison of sampled-label CE and full Feature distribution training."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
import torch.nn.functional as F

from draft_a.learning_train import SOURCE_ROOT, STUDY_ROOT, write_json

OUTPUT = Path("draft_a/runs/pf_feature_distill_20261014")
FEATURE_START, FEATURE_END = 8000, 12096
LABEL_POLICY = dict(temperature=1.0, top_p=None, dtype="FP32",
                    support="all 4096 allowed Feature ids", slots="all Feature prediction slots")


def prepare(study_root, reference_root):
    original = json.loads((reference_root / "plan.json").read_text())
    initial = torch.load(reference_root / "train_n400_seed24681/phase_00.pt",
                         map_location="cpu", weights_only=True)
    plan = dict(original, study_root=str(study_root), reference_root=str(reference_root),
                objective="feature_kd", label_policy=LABEL_POLICY, config=initial["config"],
                jobs=[dict(size=400, seed=seed) for seed in original["seeds"]],
                created_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                training_objective="P and EOS sampled-label CE; F Target soft-label CE; sum divided by all target tokens",
                comparison="Same F exposure, initialization and sequence order; training time also reported",
                evaluation="Reuse original evaluation cache at temperature0.9/top_p0.9; training softmax temperature1, no nucleus truncation")
    del initial
    path = study_root / "plan.json"
    if path.exists():
        existing = json.loads(path.read_text())
        for key in ("source_root", "reference_root", "objective", "label_policy", "config",
                    "subsets", "seeds", "jobs", "token_budgets", "val_files", "train_eval_files"):
            if existing[key] != plan[key]:
                raise ValueError(f"Saved experiment differs: {key}")
        return existing
    study_root.mkdir(parents=True, exist_ok=True)
    write_json(path, plan)
    return plan


def feature_loss(logits, labels, probabilities):
    """Preserve per-token weights and hard supervision outside Feature slots."""
    feature_mask = (labels >= FEATURE_START) & (labels < FEATURE_END)
    selected = logits[feature_mask, FEATURE_START:FEATURE_END].float()
    if selected.shape != probabilities.shape or not bool(torch.isfinite(selected).all()):
        raise ValueError("Feature labels and logits differ or allowed logits are nonfinite")
    log_q = F.log_softmax(selected, dim=-1)
    soft_sum = -(probabilities * log_q).sum()
    hard_sum = F.cross_entropy(logits[~feature_mask].float(), labels[~feature_mask], reduction="sum")
    return (hard_sum + soft_sum) / labels.numel()


def _read_training(plan):
    from draft_a.analyze_v2 import read_records
    from draft_a.train_v2 import sequence
    source_root = Path(plan["source_root"])
    source = json.loads((source_root / "plan.json").read_text())
    by_name = {entry["name"]: entry for entry in source["entries"]}
    names = plan["subsets"]["400"]
    if any(by_name[name]["split"] != "train" for name in names):
        raise ValueError("Soft supervision may only use training trajectories")
    records = read_records(source_root, [by_name[name] for name in names])
    for record in records:
        record["seq"] = sequence(record)
    return records


def _valid_labels(value, record):
    p = value["probabilities"]
    count = len(record["tokens"]) // 2
    if (value["name"] != record["name"] or value["policy"] != LABEL_POLICY
            or value["has_eos"] != record["has_eos"]
            or not torch.equal(value["tokens"], record["tokens"])
            or p.dtype != torch.float32 or p.shape != (count, 4096)
            or not bool(torch.isfinite(p).all()) or not bool((p > 0).all())
            or not torch.allclose(p.sum(-1), torch.ones(count), atol=1e-6, rtol=1e-6)):
        raise ValueError(f"Invalid full Feature probability labels: {record['name']}")
    return p


def prepare_labels(study_root, plan):
    from draft_a.corpus_v2 import load_target
    from draft_a.learning_eval import _fp32_runtime, _save_labels
    records = _read_training(plan)
    cache_root = study_root / "feature_labels"
    cache_root.mkdir(parents=True, exist_ok=True)
    summary = dict(status="preparing", complete=False, policy=LABEL_POLICY,
                   files=[record["name"] for record in records], completed=0,
                   feature_tokens=0, seconds=0.0)
    write_json(cache_root / "summary.json", summary)
    started = time.monotonic()
    target = None
    try:
        with _fp32_runtime(torch):
            for record in records:
                path = cache_root / record["name"]
                if path.exists():
                    value = torch.load(path, map_location="cpu", weights_only=True)
                    p = _valid_labels(value, record)
                else:
                    if target is None:
                        target = load_target().float().eval().requires_grad_(False)
                    seq = record["seq"]
                    logits = target(input_ids=seq[:-1].unsqueeze(0).cuda())[0]
                    feature = logits[1:len(record["tokens"]):2, FEATURE_START:FEATURE_END]
                    if not bool(torch.isfinite(feature).all()):
                        raise ValueError("Allowed Target Feature logits must be finite")
                    if not bool(torch.isneginf(logits[1:len(record["tokens"]):2, :FEATURE_START]).all()):
                        raise ValueError("Target Feature slots allow Position probability")
                    if not bool(torch.isneginf(logits[1:len(record["tokens"]):2, FEATURE_END:]).all()):
                        raise ValueError("Target Feature slots allow EOS or extra probability")
                    p = feature.softmax(-1).cpu()
                    value = dict(name=record["name"], tokens=record["tokens"],
                                 has_eos=record["has_eos"], policy=LABEL_POLICY, probabilities=p)
                    _valid_labels(value, record)
                    _save_labels(path, value, torch)
                    del logits, feature
                summary["completed"] += 1
                summary["feature_tokens"] += p.shape[0]
                summary["seconds"] = time.monotonic() - started
                if summary["completed"] % 10 == 0:
                    write_json(cache_root / "summary.json", summary)
                    print(json.dumps(dict(event="labels", **summary)), flush=True)
                del value, p
        summary.update(status="ready", complete=True, seconds=time.monotonic() - started)
        write_json(cache_root / "summary.json", summary)
    finally:
        target = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    return summary


def load_training_labels(study_root, records):
    root = study_root / "feature_labels"
    summary = json.loads((root / "summary.json").read_text())
    if (summary["status"] != "ready" or summary["complete"] is not True
            or summary["policy"] != LABEL_POLICY or summary["completed"] != 400
            or summary["files"] != [record["name"] for record in records]):
        raise ValueError("Full training distribution labels are required")
    result = {}
    for record in records:
        value = torch.load(root / record["name"], map_location="cpu", weights_only=True)
        result[record["name"]] = _valid_labels(value, record)
    if sum(p.shape[0] for p in result.values()) != summary["feature_tokens"]:
        raise ValueError("Feature supervision count differs")
    return result


def validate_loss(study_root, plan):
    """Test the real model, unequal sequence lengths and natural EOS handling."""
    from omegaconf import OmegaConf
    from draft_a.train_v2 import make_model
    from draft_a.learning_eval import _fp32_runtime
    records = _read_training(plan)
    selected = [min((r for r in records if r["has_eos"]), key=lambda r: len(r["seq"])),
                min((r for r in records if not r["has_eos"]), key=lambda r: len(r["seq"]))]
    torch.manual_seed(24681)
    model = make_model(OmegaConf.create(plan["config"])).float().eval()
    count = sum(len(record["seq"]) - 1 for record in selected)
    losses_old, losses_new, grads = [], [], []
    for record in selected:
        seq = record["seq"].cuda().unsqueeze(0)
        with _fp32_runtime(torch):
            reference_loss, actual_logits = model(input_ids=seq, labels=seq, return_logits=True)
            actual_logits = actual_logits[0].clone()
        logits = actual_logits.clone().requires_grad_(True)
        labels = seq[0, 1:]
        mask = (labels >= FEATURE_START) & (labels < FEATURE_END)
        one_hot = F.one_hot(labels[mask] - FEATURE_START, num_classes=4096).float()
        old = F.cross_entropy(logits, labels)
        new = feature_loss(logits, labels, one_hot)
        grad_old, = torch.autograd.grad(old, logits, retain_graph=True)
        grad_new, = torch.autograd.grad(new, logits)
        delta = float((grad_old - grad_new).abs().max())
        if (not torch.isclose(old.detach(), reference_loss, atol=1e-6, rtol=1e-6)
                or not torch.isclose(old.detach(), new.detach(), atol=1e-6, rtol=1e-6)
                or delta > 1e-6):
            raise ValueError("Real-model one-hot equivalence failed")
        weight = labels.numel() / count
        losses_old.append(float(old.detach()) * weight)
        losses_new.append(float(new.detach()) * weight)
        grads.append(dict(name=record["name"], target_tokens=labels.numel(),
                          has_eos=record["has_eos"], max_gradient_difference=delta,
                          loss_difference=float((old.detach() - new.detach()).abs()),
                          original_wrapper_loss_difference=float((old.detach() - reference_loss).abs())))
        del logits, actual_logits, seq, one_hot, old, new, grad_old, grad_new
    result = dict(loss_difference=abs(sum(losses_old) - sum(losses_new)), sequences=grads,
                  total_tokens=count, success=True, atol=1e-6, rtol=1e-6)
    write_json(study_root / "loss_validation.json", result)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return result


def validate_label_alignment(study_root, reference_root, plan):
    from draft_a.learning_eval import _helpers
    _, _, overlap, _ = _helpers()
    maximum = 0.0
    count = 0
    for name in plan["train_eval_files"]:
        full = torch.load(study_root / "feature_labels" / name, map_location="cpu", weights_only=True)
        old = torch.load(reference_root / "teacher_cache/train" / name, map_location="cpu", weights_only=True)
        for i, (slot, kind) in enumerate(zip(old["metadata"]["slots"], old["metadata"]["kinds"])):
            if kind != "feature":
                continue
            logits = torch.full((FEATURE_END + 2,), -torch.inf)
            logits[FEATURE_START:FEATURE_END] = full["probabilities"][slot // 2].log()
            actual = overlap.sampling_probs(logits, None, 0.9, 0.9)
            expected = torch.zeros_like(actual)
            expected[old["indices"][i]] = old["probabilities"][i]
            delta = float((actual - expected).abs().max())
            maximum = max(maximum, delta)
            count += 1
            if delta > 2e-6:
                raise ValueError(f"Full labels differ from evaluation Target: {name}/{slot}")
    result = dict(sampled_feature_slots=count, max_probability_difference=maximum, success=True)
    write_json(study_root / "label_alignment.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=OUTPUT)
    parser.add_argument("--reference-root", type=Path, default=STUDY_ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(6)
    plan = prepare(args.study_root, args.reference_root)
    if args.prepare_only:
        print(json.dumps(plan), flush=True)
        return
    def status(stage, state="running", **details):
        write_json(args.study_root / "study_status.json",
                   dict(stage=stage, state=state, pid=os.getpid(), epoch=time.time(), **details))
    try:
        from draft_a.learning_eval import require_complete_cache
        require_complete_cache(Path(plan["source_root"]), args.reference_root)
        status("loss_validation")
        if not (args.study_root / "loss_validation.json").exists():
            validate_loss(args.study_root, plan)
        status("feature_labels")
        labels = prepare_labels(args.study_root, plan)
        status("label_alignment")
        validate_label_alignment(args.study_root, args.reference_root, plan)
        status("smoke")
        shared = ["--source-root", plan["source_root"], "--study-root", str(args.study_root),
                  "--evaluation-root", str(args.reference_root), "--size", "400", "--objective", "feature_kd"]
        smoke_root = args.study_root / "smoke/train_n400_seed24681"
        if not (smoke_root / "complete.json").exists():
            command = [sys.executable, "-u", "-m", "draft_a.learning_train", *shared,
                       "--seed", "24681", "--smoke"]
            if (smoke_root / "last.pt").exists():
                command.append("--resume")
            with (args.study_root / "smoke.log").open("a") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        completed = []
        for seed in plan["seeds"]:
            name = f"train_n400_seed{seed}"
            out = args.study_root / name
            if not (out / "complete.json").exists():
                status(name, completed_jobs=completed, total_jobs=3)
                command = [sys.executable, "-u", "-m", "draft_a.learning_train", *shared, "--seed", str(seed)]
                if (out / "last.pt").exists():
                    command.append("--resume")
                with (args.study_root / f"{name}.log").open("a") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            completed.append(name)
            from draft_a.feature_distill_report import build_report
            build_report(args.study_root, args.reference_root)
        from draft_a.feature_distill_report import plot_report
        plot_report(args.study_root)
        status("complete", "complete", completed_jobs=completed, labels=labels)
    except Exception as error:
        status("failed", "failed", error=str(error))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
