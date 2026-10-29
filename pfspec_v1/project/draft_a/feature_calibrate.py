"""Diagnose Feature sampling policies for an existing FP32 Draft checkpoint.

Run as draft_a.feature_calibrate from the GaussianGPT root. Reuse the original
teacher_cache and sampled slots; validation is calibration data, not a test set.
No training, Target forward pass, or cache writes are performed.
"""

import argparse
import json
import math
from pathlib import Path


TEMPERATURES = (0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 1.2)
TOP_PS = (0.9, 1.0)
POLICIES = tuple((t, p) for t in TEMPERATURES for p in TOP_PS)
ORIGINAL = (0.9, 0.9)
METRICS = ("overlap", "top1_match", "pentropy", "qentropy")
KINDS = ("all", "position", "feature", "eos")


def _evaluation():
    if __package__:
        from . import learning_eval
    else:
        import learning_eval
    return learning_eval


def _stats(samples, token_count):
    result = {"sample_count": len(samples), "token_count": token_count}
    for metric in METRICS:
        total = math.fsum(row[metric] for row in samples)
        result[metric + "_sum"] = total
        result[metric] = total / len(samples) if samples else None
    return result


def _pool(rows):
    count = sum(row["sample_count"] for row in rows)
    observed = [row for row in rows if row["sample_count"]]
    result = {"sequence_count": len(rows), "sampled_sequence_count": len(observed),
              "sample_count": count, "token_count": sum(row["token_count"] for row in rows)}
    for metric in METRICS:
        total = math.fsum(row[metric + "_sum"] for row in rows)
        result["pooled_overlap" if metric == "overlap" else metric] = total / count if count else None
        result["equal_trajectory_mean_" + metric] = (
            math.fsum(row[metric] for row in observed) / len(observed) if observed else None)
    return result


def _sample(row, cache, index, policy, overlap, torch, evaluation):
    meta = cache["metadata"]
    q = overlap.sampling_probs(row, meta["prev_pos"][index], *policy)
    evaluation._valid_sampling_probs(q, torch)
    ids, p = cache["indices"][index], cache["probabilities"][index]
    value = float(torch.minimum(q[ids], p).sum())
    evaluation._require(math.isfinite(value) and -1e-6 <= value <= 1 + 1e-5,
                        "Invalid probability overlap")
    return {"overlap": value,
            "top1_match": int(int(q.argmax()) == cache["target_top1"][index]),
            "pentropy": cache["pentropy"][index],
            "qentropy": evaluation._entropy(q, torch)}


def _calibrate_rows(rows, cache, overlap, torch, evaluation):
    meta = cache["metadata"]
    evaluation._require(rows.dtype == torch.float32 and rows.ndim == 2 and
                        rows.shape[0] == len(meta["slots"]), "Expected sampled FP32 logits")
    original = {kind: [] for kind in KINDS}
    feature = {policy: [] for policy in POLICIES}
    for index, (row, kind) in enumerate(zip(rows, meta["kinds"])):
        unchanged = _sample(row, cache, index, ORIGINAL, overlap, torch, evaluation)
        original["all"].append(unchanged)
        original[kind].append(unchanged)
        if kind == "feature":
            for policy in POLICIES:
                feature[policy].append(unchanged if policy == ORIGINAL else
                                       _sample(row, cache, index, policy, overlap, torch, evaluation))
    counts = {"all": meta["context"], "position": meta["body_tokens"] // 2,
              "feature": meta["body_tokens"] // 2, "eos": int(meta["has_eos"])}
    return {"name": meta["name"], "split": meta["split"], "context": meta["context"],
            "has_eos": meta["has_eos"], "slots": meta["slots"],
            "kinds": meta["kinds"], "sampled_depths": meta["sampled_depths"],
            "original": {kind: _stats(values, counts[kind]) for kind, values in original.items()},
            "feature": [dict(temperature=t, top_p=p, **_stats(feature[t, p], counts["feature"]))
                        for t, p in POLICIES]}


def _summarize(per_sequence):
    splits, original = {}, {}
    for split in ("train", "val"):
        rows = [row for row in per_sequence if row["split"] == split]
        original[split] = {kind: _pool([row["original"][kind] for row in rows]) for kind in KINDS}
        position = original[split]["position"]
        feature = []
        for index, (temperature, top_p) in enumerate(POLICIES):
            metrics = _pool([row["feature"][index] for row in rows])
            f, p = metrics["pooled_overlap"], position["pooled_overlap"]
            feature.append(dict(temperature=temperature, top_p=top_p, **metrics,
                                diagnostic_goal_met=(f > 0.30 and p >= 0.90)
                                if f is not None and p is not None else None))
        splits[split] = {"position": position, "feature": feature,
                         "eos": original[split]["eos"]}
    return splits, original


def _compare_original(checkpoint, result):
    """Compare only sampling metrics; full-token CE is outside this diagnostic."""
    path = checkpoint.with_name(checkpoint.stem + "_metrics.json")
    if not path.exists():
        return {"status": "unavailable", "path": str(path),
                "reason": "No saved original evaluation for this checkpoint"}
    saved = json.loads(path.read_text())
    saved = saved.get("metrics", saved)
    for key in ("checkpoint", "checkpoint_step", "source_root", "scope", "policy", "files"):
        if saved.get(key) != result[key]:
            return {"status": "incomparable", "path": str(path), "reason": f"Different {key}"}
    differences, deltas = [], []

    def compare(label, current, previous, fields):
        for field in fields:
            a, b = current[field], previous[field]
            if a is None or b is None:
                matches = a is b
            else:
                deltas.append(abs(a - b))
                matches = math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-5)
            if not matches:
                differences.append({"location": label, "metric": field, "current": a, "previous": b})

    fields = ("sequence_count", "sampled_sequence_count", "sample_count", "token_count",
              "pooled_overlap", "equal_trajectory_mean_overlap", "top1_match", "pentropy", "qentropy")
    for split in ("train", "val"):
        for kind in KINDS:
            compare(f"{split}/{kind}", result["original_splits"][split][kind],
                    saved["splits"][split][kind], fields)
    old_rows = {(row["split"], row["name"], row["kind"]): row for row in saved["per_sequence"]}
    for row in result["per_sequence"]:
        for kind in KINDS:
            key = (row["split"], row["name"], kind)
            compare("/".join(key), row["original"][kind], old_rows[key],
                    ("sample_count", "token_count", *METRICS, *(m + "_sum" for m in METRICS)))
    return {"status": "matched" if not differences else "different", "path": str(path),
            "compared_numeric_values": len(deltas), "max_absolute_difference": max(deltas, default=0.0),
            "absolute_tolerance": 1e-6, "relative_tolerance": 1e-5, "differences": differences}


def evaluate_checkpoint(checkpoint, source_root, evaluation_root):
    import torch
    from omegaconf import OmegaConf
    from draft_a.train_v2 import NanoGPTGaussianModel

    evaluation = _evaluation()
    source_root, evaluation_root, expected, selected, entries = evaluation._plan(
        source_root, evaluation_root, None)
    _, train, overlap, analysis = evaluation._helpers()
    records = evaluation._records(source_root, selected, entries, analysis)
    caches, summary = evaluation._load_labels(source_root, evaluation_root, expected, selected,
                                             records, train, overlap, torch, True)
    checkpoint = Path(checkpoint).resolve()
    draft = state = logits = rows = seq = x = None
    per_sequence = []
    with evaluation._fp32_runtime(torch):
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            cfg = OmegaConf.create(state["config"])
            draft = NanoGPTGaussianModel(config=cfg, codebook_size=4096,
                                         tokens_per_latent=2, num_feature_tokens=1).cuda().float()
            draft.load_state_dict(state["model"], strict=True)
            step = state.get("step")
            state = None
            draft.eval().requires_grad_(False)
            for split, trajectories in records.items():
                for record, cache in zip(trajectories, caches[split]):
                    seq = train.sequence(record)
                    meta = cache["metadata"]
                    evaluation._model_ready(draft, meta["context"], torch)
                    x = seq[:-1].unsqueeze(0).cuda()
                    logits = draft(input_ids=x)[0]
                    evaluation._require(logits.dtype == torch.float32 and logits.ndim == 2 and
                                        logits.shape[0] == meta["context"] and
                                        logits.shape[1] >= evaluation.EOS + 1,
                                        "Expected full-trajectory FP32 logits")
                    rows = logits[meta["slots"]].cpu()
                    logits = x = None
                    per_sequence.append(_calibrate_rows(rows, cache, overlap, torch, evaluation))
                    rows = seq = None
        finally:
            draft = state = logits = rows = seq = x = None
    splits, original = _summarize(per_sequence)
    result = {"checkpoint": str(checkpoint), "checkpoint_step": step,
              "source_root": str(source_root), "evaluation_root": str(evaluation_root),
              "target_precision": "FP32", "draft_precision": "FP32", "scope": evaluation.SCOPE,
              "policy": dict(evaluation.POLICY), "target_policy": dict(evaluation.POLICY),
              "position_policy": dict(evaluation.POLICY), "feature_temperatures": list(TEMPERATURES),
              "feature_top_ps": list(TOP_PS), "samples_per_type": 32,
              "entropy_units": "nats", "top1_definition": "Draft / cached Target argmax match rate",
              "overlap_definition": "sum(min(Draft probability, fixed Target probability))",
              "averaging": "pooled_overlap weights sampled slots; equal_trajectory_mean_overlap weights trajectories equally",
              "split_usage": {"train": "training diagnostics", "val": "calibration diagnostics"},
              "independent_test": False, "diagnostic_only": True,
              "diagnostic_goal": {"feature_overlap_gt": 0.30, "position_overlap_gte": 0.90,
                                  "metric": "pooled_overlap", "evidence": "train and val only"},
              "complete": True, "teacher_cache_complete": summary["complete"], "files": selected,
              "splits": splits, "original_splits": original, "per_sequence": per_sequence}
    result["original_evaluation_comparison"] = _compare_original(checkpoint, result)
    json.dumps(result, allow_nan=False)
    return result


def self_test():
    """CPU arithmetic tests without model imports, packages, or file writes."""
    evaluation = _evaluation()

    def close(a, b):
        evaluation._require(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), f"{a} != {b}")

    def distribution(logits, temperature, top_p):
        maximum = max(logits)
        weights = [math.exp((x - maximum) / temperature) for x in logits]
        total = math.fsum(weights)
        probabilities = [x / total for x in weights]
        order = sorted(range(len(logits)), key=lambda i: -logits[i])
        keep, cumulative = [], 0.0
        for index in order:
            keep.append(index)
            cumulative += probabilities[index]
            if cumulative > top_p:
                break
        total = math.fsum(probabilities[i] for i in keep)
        return [probabilities[i] / total if i in keep else 0.0 for i in range(len(logits))]

    p = [0.6, 0.3, 0.1, 0.0]
    q = [0.2, 0.3, 0.0, 0.5]
    ids = [i for i, value in enumerate(p) if value > 0]
    sparse = [p[i] for i in ids]
    close(math.fsum(p), 1.0)
    close(math.fsum(q), 1.0)
    close(math.fsum(sparse), 1.0)
    close(math.fsum(min(q[i], value) for i, value in zip(ids, sparse)), 0.5)
    close(math.fsum(min(a, b) for a, b in zip(p, q)),
          1 - math.fsum(abs(a - b) for a, b in zip(p, q)) / 2)
    close(math.fsum(min(p[i], value) for i, value in zip(ids, sparse)), 1.0)
    close(-math.fsum(x * math.log(x) for x in sparse),
          -math.fsum(x * math.log(x) for x in p if x > 0))
    for temperature, top_p in POLICIES:
        target = distribution([1.2, 0.4, -0.7, -math.inf], *ORIGINAL)
        draft = distribution([0.2, 0.8, -0.5, -math.inf], temperature, top_p)
        indices = [i for i, value in enumerate(target) if value > 0]
        close(math.fsum(target), 1.0)
        close(math.fsum(draft), 1.0)
        close(draft[-1], 0.0)
        close(math.fsum(min(a, b) for a, b in zip(target, draft)),
              math.fsum(min(draft[i], target[i]) for i in indices))
        close(math.fsum(min(target[i], target[i]) for i in indices), 1.0)
    crossing = distribution([math.log(x) for x in (0.6, 0.25, 0.15)], 1.0, 0.7)
    close(crossing[0], 0.6 / 0.85)
    close(crossing[1], 0.25 / 0.85)
    close(crossing[2], 0.0)
    rows = []
    for name, n, value in (("a.pt", 1, 0.2), ("b.pt", 3, 0.8)):
        samples = [dict(overlap=value, top1_match=1, pentropy=0.5, qentropy=0.7)] * n
        stats = _stats(samples, n * 2)
        rows.append(dict(name=name, split="train", kind="feature", **stats,
                         ce_sum=0, target_ce_sum=0, target_raw_entropy_sum=0))
    current = _pool(rows)
    old = evaluation._summarize_types(rows)["train"]["feature"]
    for field in ("sample_count", "token_count", "pooled_overlap", "equal_trajectory_mean_overlap",
                  "top1_match", "pentropy", "qentropy"):
        close(current[field], old[field])
    close(current["pooled_overlap"], 0.65)
    close(current["equal_trajectory_mean_overlap"], 0.5)
    evaluation._require(_pool([_stats([], 0)])["pooled_overlap"] is None, "Empty mean must be null")
    json.dumps(current, allow_nan=False)
    return {"self_test": "passed", "device": "CPU", "file_writes": 0,
            "coverage": ["normalized vectors and sparse Target", "dense / sparse overlap equivalence",
                         "identical distributions overlap one", "14 Feature policies", "nucleus crossing token",
                         "zero support", "original reducer equivalence", "unequal sample counts", "empty means"],
            "real_checkpoint_comparison": "Performed during evaluation when saved original metrics exist"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--checkpoint", type=Path)
    action.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    else:
        if args.source_root is None or args.evaluation_root is None or args.output is None:
            parser.error("--source-root, --evaluation-root and --output are required with --checkpoint")
        result = evaluate_checkpoint(args.checkpoint, args.source_root, args.evaluation_root)
        from draft_a.corpus_v2 import write_json
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
