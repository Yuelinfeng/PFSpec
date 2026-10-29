"""Teacher-prefix P/F learning curves; cache labels, never model weights.

Run from the GaussianGPT repository root. Relative source_root paths use that
working directory and point to the original run containing corpus/ and plan.json.
Call require_complete_cache before training; limited runs are explicitly partial.
CE and raw entropy use natural logarithms and are vocabulary-dependent.
"""

import argparse
from contextlib import contextmanager
import gc
import importlib
import json
import math
from pathlib import Path


EOS = 12096
POSITION_END = 8000
KINDS = ("all", "position", "feature", "eos")
SCOPE = "teacher-prefix: full natural trajectory, no generated Draft prefix"
POLICY = {"temperature": 0.9, "top_p": 0.9,
          "position": "ids > previous position at even prediction slots",
          "nucleus": "descending logits; cumulative > top_p; shift right once"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _limit(max_records):
    _require(max_records is None or
             (type(max_records) is int and max_records > 0),
             "max_records must be a positive integer or None")


def _names(value, count, label):
    _require(isinstance(value, list) and len(value) == count,
             f"{label} must contain {count} filenames")
    _require(all(isinstance(n, str) and Path(n).name == n and
                 n.endswith(".pt") and n not in (".", "..") for n in value),
             f"{label} must contain plain .pt filenames")
    _require(len(set(value)) == count, f"Duplicate filenames in {label}")
    return value


def _plan(source_root, study_root, max_records):
    _limit(max_records)
    study_root = Path(study_root).resolve()
    plan = json.loads((study_root / "plan.json").read_text())
    _require(isinstance(plan.get("source_root"), str) and plan["source_root"],
             "plan.source_root must be a local filesystem path")
    planned_source = Path(plan["source_root"]).resolve()
    source_root = planned_source if source_root is None else Path(source_root).resolve()
    _require(source_root == planned_source, "source_root differs from plan.source_root")
    temperature = plan.get("temp", plan.get("temperature"))
    _require(plan.get("samples_per_type") == 32 and
             temperature == 0.9 and plan.get("top_p") == 0.9 and
             ("temperature" not in plan or plan["temperature"] == temperature),
             "Expected samples_per_type=32, temp=0.9, top_p=0.9")
    subsets = plan.get("subsets")
    _require(isinstance(subsets, dict), "plan.subsets must be a dictionary")
    train_names = set()
    for size in (100, 200, 400):
        train_names.update(_names(subsets.get(str(size)), size, f"subsets[{size}]"))
    val = _names(plan.get("val_files"), 50, "val_files")
    train = _names(plan.get("train_eval_files"), 32, "train_eval_files")
    _require(not set(val).intersection(train_names | set(train)),
             "Train and validation filenames must be disjoint")
    _require(set(train).issubset(set(subsets["400"])),
             "train_eval_files must belong to subsets[400]")
    source_plan = json.loads((source_root / "plan.json").read_text())
    entries = source_plan.get("entries")
    _require(isinstance(entries, list), "Source plan must contain entries")
    by_name = {}
    for entry in entries:
        _require(entry["name"] not in by_name, "Duplicate source entry name")
        by_name[entry["name"]] = entry
    for name in train_names | set(train) | set(val):
        expected = "val" if name in set(val) else "train"
        _require(name in by_name and by_name[name]["split"] == expected,
                 f"Source split mismatch: {name}")
    expected = {"val": val, "train": train}
    selected = {s: names if max_records is None else names[:max_records]
                for s, names in expected.items()}
    return source_root, study_root, expected, selected, by_name


def _helpers():
    # Deployment lives in draft_a; imports leave the original modules intact.
    corpus = importlib.import_module("draft_a.corpus_v2")
    train = importlib.import_module("draft_a.train_v2")
    try:
        overlap = importlib.import_module("draft_a.eval_overlap")
    except ModuleNotFoundError as error:
        if error.name != "draft_a.eval_overlap":
            raise
        overlap = importlib.import_module("eval_overlap")
    analysis = importlib.import_module("draft_a.analyze_v2")
    _require(corpus.EOS == EOS, "Unexpected EOS id")
    return corpus, train, overlap, analysis


def _records(source_root, selected, by_name, analysis):
    result = {}
    seen = {}
    for split, names in selected.items():
        result[split] = analysis.read_records(source_root, [by_name[n] for n in names])
        for record in result[split]:
            key = (tuple(record["tokens"].tolist()), record["has_eos"])
            _require(key not in seen or seen[key] == split,
                     "Identical trajectory occurs in train and validation")
            seen[key] = split
    return result


def _kind(token):
    if token == EOS:
        return "eos"
    if 0 <= token < POSITION_END:
        return "position"
    if POSITION_END <= token < EOS:
        return "feature"
    raise ValueError(f"Invalid label id: {token}")


def _slots(body_length, has_eos, count, choose_slots):
    slots = list(choose_slots(body_length, count))
    if has_eos:
        slots.append(body_length)
    _require(slots == sorted(set(slots)), "Slots must be sorted and unique")
    return slots


def _meta(record, seq, overlap):
    body_length = len(record["tokens"])
    slots = _slots(body_length, record["has_eos"], 32, overlap.choose_slots)
    gold = [int(seq[t + 1]) for t in slots]
    kinds = [_kind(g) for g in gold]
    _require(all(k == ("position" if t % 2 == 0 else "feature")
                 for t, k in zip(slots, kinds) if k != "eos"),
             "Token type differs from prediction parity")
    return {"name": record["name"], "split": record["split"],
            "context": len(seq) - 1, "body_tokens": body_length,
            "has_eos": record["has_eos"], "slots": slots,
            "sampled_depths": [t + 1 for t in slots], "kinds": kinds,
            "gold": gold,
            "prev_pos": [int(seq[t - 1]) if t % 2 == 0 and t >= 2 else None
                         for t in slots],
            "samples_per_type": 32, "policy": dict(POLICY),
            "target_precision": "FP32", "scope": SCOPE}


@contextmanager
def _fp32_runtime(torch):
    _require(torch.cuda.is_available(), "CUDA is required for model evaluation")
    _require(not torch.is_autocast_enabled(), "Call outside an autocast context")
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        gc.collect()
        torch.cuda.empty_cache()


def _model_ready(model, context, torch):
    _require(model.eos_token_id == EOS, "Model EOS id differs")
    _require(all(p.dtype == torch.float32 for p in model.parameters()
                 if p.is_floating_point()), "Model parameters must be FP32")
    _require(all(b.dtype == torch.float32 for b in model.buffers()
                 if b.is_floating_point()), "Model buffers must be FP32")
    cfg = getattr(model, "config", None)
    capacity = getattr(model, "n_ctx", None)
    if capacity is None and cfg is not None:
        capacity = getattr(cfg, "n_ctx", None)
    if capacity is not None:
        _require(context <= capacity, "Full trajectory exceeds model context")


def _mean(total, count):
    return total / count if count else None


def _entropy(probabilities, torch):
    nonzero = probabilities[probabilities > 0]
    return float(-(nonzero * nonzero.log()).sum())


def _valid_sampling_support(width, extra_zero):
    _require(width >= EOS + 1, "Sampling vocabulary must include EOS")
    _require(extra_zero, "PAD and extra vocabulary ids must have zero sampling probability")


def _valid_sampling_probs(probabilities, torch):
    _require(probabilities.dtype == torch.float32 and probabilities.ndim == 1,
             "Sampling probabilities must be an FP32 vector")
    _require(bool(torch.isfinite(probabilities).all()) and
             bool((probabilities >= 0).all()) and
             abs(float(probabilities.double().sum()) - 1.0) < 1e-4,
             "Sampling probabilities must be finite, nonnegative and sum to one")
    _valid_sampling_support(len(probabilities), bool((probabilities[EOS + 1:] == 0).all()))


def _raw_metrics(logits, labels, torch, include_entropy):
    import torch.nn.functional as F

    _require(logits.dtype == torch.float32 and logits.ndim == 2 and
             logits.shape[-1] >= EOS + 1,
             "Expected FP32 logits with at least 12097 vocabulary entries")
    _require(labels.dtype == torch.long and labels.ndim == 1 and
             len(labels) == logits.shape[0] and
             bool(((labels >= 0) & (labels <= EOS)).all()),
             "Labels must contain only position, feature or EOS ids")
    losses = log_probs = entropy = None
    try:
        losses = F.cross_entropy(logits.float(), labels, reduction="none")
        _require(bool(torch.isfinite(losses).all()), "Nonfinite full-token loss")
        ce = losses.cpu()
        raw_entropy = torch.empty(len(labels), dtype=torch.float32) if include_entropy else None
        if include_entropy:
            for start in range(0, len(labels), 128):
                log_probs = torch.log_softmax(logits[start:start + 128].float(), dim=-1)
                entropy = -torch.where(torch.isfinite(log_probs),
                                       log_probs.exp() * log_probs, 0.0).sum(-1)
                _require(bool(torch.isfinite(entropy).all()), "Nonfinite raw entropy")
                raw_entropy[start:start + len(entropy)] = entropy.cpu()
        y = labels.cpu()
        masks = {"all": torch.ones_like(y, dtype=torch.bool),
                 "position": (y >= 0) & (y < POSITION_END),
                 "feature": (y >= POSITION_END) & (y < EOS), "eos": y == EOS}
        result = {}
        for kind, mask in masks.items():
            count = int(mask.sum())
            loss_sum = float(ce[mask].double().sum())
            entropy_sum = float(raw_entropy[mask].double().sum()) if include_entropy else None
            result[kind] = {"token_count": count, "ce_sum": loss_sum,
                            "ce": _mean(loss_sum, count),
                            "raw_entropy_sum": entropy_sum,
                            "raw_entropy": _mean(entropy_sum, count) if include_entropy else None}
        return result
    finally:
        losses = log_probs = entropy = None


def _teacher_labels(target, record, seq, meta, overlap, torch):
    x = logits = labels = None
    try:
        _model_ready(target, meta["context"], torch)
        x = seq[:-1].unsqueeze(0).cuda()
        labels = seq[1:].cuda()
        logits = target(input_ids=x)[0]
        _require(logits.shape[0] == meta["context"], "Target context differs")
        raw = _raw_metrics(logits, labels, torch, True)
        rows = logits[meta["slots"]].float().cpu()
        indices, probabilities, pentropy, top1 = [], [], [], []
        for row, prev in zip(rows, meta["prev_pos"]):
            p = overlap.sampling_probs(row, prev, 0.9, 0.9)
            _valid_sampling_probs(p, torch)
            ids = torch.nonzero(p > 0, as_tuple=True)[0].long()
            indices.append(ids)
            probabilities.append(p[ids].float())
            pentropy.append(_entropy(p, torch))
            top1.append(int(p.argmax()))
        return {"metadata": meta, "indices": indices, "probabilities": probabilities,
                "pentropy": pentropy, "target_top1": top1, "target_raw": raw}
    finally:
        x = logits = labels = None


def _valid_cache(cache, meta, torch):
    _require(isinstance(cache, dict) and cache.get("metadata") == meta,
             f"Cache structure differs: {meta['name']}")
    count = len(meta["slots"])
    for key in ("indices", "probabilities", "pentropy", "target_top1"):
        _require(isinstance(cache.get(key), list) and len(cache[key]) == count,
                 f"Cache row count differs: {meta['name']}/{key}")
    for j, (ids, p, prev) in enumerate(zip(cache["indices"], cache["probabilities"],
                                         meta["prev_pos"])):
        _require(isinstance(ids, torch.Tensor) and isinstance(p, torch.Tensor),
                 "Sparse labels must be tensors")
        _require(ids.dtype == torch.long and p.dtype == torch.float32 and
                 ids.ndim == p.ndim == 1 and len(ids) == len(p) and len(ids) > 0,
                 "Sparse ids must be int64; probabilities must be FP32")
        _require(bool(((ids >= 0) & (ids <= EOS)).all()) and
                 bool((ids[1:] > ids[:-1]).all()), "Sparse ids must be sorted and unique")
        _require(bool(torch.isfinite(p).all()) and bool((p > 0).all()) and
                 abs(float(p.double().sum()) - 1.0) < 1e-4,
                 "Sparse probability rows must be finite, positive and sum to one")
        if prev is not None:
            _require(0 <= prev < POSITION_END and not bool((ids <= prev).any()),
                     "Sparse row violates position monotonicity")
        _require(math.isclose(cache["pentropy"][j], _entropy(p, torch),
                              rel_tol=1e-5, abs_tol=1e-5), "Cached p entropy differs")
        _require(cache["target_top1"][j] == int(ids[p.argmax()]),
                 "Cached Target top1 differs")
    raw = cache.get("target_raw", {})
    body = meta["body_tokens"]
    counts = {"all": meta["context"], "position": body // 2,
              "feature": body // 2, "eos": int(meta["has_eos"])}
    for kind, n in counts.items():
        item = raw.get(kind, {})
        _require(item.get("token_count") == n, "Cached token counts differ")
        for value_key, sum_key in (("ce", "ce_sum"), ("raw_entropy", "raw_entropy_sum")):
            total, value = item.get(sum_key), item.get(value_key)
            _require(isinstance(total, (int, float)) and math.isfinite(total) and total >= 0,
                     "Cached raw metric must be finite and nonnegative")
            _require((value is None and n == 0 and total == 0) or
                     (n > 0 and isinstance(value, (int, float)) and math.isfinite(value) and
                      math.isclose(value, total / n, rel_tol=1e-6, abs_tol=1e-6)),
                     "Cached raw mean differs")
    for sum_key in ("ce_sum", "raw_entropy_sum"):
        _require(math.isclose(raw["all"][sum_key],
                              sum(raw[k][sum_key] for k in KINDS[1:]),
                              rel_tol=1e-6, abs_tol=1e-6), "Cached type totals differ")
    return cache


def _cache_path(study_root, split, name):
    return study_root / "teacher_cache" / split / name


def _summary(source_root, expected, selected, records, max_records):
    splits = {}
    for split, names in selected.items():
        contexts = [len(r["tokens"]) + int(r["has_eos"]) for r in records[split]]
        splits[split] = {"record_count": len(names), "expected_record_count": len(expected[split]),
                         "files": list(names),
                         "context_range": [min(contexts), max(contexts)] if contexts else None}
    return {"source_root": str(source_root), "target_precision": "FP32", "scope": SCOPE,
            "samples_per_type": 32, "policy": dict(POLICY), "max_records": max_records,
            "limited": max_records is not None,
            "complete": max_records is None and selected == expected,
            "record_count": sum(len(n) for n in selected.values()), "splits": splits}


def _save_labels(path, cache, torch):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pt.tmp")
    try:
        torch.save(cache, temp)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def prepare_cache(source_root, study_root, max_records=None):
    """Load Target once in FP32, reusing structurally compatible sparse labels."""
    import torch

    source_root, study_root, expected, selected, entries = _plan(source_root, study_root, max_records)
    corpus, train, overlap, analysis = _helpers()
    records = _records(source_root, selected, entries, analysis)
    summary_path = study_root / "teacher_cache" / "summary.json"
    summary = _summary(source_root, expected, selected, records, max_records)
    # A failed or limited attempt can never advertise a completed full run.
    incomplete = dict(summary, complete=False, status="preparing", completed_record_count=0)
    corpus.write_json(summary_path, incomplete)
    reused = created = 0
    target = None
    try:
        with _fp32_runtime(torch):
            try:
                for split, rows in records.items():
                    for record in rows:
                        seq = train.sequence(record)
                        meta = _meta(record, seq, overlap)
                        path = _cache_path(study_root, split, record["name"])
                        if path.exists():
                            cache = torch.load(path, map_location="cpu", weights_only=True)
                            _valid_cache(cache, meta, torch)
                            reused += 1
                        else:
                            if target is None:
                                target = corpus.load_target().float().eval().requires_grad_(False)
                            cache = _teacher_labels(target, record, seq, meta, overlap, torch)
                            _valid_cache(cache, meta, torch)
                            _save_labels(path, cache, torch)
                            created += 1
            finally:
                target = None
    except Exception:
        incomplete.update(status="failed", completed_record_count=reused + created)
        corpus.write_json(summary_path, incomplete)
        raise
    summary.update(status="ready", completed_record_count=reused + created,
                   created_record_count=created, reused_record_count=reused)
    corpus.write_json(summary_path, summary)
    return summary


def _load_labels(source_root, study_root, expected, selected, records, train, overlap, torch,
                 require_complete):
    summary = json.loads((study_root / "teacher_cache" / "summary.json").read_text())
    _require(summary.get("source_root") == str(source_root) and
             summary.get("target_precision") == "FP32" and summary.get("scope") == SCOPE and
             summary.get("samples_per_type") == 32 and summary.get("policy") == POLICY and
             summary.get("status") == "ready", "Cache summary configuration differs")
    if require_complete:
        _require(summary.get("complete") is True and summary.get("limited") is False and
                 summary.get("max_records") is None and summary.get("record_count") == 82 and
                 summary.get("completed_record_count") == 82,
                 "Full 50-val/32-train cache is required before training or full evaluation")
    result = {}
    for split, names in selected.items():
        split_summary = summary.get("splits", {}).get(split, {})
        summary_names = split_summary.get("files", [])
        _require(summary_names == expected[split][:len(summary_names)] and
                 len(summary_names) == split_summary.get("record_count") and
                 split_summary.get("expected_record_count") == len(expected[split]),
                 "Cached file range differs from the plan")
        _require(summary_names[:len(names)] == names and
                 (not require_complete or summary_names == expected[split]),
                 "Requested files exceed the prepared cache range")
        result[split] = []
        for record in records[split]:
            seq = train.sequence(record)
            meta = _meta(record, seq, overlap)
            cache = torch.load(_cache_path(study_root, split, record["name"]),
                               map_location="cpu", weights_only=True)
            result[split].append(_valid_cache(cache, meta, torch))
    return result, summary


def require_complete_cache(source_root, study_root):
    """Trainer entry point before the first update; never loads either model."""
    import torch

    source_root, study_root, expected, selected, entries = _plan(source_root, study_root, None)
    _, train, overlap, analysis = _helpers()
    records = _records(source_root, selected, entries, analysis)
    _, summary = _load_labels(source_root, study_root, expected, selected, records,
                              train, overlap, torch, True)
    return summary


def _student_sequence(draft, seq, cache, overlap, torch):
    x = logits = labels = None
    try:
        meta = cache["metadata"]
        _model_ready(draft, meta["context"], torch)
        x = seq[:-1].unsqueeze(0).cuda()
        labels = seq[1:].cuda()
        logits = draft(input_ids=x)[0]
        _require(logits.shape[0] == meta["context"], "Draft context differs")
        raw = _raw_metrics(logits, labels, torch, False)
        rows = logits[meta["slots"]].float().cpu()
        sampled = []
        for j, (row, prev) in enumerate(zip(rows, meta["prev_pos"])):
            q = overlap.sampling_probs(row, prev, 0.9, 0.9)
            _valid_sampling_probs(q, torch)
            p = cache["probabilities"][j]
            ids = cache["indices"][j]
            value = float(torch.minimum(q[ids], p).sum())
            _require(math.isfinite(value) and -1e-6 <= value <= 1 + 1e-5,
                     "Invalid probability overlap")
            sampled.append({"kind": meta["kinds"][j], "overlap": value,
                            "top1_match": int(int(q.argmax()) == cache["target_top1"][j]),
                            "pentropy": cache["pentropy"][j], "qentropy": _entropy(q, torch)})
        result = []
        for kind in KINDS:
            values = [v for v in sampled if kind == "all" or v["kind"] == kind]
            n = len(values)
            item = {"split": meta["split"], "name": meta["name"], "kind": kind,
                    "context": meta["context"], "sample_count": n,
                    "token_count": raw[kind]["token_count"],
                    "ce_sum": raw[kind]["ce_sum"], "ce": raw[kind]["ce"],
                    "target_ce_sum": cache["target_raw"][kind]["ce_sum"],
                    "target_ce": cache["target_raw"][kind]["ce"],
                    "target_raw_entropy_sum": cache["target_raw"][kind]["raw_entropy_sum"],
                    "target_raw_entropy": cache["target_raw"][kind]["raw_entropy"]}
            for metric in ("overlap", "top1_match", "pentropy", "qentropy"):
                total = math.fsum(v[metric] for v in values)
                item[metric + "_sum"] = total
                item[metric] = _mean(total, n)
            result.append(item)
        return result
    finally:
        x = logits = labels = None


def _summarize_types(per_sequence):
    result = {}
    for split in ("val", "train"):
        result[split] = {}
        for kind in KINDS:
            rows = [r for r in per_sequence if r["split"] == split and r["kind"] == kind]
            token_count = sum(r["token_count"] for r in rows)
            sample_count = sum(r["sample_count"] for r in rows)
            trajectories = [r["overlap"] for r in rows if r["sample_count"]]
            item = {"sequence_count": len(rows), "token_count": token_count,
                    "sample_count": sample_count, "sampled_sequence_count": len(trajectories),
                    "token_weighted_ce": _mean(math.fsum(r["ce_sum"] for r in rows), token_count),
                    "target_token_weighted_ce": _mean(math.fsum(r["target_ce_sum"] for r in rows),
                                                      token_count),
                    "target_token_weighted_raw_entropy": _mean(
                        math.fsum(r["target_raw_entropy_sum"] for r in rows), token_count),
                    "pooled_overlap": _mean(math.fsum(r["overlap_sum"] for r in rows), sample_count),
                    "equal_trajectory_mean_overlap": _mean(math.fsum(trajectories), len(trajectories))}
            for metric in ("top1_match", "pentropy", "qentropy"):
                item[metric] = _mean(math.fsum(r[metric + "_sum"] for r in rows), sample_count)
            result[split][kind] = item
    return result


def evaluate_checkpoint(checkpoint: Path, source_root: Path, study_root: Path,
                        max_records=None) -> dict:
    """Evaluate one budget point with Draft only, then release its GPU tensors."""
    import torch
    from omegaconf import OmegaConf

    source_root, study_root, expected, selected, entries = _plan(source_root, study_root, max_records)
    _, train, overlap, analysis = _helpers()
    records = _records(source_root, selected, entries, analysis)
    caches, summary = _load_labels(source_root, study_root, expected, selected, records,
                                   train, overlap, torch, max_records is None)
    draft = state = None
    per_sequence = []
    step = None
    with _fp32_runtime(torch):
        try:
            state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
            cfg = OmegaConf.create(state["config"])
            _require(all(c["metadata"]["context"] <= cfg.n_ctx
                         for rows in caches.values() for c in rows),
                     "Full trajectory exceeds Draft context")
            draft = train.make_model(cfg).float()
            draft.load_state_dict(state["model"], strict=True)
            step = state.get("step")
            draft.eval().requires_grad_(False)
            state = None
            for split, rows in records.items():
                for record, cache in zip(rows, caches[split]):
                    per_sequence.extend(_student_sequence(draft, train.sequence(record), cache,
                                                           overlap, torch))
        finally:
            draft = state = None
    result = {"checkpoint": str(Path(checkpoint).resolve()), "checkpoint_step": step,
              "source_root": str(source_root), "target_precision": "FP32",
              "draft_precision": "FP32", "scope": SCOPE, "policy": dict(POLICY),
              "entropy_units": "nats", "ce_definition": "raw full-token cross entropy",
              "comparison_note": "raw_nll and entropy depend on vocabulary size; compare within type",
              "max_records": max_records, "limited": max_records is not None,
              "complete": max_records is None, "teacher_cache_complete": summary["complete"],
              "files": selected, "splits": _summarize_types(per_sequence), "per_sequence": per_sequence}
    json.dumps(result, allow_nan=False)
    return result


def self_test():
    """CPU-only mathematical tests; no torch, repository import, or file writes."""
    def close(a, b):
        _require(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7), f"{a} != {b}")

    def entropy(p):
        return -math.fsum(x * math.log(x) for x in p if x > 0)

    def nucleus(p, threshold):
        order = sorted(range(len(p)), key=lambda i: -p[i])
        cumulative, remove = 0.0, []
        for i in order:
            cumulative += p[i]
            remove.append(cumulative > threshold)
        remove = [False] + remove[:-1]
        out = [0.0] * len(p)
        for i, excluded in zip(order, remove):
            if not excluded:
                out[i] = p[i]
        total = math.fsum(out)
        return [x / total for x in out]

    p, q = [0.6, 0.3, 0.1, 0.0], [0.2, 0.3, 0.0, 0.5]
    ids = [i for i, x in enumerate(p) if x > 0]
    sparse = [p[i] for i in ids]
    close(math.fsum(sparse), 1.0)
    close(math.fsum(min(q[i], x) for i, x in zip(ids, sparse)), 0.5)
    close(math.fsum(min(a, b) for a, b in zip(p, q)),
          1 - math.fsum(abs(a - b) for a, b in zip(p, q)) / 2)
    close(entropy(sparse), entropy(p))
    close(math.fsum(min(a, b) for a, b in zip(p, p)), 1.0)
    close(math.fsum(min(a, b) for a, b in zip([1, 0], [0, 1])), 0.0)
    close(nucleus([0.2, 0.6, 0.15, 0.05], 0.7)[0], 0.25)
    close(nucleus([0.2, 0.6, 0.15, 0.05], 0.7)[1], 0.75)
    close(nucleus([0.1, 0.9], 0.5)[1], 1.0)
    close(math.fsum(nucleus(p, 1.0)), 1.0)
    # Masked PAD and extra ids preserve full-width softmax and nucleus results.
    for width in (EOS + 1, 12098, 12160):
        logits = [-math.inf] * width
        logits[2], logits[8001], logits[EOS] = math.log(0.6), math.log(0.3), math.log(0.1)
        maximum = max(logits)
        exp_logits = [math.exp(x - maximum) for x in logits]
        total = math.fsum(exp_logits)
        full_probs = [x / total for x in exp_logits]
        sampled_probs = nucleus(full_probs, 0.9)
        _valid_sampling_support(width, all(x == 0 for x in sampled_probs[EOS + 1:]))
        close(math.fsum(sampled_probs), 1.0)
        close(-math.log(full_probs[8001]), -math.log(0.3))
        close(entropy(full_probs), entropy([0.6, 0.3, 0.1]))
        close(sampled_probs[2], nucleus([0.6, 0.3, 0.1], 0.9)[0])
        _require(all(i <= EOS for i, x in enumerate(sampled_probs) if x > 0),
                 "Sparse sampling support must exclude PAD and extra ids")
    for width, extra_zero in ((EOS, True), (12098, False), (12160, False)):
        try:
            _valid_sampling_support(width, extra_zero)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid sampling support was accepted")
    gold = [2, 8001, 5, 8002, EOS]
    kinds = [_kind(t) for t in gold]
    _require(kinds == ["position", "feature", "position", "feature", "eos"],
             "EOS must have its own type")
    close(sum(k == "position" for k in kinds), 2)
    close(sum(k == "eos" for k in kinds), 1)
    # A nonuniform p proves that raw NLL and entropy describe different quantities.
    _require(not math.isclose(-math.log(p[1]), entropy(p)), "NLL differs from entropy")
    # Uniform 2-way and 8-way predictions have identical relative uncertainty.
    close(entropy([0.5] * 2), math.log(2))
    close(entropy([0.125] * 8), math.log(8))
    close(-math.log(0.125) - (-math.log(0.5)), math.log(4))
    legal = [i for i in (0, 2, 3, 7999, 8001, EOS) if i > 2]
    _require(legal == [3, 7999, 8001, EOS], "Monotonic support differs")
    _require(_mean(0, 0) is None, "Absent EOS must yield null")
    class Helpers:
        @staticmethod
        def choose_slots(length, count):
            result = []
            for parity in (0, 1):
                slots = list(range(parity, length, 2))
                n = min(len(slots), count)
                selected = {round(i * (len(slots) - 1) / max(n - 1, 1)) for i in range(n)}
                result.extend(slots[j] for j in sorted(selected))
            return sorted(result)

    close(len(_slots(200, True, 32, Helpers.choose_slots)), 65)
    _require(_slots(4, True, 32, Helpers.choose_slots) == [0, 1, 2, 3, 4],
             "Natural EOS must be appended as the last slot")
    _require(_slots(4, False, 32, Helpers.choose_slots) == [0, 1, 2, 3],
             "Do not synthesize EOS")
    # Exercise the reporting reducer with unequal trajectory lengths.
    rows = []
    for name, tokens, samples, overlap in (("a.pt", 2, 1, 0.2), ("b.pt", 6, 3, 0.8)):
        rows.append({"split": "val", "name": name, "kind": "position",
                     "token_count": tokens, "sample_count": samples,
                     "ce_sum": tokens * (1 if name == "a.pt" else 3),
                     "target_ce_sum": tokens, "target_raw_entropy_sum": tokens,
                     "overlap": overlap, "overlap_sum": overlap * samples,
                     "top1_match_sum": samples, "pentropy_sum": samples,
                     "qentropy_sum": samples})
    value = _summarize_types(rows)["val"]["position"]
    close(value["token_weighted_ce"], 2.5)
    close(value["pooled_overlap"], 0.65)
    close(value["equal_trajectory_mean_overlap"], 0.5)
    _require(_summarize_types(rows)["val"]["eos"]["pooled_overlap"] is None,
             "Missing EOS mean must be null")
    json.dumps(_summarize_types(rows), allow_nan=False)
    return {"self_test": "passed", "device": "CPU", "file_writes": 0,
            "coverage": ["sparse overlap", "nucleus shift", "masked PAD and variable width",
                         "typed EOS", "monotonic support",
                         "weighted CE", "trajectory means", "null EOS", "vocabulary-dependent NLL/entropy"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-cache", action="store_true")
    action.add_argument("--checkpoint", type=Path)
    action.add_argument("--self-test", action="store_true")
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--max-records", type=int,
                        help="First N files in each split; labels and results remain partial")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    else:
        if args.study_root is None:
            parser.error("--study-root is required")
        if args.prepare_cache:
            result = prepare_cache(args.source_root, args.study_root, args.max_records)
        else:
            if args.output is None:
                parser.error("--output is required with --checkpoint")
            result = evaluate_checkpoint(args.checkpoint, args.source_root,
                                         args.study_root, args.max_records)
            corpus = importlib.import_module("draft_a.corpus_v2")
            corpus.write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
