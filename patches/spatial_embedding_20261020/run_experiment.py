#!/usr/bin/env python3
"""Run the fixed CPU-only spatial Feature proposal comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time


FEATURES = 4096
FEATURE_OFFSET = 8000
EOS_ID = 12096
METHODS = ("S1", "S2", "S3", "S4")
SEEDS = (17, 29)
TRAIN_STEPS = 150
BATCH_SIZE = 32
L2_WEIGHT = 1e-4
RESIDUAL_MULTIPLIER = 1.0
TEMPERATURE = 0.9
TOP_P = 0.9
S4_SEED = 32043
PCA_WIDTH = 32
SLOT_COUNT = 6
HIDDEN_WIDTH = 32
FIT_SECONDS_SOFT_LIMIT = 720.0
RUN_SECONDS_LIMIT = 1200.0


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _write_progress(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    _write_json(temporary, value)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def _load_policy_module():
    path = Path(__file__).resolve().with_name("pf_round03_feature.py")
    if not path.exists():
        root = Path(__file__).resolve().parents[2]
        path = root / "patches" / "pf2x_round01" / "pf_round03_feature.py"
    spec = importlib.util.spec_from_file_location("pf_round03_feature", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load project proposal policy: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def _validate_manifest(manifest: dict) -> dict[str, list[str]]:
    source_ids = manifest.get("source_ids")
    splits = manifest.get("splits")
    if manifest.get("schema_version") != 1 or not isinstance(source_ids, list):
        raise ValueError("Expected source manifest schema_version=1")
    if len(source_ids) != 10 or source_ids != sorted(set(source_ids)):
        raise ValueError("Source IDs must be ten unique lexicographically sorted values")
    if not isinstance(splits, dict) or set(splits) != {"fit", "tune", "hold"}:
        raise ValueError("Source manifest must contain fit, tune, and hold")
    expected = {"fit": 6, "tune": 2, "hold": 2}
    result = {}
    offset = 0
    for split, count in expected.items():
        ids = splits[split]
        if not isinstance(ids, list) or len(ids) != count:
            raise ValueError(f"{split} must contain {count} sources")
        if ids != source_ids[offset:offset + count]:
            raise ValueError(f"{split} does not follow the manifest source ordering")
        result[split] = list(ids)
        offset += count
    if len(set(sum(result.values(), []))) != 10:
        raise ValueError("Source split contains repeated IDs")
    return result


def _finite_matrix(np, value, shape: tuple[int, ...], name: str):
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _load_data(np, data_dir: Path, manifest_path: Path) -> dict:
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = _validate_manifest(manifest)
    expected_metadata = {
        "status": "complete",
        "embedding_key": "gpt._orig_mod.transformer.transformer.wte.weight",
        "target_embedding_shape": [12160, 1024],
        "feature_embedding_slice": [8000, 12096],
        "feature_embedding_shape": [4096, 1024],
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Prepared metadata field {key} differs from {expected!r}")
    if metadata.get("policies") != {"temperature": 0.9, "top_p": 0.9}:
        raise ValueError("Prepared policy must use temperature=.9 and top_p=.9")
    if metadata.get("target_softcap") != 15.0:
        raise ValueError("Prepared logits metadata must report the original Target softcap")

    rows = _read_jsonl(data_dir / "rows.jsonl")
    if len(rows) != 480 or metadata.get("row_count") != 480:
        raise ValueError("Expected exactly 480 prepared R1 records")
    indices = [int(row["vector_index"]) for row in rows]
    if indices != list(range(480)):
        raise ValueError("Prepared rows must retain vector_index order")
    if len({row.get("sample_id") for row in rows}) != 480:
        raise ValueError("Prepared sample IDs must be unique")
    if any(row.get("source_id") != row.get("scene") for row in rows):
        raise ValueError("Prepared source_id and scene differ")
    if {row.get("source_id") for row in rows} != set(manifest["source_ids"]):
        raise ValueError("Prepared sources differ from the saved source manifest")
    if any(row.get("p1_accepted_actual") is not True for row in rows):
        raise ValueError("All rows must be actual P1-accepted reservoir contexts")

    with np.load(data_dir / "prepared_data.npz", allow_pickle=False) as archive:
        needed = (
            "feature_embedding", "p_f", "q1", "raw_logits", "vector_index",
            "candidate_p_xyz", "history_pf_pairs", "history_pair_offsets",
            "history_position_ids_flat", "history_feature_global_ids_flat",
            "neighbor_position_ids", "neighbor_feature_local_ids",
            "neighbor_relative_xyz", "neighbor_age_pf_pairs", "neighbor_mask",
            "recent_position_ids", "recent_feature_local_ids",
            "recent_relative_xyz", "recent_age_pf_pairs", "recent_mask",
            "p1_id", "p1_accept_probability", "p1_accepted_actual",
        )
        missing = sorted(set(needed) - set(archive.files))
        if missing:
            raise ValueError(f"prepared_data.npz is missing arrays: {missing}")
        arrays = {key: np.array(archive[key], copy=True) for key in needed}

    if not np.array_equal(arrays["vector_index"], np.arange(480, dtype=arrays["vector_index"].dtype)):
        raise ValueError("NPZ vector order differs from rows.jsonl")
    arrays["feature_embedding"] = _finite_matrix(
        np, arrays["feature_embedding"], (4096, 1024), "feature_embedding"
    ).astype(np.float32, copy=False)
    for name in ("p_f", "q1", "raw_logits"):
        arrays[name] = _finite_matrix(np, arrays[name], (480, FEATURES), name)
    for name, shape in (
        ("candidate_p_xyz", (480, 3)), ("neighbor_position_ids", (480, 6)),
        ("neighbor_feature_local_ids", (480, 6)), ("neighbor_relative_xyz", (480, 6, 3)),
        ("neighbor_age_pf_pairs", (480, 6)), ("neighbor_mask", (480, 6)),
        ("recent_position_ids", (480, 6)), ("recent_feature_local_ids", (480, 6)),
        ("recent_relative_xyz", (480, 6, 3)), ("recent_age_pf_pairs", (480, 6)),
        ("recent_mask", (480, 6)), ("history_pf_pairs", (480,)),
        ("p1_id", (480,)), ("p1_accept_probability", (480,)),
        ("p1_accepted_actual", (480,)),
    ):
        arrays[name] = _finite_matrix(np, arrays[name], shape, name)
    if arrays["history_pair_offsets"].shape != (481,):
        raise ValueError("history_pair_offsets must have 481 values")
    if arrays["history_position_ids_flat"].ndim != 1 or arrays["history_feature_global_ids_flat"].ndim != 1:
        raise ValueError("Flattened history arrays must be one-dimensional")
    if len(arrays["history_position_ids_flat"]) != len(arrays["history_feature_global_ids_flat"]):
        raise ValueError("Flattened history position and Feature arrays differ in length")
    if int(arrays["history_pair_offsets"][0]) != 0 or \
            int(arrays["history_pair_offsets"][-1]) != len(arrays["history_position_ids_flat"]):
        raise ValueError("History offsets do not span the flattened history arrays")
    if np.any(np.diff(arrays["history_pair_offsets"]) != arrays["history_pf_pairs"]):
        raise ValueError("History offsets differ from recorded complete pair counts")
    if not np.array_equal(arrays["p1_accepted_actual"], np.ones(480, dtype=bool)):
        raise ValueError("Prepared p1_accepted_actual must be true for all reservoir rows")
    if not np.isfinite(arrays["p1_accept_probability"]).all() or \
            np.any(arrays["p1_accept_probability"] < 0.0) or \
            np.any(arrays["p1_accept_probability"] > 1.0):
        raise ValueError("p1_accept_probability must be finite and in [0,1]")
    for name in ("p_f", "q1"):
        if np.any(arrays[name] < 0.0):
            raise ValueError(f"{name} contains a negative probability")
    p_mass = arrays["p_f"].astype(np.float64).sum(axis=1)
    q_mass = arrays["q1"].astype(np.float64).sum(axis=1)
    if np.any(p_mass <= 0.0) or np.any(q_mass <= 0.0):
        raise ValueError("Every Target and proposal row must have positive mass")
    p_norm = arrays["p_f"].astype(np.float64) / p_mass[:, None]
    q_norm = arrays["q1"].astype(np.float64) / q_mass[:, None]

    scene_ids = [str(row["source_id"]) for row in rows]
    source_to_split = {source: split for split, ids in splits.items() for source in ids}
    split_indices = {
        split: [index for index, source in enumerate(scene_ids) if source_to_split[source] == split]
        for split in ("fit", "tune", "hold")
    }
    for split, indices_for_split in split_indices.items():
        counts = {source: sum(scene_ids[index] == source for index in indices_for_split)
                  for source in splits[split]}
        if any(count != 48 for count in counts.values()):
            raise ValueError(f"{split} source counts differ from 48 rows each: {counts}")
    return {
        "metadata": metadata, "manifest": manifest, "splits": splits,
        "rows": rows, "arrays": arrays, "p_norm": p_norm, "q_norm": q_norm,
        "p_mass": p_mass, "q_mass": q_mass, "scene_ids": scene_ids,
        "split_indices": split_indices,
    }


def _pca_embedding(torch, embedding_np):
    source = torch.as_tensor(embedding_np, dtype=torch.float64, device="cpu")
    mean = source.mean(dim=0)
    centered = source - mean
    covariance = centered.T @ centered / float(source.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)[:PCA_WIDTH]
    eigenvalues = eigenvalues.index_select(0, order)
    components = eigenvectors.index_select(1, order)
    for column in range(components.shape[1]):
        pivot = torch.argmax(torch.abs(components[:, column]))
        if components[pivot, column] < 0:
            components[:, column].neg_()
    pca_raw = centered @ components
    global_rms = torch.sqrt(torch.mean(pca_raw.square()))
    if not torch.isfinite(global_rms) or float(global_rms) <= 0.0:
        raise ValueError("PCA embedding global RMS is invalid")
    normalized = pca_raw / global_rms
    return {
        "mean": mean.cpu().numpy(),
        "components": components.cpu().numpy(),
        "eigenvalues": eigenvalues.cpu().numpy(),
        "pca_raw": pca_raw.cpu().numpy().astype("float32"),
        "pca_embedding": normalized.cpu().numpy().astype("float32"),
        "global_rms": float(global_rms),
        "covariance_shape": list(covariance.shape),
        "input_rows": int(source.shape[0]),
        "input_width": int(source.shape[1]),
        "input_scope": "Target Feature embedding rows 8000:12096 only; no p_f, q1, fit, tune, or hold labels",
    }


def _logit_q_statistics(np, raw_logits, q_norm):
    result = np.zeros((len(raw_logits), 8), dtype=np.float64)
    for row_index, (logits, q) in enumerate(zip(raw_logits, q_norm)):
        top_logits = np.partition(logits.astype(np.float64), -2)[-2:]
        top_q = np.partition(q, -8)[-8:]
        positive = q > 0.0
        entropy = -float(np.sum(q[positive] * np.log(q[positive]), dtype=np.float64))
        result[row_index] = (
            float(np.mean(logits, dtype=np.float64)),
            float(np.std(logits, dtype=np.float64)),
            float(np.max(logits)),
            float(np.min(logits)),
            float(top_logits.max() - top_logits.min()),
            entropy / math.log(FEATURES),
            float(np.max(q)),
            float(np.sum(top_q, dtype=np.float64)),
        )
    return result


def _standardize_fit_stats(np, stats, fit_indices):
    mean = np.mean(stats[fit_indices], axis=0, dtype=np.float64)
    std = np.std(stats[fit_indices], axis=0, dtype=np.float64)
    safe_std = np.where(std > 1e-8, std, 1.0)
    return ((stats - mean) / safe_std).astype(np.float32), mean, safe_std


def _feature_slots(np, arrays, pca_embedding, rows):
    count = len(rows)
    embedded = pca_embedding
    recent_ids = arrays["recent_feature_local_ids"].astype(np.int64)
    recent_mask = arrays["recent_mask"].astype(bool)
    recent_safe = np.maximum(recent_ids, 0)
    recent_e = embedded[recent_safe].copy()
    recent_e[~recent_mask] = 0.0
    recent_delta = arrays["recent_relative_xyz"].astype(np.float32) / 19.0
    recent_delta[~recent_mask] = 0.0
    history_length = np.maximum(arrays["history_pf_pairs"].astype(np.float32), 1.0)
    recent_age = arrays["recent_age_pf_pairs"].astype(np.float32) / history_length[:, None]
    recent_age[~recent_mask] = 0.0
    recent_slots = np.concatenate((recent_e, recent_delta, recent_age[..., None]), axis=-1)
    recent_slots[~recent_mask] = 0.0

    neighbor_ids = arrays["neighbor_feature_local_ids"].astype(np.int64)
    neighbor_mask = arrays["neighbor_mask"].astype(bool)
    neighbor_safe = np.maximum(neighbor_ids, 0)
    neighbor_e = embedded[neighbor_safe].copy()
    neighbor_e[~neighbor_mask] = 0.0
    neighbor_delta = arrays["neighbor_relative_xyz"].astype(np.float32) / 19.0
    neighbor_delta[~neighbor_mask] = 0.0
    neighbor_age = arrays["neighbor_age_pf_pairs"].astype(np.float32) / history_length[:, None]
    neighbor_age[~neighbor_mask] = 0.0
    neighbor_slots = np.concatenate((neighbor_e, neighbor_delta, neighbor_age[..., None]), axis=-1)
    neighbor_slots[~neighbor_mask] = 0.0

    candidate_xyz = arrays["candidate_p_xyz"].astype(np.int64)
    offsets = arrays["history_pair_offsets"].astype(np.int64)
    history_positions = arrays["history_position_ids_flat"].astype(np.int64)
    history_features = arrays["history_feature_global_ids_flat"].astype(np.int64)
    s4_ids = np.full((count, SLOT_COUNT), -1, dtype=np.int32)
    s4_positions = np.full((count, SLOT_COUNT), -1, dtype=np.int32)
    s4_ages = np.full((count, SLOT_COUNT), -1, dtype=np.int32)
    s4_mask = np.zeros((count, SLOT_COUNT), dtype=bool)
    s4_relative = np.zeros((count, SLOT_COUNT, 3), dtype=np.int8)
    s4_available = np.zeros(count, dtype=np.int32)
    s4_rng = random.Random(S4_SEED)
    for row_index in range(count):
        neighbor_positions = {
            int(value) for value, valid in zip(arrays["neighbor_position_ids"][row_index],
                                                neighbor_mask[row_index]) if valid
        }
        start = int(offsets[row_index])
        end = int(offsets[row_index + 1])
        candidates = []
        for history_index in range(start, end):
            position_id = int(history_positions[history_index])
            if position_id == int(arrays["p1_id"][row_index]):
                raise ValueError("S4 candidate Position is already in the committed history")
            if position_id in neighbor_positions:
                continue
            global_feature_id = int(history_features[history_index])
            if not FEATURE_OFFSET <= global_feature_id < EOS_ID:
                raise ValueError(f"Invalid history Feature ID: {global_feature_id}")
            local_feature_id = global_feature_id - FEATURE_OFFSET
            age = end - start - 1 - (history_index - start)
            x, remainder = divmod(position_id, 400)
            y, z = divmod(remainder, 20)
            delta = (x - int(candidate_xyz[row_index, 0]),
                     y - int(candidate_xyz[row_index, 1]),
                     z - int(candidate_xyz[row_index, 2]))
            candidates.append((position_id, local_feature_id, age, delta))
        active_slots = [slot for slot, valid in enumerate(neighbor_mask[row_index]) if valid]
        k = min(len(active_slots), len(candidates))
        s4_available[row_index] = len(candidates)
        if k == 0:
            continue
        chosen = s4_rng.sample(candidates, k)
        chosen_slots = s4_rng.sample(active_slots, k)
        for slot, item in zip(chosen_slots, chosen):
            position_id, local_id, age, delta = item
            s4_positions[row_index, slot] = position_id
            s4_ids[row_index, slot] = local_id
            s4_ages[row_index, slot] = age
            s4_relative[row_index, slot] = delta
            s4_mask[row_index, slot] = True
    s4_e = embedded[np.maximum(s4_ids, 0)].copy()
    s4_e[~s4_mask] = 0.0
    s4_delta = s4_relative.astype(np.float32) / 19.0
    s4_delta[~s4_mask] = 0.0
    s4_age = np.maximum(s4_ages, 0).astype(np.float32) / history_length[:, None]
    s4_age[~s4_mask] = 0.0
    s4_slots = np.concatenate((s4_e, s4_delta, s4_age[..., None]), axis=-1)
    s4_slots[~s4_mask] = 0.0

    s1_slots = np.zeros((count, SLOT_COUNT, 36), dtype=np.float32)
    s1_mask = np.zeros((count, SLOT_COUNT), dtype=bool)
    return {
        "S1": (s1_slots, s1_mask),
        "S2": (recent_slots.astype(np.float32), recent_mask),
        "S3": (neighbor_slots.astype(np.float32), neighbor_mask),
        "S4": (s4_slots.astype(np.float32), s4_mask),
        "s4_positions": s4_positions,
        "s4_feature_local_ids": s4_ids,
        "s4_ages": s4_ages,
        "s4_relative_xyz": s4_relative,
        "s4_mask": s4_mask,
        "s4_available_count": s4_available,
    }


def _make_neighbor_mixture(np, q1_raw, q1_norm, neighbor_ids, neighbor_mask):
    count = len(q1_raw)
    empirical = np.zeros((count, FEATURES), dtype=np.float64)
    counts = neighbor_mask.sum(axis=1).astype(np.int32)
    for row_index in range(count):
        valid_ids = neighbor_ids[row_index][neighbor_mask[row_index]].astype(np.int64)
        if len(valid_ids) == 0:
            continue
        if np.any(valid_ids < 0) or np.any(valid_ids >= FEATURES):
            raise ValueError("Neighbor Feature ID outside local vocabulary")
        for feature_id in valid_ids:
            empirical[row_index, int(feature_id)] += 1.0 / len(valid_ids)
    mixed = 0.9 * q1_norm + 0.1 * empirical
    q_mix = np.array(mixed, dtype=np.float64, copy=True)
    for row_index in range(count):
        if counts[row_index] == 0:
            q_mix[row_index] = q1_raw[row_index]
        else:
            mass = math.fsum(float(value) for value in q_mix[row_index])
            q_mix[row_index] /= mass
    return q_mix, empirical, counts


def _source_metrics(np, p_norm, q_values, weights, rows, indices):
    per_row = {}
    grouped = {}
    for index in indices:
        q = np.asarray(q_values[index], dtype=np.float64)
        q_mass = float(np.sum(q, dtype=np.float64))
        if not math.isfinite(q_mass) or q_mass <= 0.0:
            raise ValueError(f"Invalid proposal mass at row {index}")
        q_norm = q / q_mass
        overlap = float(np.minimum(p_norm[index], q_norm).sum(dtype=np.float64))
        weight = float(weights[index])
        source = str(rows[index]["source_id"])
        per_row[index] = (overlap, weight * overlap)
        grouped.setdefault(source, []).append((overlap, weight * overlap, weight))
    by_source = {}
    for source, values in sorted(grouped.items()):
        by_source[source] = {
            "row_count": len(values),
            "mean_overlap": statistics.fmean(item[0] for item in values),
            "mean_p1_weight_times_overlap": statistics.fmean(item[1] for item in values),
            "mean_p1_probability": statistics.fmean(item[2] for item in values),
            "conditional_weighted_overlap": (
                math.fsum(item[1] for item in values) / math.fsum(item[2] for item in values)
                if math.fsum(item[2] for item in values) > 0.0 else None
            ),
        }
    return {
        "row_count": len(indices),
        "source_count": len(by_source),
        "source_equal_mean_overlap": statistics.fmean(
            value["mean_overlap"] for value in by_source.values()
        ) if by_source else None,
        "source_equal_mean_p1_weight_times_overlap": statistics.fmean(
            value["mean_p1_weight_times_overlap"] for value in by_source.values()
        ) if by_source else None,
        "source_equal_conditional_weighted_overlap": statistics.fmean(
            value["conditional_weighted_overlap"] for value in by_source.values()
            if value["conditional_weighted_overlap"] is not None
        ) if any(value["conditional_weighted_overlap"] is not None
                 for value in by_source.values()) else None,
        "by_source": by_source,
        "per_row": per_row,
    }


def _write_scores(path, rows, split, method, seed, q_values, p_norm, weights,
                  indices, update_budget):
    import numpy as np
    result = _source_metrics(np, p_norm, q_values, weights, rows, indices)
    with path.open("a", encoding="utf-8") as stream:
        for index in indices:
            overlap, weighted = result["per_row"][index]
            q = q_values[index]
            positive = [float(value) for value in q if value > 0.0]
            record = {
                "vector_index": int(index),
                "sample_id": str(rows[index]["sample_id"]),
                "source_id": str(rows[index]["source_id"]),
                "split": split,
                "method": method,
                "seed": seed,
                "step_budget": int(update_budget),
                "p1_accepted_actual": bool(rows[index]["p1_accepted_actual"]),
                "p1_accept_probability": float(weights[index]),
                "feature_overlap": float(overlap),
                "weighted_overlap": float(weighted),
                "q_mass": math.fsum(float(value) for value in q),
                "q_support": len(positive),
                "q_min_positive": min(positive) if positive else None,
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
    del result["per_row"]
    return result


def _torch_model(torch):
    from torch import nn
    from torch.nn import functional as F

    class FeatureResidual(nn.Module):
        def __init__(self, class_embedding):
            super().__init__()
            self.slot_in = nn.Linear(36, HIDDEN_WIDTH)
            self.slot_hidden = nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH)
            self.attention = nn.Linear(HIDDEN_WIDTH, 1)
            self.context_out = nn.Linear(HIDDEN_WIDTH + 8, HIDDEN_WIDTH)
            self.register_buffer("class_embedding", class_embedding.float(), persistent=False)
            nn.init.zeros_(self.context_out.weight)
            nn.init.zeros_(self.context_out.bias)

        def forward(self, slot_values, slot_mask, stats):
            batch = slot_values.shape[0]
            encoded = F.gelu(self.slot_in(slot_values))
            encoded = F.gelu(self.slot_hidden(encoded))
            scores = self.attention(encoded).squeeze(-1)
            valid = slot_mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~valid, -1e9)
            weights = torch.softmax(scores, dim=-1) * valid.to(scores.dtype)
            denominator = weights.sum(dim=-1, keepdim=True)
            weights = torch.where(denominator > 0.0,
                                  weights / denominator.clamp_min(1e-12),
                                  torch.zeros_like(weights))
            pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
            context = torch.cat((pooled, stats), dim=-1)
            latent = self.context_out(context)
            residual = F.linear(latent, self.class_embedding) / math.sqrt(HIDDEN_WIDTH)
            if residual.shape != (batch, FEATURES):
                raise RuntimeError(f"Unexpected residual shape {tuple(residual.shape)}")
            return residual

    return FeatureResidual


def _tensor_batch(torch, mode_inputs, stats_tensor, indices):
    import numpy as np
    slot_array, mask_array = mode_inputs
    index_array = np.asarray(indices, dtype=np.int64)
    return (
        torch.from_numpy(np.ascontiguousarray(slot_array[index_array])).float(),
        torch.from_numpy(np.ascontiguousarray(mask_array[index_array])).bool(),
        stats_tensor[index_array],
    )


def _evaluate_model(torch, model, arrays, mode_inputs, stats_tensor, indices,
                    policy_module, p_norm, original_q):
    import numpy as np
    model.eval()
    with torch.inference_mode():
        slot_values, slot_mask, stats = _tensor_batch(
            torch, mode_inputs, stats_tensor, indices
        )
        raw = torch.from_numpy(
            np.ascontiguousarray(arrays["raw_logits"][indices])
        ).float()
        delta = model(slot_values, slot_mask, stats)
        delta_np = delta.cpu().numpy()
    out = np.zeros((len(indices), FEATURES), dtype=np.float64)
    for local_index, source_index in enumerate(indices):
        if not np.any(delta_np[local_index] != 0.0):
            q = policy_module.apply_feature_policy(
                arrays["raw_logits"][source_index], delta_np[local_index],
                original_q=original_q[source_index], gate=0.0,
                temperature=TEMPERATURE, top_p=TOP_P,
            )
        else:
            q = policy_module.apply_feature_policy(
                arrays["raw_logits"][source_index], delta_np[local_index],
                original_q=original_q[source_index], gate=RESIDUAL_MULTIPLIER,
                temperature=TEMPERATURE, top_p=TOP_P,
            )
        out[local_index] = np.asarray(q, dtype=np.float64)
    return out, delta_np


def _save_model(np, path, model):
    arrays = {
        name.replace(".", "__"): value.detach().cpu().numpy()
        for name, value in model.named_parameters()
    }
    np.savez_compressed(path, **arrays)
    return {name: list(value.shape) for name, value in arrays.items()}


def _weighted_tune_score(metric):
    return float(metric["source_equal_mean_p1_weight_times_overlap"])


def _aggregate_q_metrics(np, rows, arrays, p_norm, weights, split_indices, q_policy,
                         scores_path, method, seed, budget):
    results = {}
    q_values = np.zeros((len(rows), FEATURES), dtype=np.float64)
    q_values[split_indices] = q_policy
    for split, indices in split_indices.items():
        metric = _write_scores(
            scores_path, rows, split, method, seed, q_values, p_norm, weights,
            indices, budget,
        )
        results[split] = metric
    return q_values, results


def _make_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Directory containing prepared_data.npz, rows.jsonl, metadata.json")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[2] /
                        "docs/project/artifacts/pf2x_loop_20261019/round03/source_manifest.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="A new directory for the CPU analysis outputs")
    parser.add_argument("--deadline-unix", type=float, default=None,
                        help="Optional shared wall-clock deadline supplied by the primary agent")
    parser.add_argument("--budget-seconds", type=float, default=RUN_SECONDS_LIMIT)
    parser.add_argument("--training-soft-seconds", type=float, default=FIT_SECONDS_SOFT_LIMIT)
    return parser


def run(args, started_wall: float | None = None,
        started_mono: float | None = None) -> int:
    if started_wall is None:
        started_wall = time.time()
    if started_mono is None:
        started_mono = time.monotonic()
    import numpy as np
    import torch

    if args.output.exists():
        raise ValueError(f"Output path already exists: {args.output}")
    args.output.mkdir(parents=True)
    progress_path = args.output / "progress.json"
    progress = {
        "status": "loading",
        "started_unix": started_wall,
        "deadline_unix": args.deadline_unix,
        "budget_seconds": float(args.budget_seconds),
        "training_soft_seconds": float(args.training_soft_seconds),
        "completed_models": [],
        "active_model": None,
        "completed_updates": 0,
    }
    _write_progress(progress_path, progress)
    overall_deadline = min(
        started_mono + float(args.budget_seconds),
        started_mono + max(0.0, args.deadline_unix - started_wall)
        if args.deadline_unix is not None else float("inf"),
    )

    torch.set_num_threads(4)
    policy_module, policy_path = _load_policy_module()
    data = _load_data(np, args.data.expanduser().resolve(),
                      args.manifest.expanduser().resolve())
    arrays = data["arrays"]
    rows = data["rows"]
    split_indices = data["split_indices"]
    fit_indices = split_indices["fit"]
    tune_indices = split_indices["tune"]
    hold_indices = split_indices["hold"]
    if time.monotonic() >= overall_deadline:
        progress.update(status="stopped_before_training", stop_reason="shared_deadline")
        _write_progress(progress_path, progress)
        return 2

    pca = _pca_embedding(torch, arrays["feature_embedding"])
    pca_embedding = pca["pca_embedding"]
    stats_raw = _logit_q_statistics(np, arrays["raw_logits"], data["q_norm"])
    stats_z, stats_mean, stats_scale = _standardize_fit_stats(np, stats_raw, fit_indices)
    slot_inputs = _feature_slots(np, arrays, pca_embedding, rows)
    with (args.output / "s4_assignments.jsonl").open("w", encoding="utf-8") as stream:
        s4_global_ids = slot_inputs["s4_feature_local_ids"].astype(np.int64) + FEATURE_OFFSET
        s4_global_ids[~slot_inputs["s4_mask"]] = -1
        for index, row in enumerate(rows):
            record = {
                "vector_index": index,
                "sample_id": str(row["sample_id"]),
                "source_id": str(row["source_id"]),
                "position_ids": slot_inputs["s4_positions"][index].astype(int).tolist(),
                "feature_global_ids": s4_global_ids[index].astype(int).tolist(),
                "mask": slot_inputs["s4_mask"][index].astype(bool).tolist(),
                "age_pf_pairs": slot_inputs["s4_ages"][index].astype(int).tolist(),
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
    q_mix, neighbor_empirical, neighbor_counts = _make_neighbor_mixture(
        np, arrays["q1"], data["q_norm"],
        arrays["neighbor_feature_local_ids"].astype(np.int64),
        arrays["neighbor_mask"].astype(bool),
    )
    weights = arrays["p1_accept_probability"].astype(np.float64)
    q_rebuilt = np.asarray([
        policy_module.policy_from_logits(row, temperature=TEMPERATURE, top_p=TOP_P)
        for row in arrays["raw_logits"]
    ], dtype=np.float64)
    q_rebuilt_difference = np.abs(q_rebuilt - arrays["q1"].astype(np.float64))
    if not np.any(weights[fit_indices] > 0.0) or not np.any(weights[tune_indices] > 0.0):
        raise ValueError("P1 theoretical probabilities must be positive in fit and tune")

    FeatureResidual = _torch_model(torch)
    class_embedding = torch.from_numpy(np.ascontiguousarray(pca_embedding))
    stats_tensor = torch.from_numpy(np.ascontiguousarray(stats_z)).float()
    history_path = args.output / "history.jsonl"
    scores_path = args.output / "scores.jsonl"
    q_archive = {
        "p_f_fit": arrays["p_f"][fit_indices].astype(np.float64),
        "p_f_tune": arrays["p_f"][tune_indices].astype(np.float64),
        "p_f_hold": arrays["p_f"][hold_indices].astype(np.float64),
        "q_S0_seed_none_fit": arrays["q1"][fit_indices].astype(np.float64),
        "q_S0_seed_none_tune": arrays["q1"][tune_indices].astype(np.float64),
        "indices_fit": np.asarray(fit_indices, dtype=np.int32),
        "indices_tune": np.asarray(tune_indices, dtype=np.int32),
        "indices_hold": np.asarray(hold_indices, dtype=np.int32),
    }
    model_reports = {}
    tune_metrics = {}
    model_parameters = {}
    fit_start_mono = time.monotonic()
    history_stream = history_path.open("w", encoding="utf-8")
    stopped = False

    def overall_time_left() -> bool:
        return time.monotonic() < overall_deadline

    for method in METHODS:
        for seed in SEEDS:
            label = f"{method}_seed_{seed}"
            if not overall_time_left():
                stopped = True
                progress.update(status="stopped_by_shared_deadline",
                                stop_reason="shared_deadline_before_model",
                                elapsed_seconds=time.monotonic() - started_mono)
                _write_progress(progress_path, progress)
                break
            if time.monotonic() - fit_start_mono >= args.training_soft_seconds:
                stopped = True
                progress.update(status="stopped_at_training_soft_limit",
                                stop_reason="training_soft_limit",
                                elapsed_seconds=time.monotonic() - started_mono)
                _write_progress(progress_path, progress)
                break

            torch.manual_seed(seed)
            model = FeatureResidual(class_embedding)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            if method not in model_reports:
                model_reports[method] = {"parameter_count": int(parameter_count)}
            elif model_reports[method]["parameter_count"] != int(parameter_count):
                raise AssertionError("Model parameter count differs across modes")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=5e-4, weight_decay=0.01,
            )
            rng = torch.Generator(device="cpu")
            rng.manual_seed(seed)
            order = torch.randperm(len(fit_indices), generator=rng).tolist()
            cursor = 0
            completed_steps = 0
            progress.update(status="training", active_model=label,
                            active_step=0, completed_updates=int(progress["completed_updates"]))
            _write_progress(progress_path, progress)
            mode_input = slot_inputs[method]
            for step in range(1, TRAIN_STEPS + 1):
                elapsed_fit = time.monotonic() - fit_start_mono
                if elapsed_fit >= args.training_soft_seconds or not overall_time_left():
                    stopped = True
                    break
                if cursor + BATCH_SIZE > len(order):
                    order = torch.randperm(len(fit_indices), generator=rng).tolist()
                    cursor = 0
                local_batch = order[cursor:cursor + BATCH_SIZE]
                cursor += BATCH_SIZE
                batch_indices = [fit_indices[position] for position in local_batch]
                slot_values, slot_mask, batch_stats = _tensor_batch(
                    torch, mode_input, stats_tensor, batch_indices
                )
                raw_logits = torch.from_numpy(
                    np.ascontiguousarray(arrays["raw_logits"][batch_indices])
                ).float()
                target = torch.from_numpy(
                    np.ascontiguousarray(data["p_norm"][batch_indices])
                ).float()
                row_weights = torch.from_numpy(
                    np.ascontiguousarray(weights[batch_indices])
                ).float()
                delta = model(slot_values, slot_mask, batch_stats)
                log_q = torch.log_softmax(
                    (raw_logits + RESIDUAL_MULTIPLIER * delta) / TEMPERATURE, dim=-1,
                )
                p_log_p = torch.special.xlogy(target, target).sum(dim=-1)
                row_kl = p_log_p - torch.sum(target * log_q, dim=-1)
                weighted_kl = torch.sum(row_weights * row_kl) / torch.sum(row_weights)
                residual_l2 = torch.mean(delta.square())
                loss = weighted_kl + L2_WEIGHT * residual_l2
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at {label} update {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                completed_steps = step
                progress["completed_updates"] += 1
                progress["active_step"] = step
                progress["elapsed_seconds"] = time.monotonic() - started_mono
                history_record = {
                    "method": method, "seed": seed, "step": step,
                    "step_budget": TRAIN_STEPS, "batch_size": BATCH_SIZE,
                    "fit_source_rows": len(fit_indices),
                    "weighted_kl": float(weighted_kl.detach()),
                    "residual_l2_mean_square": float(residual_l2.detach()),
                    "l2_coefficient": L2_WEIGHT,
                    "loss": float(loss.detach()),
                    "elapsed_seconds": float(progress["elapsed_seconds"]),
                }
                history_stream.write(json.dumps(history_record, allow_nan=False) + "\n")
                history_stream.flush()
                _write_progress(progress_path, progress)
            if stopped or completed_steps != TRAIN_STEPS:
                model_path = args.output / f"model_{label}_partial.npz"
                model_parameters[label] = _save_model(np, model_path, model)
                progress.update(status="stopped_during_training",
                                active_model=label, active_step=completed_steps,
                                stop_reason=("shared_deadline" if not overall_time_left()
                                             else "training_soft_limit"),
                                elapsed_seconds=time.monotonic() - started_mono)
                _write_progress(progress_path, progress)
                break

            model_path = args.output / f"model_{label}.npz"
            model_parameters[label] = _save_model(np, model_path, model)
            if not overall_time_left():
                stopped = True
                progress.update(status="stopped_before_tune_scoring",
                                active_model=label, active_step=completed_steps,
                                stop_reason="shared_deadline",
                                elapsed_seconds=time.monotonic() - started_mono)
                _write_progress(progress_path, progress)
                break
            q_tune, _delta_tune = _evaluate_model(
                torch, model, arrays, mode_input, stats_tensor, tune_indices,
                policy_module, data["p_norm"], arrays["q1"],
            )
            q_full = np.zeros((len(rows), FEATURES), dtype=np.float64)
            q_full[tune_indices] = q_tune
            metric = _write_scores(
                scores_path, rows, "tune", method, seed, q_full, data["p_norm"],
                weights, tune_indices, TRAIN_STEPS,
            )
            tune_metrics[(method, seed)] = metric
            q_archive[f"q_{method}_seed_{seed}_tune"] = q_tune
            model_reports[method].setdefault("seeds", {})[str(seed)] = {
                "status": "complete", "updates": completed_steps,
                "parameter_file": model_path.name,
                "parameter_count": int(parameter_count),
                "tune": {key: value for key, value in metric.items() if key != "per_row"},
            }
            progress["completed_models"].append(label)
            progress["active_model"] = None
            progress["active_step"] = None
            _write_progress(progress_path, progress)
            del model, optimizer
        if stopped:
            break
    history_stream.close()

    if stopped or any(
        model_reports.get(method, {}).get("seeds", {}).get(str(seed), {}).get("status") != "complete"
        for method in METHODS for seed in SEEDS
    ):
        progress.update(status="incomplete", elapsed_seconds=time.monotonic() - started_mono)
        _write_progress(progress_path, progress)
        _write_json(args.output / "partial_summary.json", {
            "status": "incomplete", "completed_models": progress["completed_models"],
            "completed_updates": progress["completed_updates"],
            "elapsed_seconds": time.monotonic() - started_mono,
            "reason": progress.get("stop_reason", "training_incomplete"),
            "hold_scored": False,
            "note": "No hold scores were computed because all fixed training updates did not finish.",
        })
        return 2

    selected_seeds = {}
    for method in METHODS:
        ranked = sorted(
            SEEDS,
            key=lambda seed: (-_weighted_tune_score(tune_metrics[(method, seed)]), seed),
        )
        selected_seeds[method] = int(ranked[0])
    selected_control = max(
        ("S1", "S2", "S4"),
        key=lambda method: _weighted_tune_score(tune_metrics[(method, selected_seeds[method])]),
    )
    selected_s3_seed = selected_seeds["S3"]
    s3_tune = tune_metrics[("S3", selected_s3_seed)]
    s0_tune = _source_metrics(
        np, data["p_norm"], arrays["q1"].astype(np.float64), weights, rows, tune_indices,
    )
    control_tune = tune_metrics[(selected_control, selected_seeds[selected_control])]
    s3_by_source = s3_tune["by_source"]
    s0_by_source = s0_tune["by_source"]
    control_by_source = control_tune["by_source"]
    per_source_thresholds = {}
    for source in data["splits"]["tune"]:
        gain_s0 = (s3_by_source[source]["mean_p1_weight_times_overlap"] -
                   s0_by_source[source]["mean_p1_weight_times_overlap"])
        gain_control = (s3_by_source[source]["mean_p1_weight_times_overlap"] -
                        control_by_source[source]["mean_p1_weight_times_overlap"])
        per_source_thresholds[source] = {
            "gain_vs_S0": float(gain_s0),
            "gain_vs_best_control": float(gain_control),
            "improves_vs_S0": bool(gain_s0 > 0.0),
        }
    overall_gain_s0 = (
        _weighted_tune_score(s3_tune) - _weighted_tune_score(s0_tune)
    )
    overall_gain_control = (
        _weighted_tune_score(s3_tune) - _weighted_tune_score(control_tune)
    )
    tune_screen = {
        "S3_seed": selected_s3_seed,
        "best_control_method": selected_control,
        "best_control_seed": selected_seeds[selected_control],
        "gain_vs_S0": float(overall_gain_s0),
        "gain_vs_best_S1_S2_S4": float(overall_gain_control),
        "minimum_gain_vs_S0": 0.03,
        "minimum_gain_vs_best_control": 0.015,
        "both_tune_sources_improve_vs_S0": all(
            row["improves_vs_S0"] for row in per_source_thresholds.values()
        ),
        "per_source": per_source_thresholds,
        "passes_tune_screen": bool(
            overall_gain_s0 >= 0.03 and overall_gain_control >= 0.015 and
            all(row["improves_vs_S0"] for row in per_source_thresholds.values())
        ),
    }

    q_archive["q_SMIX_0p1_seed_none_fit"] = q_mix[fit_indices].astype(np.float64)
    q_archive["q_SMIX_0p1_seed_none_tune"] = q_mix[tune_indices].astype(np.float64)
    s0_fit_result = _write_scores(
        scores_path, rows, "fit", "S0", None,
        _pad_q(np, len(rows), fit_indices, q_archive["q_S0_seed_none_fit"]),
        data["p_norm"], weights, fit_indices, 0,
    )
    s0_tune_result = _write_scores(
        scores_path, rows, "tune", "S0", None,
        _pad_q(np, len(rows), tune_indices, q_archive["q_S0_seed_none_tune"]),
        data["p_norm"], weights, tune_indices, 0,
    )
    smix_fit_result = _write_scores(
        scores_path, rows, "fit", "SMIX_0.1", None,
        _pad_q(np, len(rows), fit_indices, q_archive["q_SMIX_0p1_seed_none_fit"]),
        data["p_norm"], weights, fit_indices, 0,
    )
    smix_tune_result = _write_scores(
        scores_path, rows, "tune", "SMIX_0.1", None,
        _pad_q(np, len(rows), tune_indices, q_archive["q_SMIX_0p1_seed_none_tune"]),
        data["p_norm"], weights, tune_indices, 0,
    )

    # Fit scoring follows Tune seed selection and never changes that choice.
    fit_metrics = {}
    for method in METHODS:
        for seed in SEEDS:
            label = f"{method}_seed_{seed}"
            model = FeatureResidual(class_embedding)
            parameter_file = args.output / f"model_{label}.npz"
            with np.load(parameter_file, allow_pickle=False) as archive:
                state = {
                    name.replace("__", "."): torch.from_numpy(np.array(archive[name], copy=True))
                    for name in archive.files
                }
            model.load_state_dict(state, strict=True)
            q_fit, _delta_fit = _evaluate_model(
                torch, model, arrays, slot_inputs[method], stats_tensor, fit_indices,
                policy_module, data["p_norm"], arrays["q1"],
            )
            q_archive[f"q_{method}_seed_{seed}_fit"] = q_fit
            q_full = np.zeros((len(rows), FEATURES), dtype=np.float64)
            q_full[fit_indices] = q_fit
            metric = _write_scores(
                scores_path, rows, "fit", method, seed, q_full, data["p_norm"],
                weights, fit_indices, TRAIN_STEPS,
            )
            fit_metrics[label] = {key: value for key, value in metric.items() if key != "per_row"}
            del model

    # Hold scoring starts after all Tune choices and Fit summaries are complete.
    hold_started = True
    q_archive["q_S0_seed_none_hold"] = arrays["q1"][hold_indices].astype(np.float64)
    q_archive["q_SMIX_0p1_seed_none_hold"] = q_mix[hold_indices].astype(np.float64)
    hold_metrics = {
        "S0": _write_scores(
            scores_path, rows, "hold", "S0", None,
            _pad_q(np, len(rows), hold_indices, q_archive["q_S0_seed_none_hold"]),
            data["p_norm"], weights, hold_indices, 0,
        ),
        "SMIX_0.1": _write_scores(
            scores_path, rows, "hold", "SMIX_0.1", None,
            _pad_q(np, len(rows), hold_indices, q_archive["q_SMIX_0p1_seed_none_hold"]),
            data["p_norm"], weights, hold_indices, 0,
        ),
    }
    hold_selected = {}
    for method in METHODS:
        seed = selected_seeds[method]
        label = f"{method}_seed_{seed}"
        model = FeatureResidual(class_embedding)
        # Restore only the Tune-selected model before the single Hold scoring pass.
        parameter_file = args.output / f"model_{label}.npz"
        with np.load(parameter_file, allow_pickle=False) as archive:
            state = {
                name.replace("__", "."): torch.from_numpy(np.array(archive[name], copy=True))
                for name in archive.files
            }
        model.load_state_dict(state, strict=True)
        q_hold, _delta_hold = _evaluate_model(
            torch, model, arrays, slot_inputs[method], stats_tensor, hold_indices,
            policy_module, data["p_norm"], arrays["q1"],
        )
        q_archive[f"q_{method}_seed_{seed}_hold"] = q_hold
        q_full = np.zeros((len(rows), FEATURES), dtype=np.float64)
        q_full[hold_indices] = q_hold
        metric = _write_scores(
            scores_path, rows, "hold", method, seed, q_full, data["p_norm"],
            weights, hold_indices, TRAIN_STEPS,
        )
        hold_selected[method] = {
            "seed": seed, "score_passes": 1,
            "metrics": {key: value for key, value in metric.items() if key != "per_row"},
        }
        hold_metrics[method] = metric
        del model

    # The fixed 0.1 mixture reference is retained on every split.
    np.savez_compressed(args.output / "q_outputs.npz", **q_archive)
    np.savez_compressed(
        args.output / "condition_inputs.npz",
        pca_mean=pca["mean"], pca_components=pca["components"],
        pca_eigenvalues_top32=pca["eigenvalues"],
        pca_raw32=pca["pca_raw"], pca_embedding32=pca["pca_embedding"],
        pca_global_rms=np.asarray(pca["global_rms"], dtype=np.float64),
        stats_raw=stats_raw.astype(np.float64), stats_fit_mean=stats_mean,
        stats_fit_scale=stats_scale, s4_selected_position_ids=slot_inputs["s4_positions"],
        s4_selected_feature_local_ids=slot_inputs["s4_feature_local_ids"],
        s4_selected_age_pf_pairs=slot_inputs["s4_ages"],
        s4_selected_relative_xyz=slot_inputs["s4_relative_xyz"],
        s4_selected_mask=slot_inputs["s4_mask"],
        s4_available_nonneighbor_count=slot_inputs["s4_available_count"],
        s4_expected_neighbor_mask=arrays["neighbor_mask"].astype(bool),
        s4_expected_neighbor_count=arrays["neighbor_mask"].sum(axis=1).astype(np.int32),
        s4_actual_valid_count=slot_inputs["s4_mask"].sum(axis=1).astype(np.int32),
        neighbor_empirical=neighbor_empirical.astype(np.float32),
    )

    all_counts_equal = bool(np.array_equal(
        slot_inputs["s4_mask"].sum(axis=1),
        arrays["neighbor_mask"].astype(bool).sum(axis=1),
    ))
    neighbor_hist = {
        str(value): int(np.count_nonzero(neighbor_counts == value))
        for value in range(SLOT_COUNT + 1)
    }
    s4_hist = {
        str(value): int(np.count_nonzero(slot_inputs["s4_mask"].sum(axis=1) == value))
        for value in range(SLOT_COUNT + 1)
    }
    raw_p_mass = arrays["p_f"].astype(np.float64).sum(axis=1)
    raw_q_mass = arrays["q1"].astype(np.float64).sum(axis=1)
    source_row_counts = {
        split: {
            source: sum(data["scene_ids"][index] == source for index in indices)
            for source in data["splits"][split]
        }
        for split, indices in split_indices.items()
    }
    method_parameters_equal = len({
        int(model_reports[method]["parameter_count"]) for method in METHODS
    }) == 1
    summary = {
        "schema_version": 1,
        "status": "complete_cpu_offline_analysis",
        "started_unix": started_wall,
        "elapsed_seconds": time.monotonic() - started_mono,
        "deadline_unix": args.deadline_unix,
        "budget_seconds": float(args.budget_seconds),
        "training_soft_seconds": float(args.training_soft_seconds),
        "torch_num_threads": 4,
        "device": "cpu",
        "gpu_used": False,
        "target_forward_used": False,
        "policy_module": str(policy_path),
        "prepared_data": str(args.data.resolve()),
        "prepared_metadata": data["metadata"],
        "source_manifest": data["manifest"],
        "source_rows": source_row_counts,
        "counts": {
            "rows": len(rows), "sources": len(set(data["scene_ids"])),
            "fit_rows": len(fit_indices), "tune_rows": len(tune_indices),
            "hold_rows": len(hold_indices), "model_count": 8,
            "completed_updates": progress["completed_updates"],
            "updates_per_model": TRAIN_STEPS,
        },
        "r1_selection_condition": {
            "rows_are_actual_p1_accepted_reservoir_events": True,
            "p1_accepted_actual_true_count": int(arrays["p1_accepted_actual"].sum()),
            "p1_accept_probability_used_for_fit_and_weighted_score": True,
            "interpretation": "mean(p1_accept_probability * overlap) on this selected sample only; no inverse selection weighting and no estimate of all original PF rounds",
        },
        "probability_integrity": {
            "p_f_mean_abs_mass_error": float(np.mean(np.abs(raw_p_mass - 1.0))),
            "p_f_max_abs_mass_error": float(np.max(np.abs(raw_p_mass - 1.0))),
            "q1_mean_abs_mass_error": float(np.mean(np.abs(raw_q_mass - 1.0))),
            "q1_max_abs_mass_error": float(np.max(np.abs(raw_q_mass - 1.0))),
            "minimum_positive_p_f": float(np.min(arrays["p_f"][arrays["p_f"] > 0.0])),
            "minimum_positive_q1": float(np.min(arrays["q1"][arrays["q1"] > 0.0])),
            "probability_cutoff": None,
            "metric_definition": "normalize each p_f and q row by its float64 sum; preserve original arrays and report mass errors",
            "raw_logits_policy_reconstruction_max_abs_difference_vs_saved_q1": float(q_rebuilt_difference.max()),
            "raw_logits_policy_reconstruction_mean_abs_difference_vs_saved_q1": float(q_rebuilt_difference.mean()),
            "raw_logits_softcap": "already applied by the original Feature MLP path; runner adds no softcap",
        },
        "policies": {
            "temperature": TEMPERATURE, "top_p": TOP_P,
            "target_softcap": 15.0,
            "raw_logits_already_softcapped": True,
        },
        "pca": {
            **{key: value for key, value in pca.items()
               if key not in {"mean", "components", "eigenvalues", "pca_raw", "pca_embedding"}},
            "global_rms": pca["global_rms"],
            "fit_input": "feature_embedding only",
            "normalization": "one scalar RMS over the centered 4096x32 PCA matrix",
            "input_is_only_feature_embedding": True,
            "uses_target_labels": False,
            "mean_std_covariance_use_all_4096_vocabulary_rows": True,
        },
        "model": {
            "methods": list(METHODS),
            "parameter_count_by_method": {
                method: int(model_reports[method]["parameter_count"]) for method in METHODS
            },
            "all_methods_equal_parameter_count": method_parameters_equal,
            "parameter_shapes_by_method_seed": model_parameters,
            "slot_input_width": 36,
            "slot_encoder": "Linear(36,32), GELU, Linear(32,32), GELU; single Linear(32,1) attention score",
            "attention_pool_width": 32,
            "statistics_width": 8,
            "context_projection": "Linear(40,32), zero-initialized",
            "class_residual": "dot product with fixed PCA e32 for each of 4096 classes, divided by sqrt(32)",
            "statistics": [
                "raw_logits_mean", "raw_logits_std", "raw_logits_max", "raw_logits_min",
                "raw_top1_minus_top2", "q1_entropy_div_log4096", "q1_top1_mass", "q1_top8_mass",
            ],
            "statistics_standardization": "mean/std from fit rows only; zero or tiny std uses 1",
            "S1": "same model with all six slot vectors and masks zero",
            "S2": "most recent up-to-six committed Feature items in order; delta_xyz zero; age divided by history pair count",
            "S3": "committed six-axis neighbors; relative xyz divided by 19; age divided by history pair count",
            "S4": "same-history non-neighbor Features sampled without replacement using seed 32043; actual relative xyz/19 and age/history length; rows with insufficient candidates are recorded",
            "S4_mask_count": {
                "matches_S3_for_every_row": all_counts_equal,
                "expected_neighbor_histogram": neighbor_hist,
                "actual_s4_histogram": s4_hist,
                "rows_with_insufficient_non_neighbor_candidates": int(np.count_nonzero(
                    slot_inputs["s4_available_count"] <
                    arrays["neighbor_mask"].astype(bool).sum(axis=1)
                )),
                "selection_seed": S4_SEED,
                "positions_sampled_without_replacement": True,
            },
            "zero_residual_behavior": "when residual is exactly zero, return the original saved q values unchanged",
            "trainable_embedding": False,
        },
        "training": {
            "optimizer": "AdamW", "learning_rate": 5e-4,
            "weight_decay": 0.01, "batch_size": BATCH_SIZE,
            "updates_per_model": TRAIN_STEPS, "seeds": list(SEEDS),
            "total_updates": TRAIN_STEPS * len(METHODS) * len(SEEDS),
            "target": "p_f on fit rows only",
            "objective": "weighted mean KL(p_f || softmax((raw_logits + delta)/0.9)) + 1e-4 * mean(delta^2)",
            "row_weight": "p1_accept_probability; weighted KL divides by the sum of batch row weights",
            "gradient_policy": "temperature=.9, no top-p truncation",
            "evaluation_policy": {"temperature": TEMPERATURE, "top_p": TOP_P},
            "residual_multiplier": RESIDUAL_MULTIPLIER,
            "residual_l2_coefficient": L2_WEIGHT,
            "steps_are_fixed": True,
            "intermediate_step_selection": False,
            "training_history": "history.jsonl; appended and flushed after every optimizer update",
        },
        "training_models": model_reports,
        "tune_selection": {
            "selected_seed_by_method": selected_seeds,
            "rule": "maximize source-equal mean(p1_accept_probability * overlap) on Tune; ties choose seed 17",
            "per_method_seed_source_metrics": {
                f"{method}_seed_{seed}": tune_metrics[(method, seed)]["by_source"]
                for method in METHODS for seed in SEEDS
            },
            "hold_scores_begin_after_seed_selection": hold_started,
            "hold_seed_by_method": {
                method: hold_selected[method]["seed"] for method in METHODS
            },
            "hold_score_passes_per_selected_method": {
                method: hold_selected[method]["score_passes"] for method in METHODS
            },
        },
        "selection": {
            "tune_score": "source-equal mean of per-source mean(p1_accept_probability * overlap)",
            "tie_rule": "seed 17",
            "selected_seed_by_method": selected_seeds,
            "tune_metrics_by_method_seed": {
                f"{method}_seed_{seed}": {
                    key: value for key, value in tune_metrics[(method, seed)].items()
                    if key != "per_row"
                }
                for method in METHODS for seed in SEEDS
            },
            "hold_access_after_tune_selection": hold_started,
            "hold_method_seed": {
                method: {
                    "seed": hold_selected[method]["seed"],
                    "score_passes": hold_selected[method]["score_passes"],
                }
                for method in METHODS
            },
            "unselected_seed_hold_metrics": "not computed",
            "tune_screen": tune_screen,
            "simple_mixture": {
                "method": "SMIX_0.1",
                "fixed_neighbor_weight": 0.1,
                "selection_tuning": False,
                "used_as_control_in_tune_threshold": False,
                "fit": {key: value for key, value in smix_fit_result.items()
                        if key != "per_row"},
                "tune": {key: value for key, value in smix_tune_result.items()
                         if key != "per_row"},
                "hold": {key: value for key, value in hold_metrics["SMIX_0.1"].items()
                         if key != "per_row"},
            },
            "S0_fit": {key: value for key, value in s0_fit_result.items()
                       if key != "per_row"},
            "S0_tune": {key: value for key, value in s0_tune_result.items()
                        if key != "per_row"},
            "hold_metrics_selected_seeds": {
                method: {
                    key: value for key, value in hold_selected[method]["metrics"].items()
                }
                for method in METHODS
            },
        },
        "fit_metrics": fit_metrics,
        "limitations": [
            "The 480 rows are a finite development diagnostic from ten previously explored sources.",
            "The reservoir contains actual P1-accepted events only. The p1-weighted score is conditional on that selection and cannot estimate all original PF rounds.",
            "The two Tune and two Hold sources remain part of the existing development set; this split isolates fitting and current model selection only.",
            "No GPU, Target forward, online decoder, or wall-clock speed comparison was run.",
            "The 0.5 ms value is a future GPU design target and is not tested by this CPU experiment.",
            "Hold metrics are reported only for the Tune-selected seed of each method; the unselected seed has no Hold score.",
        ],
        "files": {
            "scores": "scores.jsonl",
            "q_outputs": "q_outputs.npz",
            "q_output_alignment": "Each q_<method>_seed_<seed>_<split> array follows indices_<split> in vector_index order.",
            "condition_inputs": "condition_inputs.npz",
            "s4_assignments": "s4_assignments.jsonl",
            "history": "history.jsonl",
            "progress": "progress.json",
            "models": [f"model_{method}_seed_{seed}.npz"
                       for method in METHODS for seed in SEEDS],
        },
    }
    _write_json(args.output / "summary.json", summary)
    progress.update(status="complete", active_model=None, active_step=None,
                    elapsed_seconds=time.monotonic() - started_mono,
                    summary="summary.json")
    _write_progress(progress_path, progress)
    return 0


def _pad_q(np, row_count, indices, q_rows):
    result = np.zeros((row_count, FEATURES), dtype=np.float64)
    result[np.asarray(indices, dtype=np.int64)] = np.asarray(q_rows, dtype=np.float64)
    return result


def main() -> int:
    parser = _make_cli()
    args = parser.parse_args()
    if args.budget_seconds <= 0.0 or args.training_soft_seconds <= 0.0:
        parser.error("time budgets must be positive")
    if args.deadline_unix is not None and not math.isfinite(args.deadline_unix):
        parser.error("--deadline-unix must be finite")
    return run(args, started_wall=time.time(), started_mono=time.monotonic())


if __name__ == "__main__":
    raise SystemExit(main())
