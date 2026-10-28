#!/usr/bin/env python3
"""Online S3 Feature adapter and saved-row distribution validation."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any


P_VOCAB = 8000
F_VOCAB = 4096
EOS_ID = 12096
TEMPERATURE = 0.9
TOP_P = 0.9
SLOT_COUNT = 6
HIDDEN_WIDTH = 32
STATS_WIDTH = 8
_DIRECTIONS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0),
               (0, 1, 0), (0, 0, -1), (0, 0, 1))


def _load_mlp_adapter():
    for module_name in ("draft_a.pf_feature_unified", "pf_feature_unified"):
        try:
            module = importlib.import_module(module_name)
            return module.MLPAdapter
        except ImportError:
            pass

    here = Path(__file__).resolve()
    candidates = (
        here.with_name("pf_feature_unified.py"),
        here.parent / "draft_a/pf_feature_unified.py",
        here.parent / "source_before/draft_a/pf_feature_unified.py",
        here.parents[2] / "docs/project/artifacts/spatial_online_20261020/"
        "source_before/draft_a/pf_feature_unified.py",
        here.parents[2] / "docs/project/artifacts/feature_ckpt_compare_20261020/"
        "raw/source_before/draft_a/pf_feature_unified.py",
    )
    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "_spatial_online_pf_feature_unified", path
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.MLPAdapter
    raise ImportError("Could not load the unified decoder MLPAdapter")


MLPAdapter = _load_mlp_adapter()


def _make_residual_model(torch, class_embedding):
    from torch import nn
    from torch.nn import functional as F

    class FeatureResidual(nn.Module):
        def __init__(self, fixed_embedding):
            super().__init__()
            self.slot_in = nn.Linear(36, HIDDEN_WIDTH)
            self.slot_hidden = nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH)
            self.attention = nn.Linear(HIDDEN_WIDTH, 1)
            self.context_out = nn.Linear(HIDDEN_WIDTH + STATS_WIDTH, HIDDEN_WIDTH)
            self.register_buffer("class_embedding", fixed_embedding.float(), persistent=False)

        def forward(self, slot_values, slot_mask, stats):
            batch = slot_values.shape[0]
            encoded = F.gelu(self.slot_in(slot_values))
            encoded = F.gelu(self.slot_hidden(encoded))
            scores = self.attention(encoded).squeeze(-1)
            valid = slot_mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~valid, -1e9)
            weights = torch.softmax(scores, dim=-1) * valid.to(scores.dtype)
            denominator = weights.sum(dim=-1, keepdim=True)
            weights = torch.where(
                denominator > 0.0,
                weights / denominator.clamp_min(1e-12),
                torch.zeros_like(weights),
            )
            pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
            latent = self.context_out(torch.cat((pooled, stats), dim=-1))
            residual = F.linear(latent, self.class_embedding) / math.sqrt(HIDDEN_WIDTH)
            if residual.shape != (batch, F_VOCAB):
                raise RuntimeError(f"Unexpected residual shape {tuple(residual.shape)}")
            return residual

    return FeatureResidual(class_embedding)


def _tokens_as_list(tokens):
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().long().cpu().tolist()
    if hasattr(tokens, "tolist") and not isinstance(tokens, (list, tuple)):
        tokens = tokens.tolist()
    return [int(value) for value in tokens]


def _committed_pairs(prefix, candidate_position):
    tokens = _tokens_as_list(prefix)
    if not tokens or tokens[0] != EOS_ID:
        raise ValueError("Prefix must begin with the BOS/EOS token")
    body = tokens[1:]
    if len(body) % 2:
        if body[-1] == candidate_position:
            body = body[:-1]
        else:
            raise ValueError("Prefix ends with an incomplete P/F pair")
    pairs = []
    seen_positions = set()
    for offset in range(0, len(body), 2):
        position = body[offset]
        feature = body[offset + 1]
        if not 0 <= position < P_VOCAB:
            raise ValueError(f"Invalid committed position token: {position}")
        if not P_VOCAB <= feature < EOS_ID:
            raise ValueError(f"Invalid committed Feature token: {feature}")
        if position in seen_positions:
            raise ValueError(f"Repeated committed position token: {position}")
        seen_positions.add(position)
        pairs.append((position, feature - P_VOCAB))
    return pairs


def _position_xyz(position):
    return position // 400, (position % 400) // 20, position % 20


def _neighbor_rows(prefix, candidate_position):
    pairs = _committed_pairs(prefix, candidate_position)
    if not 0 <= candidate_position < P_VOCAB:
        raise ValueError(f"Invalid candidate position token: {candidate_position}")
    feature_by_position = {position: local_id for position, local_id in pairs}
    pair_index_by_position = {position: index for index, (position, _) in enumerate(pairs)}
    if candidate_position in feature_by_position:
        raise ValueError("Candidate position is already present in committed history")
    candidate_xyz = _position_xyz(candidate_position)
    positions, feature_ids, relative_xyz, ages, masks = [], [], [], [], []
    for dx, dy, dz in _DIRECTIONS:
        neighbor_xyz = (candidate_xyz[0] + dx,
                        candidate_xyz[1] + dy,
                        candidate_xyz[2] + dz)
        if any(value < 0 or value >= 20 for value in neighbor_xyz):
            positions.append(-1)
            feature_ids.append(-1)
            relative_xyz.append((0, 0, 0))
            ages.append(-1)
            masks.append(False)
            continue
        position = neighbor_xyz[0] * 400 + neighbor_xyz[1] * 20 + neighbor_xyz[2]
        local_id = feature_by_position.get(position)
        if local_id is None:
            positions.append(-1)
            feature_ids.append(-1)
            relative_xyz.append((0, 0, 0))
            ages.append(-1)
            masks.append(False)
            continue
        pair_index = pair_index_by_position[position]
        positions.append(position)
        feature_ids.append(local_id)
        relative_xyz.append((dx, dy, dz))
        ages.append(len(pairs) - 1 - pair_index)
        masks.append(True)
    return {
        "pairs": pairs,
        "neighbor_position_ids": positions,
        "neighbor_feature_local_ids": feature_ids,
        "neighbor_relative_xyz": relative_xyz,
        "neighbor_age_pf_pairs": ages,
        "neighbor_mask": masks,
        "neighbor_count": sum(masks),
    }


class SpatialAdapter(MLPAdapter):
    """FP32 S3 residual adapter compatible with the unified decoder."""

    def __init__(self, feature, model_path, condition_path, prepared_path,
                 zero_residual=False, bf16=False, device="cuda"):
        if bf16:
            raise ValueError("SpatialAdapter requires FP32 Feature inference")
        super().__init__(feature, bf16=False)
        import numpy as np
        import torch

        self.model_path = Path(model_path)
        self.condition_path = Path(condition_path)
        self.prepared_path = Path(prepared_path)
        if self.model_path.name != "model_S3_seed_29.npz":
            raise ValueError("SpatialAdapter is fixed to model_S3_seed_29.npz")
        if self.prepared_path.is_dir():
            self.prepared_path = self.prepared_path / "prepared_data.npz"
        self.zero_residual = bool(zero_residual)
        self.s3_calls = 0
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("SpatialAdapter supports CUDA or CPU devices")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        with np.load(self.condition_path, allow_pickle=False) as archive:
            class_embedding = np.array(archive["pca_embedding32"], copy=True)
            stats_mean = np.array(archive["stats_fit_mean"], copy=True)
            stats_scale = np.array(archive["stats_fit_scale"], copy=True)
        if class_embedding.shape != (F_VOCAB, HIDDEN_WIDTH):
            raise ValueError("pca_embedding32 must have shape (4096, 32)")
        if stats_mean.shape != (STATS_WIDTH,) or stats_scale.shape != (STATS_WIDTH,):
            raise ValueError("Saved Feature statistics must each have width eight")
        if not np.isfinite(class_embedding).all() or not np.isfinite(stats_mean).all() \
                or not np.isfinite(stats_scale).all() or np.any(stats_scale <= 0.0):
            raise ValueError("Saved spatial model inputs must be finite with positive scales")

        self._torch = torch
        self._class_embedding = torch.as_tensor(
            class_embedding, dtype=torch.float32, device=self.device
        )
        self._stats_mean = torch.as_tensor(
            stats_mean, dtype=torch.float64, device=self.device
        )
        self._stats_scale = torch.as_tensor(
            stats_scale, dtype=torch.float64, device=self.device
        )
        self.spatial_model = _make_residual_model(torch, self._class_embedding)
        with np.load(self.model_path, allow_pickle=False) as archive:
            state = {
                key.replace("__", "."): torch.as_tensor(
                    np.array(archive[key], copy=True), dtype=torch.float32
                )
                for key in archive.files
            }
        expected = {
            "slot_in.weight", "slot_in.bias", "slot_hidden.weight", "slot_hidden.bias",
            "attention.weight", "attention.bias", "context_out.weight", "context_out.bias",
        }
        if set(state) != expected:
            raise ValueError("Saved S3 model parameter names differ from the expected model")
        parameter_count = sum(value.numel() for value in state.values())
        if parameter_count != 3585:
            raise ValueError(f"Saved S3 model has {parameter_count} parameters; expected 3585")
        self.spatial_model.load_state_dict(state, strict=True)
        self.spatial_model.to(self.device)
        self.spatial_model.eval()
        self.spatial_model.requires_grad_(False)
        self.variant = "spatial_s3_seed_29_fp32_zero" if self.zero_residual \
            else "spatial_s3_seed_29_fp32"
        self.feature_precision = "FP32"

    @property
    def mlp_calls(self):
        return self.forward_calls

    def _reset(self):
        super()._reset()
        self.s3_calls = 0

    def prepare(self, prompt, capacity):
        super().prepare(prompt, capacity)

    def rewind(self, keep):
        return None

    def close(self):
        super().close()

    def _base_policy(self, logits):
        from collect_feature_teacher_from_s3 import policy_batch

        row = logits.float().reshape(1, F_VOCAB)
        return policy_batch(row, TEMPERATURE, TOP_P)[0].float()

    def _stats(self, raw_logits, base_q):
        torch = self._torch
        values = raw_logits.reshape(-1).to(dtype=torch.float64)
        q = base_q.reshape(-1).to(dtype=torch.float64)
        q_mass = q.sum()
        q = q / q_mass
        top_logits = torch.topk(values, 2).values
        top_q = torch.topk(q, 8).values
        positive = q > 0.0
        entropy = -(q[positive] * torch.log(q[positive])).sum()
        stats = torch.stack((
            values.mean(), values.std(unbiased=False), values.max(), values.min(),
            top_logits[0] - top_logits[1], entropy / math.log(F_VOCAB),
            q.max(), top_q.sum(),
        ))
        standardized = ((stats - self._stats_mean) / self._stats_scale).float()
        return standardized.reshape(1, STATS_WIDTH), stats

    def _context_and_delta(self, prefix, candidate_position, raw_logits, base_q):
        torch = self._torch
        detail = _neighbor_rows(prefix, candidate_position)
        slots = torch.zeros((1, SLOT_COUNT, 36), dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(detail["neighbor_mask"], dtype=torch.bool,
                               device=self.device).reshape(1, SLOT_COUNT)
        valid_slots = [index for index, present in enumerate(detail["neighbor_mask"]) if present]
        pair_count = len(detail["pairs"])
        if valid_slots:
            feature_ids = torch.as_tensor(
                [detail["neighbor_feature_local_ids"][index] for index in valid_slots],
                dtype=torch.long, device=self.device,
            )
            embedding = self._class_embedding.index_select(0, feature_ids)
            delta = torch.as_tensor(
                [detail["neighbor_relative_xyz"][index] for index in valid_slots],
                dtype=torch.float32, device=self.device,
            ) / 19.0
            age = torch.as_tensor(
                [detail["neighbor_age_pf_pairs"][index] / max(pair_count, 1)
                 for index in valid_slots],
                dtype=torch.float32, device=self.device,
            ).reshape(-1, 1)
            combined = torch.cat((embedding, delta, age), dim=-1)
            slots[0, valid_slots, :] = combined
        stats, raw_stats = self._stats(raw_logits, base_q)
        old_tf32 = None
        self.s3_calls += 1
        if self.device.type == "cuda":
            old_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type, enabled=False):
                delta_logits = self.spatial_model(slots, mask, stats)[0]
        finally:
            if old_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = old_tf32
        detail["pair_count"] = pair_count
        detail["stats_raw"] = raw_stats
        detail["stats_standardized"] = stats
        detail["slot_values"] = slots[0]
        return delta_logits, detail

    def distribution_from_logits(self, prefix, p1, logits, *, diagnostics=False):
        """Return q from one saved or live raw-logit row using the online path."""
        torch = self._torch
        raw = logits.reshape(-1).float()
        if raw.numel() != F_VOCAB:
            raise ValueError("Feature raw logits must contain 4096 values")
        base_q = self._run(
            "sampling", self._base_policy, raw,
            gpu=self.device.type == "cuda", operation="spatial_base_policy",
        )
        delta, detail = self._run(
            "feature_context", self._context_and_delta, prefix, int(p1), raw, base_q,
            gpu=self.device.type == "cuda", input_tokens=1,
            operation="spatial_neighbor_stats_model",
        )
        if self.zero_residual:
            if diagnostics:
                detail = dict(detail)
                detail["zero_residual"] = True
                detail["delta_logits"] = delta
                detail["updated_logits"] = raw
                detail["q"] = base_q
                return base_q, detail
            return base_q

        def final_policy():
            from collect_feature_teacher_from_s3 import policy_batch

            updated = raw.reshape(1, F_VOCAB) + delta.reshape(1, F_VOCAB)
            return policy_batch(updated.float(), TEMPERATURE, TOP_P)[0].float()

        q = self._run(
            "sampling", final_policy, gpu=self.device.type == "cuda",
            operation="spatial_residual_policy",
        )
        if diagnostics:
            detail = dict(detail)
            detail["zero_residual"] = False
            detail["delta_logits"] = delta
            detail["updated_logits"] = raw + delta
            detail["q"] = q
            return q, detail
        return q

    def propose(self, prefix, p1, h_prev, rng):
        torch = self._torch
        shared = importlib.import_module("draft_a.pf_ranker_eval")
        shared._require(self._prepared and h_prev is not None,
                        "SpatialAdapter needs prepared state and Target prev hidden")
        if h_prev.device != self.device:
            raise ValueError(f"Target hidden is on {h_prev.device}, adapter is on {self.device}")
        if not prefix:
            raise ValueError("SpatialAdapter requires the current committed prefix")
        self.forward_calls += 1
        self.input_tokens += 1

        def forward():
            previous_feature = int(prefix[-1]) - P_VOCAB
            if not 0 <= previous_feature < F_VOCAB:
                raise ValueError("Current prefix must end in a committed Feature token")
            with torch.autocast(device_type="cuda", enabled=False):
                _, logits = self.feature(
                    h_prev.unsqueeze(0),
                    torch.tensor([previous_feature], device=h_prev.device),
                    torch.tensor([int(p1)], device=h_prev.device),
                )
            return logits.float()

        logits = self._run(
            "feature_forward", forward, gpu=True, input_tokens=1,
            operation="spatial_base_mlp",
        )
        return self.distribution_from_logits(prefix, int(p1), logits)


def _cpu_reference_rows(np, q_archive):
    references = np.full((480, F_VOCAB), np.nan, dtype=np.float64)
    for split in ("fit", "tune", "hold"):
        indices = np.asarray(q_archive[f"indices_{split}"], dtype=np.int64)
        name = f"q_S3_seed_29_{split}"
        if name not in q_archive:
            raise ValueError(f"Missing saved CPU q array: {name}")
        values = np.asarray(q_archive[name], dtype=np.float64)
        if values.shape != (len(indices), F_VOCAB):
            raise ValueError(f"Unexpected saved q shape for {name}: {values.shape}")
        references[indices] = values
    if not np.isfinite(references).all():
        raise ValueError("Saved CPU q rows do not cover all 480 records")
    return references


def _prefix_from_history(position_ids, feature_ids, offsets, row_index):
    begin, end = int(offsets[row_index]), int(offsets[row_index + 1])
    tokens = [EOS_ID]
    for position, feature in zip(position_ids[begin:end], feature_ids[begin:end]):
        tokens.extend((int(position), int(feature)))
    return tokens


def _top_p_summary(torch, logits):
    values = logits.reshape(-1).float() / TEMPERATURE
    ordered, ids = torch.sort(values, descending=True, stable=True)
    full = torch.softmax(ordered, dim=-1)
    cumulative = torch.cumsum(full, dim=-1)
    prior = torch.cat((torch.zeros_like(cumulative[:1]), cumulative[:-1]))
    keep = prior <= TOP_P
    count = int(keep.sum().item())
    boundary = count - 1
    return {
        "support_count": count,
        "boundary_token_id": int(ids[boundary].item()),
        "mass_before_boundary": float(prior[boundary].item()),
        "mass_through_boundary": float(cumulative[boundary].item()),
        "distance_before_from_top_p": float(TOP_P - prior[boundary].item()),
        "distance_through_from_top_p": float(cumulative[boundary].item() - TOP_P),
    }


def validate_saved_distribution(model_path, condition_path, prepared_path,
                                q_outputs_path, rows_path, *, device="cuda",
                                max_abs_limit=1e-5, tv_limit=1e-4,
                                include_zero_neighbor=True):
    """Compare 480 online CUDA q rows with saved CPU S3 seed-29 outputs.

    The saved raw logits and committed histories are replayed one row at a time.
    A threshold violation or neighbor reconstruction mismatch raises RuntimeError.
    """
    import numpy as np
    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA distribution validation requires CUDA")
    if max_abs_limit <= 0.0 or tv_limit <= 0.0:
        raise ValueError("Distribution limits must be positive")
    prepared_path = Path(prepared_path)
    if prepared_path.is_dir():
        prepared_path = prepared_path / "prepared_data.npz"
    rows_path = Path(rows_path)
    with np.load(prepared_path, allow_pickle=False) as archive:
        raw_logits = np.array(archive["raw_logits"], copy=True)
        p1_ids = np.array(archive["p1_id"], dtype=np.int64, copy=True)
        candidate_xyz = np.array(archive["candidate_p_xyz"], dtype=np.int64, copy=True)
        p1_accepted = np.array(archive["p1_accepted_actual"], dtype=bool, copy=True)
        offsets = np.array(archive["history_pair_offsets"], dtype=np.int64, copy=True)
        history_positions = np.array(archive["history_position_ids_flat"], copy=True)
        history_features = np.array(archive["history_feature_global_ids_flat"], copy=True)
        expected_neighbors = {
            key: np.array(archive[key], copy=True)
            for key in ("neighbor_position_ids", "neighbor_feature_local_ids",
                        "neighbor_relative_xyz", "neighbor_age_pf_pairs", "neighbor_mask")
        }
    with np.load(condition_path, allow_pickle=False) as archive:
        reference_stats = np.array(archive["stats_raw"], dtype=np.float64, copy=True)
        pca_embedding = np.array(archive["pca_embedding32"], dtype=np.float32, copy=True)
        fit_mean = np.array(archive["stats_fit_mean"], dtype=np.float64, copy=True)
        fit_scale = np.array(archive["stats_fit_scale"], dtype=np.float64, copy=True)
    if reference_stats.shape != (480, STATS_WIDTH):
        raise ValueError("Saved CPU statistics must have shape (480, 8)")
    if pca_embedding.shape != (F_VOCAB, HIDDEN_WIDTH):
        raise ValueError("Saved PCA class table must have shape (4096, 32)")
    if not np.isfinite(reference_stats).all() or not np.isfinite(fit_mean).all() \
            or not np.isfinite(fit_scale).all() or np.any(fit_scale <= 0.0):
        raise ValueError("Saved statistics must be finite with positive scales")
    rows = []
    with rows_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 480 or raw_logits.shape != (480, F_VOCAB) or p1_ids.shape != (480,) \
            or candidate_xyz.shape != (480, 3) or p1_accepted.shape != (480,):
        raise ValueError("Saved validation inputs must contain 480 aligned rows")
    if not p1_accepted.all():
        raise ValueError("All saved validation rows must be P1-accepted contexts")
    if offsets.shape != (481,) or int(offsets[-1]) != len(history_positions) \
            or len(history_positions) != len(history_features):
        raise ValueError("Saved committed history arrays have inconsistent dimensions")
    for index, row in enumerate(rows):
        if int(row.get("vector_index", -1)) != index or int(row.get("p1_id", -1)) != int(p1_ids[index]):
            raise ValueError(f"Saved JSONL and NPZ identities differ at row {index}")
        if int(row.get("context_pf_pairs", -1)) != int(offsets[index + 1] - offsets[index]):
            raise ValueError(f"Saved history length differs from JSONL at row {index}")
        if tuple(int(value) for value in candidate_xyz[index]) != _position_xyz(int(p1_ids[index])):
            raise ValueError(f"Saved candidate coordinates differ at row {index}")
    if not np.isfinite(raw_logits).all():
        raise ValueError("Saved raw logits contain a non-finite value")

    with np.load(q_outputs_path, allow_pickle=False) as q_archive:
        references = _cpu_reference_rows(np, q_archive)

    adapter = SpatialAdapter(
        feature=None, model_path=model_path, condition_path=condition_path,
        prepared_path=prepared_path, zero_residual=False, bf16=False, device=device,
    )
    cpu_adapter = SpatialAdapter(
        feature=None, model_path=model_path, condition_path=condition_path,
        prepared_path=prepared_path, zero_residual=False, bf16=False, device="cpu",
    )
    adapter.prepare([], 1)
    cpu_adapter.prepare([], 1)
    max_abs = 0.0
    max_tv = 0.0
    max_sum_error = 0.0
    support_differences = 0
    boundary_token_differences = 0
    neighbor_mismatches = 0
    max_stats_abs = 0.0
    max_standardized_stats_abs = 0.0
    max_slot_abs = 0.0
    max_residual_abs = 0.0
    boundary_examples = []
    q_sums = []
    for index in range(480):
        prefix = _prefix_from_history(
            history_positions, history_features, offsets, index
        )
        raw = torch.as_tensor(raw_logits[index], dtype=torch.float32, device=adapter.device)
        q_cuda, detail = adapter.distribution_from_logits(
            prefix, int(p1_ids[index]), raw, diagnostics=True
        )
        q_cuda_np = q_cuda.detach().cpu().numpy().astype(np.float64, copy=False)
        q_cpu = references[index]
        if not np.isfinite(q_cuda_np).all() or np.any(q_cuda_np < 0.0):
            raise RuntimeError(f"CUDA q row {index} is non-finite or negative")
        if not np.isfinite(q_cpu).all() or np.any(q_cpu < 0.0):
            raise RuntimeError(f"Saved CPU q row {index} is non-finite or negative")
        q_cuda_sum = float(np.sum(q_cuda_np, dtype=np.float64))
        q_cpu_sum = float(np.sum(q_cpu, dtype=np.float64))
        row_sum_error = abs(q_cuda_sum - 1.0)
        if row_sum_error > 2e-5:
            raise RuntimeError(f"CUDA q row {index} sums to {q_cuda_sum:.9g}")
        if abs(q_cpu_sum - 1.0) > 2e-5:
            raise RuntimeError(f"Saved CPU q row {index} sums to {q_cpu_sum:.9g}")
        runtime_stats = detail["stats_raw"].detach().cpu().numpy().astype(np.float64, copy=False)
        runtime_standardized = detail["stats_standardized"].detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        if not np.isfinite(runtime_stats).all() or not np.isfinite(runtime_standardized).all():
            raise RuntimeError(f"Runtime statistics row {index} is non-finite")
        max_stats_abs = max(
            max_stats_abs,
            float(np.max(np.abs(runtime_stats - reference_stats[index]))),
        )
        expected_standardized = ((reference_stats[index] - fit_mean) / fit_scale).astype(np.float32)
        max_standardized_stats_abs = max(
            max_standardized_stats_abs,
            float(np.max(np.abs(runtime_standardized - expected_standardized))),
        )

        expected_slots = np.zeros((SLOT_COUNT, 36), dtype=np.float32)
        expected_mask = np.asarray(expected_neighbors["neighbor_mask"][index], dtype=bool)
        history_length = max(int(offsets[index + 1] - offsets[index]), 1)
        for slot_index, present in enumerate(expected_mask):
            if not present:
                continue
            feature_id = int(expected_neighbors["neighbor_feature_local_ids"][index, slot_index])
            expected_slots[slot_index, :HIDDEN_WIDTH] = pca_embedding[feature_id]
            expected_slots[slot_index, HIDDEN_WIDTH:HIDDEN_WIDTH + 3] = (
                expected_neighbors["neighbor_relative_xyz"][index, slot_index].astype(np.float32)
                / np.float32(19.0)
            )
            expected_slots[slot_index, -1] = np.float32(
                int(expected_neighbors["neighbor_age_pf_pairs"][index, slot_index])
                / history_length
            )
        runtime_slots = detail["slot_values"].detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        if not np.isfinite(runtime_slots).all():
            raise RuntimeError(f"Runtime slot row {index} is non-finite")
        max_slot_abs = max(
            max_slot_abs, float(np.max(np.abs(runtime_slots - expected_slots)))
        )
        cpu_raw = torch.as_tensor(raw_logits[index], dtype=torch.float32, device="cpu")
        cpu_base_q = cpu_adapter._base_policy(cpu_raw)
        cpu_delta_tensor, _cpu_detail = cpu_adapter._context_and_delta(
            prefix, int(p1_ids[index]), cpu_raw, cpu_base_q
        )
        cuda_delta = detail["delta_logits"].detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        cpu_delta = cpu_delta_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(cuda_delta).all() or not np.isfinite(cpu_delta).all():
            raise RuntimeError(f"Residual logits row {index} is non-finite")
        max_residual_abs = max(
            max_residual_abs, float(np.max(np.abs(cuda_delta - cpu_delta)))
        )
        difference = np.abs(q_cuda_np - q_cpu)
        row_max = float(np.max(difference))
        row_tv = float(0.5 * np.sum(difference, dtype=np.float64))
        max_abs = max(max_abs, row_max)
        max_tv = max(max_tv, row_tv)
        max_sum_error = max(max_sum_error, row_sum_error)
        q_sums.append(q_cuda_sum)

        neighbor_fields = (
            ("neighbor_position_ids", "neighbor_position_ids"),
            ("neighbor_feature_local_ids", "neighbor_feature_local_ids"),
            ("neighbor_relative_xyz", "neighbor_relative_xyz"),
            ("neighbor_age_pf_pairs", "neighbor_age_pf_pairs"),
            ("neighbor_mask", "neighbor_mask"),
        )
        for expected_name, detail_name in neighbor_fields:
            if not np.array_equal(
                    np.asarray(detail[detail_name]), expected_neighbors[expected_name][index]):
                neighbor_mismatches += 1
                raise RuntimeError(f"Saved neighbor reconstruction differs at row {index}")

        gpu_boundary = _top_p_summary(torch, detail["updated_logits"])
        cpu_support = np.flatnonzero(q_cpu > 0.0)
        gpu_support = np.flatnonzero(q_cuda_np > 0.0)
        if not np.array_equal(cpu_support, gpu_support):
            support_differences += 1
        cpu_boundary = int(cpu_support[np.argmin(q_cpu[cpu_support])]) if len(cpu_support) else -1
        if cpu_boundary != gpu_boundary["boundary_token_id"]:
            boundary_token_differences += 1
        if index < 5 or cpu_boundary != gpu_boundary["boundary_token_id"]:
            boundary_examples.append({
                "vector_index": index,
                "cpu_boundary_token_id": cpu_boundary,
                "cuda_boundary_token_id": gpu_boundary["boundary_token_id"],
                "cpu_support_count": int(len(cpu_support)),
                "cuda_support_count": gpu_boundary["support_count"],
                "cuda_mass_before_boundary": gpu_boundary["mass_before_boundary"],
                "cuda_mass_through_boundary": gpu_boundary["mass_through_boundary"],
                "cuda_distance_before_from_top_p": gpu_boundary["distance_before_from_top_p"],
                "cuda_distance_through_from_top_p": gpu_boundary["distance_through_from_top_p"],
            })
        if row_max > max_abs_limit or row_tv > tv_limit:
            failure = {
                "vector_index": index,
                "max_abs_difference": row_max,
                "total_variation": row_tv,
                "max_abs_limit": float(max_abs_limit),
                "tv_limit": float(tv_limit),
                "cpu_boundary_token_id": cpu_boundary,
                "cuda_boundary": gpu_boundary,
                "cpu_support_count": int(len(cpu_support)),
                "cuda_support_count": int(gpu_boundary["support_count"]),
            }
            raise RuntimeError(json.dumps(failure, ensure_ascii=False, allow_nan=False))

    zero_neighbor_result = None
    if include_zero_neighbor:
        synthetic_prefix = [EOS_ID, 100, P_VOCAB + 7]
        synthetic_q, synthetic_detail = adapter.distribution_from_logits(
            synthetic_prefix, 399,
            torch.as_tensor(raw_logits[0], dtype=torch.float32, device=adapter.device),
            diagnostics=True,
        )
        synthetic_sum = float(synthetic_q.double().sum().item())
        if synthetic_detail["neighbor_count"] != 0:
            raise RuntimeError("Synthetic no-neighbor input produced an axial neighbor")
        if not torch.isfinite(synthetic_q).all() or (synthetic_q < 0.0).any() \
                or abs(synthetic_sum - 1.0) > 2e-5:
            raise RuntimeError("Synthetic no-neighbor q is non-finite or unnormalized")
        zero_neighbor_result = {
            "neighbor_count": int(synthetic_detail["neighbor_count"]),
            "history_pair_count": int(synthetic_detail["pair_count"]),
            "q_sum": synthetic_sum,
            "q_support": int((synthetic_q > 0).sum().item()),
        }
    adapter.close()
    cpu_adapter.close()

    result = {
        "status": "passed",
        "device": str(adapter.device),
        "rows_compared": 480,
        "reference": "saved CPU q_S3_seed_29 split arrays",
        "raw_input": "saved raw_logits; one row per online path call",
        "policy": {"temperature": TEMPERATURE, "top_p": TOP_P},
        "limits": {"max_abs": float(max_abs_limit), "total_variation": float(tv_limit),
                   "q_sum_abs_error": 2e-5},
        "max_abs_difference": max_abs,
        "max_total_variation": max_tv,
        "max_cuda_q_sum_error": max_sum_error,
        "max_raw_statistics_difference": max_stats_abs,
        "max_standardized_statistics_difference": max_standardized_stats_abs,
        "max_slot_value_difference": max_slot_abs,
        "max_cpu_vs_cuda_residual_difference": max_residual_abs,
        "support_difference_rows": support_differences,
        "top_p_boundary_token_difference_rows": boundary_token_differences,
        "neighbor_reconstruction_mismatches": neighbor_mismatches,
        "cuda_q_sum_min": min(q_sums),
        "cuda_q_sum_max": max(q_sums),
        "top_p_boundary_examples": boundary_examples[:20],
        "synthetic_zero_neighbor": zero_neighbor_result,
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-saved", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--condition", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--q-outputs", type=Path)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-abs-limit", type=float, default=1e-5)
    parser.add_argument("--tv-limit", type=float, default=1e-4)
    parser.add_argument("--skip-zero-neighbor", action="store_true")
    args = parser.parse_args(argv)
    if not args.validate_saved:
        parser.print_help()
        return 0
    required = (args.model, args.condition, args.prepared, args.q_outputs, args.rows)
    if any(value is None for value in required):
        parser.error("--validate-saved requires --model, --condition, --prepared, --q-outputs, and --rows")
    result = validate_saved_distribution(
        args.model, args.condition, args.prepared, args.q_outputs, args.rows,
        device=args.device, max_abs_limit=args.max_abs_limit,
        tv_limit=args.tv_limit, include_zero_neighbor=not args.skip_zero_neighbor,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
