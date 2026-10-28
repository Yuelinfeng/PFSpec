#!/usr/bin/env python3
"""Small CPU probe for committed six-neighbor Feature context."""

from __future__ import annotations

import argparse
import math
from typing import Sequence


P_VOCAB = 8000
F_VOCAB = 4096
FEATURE_OFFSET = P_VOCAB
BOS_EOS = P_VOCAB + F_VOCAB
SIDE = 20
DIRECTIONS = (
    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (0, 0, -1), (0, 0, 1),
)


def _flat_position(x: int, y: int, z: int) -> int:
    return x * SIDE * SIDE + y * SIDE + z


def _coords(position_id: int) -> tuple[int, int, int]:
    if not 0 <= int(position_id) < P_VOCAB:
        raise ValueError(f"Position ID outside 0..{P_VOCAB - 1}: {position_id}")
    position_id = int(position_id)
    return position_id // 400, (position_id // 20) % 20, position_id % 20


class CommittedNeighbors:
    """Read complete P/F pairs from one actual token history."""

    def __init__(self, pairs: Sequence[tuple[int, int]]):
        self._features: dict[int, int] = {}
        for position_id, feature_id in pairs:
            position_id, feature_id = int(position_id), int(feature_id)
            _coords(position_id)
            if not FEATURE_OFFSET <= feature_id < BOS_EOS:
                raise ValueError(f"Feature ID outside {FEATURE_OFFSET}..{BOS_EOS - 1}: {feature_id}")
            if position_id in self._features:
                raise ValueError(f"Repeated committed Position ID: {position_id}")
            self._features[position_id] = feature_id

    @classmethod
    def from_tokens(cls, tokens: Sequence[int]) -> "CommittedNeighbors":
        """Parse [BOS, P, F, ...]; a final unpaired P is ignored."""
        ids = [int(token) for token in tokens]
        if not ids or ids[0] != BOS_EOS:
            raise ValueError("History must start with BOS id 12096")

        body = ids[1:]
        if BOS_EOS in body:
            eos_at = body.index(BOS_EOS)
            if eos_at != len(body) - 1:
                raise ValueError("EOS may only appear at the end of history")
            body = body[:-1]
            if len(body) % 2:
                raise ValueError("EOS must follow a complete P/F pair")

        pairs: list[tuple[int, int]] = []
        i = 0
        while i < len(body):
            position_id = body[i]
            if not 0 <= position_id < P_VOCAB:
                raise ValueError(f"Expected Position at history offset {i + 1}, got {position_id}")
            if i + 1 == len(body):
                break
            feature_id = body[i + 1]
            if not FEATURE_OFFSET <= feature_id < BOS_EOS:
                raise ValueError(f"Expected Feature at history offset {i + 2}, got {feature_id}")
            pairs.append((position_id, feature_id))
            i += 2
        return cls(pairs)

    def batch_features(self, position_ids: Sequence[int]) -> dict[str, list]:
        """Return six fixed direction slots; absent neighbor IDs are -1."""
        neighbor_positions: list[list[int]] = []
        neighbor_features: list[list[int]] = []
        present: list[list[bool]] = []
        deltas: list[list[list[int]]] = []
        for query in position_ids:
            x, y, z = _coords(int(query))
            row_p, row_f, row_m, row_d = [], [], [], []
            for dx, dy, dz in DIRECTIONS:
                nx, ny, nz = x + dx, y + dy, z + dz
                row_d.append([dx, dy, dz])
                if not (0 <= nx < SIDE and 0 <= ny < SIDE and 0 <= nz < SIDE):
                    row_p.append(-1)
                    row_f.append(-1)
                    row_m.append(False)
                    continue
                neighbor_position = _flat_position(nx, ny, nz)
                feature_id = self._features.get(neighbor_position)
                row_p.append(neighbor_position if feature_id is not None else -1)
                row_f.append(feature_id if feature_id is not None else -1)
                row_m.append(feature_id is not None)
            neighbor_positions.append(row_p)
            neighbor_features.append(row_f)
            present.append(row_m)
            deltas.append(row_d)
        return {
            "neighbor_position_ids": neighbor_positions,
            "neighbor_feature_ids": neighbor_features,
            "present": present,
            "delta_xyz": deltas,
        }

    def __len__(self) -> int:
        return len(self._features)


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for fitting or applying the residual probe") from exc
    return np


def _as_1d_list(values, *, name: str) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a one-dimensional array or sequence")
    if values and isinstance(values[0], (list, tuple)):
        raise ValueError(f"{name} must be one-dimensional")
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be non-empty and finite")
    return result


def _probability_row(values, *, name: str) -> list[float]:
    row = _as_1d_list(values, name=name)
    if any(value < 0.0 for value in row):
        raise ValueError(f"{name} contains a negative probability")
    total = math.fsum(row)
    if total <= 0.0 or abs(total - 1.0) > 1e-4:
        raise ValueError(f"{name} must sum to one; got {total:.9g}")
    return [value / total for value in row]


def _unchanged_probability_row(values, *, name: str) -> list[float]:
    row = _as_1d_list(values, name=name)
    if any(value < 0.0 for value in row):
        raise ValueError(f"{name} contains a negative probability")
    total = math.fsum(row)
    if total <= 0.0 or abs(total - 1.0) > 1e-4:
        raise ValueError(f"{name} must sum to one; got {total:.9g}")
    return row


def policy_from_logits(logits: Sequence[float], *, temperature: float = 0.9,
                       top_p: float = 0.9) -> list[float]:
    """Match the project's temperature and nucleus filtering for one row."""
    values = _as_1d_list(logits, name="logits")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    exp_values = [math.exp(value - maximum) for value in scaled]
    total = math.fsum(exp_values)
    full = [value / total for value in exp_values]
    order = sorted(range(len(values)), key=lambda index: (-scaled[index], index))
    keep: set[int] = set()
    cumulative = 0.0
    for index in order:
        keep.add(index)
        cumulative += full[index]
        if cumulative > top_p:
            break
    kept_total = math.fsum(full[index] for index in keep)
    return [full[index] / kept_total if index in keep else 0.0
            for index in range(len(values))]


def apply_feature_policy(raw_logits, delta_logits, *, original_q=None,
                         gate: float = 1.0, temperature: float = 0.9,
                         top_p: float = 0.9) -> list[float]:
    """Apply an additive proposal residual; zero scale preserves the supplied q."""
    if not math.isfinite(float(gate)):
        raise ValueError("residual multiplier must be finite")
    if float(gate) == 0.0 and original_q is not None:
        return _unchanged_probability_row(original_q, name="original_q")
    raw = _as_1d_list(raw_logits, name="raw_logits")
    delta = _as_1d_list(delta_logits, name="delta_logits")
    if len(raw) != len(delta):
        raise ValueError("raw_logits and delta_logits must have equal length")
    updated = [base + float(gate) * change for base, change in zip(raw, delta)]
    return policy_from_logits(updated, temperature=temperature, top_p=top_p)


def overlap(p, q) -> float:
    p_row = _probability_row(p, name="p")
    q_row = _probability_row(q, name="q")
    if len(p_row) != len(q_row):
        raise ValueError("p and q must have equal length")
    return math.fsum(min(a, b) for a, b in zip(p_row, q_row))


def residual_distribution(p, q) -> tuple[list[float], float]:
    """Return normalized positive residual and its probability mass."""
    p_row = _probability_row(p, name="p")
    q_row = _probability_row(q, name="q")
    if len(p_row) != len(q_row):
        raise ValueError("p and q must have equal length")
    residual = [max(a - b, 0.0) for a, b in zip(p_row, q_row)]
    mass = math.fsum(residual)
    if mass == 0.0:
        return [0.0] * len(residual), 0.0
    return [value / mass for value in residual], mass


class DirectionalResidualProbe:
    """Low-parameter CPU fit; p appears only as the fit label."""

    _ALLOWED = {
        "rank": (2, 4),
        "neighbor_groups": (4, 8),
        "logit_buckets": (8, 16),
    }

    def __init__(self, *, mode: str = "spatial", rank: int = 4,
                 neighbor_groups: int = 8, logit_buckets: int = 16,
                 seed: int = 32003):
        if mode not in {"spatial", "context_only"}:
            raise ValueError("mode must be 'spatial' or 'context_only'")
        for name, value in (("rank", rank), ("neighbor_groups", neighbor_groups),
                            ("logit_buckets", logit_buckets)):
            if value not in self._ALLOWED[name]:
                raise ValueError(f"unsupported {name}={value}; use {self._ALLOWED[name]}")
        if F_VOCAB % int(logit_buckets):
            raise ValueError("logit_buckets must divide the Feature vocabulary")
        self.mode = mode
        self.rank = int(rank)
        self.neighbor_groups = int(neighbor_groups)
        self.logit_buckets = int(logit_buckets)
        self.seed = int(seed)
        self._weights = None

    @property
    def parameter_count(self) -> int:
        input_dim = 6 * self.neighbor_groups
        return (input_dim * self.rank + 8 * self.rank + self.rank
                + self.rank * self.logit_buckets + self.logit_buckets)

    def _inputs(self, raw_logits, position_ids, neighbor_feature_ids):
        np = _numpy()
        raw = np.asarray(raw_logits, dtype=np.float64)
        positions = np.asarray(position_ids, dtype=np.int64).reshape(-1)
        neighbors = np.asarray(neighbor_feature_ids, dtype=np.int64)
        if raw.ndim != 2 or raw.shape[1] != F_VOCAB:
            raise ValueError(f"raw_logits must have shape [N,{F_VOCAB}]")
        if len(positions) != raw.shape[0] or neighbors.shape != (raw.shape[0], 6):
            raise ValueError("position_ids and neighbor_feature_ids must align with raw_logits")
        if not np.isfinite(raw).all():
            raise ValueError("raw_logits contains a non-finite value")
        coords = np.asarray([_coords(int(value)) for value in positions], dtype=np.float64)
        if np.any((neighbors < -1) | (neighbors >= F_VOCAB + FEATURE_OFFSET)):
            raise ValueError("neighbor_feature_ids must be -1 or a global Feature token ID")
        local_neighbors = neighbors.copy()
        local_neighbors[local_neighbors >= FEATURE_OFFSET] -= FEATURE_OFFSET
        if np.any(local_neighbors >= F_VOCAB):
            raise ValueError("neighbor Feature ID exceeds the vocabulary")

        centered = raw - raw.mean(axis=1, keepdims=True)
        std = np.maximum(np.sqrt(np.mean(centered * centered, axis=1, keepdims=True)), 1e-8)
        normalized = centered / std
        top = np.sort(normalized, axis=1)[:, ::-1]
        context = np.column_stack((
            coords / (SIDE - 1) * 2.0 - 1.0,
            top[:, 0], top[:, 0] - top[:, 1],
            top[:, :8].mean(axis=1), top[:, :32].mean(axis=1),
            np.log1p(std[:, 0]),
        ))
        context = np.clip(context, -8.0, 8.0)
        if context.shape[1] != 8:
            raise AssertionError("context feature width changed")

        if self.mode == "spatial":
            x = np.zeros((raw.shape[0], 6 * self.neighbor_groups), dtype=np.float64)
            for direction in range(6):
                present = local_neighbors[:, direction] >= 0
                groups = (local_neighbors[present, direction] * self.neighbor_groups) // F_VOCAB
                rows = np.nonzero(present)[0]
                x[rows, direction * self.neighbor_groups + groups] = 1.0
        else:
            clipped = context
            funcs = (
                lambda v: v,
                lambda v: v * v,
                lambda v: np.sin(math.pi * v),
                lambda v: np.cos(math.pi * v),
                lambda v: np.tanh(v),
                lambda v: v / (1.0 + np.abs(v)),
            )
            control_context = clipped[:, :self.neighbor_groups]
            x = np.concatenate([fn(control_context) for fn in funcs], axis=1)
        return np, raw, context, x

    def _delta_from_inputs(self, x, context):
        np = _numpy()
        weights = self._weights
        if weights is None:
            raise RuntimeError("fit must be called before residual_logits")
        hidden = np.tanh(x @ weights["w_in"] + context @ weights["w_context"]
                         + weights["b_hidden"])
        bucket_logits = hidden @ weights["w_out"] + weights["b_out"]
        return np.repeat(bucket_logits, F_VOCAB // self.logit_buckets, axis=1)

    def fit(self, raw_logits, position_ids, neighbor_feature_ids, target_p, *,
            steps: int = 100, lr: float = 0.03, l2: float = 1e-3) -> dict:
        """Fit on the caller's fit rows; target_p is consumed only here."""
        np, raw, context, x = self._inputs(raw_logits, position_ids, neighbor_feature_ids)
        p = np.asarray(target_p, dtype=np.float64)
        if p.shape != raw.shape or not np.isfinite(p).all() or np.any(p < 0.0):
            raise ValueError("target_p must be finite, nonnegative, and match raw_logits")
        sums = p.sum(axis=1, keepdims=True)
        if np.any(sums <= 0.0) or np.any(np.abs(sums[:, 0] - 1.0) > 1e-4):
            raise ValueError("each target_p row must sum to one")
        p = p / sums
        if int(steps) < 1 or not math.isfinite(float(lr)) or lr <= 0.0:
            raise ValueError("steps and lr must be positive")
        if not math.isfinite(float(l2)) or l2 < 0.0:
            raise ValueError("l2 must be nonnegative")

        rng = np.random.default_rng(self.seed)
        dim_x = x.shape[1]
        scale = 0.025
        weights = {
            "w_in": rng.normal(0.0, scale, size=(dim_x, self.rank)),
            "w_context": rng.normal(0.0, scale, size=(8, self.rank)),
            "b_hidden": np.zeros(self.rank, dtype=np.float64),
            "w_out": rng.normal(0.0, scale, size=(self.rank, self.logit_buckets)),
            "b_out": np.zeros(self.logit_buckets, dtype=np.float64),
        }
        moments = {key: np.zeros_like(value) for key, value in weights.items()}
        variances = {key: np.zeros_like(value) for key, value in weights.items()}
        n = raw.shape[0]
        bucket_width = F_VOCAB // self.logit_buckets
        for step in range(1, int(steps) + 1):
            pre = x @ weights["w_in"] + context @ weights["w_context"] + weights["b_hidden"]
            hidden = np.tanh(pre)
            bucket_logits = hidden @ weights["w_out"] + weights["b_out"]
            corrected = raw + np.repeat(bucket_logits, bucket_width, axis=1)
            scaled = corrected / 0.9
            scaled -= scaled.max(axis=1, keepdims=True)
            exp_logits = np.exp(scaled)
            q = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            log_z = np.log(exp_logits.sum(axis=1))
            loss = float((-np.sum(p * (scaled - log_z[:, None])) / n)
                         + 0.5 * l2 * sum(float(np.square(w).sum())
                                          for key, w in weights.items()
                                          if key.startswith("w_")))
            grad_logits = (q - p) / (n * 0.9)
            grad_bucket = grad_logits.reshape(n, self.logit_buckets, bucket_width).sum(axis=2)
            grad_out = hidden.T @ grad_bucket + l2 * weights["w_out"]
            grad_b_out = grad_bucket.sum(axis=0)
            grad_hidden = grad_bucket @ weights["w_out"].T
            grad_pre = grad_hidden * (1.0 - hidden * hidden)
            gradients = {
                "w_in": x.T @ grad_pre + l2 * weights["w_in"],
                "w_context": context.T @ grad_pre + l2 * weights["w_context"],
                "b_hidden": grad_pre.sum(axis=0),
                "w_out": grad_out,
                "b_out": grad_b_out,
            }
            if not math.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients.values()):
                raise FloatingPointError("non-finite residual fit value")
            for key, grad in gradients.items():
                moments[key] = 0.9 * moments[key] + 0.1 * grad
                variances[key] = 0.999 * variances[key] + 0.001 * grad * grad
                m_hat = moments[key] / (1.0 - 0.9 ** step)
                v_hat = variances[key] / (1.0 - 0.999 ** step)
                weights[key] -= float(lr) * m_hat / (np.sqrt(v_hat) + 1e-8)
        self._weights = weights
        return {"fit_rows": int(n), "steps": int(steps), "fit_cross_entropy": loss,
                "parameter_count": self.parameter_count, "mode": self.mode}

    def residual_logits(self, raw_logits, position_ids, neighbor_feature_ids):
        """Predict additive logits without receiving target_p."""
        np, raw, context, x = self._inputs(raw_logits, position_ids, neighbor_feature_ids)
        return self._delta_from_inputs(x, context).astype(np.float32)


def _self_test() -> dict:
    b, e, f = BOS_EOS, BOS_EOS, FEATURE_OFFSET
    tokens = [b, 21, f + 101, 400, f + 102, 421]
    book = CommittedNeighbors.from_tokens(tokens)
    assert len(book) == 2
    center = book.batch_features([421])
    assert center["neighbor_position_ids"][0] == [21, -1, -1, -1, -1, -1]
    assert center["neighbor_feature_ids"][0] == [f + 101, -1, -1, -1, -1, -1]
    assert center["delta_xyz"][0] == [[-1, 0, 0], [1, 0, 0], [0, -1, 0],
                                      [0, 1, 0], [0, 0, -1], [0, 0, 1]]

    edge = CommittedNeighbors.from_tokens([b, 19, f + 7, 400, f + 8])
    edge_row = edge.batch_features([0])
    assert edge_row["neighbor_position_ids"][0] == [-1, 400, -1, -1, -1, -1]
    assert edge_row["neighbor_feature_ids"][0] == [-1, f + 8, -1, -1, -1, -1]
    assert edge_row["neighbor_position_ids"][0][1] != 19

    before_feature = CommittedNeighbors.from_tokens([b, 399, f + 9, 400])
    after_feature = CommittedNeighbors.from_tokens([b, 399, f + 9, 400, f + 10])
    assert before_feature.batch_features([401])["neighbor_feature_ids"][0][4] == -1
    assert after_feature.batch_features([401])["neighbor_feature_ids"][0][4] == f + 10

    ended = CommittedNeighbors.from_tokens([b, 399, f + 9, e])
    assert len(ended) == 1
    for invalid in ([b, e, 399], [b, 399, e], [b, 399, 400, f + 2], [b, 399, f + 1, e, 400]):
        try:
            CommittedNeighbors.from_tokens(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid history was accepted: {invalid}")

    q = [0.0] * F_VOCAB
    q[7], q[11] = 0.6000001, 0.4
    q_returned = apply_feature_policy([0.0] * F_VOCAB, [1.0] * F_VOCAB,
                                      original_q=q, gate=0.0)
    assert q_returned == q
    assert math.fsum(q_returned) != 1.0
    target = [0.0, 0.0, 1.0]
    proposal = [0.0, 1.0, 0.0]
    r, r_mass = residual_distribution(target, proposal)
    assert r == target and r_mass == 1.0 and overlap(target, proposal) == 0.0
    r0, m0 = residual_distribution(target, target)
    assert r0 == [0.0, 0.0, 0.0] and m0 == 0.0 and overlap(target, target) == 1.0
    rm, mm = residual_distribution([0.7, 0.3, 0.0], [0.2, 0.5, 0.3])
    assert rm == [1.0, 0.0, 0.0] and abs(mm - (1.0 - overlap(
        [0.7, 0.3, 0.0], [0.2, 0.5, 0.3]))) < 1e-15
    tiny = 2.0 ** -60
    rt, mt = residual_distribution([1.0 - tiny, tiny], [1.0, 0.0])
    assert mt == tiny and rt == [0.0, 1.0]

    spatial = DirectionalResidualProbe(mode="spatial")
    context_only = DirectionalResidualProbe(mode="context_only")
    assert spatial.parameter_count == context_only.parameter_count == 308
    for rank in DirectionalResidualProbe._ALLOWED["rank"]:
        for groups in DirectionalResidualProbe._ALLOWED["neighbor_groups"]:
            for buckets in DirectionalResidualProbe._ALLOWED["logit_buckets"]:
                left = DirectionalResidualProbe(mode="spatial", rank=rank,
                                                neighbor_groups=groups,
                                                logit_buckets=buckets)
                right = DirectionalResidualProbe(mode="context_only", rank=rank,
                                                 neighbor_groups=groups,
                                                 logit_buckets=buckets)
                assert left.parameter_count == right.parameter_count
                assert left.parameter_count <= 308
    np_fit = "skipped_no_numpy"
    try:
        np = _numpy()
    except RuntimeError:
        pass
    else:
        raw = np.zeros((2, F_VOCAB), dtype=np.float32)
        raw[0, 3], raw[1, 5] = 1.0, 1.0
        labels = np.zeros_like(raw)
        labels[0, 3], labels[1, 5] = 1.0, 1.0
        neighbor_ids = np.asarray([[f + 20, -1, -1, -1, -1, -1],
                                   [-1, f + 3000, -1, -1, -1, -1]])
        positions = np.asarray([421, 422])
        spatial.fit(raw, positions, neighbor_ids, labels, steps=2, lr=0.01)
        context_only.fit(raw, positions, neighbor_ids, labels, steps=2, lr=0.01)
        s_delta = spatial.residual_logits(raw, positions, neighbor_ids)
        c1 = context_only.residual_logits(raw, positions, neighbor_ids)
        c2 = context_only.residual_logits(raw, positions, np.full_like(neighbor_ids, -1))
        assert s_delta.shape == c1.shape == (2, F_VOCAB)
        assert np.isfinite(s_delta).all() and np.isfinite(c1).all()
        assert np.array_equal(c1, c2)
        for mode in ("spatial", "context_only"):
            small = DirectionalResidualProbe(mode=mode, rank=2,
                                             neighbor_groups=4, logit_buckets=8)
            small.fit(raw, positions, neighbor_ids, labels, steps=1, lr=0.01)
            assert sum(int(weight.size) for weight in small._weights.values()) == small.parameter_count
        np_fit = "passed_numpy_fit_inference"

    return {"passed": True, "neighbor_boundary": "passed",
            "committed_pair_and_eos": "passed", "future_feature_excluded_until_present": "passed",
            "direction_order_xyz": "passed", "zero_scale_exact_q": "passed",
            "residual_mass_and_tiny_positive": "passed", "parameter_count": 308,
            "same_parameter_modes_all_configs": "passed",
            "numpy_fit_inference": np_fit}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        import json
        print(json.dumps(_self_test(), sort_keys=True))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
