"""Small adapter that adds one residual before the existing Feature policy."""
from __future__ import annotations

import time
import statistics

from transition_model import (
    EOS_ID,
    FEATURE_OFFSET,
    POSITION_VOCAB,
    TEMPERATURE,
    TOP_P,
    policy_batch,
)


class TransitionAdapter:
    """Keep the existing MLP proposal and add a two-input residual head."""

    def __init__(self, feature, model, bf16: bool = False, residual_scale: float = 1.0):
        self.feature = feature
        self.model = model
        self.bf16 = bool(bf16)
        self.residual_scale = float(residual_scale)
        self.temperature = TEMPERATURE
        self.top_p = TOP_P
        self.variant = f"{model.method}_residual"
        self.feature_precision = "BF16 autocast" if self.bf16 else "FP32"
        self.forward_calls = 0
        self.input_tokens = 0
        self._profile = None
        self.profile_stages = False
        self.stage_seconds = {"old_mlp": 0.0, "transition": 0.0, "feature_policy": 0.0}
        self._prepared = False

    def _run(self, stage, fn, *args, **kwargs):
        operation = kwargs.get("operation", "")
        started = time.perf_counter() if self.profile_stages else None
        if self._profile is None:
            for key in ("gpu", "input_tokens", "operation", "context_tokens"):
                kwargs.pop(key, None)
            result = fn(*args, **kwargs)
        else:
            result = self._profile.run(stage, fn, *args, **kwargs)
        if started is not None:
            name = "feature_policy" if stage == "sampling" else (
                "old_mlp" if operation == "state_mlp" else "transition"
            )
            self.stage_seconds[name] += time.perf_counter() - started
        return result

    def prepare(self, prompt, capacity):
        self.forward_calls = 0
        self.input_tokens = 0
        self.stage_seconds = {"old_mlp": 0.0, "transition": 0.0, "feature_policy": 0.0}
        self._prepared = True

    def propose(self, prefix, p1, h_prev, rng):
        import torch

        if not self._prepared or h_prev is None:
            raise RuntimeError("Transition adapter needs prepared state and h_prev")
        if len(prefix) < 2:
            raise ValueError("Prefix must end in a Feature with its preceding Position")
        f0 = int(prefix[-1])
        p0 = int(prefix[-2])
        p1 = int(p1)
        if not FEATURE_OFFSET <= f0 < EOS_ID:
            raise ValueError(f"Expected global Feature ID, got {f0}")
        if not (0 <= p0 < POSITION_VOCAB and 0 <= p1 < POSITION_VOCAB):
            raise ValueError(f"Position IDs out of range: {p0}, {p1}")

        def old_forward():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):
                _, logits = self.feature(
                    h_prev.unsqueeze(0),
                    torch.tensor([f0 - FEATURE_OFFSET], device=h_prev.device),
                    torch.tensor([p1], device=h_prev.device),
                )
            return logits

        old_logits = self._run(
            "feature_forward", old_forward, gpu=True, input_tokens=1,
            operation="state_mlp",
        )
        self.forward_calls += 1
        self.input_tokens += 1

        new_logits = self._run(
            "feature_forward",
            lambda: self.model.proposal_logits(
                old_logits.float(),
                h_prev.unsqueeze(0),
                torch.tensor([f0], device=h_prev.device),
                torch.tensor([p0], device=h_prev.device),
                torch.tensor([p1], device=h_prev.device),
                residual_scale=self.residual_scale,
            ),
            gpu=True, input_tokens=1, operation="position_transition",
        )

        q = self._run(
            "sampling",
            lambda: policy_batch(new_logits.float(), self.temperature, self.top_p)[0].float(),
            gpu=True, operation="feature_policy",
        )
        if q.ndim != 1 or q.numel() != 4096:
            raise RuntimeError("Feature policy must return 4096 probabilities")
        return q

    def rewind(self, keep):
        return None

    def close(self):
        self._prepared = False


def cost_probe(
    old_call,
    new_call,
    cases,
    *,
    pairs: int = 32,
    warmup_pairs: int = 2,
    deadline_unix: float,
    synchronize=None,
):
    """Measure paired full proposal callables under one absolute deadline.

    Each callable receives one case and should invoke the complete adapter
    ``propose`` path. The caller supplies cached inputs and an optional device
    synchronization function; no Target model or weights are loaded here.
    """
    if pairs < 32:
        raise ValueError("At least 32 paired contexts are required")
    if warmup_pairs < 0:
        raise ValueError("warmup_pairs must be nonnegative")
    if not cases:
        raise ValueError("At least one prepared context is required")
    cases = list(cases)
    if synchronize is None:
        synchronize = lambda: None

    started_unix = time.time()
    warmups = 0
    for warmup_id in range(max(0, int(warmup_pairs))):
        if time.time() >= float(deadline_unix):
            break
        case = cases[warmup_id % len(cases)]
        order = ("old", "new") if warmup_id % 2 == 0 else ("new", "old")
        for label in order:
            if time.time() >= float(deadline_unix):
                break
            synchronize()
            (old_call if label == "old" else new_call)(case)
            synchronize()
            warmups += 1

    rows = []
    for pair_id in range(pairs):
        if time.time() >= float(deadline_unix):
            break
        case = cases[pair_id % len(cases)]
        order = ("old", "new") if pair_id % 2 == 0 else ("new", "old")
        times = {}
        for label in order:
            if time.time() >= float(deadline_unix):
                break
            synchronize()
            started = time.perf_counter()
            (old_call if label == "old" else new_call)(case)
            synchronize()
            times[label] = time.perf_counter() - started
        if len(times) != 2:
            break
        rows.append({
            "pair_id": pair_id,
            "sample_id": case.get("sample_id", str(pair_id)) if isinstance(case, dict) else str(pair_id),
            "order": list(order),
            "old_propose_seconds": times["old"],
            "new_propose_seconds": times["new"],
            "delta_seconds": times["new"] - times["old"],
        })
    old_values = [row["old_propose_seconds"] for row in rows]
    new_values = [row["new_propose_seconds"] for row in rows]
    deltas = [row["delta_seconds"] for row in rows]
    summary = {
        "old_mean_seconds": statistics.mean(old_values) if old_values else None,
        "new_mean_seconds": statistics.mean(new_values) if new_values else None,
        "paired_delta_mean_seconds": statistics.mean(deltas) if deltas else None,
        "old_median_seconds": statistics.median(old_values) if old_values else None,
        "new_median_seconds": statistics.median(new_values) if new_values else None,
        "paired_delta_median_seconds": statistics.median(deltas) if deltas else None,
    }
    if old_values and statistics.mean(old_values) > 0:
        summary["new_over_old_mean"] = statistics.mean(new_values) / statistics.mean(old_values)
    finished_unix = time.time()
    status = "complete" if len(rows) == pairs else "partial"
    if len(rows) == pairs and finished_unix > float(deadline_unix):
        status = "budget_overrun"
    return {
        "status": status,
        "requested_pairs": int(pairs),
        "completed_pairs": len(rows),
        "warmup_pairs_requested": int(warmup_pairs),
        "warmup_calls_completed": warmups,
        "started_unix": started_unix,
        "finished_unix": finished_unix,
        "deadline_unix": float(deadline_unix),
        "summary": summary,
        "rows": rows,
    }
