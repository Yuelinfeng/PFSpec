from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


CountRows = Tuple[Tuple[int, int], ...]
TensorRows = Tuple[torch.Tensor, torch.Tensor]


class StructuredPositionGapDraft:
    """Negligible-cost Position proposal from an exported gap/context table.

    lookup_mode=python preserves the original tuple/dictionary path.
    lookup_mode=tensor uses integer context codes plus device-resident gap
    and count tensors for one proposal at a time.
    lookup_mode=vectorized resolves a batch of contexts and performs the
    support mapping and scatter-add in one PyTorch operation.

    The table remains a gap model: every row is shifted by the last Position
    before support mapping. Target Feature proposals and correction logic stay
    outside this adapter.
    """

    LOOKUP_MODES = {"python", "tensor", "vectorized"}

    def __init__(
        self,
        model_path: str | Path,
        *,
        position_token_ids: torch.Tensor,
        position_vocab_size: int,
        eos_token_id: int,
        lookup_mode: str = "python",
    ) -> None:
        if lookup_mode not in self.LOOKUP_MODES:
            raise ValueError(f"lookup_mode must be one of {sorted(self.LOOKUP_MODES)}, got {lookup_mode!r}")
        payload = json.loads(Path(model_path).read_text())
        self.model_path = str(model_path)
        self.lookup_mode = str(lookup_mode)
        self.history_len = int(payload["history_len"])
        feature = payload.get("feature", {})
        self.feature_bits = int(feature.get("bits", 0))
        self.feature_mask = int(feature.get("mask", (1 << self.feature_bits) - 1))
        self.position_vocab_size = int(position_vocab_size)
        self.eos_token_id = int(eos_token_id)
        self.position_token_ids = position_token_ids.detach().to(dtype=torch.long)
        self.token_to_slot = {
            int(token): slot for slot, token in enumerate(self.position_token_ids.tolist())
        }
        self.eos_slot = self.token_to_slot.get(self.eos_token_id)

        self.tables: Dict[Tuple[Tuple[int, ...], int], CountRows] = {}
        for item in payload.get("tables", []):
            key = (
                tuple(int(x) for x in item.get("history", [])),
                int(item["feature"]),
            )
            self.tables[key] = tuple(
                (int(gap), int(count)) for gap, count in item.get("counts", [])
            )
        self.feature_counts: Dict[int, CountRows] = {
            int(feature_key): tuple(
                (int(gap), int(count)) for gap, count in counts
            )
            for feature_key, counts in payload.get("feature_counts", {}).items()
        }
        self.global_counts: CountRows = tuple(
            (int(gap), int(count)) for gap, count in payload.get("global_counts", [])
        )

        # Collision-free integer encoding for (history, feature). Gaps can be
        # negative in malformed/rollback contexts, so reserve an offset.
        self._gap_offset = self.position_vocab_size + 1
        self._gap_base = max(2 * self.position_vocab_size + 3, 8)
        self._length_base = max(self.history_len + 1, 1)
        self._table_codes: Dict[int, CountRows] = {
            self._encode_context(history, feature): counts
            for (history, feature), counts in self.tables.items()
        }
        self._feature_codes: Dict[int, CountRows] = dict(self.feature_counts)

        # CPU tensors are created once. Device copies are lazily materialized
        # per device and retained for the lifetime of this adapter.
        self._table_cpu: Dict[int, TensorRows] = {
            code: self._tensorize_counts(counts)
            for code, counts in self._table_codes.items()
        }
        self._feature_cpu: Dict[int, TensorRows] = {
            feature_key: self._tensorize_counts(counts)
            for feature_key, counts in self._feature_codes.items()
        }
        self._global_cpu: TensorRows = self._tensorize_counts(self.global_counts)
        self._device_cache: Dict[str, Dict[str, object]] = {}

        self.calls = 0
        self.lookup_us = 0.0
        self.backoff_counts: Dict[str, int] = {}

    @staticmethod
    def _tensorize_counts(counts: CountRows) -> TensorRows:
        if not counts:
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((0,), dtype=torch.float32),
            )
        return (
            torch.tensor([gap for gap, _ in counts], dtype=torch.long),
            torch.tensor([count for _, count in counts], dtype=torch.float32),
        )

    def _encode_context(self, history: Sequence[int], feature: int) -> int:
        code = int(feature)
        for gap in history:
            code = code * self._gap_base + int(gap) + self._gap_offset
        return code * self._length_base + len(history)

    def _device_tensors(self, device: torch.device) -> Dict[str, object]:
        device_key = str(device)
        cached = self._device_cache.get(device_key)
        if cached is not None:
            return cached

        def move_rows(rows: Dict[int, TensorRows]) -> Dict[int, TensorRows]:
            return {
                key: (gaps.to(device=device), counts.to(device=device))
                for key, (gaps, counts) in rows.items()
            }

        table_rows = move_rows(self._table_cpu)
        feature_rows = move_rows(self._feature_cpu)
        global_rows = (
            self._global_cpu[0].to(device=device),
            self._global_cpu[1].to(device=device),
        )
        token_ids = self.position_token_ids.to(device=device)
        max_token = max(
            int(self.position_vocab_size),
            int(self.position_token_ids.max().item()) if self.position_token_ids.numel() else 0,
        )
        token_to_slot = torch.full(
            (max_token + 1,), -1, dtype=torch.long, device=device
        )
        token_to_slot[token_ids] = torch.arange(
            token_ids.numel(), dtype=torch.long, device=device
        )
        cached = {
            "table": table_rows,
            "feature": feature_rows,
            "global": global_rows,
            "token_to_slot": token_to_slot,
        }
        self._device_cache[device_key] = cached
        return cached

    def _select_counts_python(
        self, positions: Sequence[int]
    ) -> Tuple[int, CountRows, str, object]:
        if positions:
            last = int(positions[-1])
            gaps = [
                int(positions[i]) - int(positions[i - 1])
                for i in range(1, len(positions))
            ]
            feature = last & self.feature_mask
        else:
            # Treat the virtual predecessor as -1 so a gap of one can propose
            # Position 0 for the first object.
            last = -1
            gaps = []
            feature = 0
        max_history = min(self.history_len, len(gaps))
        for n in range(max_history, -1, -1):
            key = (tuple(gaps[-n:]) if n else tuple(), feature)
            counts = self.tables.get(key)
            if counts:
                name = f"table_h{n}"
                self.backoff_counts[name] = self.backoff_counts.get(name, 0) + 1
                return last, counts, "table", key
        counts = self.feature_counts.get(feature)
        if counts:
            self.backoff_counts["feature"] = self.backoff_counts.get("feature", 0) + 1
            return last, counts, "feature", feature
        self.backoff_counts["global"] = self.backoff_counts.get("global", 0) + 1
        return last, self.global_counts, "global", None

    def _select_counts_encoded(
        self, positions: Sequence[int]
    ) -> Tuple[int, CountRows, str, object]:
        if positions:
            last = int(positions[-1])
            gaps = [
                int(positions[i]) - int(positions[i - 1])
                for i in range(1, len(positions))
            ]
            feature = last & self.feature_mask
        else:
            last = -1
            gaps = []
            feature = 0
        max_history = min(self.history_len, len(gaps))
        for n in range(max_history, -1, -1):
            history = gaps[-n:] if n else ()
            code = self._encode_context(history, feature)
            counts = self._table_codes.get(code)
            if counts:
                name = f"table_h{n}"
                self.backoff_counts[name] = self.backoff_counts.get(name, 0) + 1
                return last, counts, "table", code
        counts = self._feature_codes.get(feature)
        if counts:
            self.backoff_counts["feature"] = self.backoff_counts.get("feature", 0) + 1
            return last, counts, "feature", feature
        self.backoff_counts["global"] = self.backoff_counts.get("global", 0) + 1
        return last, self.global_counts, "global", None

    def _select_counts(
        self, positions: Sequence[int], *, encoded: bool
    ) -> Tuple[int, CountRows, str, object]:
        if encoded:
            return self._select_counts_encoded(positions)
        return self._select_counts_python(positions)

    def _finish_single_python(self, q: torch.Tensor, last: int) -> torch.Tensor:
        if float(q.sum().item()) <= 0.0:
            next_token = last + 1
            slot = self.token_to_slot.get(next_token)
            if slot is not None:
                q[slot] = 1.0
            elif self.eos_slot is not None:
                q[self.eos_slot] = 1.0
            else:
                q.fill_(1.0)
        q /= q.sum()
        return q

    def _finish_single_tensor(
        self, q: torch.Tensor, last: int, token_to_slot: torch.Tensor
    ) -> torch.Tensor:
        if float(q.sum().item()) <= 0.0:
            next_token = last + 1
            if 0 <= next_token < token_to_slot.numel():
                slot = int(token_to_slot[next_token].item())
            else:
                slot = -1
            if slot >= 0:
                q[slot] = 1.0
            elif self.eos_slot is not None:
                q[self.eos_slot] = 1.0
            else:
                q.fill_(1.0)
        q /= q.sum()
        return q

    def _query_python(self, positions: Sequence[int], *, device: torch.device) -> torch.Tensor:
        start = time.perf_counter()
        last, counts, _, _ = self._select_counts(positions, encoded=False)
        q = torch.zeros(
            (len(self.position_token_ids),), device=device, dtype=torch.float32
        )
        for gap, count in counts:
            token = last + int(gap)
            if token <= last or token < 0 or token >= self.position_vocab_size:
                continue
            slot = self.token_to_slot.get(token)
            if slot is not None:
                q[slot] += float(count)
        q = self._finish_single_python(q, last)
        self.calls += 1
        self.lookup_us += (time.perf_counter() - start) * 1e6
        return q.unsqueeze(0)

    def _query_tensor(self, positions: Sequence[int], *, device: torch.device) -> torch.Tensor:
        start = time.perf_counter()
        last, counts, source, key = self._select_counts(positions, encoded=True)
        cache = self._device_tensors(device)
        rows = cache[source]
        if source == "global":
            gaps, values = rows
        else:
            gaps, values = rows.get(key, self._global_cpu)
        q = torch.zeros(
            (len(self.position_token_ids),), device=device, dtype=torch.float32
        )
        if gaps.numel():
            tokens = int(last) + gaps
            token_to_slot = cache["token_to_slot"]
            safe_tokens = tokens.clamp(0, token_to_slot.numel() - 1)
            slots = token_to_slot[safe_tokens]
            valid = (
                (tokens > int(last))
                & (tokens >= 0)
                & (tokens < self.position_vocab_size)
                & (slots >= 0)
            )
            if bool(valid.any().item()):
                q.index_add_(0, slots[valid], values[valid])
        q = self._finish_single_tensor(q, last, cache["token_to_slot"])
        self.calls += 1
        self.lookup_us += (time.perf_counter() - start) * 1e6
        return q.unsqueeze(0)

    def query_batch(
        self, positions_batch: Sequence[Sequence[int]], *, device: torch.device
    ) -> torch.Tensor:
        """Return B Position distributions with one batched support mapping."""
        start = time.perf_counter()
        selected = [
            self._select_counts(positions, encoded=True)
            for positions in positions_batch
        ]
        batch_size = len(selected)
        vocab_slots = len(self.position_token_ids)
        if batch_size == 0:
            return torch.empty((0, vocab_slots), device=device, dtype=torch.float32)
        max_rows = max(1, max(len(item[1]) for item in selected))
        gaps = torch.zeros((batch_size, max_rows), dtype=torch.long, device=device)
        values = torch.zeros((batch_size, max_rows), dtype=torch.float32, device=device)
        lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        lasts = torch.tensor(
            [int(item[0]) for item in selected], dtype=torch.long, device=device
        )
        for row, (_, counts, _, _) in enumerate(selected):
            if counts:
                row_gaps = torch.tensor(
                    [gap for gap, _ in counts], dtype=torch.long, device=device
                )
                row_values = torch.tensor(
                    [count for _, count in counts], dtype=torch.float32, device=device
                )
                length = len(counts)
                gaps[row, :length] = row_gaps
                values[row, :length] = row_values
                lengths[row] = length

        cache = self._device_tensors(device)
        token_to_slot = cache["token_to_slot"]
        tokens = lasts.unsqueeze(1) + gaps
        safe_tokens = tokens.clamp(0, token_to_slot.numel() - 1)
        slots = token_to_slot[safe_tokens]
        valid = (
            (torch.arange(max_rows, device=device).unsqueeze(0) < lengths.unsqueeze(1))
            & (tokens > lasts.unsqueeze(1))
            & (tokens >= 0)
            & (tokens < self.position_vocab_size)
            & (slots >= 0)
        )
        safe_slots = slots.clamp_min(0)
        weighted = torch.where(valid, values, torch.zeros_like(values))
        q = torch.zeros((batch_size, vocab_slots), device=device, dtype=torch.float32)
        q.scatter_add_(1, safe_slots, weighted)
        sums = q.sum(dim=1)
        zero_rows = torch.nonzero(sums <= 0.0, as_tuple=False).flatten().tolist()
        for row in zero_rows:
            next_token = int(lasts[row].item()) + 1
            slot = int(token_to_slot[next_token].item()) if 0 <= next_token < token_to_slot.numel() else -1
            if slot >= 0:
                q[row, slot] = 1.0
            elif self.eos_slot is not None:
                q[row, self.eos_slot] = 1.0
            else:
                q[row].fill_(1.0)
        q /= q.sum(dim=1, keepdim=True)
        self.calls += batch_size
        self.lookup_us += (time.perf_counter() - start) * 1e6
        return q

    def query(self, positions: Sequence[int], *, device: torch.device) -> torch.Tensor:
        if self.lookup_mode == "python":
            return self._query_python(positions, device=device)
        if self.lookup_mode == "tensor":
            return self._query_tensor(positions, device=device)
        return self.query_batch([positions], device=device)

    def summary(self) -> dict:
        return {
            "model_path": self.model_path,
            "lookup_mode": self.lookup_mode,
            "history_len": self.history_len,
            "feature_bits": self.feature_bits,
            "position_vocab_size": self.position_vocab_size,
            "table_count": len(self.tables),
            "integer_key_table_count": len(self._table_codes),
            "calls": self.calls,
            "lookup_mean_us": self.lookup_us / self.calls if self.calls else None,
            "device_cache_count": len(self._device_cache),
            "backoff_counts": dict(sorted(self.backoff_counts.items())),
        }
