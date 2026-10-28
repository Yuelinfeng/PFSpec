#!/usr/bin/env python3
"""CPU 代数参考与可选候选线性层核验。无需安装依赖。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import random
import sys
from pathlib import Path
from types import SimpleNamespace


POSITION_VOCAB = 8000
FEATURE_VOCAB = 4096
EOS = POSITION_VOCAB + FEATURE_VOCAB
PAD = EOS + 1
VOCAB_SIZE = PAD + 1
SOFTCAP = 15.0
SEED = 20261020
HIDDEN = 5


def typed_kinds(start: int, count: int) -> list[str]:
    return ["position" if (start + j) % 2 == 0 else "feature" for j in range(count)]


def ids_for(kind: str) -> list[int]:
    if kind == "position":
        return list(range(POSITION_VOCAB)) + [EOS]
    if kind == "feature":
        return list(range(POSITION_VOCAB, EOS))
    raise ValueError(f"unsupported kind: {kind}")


def column_to_token(kind: str, column: int) -> int:
    if kind == "position":
        if 0 <= column < POSITION_VOCAB:
            return column
        if column == POSITION_VOCAB:
            return EOS
    elif kind == "feature" and 0 <= column < FEATURE_VOCAB:
        return POSITION_VOCAB + column
    raise IndexError((kind, column))


def token_to_column(kind: str, token: int) -> int:
    if kind == "position":
        if 0 <= token < POSITION_VOCAB:
            return token
        if token == EOS:
            return POSITION_VOCAB
    elif kind == "feature" and POSITION_VOCAB <= token < EOS:
        return token - POSITION_VOCAB
    raise ValueError((kind, token))


def scalar_dot(left: list[float], right: list[float]) -> float:
    total = 0.0
    for a, b in zip(left, right):
        total += a * b
    return total


def softcap(value: float) -> float:
    return SOFTCAP * math.tanh(value / SOFTCAP)


def project_rowwise(
    inputs: list[list[float]], kinds: list[str], weights: list[list[float]]
) -> list[list[float]]:
    result = [[-math.inf] * VOCAB_SIZE for _ in inputs]
    for row, kind in enumerate(kinds):
        for column, token in enumerate(ids_for(kind)):
            value = scalar_dot(inputs[row], weights[token])
            result[row][token] = softcap(value)
            if column_to_token(kind, column) != token:
                raise AssertionError("rowwise vocabulary map changed")
    return result


def project_grouped_reference(
    inputs: list[list[float]], kinds: list[str], weights: list[list[float]]
) -> list[list[float]]:
    result = [[-math.inf] * VOCAB_SIZE for _ in inputs]
    for kind in ("position", "feature"):
        rows = [row for row, row_kind in enumerate(kinds) if row_kind == kind]
        token_ids = ids_for(kind)
        for row in rows:
            for column, token in enumerate(token_ids):
                value = scalar_dot(inputs[row], weights[token])
                result[row][token] = softcap(value)
                if token_to_column(kind, column_to_token(kind, column)) != column:
                    raise AssertionError("grouped vocabulary map did not round-trip")
    return result


def apply_slot_mask(logits: list[list[float]], start: int) -> None:
    for row_idx, row in enumerate(logits):
        is_position = (start + row_idx) % 2 == 0
        row[PAD] = -math.inf
        if is_position:
            for token in range(POSITION_VOCAB, EOS):
                row[token] = -math.inf
        else:
            for token in range(POSITION_VOCAB):
                row[token] = -math.inf
            row[EOS] = -math.inf


def apply_monotonic_mask(
    logits: list[float], previous_position: int, rank_map: list[int]
) -> None:
    previous_rank = rank_map[previous_position]
    for token in range(POSITION_VOCAB):
        if rank_map[token] <= previous_rank:
            logits[token] = -math.inf


def stable_probability_sum(logits: list[float]) -> tuple[float, int]:
    finite = [value for value in logits if math.isfinite(value)]
    if not finite:
        raise AssertionError("slot has no finite logits")
    maximum = max(finite)
    values = [math.exp(value - maximum) for value in finite]
    total = sum(values)
    return sum(value / total for value in values), len(finite)


def valid_sequence(tokens: list[int]) -> bool:
    body = list(tokens)
    if EOS in body:
        if body[-1] != EOS or body.count(EOS) != 1:
            return False
        body.pop()
        if len(body) % 2:
            return False
    positions = body[::2]
    features = body[1::2]
    return (
        all(0 <= token < POSITION_VOCAB for token in positions)
        and all(POSITION_VOCAB <= token < EOS for token in features)
        and all(b > a for a, b in zip(positions, positions[1:]))
    )


def assert_same_logits(
    left: list[list[float]], right: list[list[float]], *, tolerance: float = 0.0
) -> float:
    max_error = 0.0
    for left_row, right_row in zip(left, right):
        for a, b in zip(left_row, right_row):
            if math.isinf(a) or math.isinf(b):
                if a != b:
                    raise AssertionError("infinity support differs")
            else:
                error = abs(a - b)
                max_error = max(max_error, error)
                if error > tolerance:
                    raise AssertionError(f"logit error {error} exceeds {tolerance}")
    return max_error


def run_standard_tests() -> dict:
    roundtrip_count = 0
    for kind in ("position", "feature"):
        for column, token in enumerate(ids_for(kind)):
            if column_to_token(kind, column) != token:
                raise AssertionError("column-to-token conversion differs")
            if token_to_column(kind, token) != column:
                raise AssertionError("token-to-column conversion differs")
            roundtrip_count += 1

    windows = [(start, count) for start in (0, 1, 4, 5) for count in (1, 2, 3, 4)]
    tail_cases = {(4, 2), (5, 2)}
    if not tail_cases.issubset(set(windows)):
        raise AssertionError("two-row tail windows are missing")
    tested_rows = 0
    row_scatter_roundtrips = 0
    expected_by_case: dict[str, list[str]] = {}
    for start, count in windows:
        kinds = typed_kinds(start, count)
        expected = ["position" if (start + j) % 2 == 0 else "feature" for j in range(count)]
        if kinds != expected:
            raise AssertionError("typed rows do not follow absolute cache parity")
        grouped_rows = {
            kind: [row for row, row_kind in enumerate(kinds) if row_kind == kind]
            for kind in ("position", "feature")
        }
        scattered_rows = [
            row for kind in ("position", "feature") for row in grouped_rows[kind]
        ]
        if sorted(scattered_rows) != list(range(count)):
            raise AssertionError("group row map lost or duplicated an absolute row")
        for kind, rows in grouped_rows.items():
            if any(kinds[row] != kind for row in rows):
                raise AssertionError("group row map returned a different slot type")
        row_scatter_roundtrips += count
        expected_by_case[f"start={start},count={count}"] = kinds
        tested_rows += count

    rng = random.Random(SEED)
    weights = [
        [rng.uniform(-1.0, 1.0) for _ in range(HIDDEN)]
        for _ in range(VOCAB_SIZE)
    ]
    projection_cases = [(0, 4), (1, 4), (4, 2), (5, 2), (3, 1), (2, 3)]
    max_projection_error = 0.0
    masked_row_count = 0
    minimum_probability_sum_error = 1.0
    maximum_probability_sum_error = 0.0
    support_counts: dict[str, int] = {}

    for case_index, (start, count) in enumerate(projection_cases):
        inputs = [
            [rng.uniform(-1.5, 1.5) for _ in range(HIDDEN)]
            for _ in range(count)
        ]
        kinds = typed_kinds(start, count)
        original = project_rowwise(inputs, kinds, weights)
        grouped = project_grouped_reference(inputs, kinds, weights)
        max_projection_error = max(
            max_projection_error, assert_same_logits(original, grouped)
        )

        apply_slot_mask(original, start)
        apply_slot_mask(grouped, start)
        assert_same_logits(original, grouped)

        for row_idx, kind in enumerate(kinds):
            row = grouped[row_idx]
            total, support = stable_probability_sum(row)
            error = abs(total - 1.0)
            minimum_probability_sum_error = min(minimum_probability_sum_error, error)
            maximum_probability_sum_error = max(maximum_probability_sum_error, error)
            expected_support = POSITION_VOCAB + 1 if kind == "position" else FEATURE_VOCAB
            if support != expected_support:
                raise AssertionError(f"wrong support for {kind}: {support}")
            support_counts[kind] = support_counts.get(kind, 0) + 1
            masked_row_count += 1

    xyz = list(range(POSITION_VOCAB))
    xzy = []
    side = 20
    for token in range(POSITION_VOCAB):
        z = token % side
        y = (token // side) % side
        x = (token // (side * side)) % side
        xzy.append(x * side * side + z * side + y)
    if sorted(xzy) != xyz:
        raise AssertionError("xzy rank map is not a permutation")

    monotonic_row = [-math.inf] * VOCAB_SIZE
    for token in range(POSITION_VOCAB):
        monotonic_row[token] = 0.0
    monotonic_row[EOS] = 0.0
    monotonic_row[PAD] = 0.0
    apply_slot_mask([monotonic_row], start=0)
    apply_monotonic_mask(monotonic_row, previous_position=10, rank_map=xyz)
    if any(math.isfinite(monotonic_row[token]) for token in range(11)):
        raise AssertionError("monotonic mask kept a non-increasing position")
    if not math.isfinite(monotonic_row[11]) or not math.isfinite(monotonic_row[EOS]):
        raise AssertionError("monotonic mask removed a later Position or EOS")
    if math.isfinite(monotonic_row[PAD]):
        raise AssertionError("slot mask kept PAD")
    monotonic_probability_sum, monotonic_support = stable_probability_sum(monotonic_row)

    xzy_monotonic_row = [-math.inf] * VOCAB_SIZE
    for token in range(POSITION_VOCAB):
        xzy_monotonic_row[token] = 0.0
    xzy_monotonic_row[EOS] = 0.0
    xzy_monotonic_row[PAD] = 0.0
    apply_slot_mask([xzy_monotonic_row], start=0)
    previous_xzy_rank = xzy[10]
    apply_monotonic_mask(xzy_monotonic_row, previous_position=10, rank_map=xzy)
    for token in range(POSITION_VOCAB):
        if xzy[token] <= previous_xzy_rank and math.isfinite(xzy_monotonic_row[token]):
            raise AssertionError("xzy monotonic mask kept a non-increasing position")
        if xzy[token] > previous_xzy_rank and not math.isfinite(xzy_monotonic_row[token]):
            raise AssertionError("xzy monotonic mask removed an increasing position")
    if not math.isfinite(xzy_monotonic_row[EOS]):
        raise AssertionError("xzy monotonic mask removed EOS")

    eos_cases = {
        "complete_pair_then_eos": valid_sequence([12, 8000, 13, 8001, EOS]),
        "eos_after_only_position": valid_sequence([12, EOS]),
        "eos_not_terminal": valid_sequence([12, 8000, EOS, 8001]),
        "feature_id_in_position_slot": valid_sequence([8000, 8001]),
    }
    if eos_cases != {
        "complete_pair_then_eos": True,
        "eos_after_only_position": False,
        "eos_not_terminal": False,
        "feature_id_in_position_slot": False,
    }:
        raise AssertionError(f"EOS sequence semantics differ: {eos_cases}")
    if EOS not in ids_for("position") or EOS in ids_for("feature"):
        raise AssertionError("EOS vocabulary support differs by type")

    return {
        "status": "passed",
        "backend": "Python standard library; scalar float64 reference",
        "seed": SEED,
        "config": {
            "position_vocab": POSITION_VOCAB,
            "feature_vocab": FEATURE_VOCAB,
            "eos_id": EOS,
            "pad_id": PAD,
            "vocab_size": VOCAB_SIZE,
            "hidden_width": HIDDEN,
            "softcap": SOFTCAP,
        },
        "vocabulary_roundtrips": roundtrip_count,
        "row_scatter_roundtrips": row_scatter_roundtrips,
        "slot_windows": {
            "cases": len(windows),
            "rows": tested_rows,
            "parities": [0, 1],
            "lengths": [1, 2, 3, 4],
            "tail_length_two_cases": len(tail_cases),
            "types": expected_by_case,
        },
        "projection_cases": len(projection_cases),
        "max_projection_abs_error": max_projection_error,
        "projection_precision": "Python float64; identical scalar accumulation order",
        "slot_mask_rows": masked_row_count,
        "support_rows": support_counts,
        "probability_row_sum_error": {
            "max_abs": maximum_probability_sum_error,
            "min_abs": minimum_probability_sum_error,
        },
        "monotonic_mask": {
            "rank_map_permutations_checked": ["xyz", "xzy"],
            "previous_position": 10,
            "positions_masked": 11,
            "next_position_kept": True,
            "eos_kept": True,
            "finite_support": monotonic_support,
            "probability_sum": monotonic_probability_sum,
            "xzy_increasing_ids_retained": sum(
                1 for token in range(POSITION_VOCAB) if xzy[token] > xzy[10]
            ),
        },
        "eos_cases": eos_cases,
        "limitations": [
            "This reference does not execute the model's PyTorch forward path.",
            "Python float64 equality does not bound PyTorch CPU/GPU FP32 GEMM differences.",
            "No checkpoint, KV cache, CUDA device, or full decoder was loaded.",
        ],
    }


def _torch_ids(torch, kind: str):
    if kind == "position":
        return torch.cat(
            (
                torch.arange(POSITION_VOCAB, dtype=torch.long),
                torch.tensor([EOS], dtype=torch.long),
            )
        )
    return torch.arange(POSITION_VOCAB, EOS, dtype=torch.long)


def _torch_rowwise(torch, functional, model, x, kinds):
    logits = torch.full(
        (x.size(0), x.size(1), VOCAB_SIZE),
        -torch.inf,
        dtype=torch.float32,
        device="cpu",
    )
    for row, kind in enumerate(kinds):
        token_ids = _torch_ids(torch, kind)
        selected_weight = model.lm_head.weight.index_select(0, token_ids)
        selected = functional.linear(x[:, row, :], selected_weight)
        selected = SOFTCAP * torch.tanh(selected.float() / SOFTCAP)
        logits[:, row, :].index_copy_(1, token_ids, selected)
    return logits


def _finite_max_error(torch, expected, actual) -> tuple[float, float]:
    expected_finite = torch.isfinite(expected)
    actual_finite = torch.isfinite(actual)
    if not torch.equal(expected_finite, actual_finite):
        raise AssertionError("candidate finite support differs before masking")
    if not torch.equal(torch.isneginf(expected), torch.isneginf(actual)):
        raise AssertionError("candidate negative-infinity support differs")
    values = expected_finite
    if not bool(values.any()):
        return 0.0, 0.0
    differences = (expected[values] - actual[values]).abs()
    max_abs = float(differences.max().item())
    denom = expected[values].abs().clamp_min(1.0)
    max_rel = float((differences / denom).max().item())
    return max_abs, max_rel


def run_torch_candidate(candidate_path: Path, repo_root: Path) -> dict:
    import torch
    import torch.nn.functional as functional

    if not hasattr(torch, "nn"):
        raise RuntimeError("PyTorch nn is unavailable")
    candidate_path = candidate_path.resolve()
    repo_root = repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    module_name = "model.gpt.nanochat_gpt_pf2x_candidate"
    spec = importlib.util.spec_from_file_location(module_name, candidate_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate source: {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    gpt_class = getattr(module, "GPT")
    if not hasattr(gpt_class, "_project_typed_logits_grouped"):
        raise AttributeError("candidate GPT lacks _project_typed_logits_grouped")

    class MiniGPT(gpt_class):
        pass

    model = MiniGPT.__new__(MiniGPT)
    torch.nn.Module.__init__(model)
    torch.manual_seed(SEED)
    model.lm_head = torch.nn.Linear(HIDDEN, VOCAB_SIZE, bias=False, device="cpu")
    model.config = SimpleNamespace(vocab_size=VOCAB_SIZE)
    model.sparse_position_vocab_size = POSITION_VOCAB
    model.sparse_feature_vocab_size = FEATURE_VOCAB
    model.sparse_feature_token_offset = POSITION_VOCAB
    model.sparse_shared_vocab = False
    model.sparse_eos_token_id = EOS
    model.sparse_pad_token_id = PAD
    model.sparse_num_position_tokens = 1
    model.sparse_tokens_per_latent = 2
    model.sparse_position_hard_constraints = True
    model.sparse_rope_side_length = 20
    model.sparse_position_rank_map = torch.arange(POSITION_VOCAB, dtype=torch.long)
    model.dense_chunk_order = "xyz"
    model.rope_layout = SimpleNamespace(is_sparse=True)

    starts_and_counts = [(0, 4), (1, 4), (4, 2), (5, 2), (3, 1), (2, 3)]
    torch.manual_seed(SEED + 1)
    maximum_abs = 0.0
    maximum_relative = 0.0
    post_mask_maximum_abs = 0.0
    probability_rows = 0
    tolerance = 1e-5

    class CpuCache:
        def __init__(self, tokens):
            self.tokens = tokens

        def get_tokens(self, count):
            return self.tokens[:, :count]

    for start, count in starts_and_counts:
        kinds = typed_kinds(start, count)
        x = torch.randn((1, count, HIDDEN), dtype=torch.float32, device="cpu")
        expected = _torch_rowwise(torch, functional, model, x, kinds)
        actual = model._project_typed_logits_grouped(x, kinds, softcap=SOFTCAP)
        if tuple(actual.shape) != (1, count, VOCAB_SIZE):
            raise AssertionError(f"candidate shape differs: {tuple(actual.shape)}")
        if actual.dtype != torch.float32 or actual.device.type != "cpu":
            raise AssertionError("candidate must return CPU float32 logits")
        abs_error, rel_error = _finite_max_error(torch, expected, actual)
        maximum_abs = max(maximum_abs, abs_error)
        maximum_relative = max(maximum_relative, rel_error)
        if not torch.allclose(
            expected, actual, atol=tolerance, rtol=tolerance, equal_nan=False
        ):
            raise AssertionError(
                f"candidate differs beyond CPU tolerance: abs={abs_error} rel={rel_error}"
            )

        expected_masked = expected.clone()
        actual_masked = actual.clone()
        model._apply_slot_logits_mask(expected_masked, start_pos=start)
        model._apply_slot_logits_mask(actual_masked, start_pos=start)
        masked_abs, _ = _finite_max_error(torch, expected_masked, actual_masked)
        post_mask_maximum_abs = max(post_mask_maximum_abs, masked_abs)

        full_token_list = [EOS]
        for pair in range(8):
            full_token_list.extend([5 + pair, POSITION_VOCAB + pair])
        full_tokens = torch.tensor([full_token_list[: start + count]], dtype=torch.long)
        cache_idx = full_tokens[:, start : start + count]
        cache = CpuCache(full_tokens)
        expected_monotonic = expected_masked.clone()
        actual_monotonic = actual_masked.clone()
        model._apply_monotonic_position_mask(
            expected_monotonic,
            idx=cache_idx,
            start_pos=start,
            kv_cache=cache,
        )
        model._apply_monotonic_position_mask(
            actual_monotonic,
            idx=cache_idx,
            start_pos=start,
            kv_cache=cache,
        )
        monotonic_abs, _ = _finite_max_error(torch, expected_monotonic, actual_monotonic)
        post_mask_maximum_abs = max(post_mask_maximum_abs, monotonic_abs)
        for row in range(count):
            if not bool(torch.isfinite(actual_monotonic[0, row]).any()):
                raise AssertionError("candidate mask removed every legal token")
            probs = torch.softmax(actual_monotonic[0, row], dim=0)
            if not torch.isfinite(probs).all() or abs(float(probs.sum()) - 1.0) > 1e-6:
                raise AssertionError("candidate masked probability row does not normalize")
            probability_rows += 1

    eos_position = torch.zeros((1, 1, VOCAB_SIZE), dtype=torch.float32)
    eos_position[..., EOS] = 0.25
    model._apply_slot_logits_mask(eos_position, start_pos=0)
    eos_feature = torch.zeros((1, 1, VOCAB_SIZE), dtype=torch.float32)
    eos_feature[..., EOS] = 0.25
    model._apply_slot_logits_mask(eos_feature, start_pos=1)
    if not torch.isfinite(eos_position[0, 0, EOS]):
        raise AssertionError("Position slot must preserve EOS")
    if not torch.isneginf(eos_feature[0, 0, EOS]):
        raise AssertionError("Feature slot must exclude EOS")

    return {
        "status": "passed",
        "backend": "real PyTorch CPU",
        "candidate_path": str(candidate_path),
        "seed": SEED,
        "cases": len(starts_and_counts),
        "tolerance_atol_rtol": tolerance,
        "max_abs_error_before_masks": maximum_abs,
        "max_relative_error_before_masks_scaled_by_max_1": maximum_relative,
        "max_abs_error_after_slot_and_monotonic_masks": post_mask_maximum_abs,
        "normalized_probability_rows": probability_rows,
        "eos_position_allowed": True,
        "eos_feature_excluded": True,
        "limitations": [
            "Synthetic CPU Linear weights only; no checkpoint or model trunk loaded.",
            "This does not measure CUDA FP32 differences or runtime speed.",
            "The CPU cache fixture supplies token history to the model mask methods; no KV tensors are computed.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu",), default="cpu")
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = {
        "script": "patches/pf2x_round01/test_pf_typed_verify.py",
        "runtime": {
            "python": platform.python_version(),
            "torch_available": importlib.util.find_spec("torch") is not None,
        },
        "standard_library": run_standard_tests(),
        "torch_candidate": {"status": "not_requested"},
    }
    return_code = 0
    if args.candidate is not None:
        try:
            result["torch_candidate"] = run_torch_candidate(args.candidate, args.repo_root)
        except ModuleNotFoundError as exc:
            result["torch_candidate"] = {
                "status": "unavailable",
                "reason": f"missing runtime module: {exc.name}",
            }
            return_code = 1
        except Exception as exc:
            result["torch_candidate"] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            return_code = 1

    output = args.output
    if output is None:
        repo_root = Path(__file__).resolve().parents[2]
        output = (
            repo_root
            / "docs/project/artifacts/pf2x_loop_20261019/round02/results/tests_cpu.json"
        )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
