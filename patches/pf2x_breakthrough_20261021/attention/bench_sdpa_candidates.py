#!/usr/bin/env python3
"""Compare FP32 attention candidates for GaussianGPT decode-shaped inputs.

The benchmark covers query lengths 1, 2, and 4 with a cached prefix length of
64, 256, 768, or 2048. It measures the default PyTorch SDPA dispatcher, the
SDPA math backend, and explicit batched matrix multiply + softmax. Accuracy is
compared with an FP64 CPU reference, including attention probabilities. This
script is intended for the existing gaussiangpt environment and does not load
project checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


CONTEXT_LENGTHS = (64, 256, 768, 2048)
QUERY_LENGTHS = (1, 2, 4)
HEADS = 16
HEAD_DIM = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-prob-error", type=float, default=1.0e-4)
    parser.add_argument("--max-output-error", type=float, default=1.0e-4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profiling/pf2x_breakthrough_20261021/attention"),
    )
    return parser.parse_args()


def make_allowed_mask(
    context_length: int, query_length: int, device: torch.device
) -> torch.Tensor | None:
    # The one-token decode path sees its full cache. Multi-token verification
    # sees the prefix plus a causal mask within the newly supplied block.
    if query_length == 1:
        return None
    total_keys = context_length + query_length
    mask = torch.ones(
        (query_length, total_keys), dtype=torch.bool, device=device
    )
    for row in range(query_length):
        mask[row, context_length + row + 1 :] = False
    return mask


def explicit_bmm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v), probabilities


def sdpa_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        is_causal=False,
        dropout_p=0.0,
        enable_gqa=False,
    )


def backend_context(implementation: str):
    if implementation == "math":
        return sdpa_kernel(SDPBackend.MATH)
    return nullcontext()


def run_candidate(
    implementation: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if implementation == "explicit_bmm":
        return explicit_bmm(q, k, v, mask)
    return sdpa_call(q, k, v, mask), None


def recover_sdpa_probabilities(
    implementation: str,
    q: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Read SDPA probabilities through basis-valued V tensors.

    SDPA exposes the weighted value output, not its internal probabilities.
    Each basis call places up to HEAD_DIM cache positions on separate value
    channels, so the output channels recover those probabilities directly.
    This runs only during accuracy reporting, outside the timed section.
    """
    key_count = int(k.size(-2))
    chunks = []
    device = q.device
    with backend_context(implementation):
        for start in range(0, key_count, HEAD_DIM):
            count = min(HEAD_DIM, key_count - start)
            values = torch.zeros(
                (q.size(0), q.size(1), key_count, HEAD_DIM),
                dtype=q.dtype,
                device=device,
            )
            basis = torch.eye(HEAD_DIM, dtype=q.dtype, device=device)[:count]
            values[:, :, start : start + count, :] = basis.view(
                1, 1, count, HEAD_DIM
            )
            recovered = sdpa_call(q, k, values, mask)
            chunks.append(recovered[..., :count])
    return torch.cat(chunks, dim=-1)


def fp64_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q64 = q.detach().to(device="cpu", dtype=torch.float64)
    k64 = k.detach().to(device="cpu", dtype=torch.float64)
    v64 = v.detach().to(device="cpu", dtype=torch.float64)
    scores = torch.matmul(q64, k64.transpose(-2, -1)) / math.sqrt(q64.size(-1))
    if mask is not None:
        mask64 = mask.detach().to(device="cpu")
        scores = scores.masked_fill(~mask64, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, v64)
    return probabilities, output


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        (actual.detach().to(device="cpu", dtype=torch.float64) - expected)
        .abs()
        .max()
        .item()
    )


def timed_repeats(
    implementation: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[dict[str, float | int | str]]:
    samples: list[dict[str, float | int | str]] = []
    with backend_context(implementation):
        for _ in range(warmup):
            run_candidate(implementation, q, k, v, mask)
        torch.cuda.synchronize(q.device)

        for repeat in range(repeats):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(q.device)
            wall_start = time.perf_counter()
            start_event.record()
            for _ in range(iterations):
                run_candidate(implementation, q, k, v, mask)
            end_event.record()
            torch.cuda.synchronize(q.device)
            wall_elapsed_ms = (time.perf_counter() - wall_start) * 1000.0
            device_elapsed_ms = float(start_event.elapsed_time(end_event))
            samples.append(
                {
                    "implementation": implementation,
                    "repeat": repeat + 1,
                    "device_us_per_call": device_elapsed_ms * 1000.0 / iterations,
                    "wall_us_per_call": wall_elapsed_ms * 1000.0 / iterations,
                }
            )
    return samples


def run_cell(
    context_length: int,
    query_length: int,
    generator: torch.Generator,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    key_length = context_length + query_length
    q_bthd = torch.randn(
        (1, query_length, HEADS, HEAD_DIM),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    k_bthd = torch.randn(
        (1, key_length, HEADS, HEAD_DIM),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    v_bthd = torch.randn(
        (1, key_length, HEADS, HEAD_DIM),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    # The production path forms B,T,H,D tensors, applies Q/K RMSNorm, then
    # transposes to B,H,T,D. Keep those non-contiguous strides for dispatch.
    q = F.rms_norm(q_bthd, (HEAD_DIM,)).transpose(1, 2)
    k = F.rms_norm(k_bthd, (HEAD_DIM,)).transpose(1, 2)
    v = v_bthd.transpose(1, 2)
    mask = make_allowed_mask(context_length, query_length, device)
    reference_probabilities, reference_output = fp64_reference(q, k, v, mask)

    accuracy: dict[str, dict[str, float]] = {}
    for implementation in ("default_sdpa", "math_sdpa", "explicit_bmm"):
        backend_name = "math" if implementation == "math_sdpa" else implementation
        with backend_context(backend_name):
            output, probabilities = run_candidate(
                backend_name, q, k, v, mask
            )
        if probabilities is None:
            probabilities = recover_sdpa_probabilities(
                backend_name, q, k, mask
            )
        probability_error = max_abs_error(
            probabilities, reference_probabilities
        )
        output_error = max_abs_error(output, reference_output)
        accuracy[implementation] = {
            "max_probability_abs_error": probability_error,
            "max_output_abs_error": output_error,
        }
        if probability_error > args.max_prob_error:
            raise RuntimeError(
                f"{implementation} probability error {probability_error:.3g} "
                f"exceeds {args.max_prob_error:.3g} for ctx={context_length}, "
                f"T={query_length}"
            )
        if output_error > args.max_output_error:
            raise RuntimeError(
                f"{implementation} output error {output_error:.3g} "
                f"exceeds {args.max_output_error:.3g} for ctx={context_length}, "
                f"T={query_length}"
            )

    measurements: list[dict[str, object]] = []
    implementations = ("default_sdpa", "math_sdpa", "explicit_bmm")
    for repeat in range(args.repeats):
        order = (
            implementations
            if repeat % 2 == 0
            else tuple(reversed(implementations))
        )
        for implementation in order:
            backend_name = "math" if implementation == "math_sdpa" else implementation
            for sample in timed_repeats(
                backend_name,
                q,
                k,
                v,
                mask,
                warmup=args.warmup if repeat == 0 else 0,
                iterations=args.iterations,
                repeats=1,
            ):
                measurements.append(
                    {
                        "context_length": context_length,
                        "query_length": query_length,
                        "implementation": implementation,
                        "repeat": repeat + 1,
                        "device_us_per_call": sample["device_us_per_call"],
                        "wall_us_per_call": sample["wall_us_per_call"],
                        **accuracy[implementation],
                    }
                )

    summary_rows = []
    for implementation in implementations:
        rows = [
            row
            for row in measurements
            if row["implementation"] == implementation
        ]
        summary_rows.append(
            {
                "context_length": context_length,
                "query_length": query_length,
                "implementation": implementation,
                "median_device_us_per_call": statistics.median(
                    float(row["device_us_per_call"]) for row in rows
                ),
                "median_wall_us_per_call": statistics.median(
                    float(row["wall_us_per_call"]) for row in rows
                ),
                **accuracy[implementation],
            }
        )
    return measurements, {"accuracy": accuracy, "timings": summary_rows}


def main() -> None:
    args = parse_args()
    if args.device == "cpu" or not torch.cuda.is_available():
        raise SystemExit("A CUDA device is required; this script does not run attention on CPU.")
    if args.warmup < 0 or args.iterations < 1 or args.repeats < 2:
        raise SystemExit("Use warmup>=0, iterations>=1, and repeats>=2.")
    if args.max_prob_error <= 0 or args.max_output_error <= 0:
        raise SystemExit("Accuracy limits must be positive.")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    cells: list[dict[str, object]] = []
    for context_length in CONTEXT_LENGTHS:
        for query_length in QUERY_LENGTHS:
            rows, cell = run_cell(
                context_length, query_length, generator, device, args
            )
            all_rows.extend(rows)
            cells.append(
                {
                    "context_length": context_length,
                    "query_length": query_length,
                    **cell,
                }
            )
            print(
                f"ctx={context_length:4d} T={query_length} "
                + " | ".join(
                    f"{row['implementation']}="
                    f"{row['median_device_us_per_call']:.2f}us"
                    for row in cell["timings"]
                ),
                flush=True,
            )

    csv_path = args.output_dir / "attention_candidates.csv"
    json_path = args.output_dir / "attention_candidates.json"
    fieldnames = list(all_rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    payload = {
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "dtype": "float32",
        "allow_tf32": False,
        "matmul_precision": "highest",
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "context_lengths": list(CONTEXT_LENGTHS),
        "query_lengths": list(QUERY_LENGTHS),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "max_probability_abs_error_allowed": args.max_prob_error,
        "max_output_abs_error_allowed": args.max_output_error,
        "cells": cells,
        "raw_rows": all_rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
