#!/usr/bin/env python3
"""CUDA Graph replay comparison for the isolated SDPA candidate benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import bench_sdpa_candidates as bench


IMPLEMENTATIONS = ("default_sdpa", "math_sdpa", "explicit_bmm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--graph-iterations", type=int, default=16)
    parser.add_argument("--graph-replays", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-prob-error", type=float, default=1.0e-4)
    parser.add_argument("--max-output-error", type=float, default=1.0e-4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profiling/pf2x_breakthrough_20261021/attention_graph"),
    )
    return parser.parse_args()


def backend_name(implementation: str) -> str:
    return "math" if implementation == "math_sdpa" else implementation


def make_inputs(context_length: int, query_length: int, generator, device):
    key_length = context_length + query_length
    q_bthd = torch.randn(
        (1, query_length, bench.HEADS, bench.HEAD_DIM),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    k_bthd = torch.randn(
        (1, key_length, bench.HEADS, bench.HEAD_DIM),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    v_bthd = torch.randn(
        (1, key_length, bench.HEADS, bench.HEAD_DIM),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    q = F.rms_norm(q_bthd, (bench.HEAD_DIM,)).transpose(1, 2)
    k = F.rms_norm(k_bthd, (bench.HEAD_DIM,)).transpose(1, 2)
    v = v_bthd.transpose(1, 2)
    mask = bench.make_allowed_mask(context_length, query_length, device)
    return q, k, v, mask


def candidate(implementation: str, q, k, v, mask):
    return bench.run_candidate(backend_name(implementation), q, k, v, mask)


def graph_measure(
    implementation: str,
    q,
    k,
    v,
    mask,
    warmup: int,
    graph_iterations: int,
    graph_replays: int,
    repeats: int,
):
    backend = backend_name(implementation)
    warm_stream = torch.cuda.Stream(device=q.device)
    warm_stream.wait_stream(torch.cuda.current_stream(q.device))
    with bench.backend_context(backend), torch.cuda.stream(warm_stream):
        for _ in range(warmup):
            candidate(implementation, q, k, v, mask)
    torch.cuda.current_stream(q.device).wait_stream(warm_stream)
    torch.cuda.synchronize(q.device)

    graph = torch.cuda.CUDAGraph()
    static_output = None
    with bench.backend_context(backend):
        with torch.cuda.graph(graph):
            for _ in range(graph_iterations):
                static_output, _ = candidate(implementation, q, k, v, mask)
    graph.replay()
    torch.cuda.synchronize(q.device)

    samples = []
    for repeat in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(q.device)
        wall_start = time.perf_counter()
        start_event.record()
        for _ in range(graph_replays):
            graph.replay()
        end_event.record()
        torch.cuda.synchronize(q.device)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        device_ms = float(start_event.elapsed_time(end_event))
        calls = graph_replays * graph_iterations
        samples.append(
            {
                "implementation": implementation,
                "repeat": repeat + 1,
                "graph_calls": calls,
                "device_us_per_call": device_ms * 1000.0 / calls,
                "wall_us_per_call": wall_ms * 1000.0 / calls,
            }
        )
    return samples, static_output


def main() -> None:
    args = parse_args()
    if args.device == "cpu" or not torch.cuda.is_available():
        raise SystemExit("A CUDA device is required; this script does not run attention on CPU.")
    if (
        args.warmup < 0
        or args.graph_iterations < 1
        or args.graph_replays < 1
        or args.repeats < 2
    ):
        raise SystemExit("Use warmup>=0, graph iterations/replays>=1, and repeats>=2.")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    cells = []
    for context_length in bench.CONTEXT_LENGTHS:
        for query_length in bench.QUERY_LENGTHS:
            q, k, v, mask = make_inputs(
                context_length, query_length, generator, device
            )
            reference_probabilities, reference_output = bench.fp64_reference(
                q, k, v, mask
            )
            cell_rows = []
            for repeat in range(args.repeats):
                order = (
                    IMPLEMENTATIONS
                    if repeat % 2 == 0
                    else tuple(reversed(IMPLEMENTATIONS))
                )
                for implementation in order:
                    samples, graph_output = graph_measure(
                        implementation,
                        q,
                        k,
                        v,
                        mask,
                        warmup=args.warmup if repeat == 0 else 0,
                        graph_iterations=args.graph_iterations,
                        graph_replays=args.graph_replays,
                        repeats=1,
                    )
                    sample = samples[0]
                    with bench.backend_context(backend_name(implementation)):
                        _, explicit_probabilities = candidate(
                            implementation, q, k, v, mask
                        )
                    if explicit_probabilities is None:
                        probabilities = bench.recover_sdpa_probabilities(
                            backend_name(implementation), q, k, mask
                        )
                    else:
                        probabilities = explicit_probabilities
                    probability_error = bench.max_abs_error(
                        probabilities, reference_probabilities
                    )
                    output_error = bench.max_abs_error(
                        graph_output, reference_output
                    )
                    if probability_error > args.max_prob_error:
                        raise RuntimeError(
                            f"{implementation} probability error "
                            f"{probability_error:.3g} exceeds "
                            f"{args.max_prob_error:.3g} at ctx={context_length}, "
                            f"T={query_length}"
                        )
                    if output_error > args.max_output_error:
                        raise RuntimeError(
                            f"{implementation} output error {output_error:.3g} "
                            f"exceeds {args.max_output_error:.3g} at "
                            f"ctx={context_length}, T={query_length}"
                        )
                    row = {
                        "context_length": context_length,
                        "query_length": query_length,
                        **sample,
                        "max_probability_abs_error": probability_error,
                        "max_output_abs_error": output_error,
                    }
                    raw_rows.append(row)
                    cell_rows.append(row)

            timings = []
            for implementation in IMPLEMENTATIONS:
                rows = [
                    row
                    for row in cell_rows
                    if row["implementation"] == implementation
                ]
                timings.append(
                    {
                        "implementation": implementation,
                        "median_device_us_per_call": statistics.median(
                            float(row["device_us_per_call"]) for row in rows
                        ),
                        "median_wall_us_per_call": statistics.median(
                            float(row["wall_us_per_call"]) for row in rows
                        ),
                        "max_probability_abs_error": max(
                            float(row["max_probability_abs_error"])
                            for row in rows
                        ),
                        "max_output_abs_error": max(
                            float(row["max_output_abs_error"]) for row in rows
                        ),
                    }
                )
            cells.append(
                {
                    "context_length": context_length,
                    "query_length": query_length,
                    "timings": timings,
                }
            )
            print(
                f"ctx={context_length:4d} T={query_length} "
                + " | ".join(
                    f"{row['implementation']}="
                    f"{row['median_device_us_per_call']:.2f}us"
                    for row in timings
                ),
                flush=True,
            )

    csv_path = args.output_dir / "attention_candidates_graph.csv"
    json_path = args.output_dir / "attention_candidates_graph.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    payload = {
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "dtype": "float32",
        "allow_tf32": False,
        "matmul_precision": "highest",
        "heads": bench.HEADS,
        "head_dim": bench.HEAD_DIM,
        "context_lengths": list(bench.CONTEXT_LENGTHS),
        "query_lengths": list(bench.QUERY_LENGTHS),
        "seed": args.seed,
        "warmup": args.warmup,
        "graph_iterations": args.graph_iterations,
        "graph_replays": args.graph_replays,
        "repeats": args.repeats,
        "max_probability_abs_error_allowed": args.max_prob_error,
        "max_output_abs_error_allowed": args.max_output_error,
        "cells": cells,
        "raw_rows": raw_rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
