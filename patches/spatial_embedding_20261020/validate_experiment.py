#!/usr/bin/env python3
"""Independently summarize the fixed N2 CPU run using the standard library."""

from __future__ import annotations

import argparse
import array
import ast
import json
import math
import re
import struct
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


FEATURES = 4096
FEATURE_OFFSET = 8000
EOS_ID = 12096
SEEDS = (17, 29)
METHODS = ("S1", "S2", "S3", "S4")
SCORE_METHODS = ("S0", "SMIX_0.1", *METHODS)
SPLITS = ("fit", "tune", "hold")


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.notes: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{number}")
            rows.append(value)
    return rows


def _product(shape: tuple[int, ...]) -> int:
    value = 1
    for dim in shape:
        value *= dim
    return value


def _npy_array(stream, label: str) -> tuple[tuple[int, ...], str, array.array]:
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError(f"Invalid NPY header for {label}")
    version = stream.read(2)
    if len(version) != 2:
        raise ValueError(f"Truncated NPY version for {label}")
    if version[0] == 1:
        length_raw = stream.read(2)
        if len(length_raw) != 2:
            raise ValueError(f"Truncated NPY header length for {label}")
        header_len = struct.unpack("<H", length_raw)[0]
    elif version[0] in (2, 3):
        length_raw = stream.read(4)
        if len(length_raw) != 4:
            raise ValueError(f"Truncated NPY header length for {label}")
        header_len = struct.unpack("<I", length_raw)[0]
    else:
        raise ValueError(f"Unsupported NPY version {tuple(version)} for {label}")
    encoding = "utf-8" if version[0] == 3 else "latin1"
    header = ast.literal_eval(stream.read(header_len).decode(encoding).strip())
    if header.get("fortran_order") is not False:
        raise ValueError(f"Fortran-order NPY array is unsupported: {label}")
    shape = tuple(int(dim) for dim in header["shape"])
    descriptor = str(header["descr"])
    match = re.fullmatch(r"([<>=|])([biuf?])(\d+)", descriptor)
    if not match:
        raise ValueError(f"Unsupported NPY dtype {descriptor!r} for {label}")
    endian, kind, width_text = match.groups()
    width = int(width_text)
    code_by_dtype = {
        ("f", 8): "d", ("f", 4): "f",
        ("i", 8): "q", ("i", 4): "i", ("i", 2): "h", ("i", 1): "b",
        ("u", 8): "Q", ("u", 4): "I", ("u", 2): "H", ("u", 1): "B",
        ("b", 1): "b", ("?", 1): "B",
    }
    typecode = code_by_dtype.get((kind, width))
    if typecode is None:
        raise ValueError(f"Unsupported NPY item type {descriptor!r} for {label}")
    raw = stream.read()
    expected = _product(shape) * width
    if len(raw) != expected:
        raise ValueError(f"NPY payload size differs for {label}: {len(raw)} != {expected}")
    values = array.array(typecode)
    if values.itemsize != width:
        raise ValueError(f"Host array item size differs for {label}")
    values.frombytes(raw)
    host_little = sys.byteorder == "little"
    if width > 1 and endian in ("<", ">") and ((endian == "<") != host_little):
        values.byteswap()
    return shape, descriptor, values


def _read_npz_member(archive: zipfile.ZipFile, name: str):
    member = name if name.endswith(".npy") else name + ".npy"
    with archive.open(member) as stream:
        return _npy_array(stream, member)


def _require_shape(audit: Audit, shape: tuple[int, ...], expected: tuple[int, ...], name: str) -> None:
    audit.require(shape == expected, f"{name}: shape {shape}, expected {expected}")


def _close(a: float, b: float, *, tol: float = 2e-7) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _source_metrics(rows: list[dict]) -> dict:
    grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        overlap = float(row["_recomputed_overlap"])
        weight = float(row["p1_accept_probability"])
        grouped[str(row["source_id"])].append((overlap, overlap * weight, weight))
    by_source = {}
    for source, values in sorted(grouped.items()):
        weight_sum = math.fsum(item[2] for item in values)
        by_source[source] = {
            "row_count": len(values),
            "mean_overlap": math.fsum(item[0] for item in values) / len(values),
            "mean_p1_weight_times_overlap": math.fsum(item[1] for item in values) / len(values),
            "mean_p1_probability": weight_sum / len(values),
            "conditional_weighted_overlap": (
                math.fsum(item[1] for item in values) / weight_sum if weight_sum > 0.0 else None
            ),
        }
    metrics = {
        "row_count": len(rows),
        "source_count": len(by_source),
        "by_source": by_source,
    }
    for source_key, output_key in (
        ("mean_overlap", "source_equal_mean_overlap"),
        ("mean_p1_weight_times_overlap", "source_equal_mean_p1_weight_times_overlap"),
        ("conditional_weighted_overlap", "source_equal_conditional_weighted_overlap"),
    ):
        values = [entry[source_key] for entry in by_source.values()]
        values = [float(value) for value in values if value is not None]
        metrics[output_key] = math.fsum(values) / len(values) if values else None
    return metrics


def _split_maps(rows: list[dict], manifest: dict) -> tuple[dict, dict]:
    source_ids = manifest.get("source_ids")
    splits = manifest.get("splits")
    if manifest.get("schema_version") != 1 or not isinstance(source_ids, list):
        raise ValueError("Unsupported source manifest")
    if source_ids != sorted(set(source_ids)) or len(source_ids) != 10:
        raise ValueError("Source manifest must list ten unique IDs in sorted order")
    if manifest.get("seed") != 32003 or manifest.get("seed_used_for_split") is not False:
        raise ValueError("Source manifest must retain split note seed 32003 without using it to shuffle")
    expected = {"fit": 6, "tune": 2, "hold": 2}
    if not isinstance(splits, dict) or set(splits) != set(expected):
        raise ValueError("Source manifest must contain fit, tune, and hold")
    source_to_split = {}
    for split, count in expected.items():
        values = splits[split]
        if values != source_ids[sum(expected[s] for s in SPLITS[:SPLITS.index(split)]):
                          sum(expected[s] for s in SPLITS[:SPLITS.index(split) + 1])]:
            raise ValueError(f"{split} IDs do not follow sorted source order")
        if len(values) != count:
            raise ValueError(f"{split} must contain {count} source IDs")
        for source in values:
            if source in source_to_split:
                raise ValueError(f"Source appears in multiple splits: {source}")
            source_to_split[str(source)] = split
    expected_counts = {source: 48 for source in source_ids}
    actual_counts = Counter(str(row.get("source_id")) for row in rows)
    if dict(actual_counts) != expected_counts:
        raise ValueError("Prepared rows do not contain exactly 48 entries per source")
    for row in rows:
        row["_split"] = source_to_split[str(row["source_id"])]
    split_indices = {
        split: [int(row["vector_index"]) for row in rows if row["_split"] == split]
        for split in SPLITS
    }
    if {key: len(value) for key, value in split_indices.items()} != {
        "fit": 288, "tune": 96, "hold": 96,
    }:
        raise ValueError("Prepared rows do not split into 288/96/96")
    return source_to_split, split_indices


def _check_s4(audit: Audit, data_dir: Path, run_dir: Path, rows: list[dict]) -> dict:
    with zipfile.ZipFile(data_dir / "prepared_data.npz") as data_npz:
        needed = (
            "history_pair_offsets", "history_position_ids_flat", "history_feature_global_ids_flat",
            "neighbor_position_ids", "neighbor_mask", "p1_id",
        )
        arrays = {name: _read_npz_member(data_npz, name) for name in needed}
    offsets = arrays["history_pair_offsets"][2]
    positions = arrays["history_position_ids_flat"][2]
    features = arrays["history_feature_global_ids_flat"][2]
    neighbor_positions = arrays["neighbor_position_ids"][2]
    neighbor_mask = arrays["neighbor_mask"][2]
    candidate_positions = arrays["p1_id"][2]
    audit.require(arrays["history_pair_offsets"][0] == (481,), "S4: history offsets must have 481 entries")
    audit.require(arrays["neighbor_position_ids"][0] == (480, 6), "S4: six neighbor positions per row are required")
    audit.require(arrays["neighbor_mask"][0] == (480, 6), "S4: six neighbor mask entries per row are required")
    assignments = read_jsonl(run_dir / "s4_assignments.jsonl")
    audit.require(len(assignments) == 480, f"S4: expected 480 assignments, got {len(assignments)}")
    row_by_index = {int(row["vector_index"]): row for row in rows}
    valid_total = 0
    for index, record in enumerate(assignments):
        if index >= len(rows):
            break
        source = row_by_index[index]
        audit.require(record.get("vector_index") == index, f"S4: assignment order differs at row {index}")
        audit.require(record.get("sample_id") == source["sample_id"], f"S4: sample ID differs at row {index}")
        audit.require(record.get("source_id") == source["source_id"], f"S4: source ID differs at row {index}")
        fields = ("position_ids", "feature_global_ids", "mask", "age_pf_pairs")
        for field in fields:
            value = record.get(field)
            audit.require(isinstance(value, list) and len(value) == 6,
                          f"S4: {field} must contain six entries at row {index}")
        if not all(isinstance(record.get(field), list) and len(record[field]) == 6 for field in fields):
            continue
        expected_mask = [bool(neighbor_mask[index * 6 + slot]) for slot in range(6)]
        actual_mask = [bool(value) for value in record["mask"]]
        audit.require(actual_mask == expected_mask, f"S4: mask differs from S3 at row {index}")
        start, end = int(offsets[index]), int(offsets[index + 1])
        history = {
            int(positions[pos]): (int(features[pos]), end - start - 1 - (pos - start))
            for pos in range(start, end)
        }
        adjacent = {
            int(neighbor_positions[index * 6 + slot])
            for slot in range(6) if expected_mask[slot]
        }
        valid_positions = []
        for slot, is_valid in enumerate(actual_mask):
            pos = int(record["position_ids"][slot])
            feature = int(record["feature_global_ids"][slot])
            age = int(record["age_pf_pairs"][slot])
            if not is_valid:
                audit.require((pos, feature, age) == (-1, -1, -1),
                              f"S4: masked slot must use sentinel values at row {index}, slot {slot}")
                continue
            valid_total += 1
            valid_positions.append(pos)
            audit.require(pos in history, f"S4: position is absent from committed history at row {index}")
            if pos in history:
                expected_feature, expected_age = history[pos]
                audit.require(feature == expected_feature,
                              f"S4: Feature ID differs from committed history at row {index}")
                audit.require(age == expected_age,
                              f"S4: age differs from committed history at row {index}")
            audit.require(pos not in adjacent, f"S4: selected position is a six-face neighbor at row {index}")
            audit.require(pos != int(candidate_positions[index]),
                          f"S4: selected position is the candidate P at row {index}")
            audit.require(FEATURE_OFFSET <= feature < EOS_ID,
                          f"S4: Feature ID is outside the Feature vocabulary at row {index}")
        audit.require(len(valid_positions) == len(set(valid_positions)),
                      f"S4: repeated historical position at row {index}")
        audit.require(len(valid_positions) == sum(expected_mask),
                      f"S4: valid count differs from S3 at row {index}")
    return {"assignments": len(assignments), "valid_slots": valid_total,
            "mask_and_count_match": not any("S4: mask differs" in item or
                                              "S4: valid count differs" in item for item in audit.errors)}


def _main(args) -> int:
    audit = Audit()
    data_dir = args.data.expanduser().resolve()
    run_dir = args.run.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    metadata = read_json(data_dir / "metadata.json")
    summary = read_json(run_dir / "summary.json")
    manifest = read_json(manifest_path)
    rows = read_jsonl(data_dir / "rows.jsonl")
    if len(rows) != 480:
        audit.errors.append(f"Prepared rows must be 480, got {len(rows)}")
    audit.require(metadata.get("target_probabilities_used_for_features") is False,
                  "Prepared context feature derivation reports target probabilities as input")
    audit.require(metadata.get("history_source") ==
                  "context_prefix only; includes the latest complete PF pair even when its F had not yet entered Target KV",
                  "Prepared history source is not the saved context_prefix")
    audit.require(metadata.get("policies") == {"temperature": 0.9, "top_p": 0.9},
                  "Prepared Target policy differs from T=.9, top-p=.9")
    for index, row in enumerate(rows):
        audit.require(int(row.get("vector_index", -1)) == index,
                      f"Prepared rows lost original order at row {index}")
        audit.require(row.get("p1_accepted_actual") is True,
                      f"Prepared row {index} is not an actual accepted-P1 context")
        audit.require(row.get("source_id") == row.get("scene"),
                      f"Prepared source ID differs from scene at row {index}")
    source_to_split, expected_indices = _split_maps(rows, manifest)

    training = summary.get("training", {})
    model_meta = summary.get("model", {})
    selection = summary.get("selection", {})
    pca = summary.get("pca", {})
    audit.require(summary.get("status") == "complete_cpu_offline_analysis", "Run summary is incomplete")
    audit.require(summary.get("device") == "cpu" and summary.get("gpu_used") is False,
                  "Run summary does not confirm CPU-only execution")
    audit.require(summary.get("target_forward_used") is False,
                  "Run summary reports a Target forward")
    counts = summary.get("counts", {})
    for name, expected in (("rows", 480), ("sources", 10), ("fit_rows", 288),
                           ("tune_rows", 96), ("hold_rows", 96), ("model_count", 8),
                           ("completed_updates", 1200), ("updates_per_model", 150)):
        audit.require(counts.get(name) == expected, f"Run summary count {name} differs from {expected}")
    audit.require(training.get("updates_per_model") == 150, "Each trained model must have 150 updates")
    audit.require(training.get("total_updates") == 1200, "Expected 1200 total model updates")
    audit.require(training.get("seeds") == [17, 29], "Training seeds must be 17 and 29")
    audit.require(training.get("batch_size") == 32, "Training batch size differs from 32")
    audit.require(training.get("evaluation_policy") == {"temperature": 0.9, "top_p": 0.9},
                  "Evaluation policy differs from T=.9, top-p=.9")
    audit.require(summary.get("policies", {}).get("temperature") == 0.9 and
                  summary.get("policies", {}).get("top_p") == 0.9,
                  "Saved evaluation policy differs from T=.9, top-p=.9")
    audit.require(model_meta.get("all_methods_equal_parameter_count") is True,
                  "S1-S4 parameter counts are not reported equal")
    audit.require(pca.get("input_is_only_feature_embedding") is True,
                  "PCA input is not restricted to the fixed Feature embedding")
    audit.require(pca.get("fit_input") == "feature_embedding only" and
                  pca.get("uses_target_labels") is False,
                  "PCA must use only the fixed Feature embedding, without Target labels")
    policy_rebuild = summary.get("probability_integrity", {}).get(
        "raw_logits_policy_reconstruction_max_abs_difference_vs_saved_q1", math.inf
    )
    audit.require(float(policy_rebuild) <= 5e-6,
                  "Saved q1 differs from temperature/top-p reconstruction of the original logits")
    audit.require(selection.get("hold_access_after_tune_selection") is True,
                  "Summary does not show Tune selection before Hold scoring")
    audit.require(selection.get("unselected_seed_hold_metrics") == "not computed",
                  "Unselected seeds must not have Hold scores")

    with zipfile.ZipFile(data_dir / "prepared_data.npz") as data_npz:
        prepared_pf = _read_npz_member(data_npz, "p_f")
        prepared_q1 = _read_npz_member(data_npz, "q1")
    _require_shape(audit, prepared_pf[0], (480, FEATURES), "prepared p_f")
    _require_shape(audit, prepared_q1[0], (480, FEATURES), "prepared q1")
    weights = [float(row["p1_accept_probability"]) for row in rows]
    audit.require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in weights),
                  "P1 probabilities must be finite values in [0,1]")

    with zipfile.ZipFile(run_dir / "q_outputs.npz") as q_archive:
        archive_keys = {name[:-4] for name in q_archive.namelist() if name.endswith(".npy")}
        score_rows = read_jsonl(run_dir / "scores.jsonl")
        split_data = {}
        for split in SPLITS:
            indices_shape, _indices_dtype, indices_data = _read_npz_member(q_archive, f"indices_{split}")
            _require_shape(audit, indices_shape, (len(expected_indices[split]),), f"indices_{split}")
            audit.require(list(indices_data) == expected_indices[split],
                          f"q output vector order differs for {split}")
            p_shape, _p_dtype, p_data = _read_npz_member(q_archive, f"p_f_{split}")
            expected_shape = (len(expected_indices[split]), FEATURES)
            _require_shape(audit, p_shape, expected_shape, f"p_f_{split}")
            for local, global_index in enumerate(expected_indices[split]):
                offset = local * FEATURES
                prepared_offset = global_index * FEATURES
                for class_index in range(FEATURES):
                    audit.require(_close(float(p_data[offset + class_index]),
                                         float(prepared_pf[2][prepared_offset + class_index]), tol=2e-6),
                                         f"Saved p_f differs from prepared source at {split} row {global_index}")
            split_data[split] = {
                "indices": indices_data, "local_index": {
                    int(value): position for position, value in enumerate(indices_data)
                },
                "p": p_data,
            }
        groups: dict[tuple[str, str, int | None], list[dict]] = defaultdict(list)
        seen = set()
        for record in score_rows:
            split = str(record.get("split"))
            method = str(record.get("method"))
            if split not in SPLITS or method not in SCORE_METHODS:
                audit.errors.append(f"Unknown score split/method: {split}/{method}")
                continue
            seed_value = record.get("seed")
            seed = None if seed_value is None else int(seed_value)
            key = (method, split, seed)
            vector_index = int(record.get("vector_index", -1))
            if not 0 <= vector_index < len(rows):
                audit.errors.append(f"Score vector_index outside prepared rows: {vector_index}")
                continue
            source = rows[vector_index]
            audit.require(record.get("sample_id") == source.get("sample_id"),
                          f"Score sample ID differs at vector_index {vector_index}")
            audit.require(record.get("source_id") == source.get("source_id"),
                          f"Score source ID differs at vector_index {vector_index}")
            audit.require(split == source["_split"],
                          f"Score split differs from manifest at vector_index {vector_index}")
            row_key = (method, split, seed, vector_index)
            if row_key in seen:
                audit.errors.append(f"Duplicate score row: {row_key}")
            seen.add(row_key)
            groups[key].append(record)

        selected = selection.get("selected_seed_by_method", {})
        audit.require(set(selected) == set(METHODS), "Tune selection must choose one seed for S1-S4")
        expected_groups: dict[tuple[str, str, int | None], int] = {}
        for method in ("S0", "SMIX_0.1"):
            for split in SPLITS:
                expected_groups[(method, split, None)] = len(expected_indices[split])
        for method in METHODS:
            for seed in SEEDS:
                expected_groups[(method, "fit", seed)] = 288
                expected_groups[(method, "tune", seed)] = 96
                expected_groups[(method, "hold", seed)] = (
                    96 if selected.get(method) == seed else 0
                )
        for key, expected_count in expected_groups.items():
            actual_count = len(groups.get(key, []))
            audit.require(actual_count == expected_count,
                          f"Score row count for {key} is {actual_count}, expected {expected_count}")

        score_metrics: dict[tuple[str, str, int | None], dict] = {}
        for key, records in groups.items():
            method, split, seed = key
            if not records:
                continue
            split_values = split_data[split]
            p_data = split_values["p"]
            local_index = split_values["local_index"]
            expected_split_indices = expected_indices[split]
            if method == "S0":
                q_key = f"q_S0_seed_none_{split}"
                budget = 0
            elif method == "SMIX_0.1":
                q_key = f"q_SMIX_0p1_seed_none_{split}"
                budget = 0
            else:
                q_key = f"q_{method}_seed_{seed}_{split}"
                budget = 150
            audit.require(q_key in archive_keys, f"Missing saved q array {q_key}")
            if q_key not in archive_keys:
                continue
            q_shape, q_dtype, q_data = _read_npz_member(q_archive, q_key)
            _require_shape(audit, q_shape, (len(expected_split_indices), FEATURES), q_key)
            audit.require(q_dtype == "<f8" or q_dtype == "=f8", f"{q_key} must be saved as float64")
            if method == "S0":
                same_as_old_q = True
                for local, global_index in enumerate(expected_split_indices):
                    offset = local * FEATURES
                    prepared_offset = global_index * FEATURES
                    for class_index in range(FEATURES):
                        if not _close(float(q_data[offset + class_index]),
                                      float(prepared_q1[2][prepared_offset + class_index]), tol=2e-6):
                            same_as_old_q = False
                            break
                    if not same_as_old_q:
                        break
                audit.require(same_as_old_q, f"{q_key} differs from the prepared old q for {split}")
            metric_rows = []
            group_records = sorted(records, key=lambda row: int(row["vector_index"]))
            audit.require(len({int(row["vector_index"]) for row in group_records}) == len(group_records),
                          f"Repeated vector index within {key}")
            for record in group_records:
                index = int(record["vector_index"])
                local = local_index.get(index)
                if local is None:
                    audit.errors.append(f"Score row {index} is absent from {split} q indices")
                    continue
                begin = local * FEATURES
                p_sum = math.fsum(float(p_data[begin + j]) for j in range(FEATURES))
                q_sum = math.fsum(float(q_data[begin + j]) for j in range(FEATURES))
                audit.require(p_sum > 0.0 and q_sum > 0.0,
                              f"Nonpositive probability mass at {key}, row {index}")
                if p_sum <= 0.0 or q_sum <= 0.0:
                    continue
                overlap = math.fsum(
                    min(float(p_data[begin + j]) / p_sum, float(q_data[begin + j]) / q_sum)
                    for j in range(FEATURES)
                )
                weight = weights[index]
                audit.require(record.get("p1_accepted_actual") is True,
                              f"Score row {index} is outside the accepted-P1 reservoir")
                audit.require(_close(float(record.get("p1_accept_probability", -1.0)), weight),
                              f"Score P1 weight differs from prepared row {index}")
                audit.require(_close(float(record.get("feature_overlap", -1.0)), overlap),
                              f"Saved overlap differs from independent q calculation at row {index}")
                audit.require(_close(float(record.get("weighted_overlap", -1.0)), weight * overlap),
                              f"Saved P1-weighted overlap differs at row {index}")
                audit.require(int(record.get("step_budget", -1)) == budget,
                              f"Step budget differs for {key} at row {index}")
                support = 0
                positive_values = []
                for j in range(FEATURES):
                    value = float(q_data[begin + j])
                    if value > 0.0:
                        support += 1
                        positive_values.append(value)
                audit.require(int(record.get("q_support", -1)) == support,
                              f"q support differs at {key}, row {index}")
                audit.require(_close(float(record.get("q_mass", -1.0)), q_sum),
                              f"q mass differs at {key}, row {index}")
                recorded_min = record.get("q_min_positive")
                actual_min = min(positive_values) if positive_values else None
                audit.require((recorded_min is None and actual_min is None) or
                              (recorded_min is not None and actual_min is not None and
                               _close(float(recorded_min), actual_min, tol=2e-7)),
                              f"Minimum positive q differs at {key}, row {index}")
                metric_row = dict(record)
                metric_row["_recomputed_overlap"] = overlap
                metric_rows.append(metric_row)
            score_metrics[key] = _source_metrics(metric_rows)

    expected_score_count = 4416
    audit.require(len(score_rows) == expected_score_count,
                  f"Expected {expected_score_count} score rows after unique grouping, got {len(score_rows)}")
    for method in METHODS:
        method_reports = model_meta.get("parameter_count_by_method", {})
        audit.require(method in method_reports, f"Missing parameter count for {method}")
    model_parameter_counts = {}
    for method in METHODS:
        for seed in SEEDS:
            model_path = run_dir / f"model_{method}_seed_{seed}.npz"
            audit.require(model_path.is_file(), f"Missing trained model weights: {model_path.name}")
            if not model_path.is_file():
                continue
            count = 0
            with zipfile.ZipFile(model_path) as model_npz:
                for member in model_npz.namelist():
                    if member.endswith(".npy"):
                        shape, _dtype, _values = _read_npz_member(model_npz, member)
                        count += _product(shape)
            model_parameter_counts[f"{method}_seed_{seed}"] = count
            audit.require(count == model_meta.get("parameter_count_by_method", {}).get(method),
                          f"Saved parameter count differs from summary for {method}, seed {seed}")
    audit.require(len(set(model_parameter_counts.values())) == 1,
                  "Saved S1-S4 model parameter counts differ")

    history = read_jsonl(run_dir / "history.jsonl")
    audit.require(len(history) == 1200, f"Expected 1200 optimizer records, got {len(history)}")
    history_groups = defaultdict(list)
    for record in history:
        key = (str(record.get("method")), int(record.get("seed", -1)))
        history_groups[key].append(record)
        audit.require(int(record.get("step_budget", -1)) == 150,
                      f"Training record has an unexpected fixed budget: {key}")
        audit.require(int(record.get("batch_size", -1)) == 32,
                      f"Training record batch size differs from 32: {key}")
        for field in ("weighted_kl", "residual_l2_mean_square", "loss", "elapsed_seconds"):
            audit.require(math.isfinite(float(record.get(field, math.nan))),
                          f"Nonfinite training field {field} in {key}")
    for method in METHODS:
        for seed in SEEDS:
            updates = sorted(int(row.get("step", -1)) for row in history_groups[(method, seed)])
            audit.require(updates == list(range(1, 151)),
                          f"Training history for {method}/{seed} must contain steps 1..150")

    tune_scores = {}
    for method in METHODS:
        for seed in SEEDS:
            metric = score_metrics.get((method, "tune", seed))
            tune_scores[(method, seed)] = (
                float(metric["source_equal_mean_p1_weight_times_overlap"])
                if metric and metric.get("source_equal_mean_p1_weight_times_overlap") is not None
                else math.nan
            )
    selected_recomputed = {
        method: (17 if tune_scores[(method, 17)] >= tune_scores[(method, 29)] else 29)
        for method in METHODS
    }
    for method in METHODS:
        audit.require(int(selected.get(method, -1)) == selected_recomputed[method],
                      f"Selected seed for {method} differs from independent Tune selection")
        hold_report = selection.get("hold_method_seed", {}).get(method, {})
        audit.require(hold_report.get("seed") == selected_recomputed[method] and
                      hold_report.get("score_passes") == 1,
                      f"Hold record for {method} does not use only its Tune-selected seed")
        for seed in SEEDS:
            key = f"{method}_seed_{seed}"
            saved = selection.get("tune_metrics_by_method_seed", {}).get(key, {})
            computed = score_metrics.get((method, "tune", seed), {})
            for metric_name in ("source_equal_mean_overlap",
                                "source_equal_mean_p1_weight_times_overlap",
                                "source_equal_conditional_weighted_overlap"):
                expected_value = computed.get(metric_name)
                saved_value = saved.get(metric_name)
                audit.require(expected_value is not None and saved_value is not None and
                              _close(float(saved_value), float(expected_value)),
                              f"Tune summary metric {metric_name} differs for {key}")

    s0_tune = score_metrics.get(("S0", "tune", None), {})
    chosen_scores = {method: tune_scores[(method, selected_recomputed[method])] for method in METHODS}
    selected_control = max(("S1", "S2", "S4"), key=lambda method: chosen_scores[method])
    s3_score = chosen_scores["S3"]
    s0_score = float(s0_tune.get("source_equal_mean_p1_weight_times_overlap", math.nan))
    control_score = chosen_scores[selected_control]
    tune_per_source = {}
    source_ids = manifest["splits"]["tune"]
    for source in source_ids:
        s3_value = score_metrics[("S3", "tune", selected_recomputed["S3"])]["by_source"][source]["mean_p1_weight_times_overlap"]
        s0_value = s0_tune["by_source"][source]["mean_p1_weight_times_overlap"]
        control_value = score_metrics[(selected_control, "tune", selected_recomputed[selected_control])]["by_source"][source]["mean_p1_weight_times_overlap"]
        tune_per_source[source] = {
            "S3_minus_S0": s3_value - s0_value,
            "S3_minus_best_control": s3_value - control_value,
            "improves_vs_S0": bool(s3_value > s0_value),
        }
    passes_screen = bool(
        s3_score - s0_score >= 0.03 and s3_score - control_score >= 0.015 and
        all(row["improves_vs_S0"] for row in tune_per_source.values())
    )
    tune_screen = {
        "S0_source_equal_weighted_overlap": s0_score,
        "S3_source_equal_weighted_overlap": s3_score,
        "best_control": selected_control,
        "best_control_seed": selected_recomputed[selected_control],
        "best_control_source_equal_weighted_overlap": control_score,
        "gain_vs_S0": s3_score - s0_score,
        "gain_vs_best_control": s3_score - control_score,
        "per_source": tune_per_source,
        "passes_fixed_thresholds": passes_screen,
    }
    saved_screen = selection.get("tune_screen", {})
    audit.require(saved_screen.get("best_control_method") == selected_control,
                  "Summary Tune control differs from independently selected S1/S2/S4 control")
    audit.require(_close(float(saved_screen.get("gain_vs_S0", math.nan)), s3_score - s0_score),
                  "Summary S3 gain over S0 differs from recomputed Tune score")
    audit.require(_close(float(saved_screen.get("gain_vs_best_S1_S2_S4", math.nan)),
                         s3_score - control_score),
                  "Summary S3 gain over the strongest control differs from recomputed Tune score")
    audit.require(saved_screen.get("both_tune_sources_improve_vs_S0") is
                  all(row["improves_vs_S0"] for row in tune_per_source.values()),
                  "Summary Tune source-wise improvement differs from recomputation")
    audit.require(saved_screen.get("passes_tune_screen") is passes_screen,
                  "Summary Tune decision differs from recomputed thresholds")
    for method, result_key in (("S0", "S0_tune"), ("SMIX_0.1", "simple_mixture")):
        saved = selection.get(result_key, {})
        if method == "SMIX_0.1":
            saved = saved.get("tune", {})
        computed = score_metrics.get((method, "tune", None), {})
        for metric_name in ("source_equal_mean_overlap",
                            "source_equal_mean_p1_weight_times_overlap",
                            "source_equal_conditional_weighted_overlap"):
            audit.require(computed.get(metric_name) is not None and saved.get(metric_name) is not None and
                          _close(float(saved[metric_name]), float(computed[metric_name])),
                          f"Summary Tune metric {metric_name} differs for {method}")
    for method in METHODS:
        selected_seed = selected_recomputed[method]
        saved = selection.get("hold_metrics_selected_seeds", {}).get(method, {})
        computed = score_metrics.get((method, "hold", selected_seed), {})
        for metric_name in ("source_equal_mean_overlap",
                            "source_equal_mean_p1_weight_times_overlap",
                            "source_equal_conditional_weighted_overlap"):
            audit.require(computed.get(metric_name) is not None and saved.get(metric_name) is not None and
                          _close(float(saved[metric_name]), float(computed[metric_name])),
                          f"Summary Hold metric {metric_name} differs for {method}")
    s4_summary = _check_s4(audit, data_dir, run_dir, rows)
    report_metrics = {
        "/".join((method, split, "none" if seed is None else str(seed))): metric
        for (method, split, seed), metric in sorted(score_metrics.items(), key=lambda item: str(item[0]))
    }
    result = {
        "status": "passed" if not audit.errors else "failed",
        "errors": audit.errors,
        "counts": {"prepared_rows": len(rows), "score_rows": len(score_rows),
                   "source_count": len(source_to_split), "model_updates": len(history)},
        "selected_seeds_from_tune": selected_recomputed,
        "tune_screen_recomputed": tune_screen,
        "source_metrics_recomputed": report_metrics,
        "saved_model_parameter_counts": model_parameter_counts,
        "s4_context_summary": s4_summary,
        "scope": "480 actual P1-accepted rows from previously explored sources; no all-round rate or online speed inference",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if not audit.errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Prepared data directory from prepare_data.py")
    parser.add_argument("--run", type=Path, required=True,
                        help="Completed output directory from run_experiment.py")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Saved 6/2/2 source split JSON")
    args = parser.parse_args()
    try:
        return _main(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "errors": [f"{type(exc).__name__}: {exc}"]},
                         ensure_ascii=False, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
