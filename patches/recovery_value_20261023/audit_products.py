#!/usr/bin/env python3
"""Read saved E2E products and compare each method with its W1 request."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable


EXPECTED_FIELDS = {
    "means": 3,
    "scales": 3,
    "opacities": 1,
    "quats": 4,
    "colors": 3,
}
EOS_ID = 12096


def _jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return rows, [f"missing file: {path.name}"]
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row is not an object")
            rows.append(value)
        except Exception as error:
            errors.append(f"{path.name}:{line_number}: {type(error).__name__}: {error}")
    return rows, errors


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _token_payload(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing tokens.pt"
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("tokens.pt must contain a mapping")
        required = {"tokens", "suffix", "has_eos", "prefix_tokens"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"tokens.pt is missing keys: {missing}")
        normalized = {
            "tokens": [int(value) for value in _plain(payload["tokens"])],
            "suffix": [int(value) for value in _plain(payload["suffix"])],
            "has_eos": bool(payload["has_eos"]),
            "prefix_tokens": int(_plain(payload["prefix_tokens"])),
        }
        return normalized, None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def _record_eos(output: list[int]) -> bool:
    return bool(output and output[-1] == EOS_ID)


def _token_checks(token_data: dict[str, Any], row: dict[str, Any], prompt: list[int] | None) -> dict[str, Any]:
    output = [int(token) for token in row.get("output_tokens", [])]
    checks: dict[str, Any] = {
        "record_suffix_matches": token_data["suffix"] == output,
        "record_eos_matches": token_data["has_eos"] == bool(row.get("eos")),
        "eos_matches_suffix": token_data["has_eos"] == _record_eos(token_data["suffix"]),
        "prefix_matches_record": token_data["prefix_tokens"] == int(row.get("prefix", -1)),
        "prompt_body_matches": None,
    }
    if prompt is not None:
        body_suffix = token_data["suffix"][:-1] if token_data["has_eos"] else token_data["suffix"]
        checks["prompt_body_matches"] = token_data["tokens"] == prompt[1:] + body_suffix
    return checks


def _planned_keys(run_root: Path, plan: dict[str, Any], errors: list[str]) -> tuple[set[tuple[str, str]], dict[str, list[int]]]:
    methods = plan.get("methods")
    if not isinstance(methods, list) or not methods:
        errors.append("plan.json has no nonempty methods list")
        methods = []
    prompts_path = run_root / "prompts.json"
    pair_prompts: dict[str, list[int]] = {}
    if not prompts_path.is_file():
        errors.append("missing prompts.json")
        return set(), pair_prompts
    try:
        prompt_data = json.loads(prompts_path.read_text())
        prompts = prompt_data["prompts"]
        if not prompts:
            errors.append("prompts.json contains no requests")
        for prompt in prompts:
            pair_id = str(prompt["pair_id"])
            if pair_id in pair_prompts:
                errors.append(f"duplicate pair_id in prompts.json: {pair_id}")
            pair_prompts[pair_id] = [int(token) for token in prompt["prompt"]]
    except Exception as error:
        errors.append(f"invalid prompts.json: {type(error).__name__}: {error}")
        return set(), pair_prompts
    return {(pair_id, str(method)) for pair_id in pair_prompts for method in methods}, pair_prompts


def _load_npz(path: Path, expected_count: int | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    report: dict[str, Any] = {"path": str(path), "readable": False, "finite": False}
    if not path.is_file():
        report["error"] = "missing gaussians.npz"
        return None, report
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            names = sorted(archive.files)
            report["fields"] = names
            arrays = {name: archive[name] for name in names}
        missing = sorted(set(EXPECTED_FIELDS) - set(arrays))
        extra = sorted(set(arrays) - set(EXPECTED_FIELDS))
        report["missing_fields"] = missing
        report["extra_fields"] = extra
        if missing or extra:
            report["error"] = "Gaussian field set differs from the saved E2E format"
            return None, report
        count = int(arrays["means"].shape[0]) if arrays["means"].ndim else -1
        report["gaussian_count"] = count
        report["shapes"] = {name: list(value.shape) for name, value in arrays.items()}
        report["finite"] = all(bool(np.isfinite(value).all()) for value in arrays.values())
        shapes_ok = count >= 0 and all(
            arrays[name].shape == ((count,) if width == 1 else (count, width))
            for name, width in EXPECTED_FIELDS.items()
        )
        report["shapes_valid"] = bool(shapes_ok)
        if expected_count is not None:
            report["count_matches_record"] = count == expected_count
        else:
            report["count_matches_record"] = None
        if not shapes_ok:
            report["error"] = "unexpected Gaussian array shape"
            return None, report
        if not report["finite"]:
            report["error"] = "Gaussian arrays contain nonfinite values"
            return None, report
        if expected_count is not None and count != expected_count:
            report["error"] = "Gaussian count differs from records.jsonl"
            return None, report
        report["readable"] = True
        report["saved_bytes"] = path.stat().st_size
        return arrays, report
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return None, report


def _read_pngs(directory: Path) -> tuple[list[Any], dict[str, Any]]:
    report: dict[str, Any] = {"count": 0, "valid": False, "views": []}
    try:
        import numpy as np
        from PIL import Image

        images: list[Any] = []
        total_bytes = 0
        for index in range(6):
            path = directory / f"view_{index:02d}.png"
            entry: dict[str, Any] = {"index": index, "path": str(path), "valid": False}
            if not path.is_file():
                entry["error"] = "missing PNG"
                report["views"].append(entry)
                continue
            try:
                with Image.open(path) as image:
                    pixels = np.asarray(image)
                    entry.update(format=image.format, mode=image.mode, width=int(image.width), height=int(image.height),
                                 shape=list(pixels.shape), dtype=str(pixels.dtype))
                    if pixels.size:
                        nonzero_count = int(np.count_nonzero(pixels))
                        pixel_min = pixels.min().item()
                        pixel_max = pixels.max().item()
                        if np.issubdtype(pixels.dtype, np.floating):
                            pixel_min = float(pixel_min)
                            pixel_max = float(pixel_max)
                        else:
                            pixel_min = int(pixel_min)
                            pixel_max = int(pixel_max)
                        entry.update(
                            pixel_min=pixel_min,
                            pixel_max=pixel_max,
                            nonzero_channel_count=nonzero_count,
                            channel_value_count=int(pixels.size),
                            nonzero_channel_fraction=float(nonzero_count / pixels.size),
                            blank_view=nonzero_count == 0,
                        )
                    else:
                        entry.update(
                            pixel_min=None,
                            pixel_max=None,
                            nonzero_channel_count=0,
                            channel_value_count=0,
                            nonzero_channel_fraction=None,
                            blank_view=None,
                        )
                    valid = image.format == "PNG" and image.mode == "RGB" and image.size == (384, 384) and pixels.shape == (384, 384, 3)
                    entry["valid"] = bool(valid)
                    if not valid:
                        entry["error"] = "expected 384x384 RGB PNG"
                    else:
                        images.append(pixels.copy())
                        total_bytes += path.stat().st_size
            except Exception as error:
                entry["error"] = f"{type(error).__name__}: {error}"
            report["views"].append(entry)
        report["count"] = sum(bool(view["valid"]) for view in report["views"])
        report["valid"] = report["count"] == 6
        report["blank_view_count"] = sum(view.get("blank_view") is True for view in report["views"])
        report["nonblank_view_count"] = sum(view.get("blank_view") is False for view in report["views"])
        report["saved_bytes"] = total_bytes
        return images if report["valid"] else [], report
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return [], report


def _product_for_record(run_root: Path, row: dict[str, Any], prompt: list[int] | None) -> dict[str, Any]:
    pair_id = str(row.get("pair_id", ""))
    method = str(row.get("method", ""))
    directory = run_root / "samples" / pair_id / method
    result: dict[str, Any] = {"directory": str(directory)}
    token_data, token_error = _token_payload(directory / "tokens.pt")
    result["tokens_readable"] = token_data is not None
    if token_error:
        result["tokens_error"] = token_error
    token_checks: dict[str, Any] = {}
    if token_data is not None:
        token_checks = _token_checks(token_data, row, prompt)
        result["token_payload"] = token_data
    result["token_checks"] = token_checks
    expected_gaussians = None
    stages = row.get("stages", {})
    quality = stages.get("quality", {}) if isinstance(stages, dict) else {}
    if isinstance(quality, dict) and quality.get("gaussians") is not None:
        expected_gaussians = int(quality["gaussians"])
    arrays, npz_report = _load_npz(directory / "gaussians.npz", expected_gaussians)
    result["gaussians"] = npz_report
    pngs, png_report = _read_pngs(directory)
    result["pngs"] = png_report
    result["valid"] = bool(
        token_data is not None and arrays is not None and png_report["valid"]
        and all(value is True for value in token_checks.values() if value is not None)
    )
    return result


def _coordinate_mapping(left_xyz: Any, right_xyz: Any, tree_factory: Callable[[Any], Any] | None = None) -> dict[str, Any]:
    if len(left_xyz) != len(right_xyz):
        return {"bijective": False, "reason": "Gaussian counts differ"}
    if not len(left_xyz):
        return {"bijective": False, "reason": "empty coordinate arrays"}
    try:
        if tree_factory is None:
            from scipy.spatial import cKDTree

            tree_factory = cKDTree
        distances, indices = tree_factory(right_xyz).query(left_xyz, k=1)
        if hasattr(distances, "reshape"):
            distances = distances.reshape(-1).tolist()
        if hasattr(indices, "reshape"):
            indices = indices.reshape(-1).tolist()
        distances = [float(value) for value in distances]
        indices = [int(value) for value in indices]
        unique = len(set(indices)) == len(indices)
        valid_indices = len(indices) == len(right_xyz) and all(0 <= index < len(right_xyz) for index in indices)
        if not valid_indices or not unique:
            return {"bijective": False, "reason": "nearest-coordinate mapping is not one-to-one",
                    "matched_count": int(len(indices)), "unique_matches": int(len(set(indices)))}
        if not all(math.isfinite(value) for value in distances):
            return {"bijective": False, "reason": "nonfinite coordinate distance"}
        return {"bijective": True, "indices": indices, "distances": distances,
                "max_distance": max(distances),
                "changed_row_fraction": sum(index != row for row, index in enumerate(indices)) / len(indices)}
    except Exception as error:
        return {"bijective": False, "reason": f"{type(error).__name__}: {error}"}


def _array_metrics(reference: Any, candidate: Any) -> dict[str, Any]:
    import numpy as np

    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    delta = np.abs(left - right)
    return {
        "equal": bool(np.array_equal(reference, candidate)),
        "max_abs": float(delta.max()) if delta.size else 0.0,
        "relative_l2": float(np.linalg.norm(delta) / max(float(np.linalg.norm(left)), 1e-30)),
    }


def _compare_gaussians(left_path: Path, right_path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(left_path, allow_pickle=False) as left_archive, np.load(right_path, allow_pickle=False) as right_archive:
        left = {name: left_archive[name] for name in left_archive.files}
        right = {name: right_archive[name] for name in right_archive.files}
    if set(left) != set(right):
        return {"available": False, "reason": "Gaussian field sets differ"}
    if left["means"].shape != right["means"].shape:
        return {"available": False, "reason": "Gaussian coordinate shapes differ"}
    mapping = _coordinate_mapping(left["means"].astype(np.float64), right["means"].astype(np.float64))
    if not mapping["bijective"]:
        return {"available": False, "coordinate_matching": mapping}
    indices = np.asarray(mapping.pop("indices"), dtype=np.int64)
    distances = mapping.pop("distances")
    max_distance = mapping["max_distance"]
    changed_row_fraction = mapping["changed_row_fraction"]
    return {
        "available": True,
        "coordinate_matching": mapping,
        "fields_aligned_by_xyz": {
            name: _array_metrics(left[name], right[name][indices]) for name in sorted(left)
        },
        "coordinate_distance_max": float(max_distance),
        "raw_row_order_changed_fraction": float(changed_row_fraction),
    }


def _compare_pngs(left_directory: Path, right_directory: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    views = []
    try:
        for index in range(6):
            left_path = left_directory / f"view_{index:02d}.png"
            right_path = right_directory / f"view_{index:02d}.png"
            with Image.open(left_path) as left_image, Image.open(right_path) as right_image:
                left = np.asarray(left_image)
                right = np.asarray(right_image)
            if left.shape != right.shape or left.shape != (384, 384, 3):
                return {"available": False, "reason": f"view {index} shapes differ or are invalid"}
            difference = np.abs(left.astype(np.int16) - right.astype(np.int16))
            views.append({
                "view": index,
                "equal": bool(np.array_equal(left, right)),
                "changed_channel_fraction": float((difference != 0).mean()),
                "changed_pixel_fraction": float(np.any(difference != 0, axis=2).mean()),
                "max_channel_difference": int(difference.max()) if difference.size else 0,
            })
        return {"available": True, "views": views,
                "interpretation": "Pixel differences describe saved output variation; they do not establish a quality change."}
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


def _audit_run(run_path: Path) -> dict[str, Any]:
    run_root = run_path.expanduser().resolve()
    report: dict[str, Any] = {"run": str(run_root), "errors": [], "products": [], "comparisons": []}
    errors: list[str] = report["errors"]
    try:
        plan = json.loads((run_root / "plan.json").read_text())
    except Exception as error:
        report.update(status="failed", errors=[f"plan.json: {type(error).__name__}: {error}"])
        return report
    try:
        progress = json.loads((run_root / "progress.json").read_text())
    except Exception as error:
        progress = {}
        errors.append(f"progress.json: {type(error).__name__}: {error}")
    expected, prompt_by_pair = _planned_keys(run_root, plan, errors)
    rows, row_errors = _jsonl(run_root / "records.jsonl")
    errors.extend(row_errors)
    actual_counts: dict[tuple[str, str], int] = {}
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("pair_id", "")), str(row.get("method", "")))
        actual_counts[key] = actual_counts.get(key, 0) + 1
        rows_by_key.setdefault(key, row)
    duplicates = sorted([list(key) for key, count in actual_counts.items() if count > 1])
    actual = set(actual_counts)
    missing = sorted([list(key) for key in expected - actual])
    extra = sorted([list(key) for key in actual - expected]) if expected else []
    if duplicates:
        errors.append("records.jsonl contains duplicate pair/method rows")
    if missing:
        errors.append("records.jsonl omits planned pair/method rows")
    if extra:
        errors.append("records.jsonl contains rows absent from plan.json/prompts.json")
    if progress.get("status") != "complete":
        errors.append(f"run progress is {progress.get('status', 'missing')!r}")
    if progress.get("completed") != progress.get("expected"):
        errors.append("progress completed count differs from expected count")
    if progress.get("expected") != len(expected):
        errors.append("progress expected count differs from plan/prompt count")
    if len(rows) != len(expected):
        errors.append("record row count differs from planned request count")
    methods = [str(method) for method in plan.get("methods", [])]
    if "w1" not in methods:
        errors.append("plan does not include W1, so saved tokens have no reference arm")

    products_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for key in sorted(actual):
        row = rows_by_key[key]
        product = _product_for_record(run_root, row, prompt_by_pair.get(key[0]))
        product.update(pair_id=key[0], method=key[1], source=row.get("source"),
                       prefix=row.get("prefix"), seed=row.get("seed"))
        report["products"].append(product)
        products_by_key[key] = product
        if not product["valid"]:
            errors.append(f"invalid saved products for pair {key[0]} method {key[1]}")

    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for (pair_id, method), row in rows_by_key.items():
        by_pair.setdefault(pair_id, {})[method] = row
    for pair_id, arms in sorted(by_pair.items()):
        if "w1" not in arms:
            report["comparisons"].append({"pair_id": pair_id, "available": False, "reason": "W1 row missing"})
            continue
        reference = arms["w1"]
        reference_key = (pair_id, "w1")
        reference_product = products_by_key.get(reference_key)
        for method, candidate in sorted(arms.items()):
            if method == "w1":
                continue
            candidate_product = products_by_key.get((pair_id, method))
            comparison: dict[str, Any] = {
                "pair_id": pair_id,
                "source": candidate.get("source"),
                "prefix": candidate.get("prefix"),
                "seed": candidate.get("seed"),
                "method": method,
                "reference_method": "w1",
                "tokens_equal": candidate.get("output_tokens") == reference.get("output_tokens"),
                "eos_equal": candidate.get("eos") == reference.get("eos"),
            }
            if not comparison["tokens_equal"]:
                errors.append(f"output token mismatch for pair {pair_id} method {method} vs W1")
            if reference_product and candidate_product and reference_product.get("valid") and candidate_product.get("valid"):
                left_directory = Path(reference_product["directory"])
                right_directory = Path(candidate_product["directory"])
                comparison["token_files_equal"] = (
                    reference_product.get("token_payload", {}).get("tokens") == candidate_product.get("token_payload", {}).get("tokens")
                    and reference_product.get("token_payload", {}).get("suffix") == candidate_product.get("token_payload", {}).get("suffix")
                    and reference_product.get("token_payload", {}).get("has_eos") == candidate_product.get("token_payload", {}).get("has_eos")
                    and reference_product.get("token_payload", {}).get("prefix_tokens") == candidate_product.get("token_payload", {}).get("prefix_tokens")
                )
                comparison["gaussians"] = _compare_gaussians(left_directory / "gaussians.npz",
                                                               right_directory / "gaussians.npz")
                comparison["pngs"] = _compare_pngs(left_directory, right_directory)
                if not comparison["token_files_equal"]:
                    errors.append(f"tokens.pt mismatch for pair {pair_id} method {method} vs W1")
                if not comparison["gaussians"].get("available"):
                    errors.append(f"Gaussian coordinate pairing failed for pair {pair_id} method {method} vs W1")
                if not comparison["pngs"].get("available"):
                    errors.append(f"PNG comparison failed for pair {pair_id} method {method} vs W1")
            else:
                comparison.update(available=False, reason="one or both paired products are invalid")
            report["comparisons"].append(comparison)

    compared = report["comparisons"]
    report["summary"] = {
        "planned_requests": len(expected),
        "record_rows": len(rows),
        "valid_products": sum(bool(product["valid"]) for product in report["products"]),
        "product_rows": len(report["products"]),
        "w1_comparisons": len(compared),
        "token_equal_pairs": sum(item.get("tokens_equal") is True and item.get("token_files_equal") is True for item in compared),
        "gaussian_coordinate_matches": sum(item.get("gaussians", {}).get("available") is True for item in compared),
        "png_comparisons": sum(item.get("pngs", {}).get("available") is True for item in compared),
    }
    report["status"] = "passed" if not errors else "failed"
    report["interpretation"] = (
        "This report validates saved-file readability, finiteness, layout, and token parity. "
        "Gaussian and PNG differences are descriptive and do not establish improved scene quality."
    )
    return report


def _synthetic_case() -> dict[str, Any]:
    class SmallTree:
        def __init__(self, points: list[tuple[float, float, float]]):
            self.points = points

        def query(self, points: list[tuple[float, float, float]], k: int = 1):
            distances, indices = [], []
            for point in points:
                options = [(math.dist(point, other), index) for index, other in enumerate(self.points)]
                distance, index = min(options)
                distances.append(distance)
                indices.append(index)
            return distances, indices

    left = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    right = [(0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    mapping = _coordinate_mapping(left, right, tree_factory=SmallTree)
    assert mapping["bijective"] and mapping["indices"] == [1, 2, 0]
    collision = _coordinate_mapping([(0.0, 0.0, 0.0), (0.001, 0.0, 0.0)],
                                   [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], tree_factory=SmallTree)
    assert not collision["bijective"]
    suffix = [5, EOS_ID]
    prompt = [0, 1, 2, 3, 4]
    payload = {"tokens": [1, 2, 3, 4, 5], "suffix": suffix, "has_eos": True, "prefix_tokens": 4}
    row = {"output_tokens": suffix, "eos": True, "prefix": 4}
    checks = _token_checks(payload, row, prompt)
    assert all(value is True for value in checks.values())
    return {"coordinate_indices": mapping["indices"], "max_distance": mapping["max_distance"],
            "collision_rejected": not collision["bijective"], "body_example": payload["tokens"],
            "token_checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="one or more completed E2E run directories")
    parser.add_argument("--out", required=True, type=Path, help="new JSON report path")
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(out)
    result = {"status": "passed", "runs": [_audit_run(path) for path in args.runs],
              "interpretation": "Per-run product audit; image and Gaussian deltas are descriptive only."}
    result["status"] = "passed" if all(run["status"] == "passed" for run in result["runs"]) else "failed"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]),
                      "out": str(out)}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
