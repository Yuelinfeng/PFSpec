#!/usr/bin/env python3
"""Read exported Gaussian arrays and PNGs independently after GPU work ends."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    progress = json.loads((root / "progress.json").read_text())
    if progress["status"] != "complete":
        raise ValueError("GPU run must finish before this readback")
    records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    if len(records) != progress["expected"]:
        raise ValueError("Missing output rows")
    results = []
    for row in records:
        directory = Path(row["output_dir"])
        count = row["quality"]["gaussians"]
        shapes = dict(means=(count, 3), scales=(count, 3), opacities=(count,),
                      quats=(count, 4), colors=(count, 3))
        with np.load(directory / "gaussians.npz") as arrays:
            if set(arrays.files) != set(shapes):
                raise ValueError("Unexpected Gaussian fields")
            for name, shape in shapes.items():
                if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                    raise ValueError("Invalid saved Gaussian array")
            norms = np.linalg.norm(arrays["quats"], axis=-1)
            if not np.allclose(norms, 1, atol=1e-5, rtol=0):
                raise ValueError("Invalid saved quaternion norms")
            for name in ("colors", "opacities"):
                if arrays[name].min() < 0 or arrays[name].max() > 1:
                    raise ValueError("Unexpected saved color or opacity range")
        byte_count = (directory / "gaussians.npz").stat().st_size
        for index in range(6):
            path = directory / f"view_{index:02d}.png"
            with Image.open(path) as view:
                pixels = np.asarray(view)
                if view.size != (384, 384) or view.mode != "RGB" or pixels.max() <= 0:
                    raise ValueError("Invalid or empty saved RGB view")
            byte_count += path.stat().st_size
        results.append(dict(cohort=row["cohort"], pair_id=row["pair_id"], method=row["method"],
                            gaussians=count, png_count=6, saved_bytes=byte_count, passed=True))
    result = dict(status="complete", runs=len(results), passed=len(results),
                  gaussian_files=len(results), png_files=6*len(results), rows=results)
    paired = {}
    for row in records:
        paired.setdefault((row["cohort"], row["pair_id"]), {})[row["method"]] = row
    comparisons = []
    for (cohort, pair_id), arms in paired.items():
        if "w1" not in arms or "rescue" not in arms:
            continue
        left, right = arms["w1"], arms["rescue"]
        a, b = Path(left["output_dir"]), Path(right["output_dir"])
        differences = {}
        with np.load(a / "gaussians.npz") as ga, np.load(b / "gaussians.npz") as gb:
            if set(ga.files) != set(gb.files):
                raise ValueError("Paired Gaussian fields differ")
            for name in ga.files:
                if ga[name].shape != gb[name].shape:
                    raise ValueError("Paired Gaussian shapes differ")
                delta = np.abs(ga[name].astype(np.float64)-gb[name].astype(np.float64))
                differences[name] = dict(equal=bool(np.array_equal(ga[name], gb[name])),
                    max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                    relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(ga[name].astype(np.float64)),1e-30)))
            from scipy.spatial import cKDTree
            distance, indices = cKDTree(gb["means"]).query(ga["means"], k=1)
            unique = len(np.unique(indices)) == len(indices)
            aligned = {}
            if unique:
                for name in ga.files:
                    aa, bb = ga[name].astype(np.float64), gb[name][indices].astype(np.float64)
                    delta = np.abs(aa-bb)
                    aligned[name] = dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),
                        relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(aa),1e-30)))
        pixels = []
        for index in range(6):
            with Image.open(a / f"view_{index:02d}.png") as ia, \
                    Image.open(b / f"view_{index:02d}.png") as ib:
                pa, pb = np.asarray(ia), np.asarray(ib)
                delta = np.abs(pa.astype(np.int16)-pb.astype(np.int16))
                pixels.append(dict(view=index, equal=bool(np.array_equal(pa, pb)),
                    max_abs=int(delta.max()), mean_abs=float(delta.mean()),
                    changed_fraction=float((delta != 0).mean()),
                    mse=float(np.square(delta.astype(np.float64)).mean())))
        comparisons.append(dict(cohort=cohort, pair_id=pair_id,
            tokens_equal=left["output_tokens"] == right["output_tokens"],
            eos_equal=left["eos"] == right["eos"],
            gaussian_fields=differences, png_views=pixels,
            coordinate_matching=dict(bijective=unique, max_distance=float(distance.max()),
                changed_row_fraction=float((indices != np.arange(len(indices))).mean()),
                aligned_fields=aligned,
                scope="One-to-one nearest-coordinate correspondence within the measured small distances; raw row order is reported separately."),
            gaussian_equal=all(v["equal"] for v in differences.values()),
            png_equal=all(v["equal"] for v in pixels)))
    result["w1_rescue_comparisons"] = comparisons
    result["w1_rescue_summary"] = dict(pairs=len(comparisons),
        tokens_equal=sum(v["tokens_equal"] for v in comparisons),
        gaussian_equal=sum(v["gaussian_equal"] for v in comparisons),
        png_equal=sum(v["png_equal"] for v in comparisons),
        scope="Paired output preservation relative to W1; no claim of improved absolute scene quality.")
    (root / "output_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items()
                      if k not in {"rows", "w1_rescue_comparisons"}}))


if __name__ == "__main__":
    main()
