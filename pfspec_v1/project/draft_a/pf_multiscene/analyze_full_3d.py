#!/usr/bin/env python3
"""Summarize stage F outputs and encode fixed-view comparison reels."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

MODES = ("target_only", "p_only", "step12000")
MODE_LABELS = {
    "target_only": "Target",
    "p_only": "P-only",
    "step12000": "Light PF",
}
BOOTSTRAP_DRAWS = 10000


def read_json(path: Path):
    return json.loads(path.read_text())


def quantile(values, probability):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    point = (len(values) - 1) * probability
    lower = math.floor(point)
    upper = math.ceil(point)
    return values[lower] + (values[upper] - values[lower]) * (point - lower)


def describe(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p25": quantile(values, 0.25),
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def geometric_mean(values):
    values = [float(value) for value in values if value is not None and value > 0]
    if not values:
        return None
    return math.exp(statistics.mean(math.log(value) for value in values))


def bootstrap_geomean(scene_values, seed):
    names = sorted(scene_values)
    if not names:
        return {"draws": BOOTSTRAP_DRAWS, "low": None, "high": None}
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        selected = [scene_values[rng.choice(names)] for _ in names]
        draws.append(geometric_mean(selected))
    return {
        "draws": BOOTSTRAP_DRAWS,
        "low": quantile(draws, 0.025),
        "high": quantile(draws, 0.975),
    }


def read_rows(root: Path, name: str):
    summary_path = root / name / "quality_summary.json"
    summary = read_json(summary_path)
    rows = summary.get("rows", [])
    if summary.get("status") != "complete" or summary.get("failed") != 0:
        raise ValueError(f"Incomplete run summary: {summary_path}")
    if any(row.get("status") != "complete" for row in rows):
        raise ValueError(f"A row is incomplete: {summary_path}")
    return summary, rows


def index_rows(rows):
    result = {}
    for row in rows:
        key = (row["sample"], row["variant"])
        if key in result:
            raise ValueError(f"Duplicate result row: {key}")
        result[key] = row
    return result


def paired_speed(rows, group_field, seed):
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        groups[row[group_field]][row["sample"]][row["variant"]] = row
    result = {}
    for mode_index, mode in enumerate(MODES[1:]):
        sample_ratios = []
        group_ratios = {}
        for group, samples in groups.items():
            ratios = []
            for modes in samples.values():
                if set(MODES).issubset(modes):
                    ratio = modes["target_only"]["total_seconds"] / modes[mode]["total_seconds"]
                    sample_ratios.append(ratio)
                    ratios.append(ratio)
            if ratios:
                group_ratios[group] = geometric_mean(ratios)
        result[mode] = {
            "paired_rows": len(sample_ratios),
            "paired_geometric_mean": geometric_mean(sample_ratios),
            "paired_median": statistics.median(sample_ratios) if sample_ratios else None,
            "paired_win_fraction": sum(ratio > 1 for ratio in sample_ratios) / len(sample_ratios)
            if sample_ratios else None,
            "group_count": len(group_ratios),
            "groups_faster_than_target": sum(value > 1 for value in group_ratios.values()),
            "group_geometric_mean": geometric_mean(group_ratios.values()),
            "group_95pct_cluster_bootstrap": bootstrap_geomean(group_ratios, seed + mode_index),
            "pooled_wall_time_ratio": sum(
                row["total_seconds"] for row in rows if row["variant"] == "target_only"
            ) / sum(row["total_seconds"] for row in rows if row["variant"] == mode),
        }
    return result


def direct_speed_comparison(rows, reference_mode, candidate_mode, group_field, seed):
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        groups[row[group_field]][row["sample"]][row["variant"]] = row
    ratios = []
    group_ratios = {}
    for group, samples in groups.items():
        within_group = []
        for modes in samples.values():
            if reference_mode in modes and candidate_mode in modes:
                ratio = modes[reference_mode]["total_seconds"] / modes[candidate_mode]["total_seconds"]
                ratios.append(ratio)
                within_group.append(ratio)
        if within_group:
            group_ratios[group] = geometric_mean(within_group)
    return {
        "reference": reference_mode,
        "candidate": candidate_mode,
        "paired_rows": len(ratios),
        "paired_geometric_mean": geometric_mean(ratios),
        "paired_median": statistics.median(ratios) if ratios else None,
        "paired_win_fraction": sum(value > 1 for value in ratios) / len(ratios) if ratios else None,
        "group_count": len(group_ratios),
        "groups_faster_than_reference": sum(value > 1 for value in group_ratios.values()),
        "group_geometric_mean": geometric_mean(group_ratios.values()),
        "group_95pct_cluster_bootstrap": bootstrap_geomean(group_ratios, seed),
        "pooled_wall_time_ratio": sum(
            row["total_seconds"] for row in rows if row["variant"] == reference_mode
        ) / sum(row["total_seconds"] for row in rows if row["variant"] == candidate_mode),
    }


def mode_summary(rows, prefix_tokens):
    result = {}
    for mode in MODES:
        selected = [row for row in rows if row["variant"] == mode]
        seconds = [row["total_seconds"] for row in selected]
        generated = [row["generated_tokens"] for row in selected]
        quality = [row.get("quality", {}) for row in selected]
        acceptance = [row.get("acceptance", {}) for row in selected]
        result[mode] = {
            "count": len(selected),
            "success": sum(row["status"] == "complete" for row in selected),
            "natural_eos_count": sum(row.get("eos") is True for row in selected),
            "capped_count": sum(row.get("capped") is True for row in selected),
            "total_seconds": describe(seconds),
            "ar_seconds": describe([row.get("ar_seconds") for row in selected]),
            "decode_seconds": describe([row.get("decode_seconds") for row in selected]),
            "export_seconds": describe([row.get("export_seconds") for row in selected]),
            "render_seconds": describe([row.get("render_seconds") for row in selected]),
            "png_seconds": describe([row.get("png_seconds") for row in selected]),
            "body_tokens_including_prefix": describe([row.get("body_tokens") for row in selected]),
            "new_content_tokens_excluding_prefix": describe([
                max(0, row.get("body_tokens", 0) - prefix_tokens) for row in selected
            ]),
            "generated_tokens_per_ar_second": sum(generated) / sum(
                row["ar_seconds"] for row in selected
            ) if selected else None,
            "generated_tokens_per_end_to_end_second": sum(generated) / sum(seconds) if selected else None,
            "gaussian_count": describe([value.get("gaussians") for value in quality]),
            "scale_clipped_fraction": describe([value.get("scale_clipped_fraction") for value in quality]),
            "mean_opacity": describe([value.get("mean_opacity") for value in quality]),
            "alpha_coverage": describe([row.get("alpha_coverage") for row in selected]),
            "full_reference_psnr_db": describe([row.get("full_reference_psnr_db") for row in selected]),
            "full_reference_ssim": describe([row.get("full_reference_ssim") for row in selected]),
            "acceptance_sample_means": {
                key: describe([item.get(key) for item in acceptance])
                for key in (
                    "acceptance_verified", "acceptance_proposed",
                    "position_acceptance_verified", "position_acceptance_proposed",
                    "feature_acceptance_verified", "feature_acceptance_proposed",
                )
            },
        }
    return result


def sample_csv_rows(rows, prefix_tokens):
    result = []
    for row in rows:
        quality = row.get("quality", {})
        acceptance = row.get("acceptance", {})
        result.append({
            "sample": row["sample"],
            "scene": row["scene"],
            "seed": row["seed"],
            "variant": row["variant"],
            "status": row["status"],
            "eos": row.get("eos"),
            "capped": row.get("capped"),
            "body_tokens_including_prefix": row.get("body_tokens"),
            "new_content_tokens_excluding_prefix": max(0, row.get("body_tokens", 0) - prefix_tokens),
            "generated_tokens": row.get("generated_tokens"),
            "ar_seconds": row.get("ar_seconds"),
            "decode_seconds": row.get("decode_seconds"),
            "export_seconds": row.get("export_seconds"),
            "render_seconds": row.get("render_seconds"),
            "png_seconds": row.get("png_seconds"),
            "total_seconds": row.get("total_seconds"),
            "gaussians": quality.get("gaussians"),
            "scale_clipped_fraction": quality.get("scale_clipped_fraction"),
            "mean_opacity": quality.get("mean_opacity"),
            "alpha_coverage": row.get("alpha_coverage"),
            "full_reference_psnr_db": row.get("full_reference_psnr_db"),
            "full_reference_ssim": row.get("full_reference_ssim"),
            "acceptance_verified": acceptance.get("acceptance_verified"),
            "acceptance_proposed": acceptance.get("acceptance_proposed"),
            "position_acceptance_verified": acceptance.get("position_acceptance_verified"),
            "position_acceptance_proposed": acceptance.get("position_acceptance_proposed"),
            "feature_acceptance_verified": acceptance.get("feature_acceptance_verified"),
            "feature_acceptance_proposed": acceptance.get("feature_acceptance_proposed"),
            "target_calls": acceptance.get("target_calls"),
            "draft_calls": acceptance.get("draft_calls"),
            "ranker_calls": acceptance.get("ranker_calls"),
        })
    return result


def scene_csv_rows(rows):
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        grouped[row["scene"]][row["seed"]][row["variant"]] = row
    result = []
    for scene, seed_groups in sorted(grouped.items()):
        for mode in MODES:
            paired_ratios = []
            target_seconds = []
            mode_seconds = []
            selected = []
            for modes in seed_groups.values():
                if set(MODES).issubset(modes):
                    target_seconds.append(modes["target_only"]["total_seconds"])
                    mode_seconds.append(modes[mode]["total_seconds"])
                    paired_ratios.append(modes["target_only"]["total_seconds"] / modes[mode]["total_seconds"])
                    selected.append(modes[mode])
            result.append({
                "scene": scene,
                "variant": mode,
                "seed_count": len(selected),
                "mean_total_seconds": statistics.mean(mode_seconds) if mode_seconds else None,
                "mean_ar_seconds": statistics.mean(row["ar_seconds"] for row in selected) if selected else None,
                "mean_body_tokens": statistics.mean(row["body_tokens"] for row in selected) if selected else None,
                "paired_target_speed_geomean": geometric_mean(paired_ratios),
                "paired_wall_time_ratio": sum(target_seconds) / sum(mode_seconds)
                if mode_seconds and sum(mode_seconds) else None,
                "natural_eos_fraction": sum(row.get("eos") is True for row in selected) / len(selected)
                if selected else None,
            })
    return result


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def encode_comparison_video(root: Path, rows, prefix_tokens, output_path: Path,
                            group_field: str, group_names, fps: int, seconds_per_group: float):
    import av
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    grouped = defaultdict(dict)
    for row in rows:
        grouped[row[group_field]][row["variant"]] = row
    groups = [name for name in group_names if set(MODES).issubset(grouped.get(name, {}))]
    if not groups:
        raise ValueError(f"No complete three-mode groups for {output_path.name}")

    view_size = 384
    width = view_size * 6
    top_height = 48
    row_header_height = 40
    row_height = row_header_height + view_size
    height = top_height + row_height * len(MODES)
    if width % 2 or height % 2:
        raise ValueError("H.264 video dimensions must be even")
    font = ImageFont.load_default(size=22)
    small_font = ImageFont.load_default(size=18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.bit_rate = 4_000_000
    stream.options = {"crf": "22", "preset": "medium"}
    stream.codec_context.gop_size = fps * max(1, round(seconds_per_group))
    stream.time_base = Fraction(1, fps)
    frame_number = 0

    for group_index, group in enumerate(groups, start=1):
        mode_rows = grouped[group]
        longest = max(mode_rows[mode]["total_seconds"] for mode in MODES)
        if group_field == "scene":
            source_name = group
            title = f"Test scene {group_index}/{len(groups)} | {source_name} | seed={mode_rows[MODES[0]]['seed']}"
        else:
            source_name = group
            title = f"BOS trajectory {group_index}/{len(groups)} | seed={mode_rows[MODES[0]]['seed']}"
        canvas = Image.new("RGB", (width, height), "#f4f5f3")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), title, fill="#172126", font=font)
        draw.text((width - 650, 14), "6 fixed shared camera views", fill="#4b5960", font=small_font)
        for mode_index, mode in enumerate(MODES):
            row = mode_rows[mode]
            y = top_height + mode_index * row_height
            label = f"{MODE_LABELS[mode]}   E2E {row['total_seconds']:.2f} s   {row['body_tokens']} tokens   EOS {int(bool(row.get('eos')))}"
            draw.rectangle((0, y, width, y + row_header_height), fill="#dfe5e4")
            draw.text((12, y + 8), label, fill="#172126", font=small_font)
            bar_start = 900
            bar_end = width - 24
            bar_y = y + 16
            draw.rounded_rectangle((bar_start, bar_y, bar_end, bar_y + 12), radius=5, fill="#c1cbca")
            bar_fill = bar_start + round((bar_end - bar_start) * row["total_seconds"] / longest)
            draw.rounded_rectangle((bar_start, bar_y, max(bar_start + 4, bar_fill), bar_y + 12), radius=5,
                                   fill="#2c7371" if mode != "target_only" else "#465d70")
            sample_dir = Path(row["output_dir"])
            if not sample_dir.is_absolute():
                sample_dir = root / sample_dir
            for view_index in range(6):
                image_path = sample_dir / f"view_{view_index:02d}.png"
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    if image.size != (view_size, view_size):
                        raise ValueError(f"Unexpected view size: {image_path} {image.size}")
                    canvas.paste(image, (view_index * view_size, y + row_header_height))

        rgb = np.asarray(canvas, dtype=np.uint8)
        for _ in range(max(1, round(fps * seconds_per_group))):
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = frame_number
            frame.time_base = Fraction(1, fps)
            frame_number += 1
            for packet in stream.encode(frame):
                container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    preview_path = output_path.with_suffix(".preview.png")
    validation = av.open(str(output_path))
    video_stream = validation.streams.video[0]
    decoded_frames = 0
    first_frame = None
    for decoded in validation.decode(video_stream):
        decoded_frames += 1
        if first_frame is None:
            first_frame = decoded.to_ndarray(format="rgb24")
    validation.close()
    expected_frames = len(groups) * max(1, round(fps * seconds_per_group))
    if decoded_frames != expected_frames or first_frame is None:
        raise ValueError(f"Video decode yielded {decoded_frames} frames; expected {expected_frames}")
    Image.fromarray(first_frame).save(preview_path)
    return {
        "path": str(output_path),
        "preview_path": str(preview_path),
        "groups": len(groups),
        "fps": fps,
        "seconds_per_group": seconds_per_group,
        "duration_seconds": len(groups) * max(1, round(fps * seconds_per_group)) / fps,
        "width": width,
        "height": height,
        "decoded_frames": decoded_frames,
        "display_semantics": "Final renders are held for viewing; E2E labels and relative bars use measured prefix-to-output seconds.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Remote stage_f_full3d result directory")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to ROOT/analysis")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seconds-per-group", type=float, default=2.0)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)

    completion_summary, completion_rows = read_rows(root, "test_completion")
    bos_summary, bos_rows = read_rows(root, "bos_generation")
    completion_plan = read_json(root / "test_completion/plan.json")
    bos_plan = read_json(root / "bos_generation/plan.json")
    prefix_tokens = int(completion_plan.get("prefix_tokens", 256))

    expected_completion = int(completion_summary.get("expected", 0))
    expected_bos = int(bos_summary.get("expected", 0))
    if len(completion_rows) != expected_completion or len(bos_rows) != expected_bos:
        raise ValueError("Summary row count differs from its expected count")
    for name, rows, expected_groups in (
        ("test_completion", completion_rows, expected_completion // len(MODES)),
        ("bos_generation", bos_rows, expected_bos // len(MODES)),
    ):
        groups = defaultdict(set)
        for row in rows:
            groups[row["sample"]].add(row["variant"])
        if len(groups) != expected_groups or any(variants != set(MODES) for variants in groups.values()):
            raise ValueError(f"Missing mode rows in {name}")

    completion_seed = int(completion_plan.get("seeds", [51701])[0])
    video_seed = completion_seed
    video_rows = [row for row in completion_rows if int(row["seed"]) == video_seed]
    scene_order = list(dict.fromkeys(row["scene"] for row in completion_rows))
    bos_order = list(dict.fromkeys(row["sample"] for row in bos_rows))

    completion_modes = mode_summary(completion_rows, prefix_tokens)
    bos_modes = mode_summary(bos_rows, 0)
    completion_speed = paired_speed(completion_rows, "scene", 20261016)
    bos_speed = paired_speed(bos_rows, "sample", 20261018)
    pf_vs_p_completion = direct_speed_comparison(
        completion_rows, "p_only", "step12000", "scene", 20261020)
    pf_vs_p_bos = direct_speed_comparison(
        bos_rows, "p_only", "step12000", "sample", 20261021)

    completion_stats = {
        "expected": expected_completion,
        "completed": len(completion_rows),
        "scene_count": len(scene_order),
        "seed_count": len(set(row["seed"] for row in completion_rows)),
        "prefix_tokens": prefix_tokens,
        "body_limit": completion_plan.get("body_limit"),
        "modes": completion_modes,
        "paired_end_to_end_speed_vs_target": completion_speed,
        "light_pf_vs_p_only": pf_vs_p_completion,
    }
    bos_stats = {
        "expected": expected_bos,
        "completed": len(bos_rows),
        "trajectory_count": len(bos_order),
        "seed_count": len(set(row["seed"] for row in bos_rows)),
        "prefix_tokens": 0,
        "body_limit": bos_plan.get("body_limit"),
        "modes": bos_modes,
        "paired_end_to_end_speed_vs_target": bos_speed,
        "light_pf_vs_p_only": pf_vs_p_bos,
    }

    video_completion = encode_comparison_video(
        root, video_rows, prefix_tokens, output / f"test_completion_30scenes_seed{video_seed}_6views.mp4",
        "scene", scene_order, args.fps, args.seconds_per_group)
    video_bos = encode_comparison_video(
        root, bos_rows, 0, output / "bos_30trajectories_6views.mp4",
        "sample", bos_order, args.fps, args.seconds_per_group)

    report = {
        "status": "complete",
        "scope": "single 20x20x20 latent chunk, FP32, temperature/top_p=0.9",
        "interpretation": {
            "timing": "Full-output wall time runs from the prepared prefix (or BOS) through six PNGs and Gaussian NPZ save.",
            "speed_interval": "95% percentile cluster bootstrap; resampling unit is source scene, retaining all three seeds.",
            "acceptance": "Acceptance values are arithmetic means across available sample-level rates; denominators are not present in the saved row summary.",
            "reference_metrics": "Full-reference PSNR/SSIM compare whole generated chunks with one real reconstruction; sampled suffixes can have multiple valid outcomes.",
            "videos": "All scenes use the predeclared first seed; BOS uses all 30 trajectory seeds. Six shared fixed views are shown per output.",
        },
        "test_completion": completion_stats,
        "bos_generation": bos_stats,
        "videos": {"test_completion": video_completion, "bos_generation": video_bos},
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_csv(output / "test_samples.csv", sample_csv_rows(completion_rows, prefix_tokens))
    write_csv(output / "bos_samples.csv", sample_csv_rows(bos_rows, 0))
    write_csv(output / "test_scenes.csv", scene_csv_rows(completion_rows))
    pipeline_path = root / "pipeline.json"
    if pipeline_path.exists():
        pipeline = read_json(pipeline_path)
        if pipeline.get("status") != "complete":
            raise ValueError("Stage F generation must be complete before analysis is recorded")
        pipeline["phase"] = "analysis_complete"
        pipeline["analysis_summary"] = str(output / "summary.json")
        pipeline["analysis_videos"] = [video_completion["path"], video_bos["path"]]
        pipeline_path.write_text(json.dumps(pipeline, indent=2) + "\n")
    print(json.dumps({
        "summary": str(output / "summary.json"),
        "test_completion_video": video_completion,
        "bos_video": video_bos,
    }, indent=2))


if __name__ == "__main__":
    main()
