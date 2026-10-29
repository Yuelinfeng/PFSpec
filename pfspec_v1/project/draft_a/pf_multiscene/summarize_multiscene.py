#!/usr/bin/env python3
"""Summarize scene preparation metrics from a multiscene run directory."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch


def mean_l1(log: str, kind: str) -> float | None:
    match = re.search(rf"{kind} MEAN L1=([0-9.]+)", log)
    return float(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    split_data = json.loads(args.splits.read_text())
    scene_ids = split_data["splits"]["development"]["scene_ids"]
    with (root / "stage_b" / "scene_catalog.csv").open(newline="") as handle:
        catalog = {row["scene_id"]: row for row in csv.DictReader(handle)}
    timings = json.loads((root / "timings.json").read_text())["scenes"]
    asset_manifest = json.loads((root / "stage_b" / "selected_asset_manifest.json").read_text())
    crops = json.loads((root / "stage_c" / "crop_inventory.json").read_text())
    crop_rows = {row["scene_id"]: row for row in crops["scenes"]}
    status = json.loads((root / "stage_b" / "scene_pipeline_status.json").read_text())

    scenes = {}
    for scene_id in scene_ids:
        scene_root = root / "scenes" / scene_id
        camera = json.loads((scene_root / "depth_pilot" / "camera_selection.json").read_text())
        decoded = json.loads((scene_root / "gaussian_init" / "decoded_render" / "metrics.json").read_text())
        gaussian_log = (root / "logs" / scene_id / "gaussian_fit.log").read_text()
        geometry_log = (root / "logs" / scene_id / "geometry.log").read_text()
        full_section = geometry_log.split("===== FULL =====")[-1]
        triangles_match = re.search(r"Triangles: ([0-9]+)", full_section)
        token_path = scene_root / "gaussian_init" / "vfront_pilot_tokens.pt"
        token_data = torch.load(token_path, map_location="cpu", weights_only=True)
        row = catalog[scene_id]
        stages = timings[scene_id]
        stage_seconds = {name: value["seconds"] for name, value in stages.items()}
        scene_asset = asset_manifest["scenes"][scene_id]
        scenes[scene_id] = {
            "source_group_id": row["group_id"],
            "split": row["split"],
            "stratum": row["stratum"],
            "rooms": int(row["room_count"]),
            "room_types": row["room_types"].split(";"),
            "active_furniture_instances": int(row["active_furniture_instances"]),
            "active_model_ids": len(scene_asset["model_ids"]),
            "missing_models": [
                value for value in row["missing_model_ids"].split(";") if value
            ],
            "model_coverage": float(row["model_coverage"]),
            "layout_texture_ids_requested": len(scene_asset["texture_ids"]),
            "layout_texture_ids_missing_png": len(scene_asset["missing_texture_ids"]),
            "full_mesh_triangles": int(triangles_match.group(1)) if triangles_match else None,
            "camera_room": camera["room_id"],
            "camera_room_type": camera["room_type"],
            "camera_clearance_m": camera["clearance_m"],
            "camera_yaws": camera["selected_yaws"],
            "camera_hit_ratios": [view["hit_ratio"] for view in camera["view_scores"]],
            "depth_recast_p99_m": [
                float(value)
                for value in re.findall(r"Depth recast P99 error: ([0-9.eE+-]+) m", (root / "logs" / scene_id / "depth.log").read_text())
            ],
            "input_voxels": stages["gaussian_fit"].get("input_voxels"),
            "gaussians_fit": stages["gaussian_fit"].get("gaussians"),
            "initial_render_l1": mean_l1(gaussian_log, "initial"),
            "optimized_render_l1": mean_l1(gaussian_log, "optimized"),
            "fit_l1_improvement_fraction": (
                1.0 - mean_l1(gaussian_log, "optimized") / mean_l1(gaussian_log, "initial")
                if mean_l1(gaussian_log, "initial") and mean_l1(gaussian_log, "optimized") is not None
                else None
            ),
            "input_latents": int(token_data["coords"].shape[0]),
            "decoded_gaussians": decoded["decoded_gaussians"],
            "decode_mae": decoded["mean_mae"],
            "decode_psnr_db": decoded["mean_psnr_db"],
            "scale_clipped_fraction": decoded["scale_clipped_fraction"],
            "peak_decode_gpu_gib": decoded["peak_gpu_allocated_gib"],
            "stage_seconds": stage_seconds,
            "pipeline_seconds": round(sum(stage_seconds.values()), 3),
            "crop_inventory": crop_rows[scene_id],
            "pipeline_state": status["results"][scene_id]["state"],
            "render_output_bytes": sum(
                path.stat().st_size
                for path in scene_root.rglob("*")
                if path.is_file() and not path.is_symlink()
            ),
        }

    summary = {
        "run_root": str(root),
        "development_scene_count": len(scene_ids),
        "development_crop_count": crops["crop_records"],
        "development_body_tokens": crops["total_body_tokens"],
        "selected_assets": {
            key: asset_manifest[key]
            for key in [
                "selected_scene_count",
                "model_ids_requested",
                "model_ids_extracted",
                "model_bytes_extracted",
                "texture_ids_requested",
                "texture_ids_extracted",
                "texture_png_ids_available",
                "texture_bytes_extracted",
            ]
        },
        "limitations": [
            "One development scene yielded one unique crop after seeds 0 through 4.",
            "Texture IDs without texture.png use the scene reader's material-color path.",
            "Rendered image metrics cover three selected camera views per scene.",
        ],
        "scenes": scenes,
    }
    output = root / "stage_b" / "development_scene_summary.json"
    output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {output} for {len(scenes)} development scenes")


if __name__ == "__main__":
    main()
