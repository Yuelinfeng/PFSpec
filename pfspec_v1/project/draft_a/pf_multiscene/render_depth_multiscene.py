#!/usr/bin/env python3
"""Choose free-space views in the main living room and render metric depth."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scenezoo import get_dataset

WIDTH, HEIGHT = 640, 480
FOV_X_DEG = 70.0
YAW_SETS = ([0, 120, 240], [60, 180, 300])


def _basis(yaw_deg: float):
    angle = np.deg2rad(yaw_deg)
    forward = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(right, forward)
    up /= max(np.linalg.norm(up), 1e-12)
    return right, up, forward


def _rays(eye, yaw_deg, width, height):
    focal = width / (2 * np.tan(np.deg2rad(FOV_X_DEG / 2)))
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32) + 0.5,
        np.arange(height, dtype=np.float32) + 0.5,
    )
    right, up, forward = _basis(yaw_deg)
    directions = (
        forward[None, None, :]
        + ((u - width / 2) / focal)[..., None] * right
        + ((height / 2 - v) / focal)[..., None] * up
    )
    origins = np.broadcast_to(np.asarray(eye, dtype=np.float32), (height, width, 3))
    return np.concatenate([origins, directions], axis=-1).astype(np.float32)


def _depth(ray_scene, eye, yaw_deg, width, height):
    hits = ray_scene.cast_rays(o3d.core.Tensor(_rays(eye, yaw_deg, width, height)))
    depth = hits["t_hit"].numpy()
    valid = np.isfinite(depth) & (depth > 0)
    return np.where(valid, depth, 0).astype(np.float32), valid


def _score(ray_scene, eye, yaw_deg):
    depth, valid = _depth(ray_scene, eye, yaw_deg, 80, 60)
    values = depth[valid]
    if values.size == 0:
        return {"score": -10.0, "hit_ratio": 0.0, "p10": 0.0, "p50": 0.0}
    p10, p50 = np.percentile(values, [10, 50]).tolist()
    score = float(valid.mean()) - 0.25 * max(0.0, 0.35 - p10)
    score -= 0.15 * max(0.0, 0.5 - p50)
    return {"score": score, "hit_ratio": float(valid.mean()), "p10": float(p10), "p50": float(p50)}


def _select_room(dataset, scene_id, scene_data):
    room_types = {
        room.get("instanceid"): room.get("type", "")
        for room in scene_data.get("scene", {}).get("room", [])
    }
    choices = []
    for room_id in dataset.get_room_ids(scene_id):
        room_type = room_types.get(room_id, "")
        if room_type not in {"LivingRoom", "LivingDiningRoom"}:
            continue
        mesh = dataset.get_mesh(scene_id, mesh_type="full", room_id=room_id)
        if mesh.is_empty():
            continue
        bounds = mesh.get_axis_aligned_bounding_box()
        lower, upper = bounds.get_min_bound(), bounds.get_max_bound()
        area = max(0.0, float(upper[0] - lower[0])) * max(
            0.0, float(upper[1] - lower[1])
        )
        score = len(mesh.triangles) + 50.0 * area
        choices.append((score, room_id, room_type, mesh, lower, upper, area))
    if not choices:
        raise RuntimeError(f"No living room found for {scene_id}")
    _, room_id, room_type, mesh, lower, upper, area = max(choices, key=lambda item: item[0])
    return {
        "room_id": room_id,
        "room_type": room_type,
        "mesh": mesh,
        "lower": lower,
        "upper": upper,
        "area": area,
        "triangles": len(mesh.triangles),
    }


def _choose_eye(ray_scene, room):
    lower, upper = room["lower"], room["upper"]
    z = float(lower[2] + 1.45)
    if z >= upper[2] - 0.15:
        z = float((lower[2] + upper[2]) / 2)
    fractions = np.linspace(0.1, 0.9, 9)
    candidates = np.asarray(
        [
            [lower[0] + fx * (upper[0] - lower[0]), lower[1] + fy * (upper[1] - lower[1]), z]
            for fx in fractions
            for fy in fractions
        ],
        dtype=np.float32,
    )
    clearances = ray_scene.compute_distance(o3d.core.Tensor(candidates)).numpy()
    best = None
    yaws = sorted(set(YAW_SETS[0] + YAW_SETS[1]))
    for index in np.argsort(clearances)[::-1][:30]:
        eye = candidates[index].astype(np.float64)
        clearance = float(clearances[index])
        if clearance < 0.35:
            continue
        scores = {yaw: _score(ray_scene, eye, yaw) for yaw in yaws}
        for yaw_set in YAW_SETS:
            metrics = [scores[yaw] for yaw in yaw_set]
            view_score = float(np.mean([item["score"] for item in metrics]))
            candidate_score = view_score + 0.015 * min(clearance, 2.0)
            if best is None or candidate_score > best["candidate_score"]:
                best = {
                    "eye": eye,
                    "clearance": clearance,
                    "yaws": list(yaw_set),
                    "metrics": metrics,
                    "candidate_score": candidate_score,
                }
    if best is None:
        raise RuntimeError(f"No camera point clears scene geometry in {room['room_id']}")
    if min(item["hit_ratio"] for item in best["metrics"]) < 0.35:
        raise RuntimeError(f"Selected views have inadequate hit coverage: {best}")
    if min(item["p10"] for item in best["metrics"]) < 0.05:
        raise RuntimeError(f"Selected camera is too close to geometry: {best}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    args = parser.parse_args()
    start = time.perf_counter()

    scene_root = args.scene_root.resolve()
    scene_id = args.scene_id
    scene_data = json.loads((scene_root / "3D-FRONT" / f"{scene_id}.json").read_text())
    dataset = get_dataset("3dfront", scene_root)
    room = _select_room(dataset, scene_id, scene_data)
    mesh_path = scene_root / "pilot_full_zup.obj"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Empty full mesh: {mesh_path}")
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    choice = _choose_eye(ray_scene, room)

    out = scene_root / "depth_pilot"
    out.mkdir(parents=True, exist_ok=True)
    focal = WIDTH / (2 * np.tan(np.deg2rad(FOV_X_DEG / 2)))
    frames = []
    for index, yaw in enumerate(choice["yaws"]):
        name = f"living_{index}"
        depth, valid = _depth(ray_scene, choice["eye"], yaw, WIDTH, HEIGHT)
        depth_path = out / f"{name}_depth.npy"
        preview_path = out / f"{name}_depth.png"
        np.save(depth_path, depth)
        normalized = np.clip((depth - 0.2) / (12.0 - 0.2), 0, 1)
        preview = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        preview[valid] = ((1 - normalized[valid]) * 230 + 20).astype(np.uint8)
        Image.fromarray(preview).save(preview_path)

        right, up, forward = _basis(yaw)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = right, up, -forward
        c2w[:3, 3] = choice["eye"]
        frames.append(
            {
                "name": name,
                "depth_file": depth_path.name,
                "preview_file": preview_path.name,
                "camera_to_world": c2w.tolist(),
                "yaw_degrees": yaw,
                "hit_ratio": float(valid.mean()),
                "depth_p10": float(np.percentile(depth[valid], 10)),
                "depth_p50": float(np.percentile(depth[valid], 50)),
                "depth_p95": float(np.percentile(depth[valid], 95)),
            }
        )
        item = frames[-1]
        print(
            f"{name} yaw={yaw} hit={item['hit_ratio']:.4f} "
            f"p10/p50/p95={item['depth_p10']:.3f}/{item['depth_p50']:.3f}/{item['depth_p95']:.3f}",
            flush=True,
        )

    camera_data = {
        "width": WIDTH,
        "height": HEIGHT,
        "fx": float(focal),
        "fy": float(focal),
        "cx": WIDTH / 2,
        "cy": HEIGHT / 2,
        "coordinate_system": "Z-up world; OpenGL camera-to-world",
        "depth_type": "camera-forward Z depth, metres",
        "camera_selection": "living-room grid point ranked by geometry clearance and low-resolution ray coverage",
        "room_id": room["room_id"],
        "room_type": room["room_type"],
        "frames": frames,
    }
    (out / "cameras.json").write_text(json.dumps(camera_data, indent=2))
    record = {
        "scene_id": scene_id,
        "room_id": room["room_id"],
        "room_type": room["room_type"],
        "room_triangles": room["triangles"],
        "room_xy_area_m2": room["area"],
        "room_bounds_min": room["lower"].tolist(),
        "room_bounds_max": room["upper"].tolist(),
        "eye": choice["eye"].tolist(),
        "clearance_m": choice["clearance"],
        "selected_yaws": choice["yaws"],
        "view_scores": choice["metrics"],
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    (out / "camera_selection.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)
    print("MULTISCENE DEPTH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
