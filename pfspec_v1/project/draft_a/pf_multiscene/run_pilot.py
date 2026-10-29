#!/usr/bin/env python3
"""Run the established GaussianGPT VFront pipeline in isolated scene folders."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("/root/autodl-tmp/pf_multiscene_pilot_20261015")
STAGES = [
    ("geometry", "export_vfront_pilot.py"),
    ("depth", "render_depth_multiscene.py"),
    ("rgb", "render_vfront_rgb_cpu.py"),
    ("voxel", "build_vfront_init_cloud.py"),
    ("gaussian_fit", "gsplat_vfront_multiview_export.py"),
    ("tokenize", "tokenize_vfront_pilot.py"),
    ("decode_render", "decode_and_render.py"),
]


def run(scene_id: str, root: Path) -> None:
    scene_root = root / "scenes" / scene_id
    scene_json = scene_root / "3D-FRONT" / f"{scene_id}.json"
    if not scene_json.is_file():
        raise FileNotFoundError(scene_json)

    scripts_dir = root / "scripts" / scene_id
    logs_dir = root / "logs" / scene_id
    work_dir = root / "work" / scene_id
    for path in (scripts_dir, logs_dir, work_dir / "draft_a"):
        path.mkdir(parents=True, exist_ok=True)
    (work_dir / "draft_a" / "vfront_pilot_scene.txt").write_text(scene_id + "\n")

    timing_path = root / "timings.json"
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else {"scenes": {}}
    timings.setdefault("scenes", {}).setdefault(scene_id, {})
    timings["run_root"] = str(root)
    timings["camera_method"] = "living-room free-space point and low-resolution ray coverage search"

    old_root = "/root/autodl-tmp/3D-FRONT-pilot"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "8")

    for stage, filename in STAGES:
        if stage in {"depth", "decode_render"}:
            source = PROJECT / "draft_a" / "pf_multiscene" / filename
        else:
            source = PROJECT / "draft_a" / filename
        content = source.read_text().replace(old_root, str(root / "scenes"))
        content = content.replace(
            '"checkpoints/vqvae_vfront.ckpt"',
            f'"{PROJECT / "checkpoints/vqvae_vfront.ckpt"}"',
        )
        staged = scripts_dir / filename
        staged.write_text(content)

        command = [sys.executable, str(staged)]
        if stage == "depth":
            command += ["--scene-root", str(scene_root), "--scene-id", scene_id]
        elif stage == "decode_render":
            command += [
                "--scene-root",
                str(scene_root),
                "--scene-id",
                scene_id,
                "--vqvae-checkpoint",
                str(PROJECT / "checkpoints" / "vqvae_vfront.ckpt"),
            ]
        if stage == "gaussian_fit":
            with np.load(scene_root / "gaussian_init" / "voxel_init_025m.npz", allow_pickle=False) as data:
                n_voxels = int(data["xyz"].shape[0])
            if n_voxels <= 0:
                raise RuntimeError(f"No input voxels in {scene_root}")
            if n_voxels < 30000:
                staged.write_text(staged.read_text().replace("N = 30000", f"N = {n_voxels}"))

        log_path = logs_dir / f"{stage}.log"
        print(f"[{scene_id}] starting {stage}", flush=True)
        start = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=work_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - start
        log_path.write_text(result.stdout)
        record = {
            "seconds": round(elapsed, 3),
            "exit_code": result.returncode,
            "log": str(log_path),
        }
        if stage == "gaussian_fit":
            record["input_voxels"] = n_voxels
            record["gaussians"] = min(n_voxels, 30000)
        timings["scenes"][scene_id][stage] = record
        timing_path.write_text(json.dumps(timings, indent=2))
        print(f"[{scene_id}] {stage}: {elapsed:.1f}s, exit={result.returncode}", flush=True)
        tail = "\n".join(result.stdout.splitlines()[-12:])
        if tail:
            print(tail, flush=True)
        if result.returncode:
            raise RuntimeError(f"{stage} failed; see {log_path}")

    print(f"PILOT PIPELINE COMPLETE: {scene_id}", flush=True)
    print(f"Outputs: {scene_root}", flush=True)
    print(f"Timings: {timing_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    run(args.scene_id, args.root)
