#!/usr/bin/env python3
"""Run selected scene preparation pipelines sequentially and record progress."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


STAGES = [
    "geometry",
    "depth",
    "rgb",
    "voxel",
    "gaussian_fit",
    "tokenize",
    "decode_render",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["development", "final_test", "adaptation_pool"],
        default="development",
    )
    parser.add_argument("--scene-ids", nargs="*")
    args = parser.parse_args()
    split_data = json.loads(args.splits.read_text())
    scene_ids = args.scene_ids or split_data["splits"][args.split]["scene_ids"]
    status_path = args.run_root / "stage_b" / "scene_pipeline_status.json"
    log_dir = args.run_root / "stage_b" / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status = {"run_root": str(args.run_root), "results": {}}
    if status_path.exists():
        status = json.loads(status_path.read_text())
        status.setdefault("results", {})

    timings_path = args.run_root / "timings.json"
    runner = args.project_root / "draft_a" / "pf_multiscene" / "run_pilot.py"
    for index, scene_id in enumerate(scene_ids, 1):
        timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
        previous = timings.get("scenes", {}).get(scene_id, {})
        if previous and all(
            previous.get(stage, {}).get("exit_code") == 0 for stage in STAGES
        ):
            status["results"][scene_id] = {
                "state": "complete",
                "source": "existing successful scene outputs",
                "updated_at": now(),
            }
            status_path.write_text(json.dumps(status, indent=2))
            print(f"[{index}/{len(scene_ids)}] {scene_id}: already complete", flush=True)
            continue

        status["running_scene"] = scene_id
        status["requested_scene_ids"] = list(scene_ids)
        status["started_at"] = status.get("started_at", now())
        status_path.write_text(json.dumps(status, indent=2))
        log_path = log_dir / f"{scene_id}.log"
        start = time.perf_counter()
        with log_path.open("w") as output:
            result = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "--scene-id",
                    scene_id,
                    "--root",
                    str(args.run_root),
                ],
                cwd=args.project_root,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.perf_counter() - start
        status["results"][scene_id] = {
            "state": "complete" if result.returncode == 0 else "failed",
            "exit_code": result.returncode,
            "seconds": round(elapsed, 3),
            "log": str(log_path),
            "updated_at": now(),
        }
        status["completed_count"] = sum(
            value.get("state") == "complete" for value in status["results"].values()
        )
        status["failed_count"] = sum(
            value.get("state") == "failed" for value in status["results"].values()
        )
        status_path.write_text(json.dumps(status, indent=2))
        print(
            f"[{index}/{len(scene_ids)}] {scene_id}: "
            f"{status['results'][scene_id]['state']} in {elapsed:.1f}s",
            flush=True,
        )

    status.pop("running_scene", None)
    status["finished_at"] = now()
    status_path.write_text(json.dumps(status, indent=2))
    print(json.dumps({k: status.get(k) for k in ["completed_count", "failed_count", "finished_at"]}, indent=2))


if __name__ == "__main__":
    main()
