#!/usr/bin/env python3
"""Continue the existing test-scene job through crops and FP32 evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from run_adaptation_pipeline import read_json, write_json


def scene_counts(scene_ids: list[str], results: dict) -> dict:
    return {state: sum(results.get(scene, {}).get("state") == state for scene in scene_ids)
            for state in ("complete", "failed")}


def preparation_running(pid: int) -> bool:
    try:
        return b"run_scene_batch.py" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scene-pid", type=int, required=True)
    args = parser.parse_args()
    root, project = args.run_root.resolve(), args.project_root.resolve()
    out = root / "stage_e_final"
    source = project / "draft_a/pf_multiscene"
    selection = read_json(out / "method_selection.json")
    split_path = root / "stage_b/scene_splits.json"
    scene_ids = read_json(split_path)["splits"]["final_test"]["scene_ids"]
    if len(scene_ids) != 30 or len(set(scene_ids)) != 30:
        raise ValueError("The selected final test set must contain 30 distinct scenes")
    if (selection["precision"] != "fp32" or selection["modes"] !=
            ["target_only", "p_only", "step12000", "d8_feature", "d8_joint"]):
        raise ValueError("This runner implements the selected five-mode FP32 comparison")
    logs = out / "pipeline_logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = out / "pipeline.json"
    status = dict(status="running", phase="waiting_scenes", pid=os.getpid(),
                  scene_pid=args.scene_pid, scene_ids=scene_ids, stages=[],
                  method_selection=str(out / "method_selection.json"))
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(str(project / item) for item in
                                (".", "draft_a", "draft_a/pf_multiscene", "profiling/target_eval"))
    started = time.monotonic()

    def run_stage(name: str, command: list[str], output: Path, *, inventory=False) -> dict:
        if output.exists():
            existing = read_json(output)
            complete = (inventory and (out / "final_test_crops.pt").exists()
                        and existing.get("unique_scenes") == len(scene_ids)
                        and existing.get("requested_scene_ids") == scene_ids)
            if complete or existing.get("status") == "complete":
                status["stages"].append(dict(name=name, status="reused", output=str(output)))
                return existing
            raise FileExistsError(f"Incomplete stage output is retained: {output}")
        log = logs / f"{name}.log"
        status.update(phase=name, log=str(log))
        with log.open("a") as handle:
            process = subprocess.Popen(command, cwd=project, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            status["stage_pid"] = process.pid
            write_json(status_path, status)
            code = process.wait()
        status["stages"].append(dict(name=name, exit_code=code, output=str(output), log=str(log)))
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(status_path, status)
        if code:
            raise RuntimeError(f"{name} exited with code {code}; see {log}")
        value = read_json(output)
        if not inventory and value.get("status") != "complete":
            raise RuntimeError(f"{name} did not finish: {output}")
        return value

    try:
        while True:
            preparation = read_json(root / "stage_b/scene_pipeline_status.json")
            counts = scene_counts(scene_ids, preparation["results"])
            status.update(preparation=counts, elapsed_seconds=time.monotonic() - started)
            write_json(status_path, status)
            if not preparation_running(args.scene_pid):
                if counts["complete"] != len(scene_ids):
                    status.update(status="needs_scene_review", phase="scene_preparation_incomplete",
                                  failed_scenes=[scene for scene in scene_ids
                                                 if preparation["results"].get(scene, {}).get("state") != "complete"])
                    write_json(status_path, status)
                    return
                break
            time.sleep(10)

        crops = out / "final_test_crops.pt"
        inventory = run_stage("crops", [sys.executable, str(source / "prepare_crop_records.py"),
                              "--run-root", str(root), "--splits", str(split_path),
                              "--split", "final_test", "--output", str(crops)],
                              out / "crop_inventory.json", inventory=True)
        status["crop_inventory"] = dict(scenes=inventory["unique_scenes"],
                                        crops=inventory["crop_records"],
                                        tokens=inventory["total_body_tokens"])
        shared = ["--records", str(crops), "--precision", "fp32",
                  "--feature-checkpoint", f"step12000={project / selection['primary_feature']}",
                  "--gpt", str(project / selection["d8_checkpoint"]), "--joint"]
        smoke = out / "fp32_smoke"
        full = out / "fp32_test_full"
        run_stage("smoke", [sys.executable, str(source / "feature_real_eval_multiscene.py"),
                            "--root", str(smoke), *shared, "--smoke", "--budget", "64"],
                  smoke / "online.json")
        run_stage("evaluation", [sys.executable, str(source / "feature_real_eval_multiscene.py"),
                                 "--root", str(full), *shared, "--budget", str(selection["budget"]),
                                 "--prefixes", *map(str, selection["prefixes"]),
                                 "--seeds", *map(str, selection["seeds"])], full / "online.json")
        run_stage("summary", [sys.executable, str(source / "summarize_online_results.py"),
                              "--eval-root", str(full), "--catalog", str(root / "stage_b/scene_catalog.csv"),
                              "--split-label", "final test"], full / "multiscene_summary.json")
        status.update(status="complete", phase="ready_for_full_3d_stage",
                      elapsed_seconds=time.monotonic() - started,
                      summary=str(full / "multiscene_summary.json"))
        status.pop("stage_pid", None)
        write_json(status_path, status)
    except Exception:
        status.update(status="failed", error=traceback.format_exc(),
                      elapsed_seconds=time.monotonic() - started)
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
