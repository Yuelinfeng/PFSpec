#!/usr/bin/env python3
"""Continue existing rollouts through labels, equal-budget training, and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def original_rollout_running(pid: int) -> bool:
    path = Path(f"/proc/{pid}/cmdline")
    try:
        return b"generate_adaptation_rollouts.py" in path.read_bytes()
    except FileNotFoundError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--rollout-pid", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42001)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.project_root = args.project_root.resolve()
    source = args.project_root / "draft_a/pf_multiscene"
    pool = args.run_root / "stage_d2_adaptation/stage_c"
    rollouts = pool / "rollouts"
    labels = pool / "labels"
    training = pool / f"training_pair_seed{args.seed}"
    evaluation = pool / f"dev_pair_seed{args.seed}"
    smoke = pool / f"dev_smoke_seed{args.seed}"
    status_path = pool / f"pipeline_seed{args.seed}.json"
    logs = pool / "pipeline_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(str(args.project_root / relative) for relative in
                                 (".", "draft_a", "draft_a/pf_multiscene", "profiling/target_eval"))
    python = sys.executable
    initial = args.project_root / "draft_a/runs/feature_target_continue_20261014/selected.pt"
    checkpoints = {
        "step12000": initial,
        "adapt_A": training / "A_old_only/selected.pt",
        "adapt_B": training / "B_real_mixed/selected.pt",
    }
    shared_eval = ["--records", str(args.run_root / "stage_c/dev_crops.pt"),
                   "--precision", "fp32"]
    for name, path in checkpoints.items():
        shared_eval.extend(["--feature-checkpoint", f"{name}={path}"])
    commands = [
        ("labels", [python, str(source / "prepare_adaptation_labels.py"),
                    "--rollouts", str(rollouts), "--development-crops", str(args.run_root / "stage_c/dev_crops.pt"),
                    "--out", str(labels)], labels / "label_plan.json"),
        ("training", [python, str(source / "train_real_adaptation_pair.py"),
                      "--old-data-plan", str(args.project_root / "draft_a/runs/feature_target_full_v2_20261014/data_plan.json"),
                      "--label-plan", str(labels / "label_plan.json"), "--initial-checkpoint", str(initial),
                      "--out-root", str(training), "--seed", str(args.seed),
                      "--resume-complete-groups"], training / "pair_summary.json"),
        ("smoke", [python, str(source / "feature_real_eval_multiscene.py"),
                   "--root", str(smoke), *shared_eval, "--smoke", "--budget", "64"], smoke / "online.json"),
        ("development", [python, str(source / "feature_real_eval_multiscene.py"),
                         "--root", str(evaluation), *shared_eval, "--budget", "512",
                         "--prefixes", "64", "256", "1024", "--seeds", "34701", "34702", "34703"], evaluation / "online.json"),
        ("summary", [python, str(source / "summarize_online_results.py"), "--eval-root", str(evaluation),
                     "--catalog", str(args.run_root / "stage_b/scene_catalog.csv")], evaluation / "multiscene_summary.json"),
        ("comparison", [python, str(source / "compare_adaptation_groups.py"), "--eval-root", str(evaluation),
                        "--catalog", str(args.run_root / "stage_b/scene_catalog.csv")], evaluation / "adaptation_comparison.json"),
    ]
    status = dict(status="running", phase="waiting_rollouts", pid=os.getpid(),
                  rollout_pid=args.rollout_pid, seed=args.seed,
                  started_at=datetime.now(timezone.utc).isoformat(),
                  project_root=str(args.project_root), run_root=str(args.run_root),
                  stages=[], commands={name: command for name, command, _ in commands})
    started = time.monotonic()
    try:
        while True:
            progress = read_json(rollouts / "rollout_progress.json")
            status["rollouts"] = progress
            status["elapsed_seconds"] = time.monotonic() - started
            write_json(status_path, status)
            if progress["status"] == "complete":
                plan = read_json(rollouts / "rollout_plan.json")
                if plan["status"] != "complete" or plan["completed_trajectories"] != plan["planned_trajectories"]:
                    raise ValueError("Rollout plan and progress disagree at completion")
                if not original_rollout_running(args.rollout_pid):
                    break
            elif not original_rollout_running(args.rollout_pid):
                raise RuntimeError("The rollout process exited before completing its plan")
            time.sleep(10)

        for name, command, completed_path in commands:
            if completed_path.exists() and read_json(completed_path).get("status") == "complete":
                status["stages"].append(dict(name=name, status="reused", output=str(completed_path)))
                write_json(status_path, status)
                continue
            log = logs / f"{args.seed}_{name}.log"
            status.update(phase=name, stage_log=str(log))
            stage_started = time.monotonic()
            with log.open("a") as handle:
                handle.write(f"Started {datetime.now(timezone.utc).isoformat()}\n")
                handle.flush()
                process = subprocess.Popen(command, cwd=args.project_root, env=env,
                                           stdout=handle, stderr=subprocess.STDOUT)
                status["stage_pid"] = process.pid
                write_json(status_path, status)
                code = process.wait()
            status["stages"].append(dict(name=name, exit_code=code,
                                          elapsed_seconds=time.monotonic() - stage_started,
                                          log=str(log), output=str(completed_path)))
            status["elapsed_seconds"] = time.monotonic() - started
            write_json(status_path, status)
            if code:
                raise RuntimeError(f"{name} exited with code {code}; see {log}")
            if not completed_path.exists() or read_json(completed_path).get("status") != "complete":
                raise RuntimeError(f"{name} did not record complete output: {completed_path}")
            print(json.dumps(dict(stage=name, status="complete", output=str(completed_path))), flush=True)
        status.update(status="complete", phase="ready_for_development_analysis",
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic() - started,
                      comparison=str(evaluation / "adaptation_comparison.json"))
        status.pop("stage_pid", None)
        write_json(status_path, status)
        print(json.dumps(status, indent=2), flush=True)
    except Exception:
        status.update(status="failed", error=traceback.format_exc(),
                      elapsed_seconds=time.monotonic() - started)
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
