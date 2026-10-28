"""Finish the single D4 experiment after its already-running training job."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from draft_a.d4_joint_cost import ROOT
from draft_a.feature_target_v2 import save


def run(args):
    while True:
        path = args.root / "training_progress.json"
        progress = json.loads(path.read_text()) if path.exists() else {}
        alive = Path(f"/proc/{args.train_pid}").exists()
        if not alive:
            if progress.get("status") != "complete" or progress.get("step") != 8600:
                raise RuntimeError("The training process ended before its agreed budget completed")
            break
        try:
            command = Path(f"/proc/{args.train_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except FileNotFoundError:
            continue
        if "draft_a.d4_joint_train" not in command:
            if progress.get("status") == "complete" and progress.get("step") == 8600:
                break
            raise RuntimeError("The referenced PID no longer belongs to D4 training")
        save(args.root / "pipeline.json", dict(status="running", phase="training", train_pid=args.train_pid,
                                              step=progress.get("step")))
        time.sleep(5)
    for name, root, smoke in [("evaluation_smoke", Path("draft_a/runs/d4_joint_smoke_20261014"), True),
                             ("evaluation", args.root, False)]:
        save(args.root / "pipeline.json", dict(status="running", phase=name))
        command = [sys.executable, "-u", "-m", "draft_a.d4_joint_eval", "--root", str(root)]
        if smoke:
            command.append("--smoke")
        with (args.root / f"{name}.log").open("a") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    save(args.root / "pipeline.json", dict(status="complete", phase="complete", training_steps=8600))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--train-pid", type=int, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        save(args.root / "pipeline.json", dict(status="failed", error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
