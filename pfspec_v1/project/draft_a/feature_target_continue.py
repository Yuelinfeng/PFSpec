"""Finite additional training of the existing direct-Target Feature model."""

import argparse
import json
import os
from pathlib import Path
import traceback

from draft_a.feature_target_v2 import train, online, save


START = Path("draft_a/runs/feature_target_full_v2_20261014")
ROOT = Path("draft_a/runs/feature_target_continue_20261014")


def run(args):
    if args.root.exists():
        raise FileExistsError("Use a new output directory; previous results are retained")
    state = json.loads((START/"pipeline.json").read_text())
    selected = json.loads((START/"selected.json").read_text())
    if state["status"] != "complete" or selected["step"] != 3600:
        raise ValueError("The completed step3600 model is required")
    data = json.loads((START/"data_plan.json").read_text())
    if data["status"] != "complete" or data["teacher"] != "checkpoints/gpt_vfront.ckpt":
        raise ValueError("Complete original-Target state data is required")
    if args.smoke:
        for split in ("train", "val"):
            rows = data["splits"][split]["records"][:2]
            data["splits"][split] = dict(scenes=len(rows), rows=sum(r["rows"] for r in rows), records=rows)
            data["files"][split] = [r["name"] for r in rows]
        data["smoke"] = True
    steps = 2 if args.smoke else args.steps
    interval = 1 if args.smoke else args.eval_every
    if steps <= 0 or interval <= 0:
        raise ValueError("Training steps and measurement interval must be positive")
    args.root.mkdir(parents=True)
    save(args.root/"data_plan.json", data)
    save(args.root/"continuation_plan.json", dict(start_checkpoint=str(START/"selected.pt"),
         start_step=3600, additional_steps=steps, final_step=3600+steps, evaluation_interval=interval,
         objective="same raw Target CE + 2 * policy TV", teacher="checkpoints/gpt_vfront.ckpt",
         lr=1e-5, final_lr=1e-6, optimizer="new AdamW; previous checkpoint contains model weights only",
         model_selection="full validation overlap only; initial step3600 retained if no improvement",
         online_reference="step3600, same run Target-only", smoke=args.smoke))
    try:
        save(args.root/"pipeline.json", dict(status="running", phase="train", pid=os.getpid()))
        train(args.root, chunk=256, initial_path=START/"selected.pt",
              phase_plan=[dict(name="policy_tv_continue", lr=1e-5, weight=2., seed=24689, steps=steps)],
              eval_every=interval, train_probe=16)
        save(args.root/"pipeline.json", dict(status="running", phase="online", pid=os.getpid()))
        online(args.root, tokens=32 if args.smoke else 512, limit=1 if args.smoke else None,
               reference_checkpoint=START/"selected.pt", reference_name="step3600", candidate_name="continued")
        save(args.root/"pipeline.json", dict(status="complete", phase="complete", pid=os.getpid()))
    except Exception:
        save(args.root/"pipeline.json", dict(status="failed", error=traceback.format_exc(), pid=os.getpid()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--steps", type=int, default=8400)
    parser.add_argument("--eval-every", type=int, default=1200)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
