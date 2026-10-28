#!/usr/bin/env python3
"""Run three sequential LFQ training comparisons in the current foreground task."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if min(args.epochs, args.samples_per_epoch, args.batch_size, args.patience) < 1:
        parser.error("Training sizes must be positive")
    if args.out_dir.exists():
        parser.error("Use a new comparison output directory")
    script = Path(__file__).resolve().with_name("train_lfq_heads.py")
    if not script.is_file():
        parser.error("Missing training entry point next to this file")
    args.out_dir.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    if args.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    results = {}
    variants = (("original_rows", "rows", 0, 0), ("heads_rows", "rows", 0.5, 0.5),
                ("original_trajectories", "trajectories", 0, 0))
    for name, sampling, low, conditional in variants:
        print("START", name, flush=True)
        command = [sys.executable, "-u", str(script), "--device", args.device, "--threads", "1",
                   "--epochs", str(args.epochs), "--samples-per-epoch", str(args.samples_per_epoch),
                   "--batch-size", str(args.batch_size), "--patience", str(args.patience),
                   "--sampling", sampling, "--low-overlap-weight", str(low),
                   "--conditional-overlap-weight", str(conditional), "--out-dir", str(args.out_dir/name)]
        with (args.out_dir/(name+".log")).open("x") as log:
            completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"{name} failed with exit code {completed.returncode}; see its log")
        saved = json.loads((args.out_dir/name/"summary.json").read_text())
        parent = sum(item["sequence_overlap"] for item in saved["parent_validation"]) / len(saved["parent_validation"])
        epoch = saved["best_epoch"]
        selected = saved["parent_validation"] if epoch == 0 else next(
            item["validation"] for item in saved["history"] if item["epoch"] == epoch)
        results[name] = {"best_epoch": epoch, "epochs_completed": len(saved["history"]),
                         "parent_sequence_overlap": parent,
                         "best_sequence_overlap": saved["best_sequence_overlap"],
                         "delta_sequence_pp": 100*(saved["best_sequence_overlap"]-parent),
                         "selected_validation": selected, "sampling_notes": saved["sampling_notes"],
                         "last_validation": saved["history"][-1]["validation"],
                         "training_wall_seconds": sum(item["wall_seconds"] for item in saved["history"])}
        result = {"device": args.device, "epochs_requested": args.epochs,
                  "samples_per_epoch": args.samples_per_epoch, "batch_size": args.batch_size,
                  "results": results, "test_read": False,
                  "limitations": "Single training seed; validation used for model selection; no online speed measurement."}
        temporary = args.out_dir/"comparison.json.tmp"
        temporary.write_text(json.dumps(result, indent=2)+"\n")
        os.replace(temporary, args.out_dir/"comparison.json")
        print("COMPLETE", name, json.dumps(results[name]), flush=True)


if __name__ == "__main__":
    main()
