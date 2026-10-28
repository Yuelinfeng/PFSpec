"""Evaluate the high-acceptance trained D8 on the existing real VFront inputs."""

import argparse
from pathlib import Path
import traceback

from draft_a.feature_real_eval import run
from draft_a.feature_target_v2 import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("draft_a/runs/feature_d8_real_eval_20261014"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        run(args, model_paths={"step12000": "draft_a/runs/feature_target_continue_20261014/selected.pt"},
            gpt_path=Path("draft_a/runs/feature_acceptance_20261014/adapt_d8_acceptance/step_3600.pt"),
            joint=True)
    except Exception:
        if args.root.exists():
            save(args.root / "progress.json", dict(status="failed", error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
