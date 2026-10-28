#!/usr/bin/env python3
"""Create reproducible spatial token crops for the development scenes."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from data.common import PreprocessedDataset
from draft_a.corpus_v2 import validate, write_json


def dataset_for(path: Path) -> PreprocessedDataset:
    return PreprocessedDataset(
        file_list=[str(path)],
        num_position_tokens=1,
        position_vocab_size=8000,
        codebook_size=4096,
        chunk_shape=[20, 20, 20],
        dense_chunks=False,
        chunk_order="xyz",
        min_chunk_occupancy=0.3,
        max_chunk_attempts=10,
        chunk_origin=[None, None, -1],
        load_augmented_tokens=False,
        shared=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "final_test", "adaptation_pool"),
                        default="development")
    parser.add_argument("--scene-ids", nargs="*")
    parser.add_argument("--target-crops", type=int, default=2)
    parser.add_argument("--max-seed", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    scene_ids = args.scene_ids or json.loads(args.splits.read_text())["splits"][args.split]["scene_ids"]
    records = []
    scene_results = []
    for scene_id in scene_ids:
        path = args.run_root / "scenes" / scene_id / "gaussian_init" / "vfront_pilot_tokens.pt"
        dataset = dataset_for(path)
        selected = []
        attempts = []
        seen = set()
        for seed in range(args.max_seed + 1):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            tokens = dataset[0].long().cpu()
            row = dict(tokens=tokens, has_eos=False, crop_seed=seed, path=str(path))
            validate(row)
            key = tuple(tokens.tolist())
            duplicate_of = next(
                (item["crop_seed"] for item in selected if tuple(item["tokens"].tolist()) == key),
                None,
            )
            attempts.append(
                dict(seed=seed, body_tokens=len(tokens), duplicate_of=duplicate_of)
            )
            if duplicate_of is not None:
                continue
            row["scene"] = scene_id
            selected.append(row)
            seen.add(key)
            if len(selected) >= args.target_crops:
                break
        records.extend(selected)
        scene_results.append(
            dict(
                scene_id=scene_id,
                token_file=str(path),
                attempts=attempts,
                selected_crop_seeds=[row["crop_seed"] for row in selected],
                selected_body_tokens=[len(row["tokens"]) for row in selected],
                available_prefixes={
                    str(prefix): sum(len(row["tokens"]) >= prefix for row in selected)
                    for prefix in (64, 256, 1024)
                },
                crop_count=len(selected),
            )
        )

    default_name = "dev_crops.pt" if args.split == "development" else f"{args.split}_crops.pt"
    output = args.output or args.run_root / "stage_c" / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    torch.save(records, temp)
    temp.replace(output)
    inventory = dict(
        source=f"{args.split} scene VQ-VAE token files",
        split=args.split,
        requested_scene_ids=scene_ids,
        seed_order=list(range(args.max_seed + 1)),
        target_crops_per_scene=args.target_crops,
        crop_records=len(records),
        unique_scenes=len({row["scene"] for row in records}),
        total_body_tokens=sum(len(row["tokens"]) for row in records),
        short_records=[
            dict(scene=row["scene"], crop_seed=row["crop_seed"], body_tokens=len(row["tokens"]))
            for row in records
            if len(row["tokens"]) < 1024
        ],
        scenes=scene_results,
        output=str(output),
        elapsed_seconds=round(time.perf_counter() - started, 2),
    )
    write_json(output.with_name("crop_inventory.json"), inventory)
    print(json.dumps(inventory, indent=2), flush=True)


if __name__ == "__main__":
    main()
