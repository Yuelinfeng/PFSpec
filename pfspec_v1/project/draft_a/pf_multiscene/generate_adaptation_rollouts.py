#!/usr/bin/env python3
"""Sample bounded Target continuations from held-out real-scene prefixes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from draft_a.corpus_v2 import EOS, load_target, validate
from draft_a.eval_overlap import sampling_probs
from draft_a.learning_eval import _fp32_runtime
from draft_a.pf_ranker_eval import validate_sequence
from model.gpt.engine import KVCache


def typed_kinds(start: int, count: int) -> list[str]:
    return ["position" if (start + index) % 2 == 0 else "feature"
            for index in range(count)]


def sample_token(probabilities: torch.Tensor, generator: torch.Generator) -> int:
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def generate_one(target, record: dict, prefix_tokens: int, seed_base: int,
                 max_new_tokens: int, output_path: Path) -> dict:
    core = target.transformer
    tokens = record["tokens"].long().cpu()
    if len(tokens) < prefix_tokens:
        raise ValueError(f"Crop is shorter than requested prefix: {record['scene']}/{len(tokens)}")
    prompt = [EOS, *tokens[:prefix_tokens].tolist()]
    sequence = list(prompt)
    ranks = core.sparse_position_rank_map.detach().cpu().tolist()
    config = core.config
    cache = KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=len(prompt) + max_new_tokens + 2,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
        device="cuda",
        dtype=torch.float32,
    )
    if len(prompt) > 1:
        prefill = prompt[:-1]
        core(
            torch.tensor([prefill], device="cuda"),
            kv_cache=cache,
            logit_types=typed_kinds(0, len(prefill)),
        )
    generator_seed = (
        seed_base + record["crop_seed"] * 1009 + prefix_tokens * 17
    ) % (2**63 - 1)
    generator = torch.Generator(device="cuda").manual_seed(generator_seed)
    generated = 0
    has_eos = False
    while generated < max_new_tokens:
        start = cache.get_pos()
        kind = "position" if start % 2 == 0 else "feature"
        logits = core(
            torch.tensor([[sequence[-1]]], device="cuda"),
            kv_cache=cache,
            logit_types=[kind],
        )[0, -1]
        previous_position = sequence[-2] if kind == "position" and len(sequence) > 1 else None
        probabilities = sampling_probs(logits, previous_position, 0.9, 0.9)
        token = sample_token(probabilities, generator)
        generated += 1
        if token == EOS:
            has_eos = True
            break
        sequence.append(token)

    body = torch.tensor(sequence[1:], dtype=torch.long)
    result = dict(
        name=(f"{record['scene']}_crop_{record['crop_seed']}"
              f"_prefix_{prefix_tokens}_seed_{seed_base}"),
        split="adaptation_pool",
        scene=record["scene"],
        crop_seed=record["crop_seed"],
        real_prefix_tokens=prefix_tokens,
        seed_base=seed_base,
        rollout_seed=generator_seed,
        max_new_tokens=max_new_tokens,
        generated_tokens=generated,
        temperature=0.9,
        top_p=0.9,
        has_eos=has_eos,
        tokens=body,
        source_crop=record["path"],
        output_path=str(output_path),
        teacher="checkpoints/gpt_vfront.ckpt",
    )
    validate(result)
    validate_sequence([EOS], body.tolist() + ([EOS] if has_eos else []), ranks)
    if not torch.equal(body[:prefix_tokens], tokens[:prefix_tokens]):
        raise ValueError("Real prefix changed during Target continuation")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefixes", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[34701, 34702, 34703, 34704])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.max_new_tokens % 2:
        raise ValueError("The continuation budget must be a positive even number")
    if args.out.exists() and not args.out.is_dir():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    crops = torch.load(args.crops, map_location="cpu", weights_only=True)
    jobs = [
        (record, prefix, seed)
        for record in crops
        for prefix in args.prefixes
        if len(record["tokens"]) >= prefix
        for seed in args.seeds
    ]
    plan_path = args.out / "rollout_plan.json"
    plan = dict(
        status="running",
        phase="target_sampling",
        teacher="checkpoints/gpt_vfront.ckpt",
        crops=str(args.crops),
        crop_count=len(crops),
        unique_scenes=len({record["scene"] for record in crops}),
        prefixes=args.prefixes,
        seeds=args.seeds,
        max_new_tokens=args.max_new_tokens,
        temperature=0.9,
        top_p=0.9,
        label_temperature=1.0,
        label_top_p=None,
        planned_trajectories=len(jobs),
        completed_trajectories=0,
        trajectories=[],
    )
    plan_path.write_text(json.dumps(plan, indent=2))
    progress_path = args.out / "rollout_progress.json"
    started = time.monotonic()
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("highest")
    with torch.inference_mode(), _fp32_runtime(torch):
        target = load_target().float().eval().requires_grad_(False)
        for index, (record, prefix, seed_base) in enumerate(jobs, 1):
            name = (f"{record['scene']}_crop_{record['crop_seed']}"
                    f"_prefix_{prefix}_seed_{seed_base}.pt")
            path = args.out / name
            if path.exists():
                value = torch.load(path, map_location="cpu", weights_only=True)
                if (value["scene"] != record["scene"]
                        or value["crop_seed"] != record["crop_seed"]
                        or value["real_prefix_tokens"] != prefix
                        or value["seed_base"] != seed_base
                        or value["max_new_tokens"] != args.max_new_tokens
                        or not torch.equal(value["tokens"][:prefix], record["tokens"][:prefix])):
                    raise ValueError(f"Existing rollout differs from this plan: {path}")
            else:
                value = generate_one(target, record, prefix, seed_base,
                                     args.max_new_tokens, path)
                torch.save(value, path)
            plan["trajectories"].append(dict(
                name=value["name"],
                scene=value["scene"],
                crop_seed=value["crop_seed"],
                real_prefix_tokens=value["real_prefix_tokens"],
                seed_base=value["seed_base"],
                generated_tokens=value["generated_tokens"],
                has_eos=value["has_eos"],
                path=str(path),
            ))
            plan["completed_trajectories"] = index
            plan["elapsed_seconds"] = time.monotonic() - started
            progress_path.write_text(json.dumps({
                "status": "running",
                "completed": index,
                "expected": len(jobs),
                "last": name,
                "elapsed_seconds": round(plan["elapsed_seconds"], 2),
            }, indent=2))
            if index % 10 == 0 or index == len(jobs):
                plan_path.write_text(json.dumps(plan, indent=2))
                print(json.dumps({
                    "completed": index,
                    "expected": len(jobs),
                    "scene": value["scene"],
                    "prefix": prefix,
                    "seed": seed_base,
                    "generated_tokens": value["generated_tokens"],
                    "eos": value["has_eos"],
                }), flush=True)
            del value
    plan.update(status="complete", elapsed_seconds=time.monotonic() - started)
    plan_path.write_text(json.dumps(plan, indent=2))
    progress_path.write_text(json.dumps({
        "status": "complete",
        "completed": len(jobs),
        "expected": len(jobs),
        "elapsed_seconds": round(plan["elapsed_seconds"], 2),
    }, indent=2))
    print(json.dumps({
        "status": plan["status"],
        "completed_trajectories": plan["completed_trajectories"],
        "elapsed_seconds": round(plan["elapsed_seconds"], 2),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
