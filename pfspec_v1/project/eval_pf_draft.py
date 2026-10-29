"""Paired development measurements of Joint and Ranker/Feature EAGLE."""
import argparse
import json
from pathlib import Path

import torch

from speculative.pf_draft import DraftConfig, PFDraft
from speculative.pf_engine import PositionRanker, decode, policy, target_cache
from train_pf_draft import load_core, save_json


def load_draft(core, path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    draft = PFDraft(core, DraftConfig(**saved["config"])).cuda().eval()
    draft.load_state_dict(saved["state_dict"], strict=True)
    return draft


def totals(rows):
    result = {}
    for name in sorted({r["variant"] for r in rows}):
        records = [r for r in rows if r["variant"] == name]
        generated = sum(r["generated_tokens"] for r in records)
        seconds = sum(r["seconds"] for r in records)
        item = dict(runs=len(records), generated_tokens=generated, seconds=seconds,
                    tokens_per_second=generated/seconds)
        for kind in ("position", "feature"):
            n = sum(r.get(kind+"_verified", 0) for r in records)
            accepted = sum(r.get(kind+"_accepted", 0) for r in records)
            item[kind+"_verified"] = n
            item[kind+"_accepted"] = accepted
            item[kind+"_acceptance"] = accepted/n if n else None
        for key in ("rounds", "target_calls", "verify_calls", "recovery_calls", "draft_calls",
                    "bonus_position", "bonus_feature", "reject_position", "reject_feature"):
            item[key] = sum(r.get(key, 0) for r in records)
        item["feature_by_depth"] = {}
        for depth in sorted({d for r in records for d in r.get("feature_by_depth", {})}):
            counts = {field: sum(r.get("feature_by_depth", {}).get(depth, {}).get(field, 0) for r in records)
                      for field in ("proposed", "verified", "accepted")}
            counts["acceptance"] = counts["accepted"]/counts["verified"] if counts["verified"] else None
            item["feature_by_depth"][depth] = counts
        item["reject_by_slot"] = {slot: sum(r.get("reject_by_slot", {}).get(slot, 0) for r in records)
                                  for slot in sorted({s for r in records for s in r.get("reject_by_slot", {})})}
        item["accepted_candidates_per_verify"] = sum(sum(r.get("accepted_lengths", [])) for r in records)/item["verify_calls"] if item["verify_calls"] else None
        item["progress_per_verification"] = generated/item["verify_calls"] if item["verify_calls"] else None
        result[name] = item
    if "target" in result:
        for item in result.values():
            item["speedup"] = item["tokens_per_second"]/result["target"]["tokens_per_second"]
    return result


@torch.inference_mode()
def verify_real(core, drafts, prompt):
    ids = torch.tensor([prompt], device="cuda")
    full = core(ids, kv_cache=target_cache(core, len(prompt)))
    cache = target_cache(core, len(prompt))
    split = min(17, len(prompt)-1)
    a = core(ids[:, :split], kv_cache=cache)
    b = core(ids[:, split:], kv_cache=cache)
    chunked = torch.cat((a, b), 1)
    tv = (policy(full)-policy(chunked)).abs().sum(-1)/2
    result = dict(max_target_block_tv=float(tv.max()), greedy_equal={})
    if result["max_target_block_tv"] > 0.001:
        raise ArithmeticError("FP32 Target block and full-sequence distributions differ")
    _, serial = decode(core, None, prompt, 24, mode="target", temperature=0)
    if "joint" in drafts:
        _, output = decode(core, drafts["joint"], prompt, 24, mode="joint", gamma=3, temperature=0)
        result["greedy_equal"]["joint"] = output == serial
        if output != serial:
            raise ArithmeticError("Joint greedy output differs from serial Target")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crops", type=Path, default=Path("/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_c/dev_crops.pt"))
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--prefixes", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--seeds", nargs="+", type=int, default=[34701])
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--draft-context", type=int, default=0)
    parser.add_argument("--old-pf", action="store_true")
    parser.add_argument("--feature-only", action="store_true")
    parser.add_argument("--feature-checkpoint", type=Path)
    parser.add_argument("--reference-feature", type=Path)
    parser.add_argument("--single-feature", type=Path)
    parser.add_argument("--gammas", nargs="+", type=int, choices=[1, 3, 4, 5])
    parser.add_argument("--reference-label", default="feature_v2")
    parser.add_argument("--profile-budget", type=int, default=64)
    parser.add_argument("--reuse-draft-cache", action="store_true")
    args = parser.parse_args()
    if args.budget < 2 or args.budget % 2:
        parser.error("use an even output budget of at least two")
    if args.profile_budget < 2 or args.profile_budget % 2:
        parser.error("profile budget must be even and at least two")
    wrapper, core = load_core()
    from model.gpt.flash_attention import HAS_FA3
    attention_backend = "Flash Attention 3" if HAS_FA3 else "PyTorch SDPA fallback"
    from joblib import load
    from bench_fast_hgb import FastHistRanker
    from draft_a.feature_target_v2 import RANKER
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    ranker_model = FastHistRanker(load(RANKER))
    ranks = core.sparse_position_rank_map.cpu().tolist()
    drafts = {mode: load_draft(core, args.feature_checkpoint if mode == "feature" and args.feature_checkpoint else args.training/mode/"selected.pt")
              for mode in (("feature",) if args.feature_only else ("feature", "joint"))}
    if args.reference_feature:
        drafts["feature_v2"] = load_draft(core, args.reference_feature)
    if args.single_feature:
        drafts["feature_single"] = load_draft(core, args.single_feature)
    records = torch.load(args.crops, map_location="cpu", weights_only=False)
    selected, seen = [], set()
    for row in records:
        if row["scene"] not in seen and len(row["tokens"]) >= max(args.prefixes):
            selected.append(row)
            seen.add(row["scene"])
        if len(selected) >= args.scenes:
            break
    if not selected:
        raise ValueError("no development scenes satisfy the requested prefixes")
    sample = [core.sparse_eos_token_id]+selected[0]["tokens"][:args.prefixes[0]].tolist()
    validation = verify_real(core, drafts, sample)
    _, serial = decode(core, None, sample, 24, mode="target", temperature=0)
    for gamma in args.gammas or [3]:
        _, feature_output = decode(core, drafts["feature"], sample, 24, mode="feature", gamma=gamma,
                                   ranker=PositionRanker(ranker_model, ranks), temperature=0)
        key = f"feature_g{gamma}" if args.gammas else "feature"
        validation["greedy_equal"][key] = serial == feature_output
        if serial != feature_output:
            raise ArithmeticError("Feature-only greedy output differs from serial Target")
    save_json(args.output/"validation.json", validation)
    variants = [("target", "target", 1)]+[(f"{mode}_g{g}", mode, g)
                for mode in (("feature",) if args.feature_only else ("feature", "joint")) for g in (args.gammas or [1, 3, 5])]
    if args.reuse_draft_cache:
        variants.append(("feature_v4_g3", "feature_v4", 3))
    if args.gammas is None:
        variants.append(("feature_g4", "feature", 4))
    if args.reference_feature:
        variants.extend((f"{args.reference_label}_g{g}", "feature_v2", g) for g in (args.gammas or [3, 5]))
    if args.single_feature:
        variants.append(("feature_single_g3", "feature_single", 3))
    old_adapter = None
    if args.old_pf:
        from bench_gaussiangpt_pf_block2_cow import load_feature_model
        from draft_a.feature_target_v2 import HEAD
        from draft_a.pf_feature_unified import MLPAdapter
        feature, _, _ = load_feature_model(Path("draft_a/runs/feature_target_continue_20261014/selected.pt"),
                                            HEAD, core, "cuda", 8000, 12096)
        old_adapter = MLPAdapter(feature)
        variants.append(("old_pf", "old_pf", 3))

    def run(mode, gamma, prompt, budget, seed, profile=False, trace=False):
        if mode == "old_pf":
            from draft_a.pf_feature_unified import decode as old_decode
            stats, output = old_decode(core, old_adapter, ranker_model, ranks, prompt, budget, seed, profile)
            stats["feature_by_depth"] = {"1": {k: stats.get("feature_"+k, 0) for k in ("proposed", "verified", "accepted")}}
            return stats, output
        engine_mode = "feature" if mode.startswith("feature") else mode
        return decode(core, drafts.get("feature" if mode == "feature_v4" else mode), prompt, budget,
                      mode=engine_mode, gamma=gamma,
                      ranker=PositionRanker(ranker_model, ranks), seed=seed,
                      draft_context=args.draft_context, profile=profile, trace=trace,
                      reuse_draft_cache=mode == "feature_v4")

    warmups = []
    for name, mode, gamma in variants:
        stats, _ = run(mode, gamma, sample, 16, 51699)
        warmups.append(dict(stats, variant=name, method=name,
                            checkpoint="checkpoints/gpt_vfront.ckpt",
                            batch_size=1, max_new_tokens=16, temperature=0.9, top_p=0.9,
                            gpu=torch.cuda.get_device_name(0), attention_backend=attention_backend,
                            run_id=f"{name}_warmup_s51699_p{args.prefixes[0]}",
                            profiled=False, warmup=True))
    save_json(args.output/"warmups.json", warmups)
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for scene_index, scene in enumerate(selected):
        for prefix_length in args.prefixes:
            prompt = [core.sparse_eos_token_id]+scene["tokens"][:prefix_length].tolist()
            for seed in args.seeds:
                offset = (scene_index+prefix_length+seed) % len(variants)
                order = variants[offset:]+variants[:offset]
                for name, mode, gamma in order:
                    stats, output = run(mode, gamma, prompt, args.budget, seed)
                    checkpoint = ("checkpoints/gpt_vfront.ckpt" if mode == "target" else
                                  "draft_a/runs/feature_target_continue_20261014/selected.pt" if mode == "old_pf" else
                                  str(args.feature_checkpoint) if args.feature_checkpoint else str(args.training/"feature"/"selected.pt"))
                    row = dict(stats, variant=name, method=name, checkpoint="checkpoints/gpt_vfront.ckpt",
                               draft_checkpoint=checkpoint if mode != "target" else None,
                               batch_size=1, max_new_tokens=args.budget, temperature=0.9, top_p=0.9,
                               gpu=torch.cuda.get_device_name(0), attention_backend=attention_backend,
                               run_id=f"{name}_s{seed}_p{prefix_length}", profiled=False,
                               scene=scene["scene"], prefix=prefix_length, output=output, seed=seed)
                    rows.append(row)
                    save_json(args.output/"records.json", rows)
                    print(json.dumps(dict(variant=name, scene=scene["scene"], prefix=prefix_length,
                                          seed=seed, generated_tokens=stats["generated_tokens"],
                                          seconds=stats["seconds"])), flush=True)
    save_json(args.output/"summary.json", dict(scope="development single-chunk token decoding",
              precision="FP32", temperature=0.9, top_p=0.9, sources=len(selected),
              budget=args.budget, sampling_seeds=args.seeds, draft_context=args.draft_context,
              target_checkpoint="checkpoints/gpt_vfront.ckpt", profile_budget=args.profile_budget,
              warmups_per_variant=1, batch_size=1, gpu=torch.cuda.get_device_name(0),
              attention_backend=attention_backend,
              feature_checkpoint=str(args.feature_checkpoint) if args.feature_checkpoint else None,
              reference_feature=str(args.reference_feature) if args.reference_feature else None,
              reference_label=args.reference_label, gammas=args.gammas,
              totals=totals(rows)))
    profiles = []
    profile_references = []
    for name, mode, gamma in variants:
        ordinary, output = run(mode, gamma, sample, args.profile_budget, 51697, trace=True)
        profile_references.append(dict(ordinary, variant=name, method=name,
                                       checkpoint="checkpoints/gpt_vfront.ckpt",
                                       batch_size=1, max_new_tokens=args.profile_budget,
                                       temperature=0.9, top_p=0.9,
                                       gpu=torch.cuda.get_device_name(0), attention_backend=attention_backend,
                                       run_id=f"{name}_profile_reference_s51697_p{args.prefixes[0]}",
                                       profiled=False, profile_reference=True, output_equal=True))
        profiled, again = run(mode, gamma, sample, args.profile_budget, 51697, profile=True, trace=True)
        if output != again:
            raise ArithmeticError("profile replay changes generated tokens")
        phases = profiled.get("timings", {})
        other = max(0.0, profiled["seconds"]-sum(phases.values())) if phases else None
        profiles.append(dict(profiled, variant=name, method=name, checkpoint="checkpoints/gpt_vfront.ckpt",
                             batch_size=1, max_new_tokens=args.profile_budget, temperature=0.9, top_p=0.9,
                             gpu=torch.cuda.get_device_name(0), attention_backend=attention_backend,
                             run_id=f"{name}_profile_s51697_p{args.prefixes[0]}", profiled=True,
                             other_seconds=other, output_equal=True))
    save_json(args.output/"profiles.json", profiles)
    save_json(args.output/"profile_references.json", profile_references)


if __name__ == "__main__":
    main()
