"""Measure direct-Target Feature drafts using reconstructed 3D-FRONT inputs."""

import argparse
import json
from pathlib import Path
import time

from draft_a.feature_target_v2 import HEAD, RANKER, evaluate, save


ROOT = Path("draft_a/runs/feature_real_eval_20261014")
CROPS = Path("draft_a/runs/rollout500_20261013/real_crops.pt")
MODELS = {
    "step3600": "draft_a/runs/feature_target_full_v2_20261014/selected.pt",
    "step12000": "draft_a/runs/feature_target_continue_20261014/selected.pt",
}
PREFIXES = [64, 256, 1024]
SEEDS = [34701, 34702, 34703]
MODE_LABELS = {
    "target_only": "T", "p_only": "P", "step12000": "PF",
    "step3600": "PF step3600", "d8_feature": "PF-D8",
    "d8_joint": "GPT-D8", "adapt_A": "PF A", "adapt_B": "PF B",
}


def aggregate(rows):
    keys = ["generated_tokens", "seconds", "decoder_seconds", "target_calls",
            "position_proposed", "position_verified", "position_accepted",
            "feature_proposed", "feature_verified", "feature_accepted", "rounds"]
    total = {k: sum(r.get(k, 0) for r in rows) for k in keys}
    total["runs"] = len(rows)
    total["tps"] = total["generated_tokens"] / total["seconds"]
    for kind in ("position", "feature"):
        for denom in ("verified", "proposed"):
            count = total[kind + "_" + denom]
            total[kind + "_acceptance_" + denom] = total[kind + "_accepted"] / count if count else None
    return total


def summaries(rows, variants=None):
    variants = variants or ["target_only", *MODELS]
    out = {v: aggregate([r for r in rows if r["variant"] == v])
           for v in variants}
    for name in variants[1:]:
        out[name]["speedup"] = out[name]["tps"] / out["target_only"]["tps"]
    if "step3600" in out and "step12000" in out:
        out["new_over_old_speed"] = out["step12000"]["tps"] / out["step3600"]["tps"]
    elif "step12000" in out:
        out["relative_to_mlp"] = {name: out[name]["tps"]/out["step12000"]["tps"]
                                  for name in variants[1:] if name != "step12000"}
    return out


def run(args, model_paths=None, gpt_path=None, joint=False):
    import torch
    from importlib import import_module
    from joblib import load
    from threadpoolctl import threadpool_limits
    from model.gpt.engine import KVCache
    from bench_fast_hgb import FastHistRanker
    from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock, load_feature_model
    from draft_a.analyze_v2 import real_records
    from draft_a.corpus_v2 import EOS, load_target, validate
    from draft_a.learning_eval import _fp32_runtime
    from draft_a.pf_feature_unified import MLPAdapter, GPTAdapter, decode
    from draft_a.pf_ranker_eval import decode as ordinary, validate_sequence
    from draft_a.feature_adapt_continue import policy_probs

    is_bf16 = args.precision == "bf16"
    if is_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support BF16")
    acceptance_eval = import_module("draft_a.feature_acceptance_eval")
    ranker_eval = import_module("draft_a.pf_ranker_eval")
    unified_eval = import_module("draft_a.pf_feature_unified")
    original_cache_for = acceptance_eval.cache_for
    original_require = acceptance_eval._require

    def precision_cache_for(core, length):
        if not is_bf16:
            return original_cache_for(core, length)
        cfg = core.config
        return KVCache(batch_size=1, num_heads=cfg.n_kv_head, seq_len=length,
                       head_dim=cfg.n_embd // cfg.n_head, num_layers=cfg.n_layer,
                       device="cuda", dtype=torch.bfloat16)

    if is_bf16:
        acceptance_eval.cache_for = precision_cache_for
        ranker_eval.cache_for = precision_cache_for

        def precision_require(condition, message):
            if message == "Online logits must be FP32":
                return
            return original_require(condition, message)

        acceptance_eval._require = precision_require

    def call_precision(fn, *fn_args, **fn_kwargs):
        if is_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return fn(*fn_args, **fn_kwargs)
        return fn(*fn_args, **fn_kwargs)

    def feature_decode(target_core, adapter, ranker, ranks, prompt, budget, seed, profile=False):
        if is_bf16:
            return call_precision(unified_eval._decode, target_core, adapter, ranker,
                                  ranks, prompt, budget, seed, profile)
        return decode(target_core, adapter, ranker, ranks, prompt, budget, seed, profile)

    def verify_precision_blocks(core, seq):
        if not is_bf16:
            from draft_a.feature_acceptance_eval import verify_blocks
            return verify_blocks(core, seq)
        seq = seq[:129].reshape(1, -1).cuda()
        if seq.shape[1] == 0:
            raise ValueError("Empty block-validation sequence")
        step_cache = precision_cache_for(core, seq.shape[1] + 17)
        refs = []
        for t in range(seq.shape[1]):
            row = call_precision(core, seq[:, t:t + 1], kv_cache=step_cache)[0, -1]
            refs.append(acceptance_eval.policy(row).cpu())
        maximum = 0.0
        width_maximum = {}
        failures = []
        for width in (2, 4, 8, 16):
            width_maximum[width] = 0.0
            cached = precision_cache_for(core, seq.shape[1] + 17)
            for start in range(0, seq.shape[1], width):
                logits = call_precision(core, seq[:, start:start + width], kv_cache=cached)[0]
                for j, row in enumerate(logits):
                    tv = float((acceptance_eval.policy(row).cpu() - refs[start + j]).abs().sum() / 2)
                    if tv >= 0.01:
                        failures.append(dict(width=width, token=start + j, total_variation=tv))
                    width_maximum[width] = max(width_maximum[width], tv)
                    maximum = max(maximum, tv)
            if seq.shape[1] > 15:
                cached.cache_seqlens.fill_(7)
                logits = call_precision(core, seq[:, 7:15], kv_cache=cached)[0]
                for j, row in enumerate(logits):
                    tv = float((acceptance_eval.policy(row).cpu() - refs[7 + j]).abs().sum() / 2)
                    if tv >= 0.01:
                        failures.append(dict(width=width, token=7 + j, total_variation=tv,
                                             rewind=True))
                    width_maximum[width] = max(width_maximum[width], tv)
                    maximum = max(maximum, tv)
        return dict(max_probability_tv=maximum, max_probability_tv_by_width=width_maximum,
                    tolerance=0.01, failures=failures,
                    cache_dtype="bfloat16", widths=[2, 4, 8, 16])

    model_paths = model_paths or MODELS
    declared_models = dict(model_paths)
    if gpt_path is not None:
        declared_models["d8_feature"] = str(gpt_path)
    if joint:
        if gpt_path is None:
            raise ValueError("Joint GPT decoding requires GPT weights")
        declared_models["d8_joint"] = str(gpt_path)

    if args.root.exists():
        raise FileExistsError("Use a new result directory; previous results are retained")
    if args.records is None:
        cached = torch.load(CROPS, map_location="cpu", weights_only=True)
        regenerated = real_records()
        if len(cached) != len(regenerated) or any(
                not torch.equal(a["tokens"], b["tokens"]) or a["path"] != b["path"]
                for a, b in zip(cached, regenerated)):
            raise ValueError("Regenerated real crops differ from recorded inputs")
    else:
        regenerated = torch.load(args.records, map_location="cpu", weights_only=True)
    records, duplicates = [], []
    for row in regenerated:
        validate(row)
        scene = row.get("scene") or Path(row["path"]).parents[1].name
        if any(old["scene"] == scene and torch.equal(row["tokens"], old["tokens"])
               for old in records):
            duplicates.append(dict(scene=scene, crop_seed=row["crop_seed"]))
            continue
        row["name"] = f"{scene}_crop_{row['crop_seed']}"
        row["scene"] = scene
        row["seq"] = torch.cat([torch.tensor([EOS]), row["tokens"]])
        records.append(row)
    if args.smoke:
        records = records[:1]
    prefixes = [64] if args.smoke else args.prefixes
    seeds = args.seeds[:1] if args.smoke else args.seeds
    budget = args.budget if args.budget is not None else (128 if args.smoke else 512)
    prefix_records = {
        record["name"]: [p for p in prefixes if len(record["tokens"]) >= p]
        for record in records
    }
    args.root.mkdir(parents=True)
    metadata = dict(source="3D-FRONT mesh -> multiview Gaussian optimization -> original VQ-VAE -> real tokens",
                    cached_crops=str(args.records or CROPS), unique_scenes=len({r["scene"] for r in records}),
                    unique_crops=len(records), skipped_duplicate_crop_seeds=duplicates,
                    original_target="checkpoints/gpt_vfront.ckpt", vqvae="checkpoints/vqvae_vfront.ckpt",
                    ranker=str(RANKER), models=declared_models, prefixes=prefixes, seeds=seeds,
                    budget=budget,
                    precision=("BF16 autocast with BF16 KV" if is_bf16 else "FP32"),
                    temperature=.9, top_p=.9, k=2,
                    proposal_widths={"target_only": 0, "p_only": 1,
                                     **{name: 3 for name in model_paths},
                                     **({"d8_feature": 3, "d8_joint": 2} if joint else {})},
                    mode_labels={name: MODE_LABELS.get(name, name) for name in
                                 ["target_only", "p_only", *model_paths,
                                  *(["d8_feature"] if gpt_path is not None else []),
                                  *(["d8_joint"] if joint else [])]},
                    joint_gamma=2 if joint else None,
                    scope="continuation from real prefixes; generated suffixes are model samples",
                    timing="common outer synchronized wall time: validation, cache allocation/release, prefill, proposal, verification, rejection handling, sampling; excludes loading and data preparation",
                    uncertainty=f"{len({r['scene'] for r in records})} scenes; crops are within-scene variants and do not replace scene-level comparisons",
                    crops=[dict(name=r["name"], scene=r["scene"], crop_seed=r["crop_seed"],
                                source=r["path"], body_tokens=len(r["tokens"])) for r in records])
    save(args.root / "plan.json", metadata)
    save(args.root / "progress.json", dict(status="running", phase="fixed_prefix"))
    with torch.inference_mode(), _fp32_runtime(torch), threadpool_limits(limits=1, user_api="openmp"):
        target = load_target().float().eval().requires_grad_(False)
        ranker = FastHistRanker(load(RANKER))
        ranks = target.transformer.sparse_position_rank_map.cpu().tolist()
        models = {name: load_feature_model(Path(path), HEAD, target.transformer,
                                         torch.device("cuda"), 8000, 12096)[0]
                  for name, path in model_paths.items()}
        gpt = None
        if gpt_path is not None:
            from draft_a.feature_reuse import make_model
            state = torch.load(gpt_path, map_location="cpu", weights_only=True)
            size = state["config"]["gpt_size"]
            if (size["n_layer"], size["n_embd"], size["n_head"]) != (8, 512, 8):
                raise ValueError("The requested 8-layer 512-width D8 is required")
            gpt = make_model(state["config"]).float().eval().requires_grad_(False)
            gpt.load_state_dict(state["model"], strict=True)
            if not torch.equal(gpt.transformer.sparse_position_rank_map,
                               target.transformer.sparse_position_rank_map):
                raise ValueError("GPT and Target Position ordering differs")
            metadata["d8_weights"] = dict(step=state["step"], parameters=sum(p.numel() for p in gpt.parameters()),
                                         size=size, checkpoint=str(gpt_path))
            save(args.root / "plan.json", metadata)
            del state
        for record in records:
            validate_sequence(record["seq"].tolist(), [], ranks)
        fixed = []
        capture = CaptureLastBlock(target.transformer)
        try:
            for record in records:
                seq = record["seq"]
                depth = torch.arange(1, len(record["tokens"]) // 2)
                indices = (2 * depth + 1).cuda()
                logits = call_precision(target, input_ids=seq[:-1].unsqueeze(0).cuda())[0]
                states = capture.output[0]
                data = dict(name=record["name"], depth=depth, position=seq[2*depth+1],
                            sampled_feature=seq[2*depth+2]-8000, previous_feature=seq[2*depth]-8000,
                            previous_state=states.index_select(0, indices-2).float().cpu(),
                            target=logits.index_select(0, indices)[:, 8000:12096].float().softmax(-1).cpu())
                capture.output = None
                del logits, states, indices
                values = {name: call_precision(evaluate, model, [data], 256)
                          for name, model in models.items()}
                if gpt is not None:
                    logits = call_precision(gpt, input_ids=seq[:-1].unsqueeze(0).cuda())[0]
                    selected = logits.index_select(0, (2*depth+1).cuda())[:,8000:12096].float()
                    q = policy_probs(selected)
                    p = policy_probs(data["target"].cuda().log())
                    values["d8_feature"] = dict(pooled_overlap=float(torch.minimum(p,q).sum(-1).mean()))
                    del logits, selected, p, q
                fixed.append(dict(name=record["name"], rows=len(depth), **values))
                del data
        finally:
            capture.close()
        slots = sum(r["rows"] for r in fixed)
        fixed_summary = dict(rows=slots, crops=fixed, unique_scenes=metadata["unique_scenes"],
                             scope="identical complete real prefixes; expected one-step F acceptance")
        fixed_names = [*models, *(["d8_feature"] if gpt is not None else [])]
        for name in fixed_names:
            fixed_summary[name] = sum(r[name]["pooled_overlap"]*r["rows"] for r in fixed)/slots
        save(args.root / "fixed_real_overlap.json", fixed_summary)
        adapters = {name: MLPAdapter(model, bf16=is_bf16)
                    for name, model in models.items()}
        if gpt is not None:
            if is_bf16:
                class BF16GPTAdapter(GPTAdapter):
                    def _forward(self, tokens):
                        self.forward_calls += 1
                        self.input_tokens += len(tokens)
                        return call_precision(
                            self.core,
                            torch.tensor([tokens], device="cuda"),
                            kv_cache=self.cache,
                            logit_types=["feature"] * len(tokens),
                        )[0]

                adapters["d8_feature"] = BF16GPTAdapter(gpt.transformer)
            else:
                adapters["d8_feature"] = GPTAdapter(gpt.transformer)
        variants_all = ["target_only", "p_only", *adapters, *(["d8_joint"] if joint else [])]
        validation = []
        if joint:
            from draft_a.feature_acceptance_eval import decode as joint_decode
            numerical = {name: verify_precision_blocks(core, records[0]["seq"])
                         for name, core in [("target",target.transformer),("d8",gpt.transformer)]}
            save(args.root / "numerical_validation.json", numerical)
        for prefix in prefixes:
            source_record = next((r for r in records if len(r["tokens"]) >= prefix), None)
            if source_record is None:
                continue
            prompt = source_record["seq"][:prefix+1].tolist()
            for name, adapter in adapters.items():
                row, seq = feature_decode(target.transformer, adapter, ranker, ranks,
                                          prompt, 32, 34700)
                timed, seq2 = feature_decode(target.transformer, adapter, ranker, ranks,
                                             prompt, 32, 34700, profile=True)
                if seq != seq2 or any(row[k] != timed[k] for k in ["accepted", "verified", "proposed", "rounds"]):
                    raise ValueError("Stage timing changed output or proposal counts")
                validation.append(dict(prefix=prefix, variant=name, status="passed"))
            call_precision(ordinary, target.transformer, None, ranker, ranks,
                           prompt, 32, 0, 34700)
            call_precision(ordinary, target.transformer, None, ranker, ranks,
                           prompt, 32, 1, 34700)
            validation.append(dict(prefix=prefix, variant="p_only", status="passed"))
            if joint:
                same = call_precision(joint_decode, target.transformer, target.transformer,
                                      prompt, 32, 2, 34700)
                if same["accepted"] != same["verified"]:
                    raise ValueError("Target proposing to itself must accept every verified token")
                a = call_precision(joint_decode, target.transformer, gpt.transformer,
                                   prompt, 32, 2, 34700)
                b = call_precision(joint_decode, target.transformer, gpt.transformer,
                                   prompt, 32, 2, 34700)
                if any(a[k] != b[k] for k in ["accepted", "verified", "proposed", "rounds", "generated_tokens"]):
                    raise ValueError("Independent GPT decode repeats differ")
                validation.append(dict(prefix=prefix, variant="d8_joint", status="passed",
                                       target_self_accepted=same["accepted"], target_self_verified=same["verified"]))
        result = dict(status="running", parameters=metadata, validation=validation, rows=[])
        expected = sum(len(prefix_records[record["name"]]) for record in records) * len(seeds) * len(variants_all)
        for ci, record in enumerate(records):
            record_prefixes = prefix_records[record["name"]]
            for prefix in record_prefixes:
                prompt = record["seq"][:prefix+1].tolist()
                for repeat, seed_base in enumerate(seeds):
                    seed = seed_base + record["crop_seed"]*1009 + prefix*17
                    variants = list(variants_all)
                    shift = (ci + prefixes.index(prefix) + repeat) % len(variants)
                    variants = variants[shift:] + variants[:shift]
                    for name in variants:
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        if name == "target_only":
                            row = call_precision(ordinary, target.transformer, None,
                                                 ranker, ranks, prompt, budget, 0, seed)
                        elif name == "p_only":
                            row = call_precision(ordinary, target.transformer, None,
                                                 ranker, ranks, prompt, budget, 1, seed)
                        elif name == "d8_joint":
                            row = call_precision(joint_decode, target.transformer, gpt.transformer,
                                                 prompt, budget, 2, seed)
                        else:
                            row, suffix = feature_decode(target.transformer, adapters[name], ranker,
                                                         ranks, prompt, budget, seed)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter()-started
                        row.update(decoder_seconds=row["seconds"], seconds=elapsed,
                                   tokens_per_second=row["generated_tokens"]/elapsed,
                                   variant=name, name=record["name"], scene=record["scene"],
                                   prefix=prefix, seed_base=seed_base)
                        result["rows"].append(row)
                        save(args.root / "online.json", result)
                        save(args.root / "progress.json", dict(status="running", phase="online",
                                                               completed=len(result["rows"]), expected=expected))
                        print(json.dumps(dict(run=len(result["rows"]), total=expected, variant=name,
                                              crop=record["name"], prefix=prefix, seed_base=seed_base,
                                              P=row["position_acceptance_verified"],
                                              F=row["feature_acceptance_verified"], tps=row["tokens_per_second"])), flush=True)
        result["status"] = "complete"
        save(args.root / "online.json", result)
        summary = dict(overall=summaries(result["rows"], variants_all),
                       by_prefix={str(p): summaries([r for r in result["rows"] if r["prefix"]==p], variants_all)
                                  for p in prefixes if any(r["prefix"] == p for r in result["rows"])},
                       by_seed={str(s): summaries([r for r in result["rows"] if r["seed_base"]==s], variants_all) for s in seeds},
                       fixed_prefix_overlap={name: fixed_summary[name] for name in fixed_names},
                       limitations=metadata["uncertainty"])
        save(args.root / "summary.json", summary)
        save(args.root / "progress.json", dict(status="complete", completed=expected, expected=expected))
        print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--prefixes", nargs="+", type=int, default=PREFIXES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS))
    parser.add_argument("--step12000-checkpoint", type=Path)
    parser.add_argument("--feature-checkpoint", action="append", metavar="NAME=PATH",
                        help="Compare named Feature weights in one interleaved run")
    parser.add_argument("--gpt", type=Path)
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        if args.feature_checkpoint:
            if args.step12000_checkpoint is not None:
                parser.error("Use either --feature-checkpoint or --step12000-checkpoint")
            model_paths = {}
            reserved = {"target_only", "p_only", "d8_feature", "d8_joint"}
            for value in args.feature_checkpoint:
                name, separator, path = value.partition("=")
                if not separator or not name or not path or name in reserved or name in model_paths:
                    parser.error("Feature weights require unique non-reserved NAME=PATH entries")
                model_paths[name] = path
        else:
            model_paths = {name: MODELS[name] for name in args.models}
        if args.step12000_checkpoint is not None:
            if "step12000" not in model_paths:
                parser.error("--step12000-checkpoint requires --models step12000")
            model_paths["step12000"] = str(args.step12000_checkpoint)
        run(args, model_paths=model_paths, gpt_path=args.gpt, joint=args.joint)
    except Exception:
        import traceback
        if args.root.exists():
            save(args.root / "progress.json", dict(status="failed", error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
