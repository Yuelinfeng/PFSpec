"""Compare D8 joint P/F Gamma 2, 4 and 6 on recorded real VFront prefixes."""

import argparse
import json
from pathlib import Path
import time
import traceback

from draft_a.feature_real_eval import CROPS, PREFIXES, SEEDS, aggregate
from draft_a.feature_target_v2 import save


CHECKPOINT = Path("draft_a/runs/feature_acceptance_20261014/adapt_d8_acceptance/step_3600.pt")
ROOT = Path("draft_a/runs/feature_d8_gamma_eval_20261014")
GAMMAS = [2, 4, 6]


def total(rows):
    value = aggregate(rows)
    for key in ["accepted", "verified", "proposed", "draft_calls", "target_input_tokens"]:
        value[key] = sum(r.get(key, 0) for r in rows)
    value["mean_accepted_length"] = value["accepted"]/value["rounds"] if value["rounds"] else None
    value["mean_generated_per_round"] = value["generated_tokens"]/value["rounds"] if value["rounds"] else None
    value["eos_runs"] = sum(r["eos"] for r in rows)
    return value


def summary(rows):
    out = {name: total([r for r in rows if r["variant"] == name])
           for name in ["target_only", *[f"gamma{g}" for g in GAMMAS]]}
    for g in GAMMAS:
        out[f"gamma{g}"]["speedup"] = out[f"gamma{g}"]["tps"]/out["target_only"]["tps"]
        out[f"gamma{g}"]["relative_to_gamma2"] = out[f"gamma{g}"]["tps"]/out["gamma2"]["tps"]
    return out


def run(args):
    import torch
    from threadpoolctl import threadpool_limits
    from draft_a.analyze_v2 import real_records
    from draft_a.corpus_v2 import EOS, load_target, validate
    from draft_a.feature_reuse import make_model
    from draft_a.learning_eval import _fp32_runtime
    from draft_a.feature_acceptance_eval import decode, verify_blocks
    from draft_a.pf_ranker_eval import decode as ordinary, validate_sequence

    if args.root.exists():
        raise FileExistsError("Use a new result directory; prior results are retained")
    cached = torch.load(CROPS, map_location="cpu", weights_only=True)
    rebuilt = real_records()
    if len(cached) != len(rebuilt) or any(not torch.equal(a["tokens"], b["tokens"])
                                        or a["path"] != b["path"] for a,b in zip(cached,rebuilt)):
        raise ValueError("Real inputs differ from the recorded crops")
    records, duplicates = [], []
    for row in rebuilt:
        validate(row)
        if any(torch.equal(row["tokens"], old["tokens"]) for old in records):
            duplicates.append(row["crop_seed"])
            continue
        row.update(name=f"real_crop_{row['crop_seed']}", scene=Path(row["path"]).parents[1].name,
                   seq=torch.cat([torch.tensor([EOS]), row["tokens"]]))
        records.append(row)
    if args.smoke:
        records = records[:1]
    prefixes = [64] if args.smoke else PREFIXES
    seeds = SEEDS[:1] if args.smoke else SEEDS
    budget = 32 if args.smoke else 512
    args.root.mkdir(parents=True)
    metadata = dict(target="checkpoints/gpt_vfront.ckpt", draft=str(CHECKPOINT), gammas=GAMMAS,
                    source_crops=str(CROPS), source="3D-FRONT reconstructed Gaussians encoded by original VQ-VAE",
                    unique_scenes=len({r["scene"] for r in records}), unique_crops=len(records),
                    duplicate_crop_seeds=duplicates, prefixes=prefixes, seed_groups=seeds,
                    budget=budget, precision="FP32", temperature=.9, top_p=.9,
                    gamma_definition="number of consecutive token proposals; may begin with P or F",
                    timing="common outer synchronized wall time includes input/output validation, cache allocation/release, prefill, GPT proposal, Target verification, rejection handling and sampling; excludes loading/data preparation",
                    uncertainty="one scene; crop/seed variation cannot estimate scene generalization",
                    crops=[dict(name=r["name"],scene=r["scene"],crop_seed=r["crop_seed"],source=r["path"],
                                body_tokens=len(r["tokens"])) for r in records])
    save(args.root/"plan.json", metadata)
    save(args.root/"progress.json",dict(status="running",phase="validation"))
    with torch.inference_mode(), _fp32_runtime(torch), threadpool_limits(limits=1,user_api="openmp"):
        target = load_target().float().eval().requires_grad_(False)
        state = torch.load(CHECKPOINT,map_location="cpu",weights_only=True)
        size = state["config"]["gpt_size"]
        if (size["n_layer"],size["n_embd"],size["n_head"]) != (8,512,8) or state["step"] != 3600:
            raise ValueError("Use the selected high-acceptance step3600 D8")
        draft = make_model(state["config"]).float().eval().requires_grad_(False)
        draft.load_state_dict(state["model"],strict=True)
        del state
        if not torch.equal(draft.transformer.sparse_position_rank_map,target.transformer.sparse_position_rank_map):
            raise ValueError("D8 and Target Position ordering differs")
        ranks = target.transformer.sparse_position_rank_map.cpu().tolist()
        for r in records:
            validate_sequence(r["seq"].tolist(),[],ranks)
        numerical = {name:verify_blocks(core,records[0]["seq"])
                     for name,core in [("target",target.transformer),("d8",draft.transformer)]}
        validation = []
        for prefix in prefixes:
            prompt = records[0]["seq"][:prefix+1].tolist()
            ordinary(target.transformer,None,None,ranks,prompt,32,0,34700)
            for gamma in GAMMAS:
                same = decode(target.transformer,target.transformer,prompt,32,gamma,34700)
                if same["accepted"] != same["verified"]:
                    raise ValueError("Target proposals to itself must all be accepted")
                a = decode(target.transformer,draft.transformer,prompt,32,gamma,34700)
                b = decode(target.transformer,draft.transformer,prompt,32,gamma,34700)
                if any(a[k] != b[k] for k in ["accepted","verified","proposed","rounds","generated_tokens"]):
                    raise ValueError("Repeated GPT decoding counts differ")
                validation.append(dict(prefix=prefix,gamma=gamma,status="passed",
                                       target_self_accepted=same["accepted"],target_self_verified=same["verified"]))
        save(args.root/"numerical_validation.json",dict(models=numerical,short_runs=validation))
        result = dict(status="running",parameters=metadata,rows=[])
        variants = ["target_only",*[f"gamma{g}" for g in GAMMAS]]
        expected = len(records)*len(prefixes)*len(seeds)*len(variants)
        for ci,record in enumerate(records):
            for pi,prefix in enumerate(prefixes):
                prompt = record["seq"][:prefix+1].tolist()
                for repeat,seed_base in enumerate(seeds):
                    seed = seed_base+record["crop_seed"]*1009+prefix*17
                    shift = (ci+pi+repeat)%len(variants)
                    order = variants[shift:]+variants[:shift]
                    for name in order:
                        gamma = 0 if name=="target_only" else int(name.removeprefix("gamma"))
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        if not gamma:
                            row = ordinary(target.transformer,None,None,ranks,prompt,budget,0,seed)
                        else:
                            row = decode(target.transformer,draft.transformer,prompt,budget,gamma,seed)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter()-started
                        row.update(decoder_seconds=row["seconds"],seconds=elapsed,
                                   tokens_per_second=row["generated_tokens"]/elapsed,
                                   variant=name,gamma=gamma,name=record["name"],scene=record["scene"],
                                   prefix=prefix,seed_base=seed_base)
                        result["rows"].append(row)
                        save(args.root/"online.json",result)
                        save(args.root/"progress.json",dict(status="running",phase="online",
                                                            completed=len(result["rows"]),expected=expected))
                        print(json.dumps(dict(run=len(result["rows"]),total=expected,variant=name,
                                              crop=record["name"],prefix=prefix,seed_base=seed_base,
                                              P=row["position_acceptance_verified"],F=row["feature_acceptance_verified"],
                                              tps=row["tokens_per_second"])),flush=True)
        result["status"] = "complete"
        save(args.root/"online.json",result)
        report = dict(overall=summary(result["rows"]),
                      by_prefix={str(p):summary([r for r in result["rows"] if r["prefix"]==p]) for p in prefixes},
                      by_seed={str(s):summary([r for r in result["rows"] if r["seed_base"]==s]) for s in seeds},
                      limitations=metadata["uncertainty"])
        save(args.root/"summary.json",report)
        save(args.root/"progress.json",dict(status="complete",completed=expected,expected=expected))
        print(json.dumps(report),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    existed = args.root.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.root.exists():
            save(args.root/"progress.json",dict(status="failed",error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
