"""Measure unified Feature adapter costs and selected-student online behavior."""

import argparse
import json
from pathlib import Path
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(args):
    import torch
    from joblib import load
    from threadpoolctl import threadpool_limits
    from draft_a import learning_eval as ev
    from draft_a.feature_reuse import make_model
    from draft_a.pf_feature_unified import decode, MLPAdapter, GPTAdapter
    from draft_a.pf_matched_eval import mlp_decode
    from draft_a.pf_ranker_eval import decode as original_gpt, _prompts
    from bench_fast_hgb import FastHistRanker
    from bench_gaussiangpt_pf_block2_cow import load_feature_model
    from train_feature_state_distill import FeatureStateDraft

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    corpus, _, _, analysis = ev._helpers()
    plan = json.loads((args.source_root / "plan.json").read_text())
    split = "val" if args.mode == "validate-online" else "test"
    entries = [e for e in plan["entries"] if e["split"] == split][:args.scenes]
    records = analysis.read_records(args.source_root, entries)
    result = dict(status="running", split=split, parameters=dict(scenes=args.scenes, tokens=args.tokens,
                  prefixes=[0,64,256], seed=24685, k=2, profile=args.mode == "cost"),
                  policy=dict(temperature=.9,top_p=.9), rows=[], validation={}, timings={},
                  scope="same unified K2 decoder; cache allocation, prefix, ranker, proposal, Target verify, sampling, cleanup included")
    output = root / {"cost":"cost.json","validate-online":"validation_online.json","test":"online.json"}[args.mode]
    started = time.perf_counter()

    def save():
        result["timings"]["seconds"] = time.perf_counter() - started
        corpus.write_json(output, result)

    with ev._fp32_runtime(torch), threadpool_limits(limits=1, user_api="openmp"):
        target = corpus.load_target().float().eval().requires_grad_(False)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        draft = make_model(state["config"]).float().eval().requires_grad_(False)
        draft.load_state_dict(state["model"], strict=True)
        del state
        ranker = FastHistRanker(load(args.ranker))
        ranks = target.transformer.sparse_position_rank_map.cpu().tolist()
        old, _, _ = load_feature_model(args.old_checkpoint,args.old_head,target.transformer,torch.device("cuda"),8000,12096)
        adapters = dict(old_mlp_fp32=MLPAdapter(old),current_gpt_fp32=GPTAdapter(draft.transformer))
        if args.mode != "cost":
            training = json.loads((root / "training.json").read_text())
            require(training["status"] == "complete", "All training jobs must complete before online evaluation")
            for teacher in ("gpt", "target"):
                job = next(j for j in training["jobs"] if j["teacher"] == teacher and j["seed"] == 24685)
                path = Path(job["selected_checkpoint"])
                student, _, _ = load_feature_model(path,args.old_head,target.transformer,torch.device("cuda"),8000,12096)
                adapter = MLPAdapter(student)
                adapter.variant = teacher + "_student_fp32"
                adapters[adapter.variant] = adapter
            result["selected_weights"] = {j["teacher"]:dict(step=j["selected_step"],path=j["selected_checkpoint"],seed=j["seed"])
                                            for j in training["jobs"] if j["seed"] == 24685}
        result["variants"] = list(adapters)
        # Validate the real shared loop against prior model-specific paths.
        matches=[]
        for prefix,prompt in _prompts(records[0]):
            for variant,adapter in adapters.items():
                row, out = decode(target.transformer,adapter,ranker,ranks,prompt,32,24685)
                profiled, measured = decode(target.transformer,adapter,ranker,ranks,prompt,32,24685,profile=True)
                require(out == measured, "Stage timing changed the generated sequence")
                require(all(row[k] == profiled[k] for k in ["accepted","verified","proposed","position_accepted","feature_accepted","rounds"]), "Stage timing changed counts")
                if variant == "old_mlp_fp32":
                    prior, prior_out = mlp_decode(target.transformer,old,ranker,ranks,prompt,32,24685)
                    require(prior_out == out, "Unified MLP differs from previous decoder")
                matches.append(dict(variant=variant,prefix=prefix,tokens=len(out),status="passed"))
        result["validation"] = dict(status="passed",profile_sequence_matches=matches)
        save()
        for index,record in enumerate(records):
            for prefix,prompt in _prompts(record):
                seed=(24685+index*1009+prefix*17)%(2**63-1)
                names=list(adapters)
                shift=(index+prefix//64)%len(names)
                names=names[shift:]+names[:shift]
                if args.mode != "cost":
                    row=original_gpt(target.transformer,draft.transformer,ranker,ranks,prompt,args.tokens,0,seed)
                    row.update(name=record["name"],split=split,variant="target_only",requested_tokens=args.tokens)
                    result["rows"].append(row)
                    save()
                for variant in names:
                    row,out=decode(target.transformer,adapters[variant],ranker,ranks,prompt,args.tokens,seed)
                    row.update(name=record["name"],split=split,variant=variant)
                    if args.mode == "cost":
                        profiled,profile_out=decode(target.transformer,adapters[variant],ranker,ranks,prompt,args.tokens,seed,profile=True)
                        require(out==profile_out, "Profile changed output on full trajectory")
                        row.update(stage_profile=profiled["stage_profile"],profile_seconds=profiled["seconds"],
                                   profile_overhead_ratio=profiled["seconds"]/row["seconds"],profile_output_matches=True)
                    result["rows"].append(row)
                    save()
                    print(json.dumps(dict(event="online",mode=args.mode,name=record["name"],prefix=prefix,variant=variant,
                                          F=row["feature_acceptance_verified"],P=row["position_acceptance_verified"],
                                          tps=row["tokens_per_second"],mean_output=row["mean_generated_per_round"])),flush=True)
    result["status"]="complete"
    save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["cost","validate-online","test"])
    p.add_argument("--root",type=Path,default=Path("draft_a/runs/feature_student_transfer_20261014"))
    p.add_argument("--scenes",type=int,default=12)
    p.add_argument("--tokens",type=int,default=512)
    p.add_argument("--source-root",type=Path,default=Path("draft_a/runs/rollout500_20261013"))
    p.add_argument("--checkpoint",type=Path,default=Path("draft_a/runs/feature_acceptance_20261014/adapt_d8_acceptance/step_3600.pt"))
    p.add_argument("--old-checkpoint",type=Path,default=Path("profiling/feature_draft/state_distill_mixed_overlap/feature_state_draft_best.pt"))
    p.add_argument("--old-head",type=Path,default=Path("profiling/feature_draft/state_generation_512/states/frozen_feature_head.pt"))
    p.add_argument("--ranker",type=Path,default=Path("profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib"))
    run(p.parse_args())


if __name__=="__main__":main()
