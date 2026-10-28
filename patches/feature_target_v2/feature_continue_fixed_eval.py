"""Compare the two selected models on identical complete test prefixes."""

import json
from pathlib import Path

from draft_a.feature_target_v2 import records_for, evaluate, save
from draft_a.feature_target_continue import START, ROOT


def main():
    import numpy as np
    import torch
    from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock, load_feature_model
    from draft_a.corpus_v2 import load_target
    from draft_a.learning_eval import _fp32_runtime
    if json.loads((ROOT/"pipeline.json").read_text())["status"] != "complete":
        raise ValueError("Finish training and the planned online evaluation first")
    records = records_for("test")
    training = json.loads((ROOT/"training_plan.json").read_text())
    test_names = {r["name"] for r in records}
    if test_names & (set(training["train_files"]) | set(training["val_files"])):
        raise ValueError("Test files appeared in training or model selection")
    result = dict(scope="all eligible F slots on identical test prefixes; expected single-step acceptance",
                  split="test", scenes=len(records), policy=dict(temperature=.9,top_p=.9),
                  checkpoints=dict(step3600=str(START/"selected.pt"), continued=str(ROOT/"selected.pt")), rows=[])
    with torch.inference_mode(), _fp32_runtime(torch):
        target = load_target().float().eval().requires_grad_(False)
        head = Path("profiling/feature_draft/state_generation_512/states/frozen_feature_head.pt")
        models = {name:load_feature_model(Path(path),head,target.transformer,torch.device("cuda"),8000,12096)[0]
                  for name,path in result["checkpoints"].items()}
        capture = CaptureLastBlock(target.transformer)
        try:
            for record in records:
                seq = record["seq"]
                depth = torch.arange(1,len(record["tokens"])//2)
                indices = (2*depth+1).cuda()
                capture.output = None
                logits = target(input_ids=seq[:-1].unsqueeze(0).cuda())[0]
                states = capture.output[0]
                data = dict(name=record["name"],depth=depth, position=seq[2*depth+1],
                            sampled_feature=seq[2*depth+2]-8000, previous_feature=seq[2*depth]-8000,
                            previous_state=states.index_select(0,indices-2).float().cpu(),
                            target=logits.index_select(0,indices)[:,8000:12096].float().softmax(-1).cpu())
                capture.output = None
                del logits,states,indices
                row = dict(name=record["name"], rows=len(depth),
                           **{name:evaluate(model,[data],256)["pooled_overlap"] for name,model in models.items()})
                row["difference"] = row["continued"]-row["step3600"]
                result["rows"].append(row)
                del data
        finally:
            capture.close()
    rows = result["rows"]
    total = sum(r["rows"] for r in rows)
    result.update(feature_slots=total, pooled={name:sum(r[name]*r["rows"] for r in rows)/total
                                             for name in models},
                  scene_mean={name:sum(r[name] for r in rows)/len(rows) for name in models},
                  improved_trajectories=sum(r["difference"]>0 for r in rows))
    rng = np.random.default_rng(24689)
    differences = []
    for _ in range(3000):
        sample = [rows[i] for i in rng.integers(0,len(rows),len(rows))]
        differences.append(sum(r["difference"]*r["rows"] for r in sample)/sum(r["rows"] for r in sample))
    result["pooled_difference"] = result["pooled"]["continued"]-result["pooled"]["step3600"]
    result["difference_ci95"] = np.quantile(differences,[.025,.975]).tolist()
    save(ROOT/"fixed_test_overlap.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__ == "__main__":
    main()
