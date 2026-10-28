"""Compare both Draft weights on the same unseen Teacher trajectories."""
import argparse
import collections
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from draft_a.corpus_v2 import ROOT, EOS, load_target, write_json
from draft_a.analyze_v2 import read_records
from draft_a.train_v2 import make_model, sequence
from draft_a.eval_overlap import sampling_probs, choose_slots


def load_draft(path):
    state=torch.load(path,map_location="cpu",weights_only=True)
    draft=make_model(OmegaConf.create(state["config"]))
    draft.load_state_dict(state["model"],strict=True)
    return draft.eval().requires_grad_(False)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--checkpoint",type=Path)
    ap.add_argument("--split",choices=["val","test"],default="test")
    ap.add_argument("--max-scenes",type=int,default=50)
    ap.add_argument("--samples-per-type",type=int,default=32)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("high")
    plan=json.loads((args.root/"plan.json").read_text())
    entries=[e for e in plan["entries"] if e["split"]==args.split]
    if args.smoke:
        entries=[e for e in plan["entries"] if (args.root/"corpus"/e["name"]).exists()][-1:]
    records=read_records(args.root,entries[:args.max_scenes])
    assert records
    checkpoint=args.checkpoint or args.root/"training/best.pt"
    target=load_target()
    models={"previous160":load_draft(plan["old_checkpoint"]),"v2":load_draft(checkpoint)}
    results=[]
    real_path=args.root/"real_crops.pt"
    if real_path.exists() and not args.smoke:
        real=torch.load(real_path,map_location="cpu",weights_only=True)
        records += [dict(r,name=f"real_crop_{r['crop_seed']}",split="real") for r in real]
    for record in records:
        seq=sequence(record)
        if args.smoke:
            seq=seq[:65]
        x=seq[:-1].unsqueeze(0).cuda()
        slots=choose_slots(x.shape[1],args.samples_per_type)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            teacher=target(x)[0,slots].float().cpu()
            students={name:model(x)[0,slots].float().cpu() for name,model in models.items()}
        rows=[]
        for j,t in enumerate(slots):
            pos=t%2==0
            prev=int(seq[t-1]) if pos and t>=2 else None
            p=sampling_probs(teacher[j],prev,0.9,0.9)
            gold=int(seq[t+1])
            row=dict(depth=t,kind="position" if pos else "feature",eos=gold==EOS,
                     target_eos=float(p[EOS]),target_raw_nll=float(F.cross_entropy(teacher[j:j+1],torch.tensor([gold]))))
            for name,logits in students.items():
                q=sampling_probs(logits[j],prev,0.9,0.9)
                row[name]=dict(overlap=float(torch.minimum(p,q).sum()),draft_eos=float(q[EOS]),
                               top1_match=int(p.argmax()==q.argmax()))
            rows.append(row)
        result=dict(name=record["name"],split=record["split"],context=x.shape[1],rows=rows)
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)
    summary={}
    for split in sorted({r["split"] for r in results}):
        subset=[row for r in results if r["split"]==split for row in r["rows"]]
        values={}
        for kind in ("position","feature","eos"):
            rows=[r for r in subset if (r["eos"] if kind=="eos" else r["kind"]==kind)]
            values[kind]=dict(n=len(rows))
            for name in models:
                values[kind][name]={key:sum(r[name][key] for r in rows)/len(rows) if rows else None
                    for key in ("overlap","top1_match","draft_eos")}
            values[kind]["target_raw_nll"]=sum(r["target_raw_nll"] for r in rows)/len(rows) if rows else None
        summary[split]=values
    output=dict(checkpoint=str(checkpoint),temperature=.9,top_p=.9,dtype="bf16 autocast",
                scope="Teacher prefixes; overlaps do not measure online acceptance",summary=summary,scenes=results)
    name="overlap_smoke.json" if args.smoke else f"overlap_{args.split}.json"
    write_json(args.root/name,output)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
