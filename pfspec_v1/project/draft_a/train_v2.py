"""CE training with two complete trajectories per optimizer update."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from draft_a.corpus_v2 import ROOT, EOS, write_json
from draft_a.analyze_v2 import read_records
from draft_a.train_draft_a_v2 import evaluate
from model.gpt.gpt import NanoGPTGaussianModel


def make_model(cfg):
    assert (cfg.gpt_size.n_layer,cfg.gpt_size.n_embd,cfg.gpt_size.n_head)==(8,512,8)
    assert cfg.position_vocab_size==8000 and cfg.rope_basis=="position"
    assert cfg.sparse_position_hard_constraints and str(cfg.chunk_order)=="xyz"
    model=NanoGPTGaussianModel(config=cfg,codebook_size=4096,tokens_per_latent=2,num_feature_tokens=1).cuda()
    assert model.eos_token_id==EOS
    return model


def sequence(record):
    return torch.cat([torch.tensor([EOS]),record["tokens"],
                      torch.tensor([EOS],dtype=torch.long) if record["has_eos"] else torch.empty(0,dtype=torch.long)])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--steps",type=int,default=5000)
    ap.add_argument("--eval-every",type=int,default=250)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--seed",type=int,default=24681)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    plan=json.loads((args.root/"plan.json").read_text())
    entries=plan["entries"]
    if args.smoke:
        entries=[e for e in entries if (args.root/"corpus"/e["name"]).exists()]
    else:
        analysis=json.loads((args.root/"corpus_analysis.json").read_text())
        assert analysis["all"]["sequences"]==500
    records=read_records(args.root,entries)
    for r in records:
        r["seq"]=sequence(r)
    train=[r for r in records if r["split"]=="train"]
    val=[r for r in records if r["split"]=="val"]
    if args.smoke:
        train=sorted(train,key=lambda r:len(r["seq"]))
        val=train[:1]
    cfg=OmegaConf.load("conf/model/gpt_draft_a.yaml")
    assert all(len(r["seq"])-1<=cfg.n_ctx for r in records)
    model=make_model(cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    out=args.root/("smoke_training" if args.smoke else "training")
    out.mkdir(parents=True,exist_ok=True)
    if (out/"last.pt").exists() and not args.resume:
        raise FileExistsError(out/"last.pt")
    rng=random.Random(args.seed+1)
    order=[]
    cursor=0
    start_step=0
    best=math.inf
    if args.resume:
        state=torch.load(out/"last.pt",map_location="cpu",weights_only=True)
        assert state["train_files"]==[r["name"] for r in train]
        assert state["val_files"]==[r["name"] for r in val]
        assert state["lr"]==args.lr and state["seed"]==args.seed
        model.load_state_dict(state["model"],strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step=state["step"]
        best=state["best_ce"]
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        order,cursor=state["order"],state["cursor"]
    bf16=torch.cuda.is_bf16_supported()
    print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),
        train=len(train),val=len(val),steps=args.steps,lr=args.lr,effective_batch=2)),flush=True)
    model.train()
    started=time.monotonic()
    initial=evaluate(model,val,bf16)
    write_json(out/"initial_ce.json", {k:(v if math.isfinite(v) else None) for k,v in initial.items()})
    for step in range(start_step+1,args.steps+1):
        batch=[]
        for _ in range(2):
            if cursor>=len(order):
                order=list(range(len(train)))
                rng.shuffle(order)
                cursor=0
            batch.append(train[order[cursor]])
            cursor+=1
        optimizer.zero_grad(set_to_none=True)
        num_tokens=sum(len(r["seq"])-1 for r in batch)
        total_loss=0.
        for r in batch:
            seq=r["seq"].unsqueeze(0).cuda()
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=bf16):
                loss=model(input_ids=seq,labels=seq)
            assert torch.isfinite(loss), step
            weight=(len(r["seq"])-1)/num_tokens
            (loss*weight).backward()
            total_loss+=loss.item()*weight
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(norm),step
        optimizer.step()
        if step==1 or step%10==0 or args.smoke:
            print(json.dumps(dict(step=step,train_ce=total_loss,tokens=num_tokens,
                grad_norm=norm.item(),elapsed_seconds=time.monotonic()-started)),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            metrics=evaluate(model,val,bf16)
            assert math.isfinite(metrics["all"])
            improved=metrics["all"]<best
            best=min(best,metrics["all"])
            payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                config=OmegaConf.to_container(cfg,resolve=True),step=step,seed=args.seed,lr=args.lr,
                metrics=metrics,best_ce=best,train_files=[r["name"] for r in train],
                val_files=[r["name"] for r in val],test_files=[r["name"] for r in records if r["split"]=="test"],
                corpus=str(args.root/"corpus"),python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),order=order,cursor=cursor)
            for name in (["last.pt","best.pt"] if improved else ["last.pt"]):
                temp=out/(name+".tmp")
                torch.save(payload,temp)
                temp.replace(out/name)
            metrics={k:(v if math.isfinite(v) else None) for k,v in metrics.items()}
            progress=dict(step=step,steps=args.steps,metrics=metrics,best_ce=best,
                elapsed_seconds=time.monotonic()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
            write_json(out/"progress.json",progress)
            with (out/"validation.jsonl").open("a") as f:
                f.write(json.dumps(progress)+"\n")
            print(json.dumps(progress),flush=True)
    write_json(out/"complete.json",dict(step=args.steps,best_ce=best,smoke=args.smoke))


if __name__=="__main__":
    main()
