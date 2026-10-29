"""Exact token proposal, Target verification, and cache rollback experiments."""
import argparse
import json
import time
from pathlib import Path

import torch

from draft_a.corpus_v2 import ROOT, EOS, load_target, write_json
from draft_a.evaluate_v2 import load_draft
from draft_a.eval_overlap import sampling_probs
from model.gpt.engine import KVCache


def cache_for(core,length):
    cfg=core.config
    return KVCache(batch_size=1,num_heads=cfg.n_kv_head,seq_len=length,
        head_dim=cfg.n_embd//cfg.n_head,num_layers=cfg.n_layer,
        device="cuda",dtype=next(core.parameters()).dtype)


def policy(logits):
    return sampling_probs(logits,None,.9,.9)


def draw(p,rng):
    return int(torch.multinomial(p,1,generator=rng))


def residual(p,q):
    r=(p-q).clamp_min(0)
    mass=r.sum()
    if not torch.isfinite(mass) or mass<=0:
        raise RuntimeError("Rejected proposal has no residual mass")
    return r/mass


def toy_test():
    p=torch.tensor([.05,.25,.0,.7],dtype=torch.float64)
    q=torch.tensor([.4,.1,.5,.0],dtype=torch.float64)
    r=residual(p,q)
    accepted=torch.minimum(p,q)
    expected=accepted+(1-accepted.sum())*r
    torch.testing.assert_close(expected,p,atol=1e-12,rtol=0)
    rng=torch.Generator().manual_seed(34)
    proposals=torch.multinomial(q,100000,replacement=True,generator=rng)
    accept=torch.rand(100000,generator=rng)<(p[proposals]/q[proposals]).clamp_max(1)
    corrected=torch.multinomial(r,int((~accept).sum()),replacement=True,generator=rng)
    proposals[~accept]=corrected
    empirical=torch.bincount(proposals,minlength=4)/100000
    assert float((empirical-p).abs().max())<.007
    return dict(expected=p.tolist(),empirical=empirical.tolist(),acceptance=float(accept.float().mean()))


@torch.inference_mode()
def verify_blocks(core,seq):
    seq=seq[:129].reshape(1,-1).cuda()
    step_cache=cache_for(core,seq.shape[1]+17)
    refs=[]
    for t in range(seq.shape[1]):
        refs.append(policy(core(seq[:,t:t+1],kv_cache=step_cache)[0,-1]).cpu())
    max_tv=0.
    for width in (2,4,8,16):
        cached=cache_for(core,seq.shape[1]+17)
        for start in range(0,seq.shape[1],width):
            logits=core(seq[:,start:start+width],kv_cache=cached)[0]
            for j,row in enumerate(logits):
                tv=float((policy(row).cpu()-refs[start+j]).abs().sum()/2)
                max_tv=max(max_tv,tv)
                assert tv<.01,(width,start+j,tv)
        # Rewind into an earlier prefix and verify that stale suffix values are ignored.
        cached.cache_seqlens.fill_(7)
        logits=core(seq[:,7:15],kv_cache=cached)[0]
        for j,row in enumerate(logits):
            tv=float((policy(row).cpu()-refs[7+j]).abs().sum()/2)
            max_tv=max(max_tv,tv)
            assert tv<.01,("rewind",j,tv)
    return max_tv


def validate_output(prompt,output):
    full=prompt[1:]+output
    if EOS in full:
        assert full[-1]==EOS and full.count(EOS)==1
        full=full[:-1]
        assert len(full)%2==0
    pos=full[::2]
    feat=full[1::2]
    assert all(0<=p<8000 for p in pos)
    assert all(8000<=f<EOS for f in feat)
    assert all(b>a for a,b in zip(pos,pos[1:]))


@torch.inference_mode()
def decode(target,draft,prompt,budget,k,seed):
    rng=torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    started=time.perf_counter()
    length=len(prompt)
    capacity=length+budget+k+2
    tc=cache_for(target,capacity)
    dc=cache_for(draft,capacity) if k else None
    stats=dict(target_calls=0,draft_calls=0,proposed=0,verified=0,accepted=0,rounds=0,
        position_proposed=0,feature_proposed=0,position_accepted=0,feature_accepted=0,
        accepted_lengths=[],target_input_tokens=0)
    def tf(tokens):
        stats["target_calls"]+=1
        stats["target_input_tokens"]+=len(tokens)
        return target(torch.tensor([tokens],device="cuda"),kv_cache=tc)[0]
    def df(tokens):
        stats["draft_calls"]+=1
        return draft(torch.tensor([tokens],device="cuda"),kv_cache=dc)[0]
    if length>1:
        tf(prompt[:-1])
        if k:
            df(prompt[:-1])
    prefix=list(prompt)
    while len(prefix)-length<budget:
        assert tc.get_pos()==len(prefix)-1
        if not k:
            token=draw(policy(tf(prefix[-1:])[-1]),rng)
            prefix.append(token)
            if token==EOS:
                break
            continue
        assert dc.get_pos()==len(prefix)-1
        stats["rounds"]+=1
        n=min(k,budget-(len(prefix)-length))
        candidates=[]
        qs=[]
        token=prefix[-1]
        for _ in range(n):
            q=policy(df([token])[-1])
            token=draw(q,rng)
            candidates.append(token)
            qs.append(q)
            if token==EOS:
                break
        # The final row predicts a bonus token after a completely accepted block.
        logits=tf(prefix[-1:]+candidates)
        accepted=0
        rejected=False
        base=len(prefix)
        stats["proposed"]+=len(candidates)
        for j in range(len(candidates)):
            kind="position" if (base+j-1)%2==0 else "feature"
            stats[kind+"_proposed"]+=1
        for j,(token,q) in enumerate(zip(candidates,qs)):
            p=policy(logits[j])
            kind="position" if (base+j-1)%2==0 else "feature"
            stats["verified"]+=1
            if float(torch.rand((),device="cuda",generator=rng))<min(1.,float(p[token]/q[token])):
                prefix.append(token)
                accepted+=1
                stats["accepted"]+=1
                stats[kind+"_accepted"]+=1
                if token==EOS:
                    break
            else:
                prefix.append(draw(residual(p,q),rng))
                # Both caches retain exactly the prefix preceding the replacement.
                tc.cache_seqlens.fill_(base+j)
                dc.cache_seqlens.fill_(base+j)
                rejected=True
                break
        stats["accepted_lengths"].append(accepted)
        if prefix[-1]==EOS or len(prefix)-length>=budget:
            break
        if not rejected:
            assert accepted==len(candidates)
            bonus=draw(policy(logits[len(candidates)]),rng)
            df(candidates[-1:])
            prefix.append(bonus)
            if bonus==EOS:
                break
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    output=prefix[length:]
    validate_output(prompt,output)
    stats.update(generated_tokens=len(output),seconds=elapsed,tokens_per_second=len(output)/elapsed,
        eos=bool(output and output[-1]==EOS),k=k,seed=seed,prompt_tokens=len(prompt)-1,
        acceptance=stats["accepted"]/stats["proposed"] if stats["proposed"] else None,
        mean_accepted_length=sum(stats["accepted_lengths"])/stats["rounds"] if stats["rounds"] else None)
    return stats


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--checkpoint",type=Path)
    ap.add_argument("--tokens",type=int,default=256)
    ap.add_argument("--output-tag",choices=["v2","previous"],default="v2")
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--toy-only",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("high")
    toy=toy_test()
    if args.toy_only:
        print(json.dumps(toy))
        return
    target=load_target().transformer
    checkpoint=args.checkpoint or args.root/"training/best.pt"
    draft=load_draft(checkpoint).transformer
    plan=json.loads((args.root/"plan.json").read_text())
    entries=[e for e in plan["entries"] if e["split"]=="test"]
    if args.self_test:
        entries=[e for e in plan["entries"] if (args.root/"corpus"/e["name"]).exists()]
    candidates=[torch.load(args.root/"corpus"/e["name"],map_location="cpu",weights_only=True) for e in entries]
    record=max(candidates,key=lambda r:len(r["tokens"]))
    seq=torch.cat([torch.tensor([EOS]),record["tokens"]])
    tv=dict(target=verify_blocks(target,seq),draft=verify_blocks(draft,seq))
    rows=[]
    # Warm both models and the same allocation/sampling paths used for timings.
    decode(target,draft,[EOS],16,0,0)
    decode(target,draft,[EOS],16,4,1)
    if args.self_test:
        same_model=[]
        ending=[]
        for k in (2,4,8,16):
            rows.append(decode(target,draft,[EOS],32,k,101+k))
            matched=decode(target,target,[EOS],32,k,601+k)
            assert matched["acceptance"]>.95,matched
            same_model.append(matched)
            terminal=decode(target,target,[EOS,7999,8000],8,k,901+k)
            assert terminal["eos"] and terminal["generated_tokens"]==1,terminal
            ending.append(terminal)
        write_json(args.root/"decoder_self_test.json",dict(toy=toy,max_tv=tv,rows=rows,
            same_model=same_model,terminal_eos=ending))
        return
    for context in (0,1024):
        body=record["tokens"][:context].tolist()
        prompt=[EOS]+body
        for k in (0,2,4,8,16):
            decode(target,draft,prompt,16,k,800+k)
        for seed in (3101,3102,3103):
            for k in (0,2,4,8,16):
                row=decode(target,draft,prompt,args.tokens,k,seed)
                row["requested_context"]=context
                rows.append(row)
                print(json.dumps(row),flush=True)
                write_json(args.root/f"benchmark_{args.output_tag}_progress.json",dict(completed=len(rows),total=30,last=row))
    summary={}
    for context in (0,1024):
        reference=[r for r in rows if r["requested_context"]==context and r["k"]==0]
        target_tps=sum(r["generated_tokens"] for r in reference)/sum(r["seconds"] for r in reference)
        summary[str(context)]={"target_tokens_per_second":target_tps}
        for k in (2,4,8,16):
            subset=[r for r in rows if r["requested_context"]==context and r["k"]==k]
            tps=sum(r["generated_tokens"] for r in subset)/sum(r["seconds"] for r in subset)
            summary[str(context)][str(k)]=dict(tokens_per_second=tps,speedup=tps/target_tps,
                acceptance=sum(r["accepted"] for r in subset)/sum(r["proposed"] for r in subset),
                mean_accepted_length=sum(sum(r["accepted_lengths"]) for r in subset)/sum(r["rounds"] for r in subset),
                target_calls=sum(r["target_calls"] for r in subset),
                actual_tokens=sum(r["generated_tokens"] for r in subset))
    write_json(args.root/f"benchmark_{args.output_tag}.json",dict(checkpoint=str(checkpoint),max_tv=tv,toy=toy,rows=rows,
        temperature=.9,top_p=.9,dtype="float32 weights and cache",summary=summary,
        scope="Includes cache allocation, prefill, proposal, verification, correction and sampling; excludes model loading"))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
