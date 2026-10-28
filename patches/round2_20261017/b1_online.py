"""Rotate seven methods on the same development prompts and seeds."""
import argparse
import collections
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from joblib import load

PROJECT=Path('/root/workspace/GaussianGPT')
OLD=Path('/root/autodl-tmp/gaussiangpt_next_round_20261017')
OUT=Path('/root/autodl-tmp/gaussiangpt_round2_20261017')
sys.path[:0]=[str(PROJECT),str(OLD),str(OUT)]
import step5_method_compare as old
import bench_gaussiangpt_medusa_linear as medusa
from draft_a.pf_feature_unified import MLPAdapter, decode as pf_decode
from draft_a.pf_ranker_eval import decode as p_decode, validate_sequence
from bench_gaussiangpt_pf_block2_cow import load_feature_model
from collect_feature_teacher_from_s3 import policy_batch
from b1_feature_models import make_model


def summarize(rows, methods):
    scenes=sorted({r['scene'] for r in rows})
    by=collections.defaultdict(list)
    for r in rows:by[(r['scene'],r['method'])].append(r)
    def speed(rr):return sum(r['generated_tokens'] for r in rr)/sum(r['seconds'] for r in rr)
    result={}
    for m in methods:
        rr=[r for r in rows if r['method']==m]
        if not rr:continue
        count=collections.Counter()
        for r in rr:
            count.update({k:v for k,v in r['stats'].items() if isinstance(v,int) and any(s in k for s in ['accepted','verified','proposed','target_calls','rounds'])})
        item=dict(runs=len(rr),micro_tokens_per_second=speed(rr),counts=dict(count),comparisons={})
        for ref in ['target','p_only','old_pf','b0']:
            if ref==m:continue
            ratios=[]
            for scene in scenes:
                a,b=by[(scene,m)],by[(scene,ref)]
                if a and b and {(r['prefix'],r['seed']) for r in a}=={(r['prefix'],r['seed']) for r in b}:
                    ratios.append(speed(a)/speed(b))
            if not ratios:continue
            rng=np.random.default_rng(20261020)
            boot=np.exp(np.mean(np.log(rng.choice(ratios,(10000,len(ratios)),replace=True)),axis=1))
            item['comparisons'][ref]=dict(sources=len(ratios),geomean=float(np.exp(np.mean(np.log(ratios)))),
                ci95=np.quantile(boot,[.025,.975]).tolist(),sources_faster=sum(x>1 for x in ratios))
        result[m]=item
    return result


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--profile',action='store_true')
    ap.add_argument('--training-seed',type=int,default=20261018)
    args=ap.parse_args();torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    target,dev,pv,eos,ranks,cache=medusa.target_loader(SimpleNamespace(gpt=PROJECT/'checkpoints/gpt_vfront.ckpt',vqvae=PROJECT/'checkpoints/vqvae_vfront.ckpt'))
    ranker=old.FastHistRanker(load(old.RANKER))
    feature=load_feature_model(old.FEATURE,old.FEATURE_HEAD,target,dev,pv,eos)[0].eval().requires_grad_(False)
    models={'old_pf':feature}
    for name in ['b0','b1_m','b1_h','b1_e']:
        p=OUT/'b1_training'/str(args.training_seed)/name/'selected.pt'
        if p.exists():
            model=make_model(name,target).to(dev).float()
            model.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['state'])
            models[name]=model.eval().requires_grad_(False)
    methods=['target','p_only']+list(models)
    records=torch.load(old.RECORDS,map_location='cpu',weights_only=True);first={}
    for r in records:first.setdefault(r['scene'],r)
    selected=sorted(first.items())[:1 if args.smoke else 3 if args.profile else len(first)]
    prefixes=(64,) if args.smoke else (64,256,1024)
    seeds=(54751,) if args.smoke or args.profile else (54751,54752)
    budget=32 if args.smoke else 256
    label='smoke' if args.smoke else 'profile' if args.profile else 'online'
    dest=OUT/f'b1_{label}'/str(args.training_seed);dest.mkdir(parents=True,exist_ok=True)
    def run(method,prompt,seed,n,profile=False):
        torch.cuda.synchronize();start=time.perf_counter()
        if method in ('target','p_only'):
            stats=p_decode(target,None,ranker if method=='p_only' else None,ranks,prompt,n,0 if method=='target' else 1,seed)
        else:
            stats,output=pf_decode(target,MLPAdapter(models[method]),ranker,ranks,prompt,n,seed,profile=profile)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        return dict(generated_tokens=int(stats['generated_tokens']),seconds=elapsed,
            stats={k:v for k,v in stats.items() if isinstance(v,(int,float,str,bool,list,dict))})
    prompt=[eos]+first[selected[0][0]]['tokens'][:64].tolist()
    for name in methods:run(name,prompt,55777,32)
    # Independent Feature head must be invariant to candidate Position.
    tests={}
    h=torch.randn(2,1024,device=dev);f=torch.tensor([0,1],device=dev);p=torch.tensor([1,2],device=dev)
    for name in ['b1_m','b1_h','b1_e']:
        if name in models:
            a=models[name](h,f,p)[1];b=models[name](h,f,p+10)[1]
            tests[name]=dict(candidate_change_max_abs=float((a-b).abs().max()),finite=bool(torch.isfinite(a).all()))
    if 'b1_m' in tests:assert tests['b1_m']['candidate_change_max_abs']==0
    for name in ['b1_h','b1_e']:
        if name in tests:assert tests[name]['candidate_change_max_abs']>0
    (dest/'interface_validation.json').write_text(json.dumps(tests,indent=2))
    rows=[]
    progress=dest/'records.jsonl'
    if progress.exists():rows=[json.loads(x) for x in progress.read_text().splitlines() if x.strip()]
    done={(r['scene'],r['prefix'],r['seed'],r['method']) for r in rows}
    total=sum(len(methods)*len(seeds) for _,r in selected for p in prefixes if len(r['tokens'])>=p)
    idx=0
    for si,(scene,r) in enumerate(selected):
        for prefix in prefixes:
            if len(r['tokens'])<prefix:continue
            prompt=[eos]+r['tokens'][:prefix].tolist();validate_sequence(prompt,[],ranks)
            for seed in seeds:
                order=methods[idx%len(methods):]+methods[:idx%len(methods)];idx+=1
                for m in order:
                    key=(scene,prefix,seed,m)
                    if key in done:continue
                    row=dict(scene=scene,prefix=prefix,seed=seed,method=m,training_seed=args.training_seed,
                        **run(m,prompt,seed+si*1009+prefix*17,budget,args.profile))
                    assert 0<=row['generated_tokens']<=budget+1
                    rows.append(row)
                    with progress.open('a') as stream:stream.write(json.dumps(row)+'\n')
                    print(json.dumps(dict(done=len(rows),total=total,method=m,scene=scene,prefix=prefix,tps=row['generated_tokens']/row['seconds'])),flush=True)
    summary=summarize(rows,methods)
    (dest/'summary.json').write_text(json.dumps(dict(status='complete',methods=methods,summary=summary,
        temperature=.9,top_p=.9,precision='FP32',budget=budget,model_load_and_warmup_timed=False,
        cache_allocation_and_prefill_timed=True,method_order='rotate per scene/prefix/run-seed',rows=len(rows)),indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
