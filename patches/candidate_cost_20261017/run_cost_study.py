"""A1/A2/A3: branch timing, P2 repair cost, and same-prefix action comparisons."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
from joblib import load

PROJECT = Path('/root/workspace/GaussianGPT')
OLD = Path('/root/autodl-tmp/gaussiangpt_next_round_20261017')
ROOT = Path('/root/autodl-tmp/gaussiangpt_candidate_cost_20261017')
sys.path[:0] = [str(PROJECT), str(OLD)]
import step5_method_compare as prior
import bench_gaussiangpt_medusa_linear as medusa
from draft_a.pf_feature_unified import decode as original_decode, MLPAdapter
from draft_a.pf_ranker_eval import decode as position_decode
from bench_gaussiangpt_pf_block2_cow import load_feature_model
from branch_probe import build_decoder, run_observed


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row)+'\n')


def environment():
    def output(cmd):
        return subprocess.run(cmd, cwd=PROJECT, capture_output=True, text=True).stdout
    assets = [prior.FEATURE, prior.RANKER, PROJECT/'checkpoints/gpt_vfront.ckpt',
              PROJECT/'checkpoints/vqvae_vfront.ckpt', prior.RECORDS,
              PROJECT/'draft_a/pf_feature_unified.py', PROJECT/'draft_a/pf_ranker_eval.py']
    return dict(time_unix=time.time(), pid=os.getpid(), python=sys.version,
        torch=torch.__version__, cuda=torch.version.cuda,
        git_head=output(['git','rev-parse','HEAD']).strip(),
        git_status=output(['git','status','--short']),
        gpu=output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader']),
        assets=[dict(path=str(p),bytes=p.stat().st_size,mtime=p.stat().st_mtime) for p in assets])


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    dest=ROOT/('smoke' if args.smoke else 'full');dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/'environment_start.json',environment())
    fn=build_decoder(ROOT/'instrumented_decode.py')
    target,device,pv,eos,ranks,_=medusa.target_loader(SimpleNamespace(
        gpt=PROJECT/'checkpoints/gpt_vfront.ckpt',vqvae=PROJECT/'checkpoints/vqvae_vfront.ckpt'))
    ranker=prior.FastHistRanker(load(prior.RANKER))
    feature=load_feature_model(prior.FEATURE,prior.FEATURE_HEAD,target,device,pv,eos)[0].float().eval().requires_grad_(False)
    records=torch.load(prior.RECORDS,map_location='cpu',weights_only=True)
    first={}
    for row in records:first.setdefault(row['scene'],row)
    prompts=[]
    for si,(scene,row) in enumerate(sorted(first.items())[:1 if args.smoke else None]):
        for prefix in ((64,) if args.smoke else (64,256,1024)):
            if len(row['tokens'])>=prefix:
                prompts.append(dict(scene=scene,prefix=prefix,index=si,
                    prompt=[eos]+row['tokens'][:prefix].tolist()))
    seeds=(54751,) if args.smoke else (54751,54752)
    budget=32 if args.smoke else 256
    prompt=prompts[0]['prompt']
    original_decode(target,MLPAdapter(feature),ranker,ranks,prompt,32,55777)
    position_decode(target,None,ranker,ranks,prompt,32,1,55777)
    run_observed(fn,target,feature,ranker,ranks,prompt,32,55777)
    validations=[]
    for mode in ('pf','p_only'):
        a,ao=run_observed(fn,target,feature,ranker,ranks,prompt,32,55777,mode=mode,profile=False)
        b,bo=run_observed(fn,target,feature,ranker,ranks,prompt,32,55777,mode=mode)
        fields=['target_calls','generated_tokens','rounds','position_accepted','position_verified','feature_accepted','feature_verified']
        item=dict(mode=mode,outputs_equal=ao==bo,counts_equal=all(a[k]==b[k] for k in fields))
        if mode=='pf':
            c,co=original_decode(target,MLPAdapter(feature),ranker,ranks,prompt,32,55777)
            item.update(original_outputs_equal=ao==co,original_counts_equal=all(a[k]==c[k] for k in fields))
        else:
            c=position_decode(target,None,ranker,ranks,prompt,32,1,55777)
            item['position_decoder_counts_equal']=all(a[k]==c[k] for k in fields)
        assert all(v for k,v in item.items() if k!='mode'),item
        validations.append(item)
    last=max(range(pv),key=lambda token:ranks[token])
    terminal=[eos,last,pv]
    a,ao=original_decode(target,MLPAdapter(feature),ranker,ranks,terminal,8,55779)
    b,bo=run_observed(fn,target,feature,ranker,ranks,terminal,8,55779)
    assert ao==bo==[eos] and a['eos'] and b['eos']
    validations.append(dict(mode='terminal',outputs_equal=True,natural_eos=True))
    write_json(dest/'sampler_validation.json',validations)
    path=dest/'runs.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done={(r['scene'],r['prefix'],r['seed'],r['method']) for r in rows}
    methods=['pf_plain','p_only_plain','pf_profile']
    total=len(prompts)*len(seeds)*len(methods)
    iteration=0
    for context in prompts:
        for run_seed in seeds:
            seed=run_seed+context['index']*1009+context['prefix']*17
            order=methods[iteration%3:]+methods[:iteration%3];iteration+=1
            for method in order:
                key=(context['scene'],context['prefix'],run_seed,method)
                if key in done:continue
                torch.cuda.synchronize();begin=time.perf_counter()
                if method=='pf_plain':
                    stats,output=original_decode(target,MLPAdapter(feature),ranker,ranks,context['prompt'],budget,seed)
                elif method=='p_only_plain':
                    stats=position_decode(target,None,ranker,ranks,context['prompt'],budget,1,seed);output=None
                else:
                    stats,output=run_observed(fn,target,feature,ranker,ranks,context['prompt'],budget,seed)
                torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                row=dict(scene=context['scene'],prefix=context['prefix'],seed=run_seed,
                    actual_seed=seed,method=method,seconds=elapsed,
                    generated_tokens=stats['generated_tokens'],stats=stats,output=output)
                if method=='pf_profile':
                    other=next((r for r in rows if r['scene']==row['scene'] and r['prefix']==row['prefix'] and r['seed']==run_seed and r['method']=='pf_plain'),None)
                    if other:assert other['output']==output
                rows.append(row);append(path,row)
                write_json(dest/'progress.json',dict(stage='online',done=len(rows),total=total))
                print(json.dumps(dict(stage='online',done=len(rows),total=total,method=method)),flush=True)
    for context in prompts:
        for seed in seeds:
            selected=[r for r in rows if r['scene']==context['scene'] and r['prefix']==context['prefix'] and r['seed']==seed]
            raw=next(r for r in selected if r['method']=='pf_plain')
            prof=next(r for r in selected if r['method']=='pf_profile')
            assert raw['output']==prof['output']
            rr=prof['stats']['round_profile']['rounds']
            assert len(rr)==prof['stats']['rounds']
            assert sum(r['generated_tokens'] for r in rr)==prof['generated_tokens']
            assert sum(r['counts']['target_calls'] for r in rr)+1==prof['stats']['target_calls']
    contexts=[]
    for item in prompts:
        contexts.append(dict(**item,origin='initial'))
        row=next(r for r in rows if r['method']=='pf_profile' and r['scene']==item['scene'] and r['prefix']==item['prefix'] and r['seed']==seeds[0])
        rr=row['stats']['round_profile']['rounds'];middle=rr[len(rr)//2]
        contexts.append(dict(scene=item['scene'],prefix=item['prefix'],index=item['index'],
            prompt=item['prompt']+row['output'][:middle['output_start']],origin='pf_middle'))
    write_json(dest/'matched_contexts.json',contexts)
    pair_path=dest/'matched_rounds.jsonl'
    pair_rows=[json.loads(s) for s in pair_path.read_text().splitlines()] if pair_path.exists() else []
    pair_done={(r['context_id'],r['trial'],r['mode']) for r in pair_rows}
    trials=2 if args.smoke else 8
    for ci,context in enumerate(contexts):
        for trial in range(trials):
            seed=651001+ci*1009+trial*17
            modes=('pf','p_only') if (ci+trial)%2==0 else ('p_only','pf')
            for mode in modes:
                if (ci,trial,mode) in pair_done:continue
                stats,output=run_observed(fn,target,feature,ranker,ranks,context['prompt'],8,seed,mode=mode,max_rounds=1,detail=False)
                rr=stats['round_profile']['rounds'];assert len(rr)==1
                row=dict(context_id=ci,scene=context['scene'],prefix=context['prefix'],origin=context['origin'],
                    body_tokens=len(context['prompt'])-1,trial=trial,seed=seed,mode=mode,
                    generated_tokens=len(output),output=output,round=rr[0])
                pair_rows.append(row);append(pair_path,row)
        write_json(dest/'progress.json',dict(stage='matched_rounds',done=len(pair_rows),total=len(contexts)*trials*2))
        print(json.dumps(dict(stage='matched_rounds',context=ci+1,total=len(contexts))),flush=True)
    sync_path=dest/'sync_p2.jsonl'
    sync_rows=[json.loads(s) for s in sync_path.read_text().splitlines()] if sync_path.exists() else []
    sync_done={(r['scene'],r['prefix']) for r in sync_rows}
    for context in [c for c in prompts if c['index']<3]:
        if (context['scene'],context['prefix']) in sync_done:continue
        seed=seeds[0]+context['index']*1009+context['prefix']*17
        stats,output=run_observed(fn,target,feature,ranker,ranks,context['prompt'],budget,seed,sync_p2=True)
        raw=next(r for r in rows if r['method']=='pf_plain' and r['scene']==context['scene'] and r['prefix']==context['prefix'] and r['seed']==seeds[0])
        assert raw['output']==output
        append(sync_path,dict(scene=context['scene'],prefix=context['prefix'],stats=stats))
    write_json(dest/'environment_end.json',environment())
    write_json(dest/'progress.json',dict(stage='A1_A2_A3',status='complete',runs=len(rows),
        paired_rounds=len(pair_rows),contexts=len(contexts),temperature=.9,top_p=.9,
        precision='FP32',no_new_training=True))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
