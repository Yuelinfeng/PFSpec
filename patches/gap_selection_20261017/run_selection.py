"""G1/G2 runner using existing development crops and unchanged model weights."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from joblib import load

ROOT=Path('/root/autodl-tmp/gaussiangpt_gap_selection_20261017')
PROJECT=Path('/root/workspace/GaussianGPT')
sys.path[:0]=[str(PROJECT),'/root/autodl-tmp/gaussiangpt_next_round_20261017',
             '/root/autodl-tmp/gaussiangpt_candidate_cost_20261017']
import step5_method_compare as prior
import bench_gaussiangpt_medusa_linear as medusa
from bench_gaussiangpt_pf_block2_cow import load_feature_model
from draft_a.pf_feature_unified import decode as original_decode,MLPAdapter
from gap_decoder import build_decoder,run,Selector
from validate_selection import validate,position_with_output
from run_cost_study import environment,write_json,append


def microseconds():
    values={}
    for mode in ('pf','p_only','gap','random'):
        selector=Selector(mode,.8,13579)
        count=200000
        start=time.perf_counter()
        for i in range(count):
            selector.choose((101 if i%2 else 120,4,0),100,256)
        values[mode]=(time.perf_counter()-start)/count*1e6
    return values


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    dest=ROOT/('smoke' if args.smoke else 'full');dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/'environment_start.json',environment())
    fn=build_decoder(ROOT/'selected_decode.py')
    target,device,pv,eos,ranks,_=medusa.target_loader(SimpleNamespace(
        gpt=PROJECT/'checkpoints/gpt_vfront.ckpt',vqvae=PROJECT/'checkpoints/vqvae_vfront.ckpt'))
    ranker=prior.FastHistRanker(load(prior.RANKER))
    feature=load_feature_model(prior.FEATURE,prior.FEATURE_HEAD,target,device,pv,eos)[0].float().eval().requires_grad_(False)
    source=torch.load(prior.RECORDS,map_location='cpu',weights_only=True)
    unique=[]
    for i,row in enumerate(source):
        tokens=row['tokens'].tolist()
        if any(scene==row['scene'] and old==tokens for scene,old,_ in unique):continue
        unique.append((row['scene'],tokens,i))
    scenes=sorted({r[0] for r in unique});scene_index={s:i for i,s in enumerate(scenes)}
    prompts=[]
    for crop_index,(scene,tokens,record_index) in enumerate(unique):
        for prefix in (64,256,1024):
            if len(tokens)<prefix:continue
            prompts.append(dict(scene=scene,scene_index=scene_index[scene],crop_index=crop_index,
                record_index=record_index,prefix=prefix,prompt=[eos]+tokens[:prefix]))
    if args.smoke:prompts=prompts[:2]
    write_json(dest/'inputs.json',dict(crops=len(unique),scenes=scenes,prompts=prompts,
        rule='candidate_position_id - previous_position_id == 1',
        calibration_seed=86901,measurement_seeds=[88101,88102,88103],
        budget=32 if args.smoke else 256,methods=['p_only','pf','gap','random'],
        random_probability_source='separate development calibration before online comparison'))
    reference_p=position_with_output()
    p=prompts[0]['prompt']
    for mode in ('pf','p_only','gap','random'):
        run(fn,target,feature,ranker,ranks,p,32,85109,mode=mode,probability=.8)
    if not args.smoke and not (dest/'g1.json').exists():
        contexts=json.loads(Path('/root/autodl-tmp/gaussiangpt_candidate_cost_20261017/full/matched_contexts.json').read_text())
        write_json(dest/'progress.json',dict(stage='G1',status='running'))
        result=validate(fn,target,feature,ranker,ranks,contexts)
        write_json(dest/'g1.json',result)
        print('G1 COMPLETE',flush=True)
    if args.smoke:
        for mode in ('pf','p_only'):
            a,ao=run(fn,target,feature,ranker,ranks,p,32,85109,mode=mode)
            if mode=='pf':b,bo=original_decode(target,MLPAdapter(feature),ranker,ranks,p,32,85109)
            else:b,bo=reference_p(target,None,ranker,ranks,p,32,1,85109)
            assert ao==bo
    cal_path=dest/'calibration.jsonl'
    calibration=[json.loads(s) for s in cal_path.read_text().splitlines()] if cal_path.exists() else []
    done={r['prompt_id'] for r in calibration}
    budget=32 if args.smoke else 256
    for ci,item in enumerate(prompts):
        if ci in done:continue
        seed=86901+ci*1009
        stats,output=run(fn,target,feature,ranker,ranks,item['prompt'],budget,seed,mode='gap')
        row=dict(prompt_id=ci,scene=item['scene'],stats=stats)
        calibration.append(row);append(cal_path,row)
        write_json(dest/'progress.json',dict(stage='calibration',done=len(calibration),total=len(prompts)))
        if ci%8==0:print('calibration',len(calibration),len(prompts),flush=True)
    eligible=sum(r['stats']['selection']['counts'].get('eligible',0) for r in calibration)
    requested=sum(r['stats']['selection']['counts'].get('requested_pf',0) for r in calibration)
    probability=requested/eligible
    protocol=dict(probability=probability,eligible_rounds=eligible,requested_pf=requested,
        seed=86901,selected_before_comparison=True,rule='unit ID gap',
        selector_microseconds=microseconds())
    if (dest/'random_probability.json').exists():
        prior_protocol=json.loads((dest/'random_probability.json').read_text())
        assert prior_protocol['probability']==probability
    else:write_json(dest/'random_probability.json',protocol)
    path=dest/'runs.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done={(r['prompt_id'],r['seed'],r['method']) for r in rows}
    methods=['p_only','pf','gap','random'];seeds=[88101] if args.smoke else [88101,88102,88103]
    total=len(prompts)*len(seeds)*len(methods)
    for ci,item in enumerate(prompts):
        for si,run_seed in enumerate(seeds):
            seed=run_seed+ci*1009
            rotation=(ci*len(seeds)+si)%4
            order=methods[rotation:]+methods[:rotation]
            if (ci+si)%2:order=list(reversed(order))
            for method in order:
                if (ci,run_seed,method) in done:continue
                torch.cuda.synchronize();started=time.perf_counter()
                if method=='pf':
                    stats,output=original_decode(target,MLPAdapter(feature),ranker,ranks,item['prompt'],budget,seed)
                elif method=='p_only':
                    stats,output=reference_p(target,None,ranker,ranks,item['prompt'],budget,1,seed)
                else:
                    stats,output=run(fn,target,feature,ranker,ranks,item['prompt'],budget,seed,
                                     mode=method,probability=probability)
                torch.cuda.synchronize();elapsed=time.perf_counter()-started
                row=dict(prompt_id=ci,scene=item['scene'],crop_index=item['crop_index'],prefix=item['prefix'],
                    seed=run_seed,actual_seed=seed,method=method,seconds=elapsed,
                    generated_tokens=stats['generated_tokens'],stats=stats,output=output)
                rows.append(row);append(path,row)
                write_json(dest/'progress.json',dict(stage='G2',done=len(rows),total=total))
                if len(rows)%8==0:print('G2',len(rows),total,flush=True)
    write_json(dest/'environment_end.json',environment())
    write_json(dest/'progress.json',dict(status='complete',stage='G1_G2',runs=len(rows),
        prompts=len(prompts),scenes=len(scenes),crops=len(unique),random_probability=probability,
        no_new_neural_models=True,g3_started=False))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
