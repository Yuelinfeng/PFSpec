"""Measure complete P2 repair spans with round endpoints only."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from joblib import load

ROOT=Path('/root/autodl-tmp/gaussiangpt_candidate_cost_20261017')
sys.path[:0]=['/root/workspace/GaussianGPT','/root/autodl-tmp/gaussiangpt_next_round_20261017']
import step5_method_compare as prior
import bench_gaussiangpt_medusa_linear as medusa
from bench_gaussiangpt_pf_block2_cow import load_feature_model
from draft_a.pf_feature_unified import decode,MLPAdapter
from branch_probe import build_decoder,run_observed


@torch.inference_mode()
def main():
    dest=ROOT/'full'
    assert json.loads((dest/'progress.json').read_text()).get('status')=='complete'
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    fn=build_decoder(ROOT/'instrumented_decode.py')
    project=Path('/root/workspace/GaussianGPT')
    target,device,pv,eos,ranks,_=medusa.target_loader(SimpleNamespace(
        gpt=project/'checkpoints/gpt_vfront.ckpt',vqvae=project/'checkpoints/vqvae_vfront.ckpt'))
    ranker=prior.FastHistRanker(load(prior.RANKER))
    feature=load_feature_model(prior.FEATURE,prior.FEATURE_HEAD,target,device,pv,eos)[0].float().eval().requires_grad_(False)
    contexts=json.loads((dest/'matched_contexts.json').read_text())
    contexts={(c['scene'],c['prefix']):c['prompt'] for c in contexts if c['origin']=='initial'}
    previous=[json.loads(line) for line in (dest/'runs.jsonl').read_text().splitlines()]
    previous=[r for r in previous if r['method']=='pf_plain']
    initial=next(iter(contexts.values()))
    decode(target,MLPAdapter(feature),ranker,ranks,initial,32,55777)
    run_observed(fn,target,feature,ranker,ranks,initial,32,55777,detail=False)
    path=dest/'repair_light.jsonl'
    existing=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done={(r['scene'],r['prefix'],r['seed'],r['method']) for r in existing}
    for i,old in enumerate(previous):
        prompt=contexts[(old['scene'],old['prefix'])]
        methods=['plain','regions'] if i<8 else ['regions']
        if i%2:methods.reverse()
        for mode in methods:
            key=(old['scene'],old['prefix'],old['seed'],mode)
            if key in done:continue
            torch.cuda.synchronize();start=time.perf_counter()
            if mode=='plain':
                stats,output=decode(target,MLPAdapter(feature),ranker,ranks,prompt,256,old['actual_seed'])
            else:
                stats,output=run_observed(fn,target,feature,ranker,ranks,prompt,256,old['actual_seed'],detail=False)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            assert output==old['output']
            row=dict(scene=old['scene'],prefix=old['prefix'],seed=old['seed'],method=mode,
                seconds=elapsed,generated_tokens=len(output),stats=stats)
            with path.open('a') as stream:stream.write(json.dumps(row)+'\n')
        (dest/'repair_light_progress.json').write_text(json.dumps(dict(done=i+1,total=len(previous))))
        print(json.dumps(dict(stage='repair_light',done=i+1,total=len(previous))),flush=True)
    (dest/'repair_light_progress.json').write_text(json.dumps(dict(done=len(previous),total=len(previous),status='complete')))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
