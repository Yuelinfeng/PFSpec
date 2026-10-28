"""Evaluate historical-Feature P1 heads, then measure a small full-path comparison."""
import collections
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import torch

PROJECT=Path('/root/workspace/GaussianGPT')
OLD=Path('/root/autodl-tmp/gaussiangpt_next_round_20261017')
OUT=Path('/root/autodl-tmp/gaussiangpt_round2_20261017')
sys.path[:0]=[str(PROJECT),str(OLD),str(OUT)]
import step5_method_compare as old
import bench_gaussiangpt_medusa_linear as medusa
from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock
from draft_a.pf_ranker_eval import typed_kinds, decode as p_decode
from a2_pending_feature import PositionHead
from a01_diagnostics import metric


class CombinedHeads:
    def __init__(self, original, p1):
        self.original=original
        self.p1=p1

    def raw(self,h,pending_f):
        prevf=torch.tensor([pending_f-8000],device=h.device)
        result=[self.p1(h,prevf)]
        for j in (1,2):
            z=h+self.original.heads[j](h)
            w=self.original.lm_weight[8000:12096] if j==1 else torch.cat((self.original.lm_weight[:8000],self.original.lm_weight[12096:12097]))
            v=torch.nn.functional.linear(z,w).float()
            result.append(15.*torch.tanh(v/15.))
        return result

    def distribution(self,*args,**kwargs):
        return self.original.distribution(*args,**kwargs)


@torch.inference_mode()
def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    target,dev,pv,eos,ranks,make_cache=medusa.target_loader(SimpleNamespace(gpt=PROJECT/'checkpoints/gpt_vfront.ckpt',vqvae=PROJECT/'checkpoints/vqvae_vfront.ckpt'))
    original=old.load_medusa_heads(old.LOWRANK,target,pv,eos)
    models={}
    for name in ('position_state_only','with_pending_feature'):
        m=PositionHead(target,name=='with_pending_feature').to(dev)
        m.load_state_dict(torch.load(OUT/'a2_training'/name/'selected.pt',map_location='cpu',weights_only=False)['state'])
        models[name]=m.eval().requires_grad_(False)
    records=torch.load(old.RECORDS,map_location='cpu',weights_only=True);first={}
    for r in records:first.setdefault(r['scene'],r)
    capture=CaptureLastBlock(target);rows=[]
    for scene,r in sorted(first.items()):
        for prefix in (64,256,1024):
            body=r['tokens'].long().flatten()
            if len(body)<=prefix+1:continue
            full=torch.cat((torch.tensor([eos]),body[:prefix+2])).to(dev)
            logits,states=capture.run(target,None,full.tolist(),typed_kinds(0,len(full)),dev)
            for t in range(max(1,prefix-31),prefix,2):
                prev,pending,truth=map(int,full[t:t+3])
                p_logits=logits[t+1].clone();p_logits[:pv].masked_fill_(ranks.to(dev)<=ranks[prev],-torch.inf)
                p=medusa.policy(p_logits,.9,.9)
                result={}
                for name,m in models.items():
                    raw=m(states[t:t+1],torch.tensor([pending-pv],device=dev))[0]
                    q=original.distribution(raw,'position',prev_p=prev,ranks=ranks,temp=.9,top_p=.9)
                    result[name]=metric(p,q,truth,prev,ranks,eos)
                    if name=='with_pending_feature':
                        raw=m(states[t:t+1],torch.tensor([(pending-pv+113)%4096],device=dev))[0]
                        q=original.distribution(raw,'position',prev_p=prev,ranks=ranks,temp=.9,top_p=.9)
                        result['changed_historical_feature']=metric(p,q,truth,prev,ranks,eos)
                rows.append(dict(scene=scene,prefix=prefix,anchor=t,methods=result))
    capture.close()
    summary={name:dict(count=len(rows),overlap=sum(r['methods'][name]['overlap'] for r in rows)/len(rows),
        top1_hit=sum(r['methods'][name]['top1_hit'] for r in rows)/len(rows)) for name in rows[0]['methods']}
    (OUT/'A2_position_diagnostics.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2))
    source=inspect.getsource(old.decode_heads).replace('def decode_heads(', 'def decode_pending_heads(')
    needle='raw=heads.raw(hprev.unsqueeze(0));'
    assert source.count(needle)==1
    source=source.replace(needle,'raw=heads.raw(hprev.unsqueeze(0), pending_f);')
    namespace=old.__dict__.copy();exec(compile(source,'a2_exact_decode','exec'),namespace)
    decode=namespace['decode_pending_heads']
    methods=['target','original_medusa']+list(models)
    def run(name,prompt,budget,seed):
        torch.cuda.synchronize();start=time.perf_counter()
        if name=='target':
            stats=p_decode(target,None,None,ranks,prompt,budget,0,seed)
            generated=int(stats['generated_tokens'])
        else:
            func=old.decode_heads if name=='original_medusa' else decode
            heads=original if name=='original_medusa' else CombinedHeads(original,models[name])
            output,stats=func(target,heads,'independent',make_cache,eos=eos,p_vocab=pv,ranks=ranks,
                prompt=prompt,budget=budget,seed=seed,device=dev)
            generated=len(output)-(len(prompt)-1)+int(bool(stats.get('eos',0)))
        torch.cuda.synchronize()
        return dict(generated_tokens=generated,seconds=time.perf_counter()-start,stats=dict(stats))
    prompt=[eos]+first[sorted(first)[0]]['tokens'][:64].tolist()
    for name in methods:run(name,prompt,32,55991)
    measured=[]
    for i,(scene,r) in enumerate(sorted(first.items())[:3]):
        prompt=[eos]+r['tokens'][:64].tolist()
        order=methods[i:]+methods[:i]
        for name in order:
            row=dict(scene=scene,method=name,prefix=64,seed=54751,**run(name,prompt,256,54751+i*1009+64*17))
            measured.append(row);print(json.dumps(row),flush=True)
    (OUT/'A2_small_online.json').write_text(json.dumps(dict(scope='3 development sources, one prefix and run seed, 256-token cap; screen only',
        methods=methods,rows=measured,offline_summary=summary),indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
