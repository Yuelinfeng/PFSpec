"""Compare trained seeds on identical real Feature contexts, outside speed runs."""
import collections
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

PROJECT=Path('/root/workspace/GaussianGPT')
OLD=Path('/root/autodl-tmp/gaussiangpt_next_round_20261017')
OUT=Path('/root/autodl-tmp/gaussiangpt_round2_20261017')
sys.path[:0]=[str(PROJECT),str(OLD),str(OUT)]
import step5_method_compare as old
import bench_gaussiangpt_medusa_linear as medusa
from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock
from draft_a.pf_ranker_eval import typed_kinds
from collect_feature_teacher_from_s3 import policy_batch
from b1_feature_models import make_model


@torch.inference_mode()
def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    target,dev,pv,eos,ranks,cache=medusa.target_loader(SimpleNamespace(gpt=PROJECT/'checkpoints/gpt_vfront.ckpt',vqvae=PROJECT/'checkpoints/vqvae_vfront.ckpt'))
    path=OUT/'B1_real_fixed_contexts.pt'
    if path.exists():data=torch.load(path,map_location='cpu',weights_only=False)
    else:
        records=torch.load(old.RECORDS,map_location='cpu',weights_only=True);first={}
        for r in records:first.setdefault(r['scene'],r)
        capture=CaptureLastBlock(target);cols=collections.defaultdict(list);contexts=[]
        for scene,r in sorted(first.items()):
            body=r['tokens'].long().flatten()
            for prefix in (64,256,1024):
                if len(body)<=prefix+2:continue
                full=torch.cat((torch.tensor([eos]),body[:prefix+4])).to(dev)
                logits,states=capture.run(target,None,full.tolist(),typed_kinds(0,len(full)),dev)
                anchors=torch.arange(max(1,prefix-31),prefix,2,device=dev)
                cols['state'].append(states[anchors].cpu())
                cols['previous_feature'].append((full[anchors+1]-pv).cpu())
                cols['position'].append(full[anchors+2].cpu())
                cols['true_feature'].append((full[anchors+3]-pv).cpu())
                cols['teacher_probs'].append(policy_batch(logits[anchors+2,pv:eos],.9,.9).cpu())
                contexts.extend(dict(scene=scene,prefix=prefix,anchor=int(t)) for t in anchors)
        capture.close()
        data=dict(columns={k:torch.cat(v) for k,v in cols.items()},contexts=contexts)
        torch.save(data,path)
    d=data['columns'];rows=[];summary={}
    for seed in (20261018,20261019,20261020):
        for name in ('b0','b1_m','b1_h','b1_e'):
            p=OUT/'b1_training'/str(seed)/name/'selected.pt'
            if not p.exists():continue
            m=make_model(name,target).to(dev).float()
            m.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['state']);m.eval()
            _,logits=m(*[d[k].to(dev) for k in ('state','previous_feature','position')])
            q=policy_batch(logits,.9,.9).cpu();teacher=d['teacher_probs']
            overlap=torch.minimum(q,teacher).sum(-1)
            prob=q[torch.arange(len(q)),d['true_feature']]
            key=f'{seed}:{name}';by_source=collections.defaultdict(list)
            for i,c in enumerate(data['contexts']):
                row=dict(**c,seed=seed,method=name,overlap=float(overlap[i]),q_true=float(prob[i]))
                rows.append(row);by_source[c['scene']].append(row['overlap'])
            source_means={s:sum(v)/len(v) for s,v in by_source.items()}
            summary[key]=dict(rows=len(q),mean_overlap=float(overlap.mean()),source_equal_overlap=sum(source_means.values())/len(source_means),source_means=source_means)
            del m
    (OUT/'B1_fixed_contexts.json').write_text(json.dumps(dict(scope='real teacher-forced Feature contexts with identical actual candidate Position; not online accepted prefixes',summary=summary,rows=rows),indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
