"""Test whether a known historical Feature helps an independent P1 head."""
import collections
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT=Path('/root/workspace/GaussianGPT')
OLD=Path('/root/autodl-tmp/gaussiangpt_next_round_20261017')
OUT=Path('/root/autodl-tmp/gaussiangpt_round2_20261017')
sys.path[:0]=[str(PROJECT),str(OLD),str(OUT)]
import bench_gaussiangpt_medusa_linear as medusa


class PositionHead(nn.Module):
    def __init__(self, target, pending_feature):
        super().__init__()
        self.pending_feature=pending_feature
        self.register_buffer('weight',torch.cat((target.lm_head.weight[:8000],target.lm_head.weight[12096:12097])).detach(),persistent=False)
        self.embedding=nn.Embedding(4096,128) if pending_feature else None
        self.net=nn.Sequential(nn.Linear(1152 if pending_feature else 1024,466,bias=False),nn.SiLU(),nn.Linear(466,1024,bias=False))
        nn.init.zeros_(self.net[-1].weight)

    def forward(self,state,previous_feature,previous_position=None):
        h=F.rms_norm(state.float(),(1024,))
        x=torch.cat((h,self.embedding(previous_feature)),dim=-1) if self.pending_feature else h
        raw=F.linear(h+self.net(x),self.weight)
        logits=15.*torch.tanh(raw.float()/15.)
        if previous_position is not None:
            logits[:,:8000]=logits[:,:8000].masked_fill(torch.arange(8000,device=h.device)[None]<=previous_position[:,None],-torch.inf)
        return logits


def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    target,dev,pv,eos,ranks,cache=medusa.target_loader(SimpleNamespace(gpt=PROJECT/'checkpoints/gpt_vfront.ckpt',vqvae=PROJECT/'checkpoints/vqvae_vfront.ckpt'))
    data=torch.load(OUT/'B1_shared_data.pt',map_location='cpu',weights_only=False)
    files=[PROJECT/'profiling/baselines_5way/b3_medusa_linear'/n/'train_token_traces.pt' for n in ('pipeline_train','train_32x1024')]
    trace_items=medusa.read_trace_files(files)
    from train_hydra_shared_position import _load_traces
    from gaussiangpt_medusa_conditional_v3.medusa_conditional_heads import TokenSpec
    hydra_sequences=_load_traces([str(p) for p in files],TokenSpec())
    medusa_sequences=[[eos]+v.long().flatten().tolist() for k,v in trace_items]
    order_equal=hydra_sequences==medusa_sequences
    split_fact=dict(trace_count=len(trace_items),hydra_trace_count=len(hydra_sequences),
        sequence_order_identical=order_equal,shared_split_seed=20261010,
        medusa_validation_keys=data['keys']['validation'],complete_trajectory_split_equal=order_equal)
    (OUT/'A0_trace_split_alignment.json').write_text(json.dumps(split_fact,indent=2))
    traces=dict(trace_items);splits={}
    for split,d in data['splits'].items():
        prev=torch.cat([traces[k].long().flatten()[:-2:2] for k in data['keys'][split]])
        truth=torch.cat([traces[k].long().flatten()[2::2] for k in data['keys'][split]])
        assert torch.equal(truth,d['position'])
        splits[split]=TensorDataset(d['state'],d['previous_feature'],prev,truth)
    report={}
    for pending in (False,True):
        name='with_pending_feature' if pending else 'position_state_only'
        dest=OUT/'a2_training'/name;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'summary.json').exists():
            report[name]=json.loads((dest/'summary.json').read_text());continue
        torch.manual_seed(20261018);model=PositionHead(target,pending).to(dev)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        loaders={s:DataLoader(d,batch_size=128,shuffle=s=='train',generator=torch.Generator().manual_seed(20261018)) for s,d in splits.items()}
        best=-1.;history=[];start=time.perf_counter()
        for ep in range(1,11):
            stats={}
            for split,loader in loaders.items():
                model.train(split=='train');count=collections.Counter()
                for h,f,prev,y in loader:
                    h,f,prev,y=[v.to(dev) for v in (h,f,prev,y)]
                    if split=='train':opt.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(split=='train'):
                        logits=model(h,f,prev);loss=F.cross_entropy(logits,y)
                        if split=='train':loss.backward();opt.step()
                    count['n']+=len(y);count['loss']+=float(loss.detach())*len(y)
                    count['top1']+=int((logits.argmax(-1)==y).sum())
                stats[split]=dict(rows=count['n'],ce=count['loss']/count['n'],top1=count['top1']/count['n'])
            row=dict(epoch=ep,**stats,seconds=time.perf_counter()-start);history.append(row)
            if stats['validation']['top1']>best:
                best=stats['validation']['top1'];torch.save(dict(state=model.state_dict(),pending_feature=pending,seed=20261018,epoch=ep),dest/'selected.pt')
            print(json.dumps(dict(method=name,**row)),flush=True)
        result=dict(method=name,parameters=sum(p.numel() for p in model.parameters()),seed=20261018,
            epochs=10,best_validation_top1=best,history=history,seconds=time.perf_counter()-start,
            objective='observed next Position CE with legal rank support',training_keys=data['keys'])
        (dest/'summary.json').write_text(json.dumps(result,indent=2));report[name]=result
        del model,opt
    (OUT/'A2_training_summary.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
