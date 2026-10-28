"""Same data, optimizer and policy loss for four Feature mechanisms."""
import argparse
import collections
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

PROJECT = Path('/root/workspace/GaussianGPT')
OLD = Path('/root/autodl-tmp/gaussiangpt_next_round_20261017')
OUT = Path('/root/autodl-tmp/gaussiangpt_round2_20261017')
sys.path[:0] = [str(PROJECT), str(OLD), str(OUT)]
import bench_gaussiangpt_medusa_linear as medusa
from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock
from draft_a.pf_ranker_eval import typed_kinds
from collect_feature_teacher_from_s3 import policy_batch
from b1_feature_models import make_model


@torch.inference_mode()
def collect(target, device, eos):
    files = [PROJECT/'profiling/baselines_5way/b3_medusa_linear'/n/'train_token_traces.pt' for n in ('pipeline_train','train_32x1024')]
    traces = medusa.read_trace_files(files)
    assert len(traces) == 40
    perm = torch.randperm(len(traces), generator=torch.Generator().manual_seed(20261010)).tolist()
    val = set(perm[:4])
    capture = CaptureLastBlock(target)
    splits = {k:collections.defaultdict(list) for k in ('train','validation')}
    keys = collections.defaultdict(list)
    for i,(key,body) in enumerate(traces):
        split = 'validation' if i in val else 'train'
        full = torch.cat((torch.tensor([eos]), body.long().flatten())).to(device)
        logits, raw = capture.run(target,None,full[:-1].tolist(),typed_kinds(0,len(full)-1),device)
        anchor = torch.arange(1,len(full)-3,2,device=device)
        pos = full[anchor+2]
        prevf = full[anchor+1]-8000
        assert bool(((pos>=0)&(pos<8000)&(prevf>=0)&(prevf<4096)).all())
        d = splits[split]
        d['state'].append(raw[anchor].float().cpu())
        d['previous_feature'].append(prevf.cpu())
        d['position'].append(pos.cpu())
        d['teacher_probs'].append(torch.softmax(logits[anchor+2,8000:eos].float()/.9,-1).half().cpu())
        d['target_state'].append(F.rms_norm(raw[anchor+2].float(),(1024,)).cpu())
        keys[split].append(key)
        print(json.dumps(dict(event='collect',split=split,trace=key,rows=len(anchor))),flush=True)
    capture.close()
    return dict(splits={s:{k:torch.cat(v) for k,v in d.items()} for s,d in splits.items()},keys=dict(keys),
        state_input='pre-final-normalization output of last Target block after previous Position',
        teacher='Feature probabilities at temperature .9, before top-p', trace_count=len(traces))


def epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    totals = collections.Counter()
    for state,prevf,pos,p in loader:
        state,prevf,pos,p=[x.to(device) for x in (state,prevf,pos,p)]
        p=p.float();p=p/p.sum(-1,keepdim=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            _,logits=model(state,prevf,pos)
            loss=-(p*F.log_softmax(logits/.9,-1)).sum(-1).mean()
            if optimizer is not None:
                loss.backward();optimizer.step()
        with torch.no_grad():
            q=policy_batch(logits,.9,.9)
            # Apply the same top-p operation to the temperature-scaled teacher.
            pp=policy_batch(p.clamp_min(1e-30).log(),1.,.9)
            n=len(state);totals['n']+=n;totals['ce']+=float(loss)*n
            totals['overlap']+=float(torch.minimum(pp,q).sum())
    return dict(rows=totals['n'],ce=totals['ce']/totals['n'],overlap=totals['overlap']/totals['n'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=20261018)
    parser.add_argument('--methods',nargs='+',default=['b0','b1_m','b1_h','b1_e'])
    args=parser.parse_args();torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    target,dev,pv,eos,ranks,cache=medusa.target_loader(SimpleNamespace(gpt=PROJECT/'checkpoints/gpt_vfront.ckpt',vqvae=PROJECT/'checkpoints/vqvae_vfront.ckpt'))
    path=OUT/'B1_shared_data.pt'
    if path.exists():data=torch.load(path,map_location='cpu',weights_only=False)
    else:
        data=collect(target,dev,eos);torch.save(data,path)
    summaries={}
    for name in args.methods:
        dest=OUT/'b1_training'/str(args.seed)/name;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'summary.json').exists():
            summaries[name]=json.loads((dest/'summary.json').read_text());continue
        torch.manual_seed(args.seed)
        model=make_model(name,target).to(dev).float()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        def loader(split,shuffle):
            d=data['splits'][split]
            return DataLoader(TensorDataset(*[d[k] for k in ('state','previous_feature','position','teacher_probs')]),batch_size=128,shuffle=shuffle,
                generator=torch.Generator().manual_seed(args.seed) if shuffle else None)
        tr,va=loader('train',True),loader('validation',False)
        best=-1.;history=[];beg=time.perf_counter()
        for ep in range(1,11):
            a=epoch(model,tr,dev,opt);b=epoch(model,va,dev)
            row=dict(epoch=ep,train=a,validation=b,seconds=time.perf_counter()-beg);history.append(row)
            print(json.dumps(dict(event='epoch',method=name,seed=args.seed,**row)),flush=True)
            if b['overlap']>best:
                best=b['overlap'];torch.save(dict(state=model.state_dict(),method=name,seed=args.seed,epoch=ep,validation=b),dest/'selected.pt')
            (dest/'history.json').write_text(json.dumps(history,indent=2))
        torch.cuda.synchronize()
        result=dict(method=name,seed=args.seed,parameters=sum(p.numel() for p in model.parameters()),
            shared_fixed_parameters=sum(v.numel() for v in model.buffers()),seconds=time.perf_counter()-beg,
            best=torch.load(dest/'selected.pt',map_location='cpu',weights_only=False)['validation'],history=history,
            objective='same temperature-.9 teacher distribution cross entropy; no auxiliary hidden-state loss',
            optimizer='AdamW lr=1e-4 wd=.01',epochs=10,train_rows=len(tr.dataset),validation_rows=len(va.dataset),
            train_keys=data['keys']['train'],validation_keys=data['keys']['validation'])
        (dest/'summary.json').write_text(json.dumps(result,indent=2));summaries[name]=result
        del model,opt
        torch.cuda.empty_cache()
    (OUT/f'B1_training_summary_{args.seed}.json').write_text(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
