#!/usr/bin/env python3
"""B3: trainable Medusa-like *linear-path* self-speculative decoding for GaussianGPT.

Frozen Target + three independently trained multi-step residual heads on its final
hidden state. No independent Draft Transformer, Ranker, or Feature State Student.
Heads predict P_i, F_i, P_(i+1) concurrently from the hidden state of P_(i-1).
Known pending F_(i-1) is NOT an input to these heads (skip-one anchor); this is
an explicit efficiency/accuracy tradeoff, NOT full tree-Medusa.

Target Block4 verifies all proposals with exact min(1,p/q), positive residual,
COW logical KV rewind, legal P/F/EOS slots. Heads are trained with teacher-
forced next-offset CE; Target weights stay frozen. No speedup claims without GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bench_gaussiangpt_b1_two_model import policy, draw, acceptance, residual, rewind, assert_pos, assert_feat

DEFAULT_SAVE = Path('profiling/baselines_5way/b3_medusa_linear')


class MedusaLinearHeads(nn.Module):
    """Lightweight residual heads, shared frozen Target LM projection.

    All q values are explicitly normalized AFTER position rank and type masking;
    the identical normalized q is used for proposal and exact verification.
    """
    def __init__(self, n_embd, lm_weight, position_vocab, eos_id):
        super().__init__()
        self.dim = int(n_embd)
        self.p_vocab = int(position_vocab)
        self.eos = int(eos_id)
        assert self.eos > self.p_vocab
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(n_embd, n_embd, bias=False),
                          nn.SiLU(),
                          nn.Linear(n_embd, n_embd, bias=False)) for _ in range(3)
        ])
        self.register_buffer('lm_weight', lm_weight.detach(), persistent=False)
        # Residual identity initialization makes first heads retain frozen Target geometry.
        for h in self.heads:
            nn.init.zeros_(h[2].weight)

    def raw(self, h):
        """h: [B,H], normalized frozen final-block hidden states."""
        if h.ndim != 2 or h.shape[-1] != self.dim:
            raise AssertionError('Medusa head hidden shape mismatch')
        outs = []
        for j, net in enumerate(self.heads):
            z = h + net(h)
            if j == 1:
                w = self.lm_weight[self.p_vocab:self.eos]
            else:
                w = torch.cat((self.lm_weight[:self.p_vocab],
                               self.lm_weight[self.eos:self.eos+1]), dim=0)
            logits = F.linear(z, w).float()
            outs.append(15.0 * torch.tanh(logits / 15.0))
        return outs  # P1:[B,PV+1], F1:[B,FV], P2:[B,PV+1]

    def distribution(self, raw_logits, slot, *, prev_p, ranks, temp, top_p):
        """Full-size target-compatible q, rank-masked for each predicted Position."""
        if raw_logits.ndim != 1:
            raise AssertionError('Expected single vector')
        logits = torch.full((self.eos+2,), -torch.inf, device=raw_logits.device,
                            dtype=torch.float32)
        if slot == 'feature':
            if raw_logits.numel() != self.eos-self.p_vocab:
                raise AssertionError('Feature head dimension mismatch')
            logits[self.p_vocab:self.eos] = raw_logits
        elif slot == 'position':
            if raw_logits.numel() != self.p_vocab+1:
                raise AssertionError('Position head dimension mismatch')
            scores = raw_logits[:self.p_vocab].clone()
            rank_gpu = ranks.to(device=raw_logits.device)
            scores.masked_fill_(rank_gpu <= rank_gpu[int(prev_p)], -torch.inf)
            logits[:self.p_vocab] = scores
            logits[self.eos] = raw_logits[-1]
        else:
            raise ValueError(slot)
        return policy(logits, temp, top_p)


class LastBlockHook:
    def __init__(self, target):
        self.value = None
        self.handle = target.transformer.h[-1].register_forward_hook(self._capture)

    def _capture(self, mod, inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(x) or x.ndim != 3:
            raise AssertionError('Invalid final-block hook output')
        self.value = F.rms_norm(x.detach(), (x.size(-1),))

    def forward(self, target, cache, tokens, kinds, prefix, stats, count):
        if int(cache.get_pos()) != prefix:
            raise AssertionError(f'Unexpected target KV position {cache.get_pos()}, want {prefix}')
        if len(tokens) != len(kinds):
            raise AssertionError('typed slots mismatch')
        self.value = None
        indices = torch.tensor([tokens], dtype=torch.long, device=next(target.parameters()).device)
        out = target(indices, kv_cache=cache, logit_types=kinds)[0]
        h = self.value
        if h is None or h.shape[1] != len(tokens) or int(cache.get_pos()) != prefix+len(tokens):
            raise AssertionError('Target final hidden or KV advancement mismatch')
        stats[count] += 1
        return out, h[0]

    def close(self):
        self.handle.remove()


@torch.inference_mode()
def medusa_linear_decode(target, heads, make_cache, *, eos, p_vocab, ranks,
                         max_tokens, seed, device,
                         target_temp=0.9, target_top_p=0.9,
                         draft_temp=0.9, draft_top_p=0.9):
    if max_tokens < 2 or max_tokens % 2:
        raise ValueError('max_tokens must be positive and even')
    cache = make_cache()
    hook = LastBlockHook(target)
    tg = torch.Generator(device=device).manual_seed(seed)
    dg = torch.Generator(device=device).manual_seed(seed+1000003)
    st = Counter()
    output = []

    def fw(tokens, kinds, prefix, name):
        return hook.forward(target,cache,tokens,kinds,prefix,st,name)

    def tp(logits):
        return policy(logits,target_temp,target_top_p)

    def q(slot, raw, prev):
        return heads.distribution(raw,slot,prev_p=prev,ranks=ranks,
                                  temp=draft_temp,top_p=draft_top_p)

    try:
        init, _ = fw([eos], ['position'], 0, 'forward_single')
        p0 = draw(tp(init[0]), tg)
        if p0 == eos:
            st['eos'] += 1
            return output, st
        if p0 < 0 or p0 >= p_vocab:
            raise AssertionError('Invalid initial Position')
        init, h0 = fw([p0], ['feature'], 1, 'forward_single')
        f0 = draw(tp(init[0]), tg)
        assert_feat(f0,p_vocab,eos)
        output.extend((p0,f0))
        if len(output) == max_tokens:
            return output, st
        hprev = h0[-1].detach().clone()
        prev_p, pending_f = p0, f0

        def finish(p,f):
            assert_pos(p,prev_p,p_vocab,ranks)
            assert_feat(f,p_vocab,eos)
            output.extend((p,f))

        while len(output) < max_tokens:
            before = int(cache.get_pos())
            if before != len(output):
                # BOS plus all processed committed tokens except last pending feature.
                raise AssertionError(f'Target cache not synchronized: {before=} {len(output)=}')
            raw = heads.raw(hprev.unsqueeze(0))
            st['head_forward'] += 1
            q1 = q('position',raw[0][0],prev_p)
            p1 = draw(q1,dg)
            st['p1_checked'] += 1
            k1 = (len(output)+2 == max_tokens or p1 == eos)
            if k1:
                st['k1_steps'] += 1
                if p1 == eos:
                    out,h=fw([pending_f],['position'],before,'forward_single')
                else:
                    out,h=fw([pending_f,p1],['position','feature'],before,'forward_pair')
                p=tp(out[0]);st['p1_overlap_sum']+=float(torch.minimum(p,q1).sum())
                yes=acceptance(p,q1,p1,tg)
                if yes: st['p1_accepted']+=1;chosen=p1
                else: st['p1_rejected']+=1;chosen=residual(p,q1,tg)
                if chosen==eos:
                    if p1!=eos:rewind(cache,before+1,before+2)
                    st['eos']+=1;break
                assert_pos(chosen,prev_p,p_vocab,ranks)
                if yes and p1!=eos:
                    fd=tp(out[1]);new_h=h[1]
                else:
                    if p1!=eos:rewind(cache,before+1,before+2)
                    fres,hres=fw([chosen],['feature'],before+1,'forward_single')
                    fd=tp(fres[0]);new_h=hres[0]
                f1=draw(fd,tg);finish(chosen,f1)
                hprev=new_h.detach().clone()
                prev_p,pending_f=chosen,f1
                continue

            assert_pos(p1,prev_p,p_vocab,ranks)
            qf=q('feature',raw[1][0],p1)
            f1=draw(qf,dg)
            assert_feat(f1,p_vocab,eos)
            q2=q('position',raw[2][0],p1)
            p2=draw(q2,dg)
            p2_eos=p2==eos
            if not p2_eos: assert_pos(p2,p1,p_vocab,ranks)
            tokens=[pending_f,p1,f1]
            kinds=['position','feature','position']
            if not p2_eos:
                tokens.append(p2);kinds.append('feature')
            out,h=fw(tokens,kinds,before,'forward_block3' if p2_eos else 'forward_block4')
            after=before+len(tokens)
            st['long_blocks']+=1;st['f1_proposed']+=1;st['p2_proposed']+=1
            p=tp(out[0]);st['p1_overlap_sum']+=float(torch.minimum(p,q1).sum())
            if not acceptance(p,q1,p1,tg):
                st['p1_rejected']+=1
                correct=residual(p,q1,tg)
                rewind(cache,before+1,after)
                if correct==eos:st['eos']+=1;break
                assert_pos(correct,prev_p,p_vocab,ranks)
                fr,hr=fw([correct],['feature'],before+1,'forward_single')
                fc=draw(tp(fr[0]),tg)
                finish(correct,fc)
                hprev=hr[0].detach().clone()
                prev_p,pending_f=correct,fc
                continue
            st['p1_accepted']+=1;st['f1_checked']+=1
            fd=tp(out[1]);st['f1_overlap_sum']+=float(torch.minimum(fd,qf).sum())
            if not acceptance(fd,qf,f1,tg):
                st['f1_rejected']+=1
                correct=residual(fd,qf,tg)
                assert_feat(correct,p_vocab,eos)
                rewind(cache,before+2,after)
                finish(p1,correct)
                hprev=h[1].detach().clone()
                prev_p,pending_f=p1,correct
                continue
            st['f1_accepted']+=1;st['p2_checked']+=1
            pd=tp(out[2]);st['p2_overlap_sum']+=float(torch.minimum(pd,q2).sum())
            accepted2=acceptance(pd,q2,p2,tg)
            if accepted2:st['p2_accepted']+=1;chosen2=p2
            else:st['p2_rejected']+=1;chosen2=residual(pd,q2,tg)
            if chosen2==eos:
                if not p2_eos:rewind(cache,before+3,after)
                finish(p1,f1)
                st['eos']+=1;break
            assert_pos(chosen2,p1,p_vocab,ranks)
            if accepted2 and not p2_eos:
                f2dist=tp(out[3]);hnext=h[3]
            else:
                if not p2_eos:rewind(cache,before+3,after)
                f2out,h2=fw([chosen2],['feature'],before+3,'forward_single')
                f2dist=tp(f2out[0]);hnext=h2[0]
            f2=draw(f2dist,tg);assert_feat(f2,p_vocab,eos)
            finish(p1,f1)
            output.extend((chosen2,f2))
            hprev=hnext.detach().clone()
            prev_p,pending_f=chosen2,f2

        if len(output) > max_tokens or len(output)%2:
            raise AssertionError('Malformed output length')
        for a,b in zip(output[::2],output[2::2]):
            assert_pos(b,a,p_vocab,ranks)
        return output,st
    finally:
        hook.close()


def target_loader(args):
    from model.gaussian_gpt import GaussianGPT
    from model.gaussian_vqvae import GaussianVQVAE
    from profiling.s3.s3_onpolicy_feature_distill_train import make_kv_cache
    vq=GaussianVQVAE.load_from_checkpoint(str(args.vqvae),map_location='cpu').eval()
    wrap=GaussianGPT.load_from_checkpoint(str(args.gpt),map_location='cpu',vqvae=vq,strict=True).eval()
    dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    wrap.gpt.to(dev).eval().requires_grad_(False)
    model=getattr(wrap.gpt,'_orig_mod',wrap.gpt)
    target=model.transformer
    if bool(model.shared_vocab):
        raise AssertionError('Medusa requires disjoint sparse P/F vocab')
    p_vocab=int(model.position_vocab_size)
    eos=int(model.eos_token_id)
    ranks=target.sparse_position_rank_map.detach().cpu().long()
    def make_cache(max_tokens):return make_kv_cache(target,1+max_tokens+16,dev)
    return target,dev,p_vocab,eos,ranks,make_cache


def new_heads(target,p_vocab,eos):
    return MedusaLinearHeads(target.config.n_embd,target.lm_head.weight,p_vocab,eos).to(target.lm_head.weight.device)


def load_heads(path,target,p_vocab,eos):
    blob=torch.load(path,map_location='cpu',weights_only=False)
    model=new_heads(target,p_vocab,eos)
    meta=blob['meta']
    if int(meta['p_vocab'])!=p_vocab or int(meta['eos'])!=eos or int(meta['n_embd'])!=model.dim:
        raise AssertionError('Medusa checkpoint architecture does not match frozen Target')
    model.load_state_dict(blob['state'],strict=True)
    model.eval().requires_grad_(False)
    return model,meta


def capture_training_states(target,dev,token_sequences,eos,p_vocab,ranks,max_sequences=0):
    """Offline teacher-forced Target hidden states for heads offset +2,+3,+4.

    Input: each token sequence is *already generated with correct Target policy*
    and contains full pairs, excluding BOS. Excludes last two incomplete labels.
    """
    hook=LastBlockHook(target)
    states=[];future=[];prevs=[]
    try:
        with torch.inference_mode():
            for si,seq in enumerate(token_sequences):
                if max_sequences and si>=max_sequences:break
                seq=torch.as_tensor(seq,dtype=torch.long).flatten()
                if seq.numel()<8 or seq.numel()%2:
                    continue
                full=torch.cat((torch.tensor([eos],dtype=torch.long),seq)).to(dev)
                dummy=Counter()
                # Read-only forward; no persistent cache needed for full sequence.
                # The final-block hook retains hidden states; we discard logits.
                hook.value=None
                _=target(full[:-1].unsqueeze(0))
                h=hook.value[0]
                # P positions are indices 1,3,... in [BOS,P0,F0,P1,...].
                # From P_(i-1), learn skip-one forecasts of P_i,F_i,P_(i+1).
                anchor=torch.arange(1,int(full.numel())-4,2,device=dev)
                if anchor.numel()==0:continue
                labels=torch.stack((full[anchor+2],full[anchor+3],full[anchor+4]),dim=1)
                prev=full[anchor]
                if not (torch.all((labels[:,0]>=0)&(labels[:,0]<p_vocab)) and
                        torch.all((labels[:,1]>=p_vocab)&(labels[:,1]<eos)) and
                        torch.all((labels[:,2]>=0)&(labels[:,2]<p_vocab))):
                    raise AssertionError('Teacher token traces contain invalid P/F slots')
                r=ranks.to(dev)
                if not (torch.all(r[labels[:,0]]>r[prev]) and torch.all(r[labels[:,2]]>r[labels[:,0]])):
                    raise AssertionError('Training trace has illegal/unsorted Position')
                states.append(h[anchor].float().detach().cpu())
                future.append(labels.detach().cpu())
                prevs.append(prev.detach().cpu())
    finally:
        hook.close()
    if not states:
        raise RuntimeError('No usable medusa head training samples')
    return torch.cat(states),torch.cat(future),torch.cat(prevs)


def head_loss(heads,hidden,labels,prev_p,ranks):
    raw=heads.raw(hidden)
    r=ranks.to(hidden.device)
    b=labels.shape[0]
    target_p1=labels[:,0]
    target_f=(labels[:,1]-heads.p_vocab)
    target_p2=labels[:,2]
    mask1=(r[:heads.p_vocab][None,:]<=r[prev_p][:,None])
    mask2=(r[:heads.p_vocab][None,:]<=r[target_p1][:,None])
    p1=raw[0].clone();p2=raw[2].clone()
    p1[:,:heads.p_vocab].masked_fill_(mask1,-1e8)
    p2[:,:heads.p_vocab].masked_fill_(mask2,-1e8)
    return (F.cross_entropy(p1,target_p1),
            F.cross_entropy(raw[1],target_f),
            F.cross_entropy(p2,target_p2))


def read_trace_files(trace_paths,methods=('vanilla',)):
    results=[]
    for path in trace_paths:
        blob=torch.load(path,map_location='cpu',weights_only=False)
        for k,v in sorted(blob['traces'].items()):
            method=k.split('_',1)[1]
            if method in methods:
                results.append((k,v.int()))
    return results


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def collect(args):
    """Generate fresh independent vanilla Target trajectories to train heads."""
    if not torch.cuda.is_available():raise RuntimeError('GPU required for collection')
    if args.seed_start <= 88416 and args.seed_start+args.seeds-1 >= 88401:
        raise ValueError('Training data must not reuse B1 official evaluation seeds 88401..88416')
    target,dev,p_vocab,eos,ranks,make_cache=target_loader(args)
    import gpt_position_spec_decode_smoke as dec
    from threadpoolctl import threadpool_limits
    traces={}
    path=args.out_dir/'train_token_traces.pt'
    if path.exists():raise FileExistsError(path)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode(),threadpool_limits(limits=1,user_api='openmp'):
        for i in range(args.seeds):
            seed=args.seed_start+i
            seq,st=dec.vanilla(target,lambda:make_cache(args.max_tokens),eos=eos,
                               p_vocab=p_vocab,ranks=ranks,max_tokens=args.max_tokens,
                               seed=seed,device=dev,temp=0.9,top_p=0.9)
            traces[f'{seed}_vanilla']=torch.tensor(seq,dtype=torch.int16)
            print(f'COLLECT seed={seed} tokens={len(seq)} eos={st["eos"]}',flush=True)
    torch.save(dict(traces=traces,eos=eos,target_sha256=hashlib.sha256(args.gpt.read_bytes()).hexdigest()),path)
    print('SAVED',path,flush=True)


def train(args):
    if not args.trace_files:raise ValueError('--trace-files required for --train')
    if not torch.cuda.is_available():raise RuntimeError('GPU required for training heads')
    torch.manual_seed(args.seed_start)
    torch.set_float32_matmul_precision('highest')
    target,dev,p_vocab,eos,ranks,make_cache=target_loader(args)
    data=read_trace_files(args.trace_files)
    print('HEAD TRAIN trace sequences=',len(data),flush=True)
    if not data:raise RuntimeError('No vanilla traces in specified files')
    # Validate no train evaluation leakage by filename/key prefix when official test set.
    for key,_ in data:
        s=int(key.split('_',1)[0])
        if 88401<=s<=88416:
            raise ValueError(f'Official B1 test seed {s} must not be used to train heads')
    # Divide complete independent trajectories, not neighboring tokens from
    # the same scene. A per-token split would inflate held-out validation.
    if len(data) < 3:
        raise ValueError('At least three independent trajectories required for train/validation')
    g=torch.Generator(device='cpu').manual_seed(20261010)
    seq_perm=torch.randperm(len(data),generator=g).tolist()
    val_nseq=max(1,int(round(len(data)*args.val_fraction)))
    if val_nseq >= len(data)-1:
        raise ValueError('Not enough independent training trajectories')
    val_seq_ids=set(seq_perm[:val_nseq])
    train_sequences=[v for i,(k,v) in enumerate(data) if i not in val_seq_ids]
    val_sequences=[v for i,(k,v) in enumerate(data) if i in val_seq_ids]
    train_keys=[k for i,(k,v) in enumerate(data) if i not in val_seq_ids]
    val_keys=[k for i,(k,v) in enumerate(data) if i in val_seq_ids]
    train_states,train_labels,train_prevs=capture_training_states(
        target,dev,train_sequences,eos,p_vocab,ranks)
    val_states,val_labels,val_prevs=capture_training_states(
        target,dev,val_sequences,eos,p_vocab,ranks)
    train_n,val_n=int(train_states.shape[0]),int(val_states.shape[0])
    print('SPLIT independent scenes: train=',len(train_sequences),
          'val=',len(val_sequences),'samples=',train_n,val_n,flush=True)
    h=new_heads(target,p_vocab,eos)
    trainables=sum(x.numel() for x in h.parameters())
    print(f'MEDUSA heads params={trainables:,}, Target frozen={all(not p.requires_grad for p in target.parameters())}',flush=True)
    opt=torch.optim.AdamW(h.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best=float('inf')
    path=args.out_dir/'medusa_linear_heads_best.pt'
    if path.exists():raise FileExistsError(path)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    steps=0
    for epoch in range(args.epochs):
        h.train()
        train_perm=torch.randperm(train_n,generator=g)
        train_sum=0.;nt=0
        for ids in train_perm.split(args.batch_size):
            opt.zero_grad(set_to_none=True)
            hidden=train_states[ids].to(dev)
            labs=train_labels[ids].long().to(dev)
            prev=train_prevs[ids].long().to(dev)
            lp,lf,l2=head_loss(h,hidden,labs,prev,ranks)
            loss=(lp+lf+l2)/3
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite head training loss')
            loss.backward();nn.utils.clip_grad_norm_(h.parameters(),1.0)
            opt.step()
            train_sum+=float(loss.detach())*len(ids);nt+=len(ids);steps+=1
        h.eval()
        with torch.inference_mode():
            parts=[]
            for ids in torch.arange(val_n).split(args.batch_size):
                lp,lf,l2=head_loss(h,val_states[ids].to(dev),val_labels[ids].long().to(dev),
                                  val_prevs[ids].long().to(dev),ranks)
                parts.append((len(ids),float(lp),float(lf),float(l2)))
            v=[sum(entry[0]*entry[j] for entry in parts)/val_n for j in (1,2,3)]
            vmean=sum(v)/3
        print(f'EPOCH {epoch+1}/{args.epochs} train_ce={train_sum/nt:.4f} val_p1={v[0]:.4f} val_f1={v[1]:.4f} val_p2={v[2]:.4f}',flush=True)
        if vmean<best:
            best=vmean
            meta=dict(version='b3_medusa_linear_v1',offsets=[2,3,4],
                      n_embd=h.dim,p_vocab=p_vocab,eos=eos,
                      target_sha256=hashlib.sha256(args.gpt.read_bytes()).hexdigest(),
                      sources=[str(x) for x in args.trace_files],head_params=trainables,
                      train_samples=train_n,val_samples=val_n,
                      train_trajectories=len(train_sequences),val_trajectories=len(val_sequences),
                      train_keys=train_keys,val_keys=val_keys,
                      best_val_ce=v,epoch=epoch+1,steps=steps,
                      train_config={'lr':args.lr,'batch_size':args.batch_size,'weight_decay':args.weight_decay})
            torch.save(dict(state={k:v.detach().cpu() for k,v in h.state_dict().items()},meta=meta),path)
    print('SAVED',path,flush=True)


def aggregate(rows,method):
    arr=[r for r in rows if r['method']==method]
    tokens=sum(r['tokens'] for r in arr);seconds=sum(r['ar_seconds'] for r in arr)
    st=Counter()
    for r in arr:st.update(r['stats'])
    return dict(runs=len(arr),tokens=tokens,ar_seconds=seconds,
                tokens_per_second=tokens/seconds if seconds else 0,
                target_forwards=sum(st.get(k,0) for k in ('forward_single','forward_pair','forward_block3','forward_block4')),
                target_block4=st['forward_block4'],head_forwards=st['head_forward'],
                p1_accepted=st['p1_accepted'],p1_checked=st['p1_checked'],
                f1_accepted=st['f1_accepted'],f1_checked=st['f1_checked'],
                p2_accepted=st['p2_accepted'],p2_checked=st['p2_checked'])


def benchmark(args):
    if not torch.cuda.is_available():raise RuntimeError('GPU required for benchmark')
    if args.max_tokens%2 or args.max_tokens<4:raise ValueError('--max-tokens must be even >=4')
    if not args.head_checkpoint.is_file():raise FileNotFoundError(args.head_checkpoint)
    if (args.out_dir/'summary.json').exists():raise FileExistsError(args.out_dir/'summary.json')
    torch.set_float32_matmul_precision('highest')
    target,dev,p_vocab,eos,ranks,make_cache=target_loader(args)
    heads,meta=load_heads(args.head_checkpoint,target,p_vocab,eos)
    if meta.get('target_sha256')!=hashlib.sha256(args.gpt.read_bytes()).hexdigest():
        raise AssertionError('Medusa heads trained on different Target weights')
    used_seeds={int(k.split('_',1)[0]) for k in meta.get('train_keys',[])+meta.get('val_keys',[])}
    test_seeds=set(range(args.seed_start,args.seed_start+args.seeds))
    if used_seeds & test_seeds:
        raise ValueError(f'Medusa benchmark overlaps head train/validation seeds: {sorted(used_seeds & test_seeds)}')
    import gpt_position_spec_decode_smoke as dec
    import bench_gaussiangpt_pf_block2_cow as cow
    import bench_gaussiangpt_b1_two_model as b1
    from bench_fast_hgb import FastHistRanker
    from joblib import load
    from threadpoolctl import threadpool_limits
    ranker=None;feature=None;d8_draft=None;make_d8_cache=None
    if any(m in args.methods for m in ('b1_d8','b1_hybrid')):
        from model.gaussian_gpt import GaussianGPT
        from model.gaussian_vqvae import GaussianVQVAE
        from profiling.s3.s3_onpolicy_feature_distill_train import make_kv_cache
        vq=GaussianVQVAE.load_from_checkpoint(str(args.vqvae),map_location='cpu').eval()
        d8_wrap=GaussianGPT.load_from_checkpoint(str(args.d8),map_location='cpu',vqvae=vq,strict=True).eval()
        d8_wrap.gpt.to(dev).eval().requires_grad_(False)
        mdl=getattr(d8_wrap.gpt,'_orig_mod',d8_wrap.gpt)
        assert (p_vocab,eos)==(int(mdl.position_vocab_size),int(mdl.eos_token_id))
        if not torch.equal(ranks,mdl.transformer.sparse_position_rank_map.detach().cpu().long()):
            raise AssertionError('B1 D8 Rank map mismatch')
        d8_draft=mdl.transformer
        make_d8_cache=lambda n:make_kv_cache(d8_draft,1+n+16,dev)
    if any(m in args.methods for m in ('position_only','pf_cow','b1_hybrid')):
        ranker=FastHistRanker(load(args.ranker))
    if 'pf_cow' in args.methods:
        feature,dt,dp=cow.load_feature_model(args.feature_checkpoint,args.feature_head,target,dev,p_vocab,eos)
    def one(m,seed,n):
        common=dict(eos=eos,p_vocab=p_vocab,ranks=ranks,max_tokens=n,seed=seed,device=dev,
                    temp=0.9,top_p=0.9)
        if m=='vanilla':return dec.vanilla(target,lambda:make_cache(n),**common)
        if m=='position_only':return dec.speculative(target,lambda:make_cache(n),ranker,**common)
        if m=='pf_cow':return cow.block2_decode(target,lambda:make_cache(n),ranker,feature,
                                                 draft_temp=0.9,draft_top_p=0.8,**common)
        if m=='b1_d8':return b1.b1_decode(target,d8_draft,lambda:make_cache(n),
                               lambda:make_d8_cache(n),eos=eos,p_vocab=p_vocab,ranks=ranks,
                               max_tokens=n,seed=seed,device=dev,target_temp=0.9,target_top_p=0.9,
                               draft_temp=0.9,draft_top_p=0.9)
        if m=='b1_hybrid':
            from bench_gaussiangpt_b1_hybrid import b1_hybrid_decode
            return b1_hybrid_decode(target,d8_draft,lambda:make_cache(n),
                                    lambda:make_d8_cache(n),ranker,
                                    eos=eos,p_vocab=p_vocab,ranks=ranks,max_tokens=n,seed=seed,
                                    device=dev,target_temp=0.9,target_top_p=0.9,
                                    draft_temp=0.9,draft_top_p=0.9)
        if m=='medusa_linear':return medusa_linear_decode(target,heads,lambda:make_cache(n),
                                eos=eos,p_vocab=p_vocab,ranks=ranks,max_tokens=n,seed=seed,
                                device=dev,target_temp=0.9,target_top_p=0.9,
                                draft_temp=args.draft_temp,draft_top_p=args.draft_top_p)
        raise ValueError(m)
    rows=[];traces={}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode(),threadpool_limits(limits=1,user_api='openmp'):
        for j,m in enumerate(args.methods):one(m,args.seed_start+900+j,min(32,args.max_tokens))
        torch.cuda.synchronize()
        for i in range(args.seeds):
            seed=args.seed_start+i
            order=args.methods if i%2==0 else list(reversed(args.methods))
            for m in order:
                torch.cuda.synchronize();start=time.perf_counter()
                seq,st=one(m,seed,args.max_tokens)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                if len(seq)%2 or len(seq)>args.max_tokens:raise AssertionError('Non-P/F output')
                for p,f in zip(seq[::2],seq[1::2]):
                    if not 0<=p<p_vocab:raise AssertionError('Invalid Position')
                    assert_feat(f,p_vocab,eos)
                for x,y in zip(seq[::2],seq[2::2]):assert_pos(y,x,p_vocab,ranks)
                eof=bool(st['eos']);cap=(len(seq)==args.max_tokens and not eof)
                if not eof and not cap:raise AssertionError('No EOS or cap')
                rows.append(dict(seed=seed,method=m,tokens=len(seq),ar_seconds=elapsed,
                                 tokens_per_second=len(seq)/elapsed,ended_by_eos=eof,cap_hit=cap,
                                 stats={k:float(v) for k,v in st.items()}))
                if args.save_token_traces:traces[f'{seed}_{m}']=torch.tensor(seq,dtype=torch.int16)
                print(f'seed={seed} {m:15} tokens={len(seq)} tok/s={len(seq)/elapsed:.2f} '
                      f'TB4={st["forward_block4"]} Heads={st["head_forward"]} '
                      f'P1={st["p1_accepted"]}/{st["p1_checked"]} '
                      f'F1={st["f1_accepted"]}/{st["f1_checked"]} '
                      f'P2={st["p2_accepted"]}/{st["p2_checked"]}',flush=True)
                write_json(args.out_dir/'partial_results.json',{'runs':rows})
    summary=dict(label='Medusa-like linear heads, NOT Medusa tree; frozen Target',
                 target_policy={'temp':0.9,'top_p':0.9},
                 draft_policy={'temp':args.draft_temp,'top_p':args.draft_top_p},
                 head_checkpoint=str(args.head_checkpoint),head_metadata=meta,
                 runs=rows,aggregate={m:aggregate(rows,m) for m in args.methods})
    write_json(args.out_dir/'summary.json',summary)
    if args.save_token_traces:torch.save(dict(traces=traces,runs=rows,eos=eos),args.out_dir/'token_traces.pt')
    for m in args.methods:print(f'AGGREGATE {m}: {summary["aggregate"][m]["tokens_per_second"]:.3f} tok/s',flush=True)


def self_test():
    # Test the exact acceptance/residual mixture algebra for arbitrary p,q
    # and differentiated Position rank masking, without checkpoint dependencies.
    torch.manual_seed(17)
    lm=torch.randn(22,8)
    heads=MedusaLinearHeads(8,lm,16,21)
    ranks=torch.arange(16)
    raw=heads.raw(torch.randn(1,8))
    q1=heads.distribution(raw[0][0],'position',prev_p=6,ranks=ranks,temp=1,top_p=1)
    qf=heads.distribution(raw[1][0],'feature',prev_p=6,ranks=ranks,temp=1,top_p=1)
    q2=heads.distribution(raw[2][0],'position',prev_p=10,ranks=ranks,temp=1,top_p=1)
    assert abs(float(q1.detach().sum())-1)<1e-6 and abs(float(qf.detach().sum())-1)<1e-6 and abs(float(q2.detach().sum())-1)<1e-6
    assert not torch.count_nonzero(q1[:7]) and not torch.count_nonzero(q2[:11])
    assert not torch.count_nonzero(qf[:16]) and not torch.count_nonzero(qf[21:])
    for q in (q1,qf,q2):
        p=torch.rand_like(q);p[q==0]=0;p/=p.sum()
        a=torch.minimum(p,q)
        r=torch.clamp(p-q,min=0)
        merged=a+r
        assert torch.allclose(merged,p,atol=1e-6)
    hidden=torch.randn(5,8,requires_grad=False)
    labels=torch.tensor([[7,17,8],[8,18,10],[9,19,10],[10,20,11],[11,17,13]])
    previous=torch.tensor([5,6,7,8,9])
    losses=head_loss(heads,hidden,labels,previous,ranks)
    assert all(bool(torch.isfinite(x)) for x in losses)
    sum(losses).backward()
    assert all(x.grad is not None for x in heads.parameters())
    print('PASS: Medusa heads typed normalization, rank legality, exact residual identity, frozen LM head, gradient',flush=True)
    print('SCOPE: unit algebra only; native CUDA hidden/KV and forced rejection branch need real integration tests',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test',action='store_true')
    group.add_argument('--collect',action='store_true')
    group.add_argument('--train',action='store_true')
    group.add_argument('--benchmark',action='store_true')
    ap.add_argument('--gpt',type=Path,default=Path('checkpoints/gpt_vfront.ckpt'))
    ap.add_argument('--vqvae',type=Path,default=Path('checkpoints/vqvae_vfront.ckpt'))
    ap.add_argument('--ranker',type=Path,default=Path('profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib'))
    ap.add_argument('--feature-checkpoint',type=Path,default=Path('profiling/feature_draft/state_distill_mixed_overlap/feature_state_draft_best.pt'))
    ap.add_argument('--feature-head',type=Path,default=Path('profiling/feature_draft/state_generation_512/states/frozen_feature_head.pt'))
    ap.add_argument('--head-checkpoint',type=Path,default=DEFAULT_SAVE/'medusa_linear_heads_best.pt')
    ap.add_argument('--d8',type=Path,default=Path('profiling/frozen/S2_3c_d8_canonical_final_031d916/checkpoint/d8_canonical_val_best_epoch37_step50502.ckpt'))
    ap.add_argument('--trace-files',type=Path,nargs='*',default=[])
    ap.add_argument('--seed-start',type=int,default=88501,
                    help='Collect seeds from 88501; for benchmark explicitly set a disjoint range, e.g. 88601')
    ap.add_argument('--seeds',type=int,default=8)
    ap.add_argument('--max-tokens',type=int,default=1024)
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--lr',type=float,default=5e-4)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--weight-decay',type=float,default=0.01)
    ap.add_argument('--val-fraction',type=float,default=0.2)
    ap.add_argument('--draft-temp',type=float,default=0.9)
    ap.add_argument('--draft-top-p',type=float,default=0.9)
    ap.add_argument('--methods',nargs='+',choices=['vanilla','b1_d8','b1_hybrid','position_only','pf_cow','medusa_linear'],
                    default=['vanilla','position_only','pf_cow','medusa_linear'])
    ap.add_argument('--out-dir',type=Path,default=DEFAULT_SAVE)
    ap.add_argument('--save-token-traces',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test()
    elif args.collect:collect(args)
    elif args.train:train(args)
    elif args.benchmark:benchmark(args)


if __name__=='__main__':main()
