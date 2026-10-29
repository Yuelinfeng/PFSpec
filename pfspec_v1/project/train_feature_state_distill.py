#!/usr/bin/env python3
"""Train a GaussianGPT Feature-state Draft using frozen Target Feature lm_head.

Stage 1 (warmup): SmoothL1 on FINAL-NORMALIZED current-position Target state.
Stage 2 (joint): hidden SmoothL1 + KL(p_T_feature_policy || q_phi_feature_policy).
Input is *only* previous-position state + accepted previous Feature + proposed Position.
Validation reports actual q=softmax(logits/temp) and q=top_p(logits/temp) overlap.
The frozen Target head and final Target norm are reproduced exactly.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from collect_feature_teacher_from_s3 import policy_batch


class FeatureStateDraft(nn.Module):
    """Tiny residual predictor of the PRE-final-norm target state at current P_i.

    Frozen Feature head contains rows [8000:12096] of target.transformer.lm_head.
    It is stored as a nonpersistent buffer (not part of trainable student weights).
    """
    def __init__(self, frozen_feature_weight: torch.Tensor, position_vocab: int = 8000,
                 feature_vocab: int = 4096, d_model: int = 256, emb_dim: int = 128,
                 mlp_dim: int = 512, target_softcap: float = 15.0):
        super().__init__()
        w = frozen_feature_weight.detach().float().contiguous()
        if w.ndim != 2 or w.shape[0] != feature_vocab:
            raise ValueError(f'Feature head must have shape [{feature_vocab}, hidden_dim]')
        hidden_dim = w.shape[1]
        self.config = dict(hidden_dim=hidden_dim, position_vocab=position_vocab,
                           feature_vocab=feature_vocab, d_model=d_model, emb_dim=emb_dim,
                           mlp_dim=mlp_dim, target_softcap=target_softcap)
        self.register_buffer('frozen_head',w, persistent=False)
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.project = nn.Linear(hidden_dim,d_model)
        self.feature_embed = nn.Embedding(feature_vocab,emb_dim)
        self.position_embed = nn.Embedding(position_vocab,emb_dim)
        inputs = d_model + 2*emb_dim
        self.delta = nn.Sequential(
            nn.LayerNorm(inputs), nn.Linear(inputs,mlp_dim), nn.GELU(),
            nn.Linear(mlp_dim,mlp_dim), nn.GELU(),
            nn.Linear(mlp_dim,hidden_dim))
        # Start with h_hat = h_prev, rather than an uncalibrated hidden state.
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, prev_state, prev_feature, position):
        from model.gpt.nanochat_gpt import norm as teacher_norm
        hp = prev_state.float()
        inp = torch.cat((self.project(self.in_norm(hp)),
                         self.feature_embed(prev_feature.long()),
                         self.position_embed(position.long())),dim=-1)
        pred_state = hp + self.delta(inp).float()
        normalized_state = teacher_norm(pred_state.float())
        # Full and typed sparse logits share this exact frozen feature-only linear + softcap.
        raw = F.linear(normalized_state, self.frozen_head)
        cap = self.config['target_softcap']
        logits = cap * torch.tanh(raw.float() / cap)
        return normalized_state,logits


def loader_for(pair, old, batch_size, shuffle):
    rows = pair['row_index'].long()
    if rows.ndim != 1 or rows.numel() != pair['prev_state'].shape[0]:
        raise ValueError('Invalid state row ids')
    if len(rows) < 1 or rows.min().item() < 0 or rows.max().item() >= len(old['teacher_probs']):
        raise ValueError('State row ids out of range')
    if not torch.equal(pair['current_position'].long(),old['input_ids'][rows,-1].long()):
        raise ValueError('State/current Position labels not aligned')
    if pair['next_state'].shape != pair['prev_state'].shape:
        raise ValueError('Previous/next state dimensions differ')
    ds = TensorDataset(pair['prev_state'], pair['next_state'],
                       pair['prev_feature'], pair['current_position'],
                       old['teacher_probs'].index_select(0,rows))
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,
                      pin_memory=torch.cuda.is_available(),num_workers=0)


def epoch(model, dl, *, device, temp, student_top_p, optimizer=None,
          hidden_weight=1., kl_weight=1., amp=True):
    from model.gpt.nanochat_gpt import norm as teacher_norm
    training = optimizer is not None
    model.train(training)
    # [Lh, KL, acceptance_softmax, acceptance_topp, n]
    totals = torch.zeros(5,dtype=torch.float64)
    for prev_h, next_h, prev_f, pos, p in dl:
        prev_h = prev_h.to(device,non_blocking=True)
        next_h = next_h.to(device,non_blocking=True)
        prev_f = prev_f.to(device=device,dtype=torch.long,non_blocking=True)
        pos = pos.to(device=device,dtype=torch.long,non_blocking=True)
        p = p.to(device=device,dtype=torch.float32,non_blocking=True)
        p = p / p.sum(-1,keepdim=True).clamp_min(1e-12)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,
                                enabled=amp and device.type=='cuda'):
                pred_normalized,logits = model(prev_h,prev_f,pos)
            # Both state targets and losses in FP32.
            target_normalized = teacher_norm(next_h.float()).detach()
            hidden_loss = F.smooth_l1_loss(pred_normalized.float(),target_normalized.float())
            log_q = F.log_softmax(logits.float()/temp,dim=-1)
            kl = (p*(p.clamp_min(1e-30).log()-log_q)).sum(-1).mean()
            loss = hidden_weight*hidden_loss + kl_weight*kl
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
        with torch.no_grad():
            q_raw = log_q.exp()
            overlap_raw = torch.minimum(p,q_raw).sum(-1)
            q_topp = policy_batch(logits.detach().float(),temp,student_top_p)
            overlap_top = torch.minimum(p,q_topp).sum(-1)
            n = len(prev_h)
            totals += torch.tensor([hidden_loss.item()*n,kl.item()*n,
                                    overlap_raw.sum().item(),overlap_top.sum().item(),n],dtype=torch.float64)
    if totals[-1] == 0:
        raise ValueError('Empty training/validation data')
    return dict(hidden_loss=float(totals[0]/totals[-1]),
                kl=float(totals[1]/totals[-1]),
                accept_raw=float(totals[2]/totals[-1]),
                accept_topp=float(totals[3]/totals[-1]),rows=int(totals[-1]))


@torch.inference_mode()
def latency(model, pair, device, temp, top_p, amp=True, iters=100):
    model.eval()
    h = pair['prev_state'][:1].to(device)
    f = pair['prev_feature'][:1].long().to(device)
    p = pair['current_position'][:1].long().to(device)
    def run():
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,
                            enabled=amp and device.type=='cuda'):
            _,logits = model(h,f,p)
        q = policy_batch(logits.float(),temp,top_p)
        # Sample exactly one proposal without copying logits to the CPU.
        torch.multinomial(q,1)
    for _ in range(20):
        run()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(iters):
        run()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter()-start)*1000/iters


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=Path('profiling/feature_draft/full_data'))
    ap.add_argument('--states-dir',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--d-model',type=int,default=256)
    ap.add_argument('--emb-dim',type=int,default=128)
    ap.add_argument('--mlp-dim',type=int,default=512)
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--warmup-epochs',type=int,default=2)
    ap.add_argument('--patience',type=int,default=4)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--hidden-weight',type=float,default=1.0)
    ap.add_argument('--kl-weight',type=float,default=1.0)
    ap.add_argument('--student-temperature',type=float,default=0.9)
    ap.add_argument('--student-top-p',type=float,default=0.9)
    ap.add_argument('--seed',type=int,default=27001)
    ap.add_argument('--device',type=str,default='cuda')
    ap.add_argument('--no-amp',action='store_true')
    args=ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.warmup_epochs < 0:
        ap.error('Invalid epochs/batch-size/patience/warmup')
    if args.warmup_epochs >= args.epochs:
        ap.error('Warmup must finish before final epoch')
    if args.student_temperature <= 0 or not 0 < args.student_top_p <= 1:
        ap.error('Invalid sampling policy')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        ap.error('No CUDA device')
    device=torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(args.seed)
    old={s:torch.load(args.data_dir/f'{s}.pt',map_location='cpu',weights_only=True)
         for s in ('train','val')}
    pairs={s:torch.load(args.states_dir/f'{s}_state_pairs.pt',map_location='cpu',weights_only=True)
           for s in ('train','val')}
    head=torch.load(args.states_dir/'frozen_feature_head.pt',map_location='cpu',weights_only=True)
    if int(head['weight'].shape[0]) != 4096:
        raise ValueError('Expected frozen Feature head with 4096 tokens')
    for s in ('train','val'):
        if pairs[s]['original_rows'] != len(old[s]['teacher_probs']):
            raise ValueError(f'{s}: state and old data disagree on row count')
        if pairs[s]['prev_state'].shape[-1] != head['weight'].shape[1]:
            raise ValueError(f'{s}: target hidden dimension mismatch')
        if float(old[s]['temperature']) != 0.9 or float(old[s]['top_p']) != 0.9:
            print('WARNING: Teacher policy in data differs from expected 0.9/0.9',flush=True)
    pos_vocab=int(pairs['train']['position_vocab'])
    if int(pairs['val']['position_vocab']) != pos_vocab:
        raise ValueError('Position vocab mismatch')
    model=FeatureStateDraft(head['weight'],position_vocab=pos_vocab,
                            d_model=args.d_model,emb_dim=args.emb_dim,
                            mlp_dim=args.mlp_dim, target_softcap=float(head['softcap'])).to(device)
    tr_dl=loader_for(pairs['train'],old['train'],args.batch_size,True)
    va_dl=loader_for(pairs['val'],old['val'],args.batch_size,False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'EAGLE-inspired Feature-state Draft: trainable={params:,}, hidden={head["weight"].shape[1]}, '
          f'W_F=frozen, train_rows={len(tr_dl.dataset)}, val_rows={len(va_dl.dataset)}',flush=True)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    best=-1.;best_epoch=0;bad=0;history=[]
    amp=device.type=='cuda' and not args.no_amp
    for ep in range(1,args.epochs+1):
        tic=time.perf_counter()
        warm=ep<=args.warmup_epochs
        train=epoch(model,tr_dl,device=device,temp=args.student_temperature,
                    student_top_p=args.student_top_p,optimizer=optimizer,
                    hidden_weight=args.hidden_weight,
                    kl_weight=0.0 if warm else args.kl_weight,amp=amp)
        val=epoch(model,va_dl,device=device,temp=args.student_temperature,
                  student_top_p=args.student_top_p,amp=amp,
                  hidden_weight=args.hidden_weight,kl_weight=args.kl_weight)
        history.append(dict(epoch=ep,phase='hidden_warmup' if warm else 'joint',
                            train=train,val=val,seconds=time.perf_counter()-tic))
        print(f'epoch={ep:2} phase={"warmup" if warm else "joint":6} '
              f'train_KL={train["kl"]:.4f} val_KL={val["kl"]:.4f} '
              f'val_accept_raw={100*val["accept_raw"]:.2f}% '
              f'val_accept_top_p={100*val["accept_topp"]:.2f}% '
              f'val_H={val["hidden_loss"]:.5f} wall={history[-1]["seconds"]:.1f}s',flush=True)
        if not warm:
            score=val['accept_topp']
            if score>best:
                best=score;best_epoch=ep;bad=0
                torch.save(dict(model_state=model.state_dict(),model_config=model.config,
                                best_epoch=ep,best_acceptance=best,
                                student_temperature=args.student_temperature,
                                student_top_p=args.student_top_p,
                                frozen_head_file='frozen_feature_head.pt',
                                required_state='Target last-block output at previous Position P_(i-1)',
                                training_target_state='Target last-block output at current Position P_i'),
                           args.out_dir/'feature_state_draft_best.pt')
            else:
                bad+=1
            if bad>=args.patience:
                print(f'Early-stop: no validation top-p overlap improvement for {bad} joint epochs',flush=True)
                break
    ckpt=torch.load(args.out_dir/'feature_state_draft_best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    ms=latency(model,pairs['val'],device,args.student_temperature,args.student_top_p,amp=amp)
    print(f'BEST: epoch={best_epoch} feature_expected_accept={100*best:.2f}% '
          f'one_feature_model_plus_policy_sample={ms:.3f}ms '
          f'checkpoint={args.out_dir/"feature_state_draft_best.pt"}',flush=True)
    summary=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                 model_config=model.config,trainable_params=params,
                 best_acceptance=best,best_epoch=best_epoch,online_draft_latency_ms=ms,
                 history=history,
                 note='Only offline teacher-forced context. Next step requires online speculative acceptance and state caching; runtime must never compute current Target state before drafting.')
    (args.out_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
