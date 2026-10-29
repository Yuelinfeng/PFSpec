#!/usr/bin/env python3
"""New LFQ-aware Feature Draft trained from random initialization.

No reference to the previous FeatureStateDraft, frozen Feature LM Head, or Target forward.
Inference inputs match the cached P/F+COW boundary:
    h(P_prev), F_prev(local 0..4095), P_current(local 0..7999)
Output is 4096 Feature logits, on the local vocabulary, with 0.9/0.9 policy outside.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

V = 4096
P = 8000
N_BITS = 12


def make_bits(device):
    ids = torch.arange(V, device=device, dtype=torch.long)
    return ((ids[:, None] >> torch.arange(12, device=device)) & 1).float()


class Transition(nn.Module):
    def __init__(self, width, extra, dropout=0.05):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(width + extra),
            nn.Linear(width + extra, 2 * width),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )
        self.ln = nn.LayerNorm(width)

    def forward(self, x, extra):
        return self.ln(x + self.layer(torch.cat((x, extra), -1)))


class LFQScratchDraft(nn.Module):
    def __init__(self, *, mode='conditional', hidden_dim=1024, width=256,
                 token_emb=64, dropout=0.05, temperature=0.9):
        super().__init__()
        if mode not in ('flat','conditional'):
            raise ValueError(f'unknown head mode {mode!r}')
        if width < 16 or token_emb < 4 or temperature <= 0:
            raise ValueError('invalid dimensions or temperature')
        self.config = {'mode': mode, 'hidden_dim': hidden_dim, 'width': width,
                       'token_emb': token_emb, 'dropout': dropout,
                       'temperature': temperature,
                       'position_vocab': P, 'feature_vocab': V}
        self.input = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width),nn.LayerNorm(width))
        self.feat_embed = nn.Embedding(V, token_emb)
        self.pos_embed = nn.Embedding(P, token_emb)
        self.step_feature = Transition(width, token_emb+N_BITS,dropout)
        self.step_position = Transition(width, token_emb+3,dropout)
        self.head_norm = nn.LayerNorm(width)
        if mode=='flat':
            self.head = nn.Linear(width,V)
        else:
            self.head_lo = nn.Linear(width,64)
            self.head_hi_given_lo = nn.Linear(width,V)
        self.register_buffer('bit_codes',make_bits('cpu'),persistent=False)

    def forward(self, prev_state, prev_feature, position):
        if prev_state.ndim != 2 or prev_state.size(-1) != self.config['hidden_dim']:
            raise ValueError('prev_state must have shape [batch, hidden_dim]')
        prev_feature=prev_feature.long().reshape(-1)
        position=position.long().reshape(-1)
        x=self.input(prev_state.float())
        fb=self.bit_codes[prev_feature].to(dtype=x.dtype)
        x=self.step_feature(x,torch.cat((self.feat_embed(prev_feature),2*fb-1),-1))
        xyz=torch.stack((position//400,(position//20)%20,position%20),-1).float()/19.0
        x=self.step_position(x,torch.cat((self.pos_embed(position),xyz.to(dtype=x.dtype)),-1))
        x=self.head_norm(x)
        if self.config['mode']=='flat':
            z=self.head(x).float()
        else:
            # Mathematically exact chain rule q(f)=q(low6)q(high6|low6)
            # f = low6 + 64*high6. Table indices [high6,low6].
            lo=self.head_lo(x).float() / self.config['temperature']
            # conditional head first axis is low6, second high6.
            hi=self.head_hi_given_lo(x).float().view(-1,64,64) / self.config['temperature']
            joint=F.log_softmax(lo,-1).unsqueeze(-1) + F.log_softmax(hi,-1)
            # [B, low6, high6] -> token ID f=low6+64*high6
            z=(self.config['temperature']*joint.transpose(-1,-2).reshape(-1,V))
        return x,z


def policy_batch(logits, temperature=0.9, top_p=0.9):
    z=logits.float()/temperature
    if top_p<1.:
        vals, ids=torch.sort(z,dim=-1,descending=True)
        pp=torch.softmax(vals,dim=-1)
        remove=pp.cumsum(dim=-1)>top_p
        remove[:,1:]=remove[:,:-1].clone()
        remove[:,0]=False
        vals=vals.masked_fill(remove,float('-inf'))
        z=torch.full_like(z,float('-inf'))
        z.scatter_(1,ids,vals)
    return torch.softmax(z,dim=-1)


def selftest():
    torch.manual_seed(14)
    state=torch.randn(4,32)
    feat=torch.tensor([0,63,64,4095])
    pos=torch.tensor([0,443,7200,7999])
    for mode in ['flat','conditional']:
        m=LFQScratchDraft(mode=mode,hidden_dim=32,width=32,token_emb=8,dropout=0.0)
        _,logits=m(state,feat,pos)
        raw=torch.softmax(logits/0.9,-1)
        nucleus=policy_batch(logits,.9,.9)
        assert logits.shape==(4,V)
        torch.testing.assert_close(raw.sum(-1),torch.ones(4),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(nucleus.sum(-1),torch.ones(4),atol=1e-6,rtol=1e-6)
        if mode=='conditional':
            # Reconstruct joint by explicit chain rule, check correct LFQ ID ordering.
            h=m.head_norm(m.step_position(m.step_feature(m.input(state),
                   torch.cat((m.feat_embed(feat),2*m.bit_codes[feat]-1),-1)),
                   torch.cat((m.pos_embed(pos),torch.stack((pos//400,(pos//20)%20,pos%20),-1).float()/19.),-1)))
            lo=F.softmax(m.head_lo(h)/.9,-1)
            hi=F.softmax(m.head_hi_given_lo(h).view(4,64,64)/.9,-1)
            alt=(lo.unsqueeze(-1)*hi).transpose(1,2).reshape(4,V)
            torch.testing.assert_close(raw,alt,atol=1e-6,rtol=1e-6)
        loss=-(torch.softmax(torch.randn(4,V),-1)*F.log_softmax(logits/.9,-1)).sum(-1).mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    print('PASS: scratch flat/conditional heads, LFQ ID mapping, nucleus normalization, gradients')

if __name__=='__main__':
    selftest()
