"""G1: endpoint equivalence, branch coverage, state reuse, and probability laws."""
import inspect
from collections import Counter, defaultdict

import numpy as np
import torch

from draft_a import pf_feature_unified as original
from draft_a import pf_ranker_eval as position
from gap_decoder import run
from bench_gaussiangpt_pf_block2_cow import truncate


def point_law(p,proposal):
    q=np.zeros_like(p);q[proposal]=1.
    return full_law(p,q)


def full_law(p,q):
    accepted=np.minimum(p,q)
    rejected=1-accepted.sum()
    residual=np.maximum(p-q,0.)
    return accepted+rejected*residual/residual.sum() if residual.sum()>1e-14 else accepted


class StateAudit:
    def __init__(self,ranks):
        self.ranks=ranks;self.rows=[]

    def __call__(self,target,cache,capture,prefix,h_prev,online,stats):
        shared=original._shared()
        n=len(prefix)-1
        assert int(cache.get_pos())==n
        tokens=cache.get_tokens(n)[0].tolist()
        assert tokens==prefix[:-1]
        fresh=shared.cache_for(target,n+6)
        _,states=capture.run(target,fresh,prefix[:-1],shared.typed_kinds(0,n),torch.device('cuda'))
        differences={}
        for name in ('k_cache','v_cache'):
            a=getattr(cache,name)[:,:,:n]
            b=getattr(fresh,name)[:,:,:n]
            differences[name]=float((a-b).abs().max())
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
        fresh_logits,_=capture.run(target,fresh,[prefix[-1]],shared.typed_kinds(n,1),torch.device('cuda'))
        live_logits,_=capture.run(target,cache,[prefix[-1]],shared.typed_kinds(n,1),torch.device('cuda'))
        p_fresh=shared.policy(fresh_logits[-1])
        p_live=shared.policy(live_logits[-1])
        differences['next_token_tv']=float((p_fresh-p_live).abs().sum()/2)
        assert differences['next_token_tv']<0.01
        truncate(cache,n,n+1)
        assert int(cache.get_pos())==n
        if prefix[-1] != original.EOS:
            hidden_delta = h_prev-states[-1]
            differences['full_prefill_hidden_max_abs']=float(hidden_delta.abs().max())
            differences['full_prefill_hidden_relative_l2']=float(
                hidden_delta.norm() / states[-1].norm().clamp_min(1e-12)
            )
            differences['full_prefill_hidden_cosine']=float(
                torch.nn.functional.cosine_similarity(h_prev,states[-1],dim=0)
            )
            expected = expected_position_state(prefix)
            assert online.predictor.gaps == expected['gaps']
            assert online.n_finalized_positions == expected['n_finalized_positions']
            assert online.last_seen == expected['last_seen']
            assert online.pending is None
            for obj, name in (
                (online.predictor, 'm2'), (online.predictor, 'phase_m2'),
                (online.predictor, 'phase_only'), (online, 'm3'),
                (online, 'm3_phase'),
            ):
                assert clean_counters(getattr(obj, name)) == clean_counters(expected[name])
        self.rows.append(differences)


def clean_counters(mapping):
    return {key: dict(value) for key, value in mapping.items() if value}


def expected_position_state(prefix):
    positions = list(prefix[1::2])
    gaps = [b-a for a,b in zip(positions,positions[1:])]
    m2, phase_m2, phase_only = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    m3, m3_phase, last_seen = defaultdict(Counter), defaultdict(Counter), {}
    for i, (previous, gap) in enumerate(zip(positions,gaps)):
        past = gaps[:i]
        state = tuple(past[-2:]) if len(past) >= 2 else None
        phase = previous % 20
        tri = tuple(past[-3:]) if len(past) >= 3 else None
        if state is not None:
            m2[state][gap] += 1
            phase_m2[(state,phase)][gap] += 1
        phase_only[phase][gap] += 1
        if tri is not None:
            m3[tri][gap] += 1
            m3_phase[(tri,phase)][gap] += 1
        last_seen[(state,phase,gap)] = i
    return dict(gaps=gaps,n_finalized_positions=len(positions),last_seen=last_seen,
        m2=m2,phase_m2=phase_m2,phase_only=phase_only,m3=m3,m3_phase=m3_phase)


def position_with_output():
    source=inspect.getsource(position.decode)
    assert source.count('return _add_rates(stats)')==1
    source=source.replace('return _add_rates(stats)','return _add_rates(stats), output')
    namespace=dict(vars(position))
    exec(compile(source,'position_output_reference','exec'),namespace)
    return namespace['decode']


def validate(fn,target,feature,ranker,ranks,contexts):
    fields=['target_calls','generated_tokens','rounds','position_accepted',
            'position_verified','feature_accepted','feature_verified']
    reference_p=position_with_output()
    endpoints=[];branches=Counter();transitions=Counter()
    for ci in (0,8,16,24,32,40,48,54):
        prompt=contexts[ci]['prompt'];seed=771001+ci
        for mode in ('pf','p_only'):
            a,ao=run(fn,target,feature,ranker,ranks,prompt,64,seed,mode=mode,trace=True)
            if mode=='pf':
                b,bo=original.decode(target,original.MLPAdapter(feature),ranker,ranks,prompt,64,seed)
            else:
                with torch.inference_mode():
                    b,bo=reference_p(target,None,ranker,ranks,prompt,64,1,seed)
            assert ao==bo and all(a[k]==b[k] for k in fields),(ci,mode)
            endpoints.append(dict(context_id=ci,mode=mode,output_equal=True,counts_equal=True))
            c,co=run(fn,target,feature,ranker,ranks,prompt,64,seed,mode='random',
                     probability=1. if mode=='pf' else 0.)
            assert co==ao and all(c[k]==a[k] for k in fields)
    for ci,mode in [(0,'gap'),(8,'gap'),(16,'alternate'),(24,'random')]:
        audit=StateAudit(ranks)
        stats,output=run(fn,target,feature,ranker,ranks,contexts[ci]['prompt'],64,
                         772001+ci,mode=mode,probability=.5,trace=True,audit=audit)
        branches.update(stats['selection']['branches'])
        for k,v in stats['selection']['counts'].items():
            if '_to_' in k:transitions[k]+=v
        endpoints.append(dict(context_id=ci,mode=mode,audited_rounds=len(audit.rows),
            max_cache_abs_difference={k:max(r.get(k,0.) for r in audit.rows)
                                      for k in ('k_cache','v_cache')},
            max_full_prefill_hidden_abs_difference=max(
                r.get('full_prefill_hidden_max_abs',0.) for r in audit.rows),
            max_full_prefill_hidden_relative_l2=max(
                r.get('full_prefill_hidden_relative_l2',0.) for r in audit.rows),
            min_full_prefill_hidden_cosine=min(
                r.get('full_prefill_hidden_cosine',1.) for r in audit.rows),
            max_next_token_tv=max(r.get('next_token_tv',0.) for r in audit.rows)))
    forced=[]
    prompt=contexts[8]['prompt']
    for decisions,name in [([False],'p1_reject'),([True,False],'f1_reject'),
                           ([True,True,False],'p2_reject'),([True,True,True],'all_accept')]:
        calls=iter(decisions)
        def decide(alpha,rng,device):
            torch.rand((),device=device,generator=rng)
            return next(calls)
        audit=StateAudit(ranks)
        stats,output=run(fn,target,feature,ranker,ranks,prompt,8,773001,mode='pf',
            max_rounds=1,trace=True,decision_override=decide,audit=audit)
        assert stats['selection']['trace'][0]['branch']==name
        forced.append(dict(branch=name,passed=True,audited_rounds=len(audit.rows)))
        branches.update(stats['selection']['branches'])
    # Controlled EOS logits exercise actual decoder exits; these do not measure quality.
    forward=target.forward
    cutoff=0
    def eos_forward(idx,kv_cache=None,logit_types=None,**kwargs):
        start=int(kv_cache.get_pos())
        result=forward(idx,kv_cache=kv_cache,logit_types=logit_types,**kwargs)
        for j,kind in enumerate(logit_types):
            if kind=='position' and start+j>=cutoff:
                result[:,j,:]=-torch.inf;result[:,j,original.EOS]=0.
        return result
    target.forward=eos_forward
    eos_rows=[]
    try:
        for mode in ('gap','random','pf','p_only'):
            for prompt,label in [([original.EOS],'bos'),(contexts[8]['prompt'],'p1_residual')]:
                audit=StateAudit(ranks)
                stats,output=run(fn,target,feature,ranker,ranks,prompt,8,774001,mode=mode,
                                 probability=.5,trace=True,audit=audit)
                assert output==[original.EOS]
                eos_rows.append(dict(mode=mode,exit=label,passed=True,audited_rounds=len(audit.rows)))
        prompt=contexts[8]['prompt'];cutoff=len(prompt)+1
        calls=iter([True,True,False])
        def decide(alpha,rng,device):
            torch.rand((),device=device,generator=rng)
            return next(calls)
        audit=StateAudit(ranks)
        stats,output=run(fn,target,feature,ranker,ranks,prompt,8,774003,mode='pf',
                         trace=True,max_rounds=1,decision_override=decide,audit=audit)
        assert len(output)==3 and output[-1]==original.EOS
        assert stats['selection']['trace'][0]['branch']=='p2_reject'
        eos_rows.append(dict(mode='pf',exit='p2_residual',passed=True,audited_rounds=len(audit.rows)))
    finally:
        target.forward=forward
    last=max(range(8000),key=lambda p:int(ranks[p]))
    for mode in ('gap','random'):
        stats,output=run(fn,target,feature,ranker,ranks,[original.EOS,last,8000],8,
                         774009,mode=mode,probability=.5,trace=True)
        assert output==[original.EOS]
        eos_rows.append(dict(mode=mode,exit='natural_terminal_fallback',passed=True))
    p=np.array([.17,.28,.55]);q=np.array([.41,.51,.08])
    max_error=0.
    for proposal in range(3):
        for weight in (0.,.25,.5,.75,1.):
            combined=weight*full_law(p,q)+(1-weight)*point_law(p,proposal)
            error=float(np.max(np.abs(combined-p)));max_error=max(max_error,error)
            assert error<1e-12
    assert transitions['pf_to_p_only']>0 and transitions['p_only_to_pf']>0
    assert all(branches['pf_'+name]>0 for name in ('p1_reject','f1_reject','p2_reject','all_accept'))
    return dict(status='complete',endpoints=endpoints,branches=dict(branches),
        transitions=dict(transitions),forced_branches=forced,eos=eos_rows,
        adaptive_mixture_probability_max_error=max_error,
        scope='Actual decoder branch/state tests; full exact toy sequence enumeration is reported separately in g0_g1_static.json')
