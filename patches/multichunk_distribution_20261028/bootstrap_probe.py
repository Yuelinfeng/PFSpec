#!/usr/bin/env python3
"""Replay the full original bootstrap and examine actual termination context."""
import json
from pathlib import Path
import sys
sys.path.insert(0, '/root/autodl-tmp/gaussiangpt_multichunk_5090_20261028/source')
import run_scene as rs
import torch
from threadpoolctl import threadpool_limits
from draft_a.pf_ranker_eval import cache_for, policy, typed_kinds
from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock

OUT=Path('/root/autodl-tmp/gaussiangpt_distribution_5090_20261028')

def write(obj):
    (OUT/'bootstrap_result.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def main():
    write(dict(status='loading'))
    with torch.inference_mode(), threadpool_limits(limits=1):
        gpt,model,ctx=rs.load_models()
        adapter=rs.api.load_method('pf_mlp',ctx)
        arm=rs.arm('pf_w1',ctx,adapter,True)
        try:
            write(dict(status='replaying_full_bootstrap'))
            stats,suffix=rs.compare.run_arm(arm,ctx,[rs.EOS],16000,0,rs.api)
        finally:
            rs.compare.close_arm(arm)
        orig=json.loads((rs.ROOT/'full_pf_01/pf_w1/sampler_calls.json').read_text())[0] if hasattr(rs,'ROOT') else json.loads((Path('/root/autodl-tmp/gaussiangpt_multichunk_5090_20261028')/'full_pf_01/pf_w1/sampler_calls.json').read_text())[0]
        same_metadata=len(suffix)==orig['generated_tokens'] and bool(suffix and suffix[-1]==rs.EOS)==orig['eos']
        if not same_metadata or not suffix or suffix[-1]!=rs.EOS:
            raise ValueError('Full bootstrap replay did not reproduce saved count and EOS')
        torch.save(torch.tensor(suffix,dtype=torch.long),OUT/'bootstrap_replay_tokens.pt')
        print('replayed',len(suffix),'EOS',flush=True)
        body=suffix[:-1]
        results=[]
        for n in sorted(set((0,2,32,96,len(body)))):
            prefix=[rs.EOS]+body[:n]
            cap=CaptureLastBlock(ctx['target'])
            try:
                a=cache_for(ctx['target'],len(prefix)+4)
                typed,_=cap.run(ctx['target'],a,prefix,typed_kinds(0,len(prefix)),'cuda')
                b=cache_for(ctx['target'],len(prefix)+4)
                implicit=ctx['target'](torch.tensor([prefix],device='cuda'),kv_cache=b)[0]
                p,q=policy(implicit[-1]),policy(typed[-1])
                results.append(dict(prefix_tokens=n,eos_probability=float(p[rs.EOS]),
                                    tv=float((p-q).abs().sum()/2),
                                    support_differences=int(((p>0)!=(q>0)).sum())))
            finally:
                cap.close()
        # Exact final-position boundary: only EOS is legal, exercising both wrappers.
        rank=ctx['ranks']; max_position=max(range(len(rank)),key=lambda i:rank[i])
        terminal_prompt=[int(max_position),8000]
        native=model.sample_sequence_with_prompt(torch.tensor(terminal_prompt,dtype=torch.long),
                 max_new_tokens=2,num_samples=1,temperature=.9,top_k=None,top_p=.9,
                 stop_on_eos=True,seed=13)[0].tolist()
        methods={}
        for name in ('pf_w1','pf_rescue'):
            a=rs.arm(name,ctx,adapter,True)
            try:
                counts, generated=rs.compare.run_arm(a,ctx,[rs.EOS]+terminal_prompt,2,13,rs.api)
                methods[name]=dict(suffix=generated,eos=counts['eos'])
            finally:
                rs.compare.close_arm(a)
        write(dict(status='complete',bootstrap_replay_tokens=len(suffix),
                   original_metadata_equal=same_metadata,positions=results,
                   synthetic_terminal=dict(prompt=terminal_prompt,native_return=native,methods=methods),
                   limitation='Original raw bootstrap token contents were not saved; equal count and EOS do not establish identical replay tokens.'))
        print('complete',json.dumps(results),flush=True)

if __name__=='__main__':
    main()
