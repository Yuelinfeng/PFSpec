#!/usr/bin/env python3
"""Inspect EOS and point-proposal laws near the replayed bootstrap end."""
import json
from pathlib import Path
import sys
sys.path.insert(0,'/root/autodl-tmp/gaussiangpt_multichunk_5090_20261028/source')
sys.path.insert(0,'/root/autodl-tmp/gaussiangpt_distribution_5090_20261028/source')
import run_scene as rs
from probe import law_point
import torch
from threadpoolctl import threadpool_limits
from draft_a.pf_ranker_eval import cache_for, policy, typed_kinds, position_state
from gpt_position_spec_decode_smoke import choose_position
from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock

OUT=Path('/root/autodl-tmp/gaussiangpt_distribution_5090_20261028')

def main():
    with torch.inference_mode(),threadpool_limits(limits=1):
        gpt,model,ctx=rs.load_models()
        data=torch.load(OUT/'bootstrap_replay_tokens.pt',map_location='cpu',weights_only=True).tolist()
        assert data[-1]==rs.EOS
        body=data[:-1]
        rows=[]
        for n in range(len(body)-30,len(body)+1,2):
            prefix=[rs.EOS]+body[:n]
            cap=CaptureLastBlock(ctx['target'])
            try:
                c=cache_for(ctx['target'],len(prefix)+4)
                logits,_=cap.run(ctx['target'],c,prefix,typed_kinds(0,len(prefix)),'cuda')
                p=policy(logits[-1])
                state=position_state(prefix,ctx['ranks'])
                proposal=choose_position(ctx['ranker'],state,prefix[-2])
                row=dict(prefix_tokens=n,eos_probability=float(p[rs.EOS]),
                         positive_tokens=int((p>0).sum()),proposal=None,
                         proposal_tv=None)
                if proposal is not None:
                    row['proposal']=int(proposal[0])
                    row['proposal_probability']=float(p[row['proposal']])
                    row['proposal_tv']=law_point(p,row['proposal'])
                rows.append(row)
            finally:
                cap.close()
            print(json.dumps(row),flush=True)
        (OUT/'termination_result.json').write_text(json.dumps(dict(status='complete',rows=rows),indent=2)+'\n')

if __name__=='__main__': main()
