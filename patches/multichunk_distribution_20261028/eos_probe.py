#!/usr/bin/env python3
"""Exercise EOS through Target and both speculative decoders at a replayed prefix."""
import json
from pathlib import Path
import sys
sys.path.insert(0,'/root/autodl-tmp/gaussiangpt_multichunk_5090_20261028/source')
import run_scene as rs
import torch
from threadpoolctl import threadpool_limits

OUT=Path('/root/autodl-tmp/gaussiangpt_distribution_5090_20261028')

def main():
    with torch.inference_mode(),threadpool_limits(limits=1):
        gpt,model,ctx=rs.load_models()
        adapter=rs.api.load_method('pf_mlp',ctx)
        suffix=torch.load(OUT/'bootstrap_replay_tokens.pt',map_location='cpu',weights_only=True).tolist()
        assert suffix[-1]==rs.EOS
        prompt=suffix[:-1]
        native=model.sample_sequence_with_prompt(torch.tensor(prompt,dtype=torch.long),
                 max_new_tokens=2,num_samples=1,temperature=.9,top_k=None,top_p=.9,
                 stop_on_eos=True,seed=13)[0].tolist()
        rows={}
        for name in ('pf_w1','pf_rescue'):
            arm=rs.arm(name,ctx,adapter,True)
            try:
                stats,generated=rs.compare.run_arm(arm,ctx,[rs.EOS]+prompt,2,13,rs.api)
                rows[name]=dict(suffix=generated,eos=stats['eos'],
                                position_proposed=stats['position_proposed'],
                                position_accepted=stats['position_accepted'])
            finally:
                rs.compare.close_arm(arm)
        result=dict(status='complete',prompt_tokens=len(prompt),native_new_tokens=native[len(prompt):],methods=rows)
        (OUT/'eos_result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)

if __name__=='__main__': main()
