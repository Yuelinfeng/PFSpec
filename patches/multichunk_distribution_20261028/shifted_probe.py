#!/usr/bin/env python3
"""Reconstruct the first shifted bootstrap prompt from the replayed stream."""
import json
from pathlib import Path
import sys
sys.path.insert(0,'/root/autodl-tmp/gaussiangpt_multichunk_5090_20261028/source')
sys.path.insert(0,'/root/autodl-tmp/gaussiangpt_distribution_5090_20261028/source')
import run_scene as rs
from probe import compare,law_point,law_feature
import torch
from threadpoolctl import threadpool_limits
from draft_a.pf_ranker_eval import cache_for,typed_kinds,position_state,policy
from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock,fork_position_state
from gpt_position_spec_decode_smoke import choose_position
from dynamic_target_graph import DynamicTargetGraph
from tree_runtime import TargetTreeRuntime
from position_rescue_runtime import PositionRescueRuntime
from position_rescue_decoder import _best_two_indices

OUT=Path('/root/autodl-tmp/gaussiangpt_distribution_5090_20261028')

class Captured(Exception): pass

def main():
    with torch.inference_mode(),threadpool_limits(limits=1):
        gpt,model,ctx=rs.load_models()
        adapter=rs.api.load_method('pf_mlp',ctx)
        original_bos=model.sample_sequence
        tokens=torch.load(OUT/'bootstrap_replay_tokens.pt',map_location='cpu',weights_only=True)
        model.sample_sequence=lambda *args,**kwargs: tokens.view(1,-1).to('cuda')
        lut=rs.scene._build_pos_token_to_coord_lut(gpt)
        try:
            first=rs.scene._bootstrap_initial_context_block_like_parallel_sampling(
                gpt,scene_cols_x=60,scene_cols_y=60,context_chunks_x=20,context_chunks_y=20,z=20,
                temperature=.9,top_k=None,top_p=.9,feature_temperature=None,feature_top_k=None,
                feature_top_p=None,base_seed=0,rank=0,scene_idx=0,pos_token_to_coord=lut,
                debug_column_context=False)
        finally:
            model.sample_sequence=original_bos
        saved=model.sample_sequence_with_prompt
        captured={}
        def capture(prompt_tokens,*args,**kwargs):
            captured.update(prompt=prompt_tokens.tolist(),seed=kwargs['seed'],budget=kwargs['max_new_tokens'])
            raise Captured()
        model.sample_sequence_with_prompt=capture
        try:
            try:
                rs.scene._bootstrap_shifted_x_slices_with_y0_start(
                    gpt,scene_cols_x=60,scene_cols_y=60,context_chunks_x=20,context_chunks_y=20,
                    x_offset=5,z=20,temperature=.9,top_k=None,top_p=.9,feature_temperature=None,
                    feature_top_k=None,feature_top_p=None,base_seed=0,rank=0,scene_idx=0,
                    pos_token_to_coord=lut,initial_bootstrap_columns=first,debug_column_context=False)
            except Captured:
                pass
        finally:
            model.sample_sequence_with_prompt=saved
        original=json.loads((Path('/root/autodl-tmp/gaussiangpt_multichunk_5090_20261028')/'full_pf_01/pf_w1/sampler_calls.json').read_text())[1]
        matches=(captured['seed']==original['seed'] and len(captured['prompt'])==original['prompt_tokens'] and captured['budget']==original['budget'])
        prefix=[rs.EOS]+captured['prompt']; before=len(prefix)-1
        online=position_state(prefix,ctx['ranks']); proposed=online.propose(prefix[-2])
        ids,features,_,_=proposed
        best,backup=_best_two_indices(ctx['ranker'].predict_proba(features)[:,1])
        p1=int(ids[best]); pb=int(ids[backup]) if backup is not None else None
        shadow=fork_position_state(online); shadow.finalize(p1)
        p2=int(choose_position(ctx['ranker'],shadow,p1)[0])
        cap=CaptureLastBlock(ctx['target'])
        try:
            c=cache_for(ctx['target'],len(prefix)+8)
            _,hs=cap.run(ctx['target'],c,prefix[:-1],typed_kinds(0,before),'cuda')
            adapter.model.prepare(prefix,len(prefix)+8)
            rng=torch.Generator(device='cuda').manual_seed(67)
            q=adapter.model.propose(prefix,p1,hs[-1].clone(),rng)
            f=int(torch.multinomial(q,1,generator=rng))+8000
            adapter.model.close()
            ref,_=cap.run(ctx['target'],c,[prefix[-1],p1,f,p2],typed_kinds(before,4),'cuda'); ref=ref.clone()
            runtime=DynamicTargetGraph(ctx['target'],cache_capacity_buckets=(16384,))
            tree=TargetTreeRuntime(ctx['target'],runtime,graph=True,max_width=1)
            ct=cache_for(ctx['target'],len(prefix)+8)
            cap.run(ctx['target'],ct,prefix[:-1],typed_kinds(0,before),'cuda')
            tr,_=tree.verify(ct,prefix[-1],p1,[f],[p2]); tr=tr.reshape(-1,tr.shape[-1]).clone(); tree.commit(ct,[0])
            pairs={f'w1_node_{i}':compare(ref[i],tr[i]) for i in range(4)}
            if pb is not None:
                cb=cache_for(ctx['target'],len(prefix)+8)
                cap.run(ctx['target'],cb,prefix[:-1],typed_kinds(0,before),'cuda')
                br,_=cap.run(ctx['target'],cb,[prefix[-1],pb],typed_kinds(before,2),'cuda'); br=br.clone()
                cr=cache_for(ctx['target'],len(prefix)+8)
                cap.run(ctx['target'],cr,prefix[:-1],typed_kinds(0,before),'cuda')
                rescue=PositionRescueRuntime(ctx['target'],runtime,graph=True)
                rr,_=rescue.verify(cr,prefix[-1],p1,[f],[p2],backup_position=pb); rr=rr.reshape(-1,rr.shape[-1]).clone(); rescue.commit(cr,[0])
                pairs.update({f'rescue_node_{i}':compare(ref[i],rr[i]) for i in range(4)})
                pairs['rescue_backup']=compare(br[1],rr[4])
            result=dict(status='complete',seed=captured['seed'],budget=captured['budget'],
                        prompt_tokens=before,metadata_matches=matches,comparisons=pairs,
                        position_tv=law_point(policy(tr[0]),p1),
                        feature_law=law_feature(policy(tr[1])[8000:rs.EOS],q),
                        limitation='Prompt reconstructed from replay; original prompt contents were not saved.')
            (OUT/'shifted_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            print(json.dumps(result),flush=True)
        finally:
            cap.close()

if __name__=='__main__': main()
