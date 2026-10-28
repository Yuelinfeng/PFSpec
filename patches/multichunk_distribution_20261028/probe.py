#!/usr/bin/env python3
"""Compare real scene contexts, Target probabilities and speculative tree laws."""
import json
import sys
from pathlib import Path

ROOT = Path('/root/autodl-tmp/gaussiangpt_multichunk_5090_20261028')
sys.path.insert(0, str(ROOT / 'source'))
import run_scene as rs
import torch
from threadpoolctl import threadpool_limits
from draft_a.pf_ranker_eval import cache_for, policy, typed_kinds, position_state
from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock, fork_position_state
from position_rescue_decoder import _best_two_indices
from tree_sampling import torch_positive_residual, torch_point_residual
from dynamic_target_graph import DynamicTargetGraph
from tree_runtime import TargetTreeRuntime
from position_rescue_runtime import PositionRescueRuntime

OUT = Path('/root/autodl-tmp/gaussiangpt_distribution_5090_20261028')
OUT.mkdir(exist_ok=True)

def save(obj):
    (OUT / 'result.json').write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def native_policy(logits):
    scores = logits.float().unsqueeze(0) / .9
    ordered, indices = torch.sort(scores, descending=True, dim=-1)
    removed = ordered.softmax(-1).cumsum(-1) > .9
    removed[:, 1:] = removed[:, :-1].clone()
    removed[:, 0] = False
    filtered = torch.full_like(scores, -torch.inf)
    filtered.scatter_(1, indices, ordered.masked_fill(removed, -torch.inf))
    return filtered.softmax(-1)[0]

def compare(a, b):
    pa, pb = native_policy(a), policy(b)
    return dict(tv=float((pa-pb).abs().sum()/2),
                max_abs=float((pa-pb).abs().max()),
                support_differences=int(((pa>0)!=(pb>0)).sum()),
                eos_native=float(pa[rs.EOS]), eos_tree=float(pb[rs.EOS]))

def law_point(p, candidate):
    p = p.float()
    if float(p[candidate]) >= 1.0:
        return 0.0
    residual = torch_point_residual(p, candidate).double()
    q = p.double(); alpha = float(p[candidate])
    result = residual * (1-alpha)
    result[candidate] += alpha
    return float((result-q).abs().sum()/2)

def law_feature(p, q):
    p, q = p.float(), q.float()
    qlaw = q.double()/q.double().sum()
    ratio = torch.where(q>0, (p/q).clamp(max=1), torch.zeros_like(q)).double()
    accepted = qlaw*ratio
    rejected = float((qlaw*(1-ratio)).sum())
    if rejected == 0:
        result = accepted
    else:
        result = accepted + rejected*torch_positive_residual(p, q).double()
    return dict(tv=float((result-p.double()).abs().sum()/2),
                rejection_probability=rejected, q_sum=float(q.double().sum()),
                p_sum=float(p.double().sum()))

def reconstruct(gpt):
    src = ROOT / 'full_pf_01/pf_w1'
    data = torch.load(src/'tokens.pt', map_location='cpu', weights_only=True)
    seq = data['tokens_sequence'].view(-1)
    lengths = data['column_token_lengths']
    lut = rs.scene._build_pos_token_to_coord_lut(gpt).cpu()
    grid = [[[] for _ in range(60)] for _ in range(60)]
    i = 0
    for x in range(60):
        for y in range(60):
            n = int(lengths[x,y]); pairs = seq[i:i+n].reshape(-1,2); i += n
            for pos, feat in pairs:
                grid[x][y].append((int(lut[int(pos),2]), feat.view(1).clone()))
    assert i == seq.numel()
    calls = json.loads((src/'sampler_calls.json').read_text())
    call_by_seed = {int(c['seed']): c for c in calls}
    candidates = []
    # A seed identifies the invocation; saved metadata records its prompt length.
    for x in range(60):
        for y in range(60):
            found = []
            col = x*60+y
            seed = rs.scene._seed_for_step(0,0,0,col,0)
            if seed in call_by_seed:
                found.append(('extension', seed))
            for retry in (1,2,3,4,5):
                seed = rs.scene._seed_for_step(0,0,0,col,1000+y*131+retry)
                if seed in call_by_seed:
                    found.append(('single_retry', seed))
                for prev in range(min(4,y)+1):
                    seed = rs.scene._seed_for_step(0,0,0,col-prev,2000+y*131+retry)
                    if seed in call_by_seed:
                        found.append(('group_retry', seed))
            if not found:
                continue
            sx, sy, _, _ = rs.scene._outpainting_window_for_target(
                x,y,scene_cols_x=60,scene_cols_y=60,context_chunks_x=20,
                context_chunks_y=20,x_offset=5,y_offset=5,y_range=5)
            prompt = rs.scene._build_sparse_prompt_tokens_from_window(
                grid,target_x=x,target_y=y,start_x=sx,start_y=sy,
                scene_cols_x=60,scene_cols_y=60,context_chunks_x=20,
                context_chunks_y=20,num_position_tokens=int(gpt.num_position_tokens),
                base_side_length=int(gpt.position_side_length)).tolist()
            for stage, seed in found:
                record = call_by_seed[seed]
                if len(prompt) == record['prompt_tokens'] and len(prompt) >= 2:
                    candidates.append(dict(stage=stage, x=x, y=y, seed=seed,
                                           prompt=prompt, recorded_length=len(prompt),
                                           budget=record['budget'], eos=record['eos']))
                    break
    chosen = []
    for stage in ('extension','single_retry','group_retry'):
        items = [c for c in candidates if c['stage']==stage]
        if items:
            for index in sorted(set([0,len(items)//4,len(items)//2,3*len(items)//4,len(items)-1])):
                chosen.append(items[index])
    return chosen, len(candidates)

def main():
    save(dict(status='loading'))
    with torch.inference_mode(), threadpool_limits(limits=1):
        gpt, model, ctx = rs.load_models()
        adapter = rs.api.load_method('pf_mlp',ctx)
        target = ctx['target']
        chosen, matched = reconstruct(gpt)
        print('reconstructed', matched, 'selected', len(chosen), flush=True)
        # Recreate the early bootstrap trajectory with its original seed.
        arm = rs.arm('pf_w1',ctx,adapter,True)
        try:
            _, suffix = rs.compare.run_arm(arm,ctx,[rs.EOS],128,0,rs.api)
        finally:
            rs.compare.close_arm(arm)
        for n in (2,32,96):
            if len(suffix)>=n and rs.EOS not in suffix[:n]:
                chosen.insert(0, dict(stage='bootstrap_replay', x=0,y=0,seed=0,
                                      prompt=suffix[:n], recorded_length=n,budget=128,eos=False))
        results=[]
        for ci,case in enumerate(chosen):
            prefix=[rs.EOS]+case['prompt']; before=len(prefix)-1
            online=position_state(prefix,ctx['ranks'])
            first=online.propose(prefix[-2])
            if first is None:
                print('skip no proposal',ci,flush=True); continue
            ids, features, _, _=first
            scores=ctx['ranker'].predict_proba(features)[:,1]
            best, backup = _best_two_indices(scores)
            if backup is None:
                print('skip no backup',ci,flush=True)
                continue
            p1, pb = int(ids[best]), int(ids[backup])
            shadow=fork_position_state(online); shadow.finalize(p1)
            second=shadow.propose(p1)
            if second is None:
                print('skip no second',ci,flush=True)
                continue
            ids2, f2, _, _=second
            p2=int(ids2[_best_two_indices(ctx['ranker'].predict_proba(f2)[:,1],include_backup=False)[0]])
            cap=CaptureLastBlock(target)
            try:
                c=cache_for(target,len(prefix)+8)
                _, hidden=cap.run(target,c,prefix[:-1],typed_kinds(0,before),'cuda')
                adapter.model.prepare(prefix,len(prefix)+8)
                rng=torch.Generator(device='cuda').manual_seed(31000+ci)
                q=adapter.model.propose(prefix,p1,hidden[-1].clone(),rng)
                feature=int(torch.multinomial(q,1,generator=rng))+8000
                adapter.model.close()
                ref_logits,_=cap.run(target,c,[prefix[-1],p1,feature,p2],typed_kinds(before,4),'cuda')
                ref_logits=ref_logits.clone()
                cb=cache_for(target,len(prefix)+8)
                cap.run(target,cb,prefix[:-1],typed_kinds(0,before),'cuda')
                backup_logits,_=cap.run(target,cb,[prefix[-1],pb],typed_kinds(before,2),'cuda')
                backup_logits=backup_logits.clone()
                cn=cache_for(target,len(prefix)+8)
                implicit=target(torch.tensor([prefix],device='cuda'),kv_cache=cn)[0,-1].clone()
                runtime = DynamicTargetGraph(target, cache_capacity_buckets=(16384,))
                tree = TargetTreeRuntime(target,runtime,graph=True,max_width=1)
                rescue = PositionRescueRuntime(target,runtime,graph=True)
                ct=cache_for(target,len(prefix)+8)
                cap.run(target,ct,prefix[:-1],typed_kinds(0,before),'cuda')
                tr,_=tree.verify(ct,prefix[-1],p1,[feature],[p2]); tr=tr.reshape(-1,tr.shape[-1]).clone()
                tree.commit(ct,[0])
                cr=cache_for(target,len(prefix)+8)
                cap.run(target,cr,prefix[:-1],typed_kinds(0,before),'cuda')
                rr,_=rescue.verify(cr,prefix[-1],p1,[feature],[p2],backup_position=pb)
                rr=rr.reshape(-1,rr.shape[-1]).clone()
                rescue.commit(cr,[0])
            finally:
                cap.close()
            comparisons={'implicit_vs_typed':compare(implicit,ref_logits[0])}
            for j in range(4):
                comparisons[f'w1_node_{j}']=compare(ref_logits[j],tr[j])
                comparisons[f'rescue_node_{j}']=compare(ref_logits[j],rr[j])
            comparisons['rescue_backup']=compare(backup_logits[1],rr[4])
            p1law=law_point(policy(rr[0]),p1)
            p2law=law_point(policy(rr[2]),p2)
            flaw=law_feature(policy(rr[1])[8000:rs.EOS],q)
            row={k:v for k,v in case.items() if k!='prompt'}
            row.update(comparisons=comparisons, point1_tv=p1law, point2_tv=p2law,
                       feature_law=flaw, eos_probability=float(policy(rr[0])[rs.EOS]),
                       proposed=[p1,feature,p2], backup=pb)
            results.append(row)
            save(dict(status='running',matched_length_and_seed=matched,cases=results))
            print(json.dumps(dict(case=ci,stage=case['stage'],xy=[case['x'],case['y']],
                                  length=before,max_tv=max(x['tv'] for x in comparisons.values()),
                                  feature_tv=flaw['tv'])),flush=True)
            del c,cb,cn,ct,cr,ref_logits,backup_logits,tr,rr
            del tree,rescue,runtime
        save(dict(status='complete',matched_length_and_seed=matched,cases=results,
                  limitation='Reconstructed prompts match saved call seed and length; original prompt contents were not saved. Bootstrap is a 128-token replay, not the original full-budget raw stream.'))

if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        print('FAILED:',repr(e),flush=True)
        raise
