#!/usr/bin/env python3
"""Compare tree rows and committed KV paths with ordinary causal Target calls."""
import argparse
from pathlib import Path
import sys,json,math

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'reference'))
sys.path.insert(0,str(HERE/'reference/runtime'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--prompts',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--contexts',type=int,default=2)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Output directory already exists')
    args.out.mkdir(parents=True)
    sys.path.insert(0,str(args.project))
    import os
    os.chdir(args.project)
    import torch
    from threadpoolctl import threadpool_limits
    from method_adapters import build_context,load_method
    from dynamic_target_graph import DynamicTargetGraph
    from tree_runtime import TargetTreeRuntime
    from draft_a.pf_ranker_eval import cache_for,policy,typed_kinds
    from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock,fork_position_state
    from draft_a.pf_ranker_eval import position_state
    from gpt_position_spec_decode_smoke import choose_position
    from draft_a.learning_eval import _fp32_runtime
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    contexts=json.loads(args.prompts.read_text())['contexts'][:args.contexts]
    ctx=build_context(args.project)
    target=ctx['target']; adapter=load_method('pf_mlp',ctx).model
    rows=[]; tensors=[]

    def compare(label,left,right,probability=False):
        left=left.detach().float();right=right.detach().float()
        same_finite=torch.isfinite(left)==torch.isfinite(right)
        finite=torch.isfinite(left)&torch.isfinite(right)
        diff=(left[finite]-right[finite]).abs()
        maxdiff=float(diff.max()) if diff.numel() else 0.
        a=left[finite].double();b=right[finite].double()
        relative_l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))
        relative_max=maxdiff/max(float(b.abs().max()) if b.numel() else 0.,1e-12)
        record={'label':label,'finite_mask_equal':bool(same_finite.all()),'max_abs':maxdiff,
                'allclose':bool(torch.allclose(left,right,atol=2e-5,rtol=2e-5,equal_nan=False)),
                'all_finite':bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
                'relative_l2':relative_l2,'relative_max':relative_max}
        if probability:
            p=policy(left);q=policy(right)
            record.update(policy_tv=float((p-q).abs().sum()/2),
                          policy_support_equal=bool(((p>0)==(q>0)).all()),
                          policy_max_abs=float((p-q).abs().max()))
            tensors.append({'label':label,'left':left.cpu(),'right':right.cpu(),'p':p.cpu(),'q':q.cpu()})
        record['passed']=(record['finite_mask_equal'] and
                          ((record['allclose'] and record['policy_tv']<=1e-5 and record['policy_support_equal'])
                           if probability else(record['all_finite'] and relative_l2<=2e-6 and relative_max<=2e-6)))
        rows.append(record)

    with threadpool_limits(limits=1),torch.inference_mode(),_fp32_runtime(torch),torch.autocast('cuda',enabled=False):
        for pi,prompt_record in enumerate(contexts):
            prompt=prompt_record['prompt'];before=len(prompt)-1
            if before<2:raise ValueError('Real complete P/F prefix required')
            online=position_state(prompt,ctx['ranks'])
            first=choose_position(ctx['ranker'],online,prompt[-2])
            if first is None:raise ValueError('No legal first Ranker proposal')
            shadow=fork_position_state(online);shadow.finalize(first[0])
            second=choose_position(ctx['ranker'],shadow,first[0])
            if second is None:raise ValueError('No legal second Ranker proposal')
            p1,p2=first[0],second[0]
            after_p2=fork_position_state(shadow);after_p2.finalize(p2)
            third=choose_position(ctx['ranker'],after_p2,p2)
            p3=third[0] if third is not None else None
            setup=cache_for(target,1024)
            cap=CaptureLastBlock(target)
            try:
                _,hs=cap.run(target,setup,prompt[:-1],typed_kinds(0,before),'cuda')
                adapter.prepare(prompt,1024)
                rng=torch.Generator(device='cuda').manual_seed(77501+pi)
                q=adapter.propose(prompt,p1,hs[-1].clone(),rng)
                fs=(torch.multinomial(q,2,replacement=False,generator=rng)+8000).tolist()
                adapter.close()
            finally:cap.close()
            for width in [1,2]:
                branches=[]
                for j,f in enumerate(fs[:width]):
                    c=cache_for(target,1024);cap=CaptureLastBlock(target)
                    try:
                        cap.run(target,c,prompt[:-1],typed_kinds(0,before),'cuda')
                        tokens=[prompt[-1],p1,f,p2]
                        logits,hidden=cap.run(target,c,tokens,typed_kinds(before,4),'cuda')
                        branch={'tokens':tokens,'logits':logits.clone(),'hidden':hidden.clone(),
                                'k':c.k_cache[:,:,:before+4].clone(),'v':c.v_cache[:,:,:before+4].clone()}
                        linear_ref=DynamicTargetGraph(target,cache_capacity_buckets=(1024,))
                        lc=cache_for(target,1024)
                        cap.run(target,lc,prompt[:-1],typed_kinds(0,before),'cuda')
                        with linear_ref.attention_context():
                            ll,lh=cap.run(target,lc,tokens,typed_kinds(before,4),'cuda')
                        linear_stats=linear_ref.stats()
                        rows.append({'label':f'bmm_reference_p{pi}_w{width}_j{j}',
                                     'stats':linear_stats,
                                     'passed':linear_stats.get('calls',{}).get('graph_forward',0)>0
                                     and not linear_stats.get('fallbacks')})
                        branch.update(bmm_logits=ll.clone(),bmm_hidden=lh.clone(),
                                      bmm_k=lc.k_cache[:,:,:before+4].clone(),
                                      bmm_v=lc.v_cache[:,:,:before+4].clone())
                        branches.append(branch)
                        del lc,linear_ref
                    finally:cap.close()
                for graph in [False,True]:
                    linear=DynamicTargetGraph(target,cache_capacity_buckets=(1024,))
                    tr=TargetTreeRuntime(target,linear,graph=graph,max_width=4)
                    for branch_index in range(width):
                        for keep in [1,2,3,4]:
                            c=cache_for(target,1024)
                            cap=CaptureLastBlock(target)
                            try:cap.run(target,c,prompt[:-1],typed_kinds(0,before),'cuda')
                            finally:cap.close()
                            logits,hidden=tr.verify(c,prompt[-1],p1,fs[:width],[p2]*width)
                            logits=logits.clone();hidden=hidden.clone()
                            if int(c.get_pos())!=before:raise AssertionError('Tree advanced logical cache before commit')
                            if logits.ndim==3:logits=logits[0]
                            if hidden.ndim==3:hidden=hidden[0]
                            for j,b in enumerate(branches):
                                indices=[0,1,2+2*j,3+2*j]
                                for depth,node in enumerate(indices):
                                    compare(f'p{pi}_w{width}_g{int(graph)}_b{branch_index}_keep{keep}_branch{j}_row{depth}',logits[node],b['logits'][depth],True)
                                    compare(f'p{pi}_w{width}_g{int(graph)}_b{branch_index}_keep{keep}_branch{j}_h{depth}',hidden[node],b['hidden'][depth])
                                    compare(f'bmm_p{pi}_w{width}_g{int(graph)}_b{branch_index}_keep{keep}_branch{j}_row{depth}',logits[node],b['bmm_logits'][depth],True)
                                    compare(f'bmm_p{pi}_w{width}_g{int(graph)}_b{branch_index}_keep{keep}_branch{j}_h{depth}',hidden[node],b['bmm_hidden'][depth])
                                    if depth==1 or(depth==3 and p3 is not None):
                                        from collect_feature_teacher_from_s3 import policy_batch
                                        pending=b['tokens'][2] if depth==1 else int(torch.argmax(b['logits'][3]))
                                        prevf=torch.tensor([pending-8000],device='cuda')
                                        nextp=torch.tensor([p2 if depth==1 else p3],device='cuda')
                                        _,ql=adapter.feature(hidden[node].unsqueeze(0),prevf,nextp)
                                        _,qr=adapter.feature(b['hidden'][depth].unsqueeze(0),prevf,nextp)
                                        ql=policy_batch(ql.float(),.9,.9)[0];qr=policy_batch(qr.float(),.9,.9)[0]
                                        tv=float((ql-qr).abs().sum()/2)
                                        support=bool(((ql>0)==(qr>0)).all())
                                        finite=bool(torch.isfinite(ql).all() and torch.isfinite(qr).all())
                                        mass_error=max(abs(float(ql.sum())-1),abs(float(qr.sum())-1))
                                        rows.append({'label':f'mlp_q_p{pi}_w{width}_g{int(graph)}_b{branch_index}_keep{keep}_j{j}_depth{depth}',
                                                     'policy_tv':tv,'policy_support_equal':support,
                                                     'all_finite':finite,'mass_error':mass_error,
                                                     'passed':finite and mass_error<=1e-5 and tv<=1e-5 and support})
                                        tensors.append({'label':rows[-1]['label'],'p':ql.cpu(),'q':qr.cpu()})
                            indices=[0,1,2+2*branch_index,3+2*branch_index][:keep]
                            tr.commit(c,indices)
                            if int(c.get_pos())!=before+keep:raise AssertionError('Wrong committed cache position')
                            b=branches[branch_index]
                            compare('committed_key',c.k_cache[:,:,:before+keep],b['k'][:,:,:before+keep])
                            compare('committed_value',c.v_cache[:,:,:before+keep],b['v'][:,:,:before+keep])
                            compare('bmm_committed_key',c.k_cache[:,:,:before+keep],b['bmm_k'][:,:,:before+keep])
                            compare('bmm_committed_value',c.v_cache[:,:,:before+keep],b['bmm_v'][:,:,:before+keep])
                            expected=prompt[:-1]+b['tokens'][:keep]
                            if c.get_tokens(before+keep).tolist()[0]!=expected:raise AssertionError('Wrong committed token path')
                            continuation=int(torch.argmax(b['logits'][3]))
                            if keep<4:
                                next_p=policy(b['logits'][keep-1]).clone()
                                next_p[b['tokens'][keep]]=0
                                next_p[12096]=0
                                if bool(next_p.sum()>0):
                                    continuation=int(torch.argmax(next_p))
                                else:
                                    # A top-p point has no rejection path; still exercise
                                    # cache reuse with a different type-valid token.
                                    alternate=b['logits'][keep-1].clone()
                                    alternate[b['tokens'][keep]]=float('-inf')
                                    alternate[12096]=float('-inf')
                                    if not bool(torch.isfinite(alternate).any()):
                                        raise AssertionError('No alternate type-valid token')
                                    continuation=int(torch.argmax(alternate))
                            reference=cache_for(target,1024);cap=CaptureLastBlock(target)
                            try:
                                cap.run(target,reference,expected,typed_kinds(0,len(expected)),'cuda')
                                rl,rh=cap.run(target,reference,[continuation],typed_kinds(before+keep,1),'cuda')
                                with linear.attention_context():
                                    tl,th=cap.run(target,c,[continuation],typed_kinds(before+keep,1),'cuda')
                                compare('after_commit_logit',tl[0],rl[0],True)
                                compare('after_commit_hidden',th[0],rh[0])
                            finally:cap.close()
                            del c,reference
                    rows.append({'label':f'runtime_p{pi}_w{width}_g{int(graph)}','passed':True,'stats':tr.stats()})
                    del tr,linear
            del setup,branches
        torch.cuda.synchronize()
    status={'status':'complete' if all(r['passed'] for r in rows) else 'failed','contexts':len(contexts),
            'comparisons':len(rows),'passed':sum(r['passed'] for r in rows),
            'max_abs':max((r.get('max_abs',0) for r in rows),default=0),
            'max_policy_tv':max((r.get('policy_tv',0) for r in rows),default=0),
            'max_relative_l2':max((r.get('relative_l2',0) for r in rows),default=0),
            'max_relative_max':max((r.get('relative_max',0) for r in rows),default=0),
            'elementwise_allclose_failures':sum(r.get('allclose') is False for r in rows),
            'policy_support_differences':sum(not r.get('policy_support_equal',True) for r in rows),
            'rows':rows}
    (args.out/'verification.json').write_text(json.dumps(status,indent=2)+'\n')
    torch.save(tensors,args.out/'raw_policy_pairs.pt')
    print(json.dumps({k:v for k,v in status.items() if k!='rows'}))
    if status['status']!='complete':raise ArithmeticError('Tree numerical comparisons failed')

if __name__=='__main__':main()
