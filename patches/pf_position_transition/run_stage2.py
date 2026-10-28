"""Run the four students and apply the predeclared conditional-overlap criteria."""
import time
STARTED = time.time()
import argparse
import json
from pathlib import Path
import signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    args = parser.parse_args()
    deadline = min(args.deadline_unix, STARTED+2400)
    def alarm(*_):
        raise TimeoutError('G2 shared 2400-second limit reached')
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline-time.time()))
    import torch
    from train_transition import run_g2, load_teacher
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root=args.root.resolve()
    prior=json.loads((root/'g1_run02/pipeline.json').read_text())
    if prior.get('status')!='complete':
        raise RuntimeError('G1 has not completed')
    size=prior['selected_size']
    teacher=root/'g1_run01/teacher/teacher.pt'
    tokens=root/'g1_run01/teacher/token_embeddings.pt'
    out=root/'g2_run01'
    if out.exists():
        raise FileExistsError('G2 output already exists')
    result=run_g2(torch,teacher,tokens,out,size,1200,64,3e-4,'cuda',deadline)
    if result['status']!='complete':
        raise RuntimeError('Four-model training is incomplete')
    data,_=load_teacher(torch,teacher)
    val=[i for i,s in enumerate(data['split']) if s=='val']
    rows={}
    overlap=torch.minimum(data['q_old'][val],data['p_f_policy'][val]).sum(-1).tolist()
    weights=data['p_p1'][val].tolist()
    for i,v,w in zip(val,overlap,weights):
        rows.setdefault(data['source_id'][i],[]).append((v,w))
    per_source={s:dict(overlap=sum(v for v,w in vv)/len(vv),
        weighted_overlap=sum(v*w for v,w in vv)/sum(w for v,w in vv),rows=len(vv))
        for s,vv in rows.items()}
    original=dict(source_equal_overlap=sum(r['overlap'] for r in per_source.values())/len(per_source),
        source_equal_p1_weighted_overlap=sum(r['weighted_overlap'] for r in per_source.values())/len(per_source),
        per_source=per_source)
    runs=[]
    for method in ('direct','two_step'):
        for seed in (17,29):
            checkpoint=out/f'{method}_seed{seed}/best.pt'
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            best=saved['best_validation']
            runs.append(dict(method=method,seed=seed,update=saved['update'],
                checkpoint=str(checkpoint),validation=best['validation']))
    selected={m:max([r for r in runs if r['method']==m],
        key=lambda r:r['validation']['source_equal_overlap']) for m in ('direct','two_step')}
    b=selected['direct']['validation'];c=selected['two_step']['validation']
    c_a=c['source_equal_overlap']-original['source_equal_overlap']
    c_b=c['source_equal_overlap']-b['source_equal_overlap']
    source_changes={s:c['per_source'][s]['mean_overlap']-r['overlap'] for s,r in per_source.items()}
    criteria=dict(c_minus_a_at_least_8pp=c_a>=.08,c_minus_b_at_least_4pp=c_b>=.04,
        every_val_source_nonnegative=all(v>=0 for v in source_changes.values()))
    summary=dict(status='complete',size=size,original=original,runs=runs,selected=selected,
        c_minus_a_pp=100*c_a,c_minus_b_pp=100*c_b,
        c_minus_a_per_source_pp={s:100*v for s,v in source_changes.items()},
        criteria=criteria,proceed_g3=all(criteria.values()),
        elapsed_seconds=time.time()-STARTED,deadline_unix=deadline,
        online_acceptance='not_run',new_speedup='not_run')
    (out/'decision.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({k:summary[k] for k in ('status','size','c_minus_a_pp','c_minus_b_pp','criteria','proceed_g3','elapsed_seconds')}),flush=True)
    signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    main()
