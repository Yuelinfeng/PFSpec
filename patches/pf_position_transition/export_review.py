"""Export captured labels and CPU-evaluated saved students for review."""
import argparse
import csv
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();start=time.time()
    import numpy as np
    import torch
    from train_transition import load_teacher
    from transition_model import TransitionModel,policy_batch
    torch.set_num_threads(4)
    out=root/'review_export';out.mkdir(exist_ok=False)
    teacher_path=root/'g1_run01/teacher/teacher.pt'
    raw=torch.load(teacher_path,map_location='cpu',weights_only=True)
    data,info=load_teacher(torch,teacher_path)
    omit={'target_logits_position'}
    arrays={k:v.numpy() for k,v in raw.items() if torch.is_tensor(v) and k not in omit}
    np.savez_compressed(out/'teacher_labels.npz',**arrays)
    (out/'teacher_metadata.json').write_text(json.dumps(raw['row_metadata'],indent=1,allow_nan=False))
    val=[i for i,s in enumerate(data['split']) if s=='val']
    q={'A':data['q_old'][val]};predictions={}
    target_wte=torch.load(root/'g1_run01/teacher/token_embeddings.pt',map_location='cpu',weights_only=True)
    decision=json.loads((root/'g2_run01/decision.json').read_text())
    for run in decision['runs']:
        saved=torch.load(run['checkpoint'],map_location='cpu',weights_only=True)
        state=saved['model_state']
        mapping={name:state['hidden_'+name] for name in ('mean','basis','scale')}
        model=TransitionModel(run['method'],target_wte,mapping,size=saved['size']).eval()
        model.load_state_dict(state)
        values=[]
        with torch.inference_mode():
            for begin in range(0,len(val),128):
                ids=torch.tensor(val[begin:begin+128])
                logits=model(data['old_logits_feature'][ids],data['h0'][ids],
                    data['previous_feature_id'][ids],data['previous_position_id'][ids],
                    data['candidate_position_id'][ids])
                values.append(policy_batch(logits))
        label=('B' if run['method']=='direct' else 'C')+str(run['seed'])
        q[label]=torch.cat(values)
        predictions[label]=run
    labels={label:t.numpy() for label,t in q.items()}
    labels.update(p_f=data['p_f_policy'][val].numpy(),p_p1=data['p_p1'][val].numpy(),row_ids=np.array(val))
    np.savez_compressed(out/'validation_q.npz',**labels)
    rows=[];summary={};max_gpu_cpu_mean_difference=0.
    for label,probs in q.items():
        if not torch.isfinite(probs).all() or (probs<0).any():
            raise ValueError('Invalid saved-student probabilities')
        torch.testing.assert_close(probs.sum(-1),torch.ones(len(val)),atol=2e-6,rtol=0)
        overlaps=torch.minimum(probs,data['p_f_policy'][val]).sum(-1).tolist()
        groups={}
        for row_id,v in zip(val,overlaps):
            s=data['source_id'][row_id];w=float(data['p_p1'][row_id])
            groups.setdefault(s,[]).append((v,w))
            rows.append(dict(method=label,row_id=row_id,source_id=s,overlap=v,p_p1=w))
        source={s:dict(overlap=sum(v for v,w in vv)/len(vv),
            weighted_overlap=sum(v*w for v,w in vv)/sum(w for v,w in vv),rows=len(vv)) for s,vv in groups.items()}
        mean=sum(v['overlap'] for v in source.values())/len(source)
        summary[label]=dict(source_equal_overlap=mean,per_source=source)
        if label!='A':
            difference=abs(mean-predictions[label]['validation']['source_equal_overlap'])
            max_gpu_cpu_mean_difference=max(max_gpu_cpu_mean_difference,difference)
    if max_gpu_cpu_mean_difference>1e-5:
        raise ValueError('Saved checkpoint CPU score differs from recorded validation')
    with (out/'overlap_rows.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'cpu_validation.json').write_text(json.dumps(dict(status='complete',device='cpu',
        teacher=info,models=summary,max_gpu_cpu_mean_difference=max_gpu_cpu_mean_difference,
        elapsed_seconds=time.time()-start,omitted_array_fields=sorted(omit),
        full_teacher_retained_remote=str(teacher_path)),indent=2))
    print(json.dumps(dict(status='complete',max_gpu_cpu_mean_difference=max_gpu_cpu_mean_difference,
        elapsed_seconds=time.time()-start)))


if __name__=='__main__':
    main()
