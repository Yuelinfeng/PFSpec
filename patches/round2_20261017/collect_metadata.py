"""Record small checkpoint metadata while retaining weights at their existing paths."""
import collections
import json
from pathlib import Path
import torch

ROOT=Path('/root/autodl-tmp/gaussiangpt_round2_20261017')
PROJECT=Path('/root/workspace/GaussianGPT')
PREVIOUS=Path('/root/autodl-tmp/gaussiangpt_next_round_20261017/step5_method_compare')


def clean(value):
    if torch.is_tensor(value):return dict(shape=list(value.shape),dtype=str(value.dtype))
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    return str(value)


def main():
    paths=list((ROOT/'b1_training').glob('*/*/selected.pt'))+list((ROOT/'a2_training').glob('*/selected.pt'))
    paths += [PREVIOUS/'medusa_lowrank/medusa_linear_heads_best.pt',PREVIOUS/'hydra_width177/medusa_conditional_heads_best.pt',
        PREVIOUS/'pf_common_traces/selected.pt',PROJECT/'draft_a/runs/feature_target_continue_20261014/selected.pt',
        PROJECT/'profiling/eagle/results/e46_hprev_dual_20ep/last.pt']
    records=[]
    for path in paths:
        blob=torch.load(path,map_location='cpu',weights_only=False)
        state=next((blob[k] for k in ('state','heads','model_state') if k in blob),{})
        records.append(dict(path=str(path),bytes=path.stat().st_size,mtime_unix=path.stat().st_mtime,
            tensor_dtypes=dict(collections.Counter(str(v.dtype) for v in state.values() if torch.is_tensor(v))),
            stored_state_elements=sum(v.numel() for v in state.values() if torch.is_tensor(v)),
            metadata=clean({k:v for k,v in blob.items() if k not in ('state','heads','model_state','optimizer','optimizer_state')})))
    original=[]
    for p in [PROJECT/'checkpoints/gpt_vfront.ckpt',PROJECT/'checkpoints/vqvae_vfront.ckpt',
        PROJECT/'profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib']:
        original.append(dict(path=str(p),bytes=p.stat().st_size,mtime_unix=p.stat().st_mtime))
    (ROOT/'checkpoint_metadata.json').write_text(json.dumps(dict(checkpoints=records,original_assets=original),indent=2))
    print('Recorded',len(records),'small checkpoints and',len(original),'original asset file records')


if __name__=='__main__':main()
