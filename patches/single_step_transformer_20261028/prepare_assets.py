"""Read the existing Target embedding and Feature output weights on CPU."""
import argparse
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpt',type=Path,required=True)
    p.add_argument('--feature-head',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    value=torch.load(args.gpt,map_location='cpu',weights_only=False)['state_dict']
    token_key=next(k for k in value if k.endswith('transformer.wte.weight'))
    head_key=next(k for k in value if k.endswith('transformer.lm_head.weight'))
    tokens=value[token_key].float().contiguous()
    feature=value[head_key][8000:12096].float().contiguous().clone()
    reference=torch.load(args.feature_head,map_location='cpu',weights_only=True)['weight']
    if not torch.equal(feature,reference):raise ValueError('Feature output weights differ')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(feature_weight=feature,token_embeddings=tokens),args.out)


if __name__=='__main__':main()
