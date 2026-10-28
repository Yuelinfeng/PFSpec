"""Validate each selected Transformer before the online comparison."""
import json
from pathlib import Path
import torch

from train import validate


def main():
    root=Path(__file__).resolve().parent
    out=root/'selected_validation'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assets=torch.load(root/'fixed_assets.pt',map_location='cpu',weights_only=True)
    plan=json.loads((root/'train_01/20261018/mlp/data_plan.json').read_text())
    sample=torch.load(plan['splits']['train']['records'][0]['path'],weights_only=True)
    for seed in (20261018,20261019,20261020):
        value=torch.load(root/'train_01'/str(seed)/'transformer/selected.pt',map_location='cpu',weights_only=True)
        value['training_seed']=seed
        validate(out,assets,value,sample)


if __name__=='__main__':main()
