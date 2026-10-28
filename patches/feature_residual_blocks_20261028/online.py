"""Use the existing W1 comparison for models with extra Feature residual blocks."""
import importlib.util
from pathlib import Path
import sys

import torch

from block_model import from_checkpoint


def main():
    # Wrapper-specific arguments precede the existing comparison arguments.
    script = Path(sys.argv[1]).resolve()
    training_root = Path(sys.argv[2]).resolve()
    sys.argv = [str(script)] + sys.argv[3:]
    spec = importlib.util.spec_from_file_location('block_comparison', script)
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    original_expand = compare.expand_methods
    original_load = compare.load_feature
    custom = {'blocks0': 0, 'blocks3': 3, 'blocks5': 5}

    def expand(names, seeds):
        result = []
        for name in names:
            result.extend([name] if name in custom else original_expand([name], seeds))
        if len(set(result)) != len(result):
            raise ValueError('Duplicate method')
        return result

    def load(name, context, args, api):
        if name not in custom:
            return original_load(name, context, args, api)
        from draft_a.pf_feature_unified import MLPAdapter
        from draft_a.feature_target_v2 import HEAD
        path = training_root / f'add{custom[name]}' / 'selected.pt'
        value = torch.load(path, map_location='cpu', weights_only=True)
        head = torch.load(HEAD, map_location='cpu', weights_only=True)['weight']
        target_head = context['target'].lm_head.weight[8000:12096].detach().float().cpu()
        if not torch.equal(head.float(), target_head):
            raise ValueError('Feature output weights differ from Target')
        model = from_checkpoint(value, head, context['device']).eval().requires_grad_(False)
        metadata = dict(checkpoint=str(path), added_blocks=custom[name], total_blocks=1+custom[name],
                        parameters=sum(p.numel() for p in model.parameters()),
                        checkpoint_step=value['step'], validation=value['validation']['pooled_overlap'],
                        decoder='tree_decoder.decode(width=1)', hidden_state='pre-finalnorm')
        return api.MethodAdapter(name, metadata, model=MLPAdapter(model, bf16=False), context=context)

    compare.expand_methods = expand
    compare.load_feature = load
    compare.main()


if __name__ == '__main__':
    main()
