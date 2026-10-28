"""Load matched Feature mechanisms into the existing W1 comparison."""
import importlib.util
from pathlib import Path
import sys

import torch
from feature_model import load_model


def main():
    script, training_root, asset_path = [Path(p).resolve() for p in sys.argv[1:4]]
    sys.argv = [str(script)] + sys.argv[4:]
    spec = importlib.util.spec_from_file_location('single_step_comparison', script)
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    expand_old, load_old = compare.expand_methods, compare.load_feature
    assets = torch.load(asset_path, map_location='cpu', weights_only=True)

    def expand(names, seeds):
        result = []
        for name in names:
            if name.split(':s')[0] in ('mlp', 'transformer') and ':s' in name:
                result.append(name)
            else:
                result.extend(expand_old([name], seeds))
        if not result or len(set(result)) != len(result):
            raise ValueError('Method list must be nonempty and unique')
        return result

    def load(name, context, args, api):
        if name.split(':s')[0] not in ('mlp', 'transformer'):
            return load_old(name, context, args, api)
        from draft_a.pf_feature_unified import MLPAdapter
        kind, seed = name.split(':s')
        path = training_root / seed / kind / 'selected.pt'
        value = torch.load(path, map_location='cpu', weights_only=True)
        if value['model_config']['architecture'] != kind or value['model_config']['training_seed'] != int(seed):
            raise ValueError('Model identity differs from requested method')
        core = context['target']
        if not torch.equal(assets['feature_weight'], core.lm_head.weight[8000:12096].detach().float().cpu()):
            raise ValueError('Feature output weights differ from Target')
        if kind == 'transformer' and not torch.equal(assets['token_embeddings'], core.transformer.wte.weight.detach().float().cpu()):
            raise ValueError('Token embeddings differ from Target')
        runtime_assets = dict(feature_weight=core.lm_head.weight[8000:12096].detach(),
                             token_embeddings=core.transformer.wte.weight.detach())
        model = load_model(value, runtime_assets, context['device']).eval().requires_grad_(False)
        output_weight = model.feature_weight if kind == 'transformer' else model.frozen_head
        if output_weight.data_ptr() != runtime_assets['feature_weight'].data_ptr():
            raise RuntimeError('Feature output weights do not share Target storage')
        if kind == 'transformer' and model.token_embeddings.data_ptr() != runtime_assets['token_embeddings'].data_ptr():
            raise RuntimeError('Token embeddings do not share Target storage')
        metadata = dict(checkpoint=str(path), architecture=kind, training_seed=int(seed),
                        parameters=sum(p.numel() for p in model.parameters()),
                        step=value['step'], validation=value['validation']['pooled_overlap'],
                        decoder='tree_decoder.decode(width=1)', hidden_state='pre-finalnorm',
                        cross_round_draft_cache=False, fixed_weights_share_target_storage=True)
        return api.MethodAdapter(name, metadata, model=MLPAdapter(model, bf16=False), context=context)

    compare.expand_methods, compare.load_feature = expand, load
    compare.main()


if __name__ == '__main__':
    main()
