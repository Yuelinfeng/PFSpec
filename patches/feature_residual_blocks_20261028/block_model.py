"""Extend the trained Feature predictor with identity-initialized residual blocks."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from train_feature_state_distill import FeatureStateDraft


class ExtraBlock(nn.Module):
    def __init__(self, original):
        super().__init__()
        cfg = original.config
        self.in_norm = deepcopy(original.in_norm)
        self.project = deepcopy(original.project)
        inputs = cfg['d_model'] + 2 * cfg['emb_dim']
        width, hidden = cfg['mlp_dim'], cfg['hidden_dim']
        self.delta = nn.Sequential(
            nn.LayerNorm(inputs), nn.Linear(inputs, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, hidden))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, state, feature_embedding, position_embedding):
        inputs = torch.cat((self.project(self.in_norm(state)), feature_embedding,
                            position_embedding), dim=-1)
        return state + self.delta(inputs).float()


class StackedFeatureDraft(FeatureStateDraft):
    """Reuse token embeddings and the fixed output head across all blocks."""
    def __init__(self, feature_weight, *, added_blocks=0, **config):
        if added_blocks not in (0, 3, 5):
            raise ValueError('Supported added block counts: 0, 3, 5')
        super().__init__(feature_weight, **config)
        self.extra_blocks = nn.ModuleList(ExtraBlock(self) for _ in range(added_blocks))
        self.config.update(added_blocks=added_blocks, residual_blocks=1 + added_blocks)

    def forward(self, prev_state, prev_feature, position):
        from model.gpt.nanochat_gpt import norm as teacher_norm
        state = prev_state.float()
        feature_embedding = self.feature_embed(prev_feature.long())
        position_embedding = self.position_embed(position.long())
        inputs = torch.cat((self.project(self.in_norm(state)), feature_embedding,
                            position_embedding), dim=-1)
        state = state + self.delta(inputs).float()
        for block in self.extra_blocks:
            state = block(state, feature_embedding, position_embedding)
        normalized = teacher_norm(state.float())
        raw = F.linear(normalized, self.frozen_head)
        cap = self.config['target_softcap']
        return normalized, cap * torch.tanh(raw.float() / cap)


def from_checkpoint(value, head, device='cuda'):
    cfg = value['model_config']
    keys = ('position_vocab', 'feature_vocab', 'd_model', 'emb_dim', 'mlp_dim',
            'target_softcap')
    model = StackedFeatureDraft(head, added_blocks=int(cfg.get('added_blocks', 0)),
                               **{k: cfg[k] for k in keys})
    model.load_state_dict(value['model_state'], strict=True)
    return model.to(device).float()


def expand_checkpoint(value, head, added_blocks, seed=24690):
    torch.manual_seed(seed)
    original = from_checkpoint(value, head, device='cpu')
    cfg = value['model_config']
    keys = ('position_vocab', 'feature_vocab', 'd_model', 'emb_dim', 'mlp_dim',
            'target_softcap')
    model = StackedFeatureDraft(head, added_blocks=added_blocks,
                               **{k: cfg[k] for k in keys})
    # Only newly added blocks retain their initialization.
    state = model.state_dict()
    state.update(original.state_dict())
    model.load_state_dict(state, strict=True)
    for block in model.extra_blocks:
        block.in_norm.load_state_dict(original.in_norm.state_dict())
        block.project.load_state_dict(original.project.state_dict())
    result = dict(value)
    result.update(model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                  model_config=dict(model.config), expansion_seed=seed)
    return result
