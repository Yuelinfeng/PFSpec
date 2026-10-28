"""Two-token Feature Transformer with a persistent causal mask."""
import torch
from torch import nn
import torch.nn.functional as F

from legacy_models import SingleStepFeature as LegacySingleStep
from train_feature_state_distill import FeatureStateDraft


class SingleStepFeature(LegacySingleStep):
    def __init__(self, feature_weight, token_embeddings):
        super().__init__(feature_weight, token_embeddings)
        self.register_buffer('causal_mask', torch.tensor([[False, True], [False, False]]),
                             persistent=False)

    def forward(self, state, previous_feature, position):
        ids = torch.stack((previous_feature + 8000, position), -1)
        embed = F.embedding(ids, self.token_embeddings)
        x = self.state_project(self.state_norm(state)).unsqueeze(1) + self.token_project(embed)
        x = x + self.role.weight.unsqueeze(0)
        x = self.block(x, src_mask=self.causal_mask)
        pred = F.rms_norm(state + self.out(x[:, -1]), (1024,))
        raw = F.linear(pred, self.feature_weight)
        return pred, 15. * torch.tanh(raw.float() / 15.)


def make_model(kind, assets):
    if kind == 'transformer':
        return SingleStepFeature(assets['feature_weight'], assets['token_embeddings'])
    if kind == 'mlp':
        return FeatureStateDraft(assets['feature_weight'], position_vocab=8000,
                                 feature_vocab=4096, d_model=256, emb_dim=128,
                                 mlp_dim=512, target_softcap=15.)
    raise ValueError(kind)


def load_model(value, assets, device='cuda'):
    model = make_model(value['model_config']['architecture'], assets)
    model.load_state_dict(value['model_state'], strict=True)
    return model.to(device).float()
