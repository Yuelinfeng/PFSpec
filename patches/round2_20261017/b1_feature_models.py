"""Shared-history Feature mechanisms; inference uses no future Target state."""
import torch
from torch import nn
import torch.nn.functional as F
from train_feature_state_distill import FeatureStateDraft


class DirectFeature(nn.Module):
    def __init__(self, candidate_position=True):
        super().__init__()
        self.candidate_position = candidate_position
        self.norm = nn.LayerNorm(1024)
        self.project = nn.Linear(1024, 256)
        self.feature_embed = nn.Embedding(4096, 128)
        self.position_embed = nn.Embedding(8000, 128) if candidate_position else None
        self.net = nn.Sequential(nn.LayerNorm(512 if candidate_position else 384),
            nn.Linear(512 if candidate_position else 384, 512), nn.GELU(), nn.Linear(512, 4096))

    def forward(self, state, previous_feature, position):
        parts = [self.project(self.norm(state)), self.feature_embed(previous_feature)]
        if self.candidate_position:
            parts.append(self.position_embed(position))
        return None, self.net(torch.cat(parts, -1)).float()


class SingleStepFeature(nn.Module):
    def __init__(self, feature_weight, token_embeddings):
        super().__init__()
        self.register_buffer('feature_weight', feature_weight.detach(), persistent=False)
        self.register_buffer('token_embeddings', token_embeddings.detach(), persistent=False)
        self.state_norm = nn.LayerNorm(1024)
        self.state_project = nn.Linear(1024, 256)
        self.token_project = nn.Linear(1024, 256)
        self.role = nn.Embedding(2, 256)
        self.block = nn.TransformerEncoderLayer(256, 4, 512, dropout=0., activation='gelu',
            batch_first=True, norm_first=True)
        self.out = nn.Linear(256, 1024)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, state, previous_feature, position):
        ids = torch.stack((previous_feature+8000, position), -1)
        embed = F.embedding(ids, self.token_embeddings)
        x = self.state_project(self.state_norm(state)).unsqueeze(1) + self.token_project(embed)
        x = x + self.role.weight.unsqueeze(0)
        mask = torch.tensor([[False, True], [False, False]], device=x.device)
        x = self.block(x, src_mask=mask)
        pred = F.rms_norm(state + self.out(x[:, -1]), (1024,))
        raw = F.linear(pred, self.feature_weight)
        return pred, 15.*torch.tanh(raw.float()/15.)


def make_model(name, target):
    weights = target.lm_head.weight[8000:12096].detach().float()
    if name == 'b0':
        return FeatureStateDraft(weights, position_vocab=8000, feature_vocab=4096,
            d_model=256, emb_dim=128, mlp_dim=512, target_softcap=15.)
    if name == 'b1_m':
        return DirectFeature(False)
    if name == 'b1_h':
        return DirectFeature(True)
    if name == 'b1_e':
        return SingleStepFeature(weights, target.transformer.wte.weight.detach().float())
    raise ValueError(name)
