"""Shared narrow recurrent state predictor with Target token/readout weights."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from model.gpt.engine import KVCache
from model.gpt.nanochat_gpt import Block, GPTConfig, norm
from model.gpt.rope import RotaryPositionEmbedder


@dataclass
class DraftConfig:
    mode: str = "feature"
    width: int = 256
    layers: int = 1
    heads: int = 4


@dataclass
class TokenSpec:
    positions: int
    features: int
    offset: int
    eos: int
    pad: int
    vocab: int

    @classmethod
    def from_core(cls, core):
        if core.sparse_shared_vocab or core.sparse_num_position_tokens != 1:
            raise ValueError("P/F EAGLE requires separate vocabularies and one Position token")
        if not core.rope_layout.is_position or core.rope_dim != 4:
            raise ValueError("P/F EAGLE requires sparse position-based 4D RoPE")
        return cls(core.sparse_position_vocab_size, core.sparse_feature_vocab_size,
                   core.sparse_feature_token_offset, core.sparse_eos_token_id,
                   core.sparse_pad_token_id, core.config.vocab_size)


class PFDraft(nn.Module):
    def __init__(self, core, config=None):
        super().__init__()
        self.config = config or DraftConfig()
        cfg = self.config
        if cfg.mode not in {"joint", "feature"}:
            raise ValueError("mode must be joint or feature")
        if cfg.width % cfg.heads or (cfg.width // cfg.heads) % 8:
            raise ValueError("head dimension must be divisible by eight for 4D RoPE")
        self.spec = TokenSpec.from_core(core)
        self.register_buffer("embedding", core.transformer.wte.weight.detach(), persistent=False)
        self.register_buffer("output_weight", core.lm_head.weight.detach(), persistent=False)
        dim = self.embedding.shape[1]
        self.hidden_in = nn.Linear(dim, cfg.width, bias=False)
        self.token_in = nn.Linear(dim, cfg.width, bias=False)
        block_cfg = GPTConfig(n_embd=cfg.width, n_head=cfg.heads, n_kv_head=cfg.heads,
                              n_layer=cfg.layers, value_embed_every_n_layers=None)
        self.blocks = nn.ModuleList(Block(block_cfg, i) for i in range(cfg.layers))
        self.hidden_out = nn.Linear(cfg.width, dim, bias=False)
        nn.init.zeros_(self.hidden_out.weight)
        self.rope = RotaryPositionEmbedder(cfg.width // cfg.heads, dim=4)

    def new_cache(self, length):
        c = self.config
        return KVCache(1, c.heads, length, c.width // c.heads, c.layers,
                       self.hidden_in.weight.device, self.hidden_in.weight.dtype)

    def forward(self, previous_hidden, actual_tokens, coordinates, cache=None):
        """(h_t, x_(t+1), coord_(t+1)) -> predicted final-normalized h_(t+1)."""
        if previous_hidden.shape[:2] != actual_tokens.shape:
            raise ValueError("hidden/token time indices must agree")
        embedded = F.embedding(actual_tokens, self.embedding).to(previous_hidden.dtype)
        x = self.hidden_in(previous_hidden) + self.token_in(norm(embedded))
        phases = self.rope(coordinates)
        for block in self.blocks:
            x = block(x, None, phases, (-1, 0), cache)
        return norm(previous_hidden + self.hidden_out(norm(x)))

    def logits(self, hidden, kind):
        s = self.spec
        if kind == "P":
            if self.config.mode != "joint":
                raise ValueError("Feature-only Draft does not compute Position logits")
            weight = torch.cat((self.output_weight[:s.positions], self.output_weight[s.eos:s.eos+1]))
        elif kind == "F":
            weight = self.output_weight[s.offset:s.offset+s.features]
        else:
            raise ValueError(kind)
        raw = F.linear(hidden, weight).float()
        return 15.0 * torch.tanh(raw / 15.0)

    def losses(self, source, tokens, target, coordinates, next_tokens, ranks,
               hidden_weight=1.0, token_weight=0.1):
        pred = self(source, tokens, coordinates)
        valid_p = (tokens >= 0) & (tokens < self.spec.positions)
        valid_f = (tokens >= self.spec.offset) & (tokens < self.spec.offset+self.spec.features)
        values = {}
        for kind, mask in (("P", valid_p), ("F", valid_f)):
            values["hidden_after_"+kind] = (F.smooth_l1_loss(pred[mask], target[mask])
                                                   if mask.any() else pred.sum()*0)
        hidden_loss = values["hidden_after_P"] + values["hidden_after_F"]
        ce_loss = pred.sum()*0
        for kind in (("P", "F") if self.config.mode == "joint" else ("F",)):
            labels = next_tokens
            mask = ((labels >= self.spec.offset) & (labels < self.spec.offset+self.spec.features)
                    if kind == "F" else ((labels >= 0) & (labels < self.spec.positions)) | (labels == self.spec.eos))
            if not mask.any():
                values["ce_"+kind] = pred.sum()*0
                continue
            logits = self.logits(pred[mask], kind)
            if kind == "F":
                local = labels[mask] - self.spec.offset
            else:
                # A Position prediction follows an already consumed Feature.
                last_p = coordinates[mask][:, :3]
                side = round(self.spec.positions ** (1/3))
                position_id = last_p[:, 0]*side*side + last_p[:, 1]*side + last_p[:, 2]
                previous_rank = ranks[position_id.long()]
                logits[:, :self.spec.positions] = logits[:, :self.spec.positions].masked_fill(
                    ranks.unsqueeze(0) <= previous_rank.unsqueeze(1), -torch.inf)
                local = torch.where(labels[mask] == self.spec.eos, self.spec.positions, labels[mask])
            values["ce_"+kind] = F.cross_entropy(logits, local)
            ce_loss = ce_loss + values["ce_"+kind]
        values["loss"] = hidden_weight*hidden_loss + token_weight*ce_loss
        return values

    def description(self):
        return {"config": asdict(self.config), "tokens": asdict(self.spec),
                "trainable_parameters": sum(p.numel() for p in self.parameters())}
