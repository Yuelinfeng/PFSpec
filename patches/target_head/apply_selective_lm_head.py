from pathlib import Path


ROOT = Path(".")
nano = ROOT / "model/gpt/nanochat_gpt.py"
engine = ROOT / "model/gpt/position_selective_speculative.py"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:80]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


replace_once(nano, "from typing import Optional, Tuple", "from typing import Optional, Sequence, Tuple")
replace_once(
    nano,
    '    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):\n',
    '''    def forward(\n        self,\n        idx,\n        targets=None,\n        kv_cache=None,\n        loss_reduction="mean",\n        logit_types: Optional[Sequence[str]] = None,\n    ):\n        """Run the transformer and optionally project only typed sparse slots."""\n        if logit_types is not None:\n            if targets is not None:\n                raise ValueError("logit_types is inference-only")\n            if len(logit_types) != int(idx.size(1)):\n                raise ValueError(\n                    f"logit_types length {len(logit_types)} does not match "\n                    f"input length {int(idx.size(1))}"\n                )\n            unknown = set(logit_types) - {"position", "feature", "full"}\n            if unknown:\n                raise ValueError(f"unknown sparse logit types: {sorted(unknown)}")\n''',
)
replace_once(
    nano,
    '''        # Forward the lm_head (compute logits)\n        softcap = 15  # smoothly cap the logits to the range [-softcap, softcap]\n        logits = self.lm_head(\n            x\n        )  # (B, T, padded_vocab_size) <- very big tensor, large amount of memory\n        logits = logits[..., : self.config.vocab_size]  # slice to remove padding\n        logits = logits.float()  # switch to fp32 for logit softcap and loss computation\n        logits = softcap * torch.tanh(logits / softcap)  # squash the logits\n''',
    '''        # Project only the sparse vocabulary range needed by each inference row.\n        softcap = 15  # smoothly cap the logits to the range [-softcap, softcap]\n        if logit_types is None or any(kind == "full" for kind in logit_types):\n            logits = self.lm_head(x)\n            logits = logits[..., : self.config.vocab_size]\n            logits = logits.float()\n            logits = softcap * torch.tanh(logits / softcap)\n        else:\n            vocab_size = int(self.config.vocab_size)\n            logits = torch.full(\n                (x.size(0), x.size(1), vocab_size),\n                -torch.inf,\n                dtype=torch.float32,\n                device=x.device,\n            )\n            position_end = self.sparse_position_vocab_size\n            feature_start = (\n                0 if self.sparse_shared_vocab else self.sparse_feature_token_offset\n            )\n            feature_end = feature_start + self.sparse_feature_vocab_size\n            for row, kind in enumerate(logit_types):\n                if kind == "position":\n                    token_ids = torch.cat(\n                        (\n                            torch.arange(0, position_end, device=x.device, dtype=torch.long),\n                            torch.tensor([self.sparse_eos_token_id], device=x.device, dtype=torch.long),\n                        )\n                    )\n                elif kind == "feature":\n                    token_ids = torch.arange(\n                        feature_start, feature_end, device=x.device, dtype=torch.long\n                    )\n                else:\n                    token_ids = torch.arange(0, vocab_size, device=x.device, dtype=torch.long)\n                selected_weight = self.lm_head.weight.index_select(0, token_ids)\n                selected = F.linear(x[:, row, :], selected_weight)\n                selected = softcap * torch.tanh(selected.float() / softcap)\n                logits[:, row, :].index_copy_(1, token_ids, selected)\n''',
)

replace_once(
    engine,
    '''        context_before = cache.get_pos()\n        if not self.measure_latency:\n            return model.forward(ids, kv_cache=cache)\n''',
    '''        context_before = cache.get_pos()\n        sparse_logit_types = {\n            "draft_prefill_1": ("position",),\n            "target_verify_2": ("position", "feature"),\n            "target_correction_1": ("feature",),\n            "draft_update_2": ("feature", "position"),\n        }.get(kind)\n        if not self.measure_latency:\n            return model.forward(ids, kv_cache=cache, logit_types=sparse_logit_types)\n''',
)
replace_once(
    engine,
    '''        logits = model.forward(ids, kv_cache=cache)\n''',
    '''        logits = model.forward(ids, kv_cache=cache, logit_types=sparse_logit_types)\n''',
)
print("selective lm-head patch applied")
