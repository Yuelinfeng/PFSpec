from __future__ import annotations

import argparse
import json
import math
from typing import Dict

import torch

from scripts.profile_position_selective_sd import load_models
from model.gpt.position_selective_speculative import (
    PositionSelectiveSpeculativeEngine,
    make_kv_cache,
)


TARGET = "checkpoints/gpt_vfront.ckpt"
DRAFT = (
    "profiling/s2/d8_canonical_38ep/logs/s2_d8_canonical_38ep/"
    "lightning_logs/version_0/checkpoints/val_loss_monitor_epoch_37_step_50502.ckpt"
)
VQVAE = "checkpoints/vqvae_vfront.ckpt"


def max_selected_error(full: torch.Tensor, sparse: torch.Tensor, ids: torch.Tensor, kinds):
    errors = {}
    for row, kind in enumerate(kinds):
        if kind == "position":
            selected_ids = torch.cat(
                (
                    torch.arange(0, sparse_model.sparse_position_vocab_size, device=ids.device),
                    torch.tensor([sparse_model.sparse_eos_token_id], device=ids.device),
                )
            )
        elif kind == "feature":
            start = 0 if sparse_model.sparse_shared_vocab else sparse_model.sparse_feature_token_offset
            selected_ids = torch.arange(
                start,
                start + sparse_model.sparse_feature_vocab_size,
                device=ids.device,
            )
        else:
            selected_ids = torch.arange(sparse_model.config.vocab_size, device=ids.device)
        diff = (full[:, row, :].index_select(1, selected_ids) - sparse[:, row, :].index_select(1, selected_ids)).abs()
        errors[f"row{row}_{kind}"] = float(diff.max().item())
        invalid = torch.ones(sparse.size(-1), dtype=torch.bool, device=ids.device)
        invalid[selected_ids] = False
        errors[f"row{row}_{kind}_invalid_finite"] = int(torch.isfinite(sparse[:, row, invalid]).sum().item())
    return errors


@torch.inference_mode()
def main() -> None:
    global sparse_model
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-checkpoint", default=TARGET)
    parser.add_argument("--draft-checkpoint", default=DRAFT)
    parser.add_argument("--vqvae-checkpoint", default=VQVAE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    target, draft = load_models(args)
    sparse_model = target.gpt
    device = sparse_model.get_device()
    print(
        "MODEL",
        {
            "vocab": int(sparse_model.config.vocab_size),
            "position_vocab": int(sparse_model.sparse_position_vocab_size),
            "feature_vocab": int(sparse_model.sparse_feature_vocab_size),
            "feature_offset": int(sparse_model.sparse_feature_token_offset),
            "eos": int(sparse_model.sparse_eos_token_id),
        },
        flush=True,
    )

    # A valid verification-shaped pair: previous Feature then proposed Position.
    ids = torch.tensor(
        [[sparse_model.sparse_feature_token_offset, 0]],
        device=device,
        dtype=torch.long,
    )
    full_cache = make_kv_cache(sparse_model, 8)
    sparse_cache = make_kv_cache(sparse_model, 8)
    full = sparse_model(ids, kv_cache=full_cache)
    sparse = sparse_model(ids, kv_cache=sparse_cache, logit_types=("position", "feature"))
    errors = max_selected_error(full, sparse, ids, ("position", "feature"))
    finite_both = full.isfinite() & sparse.isfinite()
    errors["max_all_finite_abs"] = float(
        (full - sparse).abs().masked_fill(~finite_both, 0.0).max().item()
    )
    print("LOGIT_EQ", json.dumps(errors, sort_keys=True), flush=True)

    # Ensure the patched engine can complete a short deterministic run and retain
    # its route/cache invariants. The returned token stream is enough to catch
    # row-type or rollback errors before throughput measurements.
    engine = PositionSelectiveSpeculativeEngine(
        target.gpt, draft.gpt, measure_latency=False
    )
    result = engine.generate(max_tokens=32, temperature=0.0, seed=5)
    print(
        "ENGINE",
        json.dumps(
            {
                "generated_tokens": int(result.stats.generated_tokens),
                "generated_pairs": int(result.stats.generated_pairs),
                "position_accepts": int(result.stats.position_accepts),
                "position_proposals": int(result.stats.position_proposals),
                "feature_observations": int(result.stats.feature_observations),
                "terminated_on_eos": bool(result.stats.terminated_on_eos),
                "tokens_head": result.tokens[0, : min(8, result.tokens.size(1))].tolist(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
