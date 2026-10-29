import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from model.gpt.gpt import NanoGPTGaussianModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--body-tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    assert args.body_tokens > 0
    assert args.body_tokens % 2 == 0
    assert torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    print("===== LOAD DRAFT CONFIG =====", flush=True)

    cfg = OmegaConf.load(
        "conf/model/gpt_draft_a.yaml"
    )

    assert cfg.gpt_size.n_layer == 8
    assert cfg.gpt_size.n_embd == 512
    assert cfg.gpt_size.n_head == 8
    assert cfg.position_vocab_size == 8000
    assert cfg.shared_vocab is False
    assert cfg.rope_basis == "position"
    assert cfg.sparse_position_hard_constraints

    print(OmegaConf.to_yaml(cfg), flush=True)

    print("===== RANDOM INITIALIZATION =====", flush=True)

    # Fresh model. No checkpoint loading.
    draft = NanoGPTGaussianModel(
        config=cfg,
        codebook_size=4096,
        tokens_per_latent=2,
        num_feature_tokens=1,
    ).to("cuda")

    draft.train()

    total_params = sum(
        p.numel() for p in draft.parameters()
    )

    print("Parameters:", total_params, flush=True)
    print("Layers:", cfg.gpt_size.n_layer, flush=True)
    print("Embedding:", cfg.gpt_size.n_embd, flush=True)
    print("Heads:", cfg.gpt_size.n_head, flush=True)
    print("EOS:", draft.eos_token_id, flush=True)

    assert draft.eos_token_id == 12096
    assert draft.vocab_size == 12098

    print("===== LOAD TEACHER DATA =====", flush=True)

    corpus = Path(
        "draft_a/data/teacher_sequences"
    )

    files = sorted(corpus.glob("*.pt"))
    assert files, "Teacher corpus is empty"

    record = None
    selected_file = None

    for path in files:
        item = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if len(item["tokens"]) >= args.body_tokens:
            record = item
            selected_file = path
            break

    if record is None:
        raise RuntimeError(
            "No sequence long enough for smoke test"
        )

    body = record["tokens"].long()
    full_body_length = len(body)

    body = body[:args.body_tokens]

    # Only append EOS when the original Target sample
    # actually ended naturally and was not truncated.
    include_eos = (
        bool(record["has_eos"])
        and len(body) == full_body_length
    )

    eos = draft.eos_token_id

    seq = torch.cat([
        torch.tensor([eos], dtype=torch.long),
        body,
        (
            torch.tensor([eos], dtype=torch.long)
            if include_eos
            else torch.empty(0, dtype=torch.long)
        ),
    ]).unsqueeze(0).to("cuda")

    print("File:", selected_file, flush=True)
    print("Input shape:", tuple(seq.shape), flush=True)
    print("Supervised tokens:", seq.shape[1]-1, flush=True)
    print("EOS supervision:", include_eos, flush=True)
    print("First 12 tokens:",
          seq[0, :12].tolist(), flush=True)

    # Both input_ids and labels contain BOS + P/F body.
    # The official model performs next-token shifting.
    optimizer = torch.optim.AdamW(
        draft.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    use_bf16 = torch.cuda.is_bf16_supported()

    print("BF16 autocast:", use_bf16, flush=True)

    first_param = next(draft.parameters())
    initial_weight = first_param.detach().clone()

    initial_loss = None

    print("\n===== TRAINING =====", flush=True)

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = draft(
                input_ids=seq,
                labels=seq,
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )

        if initial_loss is None:
            initial_loss = float(loss.item())

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            draft.parameters(),
            max_norm=1.0,
        )

        if not torch.isfinite(grad_norm):
            raise RuntimeError("Non-finite gradients")

        optimizer.step()

        print(
            f"step={step + 1:02d} "
            f"loss={loss.item():.5f} "
            f"grad_norm={grad_norm.item():.4f}",
            flush=True,
        )

    draft.eval()

    with torch.no_grad(), torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=use_bf16,
    ):
        final_loss = draft(
            input_ids=seq,
            labels=seq,
        )

    weight_change = (
        first_param.detach() - initial_weight
    ).float().norm().item()

    print("\n===== RESULTS =====", flush=True)
    print("Initial loss:", initial_loss, flush=True)
    print("Final loss:", final_loss.item(), flush=True)
    print("Weight change:", weight_change, flush=True)
    print(
        "Peak GPU memory GiB:",
        round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3,
        ),
        flush=True,
    )

    assert weight_change > 0
    assert torch.isfinite(final_loss)

    print("\nDRAFT-A CE SMOKE TEST PASSED", flush=True)
    print(
        "No checkpoint saved: this is only a training test.",
        flush=True,
    )


if __name__ == "__main__":
    main()
