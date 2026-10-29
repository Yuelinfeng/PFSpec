import argparse
import time
from pathlib import Path

import torch

from model.gaussian_gpt import GaussianGPT
from model.gaussian_vqvae import GaussianVQVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-sequences", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument(
        "--output-dir",
        default="draft_a/data/teacher_sequences",
    )
    args = parser.parse_args()

    if args.max_new_tokens <= 0 or args.max_new_tokens % 2:
        raise ValueError("--max-new-tokens must be positive and even")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading VFront VQ-VAE...", flush=True)
    vqvae = GaussianVQVAE.load_from_checkpoint(
        "checkpoints/vqvae_vfront.ckpt",
        map_location="cpu",
    ).eval()

    print("Loading VFront Target GPT...", flush=True)
    target = GaussianGPT.load_from_checkpoint(
        "checkpoints/gpt_vfront.ckpt",
        vqvae=vqvae,
        map_location="cpu",
    ).eval().to("cuda")

    target.requires_grad_(False)

    eos = int(target.gpt.eos_token_id)
    position_vocab = int(target.position_vocab_size)
    feature_vocab = int(target.feature_vocab_size)
    feature_offset = int(target.feature_token_offset)

    print("EOS:", eos, flush=True)
    print("Position vocab:", position_vocab, flush=True)
    print("Feature vocab:", feature_vocab, flush=True)
    print("Feature offset:", feature_offset, flush=True)

    assert position_vocab == 8000
    assert feature_vocab == 4096
    assert feature_offset == 8000
    assert eos == 12096
    assert str(target.model_config.chunk_order) == "xyz"

    for index in range(args.num_sequences):
        start = time.perf_counter()
        sample_seed = args.seed + index

        print(
            f"\nGenerating sequence {index + 1}/{args.num_sequences}",
            flush=True,
        )

        with torch.inference_mode():
            raw = target.gpt.sample_sequence(
                seq_len=args.max_new_tokens,
                batch_size=1,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_on_eos=True,
                seed=sample_seed,
            )[0].detach().cpu().long()

        eos_indices = (raw == eos).nonzero(
            as_tuple=True
        )[0]

        has_eos = eos_indices.numel() > 0

        if has_eos:
            if eos_indices.numel() != 1:
                raise RuntimeError("Multiple EOS tokens")

            if eos_indices[0].item() != len(raw) - 1:
                raise RuntimeError("EOS is not terminal")

            body = raw[:-1]
        else:
            body = raw

        if len(body) % 2:
            raise RuntimeError(
                f"Incomplete P/F pair: {len(body)} tokens"
            )

        positions = body[0::2]
        features = body[1::2]

        if not ((positions >= 0) & (positions < 8000)).all():
            raise RuntimeError("Invalid Position token")

        if not (
            (features >= 8000) & (features < 12096)
        ).all():
            raise RuntimeError("Invalid Feature token")

        if len(positions) > 1:
            if not (positions[1:] > positions[:-1]).all():
                raise RuntimeError(
                    "Position order constraint violated"
                )

        if len(body) == 0:
            print("Empty generated sequence, skipping", flush=True)
            continue

        # Save P/F body only.
        # BOS is implicit; EOS is indicated by has_eos.
        # Truncated samples must NOT be given an artificial EOS
        # during subsequent Draft training.
        record = {
            "tokens": body.clone(),
            "has_eos": bool(has_eos),
            "num_voxels": len(positions),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": sample_seed,
            "source": "gpt_vfront.ckpt",
        }

        output_file = (
            output_dir / f"sequence_{sample_seed}.pt"
        )

        if output_file.exists():
            raise FileExistsError(output_file)

        torch.save(record, output_file)

        elapsed = time.perf_counter() - start

        print("Body tokens:", len(body), flush=True)
        print("Gaussian voxels:", len(positions), flush=True)
        print("Has EOS:", has_eos, flush=True)
        print("Time:", round(elapsed, 2), "seconds", flush=True)
        print("Saved:", output_file, flush=True)

    print("\nTEACHER CORPUS GENERATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
