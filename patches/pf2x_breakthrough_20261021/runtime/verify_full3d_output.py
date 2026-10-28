#!/usr/bin/env python3
"""Decode and render one naturally terminated candidate PF output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON row at {path}:{line_number}") from error
    return rows


def tensor_summary(torch, value) -> dict:
    values = value.detach().float()
    if values.numel() == 0 or not bool(torch.isfinite(values).all().item()):
        raise ValueError("Gaussian output contains an empty or nonfinite tensor")
    return dict(
        shape=list(values.shape),
        minimum=float(values.min().item()),
        maximum=float(values.max().item()),
        mean=float(values.mean().item()),
    )


def choose_sample(run_dir: Path) -> tuple[dict, dict, dict]:
    pipeline_path = run_dir / "pipeline.json"
    summary_path = run_dir / "summary.json"
    prompts_path = run_dir / "prompts.json"
    records_path = run_dir / "records.jsonl"
    for path in (pipeline_path, summary_path, prompts_path, records_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    pipeline = read_json(pipeline_path)
    summary = read_json(summary_path)
    if pipeline.get("status") != "complete" or summary.get("status") != "complete":
        raise RuntimeError("The decode run must finish before VQ decoding starts")

    prompt_data = read_json(prompts_path)
    prompts = {row["pair_id"]: row for row in prompt_data.get("contexts", [])}
    rows = read_jsonl(records_path)
    expected = int(pipeline.get("expected_runs", len(rows)))
    if len(rows) != expected:
        raise RuntimeError(f"Decode rows are incomplete: {len(rows)} of {expected}")

    selected = next((row for row in rows
                     if row.get("method") == "candidate_pf" and row.get("eos") is True), None)
    if selected is None:
        raise RuntimeError("No naturally EOS-terminated candidate_pf row exists")
    prompt = prompts.get(selected["pair_id"])
    if prompt is None:
        raise ValueError("The selected candidate has no matching prompt")
    original = next((row for row in rows
                     if row.get("pair_id") == selected["pair_id"]
                     and row.get("method") == "original_pf"), None)
    return selected, prompt, original


def token_comparison(candidate: list[int], original: dict | None, eos_id: int) -> dict:
    if original is None:
        return dict(present=False, same_output_tokens=None)
    original_tokens = [int(token) for token in original.get("output_tokens", [])]
    first_difference = next(
        (index for index, (left, right) in enumerate(zip(candidate, original_tokens))
         if left != right),
        None,
    )
    if first_difference is None and len(candidate) != len(original_tokens):
        first_difference = min(len(candidate), len(original_tokens))
    return dict(
        present=True,
        same_output_tokens=candidate == original_tokens,
        candidate_output_tokens=len(candidate),
        original_output_tokens=len(original_tokens),
        candidate_eos=bool(candidate and candidate[-1] == eos_id),
        original_eos=bool(original_tokens and original_tokens[-1] == eos_id),
        first_difference_index=first_difference,
        comparison_role="informational; a token difference does not fail Gaussian validity",
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.expanduser().resolve()
    run_dir = args.run_dir.expanduser()
    run_dir = run_dir.resolve() if run_dir.is_absolute() else (project / run_dir).resolve()
    output_dir = args.out_dir.expanduser()
    output_dir = (output_dir.resolve() if output_dir.is_absolute()
                  else (project / output_dir).resolve())
    if not project.is_dir() or not run_dir.is_dir():
        parser.error("Project or completed decode run directory is missing")
    if output_dir.exists():
        parser.error("--out-dir must name a new directory: " + str(output_dir))

    selected, prompt, original = choose_sample(run_dir)
    prompt_tokens = [int(token) for token in prompt["prompt"]]
    output_tokens = [int(token) for token in selected["output_tokens"]]
    eos_id = int(read_json(run_dir / "prompts.json").get("bos_id", 12096))
    if int(selected["prefix"]) != int(prompt["prefix"]):
        raise ValueError("Record and prompt prefix lengths differ")
    if not prompt_tokens or prompt_tokens[0] != eos_id:
        raise ValueError("Prompt must start with the recorded BOS/EOS token")
    if not selected.get("eos") or not output_tokens or output_tokens[-1] != eos_id:
        raise ValueError("Selected row does not end in its natural EOS token")
    if eos_id in output_tokens[:-1]:
        raise ValueError("EOS occurs before the end of the saved output")

    body = prompt_tokens[1:] + output_tokens[:-1]
    if not body or len(body) % 2:
        raise ValueError("Natural-EOS body must contain complete Position/Feature pairs")
    position_ids = body[::2]
    feature_ids = body[1::2]
    if any(token < 0 or token >= 8000 for token in position_ids):
        raise ValueError("Body contains a Position token outside [0, 8000)")
    if any(token < 8000 or token >= eos_id for token in feature_ids):
        raise ValueError("Body contains a Feature token outside [8000, EOS)")
    if len(prompt_tokens) - 1 != int(prompt["prefix"]):
        raise ValueError("Prompt body length differs from its recorded prefix")

    output_dir.mkdir(parents=True)
    metadata_path = output_dir / "metadata.json"
    metadata = dict(
        status="starting",
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        source=selected.get("source"),
        pair_id=selected.get("pair_id"),
        method=selected.get("method"),
        prefix_tokens=int(prompt["prefix"]),
        eos=True,
        eos_id=eos_id,
        generated_tokens=len(output_tokens),
        body_tokens=len(body),
        position_token_count=len(position_ids),
        feature_token_count=len(feature_ids),
        position_id_range=[min(position_ids), max(position_ids)],
        feature_id_range=[min(feature_ids), max(feature_ids)],
        same_pair_original_pf=token_comparison(output_tokens, original, eos_id),
        input_path=selected.get("input_path"),
        token_artifact=str(output_dir / "tokens.pt"),
        gaussian_artifact=str(output_dir / "gaussians.npz"),
        views_dir=str(output_dir / "views"),
    )
    write_json(metadata_path, metadata)
    started = time.perf_counter()

    try:
        sys.path.insert(0, str(project))
        import numpy as np
        import torch
        from draft_a.pf_multiscene import full_3d_eval
        from model.gaussian_vqvae import GaussianVQVAE
        from utils.pos_tokens import pos_tokens_to_centered_coords

        checkpoint = project / "checkpoints" / "vqvae_vfront.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the existing VQ decode and renderer")
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        grid_coords = pos_tokens_to_centered_coords(
            torch.tensor(position_ids, dtype=torch.long)[:, None], 1, 20
        ).to(torch.int32)
        unique_positions = torch.unique(grid_coords, dim=0)
        metadata["position_grid"] = dict(
            coordinate_min=grid_coords.min(0).values.tolist(),
            coordinate_max=grid_coords.max(0).values.tolist(),
            unique_coordinate_count=int(unique_positions.size(0)),
            duplicate_position_count=len(position_ids) - int(unique_positions.size(0)),
        )
        torch.save(dict(
            tokens=torch.tensor(body, dtype=torch.long),
            suffix=torch.tensor(output_tokens, dtype=torch.long),
            has_eos=True,
            prefix_tokens=int(prompt["prefix"]),
            prompt=torch.tensor(prompt_tokens, dtype=torch.long),
            pair_id=selected["pair_id"],
            method=selected["method"],
        ), output_dir / "tokens.pt")
        write_json(metadata_path, metadata)

        vqvae = GaussianVQVAE.load_from_checkpoint(
            str(checkpoint), map_location="cpu"
        ).cuda().float().eval()
        vqvae.requires_grad_(False)
        side = (20 * vqvae.autoencoder.stride
                * vqvae.autoencoder.default_grid_size)
        cameras = full_3d_eval.camera_views(side)
        write_json(output_dir / "cameras.json", cameras)
        views_dir = output_dir / "views"
        views_dir.mkdir()

        with torch.inference_mode():
            tensors, quality = full_3d_eval.gaussian_tensors(vqvae, body)
            gaussian_count = int(tensors["means"].size(0))
            if gaussian_count < 1 or int(quality["gaussians"]) != gaussian_count:
                raise ValueError("VQ decoder returned no consistent Gaussian rows")
            expected_shapes = dict(
                means=(gaussian_count, 3),
                scales=(gaussian_count, 3),
                opacities=(gaussian_count,),
                quats=(gaussian_count, 4),
                colors=(gaussian_count, 3),
            )
            for name, expected_shape in expected_shapes.items():
                if name not in tensors or tuple(tensors[name].shape) != expected_shape:
                    raise ValueError(
                        f"Gaussian field {name} shape differs: "
                        f"expected {expected_shape}, got "
                        f"{tuple(tensors[name].shape) if name in tensors else None}"
                    )
            parameter_ranges = {
                name: tensor_summary(torch, tensor)
                for name, tensor in tensors.items()
            }
            quaternion_norm = torch.linalg.vector_norm(tensors["quats"].float(), dim=-1)
            autoencoder = vqvae.autoencoder
            decoder_stages = len(autoencoder.decoder.ups)
            gaussians_per_voxel = int(autoencoder.gaussians_per_voxel)
            np.savez_compressed(
                output_dir / "gaussians.npz",
                **{name: value.detach().cpu().numpy()
                   for name, value in tensors.items()},
            )
            metadata.update(
                vqvae_checkpoint=str(checkpoint),
                chunk_side_m=float(side),
                gaussian_quality=quality,
                position_token_count=len(position_ids),
                gaussian_count=gaussian_count,
                gaussian_to_position_ratio=gaussian_count / len(position_ids),
                vq_multiplicity=dict(
                    latent_stride=int(autoencoder.stride),
                    decoder_upsample_stages=decoder_stages,
                    decoder_upsample_factor=2 ** decoder_stages,
                    gaussians_per_decoder_voxel=gaussians_per_voxel,
                    fixed_gaussians_per_position=False,
                ),
                parameter_ranges=parameter_ranges,
                quaternion_norm=dict(
                    minimum=float(quaternion_norm.min().item()),
                    maximum=float(quaternion_norm.max().item()),
                    mean=float(quaternion_norm.mean().item()),
                ),
            )
            write_json(metadata_path, metadata)
            rgb, alpha = full_3d_eval.render_views(tensors, cameras)
            expected_rgb_shape = (
                6, int(cameras["height"]), int(cameras["width"]), 3
            )
            if tuple(rgb.shape) != expected_rgb_shape:
                raise ValueError(
                    f"Unexpected rendered RGB shape: {list(rgb.shape)}; "
                    f"expected {expected_rgb_shape}"
                )
            if (alpha.ndim < 3 or int(alpha.size(0)) != 6 or
                    tuple(alpha.shape[1:3]) != expected_rgb_shape[1:3]):
                raise ValueError(f"Unexpected rendered alpha shape: {list(alpha.shape)}")
            rgb_finite = bool(torch.isfinite(rgb).all().item())
            alpha_finite = bool(torch.isfinite(alpha).all().item())
            if not rgb_finite:
                raise ValueError("Rendered RGB contains nonfinite values")
            if not alpha_finite:
                raise ValueError("Rendered alpha contains nonfinite values")
            full_3d_eval.save_views(rgb, views_dir)
            view_paths = [views_dir / f"view_{index:02d}.png" for index in range(6)]
            invalid_views = [str(path) for path in view_paths
                             if not path.is_file() or path.stat().st_size <= 0]
            if invalid_views:
                raise RuntimeError(f"Rendered PNG files are missing or empty: {invalid_views}")
            metadata.update(
                status="complete",
                rendered_rgb_finite=rgb_finite,
                rendered_alpha_finite=alpha_finite,
                rendered_rgb_shape=list(rgb.shape),
                rendered_alpha_shape=list(alpha.shape),
                alpha_coverage=float((alpha > 0.01).float().mean().item()),
                saved_view_count=len(view_paths),
                views=[str(path) for path in view_paths],
                elapsed_seconds=time.perf_counter() - started,
            )
        write_json(metadata_path, metadata)
    except Exception as error:
        metadata.update(
            status="failed",
            error=dict(type=type(error).__name__, message=str(error)),
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(metadata_path, metadata)
        raise

    print(json.dumps(dict(status=metadata["status"], output=str(output_dir),
                          pair_id=metadata["pair_id"],
                          gaussian_count=metadata["gaussian_count"],
                          elapsed_seconds=metadata["elapsed_seconds"]),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    _main()
