"""Reuse trained GaussianGPT weights and measure genuinely smaller Drafts."""

import argparse
import json
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf

from draft_a.learning_train import write_json


def make_model(config):
    from model.gpt.gpt import NanoGPTGaussianModel
    return NanoGPTGaussianModel(config=OmegaConf.create(config), codebook_size=4096,
                                tokens_per_latent=2, num_feature_tokens=1).cuda().float()


def convert(source, output):
    value = torch.load(source, map_location="cpu", weights_only=False)
    prefix = "gpt._orig_mod."
    model = {k[len(prefix):]: v for k, v in value["state_dict"].items() if k.startswith(prefix)}
    if not model:
        raise ValueError("No GaussianGPT model parameters")
    config = OmegaConf.to_container(value["hyper_parameters"]["model_config"], resolve=True)
    instance = make_model(config)
    instance.load_state_dict(model, strict=True)
    torch.save(dict(config=config, model={k: v.cpu() for k, v in instance.state_dict().items()},
                    step=value["global_step"], origin=str(source), epoch=value["epoch"],
                    parameters=sum(p.numel() for p in instance.parameters())), output)
    print(json.dumps(dict(event="converted", output=str(output))), flush=True)


def evaluate(checkpoint, source, evaluation, output, limit):
    from draft_a import learning_eval as ev
    from draft_a import train_v2
    original = train_v2.make_model
    try:
        train_v2.make_model = make_model
        started = time.monotonic()
        result = ev.evaluate_checkpoint(checkpoint, source, evaluation, limit)
        result["seconds"] = time.monotonic() - started
        write_json(output, result)
        print(json.dumps(dict(event="evaluated", checkpoint=str(checkpoint),
                              splits=result["splits"], seconds=result["seconds"])), flush=True)
    finally:
        train_v2.make_model = original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["convert", "eval"])
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, default=Path("draft_a/runs/rollout500_20261013"))
    ap.add_argument("--evaluation-root", type=Path, default=Path("draft_a/runs/pf_learning_budget_20261013"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    torch.set_num_threads(6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "convert":
        convert(args.checkpoint, args.output)
    else:
        evaluate(args.checkpoint, args.source_root, args.evaluation_root, args.output, args.limit)


if __name__ == "__main__":
    main()
