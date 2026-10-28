"""Expand the existing Teacher corpus without replacing previous artifacts."""
import argparse
import collections
import json
import random
import shutil
import time
from pathlib import Path

import torch

ROOT = Path("draft_a/runs/rollout500_20261013")
EOS = 12096


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate(item):
    body = item["tokens"]
    assert body.dtype == torch.long and body.ndim == 1
    assert len(body) > 0 and len(body) % 2 == 0
    pos, feat = body[::2], body[1::2]
    assert ((pos >= 0) & (pos < 8000)).all()
    assert ((feat >= 8000) & (feat < EOS)).all()
    assert (pos[1:] > pos[:-1]).all()
    assert isinstance(item["has_eos"], bool)


def prepare(root):
    plan_path = root / "plan.json"
    if plan_path.exists():
        return json.loads(plan_path.read_text())
    old = sorted(Path("draft_a/data/teacher_sequences").glob("*.pt"))
    assert len(old) == 43, len(old)
    entries = []
    for path in old:
        item = torch.load(path, map_location="cpu", weights_only=True)
        validate(item)
        assert (item["temperature"], item["top_p"]) == (0.9, 0.9)
        length_cap = 256 if item["seed"] == 20261013 else 2048 if item["seed"] < 20263000 else 8192
        entries.append(dict(name=path.name, seed=item["seed"], max_new_tokens=length_cap,
                            temperature=0.9, top_p=0.9, split="train", origin=str(path)))
    rng = random.Random(20261013)
    caps = [512]*83 + [1024]*125 + [2048]*159 + [4096]*82 + [8192]*8
    temps = [0.8]*200 + [0.9]*157 + [1.0]*100
    nucleus = [0.85]*152 + [0.9]*153 + [0.95]*152
    for values in (caps, temps, nucleus):
        rng.shuffle(values)
    fresh = []
    for i, (cap, temp, top_p) in enumerate(zip(caps, temps, nucleus)):
        seed = 20265000 + i
        fresh.append(dict(name=f"sequence_{seed}.pt", seed=seed, max_new_tokens=cap,
                          temperature=temp, top_p=top_p, split="train", origin="new"))
    # Allocate held-out trajectories within each requested length group.
    for cap, val_count in [(512, 9), (1024, 14), (2048, 17), (4096, 9), (8192, 1)]:
        group = [r for r in fresh if r["max_new_tokens"] == cap]
        rng.shuffle(group)
        for r in group[:val_count]:
            r["split"] = "val"
        for r in group[val_count:2*val_count]:
            r["split"] = "test"
    entries.extend(fresh)
    assert len(entries) == 500
    assert collections.Counter(r["split"] for r in entries) == {"train":400,"val":50,"test":50}
    plan = dict(entries=entries, target="checkpoints/gpt_vfront.ckpt",
                vqvae="checkpoints/vqvae_vfront.ckpt", seed=20261013,
                old_checkpoint="draft_a/checkpoints/ce_43_v2/best.pt")
    write_json(plan_path, plan)
    return plan


def load_target():
    from model.gaussian_gpt import GaussianGPT
    from model.gaussian_vqvae import GaussianVQVAE
    vqvae = GaussianVQVAE.load_from_checkpoint("checkpoints/vqvae_vfront.ckpt", map_location="cpu").eval()
    wrapper = GaussianGPT.load_from_checkpoint("checkpoints/gpt_vfront.ckpt", vqvae=vqvae, map_location="cpu").eval()
    gpt = wrapper.gpt._orig_mod if hasattr(wrapper.gpt, "_orig_mod") else wrapper.gpt
    gpt = gpt.cuda().eval().requires_grad_(False)
    assert gpt.eos_token_id == EOS
    assert wrapper.position_vocab_size == 8000 and wrapper.feature_vocab_size == 4096
    assert str(wrapper.model_config.chunk_order) == "xyz"
    assert wrapper.model_config.sparse_position_hard_constraints
    return gpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(6)
    plan = prepare(args.root)
    corpus = args.root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    pending = []
    for entry in plan["entries"]:
        path = corpus / entry["name"]
        if not path.exists() and entry["origin"] != "new":
            shutil.copy2(entry["origin"], path)
        if path.exists():
            record = torch.load(path, map_location="cpu", weights_only=True)
            validate(record)
            for key in ("seed", "temperature", "top_p"):
                assert record[key] == entry[key], (path, key)
        else:
            pending.append(entry)
    print(f"Existing={500-len(pending)} pending={len(pending)}", flush=True)
    if args.limit is not None:
        pending = pending[:args.limit]
    if not pending:
        return
    target = load_target()
    for entry in pending:
        start = time.monotonic()
        with torch.inference_mode():
            raw = target.sample_sequence(seq_len=entry["max_new_tokens"], batch_size=1,
                    temperature=entry["temperature"], top_p=entry["top_p"],
                    stop_on_eos=True, seed=entry["seed"])[0].cpu().long()
        ended = bool((raw == EOS).any())
        if ended:
            assert int((raw == EOS).sum()) == 1 and int(raw[-1]) == EOS
        record = dict(tokens=raw[:-1] if ended else raw, has_eos=ended,
                      num_voxels=(len(raw)-int(ended))//2, temperature=entry["temperature"],
                      top_p=entry["top_p"], seed=entry["seed"], source="gpt_vfront.ckpt",
                      max_new_tokens=entry["max_new_tokens"])
        validate(record)
        path = corpus / entry["name"]
        assert not path.exists(), path
        temp = path.with_suffix(".tmp")
        torch.save(record, temp)
        temp.replace(path)
        done = len(list(corpus.glob("*.pt")))
        status = dict(completed=done, total=500, last_file=path.name,
                      body_tokens=len(record["tokens"]), has_eos=ended,
                      elapsed_seconds=time.monotonic()-start)
        write_json(args.root / "generation_progress.json", status)
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
