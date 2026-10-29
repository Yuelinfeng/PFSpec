"""Describe token coverage, natural endings, and the independent data splits."""
import argparse
import collections
import json
from pathlib import Path

import torch

from draft_a.corpus_v2 import ROOT, EOS, validate, write_json


def counts(records, offset, end):
    out = torch.zeros(end-offset, dtype=torch.float64)
    for r in records:
        x = r["tokens"]
        x = x[(x >= offset) & (x < end)] - offset
        out += torch.bincount(x, minlength=end-offset)
    return out


def describe(records):
    lengths = torch.tensor([len(r["tokens"]) for r in records], dtype=torch.float64)
    result = dict(sequences=len(records), body_tokens=int(lengths.sum()),
                  natural_eos=sum(r["has_eos"] for r in records),
                  length_quantiles=torch.quantile(lengths, torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64)).tolist())
    result["eos_fraction"] = result["natural_eos"]/len(records)
    for name, offset, end in [("position",0,8000),("feature",8000,EOS)]:
        c = counts(records, offset, end)
        p = c[c > 0]/c.sum()
        result[name] = dict(unique_ids=int((c>0).sum()), entropy_bits=float(-(p*p.log2()).sum()))
    result["depth_sequences"] = {str(t):sum(len(r["tokens"]) >= t for r in records)
                                  for t in [256,512,1024,2048,4096,8192]}
    return result


def read_records(root, entries):
    records = []
    for entry in entries:
        r = torch.load(root/"corpus"/entry["name"], map_location="cpu", weights_only=True)
        validate(r)
        for key in ("seed","temperature","top_p"):
            assert r[key] == entry[key], (entry["name"],key)
        records.append(dict(r, name=entry["name"], split=entry["split"], cap=entry["max_new_tokens"]))
    return records


def real_records():
    from data.common import PreprocessedDataset
    scene = Path("draft_a/vfront_pilot_scene.txt").read_text().strip().removesuffix(".json")
    path = Path("/root/autodl-tmp/3D-FRONT-pilot")/scene/"gaussian_init/vfront_pilot_tokens.pt"
    dataset = PreprocessedDataset(file_list=[str(path)], num_position_tokens=1,
        position_vocab_size=8000, codebook_size=4096, chunk_shape=[20,20,20], dense_chunks=False,
        chunk_order="xyz", min_chunk_occupancy=0.3, max_chunk_attempts=10,
        chunk_origin=[None,None,-1], load_augmented_tokens=False, shared=False)
    import random
    import numpy as np
    rows = []
    for seed in range(5):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        body = dataset[0].long()
        row = dict(tokens=body, has_eos=False, crop_seed=seed, path=str(path))
        validate(row)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    plan = json.loads((args.root/"plan.json").read_text())
    records = read_records(args.root, plan["entries"])
    assert len(records) == 500
    seen = {}
    for r in records:
        key = (tuple(r["tokens"].tolist()), r["has_eos"])
        assert key not in seen or seen[key] == r["split"], "Identical trajectory crosses data splits"
        seen[key] = r["split"]
    old = [r for r in records if r["seed"] < 20265000]
    real = real_records()
    torch.save(real, args.root/"real_crops.pt")
    result = dict(all=describe(records), previous43=describe(old), real_one_scene_five_crops=describe(real),
        splits={split:describe([r for r in records if r["split"]==split]) for split in ["train","val","test"]},
        requested_caps=dict(collections.Counter(str(r["cap"]) for r in records)),
        temperatures=dict(collections.Counter(str(r["temperature"]) for r in records)),
        top_p=dict(collections.Counter(str(r["top_p"]) for r in records)),
        unique_trajectories=len(seen), real_coverage={})
    for name, start, end in [("position",0,8000),("feature",8000,EOS)]:
        a, b = counts(records,start,end), counts(real,start,end)
        p, q = a/a.sum(), b/b.sum()
        m = (p+q)/2
        js = sum(float((v[v>0]*(v[v>0]/m[v>0]).log2()).sum()) for v in (p,q))/2
        result["real_coverage"][name] = dict(real_token_mass_seen=float(b[a>0].sum()/b.sum()),
                                               marginal_js_bits=js)
    write_json(args.root/"corpus_analysis.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
