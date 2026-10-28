"""Matched FP32 forward and proposal costs for the D4 experiment."""

import json
from pathlib import Path
import statistics
import time

D8 = Path("draft_a/runs/feature_acceptance_20261014/adapt_d8_acceptance/step_3600.pt")
SOURCE = Path("draft_a/runs/rollout500_20261013")
LABELS = Path("draft_a/runs/pf_feature_distill_20261014")
EVALUATION = Path("draft_a/runs/pf_learning_budget_20261013")
ROOT = Path("draft_a/runs/d4_joint_20261014")


def load_model(path):
    import torch
    from draft_a.feature_reuse import make_model
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(state["config"]).eval().requires_grad_(False)
    model.load_state_dict(state["model"], strict=True)
    return model


def real_inputs():
    import torch
    from draft_a.analyze_v2 import real_records
    from draft_a.corpus_v2 import EOS, validate
    cached = torch.load(SOURCE / "real_crops.pt", map_location="cpu", weights_only=True)
    fresh = real_records()
    if len(cached) != len(fresh) or any(a["path"] != b["path"] or not torch.equal(a["tokens"], b["tokens"])
                                     for a, b in zip(cached, fresh)):
        raise ValueError("Real crops differ from the recorded inputs")
    out = []
    for row in fresh:
        validate(row)
        if any(torch.equal(row["tokens"], r["tokens"]) for r in out):
            continue
        row.update(seq=torch.cat([torch.tensor([EOS]), row["tokens"]]),
                   name=f"real_crop_{row['crop_seed']}", scene=Path(row["path"]).parents[1].name)
        out.append(row)
    return out


def timed(operation, repeats):
    import torch
    for _ in range(3):
        operation()
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start_event.record()
        operation()
        end_event.record()
        torch.cuda.synchronize()
        values.append(dict(wall_ms=(time.perf_counter() - started)*1000,
                           cuda_ms=start_event.elapsed_time(end_event)))
    return dict(repeats=repeats, wall_ms=statistics.mean(r["wall_ms"] for r in values),
                cuda_ms=statistics.mean(r["cuda_ms"] for r in values), samples=values)


def measure(models, root, smoke=False):
    import torch
    from draft_a.feature_acceptance_eval import cache_for, policy, draw
    from draft_a.learning_eval import _fp32_runtime
    from draft_a.feature_target_v2 import save
    seq = real_inputs()[0]["seq"].cuda().unsqueeze(0)
    prefixes, gammas = ([64], [2]) if smoke else ([64, 256, 1024], [2, 4, 6])
    repeats = 3 if smoke else 20
    rows = []
    with _fp32_runtime(torch):
        for prefix in prefixes:
            for name, model in models.items():
                core = model.transformer
                cache = cache_for(core, prefix+10)
                def prefill():
                    cache.cache_seqlens.zero_()
                    core(seq[:, :prefix], kv_cache=cache)
                prefill_value = timed(prefill, repeats)
                prefill()
                for gamma in gammas:
                    for mode in ("forward", "proposal"):
                        rng = torch.Generator(device="cuda").manual_seed(24690)
                        def operation():
                            cache.cache_seqlens.fill_(prefix)
                            for j in range(gamma):
                                logits = core(seq[:, prefix+j:prefix+j+1], kv_cache=cache)[0, -1]
                                if mode == "proposal":
                                    draw(policy(logits), rng)
                        value = timed(operation, repeats)
                        rows.append(dict(model=name, prefix=prefix, gamma=gamma, mode=mode, **value))
                rows.append(dict(model=name, prefix=prefix, gamma=0, mode="prefill", **prefill_value))
                del cache
    ratios = []
    for row in rows:
        if row["model"] != "d4":
            continue
        other = next(r for r in rows if r["model"] == "d8" and all(r[k] == row[k] for k in ["prefix", "gamma", "mode"]))
        ratios.append({k: row[k] for k in ["prefix", "gamma", "mode"]} |
                      dict(wall_ratio=row["wall_ms"]/other["wall_ms"], cuda_ratio=row["cuda_ms"]/other["cuda_ms"]))
    report = dict(precision="FP32", rows=rows, d4_over_d8=ratios,
                  scope="fixed real token inputs; sampled proposals are timed but known tokens advance the cache",
                  timing="cache reused; per-row position reset and forward/sampling included; allocation/release excluded",
                  prefill="fixed cache reset and complete input forward; model loading excluded")
    save(root / "cost.json", report)
    print(json.dumps(dict(event="cost", d4_over_d8=ratios)), flush=True)
    return report
