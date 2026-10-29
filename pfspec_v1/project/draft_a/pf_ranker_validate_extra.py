"""用真实 test 序列补充 typed Position 与三层 COW 数值验证。

服务器用法：python -m draft_a.pf_ranker_validate_extra --checkpoint ... --output ...
仅写指定 JSON；模型、采样策略和 Position 提案实现沿用主 eval。
"""

import argparse
import copy
import json
import math
from pathlib import Path


TOLERANCE = 0.01
PREFIXES = (0, 64, 256)
MAX_DEPTH = 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_state(actual, expected, label):
    """逐项比较全部属性，保留计数器的零值键与映射类型。"""
    import numpy as np

    require(type(actual) is type(expected), f"{label} 类型不同")
    if isinstance(actual, np.ndarray):
        require(actual.dtype == expected.dtype and np.array_equal(actual, expected),
                f"{label} 数组不同")
    elif isinstance(actual, dict):
        require(actual.keys() == expected.keys(), f"{label} 键不同")
        if hasattr(actual, "default_factory"):
            require(actual.default_factory == expected.default_factory, f"{label} 默认值类型不同")
        for key in actual:
            same_state(actual[key], expected[key], f"{label}[{key!r}]")
    elif isinstance(actual, (tuple, list)):
        require(len(actual) == len(expected), f"{label} 长度不同")
        for index, (a, b) in enumerate(zip(actual, expected)):
            same_state(a, b, f"{label}[{index}]")
    elif hasattr(actual, "__dict__"):
        same_state(vars(actual), vars(expected), label)
    else:
        require(bool(actual == expected), f"{label} 值不同")


def position_tv(core, sequence):
    import torch
    from draft_a.pf_ranker_eval import cache_for, policy, typed_kinds
    from bench_gaussiangpt_pf_block2_cow import truncate

    count = len(sequence)
    require(count == 129, "Position 分布验证需要真实 seq 的前 129 个 token")
    require(next(core.parameters()).dtype == torch.float32, "模型参数需要 FP32")
    result = {"max_tv": 0.0, "position_rows": 0, "tv_comparisons": 0,
              "cache_returns": 0, "phases": []}

    def forward(tokens, cache, typed):
        start = int(cache.get_pos())
        kwargs = {"logit_types": typed_kinds(start, len(tokens))} if typed else {}
        logits = core(tokens.reshape(1, -1), kv_cache=cache, **kwargs)[0]
        require(logits.dtype == torch.float32, "输出 logits 需要 FP32")
        require(int(cache.get_pos()) == start + len(tokens), "处理后的缓存长度不同")
        return logits

    def fresh(typed):
        cache = cache_for(core, count + 16)
        return cache, forward(sequence, cache, typed)

    full_cache, reference = fresh(False)
    typed_cache, typed = fresh(True)

    def compare(full, sparse, start, name):
        phase = {"name": name, "start": start, "input_tokens": len(full),
                 "position_rows": 0, "typed_vs_full_max_tv": 0.0,
                 "full_vs_fresh_max_tv": 0.0, "typed_vs_fresh_max_tv": 0.0}
        result["phases"].append(phase)
        for index in range(len(full)):
            if (start + index) % 2:
                continue
            probabilities = [policy(row) for row in (full[index], sparse[index], reference[start + index])]
            for p in probabilities:
                require(p.dtype == torch.float32 and bool(torch.isfinite(p).all()), "采样分布需要有限 FP32 数值")
                require(abs(float(p.sum()) - 1.0) < 1e-4, "采样分布总概率不同")
            for a, b, field in ((0, 1, "typed_vs_full_max_tv"),
                                (0, 2, "full_vs_fresh_max_tv"), (1, 2, "typed_vs_fresh_max_tv")):
                tv = float((probabilities[a] - probabilities[b]).abs().sum() / 2)
                require(math.isfinite(tv), "TV 需要有限数值")
                phase[field] = max(phase[field], tv)
                result["max_tv"] = max(result["max_tv"], tv)
                result["tv_comparisons"] += 1
                require(tv < TOLERANCE, f"{name} Position 槽 {start + index} TV={tv} 超过容差")
            phase["position_rows"] += 1
            result["position_rows"] += 1

    compare(reference, typed, 0, "完整序列")
    for typed_mode in (False, True):
        cache = cache_for(core, count + 16)
        forward(sequence[:64], cache, typed_mode)
        tail = forward(sequence[64:], cache, typed_mode)
        if typed_mode:
            compare(full_tail, tail, 64, "真实前缀64")
        else:
            full_tail = tail
    for keep in (7, 64):
        end = min(keep + 16, count)
        for cache, typed_mode in ((full_cache, False), (typed_cache, True)):
            old = int(cache.get_pos())
            if old < count:
                forward(sequence[old:], cache, typed_mode)
            truncate(cache, keep, count)
            result["cache_returns"] += 1
            tail = forward(sequence[keep:end], cache, typed_mode)
            if typed_mode:
                compare(full_tail, tail, keep, f"缓存退回{keep}")
            else:
                full_tail = tail
    return result


def cow_preview(body, ranks, ranker):
    import numpy as np
    from draft_a.pf_ranker_eval import EOS, position_state, validate_sequence
    from bench_gaussiangpt_pf_block2_cow import fork_position_state

    require(len(body) >= max(PREFIXES), "真实首条 test 序列不足以覆盖 prefix256")
    result = {"max_depth": MAX_DEPTH, "completed_layers": 0, "candidate_comparisons": 0,
              "candidate_rows": 0, "state_comparisons": 0, "parent_comparisons": 0,
              "max_feature_abs_difference": 0.0, "prefixes": []}
    for prefix in PREFIXES:
        prompt = [EOS] + body[:prefix] if prefix else [EOS, body[0]]
        validate_sequence(prompt, [], ranks)
        original = position_state(prompt, ranks)
        current, reference = copy.deepcopy(original), copy.deepcopy(original)
        parents = [(original, copy.deepcopy(original))]
        previous = prompt[-2] if prefix else body[0]
        group = {"prefix_tokens": prefix, "initial_body_tokens": len(prompt) - 1,
                 "initialization": "真实P0" if prefix == 0 else "真实前缀历史",
                 "initial_position": previous, "completed_layers": 0, "layers": [],
                 "stop": "达到三层"}
        result["prefixes"].append(group)

        def parents_unchanged():
            for index, (parent, saved) in enumerate(parents):
                same_state(parent, saved, f"prefix{prefix}.父状态{index}")
                result["parent_comparisons"] += 1

        for depth in range(1, MAX_DEPTH + 1):
            actual, expected = current.propose(previous), reference.propose(previous)
            parents_unchanged()
            same_state(actual, expected, f"prefix{prefix}.层{depth}.候选")
            same_state(current, reference, f"prefix{prefix}.层{depth}.提案状态")
            result["candidate_comparisons"] += 1
            result["state_comparisons"] += 1
            if actual is None:
                group["stop"] = "候选为空"
                break
            ids, feats, _, _ = actual
            require(len(ids) > 0 and bool(np.isfinite(feats).all()), "候选与特征需要有效数值")
            difference = float(np.max(np.abs(feats - expected[1]))) if feats.size else 0.0
            result["max_feature_abs_difference"] = max(result["max_feature_abs_difference"], difference)
            result["candidate_rows"] += len(ids)
            if len(ids) == 1:
                chosen = ids[0]
            else:
                scores = ranker.predict_proba(feats)[:, 1]
                require(bool(np.isfinite(scores).all()), "ranker 分数需要有限数值")
                chosen = ids[int(np.argmax(scores))]
            parents.append((current, copy.deepcopy(current)))
            child = fork_position_state(current)
            direct = copy.deepcopy(reference)
            child.finalize(chosen)
            direct.finalize(chosen)
            parents_unchanged()
            same_state(child, direct, f"prefix{prefix}.层{depth}.提交状态")
            result["state_comparisons"] += 1
            result["completed_layers"] += 1
            group["completed_layers"] += 1
            group["layers"].append({"depth": depth, "previous": int(previous),
                                    "chosen": int(chosen), "candidate_count": len(ids)})
            current, reference, previous = child, direct, chosen
        parents_unchanged()
    return result


def run(args, result):
    import torch
    from joblib import load
    from threadpoolctl import threadpool_limits
    from bench_fast_hgb import FastHistRanker
    from draft_a import learning_eval as evaluation
    from draft_a.feature_reuse import make_model
    from draft_a.pf_ranker_eval import EOS, _test_records

    corpus, _, _, analysis = evaluation._helpers()
    records, _ = _test_records(args.source_root.resolve(), 1, analysis)
    record = records[0]
    body = record["tokens"].tolist()
    require(len(body) >= 256, "真实首条 test 序列需要覆盖 prefix256")
    result.update(test_file=record["name"], split="test", sequence_tokens=129,
                  source_body_tokens=len(body), precision="FP32", policy=dict(evaluation.POLICY),
                  tolerance=TOLERANCE, typed_position={}, cow=None)
    with torch.inference_mode(), evaluation._fp32_runtime(torch), threadpool_limits(limits=1, user_api="openmp"):
        target = corpus.load_target().float().eval().requires_grad_(False)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        draft = make_model(state["config"]).float().eval().requires_grad_(False)
        draft.load_state_dict(state["model"], strict=True)
        result["checkpoint_step"] = state.get("step")
        del state
        ranks = target.transformer.sparse_position_rank_map.cpu().tolist()
        require(draft.transformer.sparse_position_rank_map.cpu().tolist() == ranks, "两套模型的 Position 顺序不同")
        sequence = torch.tensor([EOS] + body[:128], dtype=torch.long, device="cuda")
        for name, model in (("target", target), ("draft", draft)):
            evaluation._model_ready(model, 145, torch)
            result["typed_position"][name] = position_tv(model.transformer, sequence)
        ranker = FastHistRanker(load(args.ranker))
        result["cow"] = cow_preview(body, ranks, ranker)
        torch.cuda.synchronize()
    result["max_tv"] = max(item["max_tv"] for item in result["typed_position"].values())
    result["position_rows"] = sum(item["position_rows"] for item in result["typed_position"].values())
    result.update(status="passed", passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("draft_a/runs/rollout500_20261013"))
    parser.add_argument("--ranker", type=Path, default=Path(
        "profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib"))
    args = parser.parse_args()
    if args.output.resolve() in {args.checkpoint.resolve(), args.ranker.resolve(),
                                 args.source_root.resolve() / "plan.json"}:
        parser.error("输出路径与输入文件重合")
    result = {"status": "running", "passed": False, "checkpoint": str(args.checkpoint.resolve()),
              "ranker": str(args.ranker.resolve()), "source_root": str(args.source_root.resolve())}
    try:
        run(args, result)
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "max_tv": result["max_tv"],
                      "position_rows": result["position_rows"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
