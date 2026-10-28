#!/usr/bin/env python3
"""CPU-only R1 residual-mass feasibility probe for W1/W2/W4 proposals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import time


os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COLLECTION = ROOT / "docs/project/artifacts/pf2x_loop_20261019/round01/cost_probe_review_20261020/results/pq"
OUTPUT_DIR = ROOT / "docs/project/artifacts/pf2x_breakthrough_20261021"
OUTPUT_JSON = OUTPUT_DIR / "RESIDUAL_FEASIBILITY.json"
OUTPUT_MD = OUTPUT_DIR / "RESIDUAL_FEASIBILITY.md"
SEED = 20261022
FLAT_TEMPERATURE = 1.2
WIDTHS = (1, 2, 4)
STRATEGIES = ("iid_q1", "without_replacement_q1", "without_replacement_flat12")


def corrected_probability(values, name: str):
    import numpy as np

    out = np.asarray(values, dtype=np.float64).copy()
    if out.ndim != 1 or not np.all(np.isfinite(out)) or np.any(out < 0.0):
        raise ValueError(f"{name} must be a finite nonnegative vector")
    total = float(out.sum())
    if total <= 0.0:
        raise ValueError(f"{name} has no probability mass")
    out /= total
    pivot = int(np.argmax(out))
    out[pivot] += 1.0 - float(out.sum())
    if out[pivot] < 0.0:
        raise ValueError(f"{name} normalization produced a negative entry")
    return out


def softmax_logits(logits, temperature: float):
    import numpy as np

    x = np.asarray(logits, dtype=np.float64) / temperature
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("raw MLP logits must be a finite vector")
    x -= float(x.max())
    e = np.exp(x)
    return corrected_probability(e, "flattened q2")


def draw_rows(probability_rows, rng):
    import numpy as np

    totals = probability_rows.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ValueError("cannot draw from an empty conditional distribution")
    normalized = probability_rows / totals[:, None]
    cdf = np.cumsum(normalized, axis=1, dtype=np.float64)
    cdf[:, -1] = 1.0
    uniforms = rng.random(len(cdf))
    chosen = np.sum(cdf <= uniforms[:, None], axis=1)
    return np.minimum(chosen, cdf.shape[1] - 1).astype(np.int32)


def conditioned_proposals(base, histories):
    """Rows of q conditioned on excluding every earlier candidate in history."""
    import numpy as np

    count = len(histories)
    proposal = np.broadcast_to(base, (count, len(base))).copy()
    row_ids = np.arange(count)
    for column in range(histories.shape[1]):
        proposal[row_ids, histories[:, column]] = 0.0
    totals = proposal.sum(axis=1)
    valid = totals > 0.0
    proposal[valid] /= totals[valid, None]
    if np.any(valid):
        valid_ids = np.flatnonzero(valid)
        pivots = np.argmax(proposal[valid], axis=1)
        proposal[valid_ids, pivots] += 1.0 - proposal[valid].sum(axis=1)
    return proposal, valid


def exact_iid_mass(p, q1, acceptance1: float, rejection1: float):
    """Repeated iid q1 has a history-independent residual at each stage."""
    import numpy as np

    residual_raw = np.maximum(p - q1, 0.0)
    residual_total = float(residual_raw.sum())
    if rejection1 == 0.0:
        return {"w1": acceptance1, "w2": acceptance1, "w4": acceptance1}, 0.0
    if residual_total == 0.0:
        raise ArithmeticError("positive first rejection mass has zero Target residual")
    residual = corrected_probability(residual_raw, "r1")
    survived = 1.0
    accepted_after_rejection = 0.0
    result = {"w1": acceptance1}
    max_mass_error = 0.0
    for stage in range(2, 5):
        overlap = np.minimum(residual, q1)
        beta = float(overlap.sum())
        residual_next = residual - overlap
        reject_mass = float(residual_next.sum())
        candidate_reject_mass = float((q1 - overlap).sum())
        max_mass_error = max(max_mass_error,
                             abs(beta + reject_mass - 1.0),
                             abs(reject_mass - candidate_reject_mass))
        accepted_after_rejection += survived * beta
        survived *= reject_mass
        if stage == 2:
            result["w2"] = acceptance1 + rejection1 * accepted_after_rejection
        if reject_mass == 0.0:
            result["w4"] = acceptance1 + rejection1 * accepted_after_rejection
            break
        residual = corrected_probability(residual_next, f"r{stage}")
    result.setdefault("w2", acceptance1)
    result.setdefault("w4", acceptance1 + rejection1 * accepted_after_rejection)
    return result, max_mass_error


def simulate_without_replacement(p, q1, proposal_base, acceptance1: float,
                                 rejection1: float, draws: int, rng):
    """MC over histories of rejected proposals; each stage mass is summed exactly."""
    import numpy as np

    if rejection1 == 0.0:
        return dict(w2=acceptance1, w4=acceptance1, mc_variance_w2=0.0,
                    mc_variance_w4=0.0, mass_identity_max_error=0.0,
                    proposal_residual_mass_max_error=0.0,
                    exhausted_proposal_paths=0)

    delta = np.maximum(q1 - p, 0.0)
    delta_total = float(delta.sum())
    if delta_total == 0.0:
        raise ArithmeticError("positive first rejection mass has no rejected-candidate weights")
    first_distribution = corrected_probability(delta, "first rejected candidate weights")
    first_ids = np.empty(draws, dtype=np.int32)
    cdf = np.cumsum(first_distribution)
    cdf[-1] = 1.0
    first_ids[:] = np.minimum(
        np.searchsorted(cdf, rng.random(draws), side="right"), len(q1) - 1
    )

    residual_raw = np.maximum(p - q1, 0.0)
    residual_total = float(residual_raw.sum())
    if residual_total == 0.0:
        raise ArithmeticError("positive first rejection mass has zero Target residual")
    r1 = corrected_probability(residual_raw, "r1")
    residual = np.broadcast_to(r1, (draws, len(r1))).copy()
    histories = first_ids[:, None]
    active = np.ones(draws, dtype=bool)
    survival = np.ones(draws, dtype=np.float64)
    cumulative = np.zeros(draws, dtype=np.float64)
    accepted_after_rejection = {}
    max_identity_error = 0.0
    max_residual_mass_error = 0.0
    exhausted_paths = 0

    for stage in range(2, 5):
        active_ids = np.flatnonzero(active)
        if not len(active_ids):
            break
        proposals, has_support = conditioned_proposals(
            proposal_base, histories[active_ids]
        )
        if np.any(~has_support):
            active[active_ids[~has_support]] = False
            exhausted_paths += int((~has_support).sum())
        good_ids = active_ids[has_support]
        if not len(good_ids):
            continue
        q_rows = proposals[has_support]
        r_rows = residual[good_ids]
        overlap = np.minimum(r_rows, q_rows)
        beta = overlap.sum(axis=1)
        r_reject = r_rows - overlap
        q_reject = q_rows - overlap
        r_reject_mass = r_reject.sum(axis=1)
        q_reject_mass = q_reject.sum(axis=1)
        mass_error = np.maximum.reduce((
            np.abs(beta + r_reject_mass - 1.0),
            np.abs(r_reject_mass - q_reject_mass),
            np.abs(q_rows.sum(axis=1) - 1.0),
        ))
        max_residual_mass_error = max(max_residual_mass_error,
                                      float(mass_error.max(initial=0.0)))
        if float(mass_error.max(initial=0.0)) > 2e-10:
            raise ArithmeticError("candidate and residual rejection masses diverged")
        if np.any(beta < -1e-12) or np.any(beta > 1.0 + 1e-12):
            raise ArithmeticError("conditional candidate acceptance mass is outside [0,1]")
        beta = np.minimum(beta, 1.0)
        cumulative[good_ids] += survival[good_ids] * beta
        survival[good_ids] *= r_reject_mass
        accepted_after_rejection[f"w{stage}"] = cumulative.copy()

        if stage == 4:
            active[good_ids] = False
            continue

        can_continue = (r_reject_mass > 0.0) & (q_reject_mass > 0.0)
        impossible = (r_reject_mass > 0.0) != (q_reject_mass > 0.0)
        if np.any(impossible):
            raise ArithmeticError("positive residual has no matching rejected-candidate mass")
        stop_ids = good_ids[~can_continue]
        active[stop_ids] = False
        continue_ids = good_ids[can_continue]
        if not len(continue_ids):
            continue
        continue_mask = can_continue
        next_ids = draw_rows(q_reject[continue_mask], rng)
        next_residual = r_reject[continue_mask] / r_reject_mass[continue_mask, None]
        next_residual /= next_residual.sum(axis=1, keepdims=True)
        next_pivots = np.argmax(next_residual, axis=1)
        next_residual[np.arange(len(next_residual)), next_pivots] += (
            1.0 - next_residual.sum(axis=1)
        )
        new_histories = np.full((len(continue_ids), histories.shape[1] + 1),
                                -1, dtype=np.int32)
        new_histories[:, :-1] = histories[continue_ids]
        new_histories[:, -1] = next_ids
        expanded_histories = np.full(
            (draws, histories.shape[1] + 1), -1, dtype=np.int32
        )
        expanded_histories[:, :-1] = histories
        expanded_histories[continue_ids] = new_histories
        histories = expanded_histories
        residual[continue_ids] = next_residual
        active[continue_ids] = True

    g2 = accepted_after_rejection.get("w2", np.zeros(draws, dtype=np.float64))
    g4 = accepted_after_rejection.get("w4", cumulative)
    identity_error = np.abs(g4 + survival - 1.0)
    max_identity_error = float(identity_error.max(initial=0.0))
    w2_values = acceptance1 + rejection1 * g2
    w4_values = acceptance1 + rejection1 * g4
    return dict(
        w2=float(w2_values.mean()), w4=float(w4_values.mean()),
        mc_variance_w2=float(w2_values.var(ddof=1) / draws) if draws > 1 else 0.0,
        mc_variance_w4=float(w4_values.var(ddof=1) / draws) if draws > 1 else 0.0,
        conditional_extra_w2=float(g2.mean()), conditional_extra_w4=float(g4.mean()),
        mass_identity_max_error=max_identity_error,
        proposal_residual_mass_max_error=max_residual_mass_error,
        exhausted_proposal_paths=exhausted_paths,
    )


def summarize_rows(rows, strategy: str, width: int, draws: int):
    import numpy as np

    key = f"w{width}"
    values = np.asarray([row["strategies"][strategy][key]["accepted_mass"]
                         for row in rows], dtype=np.float64)
    variances = np.asarray([row["strategies"][strategy][key]["mc_variance"]
                            for row in rows], dtype=np.float64)
    mean = float(values.mean())
    mc_se = math.sqrt(float(variances.sum()) / (len(values) ** 2))
    return dict(
        contexts=len(values), mean_accepted_mass=mean,
        mc_standard_error=mc_se,
        mc_95_interval=[max(0.0, mean - 1.96 * mc_se),
                        min(1.0, mean + 1.96 * mc_se)],
        mean_extra_over_w1=float(np.mean(values - np.asarray(
            [row["first_accept_mass"] for row in rows], dtype=np.float64))),
        mean_conditional_extra_after_first_rejection=float(np.mean([
            row["strategies"][strategy][key]["conditional_extra_after_first_rejection"]
            for row in rows
        ])),
        monte_carlo_draws_per_context=(draws if strategy != "iid_q1" and width > 1 else 0),
    )


def summarize_group(rows, strategy: str, width: int, draws: int):
    groups = {}
    for row in rows:
        key = (row["scene"], int(row["prefix"]))
        groups.setdefault(key, []).append(row)
    output = []
    for (scene, prefix), group_rows in sorted(groups.items()):
        item = summarize_rows(group_rows, strategy, width, draws)
        item.update(scene=scene, prefix=prefix)
        output.append(item)
    return output


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def make_report(result: dict) -> str:
    lines = [
        "# 多候选残差质量与成本空间",
        "",
        "## 数据与方法",
        "",
        f"输入为 `{result['input']['vectors']}`，包含 {result['input']['rows']} 个真实 p/q 上下文、词表宽度 {result['input']['vocabulary_size']}；同目录 contexts 表提供 source、prefix 与向量索引。数据来自既有 R1 开发集，未新增采集。运行使用 NumPy CPU 路径，seed={result['sampling']['seed']}，每行每种不放回策略抽 {result['sampling']['draws_per_context']} 条拒绝历史。",
        "",
        "首候选接受质量为 `A1=Σ min(p,q1)`；首拒候选按 `δ(a)=[q1(a)-p(a)]₊` 采样并按 `R1=Σδ(a)` 加权，剩余目标为 `r1=normalize((p-q1)₊)`。每个后续候选的条件接受质量精确求和 `βj=Σ min(rj-1,qj)`；进入下一层时，候选拒绝值按 `[qj-rj-1]₊/(1-βj)` 抽取，目标残差按 `(rj-1-qj)₊/(1-βj)` 更新。累积接受质量使用各层前序拒绝质量乘本层 βj，不把“至少一支命中”直接当概率。",
        "",
        "比较 `iid_q1`（相同 q1 独立重提，因残差不依赖历史可精确递推）、`without_replacement_q1`（q1 排除所有先前候选后重归一），以及固定 `T=1.2` 的 `without_replacement_flat12`（同次 MLP raw logits 全词表 softmax，排除先前候选；不读取 p）。W1、W2、W4 分别表示 1、2、4 个候选机会。",
        "",
        "## 接受质量",
        "",
        "下表是 480 个上下文等权均值；95% 区间只反映每行有限 Monte Carlo 抽样误差，不含来源或上下文采样误差。首轮 W1 是逐行精确计算。",
        "",
        "| 策略 | W1 精确 | W2 累积 | W4 累积 | W4-W1 | W4 条件于首拒 | W4 MC 标准误 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        s1 = result["summary"][strategy]["w1"]["mean_accepted_mass"]
        s2 = result["summary"][strategy]["w2"]
        s4 = result["summary"][strategy]["w4"]
        lines.append(
            f"| {strategy} | {s1:.6f} | {s2['mean_accepted_mass']:.6f} "
            f"| {s4['mean_accepted_mass']:.6f} | {s4['mean_extra_over_w1']:.6f} "
            f"| {s4['mean_conditional_extra_after_first_rejection']:.6f} "
            f"| {s4['mc_standard_error']:.6f} |"
        )
    lines.extend([
        "",
        "每行的完整估计、MC 误差、source/prefix 汇总与质量守恒误差见配套 JSON。`min(p,k·q1)` 另作 tokenwise 乐观覆盖量：",
        "",
        "| k | 等权均值 `Σ min(p,k·q1)` |",
        "|---:|---:|",
    ])
    for k in (1, 2, 4):
        value = result["optimistic_tokenwise_coverage"][str(k)]["mean"]
        lines.append(f"| {k} | {value:.6f} |")
    lines.extend([
        "",
        "该量只描述 k 倍 tokenwise 覆盖的乐观尺度，不是连续条件提案的精确递推值，也不是自适应 qj 的严格界。实际串行候选遵循前一轮拒绝后的残差与候选排除规则，结果以 W2/W4 列为准。",
        "",
        "## 速度空间与成本上限",
        "",
        "当前 PF/Target 速度比约 1.68158x；另一阶段记录为 1.71250x。到 2x 需要吞吐再增约 18.9% 或 16.8%，等价于 PF 总时间分别下降约 15.9% 或 14.4%。候选接受质量增量本身不能换算为速度：还要乘以候选分支自己的 P2 接受概率，并确认它确实少执行一次 Target 前向。",
        "",
        "若极宽松地假定每个额外接受事件都能省下一个完整 Target 前向，令 W4-W1 为 `ΔA`，则每次相关 decode 的额外分支成本上限为 `ΔA × C_saved`。R1 既有 shape 测量中 2-token 与 4-token Target 前向约为 17.36 ms 与 18.09 ms；以约 17.72 ms 作示意，真实预算还要按 P1 到达率、各分支 P2 接受率下调，并扣除候选采样、残差、分支 KV、mask 与同步成本。若只是串行试候选却仍保留同样的后续 Target 调用，净节省为零或负值。",
        "",
        "这里的向量只给出了固定前缀上的 p、q1 和同次 MLP raw logits，没有 qj 候选分支下的 Target P2 分布，也没有分支 KV 或真实 decode 耗时。因此本结果能筛掉候选覆盖明显不足的方案，不能证明 W2/W4 可提速或达到 2x。",
        "",
        "## 限制",
        "",
        "MC 区间只覆盖固定 480 行向量内的抽样波动；来源和上下文来自既有开发集，不能代表独立评测来源。`T=1.2` 是预先固定的单一平坦分布候选，没有做温度网格。该脚本不调用 GPU、模型或生产解码器。",
        "",
    ])
    return "\n".join(lines)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def run(draws: int, collection: Path) -> dict:
    import numpy as np

    started = time.monotonic()
    vectors_path = collection / "vectors.npz"
    contexts_path = collection / "contexts.jsonl"
    if not vectors_path.is_file() or not contexts_path.is_file():
        raise FileNotFoundError(f"R1 inputs missing under {collection}")
    contexts = [json.loads(line) for line in contexts_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    with np.load(vectors_path, allow_pickle=False) as arrays:
        required = {"p_f", "q1", "raw_logits"}
        if set(arrays.files) != required:
            raise ValueError(f"unexpected NPZ keys: {arrays.files}")
        p_all = arrays["p_f"].astype(np.float64)
        q_all = arrays["q1"].astype(np.float64)
        logits_all = arrays["raw_logits"].astype(np.float64)
    if p_all.shape != q_all.shape or p_all.shape != logits_all.shape:
        raise ValueError("p_f, q1 and raw_logits shapes differ")
    if p_all.ndim != 2 or len(contexts) != p_all.shape[0] or p_all.shape[0] != 480:
        raise ValueError("expected 480 aligned R1 contexts and probability rows")
    if p_all.shape[1] != 4096:
        raise ValueError(f"expected 4096 Feature values, got {p_all.shape[1]}")
    ordered = [None] * len(contexts)
    for row in contexts:
        index = int(row["vector_index"])
        if not 0 <= index < len(ordered) or ordered[index] is not None:
            raise ValueError("context vector_index values must be unique and dense")
        ordered[index] = row
    if any(row is None for row in ordered):
        raise ValueError("context table does not cover every vector row")

    row_results = []
    max_mass_error = 0.0
    rngs = {
        "without_replacement_q1": [
            np.random.default_rng(np.random.SeedSequence([SEED, i, 1]))
            for i in range(len(ordered))
        ],
        "without_replacement_flat12": [
            np.random.default_rng(np.random.SeedSequence([SEED, i, 2]))
            for i in range(len(ordered))
        ],
    }

    for i, context in enumerate(ordered):
        p = corrected_probability(p_all[i], "p_f")
        q1 = corrected_probability(q_all[i], "q1")
        raw = logits_all[i]
        if context.get("p1_accepted_actual") is not True:
            raise ValueError("all saved R1 rows must be conditioned on an accepted P1")
        if (context.get("policy", {}).get("temperature") != 0.9
                or context.get("policy", {}).get("top_p") != 0.9):
            raise ValueError("R1 row policy differs from the recorded 0.9/0.9 setting")

        overlap1 = np.minimum(p, q1)
        acceptance1 = float(overlap1.sum())
        delta1 = np.maximum(q1 - p, 0.0)
        rejection1 = float(delta1.sum())
        residual1 = np.maximum(p - q1, 0.0)
        residual1_mass = float(residual1.sum())
        mass_error = max(abs(acceptance1 + rejection1 - 1.0),
                         abs(rejection1 - residual1_mass))
        max_mass_error = max(max_mass_error, mass_error)
        if mass_error > 2e-10:
            raise ArithmeticError("R1 first rejection and Target residual masses differ")

        iid, iid_error = exact_iid_mass(p, q1, acceptance1, rejection1)
        max_mass_error = max(max_mass_error, iid_error)
        flat = softmax_logits(raw, FLAT_TEMPERATURE)
        q1_mc = simulate_without_replacement(
            p, q1, q1, acceptance1, rejection1, draws, rngs["without_replacement_q1"][i]
        )
        flat_mc = simulate_without_replacement(
            p, q1, flat, acceptance1, rejection1, draws,
            rngs["without_replacement_flat12"][i]
        )

        row_modes = {}
        for strategy, exact, mc in (
            ("iid_q1", iid, None),
            ("without_replacement_q1", None, q1_mc),
            ("without_replacement_flat12", None, flat_mc),
        ):
            row_modes[strategy] = {"w1": dict(
                accepted_mass=acceptance1, mc_variance=0.0,
                conditional_extra_after_first_rejection=0.0,
            )}
            for width in (2, 4):
                if exact is not None:
                    accepted = float(exact[f"w{width}"])
                    var = 0.0
                    conditional_extra = ((accepted - acceptance1) / rejection1
                                         if rejection1 > 0.0 else 0.0)
                else:
                    accepted = float(mc[f"w{width}"])
                    var = float(mc[f"mc_variance_w{width}"])
                    conditional_extra = float(mc[
                        f"conditional_extra_w{width}"
                    ])
                row_modes[strategy][f"w{width}"] = dict(
                    accepted_mass=min(1.0, max(acceptance1, accepted)),
                    mc_variance=var,
                    conditional_extra_after_first_rejection=conditional_extra,
                )

        optimistic = {str(k): float(np.minimum(p, k * q1).sum()) for k in (1, 2, 4)}
        row_results.append(dict(
            vector_index=i, sample_id=context["sample_id"], scene=context["scene"],
            prefix=int(context["prefix"]), p1_accept_probability=float(
                context["p1_accept_probability"]),
            first_accept_mass=acceptance1, first_rejection_mass=rejection1,
            strategies=row_modes, optimistic_tokenwise_coverage=optimistic,
            residual_mass_error=mass_error,
            q1_no_repeat_mass_identity_max_error=q1_mc["mass_identity_max_error"],
            q1_no_repeat_transition_mass_error=q1_mc[
                "proposal_residual_mass_max_error"],
            q1_no_repeat_exhausted_paths=q1_mc["exhausted_proposal_paths"],
            flat_no_repeat_mass_identity_max_error=flat_mc[
                "mass_identity_max_error"],
            flat_no_repeat_transition_mass_error=flat_mc[
                "proposal_residual_mass_max_error"],
            flat_no_repeat_exhausted_paths=flat_mc["exhausted_proposal_paths"],
        ))
        if (i + 1) % 48 == 0:
            print(json.dumps(dict(done=i + 1, total=len(ordered),
                                  elapsed_seconds=time.monotonic() - started)), flush=True)

    summary = {}
    for strategy in STRATEGIES:
        summary[strategy] = {
            f"w{width}": summarize_rows(row_results, strategy, width, draws)
            for width in WIDTHS
        }
    group_summary = {
        strategy: {
            f"w{width}": summarize_group(row_results, strategy, width, draws)
            for width in WIDTHS
        }
        for strategy in STRATEGIES
    }
    optimistic_summary = {}
    for k in (1, 2, 4):
        vals = [row["optimistic_tokenwise_coverage"][str(k)] for row in row_results]
        optimistic_summary[str(k)] = dict(mean=float(np.mean(vals)),
                                          contexts=len(vals))

    return dict(
        status="complete", created_at=datetime.now(timezone.utc).isoformat(),
        cpu_only=True, gpu_used=False,
        input=dict(vectors=display_path(vectors_path),
                   contexts=display_path(contexts_path), rows=len(row_results),
                   vocabulary_size=int(p_all.shape[1]), arrays=["p_f", "q1", "raw_logits"],
                   source_scope="R1 existing development contexts; no new collection"),
        sampling=dict(seed=SEED, draws_per_context=draws,
                      separate_seed_streams=True,
                      history_sampling=("first rejected candidate from delta/R1; later rejected "
                                        "proposal from [qj-rj-1]+/(1-beta_j)"),
                      flattened_q2_temperature=FLAT_TEMPERATURE,
                      flattened_q2_policy="softmax(raw_logits / 1.2), full 4096-value support"),
        recurrence=dict(
            first_rejection_weight="delta(a)=[q1(a)-p(a)]+; R1=sum(delta)",
            first_residual="r1=normalize((p-q1)+)",
            stage_acceptance="beta_j=sum(min(r_{j-1},q_j))",
            stage_residual="r_j=normalize((r_{j-1}-q_j)+)",
            conditional_rejected_candidate=(
                "sample c from (q_j-min(r_{j-1},q_j))/(1-beta_j); "
                "this is distinct from the residual output distribution"),
            cumulative_mass=("A1 + R1 * E[beta_2 + (1-beta_2)*beta_3 + ...] "
                             "over correctly weighted rejected-candidate histories"),
            iid_q1="exact deterministic recursion; residual does not depend on candidate history"),
        summary=summary, per_source_prefix=group_summary,
        optimistic_tokenwise_coverage=optimistic_summary,
        max_mass_conservation_error=max_mass_error,
        uncertainty=dict(
            method="row-wise MC variance of conditional cumulative accepted mass; combine over fixed rows",
            interval="normal mean +/- 1.96 standard errors, clipped to [0,1]",
            includes_source_or_context_sampling_error=False),
        per_context=row_results,
        elapsed_seconds=time.monotonic() - started,
        environment=dict(python=platform.python_version(), numpy=np.__version__),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, choices=(128, 256), default=128)
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION,
                        help="directory containing vectors.npz and contexts.jsonl")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="directory for RESIDUAL_FEASIBILITY.json and .md")
    args = parser.parse_args()
    collection_dir = args.collection_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_json = output_dir / "RESIDUAL_FEASIBILITY.json"
    output_md = output_dir / "RESIDUAL_FEASIBILITY.md"
    existing = [path for path in (output_json, output_md) if path.exists()]
    if existing:
        parser.error("refusing to overwrite existing output file(s): "
                     + ", ".join(str(path) for path in existing))
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run(args.draws, collection_dir)
    json_text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    report = make_report(result)
    write_atomic(output_json, json_text)
    write_atomic(output_md, report)
    print(json.dumps(dict(status=result["status"], json=str(output_json),
                          report=str(output_md), rows=result["input"]["rows"],
                          draws=args.draws, elapsed_seconds=result["elapsed_seconds"]),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
