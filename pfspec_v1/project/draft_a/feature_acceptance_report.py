"""Summarize a completed independent test and online acceptance run."""

import argparse
import json
import math
from pathlib import Path


LIMITATIONS = [
    "本轮复用 38 epoch ASE 预训练权重。预训练预算不同，无法与旧随机初始化实验作配对比较。",
    "当前联合 P/F 模型的结果不能证明共享层存在干扰，也不能证明分离模型更优。",
    "本轮只有一个微调种子，尚未测量训练种子之间的变化。",
    "吞吐测量采用 FP32，生产 BF16 的速度仍需实测。",
    "在线范围为 12 个场景、512 token 续写及 0、64、256 token 前缀；未来长上下文仍待测量。",
    "30% 是各配置的平均 F 接受率目标，配置平均采用合并提案的 accepted/verified。个别轨迹仍可能低于 30%，报告保留最小值与低于目标的运行数。",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _same_weight(left, right):
    a, b = Path(left), Path(right)
    if a.is_absolute() == b.is_absolute():
        return a == b
    absolute, relative = (a, b) if a.is_absolute() else (b, a)
    return len(relative.parts) > 1 and absolute.parts[-len(relative.parts):] == relative.parts


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _interval(values, key, point):
    ci = values[key]
    _require(isinstance(ci, list) and len(ci) == 2, f"Invalid interval: {key}")
    if point is not None:
        _require(all(_number(v) for v in ci) and ci[0] <= ci[1], f"Missing interval: {key}")
    return ci


def _near(left, right, message):
    _require(_number(left) and _number(right) and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12), message)


def _kind_counts(group, kind):
    counts = {key: group[f"{kind}_{key}"] for key in ("accepted", "verified", "proposed")}
    a, v, p = (counts[key] for key in ("accepted", "verified", "proposed"))
    _require(all(isinstance(n, int) and n >= 0 for n in (a, v, p)) and a <= v <= p,
             f"Invalid {kind} proposal counts")
    ci = group["bootstrap"]["ci95"]
    for denominator in ("verified", "proposed"):
        key = f"{kind}_acceptance_{denominator}"
        rate = a / counts[denominator] if counts[denominator] else None
        if rate is None:
            _require(group[key] is None, f"Unexpected {key}")
        else:
            _near(group[key], rate, f"Count/rate mismatch: {key}")
        counts[f"accepted_over_{denominator}"] = rate
        counts[f"accepted_over_{denominator}_ci95"] = _interval(ci, key, rate)
    return counts


def _group(group):
    result = {key: group[key] for key in (
        "mode", "prompt_tokens", "gamma", "scene_count", "run_count",
        "generated_tokens", "seconds", "tokens_per_second")}
    _require(result["seconds"] > 0, "Nonpositive online duration")
    _near(result["tokens_per_second"], result["generated_tokens"] / result["seconds"],
          "Throughput/count mismatch")
    result["tokens_per_second_ci95"] = _interval(
        group["bootstrap"]["ci95"], "tokens_per_second", result["tokens_per_second"])
    result["bootstrap"] = {key: group["bootstrap"][key] for key in (
        "method", "scene_count", "resamples", "seed")}
    if group["mode"] == "draft":
        result.update({kind: _kind_counts(group, kind) for kind in ("feature", "position", "eos")})
        result["speedup_vs_target_only"] = group["speedup_vs_target_only"]
        paired = group["paired_speedup_bootstrap"]
        result["speedup_ci95"] = _interval(paired["ci95"], "speedup_vs_target_only",
                                            result["speedup_vs_target_only"])
        result["paired_speedup_bootstrap"] = {key: paired[key] for key in (
            "method", "scene_count", "resamples", "seed")}
    return result


def _feature_runs(rows):
    rates = [r["feature_accepted"] / r["feature_verified"] for r in rows if r["feature_verified"] > 0]
    return dict(per_run_feature_min=min(rates) if rates else None,
                per_run_feature_max=max(rates) if rates else None,
                feature_runs_below_30=sum(rate < 0.30 for rate in rates), feature_runs_count=len(rates))


def build_summary(history, plan, selected, evaluation, metrics, calibration=None):
    """Build the report data without file writes or model imports."""
    _require(evaluation.get("status") == "complete", "Evaluation status must be complete")
    _require(evaluation["split"] == "test", "Independent test split is required")
    _require(_same_weight(selected["checkpoint"], evaluation["checkpoint"]), "Selected weight differs")
    _require(selected["step"] == evaluation["checkpoint_step"], "Selected step differs")
    _require(evaluation["target_precision"] == evaluation["draft_precision"] == "FP32",
             "Expected FP32 evaluation")
    policy = evaluation["policy"]
    _require(policy["temperature"] == 0.9 and policy["top_p"] == 0.9, "Expected fixed 0.9/0.9 policy")
    _require(sorted(row["step"] for row in history) == [0, 200, 600, 1200], "Expected four training points")
    points = []
    for row in sorted(history, key=lambda r: r["step"]):
        measured = metrics[row["step"]]
        _require(measured.get("complete") is True, "Training-point evaluation is incomplete")
        _require(_same_weight(measured["checkpoint"], row["checkpoint"]), "Training-point weight differs")
        splits = measured["splits"]
        _near(splits["val"]["feature"]["pooled_overlap"], row["feature"], "History F differs")
        _near(splits["val"]["position"]["pooled_overlap"], row["position"], "History P differs")
        points.append(dict(step=row["step"], train_f=splits["train"]["feature"]["pooled_overlap"],
                           val_f=row["feature"], val_p=row["position"],
                           val_feature_hard_nll=splits["val"]["feature"]["token_weighted_ce"],
                           trained_tokens=row["trained_tokens"],
                           trained_feature_tokens=row["trained_feature_tokens"], seconds=row["seconds"]))
    chosen = next(row for row in history if row["step"] == selected["step"])
    _require(_same_weight(chosen["checkpoint"], selected["checkpoint"]), "Selection/history weight differs")
    _near(chosen["feature"], selected["feature"], "Selection/history F differs")
    _require(plan["steps"] == selected["step"] == 1200, "Expected completed 1200 updates")
    _require(plan["trajectories_per_update"] == 2, "Expected two trajectories per update")
    offline = evaluation["offline"]
    _require(offline.get("complete") is True and offline["split"] == "test" and
             offline["scene_count"] == offline["expected_scene_count"] == 50,
             "A completed independent test50 is required")
    _require(len(evaluation["selected_test_files"]) == 50 and
             evaluation["selected_test_files"] == evaluation["expected_test_files"], "Test50 file list differs")
    test = dict(scene_count=50, split="test")
    ci = offline["bootstrap"]["ci95"]
    for kind in ("feature", "position"):
        measured = offline["summary"][kind]
        rate = measured["pooled_overlap"]
        _require(_number(rate) and 0 <= rate <= 1, "Invalid test overlap")
        test[kind] = dict(overlap=rate, overlap_ci95=_interval(ci, kind + "/pooled_overlap", rate),
                          sample_count=measured["sample_count"], token_count=measured["token_count"],
                          equal_trajectory_mean_overlap=measured["equal_trajectory_mean_overlap"])
    test["bootstrap"] = {key: offline["bootstrap"][key] for key in (
        "method", "scene_count", "resamples", "seed")}
    online = evaluation["online"]
    params = evaluation["parameters"]
    _require(online["scene_count"] == params["online_scenes"] == 12 and params["tokens"] == 512 and
             sorted(params["prefixes"]) == [0, 64, 256] and sorted(params["gamma"]) == [2, 4],
             "Expected online12, 512 tokens, three prefixes and two gamma values")
    _require(not online["unavailable_prefixes"], "Some requested prefixes are unavailable")
    groups = online["summary"]["groups"]
    draft = sorted([_group(g) for g in groups if g["mode"] == "draft"],
                   key=lambda g: (g["prompt_tokens"], g["gamma"]))
    direct = sorted([_group(g) for g in groups if g["mode"] == "target_only"],
                    key=lambda g: g["prompt_tokens"])
    _require([(g["prompt_tokens"], g["gamma"]) for g in draft] ==
             [(p, k) for p in (0, 64, 256) for k in (2, 4)], "Expected six online Draft groups")
    _require([(g["prompt_tokens"], g["gamma"]) for g in direct] == [(p, 0) for p in (0, 64, 256)],
             "Expected three Target-only groups")
    for g in draft + direct:
        _require(g["scene_count"] == g["run_count"] == 12, "Incomplete online group")
        rows = [r for r in online["rows"] if (r["mode"], r["prompt_tokens"], r["k"]) ==
                (g["mode"], g["prompt_tokens"], g["gamma"])]
        _require(len(rows) == 12 and len({r["name"] for r in rows}) == 12, "Online scene rows differ")
        _near(sum(r["generated_tokens"] for r in rows), g["generated_tokens"], "Online token total differs")
        _near(math.fsum(r["seconds"] for r in rows), g["seconds"], "Online duration total differs")
        if g["mode"] == "draft":
            for kind in ("feature", "position", "eos"):
                for key in ("accepted", "verified", "proposed"):
                    _require(sum(r[f"{kind}_{key}"] for r in rows) == g[kind][key], "Online count total differs")
            g.update(_feature_runs(rows))
            reference = next(r for r in direct if r["prompt_tokens"] == g["prompt_tokens"])
            _near(g["speedup_vs_target_only"], g["tokens_per_second"] / reference["tokens_per_second"],
                  "Speedup/reference mismatch")
    numerical = dict(toy_test=evaluation["toy_test"], verify_blocks=online["validation"]["verify_blocks"])
    trials = online["validation"]["target_target"]
    _require(trials, "Missing Target/Target trials")
    numerical["target_target"] = dict(trial_count=len(trials),
        min_acceptance_verified=min(r["acceptance_verified"] for r in trials),
        max_acceptance_verified=max(r["acceptance_verified"] for r in trials),
        accepted=sum(r["accepted"] for r in trials), verified=sum(r["verified"] for r in trials),
        proposed=sum(r["proposed"] for r in trials))
    terminal = online["validation"]["terminal_eos"]
    numerical["terminal_eos"] = dict(trial_count=len(terminal),
        all_passed=bool(terminal) and all(r["eos"] and r["generated_tokens"] == 1 and
                                         r["eos_verified"] == 1 for r in terminal))
    numerical["calibration"] = dict(included=False, reason="可选校准记录未提供。")
    if calibration is not None:
        finished = calibration.get("status") == "complete" if "status" in calibration else calibration.get("complete") is True
        if finished and _same_weight(calibration["checkpoint"], evaluation["checkpoint"]):
            numerical["calibration"] = dict(included=True,
                original_evaluation_comparison=calibration.get("original_evaluation_comparison"),
                policy=calibration.get("policy"))
        else:
            numerical["calibration"]["reason"] = "可选校准记录未完成或权重不同，未采用。"
    training = {key: plan[key] for key in ("initial_checkpoint", "steps", "lr", "schedule", "seed",
                "objective", "feature_temperature", "train_precision", "evaluation_precision",
                "trajectories_per_update")}
    training.update(points=points, label_trajectory_count=len(plan["train_files"]),
                    trained_tokens=chosen["trained_tokens"], trained_feature_tokens=chosen["trained_feature_tokens"],
                    elapsed_seconds=chosen["seconds"], elapsed_includes_development_evaluations=True,
                    pretraining=dict(epochs=38, method="ASE", layers=8, width=512, heads=8,
                                     train_split_count=21254, val_split_count=2362))
    return dict(status="complete", checkpoint=evaluation["checkpoint"], checkpoint_step=selected["step"],
                selection=dict(source="adapt_d8/selected.json", split="val", criterion="maximum F with P >= 0.90"),
                policy=policy, parameters=params, model_parameters=evaluation["model_parameters"],
                training=training, test50=test, online=dict(scene_count=12,
                    scope="cache allocation, prefill, proposal sampling, proposal verification and residual sampling; model loading, warmup, independent numerical validation and output structure validation excluded",
                    groups=draft, target_only=direct,
                    feature_runs_below_30_details=[dict(name=r["name"], prompt_tokens=r["prompt_tokens"],
                        gamma=r["k"], acceptance_verified=r["feature_accepted"] / r["feature_verified"])
                        for r in online["rows"] if r["mode"] == "draft" and r["feature_verified"] > 0 and
                        r["feature_accepted"] / r["feature_verified"] < 0.30],
                    **_feature_runs([r for r in online["rows"] if r["mode"] == "draft"])), numerical_validation=numerical,
                timings=evaluation["timings"], limitations=list(LIMITATIONS))


def _percent(value):
    return "无样本" if value is None else f"{100 * value:.6f}%"


def _ci(value, percent=False):
    if not all(_number(v) for v in value):
        return "无有效区间"
    multiplier, suffix = (100, "%") if percent else (1, "")
    return f"[{value[0] * multiplier:.4f}, {value[1] * multiplier:.4f}]{suffix}"


def render_report(s):
    t, test, online = s["training"], s["test50"], s["online"]
    result = ["# 预训练 Draft 的 F 微调与独立接受率测量", "",
        f"固定 step {s['checkpoint_step']} 后，独立 test50 的 F Overlap 为 {_percent(test['feature']['overlap'])}，"
        f"P Overlap 为 {_percent(test['position']['overlap'])}。在线真实接受率见下方六组配置。", "",
        "先依据 val 中 P ≥ 90% 的候选点选择 F 最高的权重，再测独立 test50 和在线场景；test 未参与模型选优。"
        "Target 与 Draft 均采用 temperature=0.9、top_p=0.9，未使用温度调整。", "",
        f"最终权重：`{s['checkpoint']}`。Target 参数 {s['model_parameters']['target']:,}，"
        f"Draft 参数 {s['model_parameters']['draft']:,}。", "",
        "## 训练数据与四个测量点", "",
        "复用已有 38 epoch ASE 预训练的 8 层、宽度 512、8 个注意力头 Draft。原始数据划分为 21,254 train、2,362 val。",
        f"初始权重：`{t['initial_checkpoint']}`。本轮 {t['steps']} updates，每次 {t['trajectories_per_update']} 条完整轨迹，"
        f"累计 {t['trained_tokens']:,} target tokens，其中 F 为 {t['trained_feature_tokens']:,} tokens。"
        f"记录耗时 {t['elapsed_seconds']:.3f} 秒，包含训练和四次开发集评价。", "",
        f"微调 seed={t['seed']}，初始学习率 {t['lr']:g}，调度 `{t['schedule']}`，"
        f"训练精度 {t['train_precision']}。标签池包含 {t['label_trajectory_count']} 条轨迹。"
        "以下 F/P 均为固定 Target 轨迹前缀下的概率 Overlap；Feature 硬 NLL 为完整词表对已保存 F token 的加权交叉熵，单位 nats。", "",
        "| Updates | 训练 F | 验证 F | 验证 P | 验证 Feature 硬 NLL |",
        "| ---: | ---: | ---: | ---: | ---: |"]
    for r in t["points"]:
        result.append(f"| {r['step']} | {_percent(r['train_f'])} | {_percent(r['val_f'])} | "
                      f"{_percent(r['val_p'])} | {r['val_feature_hard_nll']:.6f} |")
    result += ["", "## 独立 test50", "",
        "场景重采样 bootstrap 的 95% 区间按轨迹计算，保留每条轨迹内部样本的关联。"
        "F/P 的点值按采样位置合并；val 仅用于选择权重。", "",
        "| 类型 | Overlap | 场景 bootstrap 95% CI | 采样位置数 | 轨迹 token 数 |",
        "| --- | ---: | --- | ---: | ---: |"]
    for kind, label in (("feature", "F"), ("position", "P")):
        r = test[kind]
        result.append(f"| {label} | {_percent(r['overlap'])} | {_ci(r['overlap_ci95'], True)} | "
                      f"{r['sample_count']} | {r['token_count']} |")
    result += ["", "## 在线真实接受率", "",
        f"online seed={s['parameters']['seed']}；12 个独立 test 场景，每次最多续写 512 tokens。"
        "前缀 0 表示仅 BOS 的空前缀，64 与 256 为已保存 Target 轨迹的前缀 token 数，均额外含 BOS。", "",
        "accepted/verified 表示已到达验证位置的提案接受率。首次拒绝后的后续提案不进入 verified。"
        "accepted/proposed 同时计入本轮未到达验证位置的提案；补充 token 与拒绝后的替代 token 不进入 accepted。"
        "EOS 单列。以下区间均为场景重采样 95% CI。", "",
        "| 前缀 | Gamma | 类型 | Accepted | Verified | Proposed | Accepted/verified | 95% CI | Accepted/proposed | 95% CI |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |"]
    for g in online["groups"]:
        for kind, label in (("feature", "F"), ("position", "P"), ("eos", "EOS")):
            r = g[kind]
            result.append(f"| {g['prompt_tokens']} | {g['gamma']} | {label} | {r['accepted']} | {r['verified']} | "
                f"{r['proposed']} | {_percent(r['accepted_over_verified'])} | {_ci(r['accepted_over_verified_ci95'], True)} | "
                f"{_percent(r['accepted_over_proposed'])} | {_ci(r['accepted_over_proposed_ci95'], True)} |")
    result += ["", "各配置中单次场景运行的 F accepted/verified 范围与低于 30% 的次数如下。"
        "仅纳入至少有一个 F verified 提案的运行；恰为 30% 不计入低于目标的次数。", "",
        "| 前缀 | Gamma | 单次 F 最小值 | 单次 F 最大值 | F 低于 30% 的运行数 / 有效运行数 |",
        "| ---: | ---: | ---: | ---: | ---: |"]
    for g in online["groups"]:
        result.append(f"| {g['prompt_tokens']} | {g['gamma']} | {_percent(g['per_run_feature_min'])} | "
                      f"{_percent(g['per_run_feature_max'])} | {g['feature_runs_below_30']} / {g['feature_runs_count']} |")
    result += ["", f"六组累计 {online['feature_runs_below_30']} / {online['feature_runs_count']} 次有效运行的 F 接受率低于 30%。"
        "累计次数包含同一场景在不同配置下的多次运行。"]
    for r in online["feature_runs_below_30_details"]:
        result += ["", f"`{r['name']}`，前缀 {r['prompt_tokens']}，gamma={r['gamma']}："
                   f"F accepted/verified={_percent(r['acceptance_verified'])}。"]
    speeds = [g["speedup_vs_target_only"] for g in online["groups"]]
    result += ["", "## FP32 吞吐", "",
        f"本轮各配置加速比为 {min(speeds):.5f}x–{max(speeds):.5f}x，未得到可靠加速。", "",
        "吞吐为总生成 token 数除以总解码时间，包含缓存分配、前缀处理、提案、验证及残差采样。"
        "计时排除模型载入、预热、独立数值验证与输出结构核对。"
        "加速比使用相同场景、相同前缀的 Target-only 对照，区间采用场景配对重采样。", "",
        "| 前缀 | 模式 | Gamma | Tokens | 秒 | Tokens/s | 95% CI | 加速比 | 95% CI |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |"]
    for g in sorted(online["groups"] + online["target_only"], key=lambda r: (r["prompt_tokens"], r["gamma"])):
        ratio = f"{g['speedup_vs_target_only']:.4f}" if g["mode"] == "draft" else "1.0000"
        interval = _ci(g["speedup_ci95"]) if g["mode"] == "draft" else "—"
        result.append(f"| {g['prompt_tokens']} | {g['mode']} | {g['gamma']} | {g['generated_tokens']} | "
                      f"{g['seconds']:.3f} | {g['tokens_per_second']:.3f} | {_ci(g['tokens_per_second_ci95'])} | {ratio} | {interval} |")
    n = s["numerical_validation"]
    result += ["", "## 数值验证与记录", "",
        f"toy_test 状态 `{n['toy_test']['status']}`。Target/Target 共 {n['target_target']['trial_count']} 次小试，"
        f"accepted/verified 范围 {_percent(n['target_target']['min_acceptance_verified'])} 至 "
        f"{_percent(n['target_target']['max_acceptance_verified'])}。EOS 终止小试共 {n['terminal_eos']['trial_count']} 次，"
        f"全部通过：{'是' if n['terminal_eos']['all_passed'] else '否'}。", ""]
    for name in ("target", "draft"):
        r = n["verify_blocks"][name]
        result.append(f"{name} verify_blocks：{r['dtype']}，最大总变差 {r['max_total_variation']:.8g}，"
                      f"容差 {r['tolerance']:g}，块宽 {r['widths']}，缓存回退覆盖：{'是' if r['rewind_tested'] else '否'}。")
    c = n["calibration"]
    if c["included"] and c["original_evaluation_comparison"]:
        r = c["original_evaluation_comparison"]
        result += ["", f"原 0.9/0.9 评价一致性：`{r['status']}`，比较 {r['compared_numeric_values']} 个数值，"
                   f"最大绝对差 {r['max_absolute_difference']:.8g}。该记录仅用于数值验证。"]
    else:
        result += ["", c.get("reason", "可选校准记录未提供原评价比较。")]
    result += ["", f"test bootstrap：{test['bootstrap']['resamples']} 次，seed={test['bootstrap']['seed']}；"
        f"online bootstrap：{s['parameters']['bootstrap_resamples']} 次。",
        "", "运行参量与已记录耗时保存在 `acceptance_summary.json`。", "", "## 解释范围", ""]
    result.extend(LIMITATIONS)
    return "\n".join(result) + "\n"


def make_figure(summary):
    """Return an optional Matplotlib figure without saving it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = summary["online"]["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    labels = [f"p{g['prompt_tokens']} / g{g['gamma']}" for g in groups]
    for ax, kind, title, color in zip(axes, ("feature", "position"), ("Feature", "Position"), ("#2374ab", "#198754")):
        for x, group in enumerate(groups):
            r = group[kind]
            if r["accepted_over_verified"] is not None:
                lo, hi = r["accepted_over_verified_ci95"]
                ax.vlines(x, 100 * lo, 100 * hi, color=color, linewidth=2)
                ax.plot(x, 100 * r["accepted_over_verified"], "o", color=color)
        ax.set(title=title, ylabel="Accepted / verified (%)", xticks=range(6), xticklabels=labels, ylim=(0, 100))
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    axes[0].axhline(30, color="#b45436", linestyle="--", label="Requested 30%")
    axes[0].legend()
    fig.suptitle("12 test scenes; scene bootstrap 95% CI; FP32; temperature=0.9, top-p=0.9")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("draft_a/runs/feature_acceptance_20261014"))
    args = parser.parse_args(argv)
    root = args.run_root
    evaluation = _read(root / "test_online.json")
    _require(evaluation.get("status") == "complete", "Evaluation status must be complete; no report written")
    adapt = root / "adapt_d8"
    history = _read(adapt / "history.json")
    metrics = {r["step"]: _read(adapt / f"step_{r['step']:04d}_metrics.json") for r in history}
    calibration = _read(root / "calibration.json") if (root / "calibration.json").exists() else None
    summary = build_summary(history, _read(adapt / "plan.json"), _read(adapt / "selected.json"),
                            evaluation, metrics, calibration)
    report = render_report(summary)
    summary_text = json.dumps(summary, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    figure = None
    try:
        figure = make_figure(summary)
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        plot_status = "Matplotlib 未提供，未生成可选图。"
    else:
        plot_status = "online_acceptance.png、online_acceptance.pdf"
    (root / "RESULTS.md").write_text(report, encoding="utf-8")
    (root / "acceptance_summary.json").write_text(summary_text, encoding="utf-8")
    if figure is not None:
        try:
            for extension in ("png", "pdf"):
                figure.savefig(root / f"online_acceptance.{extension}", dpi=180)
        finally:
            import matplotlib.pyplot as plt
            plt.close(figure)
    print(json.dumps(dict(status="complete", run_root=str(root), files=["RESULTS.md", "acceptance_summary.json"],
                          plot=plot_status), ensure_ascii=False))


if __name__ == "__main__":
    main()
