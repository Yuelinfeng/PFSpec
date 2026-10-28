# Position-Conditioned Two-Step Transition: G0 implementation

状态：单模型接口与固定参数配置已完成。本轮只做本机标准库验证；没有读取训练权重、收集教师行、训练或调用 GPU。实际 Torch CPU 自测由主代理在现有环境执行。

## 模型接口

`TransitionModel` 位于 `patches/pf_position_transition/transition_model.py`，有 `direct` 和 `two_step` 两种方法。两者接收同样的五项输入：原 Feature MLP 已 softcap 的 4096 维 logits、末块最终 norm 前的 `h0`、已提交 F0 的全局 token ID、P0 与 Ranker 候选 P1 的局部 Position ID。学生只输出 `old_logits + delta`，不再次 softcap。

TwoStep 先用 `h0、F0、P0坐标` 更新 `uF`，再用 `uF、P1嵌入、P0/P1坐标及相对坐标` 更新 `uP`。低秩头以 `[uP,u0]` 生成 32 维潜码，再映射到 4096 个 Feature logits。Direct 用同样的 `h0、F0、P1、位置坐标` 一次产生残差。固定 token 表取原 Target embedding 的前 64 个通道，Feature 行为 8000..12095，Position 行为 0..7999；这些值不会参与梯度更新。位置 ID 按 `x*400+y*20+z` 解析，xyz 各除以19，并附加 P1−P0。

旧 Feature MLP 在 `TransitionAdapter.propose()` 中只调用一次；其输出已应用 Target softcap=15。新残差直接加到该输出，再由同一个 `.9/.9` policy 生成 q。P1 的教师概率、Target 的 Feature 分布、Target Hidden `hF/hP` 均不进入提案输入。

| 尺寸 | d | r | rz | Direct 隐层 | TwoStep 可训练参数 | Direct 可训练参数 | 差异 |
|---|---:|---:|---:|---:|---:|---:|---:|
| main | 128 | 64 | 32 | 169 | 181,408 | 181,466 | 0.032% |
| narrow | 64 | 32 | 16 | 68 | 80,464 | 80,376 | 0.109% |

固定 token 表及 `Rh` 不计入可训练参数。模型文件的 CPU 自测会逐一核对实际参数数目、有限输出、归一化 q 和零残差时 logits 逐值相同。

## 教师数据和训练

采集端确认输出 `teacher.pt` 为张量字典，包含 `h0/hF/hP`（`[N,1024]`）、`previous_feature_id`（全局 Feature ID）、`previous_position_id`、`candidate_position_id`、`old_logits_feature`、`target_logits_feature`、`p_f_policy`、`q_old`、`p_p1`、`p1_accepted`、`source_id`、`scene_id`、`row_id`、`split`。采集端另存候选 Feature ID、cache 起止位置、真实 prefix tokens 与 seed，供采集重放使用。`token_embeddings.pt` 是原 Target `wte.weight` 的 CPU Tensor，不包含 Target 权重。

训练入口只取 `split=train` 行拟合 `Rh`：将 `h0/hF/hP` 合并后作确定性协方差分解，按最大绝对分量统一每个方向的符号，并以训练状态的方差缩放。验证和开发来源不会参与这一步。训练主体按普通行均值优化 `KL(softmax(target_logits_feature/.9) || softmax(new_logits/.9))`；TwoStep 额外使用两个系数为0.05的 Smooth-L1 状态项和系数为1e-4的残差平方项。AdamW 固定 `lr=3e-4`、`weight_decay=0`、batch=64，每个方法/seed最多1200步，seed为17、29。

验证只在 `split=val` 进行，主选择分数是各来源普通 post-policy overlap 的等权均值；另存各来源按 `p_p1` 加权的 overlap。P1 接受和拒绝行都保留。`p_f_policy/q_old` 只用于读取时重算对齐，模型的 `forward` 不会接收它们。`target_logits_feature` 只作为密集 KL 标签。

`train_transition.py --run-g2` 顺序运行 Direct/TwoStep × seed17/29，并要求主 runner 提供一个绝对 `--deadline-unix`。同一截止时间覆盖四个组合，最多2400秒；每个组合不能刷新预算。接近截止时间时停止新更新，保存 `last.pt`、`progress.json` 与总 `pipeline.json`，并标明 partial。保存模型包含固定 embedding 表、训练所得 `Rh` 与网络参数，可直接恢复。

## 成本函数和本轮验证

`transition_adapter.cost_probe(old_call, new_call, cases, pairs=32, deadline_unix=..., synchronize=...)` 接收主 runner 的旧、新完整 `adapter.propose()` 包装函数与已准备上下文。默认先各跑两次预热，再按配对序号交替 old→new / new→old；每次对同一 `case` 测量完整提案路径，覆盖旧 MLP、新头、张量构造和唯一 policy。同步函数由主 runner 提供，以准确计入 CUDA 执行。Target 已加载对象由调用方共享。它返回逐配对 wall 秒、均值/中位数及 deadline 前完成数；partial 数据不会补成完整配对。

本机验证命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 patches/pf_position_transition/train_transition.py --show-parameters
PYTHONDONTWRITEBYTECODE=1 python3 patches/pf_position_transition/train_transition.py --self-test
python3 - <<'PY'
import ast
from pathlib import Path

for name in ("transition_model.py", "transition_adapter.py", "train_transition.py"):
    path = Path("patches/pf_position_transition") / name
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("AST parse passed")
PY
```

当前本机 Python 没有 PyTorch，`--self-test` 会明确报告 tensor 自测未运行，同时完成标准库的参数和 Position 映射自测。主代理可在现有环境用 `CUDA_VISIBLE_DEVICES=''` 执行同一 `--self-test`，之后再决定 G1/G2；本目录没有训练或测量结果。
