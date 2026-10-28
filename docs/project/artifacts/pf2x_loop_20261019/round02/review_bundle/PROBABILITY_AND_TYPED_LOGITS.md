# Round02：P/F 类型 logits 与状态语义

## 结论

按 Position 与 Feature 分组计算输出 logits，在实数代数下成立：每一行仍使用原来的 hidden、同一组 lm_head 权重行、同一 softcap 和同一词表 ID，只把互不依赖的行合并计算，再散回原行。必须保留完整 `(B,T,V)` FP32 返回值，并继续调用现有槽位与单调 Position 掩码。分组会改变 FP32 矩阵运算的归约次序，位级相同不能由代数推导保证，Top-p 截断与接受比较附近的分支仍需真实 Torch 与固定随机数流验证。

## 词表和等价关系

Round02 的 PF 常量是 `P=8000`、`F=4096`、`EOS=12096`。`nanochat_gpt.py` 的非共享词表布局规则（第 303–322 行）由此得到 Position ID `0..7999`、Feature ID `8000..12095`、EOS `12096`、PAD `12097`，总词表 12098；lm_head 无 bias（第 259 行）。PF 适配器也要求 `q` 是 4096 项 FP32 分布，且只取 `[8000:12096]`（`draft_a/pf_feature_unified.py` 第 416–423、443 行）。

设最终 norm 后的输入为 `X`，权重行矩阵为 `W`，softcap 为 `C(z)=15·tanh(z/15)`。对一行 `r` 和类型 `τ`，原路径计算 `C(X[r]·W[Iτ]ᵀ)`，其中 `I_P=[0..7999,12096]`、`I_F=[8000..12095]`。分组路径可对同类型行集合 `Rτ` 一次计算 `C(X[Rτ]·W[Iτ]ᵀ)`，再按原行号和绝对词表 ID 散回稠密 logits。每行只进入一个类型组，未选 ID 保持 `-∞`，所以精确算术下两者逐元素相等。

原实现已经规定了必须沿用的次序：创建全 `-∞` 输出，逐行取权重行，以 `F.linear` 计算后先做 softcap，再写回绝对 ID；所有行填完后才应用 `_apply_slot_logits_mask()` 和 KV 推理的 `_apply_monotonic_position_mask()`（`nanochat_gpt.py` 第 1056–1098 行）。不能先对整张稀疏 logits 做 softcap：`15·tanh(-∞/15)=-15` 会把非法词变成有限值。若任一类型是 `full`，现有路径会对整块执行全词表 lm_head（第 1058–1062 行）；分组快速路径应继续回到该路径。

## 行槽、尾窗和 EOS

实际类型由 `typed_kinds(cache.get_pos(), len(tokens))` 决定：偶数绝对输入行对应 Position 预测，奇数行对应 Feature 预测（`draft_a/pf_ranker_eval.py` 第 39–40 行；PF 调用见 `pf_feature_unified.py` 第 278–285 行）。不可假定每个窗口都从 Position 开始。每行输出预测下一个 token，因此从偶数 cache 位置开始的四行是 P/F/P/F，从奇数位置开始是 F/P/F/P。剩两个新 token 时，PF 走二行 Target 路径 `[pending_f,p1]`，取一项 P 验证和一项 F bonus；该路径也要按 cache 奇偶生成类型（`pf_feature_unified.py` 第 371–395 行）。

非共享词表槽位掩码对 Position 行保留 `0..7999` 和 EOS，屏蔽 Feature 区间；Feature 行只保留 `8000..12095`，屏蔽 Position、EOS 与 PAD。单调掩码只在 Position ID 范围内屏蔽 `rank(candidate) <= rank(previous)`，EOS 不受该限制（`nanochat_gpt.py` 第 663–700、715–790 行）。因此 Target 可以在 Position 槽结束序列，但 Feature 槽不能输出 EOS。`validate_sequence()` 同时要求 EOS 唯一且在末尾，并且必须紧随完整 P/F 对（`pf_ranker_eval.py` 第 15–26 行）。

`test_pf_typed_verify.py` 对 P/F 的每个词表列与绝对 ID 做双向映射，枚举起点奇偶与 1–4 行窗口，并单独覆盖两个二行尾窗。它还验证完整 P/F 对后 EOS 可用，Feature 槽排除 EOS，且 EOS 后不再有 token。

## Hidden、KV 与 RoPE

`CaptureLastBlock` 在最后 Transformer block 后、最终 norm 和 lm_head 前捕获 `detach()` 状态；它返回该状态及 `GPT.forward()` 的 logits（`bench_gaussiangpt_pf_block2_cow.py` 第 35–58 行）。PF 用已提交 P 的这份 hidden 提议后续 F；分组只在最终 norm 后的词表计算发生，不能改 hook 的位置、隐藏状态索引或返回类型。

Target 前向先用当前 cache 位置生成 RoPE，再写入本次 token；`_get_rope()` 从历史 `token_cache` 和当前输入重建稀疏位置（`nanochat_gpt.py` 第 587–648、1033–1037 行）。稀疏 Position RoPE 将 Position ID 转成 `(x,y,z)`，并附加槽位作为第四维；P 与配对 F 共享 P 的空间坐标，槽位维分别为 0、1（`rope_config.py` 第 27–44 行；`nanochat_gpt.py` 第 542–585 行；`utils/pos_tokens.py` 第 15–20 行）。拒绝后若实际提交的 P 发生变化，下一次 Target 前向必须用已提交 token 重新生成坐标，不能沿用被拒分支缓存的相位。

缓存语义与输出头分组无关，但必须保持原 keep 位置。四行窗口后 P1 拒绝保留 `before+1`；接受 P1、拒绝 F1 保留 `before+2`；P1/F1 接受、P2 拒绝保留 `before+3`（`pf_feature_unified.py` 第 425–479 行）。`truncate()` 将 `cache_seqlens` 设为 keep，并把对应尾部 `token_cache` 清成 `-1`；K/V 张量尾部不清零，后续只允许使用有效长度内内容（`bench_gaussiangpt_pf_block2_cow.py` 第 61–68 行；`engine.py` 第 47–96 行）。P 槽拒绝残差为 EOS 时，或 P2 残差为 EOS 时，都不继续追加 F；Target bonus F 只在对应 P 被接受且非 EOS 时读取。

## policy 与浮点边界

现有 `sampling_probs()` 先转 FP32、按温度缩放、排序并以累计概率执行 top-p，再 softmax 归一化；PF policy 调用温度和 top-p 均为 0.9（`draft_a/eval_overlap.py` 第 14–48 行；`feature_acceptance_eval.py` 第 72–75 行）。因此 logits 小幅变化仍可能跨过排序并列或 top-p 累计阈值；接受比较也可能因 `u < p[token]` 边界而改走另一分支。分组计算的正确性证据应同时包含有限 logits 误差、有效概率质量、固定外部均匀数下逐分支结果，以及多样本分布差异。仅有 `allclose` 或同 seed 的一条轨迹不足以覆盖阈值附近情况。

## CPU 结果和范围

命令：`python patches/pf2x_round01/test_pf_typed_verify.py --mode cpu`。标准库测试已运行并写入 `results/tests_cpu.json`：12,097 个词表列往返、16 个窗口配置、40 个行槽组合（含 2 个二行尾窗）、6 组隐藏宽度为 5 的矩阵计算、16 行归一化；float64 标量参考的最大绝对差为 0，归一化最大误差为 `1.11e-16`。还覆盖 `xyz/xzy` 排名排列、单调掩码和四种 EOS 序列情形。

本机 Python 3.12.15 没有 PyTorch，因此没有运行真实 `F.linear`、候选方法或实际 KV/RoPE。脚本提供 `--candidate PATH --repo-root ROOT`，在 PyTorch 可用时直接调用候选 `GPT._project_typed_logits_grouped(x, logit_types, softcap=15)`，与同一真实 Torch CPU `F.linear` 逐行计算对照，并调用候选模型的槽位/单调掩码。该入口尚未在本机执行。所有结果都是 CPU 代数验证，不涵盖 CUDA FP32 差异、真实权重、完整模型状态或性能收益。

## 核对依据

- 本轮最新源码副本：`docs/project/artifacts/pf2x_loop_20261019/round02/source_before/`，含 `nanochat_gpt.py`、`rope_config.py`、`rope.py`、`engine.py`、`utils/pos_tokens.py`、PF 解码、policy 与 `CaptureLastBlock` 实现。
- 原计划：`docs/project/artifacts/pf2x_loop_20261019/round02/WEB_PLAN_ORIGINAL.md`，第 1–3 节及 CPU 正确性要求。
- 本轮脚本和实测结果：`patches/pf2x_round01/test_pf_typed_verify.py`、`docs/project/artifacts/pf2x_loop_20261019/round02/results/tests_cpu.json`。
