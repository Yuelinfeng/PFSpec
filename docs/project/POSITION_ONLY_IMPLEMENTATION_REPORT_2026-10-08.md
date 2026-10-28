# D8 Position Only 与结构化 Position 查表：算法实现汇报

整理日期：2026-10-08。本文面向第一次接触项目的读者，说明两个方法如何产生候选、如何保持 Target 的采样分布，以及代码怎样完成一轮生成。

实现依据来自本轮通过 Computer Use 阅读的远端 JupyterLab 源码；训练配置同时参考本地早期源码与历史记录。本轮没有运行训练、GPU 性能测量或渲染。历史性能结果的完整背景见 [阶段工作汇报](STAGE_REPORT_2026-10-08.md)。

## 1. 两个方法的关系

两条路线都采用 **Position-only speculative decoding**：先用较便宜的方法提出一个位置 P，再由 Target 验证；特征 F 始终由 Target 采样。两者共用验证引擎，区别在于候选概率 q 的来源。

| 对比项 | D8 Position Only | 结构化 Position 查表 |
| --- | --- | --- |
| 候选来源 | 已训练的 8 层 Draft Transformer | 从已有 Position 序列统计得到的 gap 计数表 |
| 使用的历史 | 已提交的全部 P/F 前缀，通过 Draft KV 缓存读取 | 已提交的 P 历史，取最近至多 5 个间隔和最后一个 P 的低 1 位 |
| 每轮提出什么 | 一个 P，可能为 EOS | 一个 P，来自表计数形成的概率 |
| 谁验证 P | Target | Target |
| 谁生成 F | Target，在最终提交的 P 条件下采样 | Target，在最终提交的 P 条件下采样 |
| 提交一对 P/F 后 | Draft 双 token 前向，得到下一轮 P 概率 | 更新历史与 token 元数据，跳过 Draft Transformer |
| 主要固定成本 | Draft 前向 + Target 验证 | 查表 + Target 验证 |

D8 中的“8”指模型层数。Position Only 描述解码时的候选选择；D8 的训练仍学习完整 P/F 序列。结构化路线也属于 Position Only，不能把这两个名称理解为不同的 Target 生成任务。

现有 S9 历史矩阵中，结构化路线平均吞吐比为 1.36149×，D8 为 1.03719×。该数字来自已有记录，本轮没有重测，也没有据此宣布独立场景上的最终排名。

## 2. 本轮读取的实现依据

远端工程根目录为 `/root/workspace/GaussianGPT`。下表路径均相对该根目录；行号来自本轮页面显示的源码，后续编辑可能改变行号。这里只列函数定位，不把远端路径写成可在本机执行的文件链接。

| 远端文件 | 直接读取的内容 | 主要定位 |
| --- | --- | --- |
| `model/gpt/position_selective_speculative.py` | 全文：概率处理、接受/拒绝、Target 双 token 验证、缓存和 proposer 接入 | `sampling_probs` 218；`accept_or_correct_position_compact` 356；`truncate_kv_cache` 388；`generate` 534 |
| `model/gpt/draft_proposal.py` | 全文：状态、返回值、D8 与结构化封装；另有 cascade 实现 | `DraftProposalState` 8；`PositionProposal` 24；`NeuralDraftProposal` 72；`StructuredPositionProposal` 91 |
| `model/gpt/structured_position_adapter.py` | 全文：JSON 载入、上下文选择、计数转概率、三种查表路径 | `_select_counts_python` 160；`_query_python` 259；`query_batch` 307；`query` 372 |
| `scripts/structured_position_lookup.py` | 全文：训练/验证语料读取、计数、mode 评价、模型导出 | `build_tables` 27；`mode_for` 72；`export_model` 145；`main` 191 |
| `scripts/run_structured_position_e2e.py` | 全文：真实权重路径、构造引擎、选择路线、预热和输出 | 权重常量 14–17；`make_engine` 42；两条生成调用 114、128 |
| `scripts/profile_position_selective_sd.py` | 页面全文中的参数、加载和 Target AR 函数 | `parse_args` 52；`timed_forward`；`profile_target_ar`；`load_models` |
| `model/gpt/nanochat_gpt.py` | 页面全文中的类型限制、位置顺序、选择性输出与前向 | `_build_sparse_position_rank_map` 702；`_apply_monotonic_position_mask` 715；`forward` 1010 |
| `profiling/s8_structured_position/h5_b1/structured_zlow3_eval.json` | 原始评价全文 | h=5、bits=1、1325 项、32/4 条序列和 3480/4092 命中 |

Jupyter 文件列表显示：主引擎保存时间为 10-05 23:59，proposer 为 10-05 23:43，adapter 为 10-05 19:39。它们已包含本地早期参考中缺少的正式接口。本轮使用浏览器页面读取，没有保存远端源码副本，也没有编辑远端文件。

本地配置依据为 [D8 配置](../../.codex/remote-source/conf/model/gpt_draft_d8.yaml)、[S2 训练配置](../../.codex/remote-source/conf/training/gpt_draft_s2.yaml)、[Target 配置](../../.codex/remote-source/conf/model/gpt.yaml)。这些属于早期参考，不能代替权重内部配置的完整核验。

## 3. 输入、输出与 token 约定

场景首先被 VQ-VAE 编码为位置和离散特征交替的序列。P 指定一个位置，F 指定该位置的特征码；生成后可由 VQ-VAE 解码为 Gaussian 场景。

```text
BOS, P₁, F₁, P₂, F₂, …, EOS
```

本报告讨论的配置使用独立 P/F 词表：Position 有 8000 个取值，对应 20³ 网格；Feature 有 4096 个取值，整体 token id 偏移为 8000。

| 类型 | 当前配置的整体 token id | 可出现的位置 |
| --- | --- | --- |
| P | 0–7999 | 每个 latent 的第一个 token |
| F | 8000–12095 | 每个 latent 的第二个 token |
| EOS | 12096 | Position 预测位置；终止后没有 F |
| PAD | 12097 | 不参与生成采样 |
| BOS | 复用 EOS id 12096 | 生成输入的第一个 token |

引擎从模型属性构造两个紧凑候选集合：P 集合为 `0…7999` 加 EOS，长度 8001；F 集合为 `8000…12095`，长度 4096。P 集合最后一项的**局部索引是 8000，整体 token id 是 12096**。代码用 `position_token_ids[index]` 转换，接受判断中的 index 不能直接当整体 id 使用。

`generate` 的主要输入是 Target、Draft、可选 proposer、最大生成 token 数、temperature/top-k/top-p、seed。当前入口从 BOS 开始，没有独立 scene/prompt 输入；输出为不含 BOS 的 `(1,T)` token 张量与统计对象。`max_tokens` 包含输出中的 EOS，可能在一个 P 后达到上限。

实现仅支持 sparse、每个 latent 恰好一个 P 和一个 F、batch size 1、`stop_on_eos=True`。Target/Draft 的词表、偏移、EOS/PAD、P/F 配置、RoPE 和设备必须匹配。批量候选表接口的存在不代表生成引擎支持多条并行序列。

## 4. 共同的概率处理与接受规则

### 4.1 接受判断使用最终采样概率

神经模型先按预测位置限制 token 类型，并在启用位置约束时排除已经经过的位置。当前 `chunk_order=xyz` 的顺序值等于 Position id，所以后续普通 P 必须严格大于前一个 P；EOS 仍可被 Target 采样。

`sampling_probs` 将 logits 转为 float32，然后依次处理：

1. temperature 为 0 时，直接取 argmax 的点概率，忽略 top-k 和 top-p。
2. temperature 大于 0 时，将 logits 除以 temperature。
3. 有效 top-k 保留不低于第 k 大值的项，边界同值可能使保留数超过 k。
4. top-p 将概率按降序累计，保留越过阈值的第一项，且至少保留一项。
5. softmax 得到最终概率，并验证有限数值。

Target 的 p 与候选方法的 q 都要采用当前 P 位置的同一组采样参数。F 使用当前 F 位置的参数；`sampling_params_resolver` 可以按生成索引分别提供参数。接受计算不能混用原始 logits、未经过采样策略的表计数和最终概率。

### 4.2 一个候选怎样接受或拒绝

在已提交历史 H 下，候选方法输出 q，先采样 `y ~ q`。Target 给出同一 H 下的 p。

```text
接受概率 a(y) = min(1, p(y) / q(y))

接受：提交 y
拒绝：r(v) = max(p(v) - q(v), 0) / Σᵤ max(p(u) - q(u), 0)
      从 r 采样最终 P*
```

实际采出的候选必须满足 `q(y)>0`。接受概率为 1 或 0 时省去一次随机抽取；其余情况比较随机数与接受概率。拒绝后剩余概率总量小于等于 `1e-12` 时，代码使用 p 作为有限精度下的备用分布。

为什么最终 P 仍服从 p：接受分支贡献 `q(v)×a(v)=min(p(v),q(v))`；拒绝总概率为 `1-Σ min(p,q)=Σ max(p-q,0)`，乘以归一化 r 后贡献 `max(p(v)-q(v),0)`。两部分相加得到 p(v)。这是精确算术下的关系，实际计算使用浮点数。

F 在最终 P 条件下由 Target 生成，因此每轮保持 Target 的条件 P/F 分布。q 与 p 无须完全一致，q 也无须覆盖 p 的每个非零项；p 中未被 q 覆盖的质量由拒绝后采样提供。低质量 q 会增加拒绝成本。

这里的“保持分布”指同一 Target、类型/位置限制与采样参数下的概率语义。不同路线消耗随机数的次数不同，同 seed 的输出无需逐 token 相同。

### 4.3 接受率与 overlap

理论平均接受概率为 `αP=Σᵥ min(p(v),q(v))`。实际接受率为 `position_accepts / position_proposals`，来自有限次数随机运行。表的 mode top-1 命中率、分布 overlap 和实际接受率是三个不同指标。

## 5. 一轮生成的执行顺序

设已经提交的历史为 `P₁,F₁,…,Pᵢ₋₁,Fᵢ₋₁`。Target 的缓存只消费到 `Pᵢ₋₁`；最后的 `Fᵢ₋₁` 等待本轮一起输入。这样可以把预测 P 与预测 F 的两步合并成一次 Target 前向。

```text
已提交历史 → D8 或查表产生 q → 从 q 采样候选 P̂ᵢ
                                      │
                    Target 输入 [Fᵢ₋₁, P̂ᵢ]
                                      │
                 第 0 行：p(Pᵢ | 已提交历史)
                 第 1 行：p(Fᵢ | 已提交历史, P̂ᵢ)
                                      │
                       按 p/q 接受或拒绝
                      /              \
              接受 P̂ᵢ              拒绝，采样 P*ᵢ
                 │                      │
           复用第 1 行             移除候选缓存，输入 P*ᵢ
                 │                      │
                 └──── Target 采样 Fᵢ ──┘
                              │
                    提交最终 [Pᵢ,Fᵢ]
                              │
                 更新候选方法，准备下一轮
```

首轮 Target 缓存为空，输入 `[BOS,P̂₁]`。因果注意力保证第 0 行预测 P 时只能看到已提交历史，第 1 行才看到候选 P。接受候选时，第 1 行正好是所需的 F 分布；拒绝时，该行的条件已经不适用，需要重新取得 F 分布。

| 操作名 | 输入 | 输出含义 | D8 路线 | 查表路线 |
| --- | --- | --- | --- | --- |
| `draft_prefill_1` | `[BOS]` | 第一轮 qP | 运行一次 Draft | 只保存 BOS token 元数据并推进长度 |
| `target_verify_2` | `[上一轮F或BOS,候选P]` | 第 0 行 pP；第 1 行候选 P 条件下的 pF | 每轮运行 | 每轮运行 |
| `target_correction_1` | `[最终P]` | 最终 P 条件下的 pF | P 被拒且还需 F 时运行 | 同左 |
| `draft_update_2` | `[最终P,Target采出的F]` | 第 0 行 qF；第 1 行下一轮 qP | 每完成一对运行 | 只保存这对 token 元数据并推进长度 |

引擎的 `_forward` 传入 `logit_types`。Target 验证两行依次请求 Position、Feature 输出；Draft 更新依次请求 Feature、Position 输出。`nanochat_gpt.forward` 保留 Transformer 主体计算，仅对所需的 LM head 权重行做线性输出，然后应用类型和位置限制。紧凑采样进一步只处理对应词表集合。

## 6. D8 Position Only 的具体实现

### 6.1 模型来自哪里

D8 配置为宽度 512、8 层、8 个 attention head 和 8 个 KV head，上下文 16384。Target 参考配置为宽度 1024、24 层、16 个 attention/KV head。D8 保持相同 P/F 词表、位置语义、3D RoPE、`chunk_order=xyz` 和位置约束。

历史 S2 常规训练使用完整序列上的 next-token 交叉熵，配置为 38 epochs、`batch_scenes=4`、累积 4、bf16 mixed，优化器采用 Muon 与 AdamW 分组。训练入口和权重形成过程属于历史记录；本轮读取配置与运行脚本，没有重新读取全部训练日志。

实际 runner 引用的 D8 权重为：

```text
/root/workspace/GaussianGPT/profiling/s2/d8_canonical_38ep/logs/s2_d8_canonical_38ep/lightning_logs/version_0/checkpoints/val_loss_monitor_epoch_37_step_50502.ckpt
```

这份 canonical D8 与后续 S3/S12/S13 蒸馏权重应分别说明，不能把所有神经草稿都称为同一 D8 结果。

### 6.2 候选概率怎样取得

初始化时，Draft 消费 BOS，返回第一轮 P logits。每完成一对最终 P/F，Draft 消费 `[最终P,Target F]`，第二行输出下一轮 P logits。这样下一轮的 q 始终对应已经提交的真实历史。

正式 `NeuralDraftProposal.propose_position(state)` 读取 `state.q_position_logits`，取 P+EOS 集合，调用共享采样函数，再返回候选整体 id、局部 index 和最终 q。该 proposer 的 `requires_draft_forward=True`。其 `commit/rewind/reset` 不保存独立历史，实际历史由引擎的 Draft KV 缓存管理。

原引擎保留未传 proposer 的直接 D8 分支；该分支也从同一 Draft logits 构造 q，并执行同一接受函数。报告使用正式接口解释两条路线的共同接入方式。

### 6.3 为什么被拒后不需要重建 Draft 候选缓存

提出 P 时，D8 已经算出了 qP，候选 P 尚未输入 Draft。Target 验证结束、最终 P 与 F 确定后，Draft 才一次性消费这对真实 token。所以候选被拒时，需要回退的是 Target 本轮验证中消费的候选；D8 直接接收最终 P/F 即可。

Draft 更新的第一行 qF 仅用于诊断分布 overlap。在本报告的 P-only 路线中，F 已由 Target 采样，qF 不决定实际输出。`feature_shadow` 的旁路统计也没有提交 Draft F；当前 runner 显式设置其为 false。

## 7. 结构化 Position 查表的具体实现

### 7.1 表怎样从序列建立

脚本读取 `teacher_trajectory_corpus.pt` 中的 `train_sequences` 与 `val_sequences`，仅使用训练部分建表。根据独立词表范围筛出 Position id，然后计算相邻 Position 的整数差：

```text
位置：p₀, p₁, …, pₙ
间隔：gⱼ = pⱼ₊₁ - pⱼ
低位：bⱼ = pⱼ & ((1 << feature_bits) - 1)
键值：(最近至多 h 个已经出现的间隔, bⱼ)
预测对象：下一个间隔 gⱼ
```

对每个训练 transition，`build_tables` 同时累加全局 gap Counter、按低位分组的 Counter，以及历史长度从 0 到 `min(h,j)` 的全部键。每个表项保留“下一个 gap → 出现次数”，无需神经网络训练或反向传播。

代码中的 `feature` 是**Position id 的低位特征**，没有使用 F token。字段名写为 `morton_low_bits`，实际表达式直接对 Position id 做按位与；此处没有调用 Morton 编码函数或重新交错 xyz 位。对外复现时应按表达式实现，不能仅按名称猜测坐标处理。

原始评价文件直接确认：训练 32 条、验证 4 条，训练 transition 32736、验证 transition 4092，Position 词表 8000，h=5、bits=1，1325 个表项。表来源为历史 Target 采样轨迹；32/4 条轨迹不能自动等同于 32/4 个独立真实场景。

### 7.2 运行时怎样选表

`StructuredPositionGapDraft` 载入 JSON 中的 `history_len`、低位参数、`tables`、`feature_counts` 和 `global_counts`。给定已提交的 P 历史，它重新计算 gap 列表，取最后一个 P 为 last，再按以下顺序查找非空计数：

```text
最近 5 个 gap + last低位
→ 最近 4 个 gap + last低位
→ … → 空历史 + last低位
→ feature_counts[last低位]
→ global_counts
```

实际历史不足 5 个 gap 时从可用最长历史开始。训练脚本已经建立 h=0 表，正常导出中 feature 级统计与 h=0 表往往重合；后两个层级也兼容缺少某些表项的输入文件。各层计数没有加权混合，命中后就使用该行。

例如已提交位置 `[10,13,14,15,18,19]`，最近 5 个 gap 为 `(3,1,1,3,1)`，bits=1 时 last=19 的低位为 1，首选键就是 `((3,1,1,3,1),1)`。

### 7.3 gap 计数怎样变成 P 概率

对选中行的每个 `(gap,count)`，adapter 计算 `token=last+gap`，保留 `token>last`、`0≤token<8000` 且属于 P 集合的项，将 count 加到对应局部位置，最后归一化。

```text
w(v) = 选中表行中满足 last+gap=v 的有效 count 之和
q_raw(v) = w(v) / Σᵤ w(u)
```

如果行计数为 `gap1:40, gap3:10` 且 last=19，原始 q 在 P=20、P=22 上分别为 0.8、0.2。last=7998 时，P=8001 超出词表范围被舍去，P=7999 归一化后为 1。

选中行的候选全部无效时，代码直接尝试 last+1；该值不在集合时尝试 EOS；两者都不可用时才对整个集合赋均匀权重。此时不会返回上一层继续找另一行。文档或复现代码都应保留这个顺序。

### 7.4 第一位置与 EOS 的特殊处理

P 历史为空时，adapter 使用虚拟前驱 last=-1、gap 历史为空、低位=0。gap=1 因而可提出 P=0。它复用了一般 transition 表，没有为第一位置使用专门分布。

训练脚本统计了首个 P 的次数，原始评价显示 P=0 出现 31 次、P=3 出现 1 次；`export_model` 没有把这组首位置计数写入模型 JSON，adapter 也没有读取它。这是当前实现细节，不应描述成“首位置按训练首 token 频率采样”。

普通 gap 行仅累积 P id，不直接计入 EOS。原始 q 的 EOS 项通常为 0，候选用尽时可以通过备用路径提出 EOS。Target 的 p 保留 EOS；即使 q(EOS)=0，拒绝后的分布仍能采出 EOS，并立即结束生成。

### 7.5 实际使用的是重新处理后的 q

正式 `StructuredPositionProposal` 没有直接将 q_raw 作为接受函数输入。它先计算：

```text
logits(v) = log(max(q_raw(v), float32最小正正常数))
q = sampling_probs(logits, 当前temperature, 当前top_k, 当前top_p)
```

因此 temperature 大于 0 时，相当于先按 `max(q_raw,tiny)^(1/temperature)` 重新加权，再执行采样过滤。zero 项经过数值下限处理后会有极小的权重，通常被当前 top-p 排除；不能声称封装后的 q 在所有参数下与原始计数概率完全相同。

返回的 `PositionProposal.probs` 是实际采样使用的 q，Target 接受函数也用这份 q。mode top-1 评价只选出现次数最多的 gap，同次数选择较小 gap，不能拿 mode 代替实际随机候选概率。

### 7.6 Python、tensor、vectorized 路径

| 模式 | 实际实现 | 当前单序列调用的含义 |
| --- | --- | --- |
| `python` | tuple 键查字典，Python 遍历 gap/count，对设备概率张量逐项累加 | runner 默认模式 |
| `tensor` | 将历史编码为整数键，gap/count 张量按设备缓存，`index_add_` 累加 | 仍在 Python 中逐个选择上下文 |
| `vectorized` | `query_batch` 先选择各行上下文，再批量映射位置、`scatter_add_` | `query` 实际调用 `query_batch([positions])`，批量大小为 1 |

三种模式都没有训练新模型。当前 adapter 每次由完整 P 历史重新计算所有 gap，引擎也每轮从 output 筛取 P；h=5 限制的是查表键长度，未使整个查询流程成为恒定成本。概率数组仍包含 8001 个位置，部分 `.item()` 和 Python/设备操作也有成本。

## 8. 正式接口与状态更新

`DraftProposalState` 包含已提交的 `position_history`、最后一个 F、可选 D8 P logits、P 集合、EOS、采样参数、随机数生成器、设备和生成索引。D8 使用 P logits；结构化封装使用 P 历史，当前不使用传入的 F。

`PositionProposal` 返回整体 `token_id`、局部 `index`、最终 `probs`、来源、可选 confidence 与元数据。`confidence` 为采中项的概率，当前两条单一路线不会据此跳过 Target 验证。

```text
新一轮 generate → proposer.reset()
每轮 → propose_position(已提交状态)
P 接受 → commit(最终P)
P 拒绝 → rewind(提议前状态) → commit(最终P)
```

结构化 proposer 保存独立 P 历史。每次提议前先与引擎传入的已提交历史同步；提交非 EOS P 时追加位置，提交 EOS 时不追加；拒绝后先恢复历史，再追加最终 P。未经 Target 确认的候选不会留在历史中。

`StructuredPositionProposal.requires_draft_forward=False` 使引擎跳过 Draft 初始化和更新前向。但当前构造器仍要求兼容的 Draft 模型，runner 仍加载 D8，`make_kv_cache` 仍分配 Draft KV 空间，生成中仍维护长度与 token 元数据；结构化诊断还构造零 F logits。因此“没有 Draft 前向”不能写成“没有 Draft 权重或 KV 显存”。

## 9. 缓存回退和停止位置

### 9.1 完整 P/F 提交后的长度

完成 i 对 P/F 后，输出不含 BOS，长度为 2i。

| 状态 | 实际包含的前缀 | 逻辑长度 |
| --- | --- | ---: |
| 输出 output | `P₁,F₁,…,Pᵢ,Fᵢ` | 2i |
| Target KV | `BOS,P₁,F₁,…,Pᵢ`；尾 F 等待下一轮 | 2i |
| D8 KV | `BOS,P₁,F₁,…,Pᵢ,Fᵢ` | 2i+1 |
| 查表路线的 Draft token 元数据 | 同上；没有通过前向生成可供神经读取的 K/V 内容 | 2i+1 |

引擎每完成一对验证 Target 长度等于 `len(output)`，Draft 长度等于 `1+len(output)`。两种长度差一的原因是输入调度，而非漏提交最后一个 F。

### 9.2 P 被拒时 Target 怎样恢复

本轮前记 `target_start=Target缓存长度`。验证输入两项后，Target 已消费上一轮 F 和候选 P。拒绝且还需要 F 时，将长度退到 `target_start+1`，保留已经正确消费的 F/BOS，舍去候选 P。

`truncate_kv_cache` 同时把对应范围的 token 元数据设为 -1。K/V 存储不清零，缩短逻辑长度后旧内容不再参与注意力，下一次前向会覆盖；token 元数据要清掉，避免位置限制读取旧候选。随后消费最终 P，得到条件正确的 F logits。

### 9.3 生成在哪些位置停止

最终 P 为 EOS 时立即停止，无 F。若最大 token 数在 P 后耗尽，也立即停止，无需补成完整一对。两种终止分支都位于后续缓存回退和 F 计算之前。

因此完整 pair 的长度关系不适用于所有终止瞬间；尤其拒绝后最终 P 为 EOS 时，局部验证缓存可仍含候选 P。当前 API 只返回 tokens 与 stats，不返回这些缓存继续生成，它们随本次调用结束而弃用。未来若增加缓存续写功能，需要另行处理终止瞬间的状态。

## 10. 可以照着理解的共同伪代码

下面描述实际控制顺序，省略计时和 Feature 诊断。它是算法说明，不是新增可执行接口。

```text
创建 Target/Draft 缓存；output = []；BOS = EOS id
重置 proposer
若 D8 路线：Draft 消费 BOS，得到 qP logits
若查表路线：仅保存 BOS token 元数据并推进长度

while 输出长度 < max_tokens:
    读取当前位置的采样参数
    D8 从 qP logits 构造 q，或查表从已提交 P 历史构造 q
    从 q 采样候选 P̂，保存其局部 index
    记住 Target 本轮前长度 start
    Target 消费 [上一轮F或BOS, P̂]，得到 pP 与 pF_given_candidate
    对 pP 应用采样参数，得到 p
    用 min(1,p/q) 接受候选；拒绝时从归一化 max(p-q,0) 采最终 P
    根据接受结果更新 proposer 历史，提交最终 P 到 output
    若最终 P 为 EOS 或输出达到上限：停止

    若接受：复用 pF_given_candidate
    若拒绝：Target 长度退至 start+1，移除旧 token 元数据
            Target 消费最终 P，重新得到 pF
    用当前位置 F 参数处理 pF，从 Target 概率采 F，提交到 output

    若 D8 路线：Draft 消费 [最终P,F]，保留下一轮 qP logits
    若查表路线：仅保存 [最终P,F] token 元数据并推进长度
    验证完整 pair 的两条缓存长度关系

返回不含 BOS 的 tokens 与 stats
```

## 11. 速度收益来自哪里

纯 Target 自回归通常需要两次单 token 前向完成一对 P/F。P-only 将它们合成一次双 token 验证；候选被拒且还需 F 时再增加一次单 token 前向。

```text
D8 每对成本 ≈ Draft双token更新 + Target双token验证
             + (1-αP)×Target拒绝后单token重算 + 采样/调度

查表每对成本 ≈ 表查询 + Target双token验证
               + (1-αP)×Target拒绝后单token重算 + 采样/调度
```

历史分项中，D8 更新约 6.5–6.7 ms/对。结构化表接受率可以略低于 D8，同时因为省去这项固定前向而更快。低成本 q 的价值在于接受收益与候选成本共同合算。

当前共用引擎每轮只提出一个 P，历史离线 K=4 mode 连续命中评价没有变成一次提出四个 P 的生成算法。Target 批量块的单位 token 计算收益，也不能直接当完整路线的吞吐倍率。

## 12. 实际运行入口与复现边界

### 12.1 表构建命令

以下命令对应已读取的参数与 h5_b1 配置。本轮仅记录，未执行；执行会重新生成 JSON。

```bash
cd /root/workspace/GaussianGPT
python scripts/structured_position_lookup.py \
  --corpus profiling/s3/s3_onpolicy_feature_distill_32x2048_256step/teacher_trajectory_corpus.pt \
  --output-dir profiling/s8_structured_position/h5_b1 \
  --vocab-size 8000 --history-len 5 --feature-bits 1
```

导出的 `structured_zlow3_model.json` 包含计数表，文件名沿用早期命名；配置以 JSON 中 h=5、bits=1 为准，不能从 `zlow3` 推断当前使用 3 bit。

### 12.2 同一入口选择两个 proposer

runner 用 `STRUCTURED_PROPOSAL_MODE=structured` 选择查表，用 `d8` 选择 `NeuralDraftProposal`。两者都使用同一个 Target、D8 和 VQ-VAE 加载入口。以下示例在恢复实验后可用于理解调用方式，本轮未运行：

```bash
cd /root/workspace/GaussianGPT
STRUCTURED_PROPOSAL_MODE=structured STRUCTURED_LOOKUP_MODE=python \
STRUCTURED_MAX_TOKENS=256 STRUCTURED_WARMUP_TOKENS=32 STRUCTURED_SEED=1 \
STRUCTURED_OUTPUT=profiling/report_reproduction/structured.json \
python -m scripts.run_structured_position_e2e

STRUCTURED_PROPOSAL_MODE=d8 \
STRUCTURED_MAX_TOKENS=256 STRUCTURED_WARMUP_TOKENS=32 STRUCTURED_SEED=1 \
STRUCTURED_OUTPUT=profiling/report_reproduction/d8.json \
python -m scripts.run_structured_position_e2e
```

Target 为 `checkpoints/gpt_vfront.ckpt`，VQ-VAE 为 `checkpoints/vqvae_vfront.ckpt`，D8 路径见第 6 节，表路径为 `profiling/s8_structured_position/h5_b1/structured_zlow3_model.json`。temperature=0.9、top-p=0.9、top-k=None 来自 profiler 参数默认值；runner 重写 `sys.argv`，不会直接接受任意额外命令行参数。

### 12.3 当前入口的计时限制

runner 先分别预热两条路线。正式纯 Target 使用 seed+10，候选路线使用 seed+11。模型加载和 `make_engine` 位于主计时区间外；`generate` 内部分配缓存，device 表副本首次建立也可发生在计时区间内。生成吞吐按实际输出长度除以 wall-clock 时间计算，可能受提前 EOS 影响。

本轮读取到的 runner 将 `p.sync` 设为 CUDA 同步函数；其纯 Target 路径调用 `profile_target_ar → timed_forward`，每次前向前后同步。候选引擎使用 `measure_latency=False`，没有同样的逐次同步。纯 Target 前向还使用完整输出头，候选引擎使用选择性输出头。这个入口可说明算法接入与当时的运行方式，**不能将它视为已统一所有同步和输出计算条件的最终公平计时入口**。

S9 历史矩阵属于另一批已有结果。本轮没有重读其全部 runner 与原始 JSON，也没有重新测量同步差异的影响。对外引用性能应注明来源和条件，不能把本文的命令直接等同于 S9 完整矩阵复现。

## 13. 已有证据、本轮验证与实际限制

### 13.1 本轮直接确认

源码与函数调用支持本文所述的 P-only 路径、最终概率接受、Target 条件 F、缓存回退、D8 更新和查表免 Draft 前向。表评价原始 JSON 与历史记录一致：3480/4092=0.85043988，表项 1325，h=5、bits=1，CPU mode 查询均值 0.71859684 μs。

CPU mode 查询只返回一个 gap，adapter 查询还要处理历史、合法位置、设备张量及概率采样。该微测量时间不能作为引擎每轮的查表成本。

### 13.2 历史结果的引用范围

| 证据 | 已记录的结果 | 可说明的内容 |
| --- | --- | --- |
| 结构化正式接口，seed1、256 token | 115/128 个 P 接受，13 个拒绝；输出 256；完整 pair 缓存长度验证通过 | 该短运行的算法与接口可用 |
| S9 structured，12 个 seed/长度单元 | 平均 1.36149×；全部单元大于 1 | 已有随机轨迹设置下的正吞吐收益 |
| S9 D8，12 个单元 | 平均 1.03719×；最低 0.96349× | D8 固定候选成本限制收益 |
| synthetic 接受/拒绝历史自测 | 理论 overlap=0.45，经验接受=0.44762，输出 L1=0.00340 | 已记录的小分布采样验证 |

这些运行结果来自 [实验记录](EXPERIMENTS.md) 与 [阶段工作汇报](STAGE_REPORT_2026-10-08.md)，本轮没有重跑。S9 从 EOS/BOS 起始，覆盖 seed0/1/2 与 256/1024/2048/4096 token，没有独立场景输入；不应写成多个真实场景的泛化证明。

### 13.3 仍需保留的实现限制

当前表按 Position id 差建模，adapter 用 id 严格递增筛候选；已使用的 xyz 配置与此一致。若改成 xzy 或其他顺序，必须重新核对 Target 的顺序值与表的表示。已有 Target 顺序代码支持 xzy，不等于查表适配器已覆盖相同语义。

查表路线的 F 诊断来自引擎构造的零 logits，再按 temperature/top-k/top-p 处理；这是人工占位输入，不能解读为表模型实际预测 F 的质量。两条路线都没有直接提议并提交 F。

本轮没有重新载入权重内容、复算神经 logits、验证全部场景的质量，或审阅 S14 文件内容。远端文件列表已出现 S14 评价/overlap 目录，因此旧文档中“S14 尚未实施”的绝对表述不再作为当前远端事实；目录存在本身也不足以判断结果是否完整。本报告范围仍为这两个 P-only 方法。

## 14. 文档交付与验收范围

本轮新增本文，并在 README、AI 入口、架构说明和阶段工作汇报中添加阅读入口及证据范围提示。正文已按实际源码核对概率、控制顺序、缓存长度与特殊分支；文档的本地链接、表格列数、代码块配对和用词规则另行验证。

交付验证：5 份新增或更新文档中的 48 个本地链接、37 个表格的列结构、代码块配对和面向读者的用词均通过；计数比例、gap/低位示例及接受/拒绝概率恒等式的确定性计算通过。该验证不包含新增神经推理或随机采样实验。

没有修改远端源码、权重、表模型或 GPU 作业；没有新增实验结果。当前成果是可以独立阅读、并按函数定位回到实现的汇报文档。
