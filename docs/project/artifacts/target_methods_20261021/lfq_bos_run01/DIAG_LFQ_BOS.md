# LFQ BOS 配对分歧诊断

日期：2026-10-21。本文只读分析 `lfq_bos_run01` 的 18 条记录、本机适配器与远端原解码实现；没有运行 GPU，也没有改动测量源码。当前不能将 LFQ 标记为与原版语义一致。

## 逐 seed 首个输出差异

输入均为单个 BOS/EOS sentinel `12096`，budget 为 512。每个 seed 在原版与优化执行中都生成 512 token，未采到 EOS。下表的 token 位置从零开始；分歧前的 suffix 完全相同。

| Seed | 首个不同 token 位置 | 原版 → 优化 | Token 类型 |
| ---: | ---: | ---: | --- |
| 44101 | 147 | 9645 → 11885 | Feature |
| 44102 | 179 | 8688 → 8692 | Feature |
| 44103 | 391 | 11158 → 11444 | Feature |

LFQ 的 P/F/P 计数在三组配对中都不同，且变化方向不一致，见下表。计数差异可能在实际 token 分歧之前已开始：逐步 accept/reject 分支会调用不同数量的随机抽样；records 只保存最终聚合计数，没有每步决策日志。

| Seed | P1 接受 原/优 | F1 接受/进入判断 原/优 | P2 接受/进入判断 原/优 | K2 块数 原/优 | 单 token Target 调用 原/优 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 44101 | 201/204 vs 191/197 | 51/200 vs 58/190 | 47/51 vs 54/58 | 203 vs 196 | 9 vs 12 |
| 44102 | 179/188 vs 176/181 | 67/178 vs 74/175 | 60/67 vs 65/74 | 187 vs 180 | 18 vs 16 |
| 44103 | 168/172 vs 176/179 | 83/167 vs 76/175 | 81/83 vs 75/76 | 171 vs 178 | 8 vs 6 |

Target 控制是 3/3 输出和计数相同；PF-MLP 是 2/3 相同，seed 44101 首差在位置 61，token 为 `11733 → 11735`，也落在 Feature 范围。LFQ 的三次都分歧，因此现有证据更指向优化 Target 路径与草稿采样联动造成的数值敏感性，暂不支持只归因于 LFQ BOS 包装。该样本仅有一个开发来源和三个 seed。

## LFQ 原解码与适配输入

远端 `/root/workspace/GaussianGPT/bench_lfq_scratch_pf_cow.py` 中的 `CaptureLastBlock` 在 `target.transformer.h[-1]` 注册普通 forward hook，取最后 Transformer block 的输出并 `detach`。这个状态处于 Target final norm 和 `lm_head` 之前，形状为 `[1,T,H]`。`block2_decode` 用它作为 `h_prev` 输入 Feature Draft；checkpoint 路径与温度、top-p 配置在本轮 metadata 中记录。

BOS 启动流程在原函数内固定为两次 Target 前向：先输入 `[EOS]` 并请求 `['position']` logits，抽样 `P0`；再输入 `[P0]` 并请求 `['feature']` logits，抽样 `F0`，把 `[P0,F0]` 加入 suffix。三组配对的首两个生成 token 相同，初始 BOS/位置对齐没有直接分歧证据。

适配器只接受 `prompt == [EOS]`，用 `runtime.cache_for` 创建或重置 Target KV cache，再以 `_runtime_context` 包住原 `block2_decode`。适配器没有改温度、top-p、随机生成器或 P/F 验证循环。原 hook 仍挂在最后一个 block；图 replay 后 runtime 从该 hook 的绑定对象找到 `.output`，写入 `pre_final_norm_hidden`。这与 LFQ 原 hook 所需状态类型一致。

三次优化执行都确实走了图：seed 44101 是 T1/T2/T4 = 12/1/196，44102 是 16/1/180，44103 是 6/1/178；均无 fallback。首个输出差异都在许多 T4 replay 之后，且为 Feature token。自然 EOS 规范化也不是本轮差异来源：三组 LFQ 的 `raw_eos_count=0`、`raw_body_tokens=512`、规范输出长度为 512。

## 可能原因与所需证据

远端 Target `nanochat_gpt.py:1016-1098` 对 typed logits 逐行选择 Position/Feature 词表线性层；runtime 图在 `dynamic_target_graph.py:_trunk` 中执行相同的输入嵌入、Transformer block、final norm 和 typed 线性层。原 Target 调用 `flash_attn_with_kvcache` 接口；本轮 RTX 4090 的 FP32 路径使用 PyTorch SDPA math 计算，没有运行 Flash Attention kernel。图内 attention 使用显式矩阵乘、softmax 和矩阵乘。T1 Target 控制输出一致，LFQ 大多数验证调用为 T4；细小 FP32 数值差异可能改变 P/F 接受率、草稿 q 或残差抽样。

已有 `runtime_verify_run02` 在另一条 prefix 上观察到 T4 同 support，policy TV 最大约 `8.91e-7`，pre-finalnorm hidden 每行 L2 relative 最大约 `1.56e-6`。这说明两个执行路径可以非常接近，但它不覆盖 BOS 起始的这些实际 cache 状态，不能说明本次三条轨迹中的首个分歧具体发生在哪个候选。

原 LFQ 使用同一个显式 seed generator 进行 Draft `torch.multinomial`、接受判定、Target 抽样和残差抽样。若一次接受判定因微小概率差异变了，随后的抽样次数与 generator 状态也会改变；所以首个不同输出 token 不必是首个数值差异点。`lfq_bos_run01` 没有逐调用 logits、pre-finalnorm hidden、Feature Draft q 或每步接受分支，无法把三条分歧进一步定位为 Target 位置分布、F 分布、残差分布或 RNG 消耗顺序。

## 最小后续诊断

先用同一 BOS、seed 和 cache 容量跑一组仅诊断的原版与优化执行，逐 Target 调用保存 `pos`、输入 token、`logit_types`、mask 后 logits 和 pre-finalnorm hidden；在每个 K2 块另存 `h_prev`、Feature Draft logits/q、P1/F1/P2 的接受概率、抽样结果与循环前后的 generator 状态。沿调用序号找第一处不同，再判断变化来自概率还是抽样分支。

另加一组禁用 Target 图、仍走同一个 LFQ adapter 的控制，确认关闭图后是否恢复 3/3 seed 一致。若需要位级同 seed 输出，应让 LFQ Target 使用与原版一致的 attention 路径，或暂时为 LFQ 关闭图；这会舍弃该方法的图执行收益。若目标是同一采样分布，则以 LFQ 实际 T1/T2/T4 状态上的 Target policy 与 Feature q support、TV，以及输入 Feature Draft 的 pre-finalnorm hidden 误差逐调用作判断；在这些证据到齐前保持“未确认”。

## 证据位置

- 本轮原始数据：`records.jsonl`、`summary.json`、`plan.json`、`warmup.json`。
- LFQ 原 hook 与启动：远端 `/root/workspace/GaussianGPT/bench_lfq_scratch_pf_cow.py:35-55,153-182`。
- LFQ K2 提案、抽样和 Target 验证：远端 `/root/workspace/GaussianGPT/bench_lfq_scratch_pf_cow.py:268-303`。
- Target typed logits：远端 `/root/workspace/GaussianGPT/model/gpt/nanochat_gpt.py:1016-1098`。
- BOS adapter 与 runtime hidden 发布：本机 `patches/target_methods_20261021/method_adapters.py:490-515`、`patches/target_methods_20261021/runtime/dynamic_target_graph.py:590-647`。
- 一般形状数值证据：`../runtime_verify_run02/metadata.json`。
