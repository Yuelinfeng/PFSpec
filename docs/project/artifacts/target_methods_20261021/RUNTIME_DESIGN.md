# Target CUDA Graph runtime

此目录提供给 EAGLE、Medusa、Hydra 及其他 Target 解码器做隔离比较。实现只在本次补丁目录内运行；已有 PF runtime 保持原样。

## 接口

主模块为 `patches/target_methods_20261021/runtime/dynamic_target_graph.py`。传入 Nanochat GPT core 实例创建一个 runtime：

```python
from dynamic_target_graph import DynamicTargetGraph

runtime = DynamicTargetGraph(core, max_query_tokens=8)
```

`attention_context(enabled=True, hidden_capture=...)` 临时包装该 core 的 `forward`。退出时恢复原方法。Target KV 工厂需由 decoder adapter 路由到 `runtime.cache_for(core, capacity, factory=original_factory)`；草稿模型的 KV 工厂保持原样。工厂以 core 身份区分模型，其他模型会继续使用传入的原工厂。

EAGLE v2 的 `speculative.pf_engine.target_cache` 是 Target cache 工厂，可在候选运行时临时包装并在 `finally` 中恢复。Medusa/Hydra 若将 Target cache 工厂作为参数传给 decode，则传入相同的包装函数；若工厂藏在其他模块全局变量中，由 runner 保存原引用、候选运行时替换、退出时恢复。

## 图内请求与输出

默认支持单 batch、CUDA inference 的 T1 到 T8，图按 cache 实例、T、read bucket 和 logit 类型建立。可将 `max_query_tokens` 设为 1 到 16；超过设置的 query、长 prefill、训练请求、非 CUDA 输入或不匹配的 cache 会走原 Target forward。

`logit_types` 支持 `position`、`feature`、`full`；省略该参数或类型列表中出现 `full` 时，按 Target 原逻辑对整个 query 生成完整词表 logits。槽位 mask 与 monotonic Position mask 在图外照原顺序执行。RoPE 使用调用时的 host position 更新，图使用固定缓冲区；future KV 位置由 causal mask 屏蔽。cache 的 token 写入、前进与 truncate 仍使用原对象。

`return_hidden=True` 返回 `(logits, post-finalnorm hidden)`，与 GPT core 的原返回值一致。末层 hook capture 收到图内的 pre-finalnorm hidden，供 PF/Medusa 类 draft 使用。若 adapter 自己持有 `value` 字段的 capture 对象，可传一个 callable：该函数接收 pre-finalnorm Tensor，再按该 decoder 原有方式转换并写入其字段。不要把 post-finalnorm 返回值用作这个 hook 输入。

`stats()` 包含按 T 的成功图调用次数、全量词表读出与 `return_hidden` 调用次数、实际图形状、cache 建立/重置耗时、图准备/预热/捕获耗时及显存增量。runner 应在 warmup 和正式配对期间保留同一个 runtime，并在结果中单列首次成本。

## 真实 Target 验证

`patches/target_methods_20261021/runtime/verify_runtime.py` 用真实 Target、crop token 和原 cache 工厂对照 eager SDPA。它先以原 forward 处理长 prefix，然后分别运行 T1/T2/T3/T4/T6 typed 请求、typed `return_hidden`、full-logits `return_hidden`，再 truncate 并换一个 Feature 重跑 T2。脚本要求实际发生图回放，确认跨过 512/1024 read bucket；输出原始输入、logits、pre-finalnorm 与 post-finalnorm Tensor，比较 policy TV、support 和逐行 hidden L2 相对误差。

```bash
python patches/target_methods_20261021/runtime/verify_runtime.py \
  --project /root/workspace/GaussianGPT \
  --crops /path/to/crop_records.pt \
  --out-dir /root/autodl-tmp/target_methods_runtime_run01 \
  --prefix-body 510
```

`--out-dir` 必须是新目录。成功状态要求每种指定 query 长度均建立图、full logits 和 post-finalnorm 返回态确实经过图、hidden capture 写入 pre-finalnorm 状态，并且 truncate 后 alternate T2 的两臂数值相符。当前本机只完成 Python 语法和 `--help` 验证；真实 GPU 结果由主代理统一运行记录。

## 真实运行结果

记录位于 `docs/project/artifacts/target_methods_20261021/runtime_verify_run01/` 和 `runtime_verify_run02/`，每轮都保留 `metadata.json` 与 `raw_outputs.pt`。run01 虽报告通过，但它把 Position 输入 `3867` 替换成了 Feature ID `12060`，原因是用输出 logits 的类型判断输入 token 类型；因此该轮的 truncate alternate 分支不代表合法换 Feature。其余调用展示了 T1/T2/T3/T4/T6、full logits 与两种 hidden 输出，最大 policy TV 为 `4.1781e-6`，最大逐行 hidden L2 相对误差为 `2.3346e-6`。

run02 改为按输入 token 范围识别 Feature，只改动 `[3867, 8435]` 中的 Feature，得到 `[3867, 8436]`，输入槽位仍为 Position/Feature。长 prefix 有 1 次原 forward；随后 7 次图回放覆盖 T1/T2/T3/T4/T6、full logits 与 `return_hidden`。truncate 将 cache 从位置 528 退回 511 后，alternate T2 仍复用图且数值比较通过。最大 policy TV 为 `1.3309e-6`，pre/post-finalnorm hidden 最大逐行 L2 相对误差为 `1.5625e-6`。6 张图覆盖 read bucket 512 与 1024；cache 容量为 1024，KV 占用 `201,326,592` bytes。该轮图捕获耗时合计约 `1.011s`，这些是正确性探针的准备开销，完整解码测时仍需将首次成本计入。

## 使用边界

默认形状覆盖 EAGLE gamma=3/5（最大 query T4/T6）及已读到的 Medusa linear/aligned T1–T4。query 超过配置值、batch 大于 1、未配对的输出模式、训练前向和超出 cache/read bucket 的位置会直接使用原 forward。完整 decoder 的收益需分别记录各自 runtime 的 `graph_forward_T*`、`graph_full_forward`、`graph_return_hidden` 与 fallback 计数；单独通过本脚本不代表某个 draft 方法已验证。
