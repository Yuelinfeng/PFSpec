# PF Target 固定位置图测试

## 当前判断

下一步应测完整 Target transformer trunk 的 CUDA Graph，并保留显式 QK、softmax、V 注意力。单层 RTX 4090 结果显示，图内显式注意力在长上下文差距很大：T=1、ctx=2048 从 252.35 µs 降到 13.19 µs；T=4、ctx=2048 从 274.48 µs 降到 30.26 µs。短上下文也有收益，例如 T=4、ctx=64 从 22.61 µs 降到 9.60 µs。完整表格与逐次结果见 [COST_DESIGN.md](COST_DESIGN.md) 和 `attention_graph_run01/`。

此前去除逐层读取 GPU cache position 的单独实验约为 1.03–1.06×，typed verify 的额外收益不到 0.5%。因此本脚本同时比较 attention 与图重放；目前还没有完整 Target 或 PF 达到 2× 的证据。

## 实验内容

[pf_target_graph_probe.py](../../../../patches/pf2x_breakthrough_20261021/runtime/pf_target_graph_probe.py) 从实际权重加载 `teacher.gpt.transformer`，使用真实 KVCache、FP32 与生产 sparse typed logits。它对 ctx=64/256/768/2048、T=1/2/4 比较四种执行方式：

| 名称 | Position 位置 | 注意力 | 执行方式 |
|---|---|---|---|
| `sdpa_gpu_pos` | CUDA 标量 | 原 SDPA | 原 Target 前向 |
| `sdpa_host_pos` | Python 已知位置 | 原 SDPA | 原 Target 前向 |
| `explicit_host_pos` | Python 已知位置 | 显式 QK、softmax、V | 原 Target 前向 |
| `graph_explicit_host_pos` | Python 已知位置 | 显式 QK、softmax、V | 完整 GPT core trunk 图 |

每种形状都先比对输出，再开始计时。图里直接捕获 GPT core 的 embedding、逐层 Block、最终 norm 和 typed linear；图返回最后 Block 的 pre-final-norm hidden，供 PF 的 Feature MLP 使用。RoPE 输入准备、slot logits mask、monotonic position mask 在图外运行，避免捕获 GPT 外层中依赖 Tensor 布尔值的分支。host-position 两臂保留 cache 内原有 CUDA `cache_seqlens`，仅将 SDPA 调用收到的位置参数换成 CPU 镜像；模型读取位置时使用已知 Python 整数。显式注意力使用 prefix 加 causal mask，跟随本地 SDPA KV-cache fallback 的处理方式；不会额外依据 window_size 裁剪 KV。metadata 会保留 Target 的逐层 window_sizes 和 cache_seqlens 所在设备。生产源码没有改动。

数值要求为 logits/hidden 的 `atol=1e-4, rtol=1e-4`，policy 的最大 TV 距离不超过 `5e-5`。若任一形状未达到要求，脚本保存当前输出并停止，不继续给出速度结果。成功形状的 `.pt` 包含四臂 logits、最后 Block hidden、输入 token、cache position 与 typed 行种类；若在后续步骤失败，文件至少保留 SDPA 参照与 CPU 位置臂输出，便于独立重算。

每臂预热 3 次，每个形状做 10 组成对计时，每组每臂运行 3 次。臂顺序逐组轮换。`pairs.jsonl` 保留每次 host wall、CUDA Event、两者差值、顺序和输入信息；`metadata.json` 另记模型加载、cache 分配、前缀处理、mask 准备、RoPE 准备、图预热和图捕获时间。速度比按配对 `reference_ms_per_call / candidate_ms_per_call` 计算，并给出中位数与均值。

## 运行方式

从本机项目目录把脚本经标准输入送到远端 Python。实验输出放在独立临时目录，不写生产 checkout。`--out-dir` 必须使用未存在的新路径；脚本遇到已有路径会直接报错。

```bash
cd /Users/lief/Documents/ChatGPT/GaussianGPT
ssh -o BatchMode=yes -S /tmp/ggpt_transition_active.sock gaussiangpt-autodl \
  "cd /root/workspace/GaussianGPT && PYTHONPATH=/root/workspace/GaussianGPT /root/miniconda3/envs/gaussiangpt/bin/python - --teacher checkpoints/gpt_vfront.ckpt --vqvae checkpoints/vqvae_vfront.ckpt --contexts 64,256,768,2048 --shapes 1,2,4 --warmup-calls 3 --pair-repeats 10 --calls-per-pair 3 --out-dir /root/autodl-tmp/pf_target_graph_probe_20261021/run01" \
  < patches/pf2x_breakthrough_20261021/runtime/pf_target_graph_probe.py
```

CPU-only self-test 可用同一 Python 确认显式 attention 的数值对齐和配对摘要方向：

```bash
ssh -o BatchMode=yes -S /tmp/ggpt_transition_active.sock gaussiangpt-autodl \
  "cd /root/workspace/GaussianGPT && PYTHONPATH=/root/workspace/GaussianGPT /root/miniconda3/envs/gaussiangpt/bin/python - --self-test" \
  < patches/pf2x_breakthrough_20261021/runtime/pf_target_graph_probe.py
```

运行生成 `metadata.json`、`pairs.jsonl`、`correctness.jsonl` 与每个形状的 `outputs_ctx{ctx}_T{T}.pt`。开始下一轮前应更换输出目录名；保留完整目录供端到端 runner 读取。

## 结果边界

该测量固定输入 token、KV 内容与 cache position。cache 创建、前缀处理、模型加载和图准备单独记录，不计入单次前向延迟。Ranker、Feature MLP、候选采样、接受与拒绝回退、cache 截短和每次 decode 的初始化均不在本脚本里，因此单个 shape 的速度比不能代表完整 PF/Target 比值。只有完整配对 decode 在相同模型、输入和采样设置下仍达到 2×，才能确认项目目标。

本轮子代理没有运行 GPU；真实 Target CUDA Graph 结果由主代理安排并验收。

## 动态位置图与实际缓存验证

`runtime/dynamic_target_graph.py` 为真实 GPT core 提供独立运行时包装，不改生产源码。图按持久 KVCache 实例、T=1/2/4、读取长度桶和 typed-logit 行种类创建；每次回放仅更新图输入 token、GPU 位置标量与 RoPE。读取长度向上归入 8 起步的二次幂桶，超过模型长度的最后一桶直接使用模型长度。图内显式 QK、softmax、V 会按绝对 token 位置写入稳定 KV 地址，并以 causal mask 屏蔽桶内未到达位置。它遵循当前 SDPA 对短 query 的有效历史范围，不按 window_size 额外裁剪。

长前缀继续走原始 forward。图返回最终 Block 的 pre-final-norm hidden；因 CUDA Graph 重放不触发 Python hook，包装器会把本次 hidden 写回 runner 提供的 `CaptureLastBlock` 实例，供 Feature MLP 使用。slot logits mask 与 monotonic position mask 留在图外执行。超过 T=1/2/4、非单 batch、训练态和无 KVCache 的调用继续走原 forward。图首次准备、预热和捕获时间分别记入 `runtime.stats()` 的 `prepare_ms`、`warmup_ms`、`capture_ms`；`cache_events` 记录容量分配或重置耗时与 KV 字节数。完整解码 runner 应保留同一个 runtime，使重复 decode 复用缓存和图，并报告这些一次性耗时。

runner 接入时，对 `draft_a.pf_ranker_eval.cache_for` 做局部替换，让相同入口同时覆盖 Target-only 与 PF Target；原函数需在 `finally` 中恢复。Target 每次 decode 得到稳定容量的缓存，复用时会调用 `reset()`；拒绝路径仍通过原有 cache 截短逻辑恢复到已提交位置，下一次图回放读取新的位置标量。调用参考如下：

```python
from dynamic_target_graph import DynamicTargetGraph
import draft_a.pf_ranker_eval as ranker_eval

runtime = DynamicTargetGraph(core, warmup_calls=2)
original_cache_for = ranker_eval.cache_for
ranker_eval.cache_for = lambda model, capacity: runtime.cache_for(
    model, capacity, factory=original_cache_for
)
try:
    with runtime.attention_context(enabled=True, hidden_capture=capture):
        result = ranker_eval.decode(...)
finally:
    ranker_eval.cache_for = original_cache_for
```

完整 runner 需在 warmup 阶段用这个持久 runtime 预建常见图，并单独输出冷启动明细；配对计时阶段继续复用。候选输出的 GPU 张量指向图内静态缓冲区，逐次保存原始 logits 与 hidden 时必须复制，避免相同形状的后续回放覆盖已有结果。

`runtime/verify_dynamic_target_graph.py` 用真实 Target、真实 crop 与原 SDPA 参照走一遍长前缀、混合 T1/T2/T4、读取桶跨越、截短到已提交位置后更换 Feature 输入并重跑 T2。每项操作立即把两臂输入、typed logits 和 pre-final-norm hidden 保存至 `raw_outputs.pt`。整段推理在 `torch.inference_mode()` 中执行；成功前还会确认实际图回放不少于 4 次、创建 T1/T2/T4 图、真实图事件跨越读取长度桶，并且除长前缀外没有运行时回退。这样原始前向自比不能被误记为动态图通过。比较要求 logit finite 支持相同、逐 token sampling policy 正值支持相同且最大 TV≤1e-5；hidden 必须全为有限数，逐行 L2 相对误差≤1e-5。元素级误差与 allclose 仅作附加摘要。数值出界时 metadata 会记录失败并保留此前原始张量。

在远端项目目录执行时，指定已有 crop 路径与全新的结果目录：

```bash
cd /root/workspace/GaussianGPT
PYTHONPATH=/root/workspace/GaussianGPT /root/miniconda3/envs/gaussiangpt/bin/python \
  patches/pf2x_breakthrough_20261021/runtime/verify_dynamic_target_graph.py \
  --project /root/workspace/GaussianGPT \
  --crops /path/to/development-crops.pt \
  --out-dir /root/autodl-tmp/pf_dynamic_target_graph_20261021/run01 \
  --prefix-body 510
```

本地完成 Python AST 语法解析与 `--help` 命令行验证；本地没有运行真实模型。远端首轮 `dynamic_verify_run01` 虽报告 passed，但 metadata 显示 `graph_count=0`、`graph_events=[]`，五个候选调用均因 `grad_enabled` 走原前向，故该轮没有动态路径证据。verifier 已加入 `inference_mode()` 和实际图事件要求，主代理负责运行更新后的版本。该实测只验证数值、缓存位置与恢复路径，不作为完整 PF 加速结论；最终收益仍须由成对完整解码结果确认。


## 完成后的结果入口

本设计后的独立确认、自然EOS与三维有效性已经完成，最终实现、数值、计时范围、原始结果与适用限制见[00_READ_ME_FIRST.md](00_READ_ME_FIRST.md)。固定原Target参照的新PF为4.519564×，相对原PF为2.613919×，同优化Target下为1.442534×。下方或上方较早阶段的未运行描述保留设计时状态，不代表当前完成状态。
