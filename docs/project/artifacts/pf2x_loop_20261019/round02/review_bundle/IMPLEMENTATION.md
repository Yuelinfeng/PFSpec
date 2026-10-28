# Round02 类型分组验证入口

2026-10-20 10:12实际结果：主代理已在既有服务器完成真实Torch CPU比较与GPU微测，全部736前向和688次数值比较通过，四token配对中位降时仅0.4875%，未进入完整解码接入。实际路径、命令、时长与限制见micro_launch.json、micro_run01/和RESULTS.md。下文保留实现准备说明，示例命令不表示另一次运行。

**当前状态：** 本机原型、命令行和纯标准库调度自测已完成；尚未运行 Target GPU 微测，也未接入 W1 解码。微测最多 900 秒，计入载入、缓存预填充、预热、前向、数值比较和结果保存。

## 变更范围

`source_candidate/model/gpt/nanochat_gpt.py` 是从 `source_before/model/gpt/nanochat_gpt.py` 复制出的候选文件。它增加 `_project_typed_logits_grouped(self, x, logit_types, softcap=15)`，按 Position 与 Feature 汇集当前窗口行，分别执行一批 `F.linear`，再把结果放回原行与词表位置。原逐行路径仍由默认 `typed_projection="original"` 使用。`typed_projection="grouped"` 仅用于推理；`logit_types=None` 和含 `full` 的输入继续使用原完整词表分支。softcap、slot mask、Position 单调 mask、EOS ID 与权重保持现有逻辑。

`patches/pf2x_round01/pf_typed_verify_probe.py` 运行真实 Target 形状微测。它用现有 `_load_target` 载入模型，再取 `target_model.transformer` 作为 nanochat GPT。候选的 `GPT.forward` 与 `_project_typed_logits_grouped` 只临时绑定到这个模型实例；结束时恢复原来的实例属性，不改类定义或生产文件。脚本沿用 `cache_for`、`typed_kinds`、`CaptureLastBlock`、`truncate` 和 `policy`。

脚本默认读取 `/root/autodl-tmp/gaussiangpt_pf2x_round01_20261020/run01/collection/contexts.jsonl`，使用 `results/expected_measurements.json` 中确定的 8 个真实 `sample_id`。样本来自 4 个开发来源，每个来源各有前缀类别 64 和 256；包含 4 条 F1 接受与 4 条拒绝，能到达 P2 的记录中接受与拒绝各 2 条。P1/P2 为 EOS 的记录不进入四行微测。每种形状使用原记录的 `target_inputs[:1]`、`[:2]` 或 `[:4]`，不会抽取新 token。

## 前向与缓存次序

每个样本先用原 `context_prefix[:-1]` 建立真实 KV，保留位置为 `len(context_prefix)-1`。每个形状先运行一次未改动 `GPT.forward` 作为数值参照；之后 original 与 grouped 各预热两次，再交错运行 12 对。每个短前向开始前都要求当前 KV 位置等于该样本的 keep，前向后按原辅助函数恢复到同一位置。配对顺序随样本、形状与配对序号轮换。

每个样本另做奇数 start 的正确性比较：先处理待处理 Feature，再将固定的 P1 输入单独送入 Target，确认该行预测 Feature。它只作正确性用途，不进入形状计时。Position 行会确认 EOS 仍为有限值，Feature 行会确认 EOS 仍被遮掉；两种候选路径都与原输出的有效词表、概率、Hidden、RoPE 及有效 KV 前缀比较。

末块 hook 取得的是最后 Transformer block 输出、最终 norm 之前的 Hidden。脚本不使用 `return_hidden=True`。`GPT._get_rope` 包装只保存本次返回对象的引用；CUDA 事件同步后再复制到 CPU，因此 RoPE 数据复制不计入前向时间。缓存比较包括 `token_cache`、`cache_seqlens` 与所有层截至有效位置的 K/V；keep 之后保留的旧槽不参与比较。

基础测量为 8 次前缀预填充、96 次预热前向和 576 次配对前向。额外正确性工作为 24 次未改动形状参照、8 次奇数 start 预处理、8 次奇数 start 原路径参照和 16 次候选奇数 start 前向。完整计划为 736 次 Target 前向、728 次缓存恢复、288 条配对摘要和 688 次数值比较；完成时脚本会逐项核对这些计数。未开始的调用不会写入为已完成记录。

Host wall 与 CUDA Event 分列记录。缓存恢复的两种时长单独保留，不并入前向。配对记录同时给出 grouped/original 时间比和原始时长，摘要按形状及来源与形状汇总中位数。它们只说明当前微测路径的相对成本，不能当作端到端加速。Host 计时包含两条路径共用的末块 hook、RoPE 引用捕获和 autocast 上下文开销。

## 精度与预算

Target 载入及调用复用项目 `_fp32_runtime(torch)`，并沿用 OpenMP 单线程设置。脚本要求 matmul 精度为 `highest`、CUDA 与 cuDNN 的 TF32 均关闭；每个 Target 调用显式禁用 CUDA autocast。运行记录保存进入微测前、微测中和恢复后的精度设置。

900 秒计时从脚本运行入口开始，包含样本读取、模块载入、模型载入、缓存预填充、预热、前向及结果写入。每次开始新 Target 前读取同一个截止时间；SIGTERM 与内部定时信号处理器只设置停止标记，不调用 CUDA。正在进行的同步模型载入或 Target 前向不能被 Python 安全中断，结束后会先恢复缓存、写入完成记录，再停止后续调用。若到时仍有未完工作，进程返回码为 2，结果标记为 `partial`。该边界无法覆盖进程收到 SIGKILL 或机器掉电的情形。

输出目录必须是新路径；已有路径会报错。`probe_rows.jsonl` 逐条保存前向、KV 恢复与数值比较；`paired_target.jsonl` 在每对完成后立即写入 Host/CUDA 时长、恢复耗时与数值结果；`progress.json`、`metadata.json` 与 `summary.json` 保存进度、预算、计数和摘要。JSONL 逐行刷盘，状态 JSON 采用临时文件替换。

## 远端调用

在现有远端工程环境执行时，命令如下。输出路径须先确认不存在；不要从本机执行这条命令。

```bash
cd /root/workspace/GaussianGPT
/root/miniconda3/envs/gaussiangpt/bin/python \
  patches/pf2x_round01/pf_typed_verify_probe.py \
  --project-root /root/workspace/GaussianGPT \
  --contexts /root/autodl-tmp/gaussiangpt_pf2x_round01_20261020/run01/collection/contexts.jsonl \
  --target checkpoints/gpt_vfront.ckpt \
  --vqvae checkpoints/vqvae_vfront.ckpt \
  --candidate docs/project/artifacts/pf2x_loop_20261019/round02/source_candidate/model/gpt/nanochat_gpt.py \
  --output /root/autodl-tmp/gaussiangpt_pf2x_round02/gate_b_typed_probe
```

本机没有 Torch，当前仅验证 Python 语法、CLI、固定样本读取、调度顺序、EOS 性能样本排除与配对摘要。候选分组函数的真实 Torch CPU 数值结果需在现有远端环境复跑；GPU 微测仍未执行。若 grouped 与原路径数值不符，或配对时长没有可重复的下降，不进入 W1 改动。
