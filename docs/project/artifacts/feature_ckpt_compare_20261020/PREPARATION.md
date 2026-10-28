# 12k 与 48k Feature 权重速度比较准备

`compare_checkpoints.py` 沿用现有 `draft_a.pf_feature_unified.decode`、Target、FastHistRanker 和 `MLPAdapter`。解码器使用 K=2，Target 与 Feature MLP 均按 FP32 运行，温度和 top-p 均为 0.9，批量为 1。Target、ranker、Head 和解码逻辑保持原样。

## 输入与运行命令

评测读取原开发集 `dev_crops.pt`，按来源名排序，每个来源选首条长度至少 256 的 crop，共 10 个来源。输入包括 prefix 64/256、seed base 48701/48702/48703、每次最多生成 512 个 token，以及 step 12000 与精确 step 48000 权重；每组来源、prefix、seed 对两权重各运行一次，共 120 次。两次运行共用相同 prompt 和 seed，并按来源、prefix、seed 的索引交替先后顺序。

启动计时预算后，程序先在 CPU 读取两个权重的 step、模型配置与张量形状，再加载原 Target、FastHistRanker 及两份 MLP。步数须分别为 12000 和 48000，结构须相同；加载器还会按现有 Head 与 Target 做兼容性核对。FP32 设置复用项目的 `_fp32_runtime`，并在开始前确认 matmul 精度为 highest、CUDA matmul 与 cuDNN 的 TF32 均关闭。

两份 MLP 各做一次 32-token 预热。随后每份权重各以相同 prompt 与 seed 重放两次 16-token 解码，核对同一权重重复调用的完整输出 token 与解码计数；两份权重之间的重放是否相同仅作为记录。测量阶段调用未启用分段计时的原解码器；每次调用前后同步 CUDA，保存完整输出 token、Target/ranker/MLP 调用量、接受计数、实际长度、EOS、解码器内部时间和外层墙钟时间。

建议的远端命令：

```bash
python patches/feature_ckpt_compare_20261020/compare_checkpoints.py \
  --project /root/workspace/GaussianGPT \
  --crops /root/autodl-tmp/pf_multiscene_pilot_20261015/stage_c/dev_crops.pt \
  --checkpoint-12000 draft_a/runs/feature_target_continue_20261014/selected.pt \
  --checkpoint-48000 /root/autodl-tmp/gaussiangpt_feature_extend_20261019/48000.pt \
  --out /root/autodl-tmp/gaussiangpt_feature_ckpt_compare_20261020/run01 \
  --deadline-seconds 1800
```

`--out` 必须是尚不存在的新目录。相对权重路径按 `--project` 解析。18 分钟由整个进程共用，从 Python 导入和权重读取开始，涵盖模型加载、预热、重放与 120 次测量。

## 产物与汇总口径

输出目录会保存 `plan.json`、`verification.json`、逐次追加并同步写入的 `records.jsonl`、`progress.json`、`pipeline.json` 和 `summary.json`。若预算到期或收到 SIGTERM/SIGINT，程序在当前解码调用返回后保存已完成记录，标记部分结果并停止后续 case；若 120 条已完成但最终写盘越过截止时间，状态记为 `budget_overrun`。程序不会改用别的来源或重新启动新预算。普通错误会写入错误类型和说明，再以失败状态退出。

速度主指标是每个来源各自合并六个上下文后的总生成 token 数除以外层同步墙钟时间，再对 10 个来源取等权几何平均。来源 bootstrap 以来源为单位有放回抽样 10,000 次，给出 95% 区间；每个来源的两权重速度比、合并 token/s、F/P 接受率、输出长度分布及 EOS 比例也会保存。另列两种权重都完整生成 512 token 的共同子集作为描述。输出长度不同时，不能直接用总墙钟时间判定快慢。

## 尚未执行与范围

本机仅运行 Python 语法解析和 `--help`；没有读取权重、加载模型、连接远端或使用 GPU。实际矩阵仍需在既有远端环境运行。墙钟截止无法打断正在执行的单次 Target 解码；该调用返回后会先落盘，再停止后续工作，因此极慢的单次调用仍可能使总时间略超 1800 秒。此次比较只覆盖这 10 个既有开发来源和指定设置，结果不能代替新来源上的泛化评估。
