# LFQ Feature Draft：输出头误差分析与训练候选

2026-10-13，用户授权先用 CPU 分析 Feature Draft 接受率，并准备训练候选。当前 RTX 4090 正执行 Draft-A v2；本任务不使用 GPU，不停止既有进程，也不创建自动启动训练的后台任务。

## 运行对象

远端主工程为 `/root/workspace/GaussianGPT`，Git 版本为 `031d916`。沿用 `/root/miniconda3/envs/gaussiangpt/bin/python`，Python 3.10.21；没有安装依赖。现有未提交源码与运行数据保留。

本轮输入权重为 `profiling/feature_draft/lfq_new_onpolicy_warmstart/lfq_scratch_conditional_best.pt`。模型接收上一 Position 的 Target 隐藏状态、上一 Feature 和当前 Position，以低 6 bit 类别分布与高 6 bit 条件分布表达 4096 类 Feature 联合分布。Feature ID 为 `low6 + 64 * high6`。Position XYZ 按当前模型的 20×20×20 网格解释；没有改变坐标和 token 规则。

数据为 `profiling/feature_draft/lfq_new_onpolicy` 内的已有在线轨迹。Target 标签来自 `checkpoints/gpt_vfront.ckpt`，temperature=0.9、top_p=0.9。分析与训练不执行 Target 或 VQ-VAE 前向，不需要载入 `checkpoints/vqvae_vfront.ckpt`。轨迹原采集窗口为 1024 token。保留测试数据未读取。

## 完整验证集结果

CPU 单线程分析覆盖 16 条验证轨迹、6138 个 Feature 位置。以下为按位置平均的分布重合度，代表已存上下文上的期望接受率，不代表新增在线解码结果。

| 指标 | 结果 |
| --- | ---: |
| 当前联合分布重合度 | 28.9468% |
| 低 6 bit 类别分布重合度 | 45.2679% |
| Target 低位概率加权的高 6 bit 条件重合度 | 42.9942% |
| Draft 支持范围覆盖的 Target 概率质量 | 78.6588% |
| Draft top_p=0.95 时的联合重合度 | 28.4589% |
| Draft top_p=1.0 时的联合重合度 | 27.6949% |
| 用 Teacher 低位分布替换对应部分 | 44.3357% |
| 用 Teacher 高位条件分布替换对应部分 | 46.6854% |

按轨迹等权的联合重合度为 29.6134%。这与按位置平均的 28.9468% 采用不同权重，不能混用。两个 Teacher 替换仅用于定位误差，运行时 Draft 无法直接取得对应概率；这些数字不构成可实现收益承诺。

按每条轨迹中已存 Feature 行的先后顺序分成四段，联合重合度依次为 40.11%、29.58%、23.14%、22.88%。数据没有保存准确的原 token 深度，本轮分段只能解释为观测行进度。

联合重合度最低四分之一的 1535 个位置，平均重合度仅为 2.44%；对应 Target 熵为 1.90，Draft 熵为 4.84。熵采用自然对数。这表明该子集的 Draft 分布远比 Target 分散，但全局温度调整仍不足以解决问题。

原始产物位于远端 `profiling/feature_draft/lfq_head_acceptance_20261013/analysis_val.json`。不把原始模型、数据和分析产物提交到 Git。

## 训练集与泛化差异

同一权重在全部 64 条训练轨迹、23823 个位置上，联合重合度为 46.8286%，低位重合度为 61.1679%，高位条件重合度为 59.5903%。按轨迹等权的联合重合度为 47.1189%。结果位于远端 `profiling/feature_draft/lfq_head_acceptance_20261013/analysis_train.json`。

训练与验证的联合重合度相差 17.8818 个百分点。这提示当前主要问题包含对新轨迹的泛化，不能仅凭训练分数上升判断候选有效。两组轨迹本身存在分布差异，因此该差值不能全部归因于过拟合。后续优先采用相同预算的损失对照与数据采样对照，不扩大网络容量。

历史材料 `/Users/lief/Downloads/P F 分离.md` 中，旧 Student 的 Spatial-Delta、Head-Aware、KV 摘要与 Full-KV 版本均未改善对应验证结果。LFQ Conditional 已在独立轨迹上取得相对旧 Student 的分布优势。当前没有理由原样重做上述旧方案；分别增加两部分类别分布重合度监督、按轨迹均衡采样，仍是待验证的独立改动。

## 训练候选

现有目标已经包含联合分布 CE、最终采样分布重合度，以及 12-bit 边际辅助监督。联合 CE 本身也可分解为低位 CE 和 Target 低位概率加权的高位条件 CE。因此本轮研究的是不同监督目标能否改善概率分配。

设 Target 已存概率为 p，Draft 完整联合概率为 q。两个新增训练量为：

- 低位重合度：对 64 类 `p_low` 与 `q_low` 的逐项最小值求和。
- 条件重合度：对每个低位类别计算高位条件重合度，再按 `p_low` 加权。

候选保留原目标，额外加入两项 `1 - overlap`，初始权重各为 0.5。Target 低位概率为零的类别不贡献条件损失。辅助量使用截断前的 Draft 联合分布，现有联合接受率目标继续使用最终 temperature/top_p 分布。

推理模型、输入、参数数量与输出格式保持不变。所有新增计算仅用于训练。第一项对照将两个辅助权重设为零，第二项设为 0.5/0.5；两者采用相同数据、预算、学习率、初始化与采样方式。按轨迹均衡采样另作独立对照，避免同时改变损失和数据权重。

使用已有训练集和验证集，不增加 Target 数据采集，不读取测试集。旧离线与新在线样本默认比例为 40%/60%。若旧数据缺少轨迹编号，轨迹均衡对照仅在有编号的来源内生效，并在运行记录中说明。

## 当前执行边界

本轮已完成 CPU 分析、训练入口实现与 CPU 功能验证，并开始相同训练预算的 CPU 对照。CUDA 训练与在线测速等待当前 GPU 工作结束后执行；没有安排轮询、定时任务或自动启动器。独立测试接受率和加速变化仍未确认。

验证集此前参与了多次权重选择，属于开发数据。候选确定后再使用保留测试轨迹评价，随后运行在线接受率与解码吞吐对照。

## 代码与 CPU 验证

远端新增 `analyze_lfq_heads.py` 与 `train_lfq_heads.py`，本机对应文件位于 `patches/lfq_feature_acceptance/`。现有模型、训练器、解码器和正式权重未修改。两项新增脚本通过语法验证。

训练入口的 CPU 自检已通过：q=p 时两个辅助损失为零；联合 CE 与两部分 CE 的分解一致；Target 低位零概率类别没有产生非有限值；两个输出头均有有效梯度；轨迹采样概率符合等权定义；一次更新后重新载入模型得到一致输出。辅助权重为零时与原训练目标一致。

真实 CPU 短测试读取现有父模型和新在线训练数据，采用轨迹均衡采样，完成 1 轮、128 个训练位置、4 次更新，并完整评价 6138 个验证位置。按位置重合度从 28.9468% 变为 28.8786%，按轨迹重合度从 29.6134% 变为 29.5447%，因此最佳轮次仍为父模型。该运行用于验证训练链路，没有接受率提升结论。

短测试使用 `--no-offline` 以减少 CPU 读取量，正式对照使用 40% 旧离线和 60% 新在线数据。短测试日志为远端 `profiling/feature_draft/lfq_head_acceptance_20261013/cpu_smoke.log`，指标为 `cpu_smoke/summary.json`。

本轮实际 CPU 命令如下。输出目录现已存在；需要重跑时显式选择新的实验目录，入口会拒绝覆盖既有结果。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  nice -n 19 /root/miniconda3/envs/gaussiangpt/bin/python -u train_lfq_heads.py \
  --device cpu --no-offline --sampling trajectories --epochs 1 --patience 1 \
  --samples-per-epoch 128 --batch-size 32 \
  --out-dir profiling/feature_draft/lfq_head_acceptance_20261013/cpu_smoke
```

短测试的最佳与最后权重均已实际载入并完成有限值前向，参数数量保持 2710110。验证完成后已移除这两个临时权重，保留日志与指标；可按相同命令重新生成。父模型和正式权重未改动。

每次正式训练从指定权重开始，重新创建 AdamW；`last.pt` 保存优化器与 CPU 随机状态，但本入口没有自动续训功能。验证按轨迹等权的重合度选择模型，同时记录按位置指标。不同验证来源之间采用等权平均。

## GPU 空闲后的运行入口

以下命令仅记录供后续执行，本轮未运行。先确认当前 GPU 作业完成；保留同一 RTX 4090、现有环境及 Target 参数。三个版本采用相同初始化、seed、学习率和训练预算，分别运行，不能同时占用 GPU。

```bash
cd /root/workspace/GaussianGPT

# 原训练目标，按位置采样。
OMP_NUM_THREADS=1 /root/miniconda3/envs/gaussiangpt/bin/python -u train_lfq_heads.py \
  --device cuda --sampling rows \
  --low-overlap-weight 0 --conditional-overlap-weight 0 \
  --out-dir profiling/feature_draft/lfq_head_acceptance_20261013/original_rows

# 新增两个输出头的重合度监督，数据采样方式保持相同。
OMP_NUM_THREADS=1 /root/miniconda3/envs/gaussiangpt/bin/python -u train_lfq_heads.py \
  --device cuda --sampling rows \
  --low-overlap-weight 0.5 --conditional-overlap-weight 0.5 \
  --out-dir profiling/feature_draft/lfq_head_acceptance_20261013/heads_rows

# 原训练目标，单独比较轨迹均衡采样。
OMP_NUM_THREADS=1 /root/miniconda3/envs/gaussiangpt/bin/python -u train_lfq_heads.py \
  --device cuda --sampling trajectories \
  --low-overlap-weight 0 --conditional-overlap-weight 0 \
  --out-dir profiling/feature_draft/lfq_head_acceptance_20261013/original_trajectories
```

共同默认参数为 8 轮、每轮 30000 个训练位置、batch_size=128、lr=3e-5、patience=3、seed=164039，推理 temperature/top_p 仍为 0.9/0.9。先比较原目标与新增监督，再根据结果决定是否需要组合两个改动。

候选首先用于 K=2 对照。当前 K=3 F2 权重与旧 F1 状态相配；新的 F1 权重不能直接视为已有 F2 的兼容替代，需要另行评价与适配。

## 同预算 CPU 对照

用户同意先完成 LFQ 对照，再依据结果决定 F1/F2 联合递归训练。GPU 仍执行 Draft-A v2，故先开展较小预算的 CPU 对照，沿用旧离线/新在线 40%/60% 混合。三个版本依次运行，每个版本 2 轮、每轮 2048 个训练位置、batch_size=32、lr=3e-5、seed=164039、patience=2。

- `original_rows`：原训练目标、按位置采样。
- `heads_rows`：新增低位与条件重合度监督、按位置采样。
- `original_trajectories`：原训练目标、按轨迹采样。

输出根目录为远端 `profiling/feature_draft/lfq_head_acceptance_20261013/cpu_comparison`，每个版本有单独日志与 `summary.json`。整个对照仅使用一个 CPU 线程，进程设置低优先级，CUDA 设备对这些进程不可见。没有新增后台调度。CPU 结果用于筛选方向，不能替代正式 GPU 训练与测速。

本轮已补齐候选权重中现有加载器要求的 `source` 字段，并保留新的训练方式字段。后续直接用现有 LFQ 加载器实际载入验证兼容性，避免候选只在新脚本中可用。

小预算三组均保留父模型。第二轮按轨迹重合度为原目标 29.4369%、新增监督 29.4811%、轨迹采样 29.4137%，均低于父模型 29.6134%。因计算耗时较短，随后使用 CPU 运行完整预算对照：最多 8 轮、每轮 30000 个位置、batch_size=128、patience=3。其余数据、初始化、seed 与学习率相同。

完整预算入口为 `run_lfq_comparison.py`，由当前会话前台顺序运行三个版本，输出根目录为远端 `profiling/feature_draft/lfq_head_acceptance_20261013/cpu_full_comparison`。该入口也保存在本机 `patches/lfq_feature_acceptance/`。CPU 使用 FP32；与原 CUDA BF16 训练存在数值差异，跨设备结果不能直接视为严格复现。

原目标和新增输出头监督均已完成 3 轮，触发原定 patience=3，选中轮次仍为父模型。两组最后一轮按位置重合度分别为 28.8678%、28.9196%，按轨迹重合度分别为 29.5292%、29.5817%；父模型对应 28.9468%、29.6134%。当前没有新增监督优于父模型的证据。

轨迹均衡版本完成 4 轮，第一轮按位置重合度为 28.9654%、按轨迹重合度为 29.6298%，相对父模型分别增加 0.0185、0.0163 个百分点；后三轮没有继续改善，按原定 patience=3 停止。旧离线来源缺少轨迹编号，沿用按位置采样；新在线来源采用轨迹均衡采样。正式权重未替换。

| 版本 | 完成轮数 | 选中轮次 | 选中模型按位置重合度 | 选中模型按轨迹重合度 |
| --- | ---: | ---: | ---: | ---: |
| 原训练目标 | 3 | 父模型 | 28.9468% | 29.6134% |
| 新增输出头监督 | 3 | 父模型 | 28.9468% | 29.6134% |
| 轨迹均衡采样 | 4 | 1 | 28.9654% | 29.6298% |

三组实际训练累计 300000 个位置，训练与每轮完整验证耗时合计约 231.65 秒，不含进程启动和读取数据。相同初始化、seed、学习率和数据来源配比；停止轮数因开发验证结果不同而不同。总体结果不支持通过这两个改动取得明显接受率收益。

轨迹均衡候选在相同验证上下文上，16 条轨迹中 10 条重合度上升；平均差为 +0.01634 个百分点，按轨迹重采样 10000 次得到描述性 95% 区间为 −0.19720 至 +0.20946 个百分点。区间包含零，且该验证集用于选择轮次，因此不能作为独立增益结论。结果为 `cpu_full_comparison/trajectory_difference.json`。

选出的轨迹均衡候选保留在 `cpu_full_comparison/original_trajectories/lfq_scratch_conditional_best.pt`，已通过现有加载器的实际载入与前向验证。其余小预算与完整预算对照中，重复父模型权重和未选中最后权重共 11 个文件已移除；日志、指标和精确删除清单 `cpu_full_comparison/weight_cleanup.json` 保留，可重新运行生成。正式父模型、旧 F2 与用户运行数据保留。

候选权重已由现有 `pf3_depth2_model.load_base`、`continue_lfq_conditional.restore` 和本轮训练入口分别加载，在同一 CPU 输入上的输出逐项一致。该验证只说明现有 LFQ 加载与前向兼容，不能使旧 F2 权重自动适配改变后的 F1。

## 下一阶段判断

不继续增加本轮输出头监督的训练轮数。下一阶段优先研究 F1/F2 联合递归训练，保持当前模型容量与真实候选输入：一部分样本监督 F1，另一部分使用已保存的 `hprev、fprev、p1、f1_draft、p2` 与 F2 Target 概率，重新计算 F1 中间状态并将 F2 损失传回该状态的生成模块。保留分别评价 F1 与 F2 的开发轨迹，避免只改善第二层却损失现有 K=2 收益。

已有 F2 数据仅覆盖旧提议器实际到达的前缀，不能用同一标签监督随意重新采样的 F1；改变 F1 后的在线分布仍需后续真实采集。用户进一步授权后，联合递归 CPU 对照已完成，F2 开发重合度增加 0.4441 个百分点，F1 按轨迹下降 0.0200 个百分点，见 [后续实验记录](PF_JOINT_RECURSIVE_2026-10-13.md)。没有运行 K=2/K=3 动态长度或空间 Delta 新实验。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  nice -n 19 /root/miniconda3/envs/gaussiangpt/bin/python -u run_lfq_comparison.py \
  --device cpu \
  --out-dir profiling/feature_draft/lfq_head_acceptance_20261013/cpu_full_comparison
```
