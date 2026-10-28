# 真实 VFront 前缀上的 P/F Draft 评价

2026-10-14。用户要求用真实数据评价当前接受率与加速比。本轮沿用远端 `/root/workspace/GaussianGPT`、Git `031d916`、RTX 4090 和 `/root/miniconda3/envs/gaussiangpt/bin/python`。开始时 GPU 为 2 MiB、利用率 0%，没有其他训练或评价作业。现有环境与权重保留。

## 数据与范围

实际可用数据为 3D-FRONT 场景 `04396802-8d89-4377-8b28-4cadc9eb07c6`。已有链路从真实 mesh 生成多视角 Gaussian，并用 `checkpoints/vqvae_vfront.ckpt` 生成坐标和 Feature token，文件为 `/root/autodl-tmp/3D-FRONT-pilot/04396802-8d89-4377-8b28-4cadc9eb07c6/gaussian_init/vfront_pilot_tokens.pt`。

复用 `draft_a/analyze_v2.py` 中已有 VFront 裁剪读取方式，重新生成 seed0–4 的五个样本，与 `draft_a/runs/rollout500_20261013/real_crops.pt` 的 token 逐个比较。seed2 与 seed1 完全相同，只保留 seed0、1、3、4，共四个不同裁剪、8848 个 body token，长度分别为 2044、2286、2084、2434。它们来自同一场景，不能解释为四个独立场景。

固定真实前缀评价覆盖全部可用 Feature 位置，跳过每个裁剪缺少前一对 P/F 的第一个 Feature。原始 Target 在完整真实序列上计算条件分布，两版 Feature Draft 使用完全相同的状态与候选 Position，比较 temperature=0.9、top-p=0.9 的分布重合度。

在线评价从真实 body 前缀 64、256、1024 token 续写，各生成最多 512 token。续写部分由模型采样，速度结论只覆盖 token 解码；不覆盖真实场景 token 编码、VQ 解码、Gaussian 重建或渲染。

## 模型与计时

三组为原始 Target 普通解码、step3600 轻量 Feature Draft、step12000 轻量 Feature Draft。Target 为 `checkpoints/gpt_vfront.ckpt`，两版分别来自 `draft_a/runs/feature_target_full_v2_20261014/selected.pt`、`draft_a/runs/feature_target_continue_20261014/selected.pt`，共用同一个 Position ranker。K=2 对应通常的 P1/F1/P2 候选与 Target Feature 补充 token，FP32，temperature/top-p 均为 0.9。

三个重复种子组为 34701、34702、34703；每个裁剪/前缀在组内派生配对种子。运行顺序轮转，预先运行各前缀的短生成，验证普通与细分计时模式输出及计数一致。正式预算为 4 裁剪 × 3 前缀 × 3 种子组 × 3 模式 = 108 次。

本轮增加共用外层计时，两条路径均在进入 decode 前 CUDA 同步，返回后再次同步。耗时包含输入与输出验证、缓存分配及释放、前缀处理、提案、Target 验证、拒绝处理与采样，排除模型加载和数据准备。内层原有耗时另存 `decoder_seconds`。因此不能把本轮绝对速度与上一轮采用内层计时的速度直接拼接比较。

接受率以 accepted/verified 汇总，另存 accepted/proposed。前序拒绝后未使用的提案不计入 verified。速度为总新增 token 除以总外层耗时，然后除以同轮 Target 吞吐。不按单次比值取平均。

只有一个真实场景，本轮不估计跨场景置信区间；按种子与前缀的变化仅作描述。真实裁剪此前已用于语料分析，不把它们描述为从未接触的独立场景。

## 状态与文件

新增入口为远端 `draft_a/feature_real_eval.py`，本地副本为 `patches/feature_target_v2/feature_real_eval.py`。结果目录为 `draft_a/runs/feature_real_eval_20261014/`，保存数据来源、去重说明、固定前缀结果、逐次在线结果、汇总和进度。短运行已通过：一个裁剪、64-token 前缀、32-token 续写，三种模式正常结束；相同前缀上两版模型的普通与细分计时模式输出及计数一致。正式 108 次均已完成，六组计时模式输出及计数验证通过。作业已退出，GPU 最终为 2 MiB、利用率 0%。

固定真实前缀评价已完成，四个裁剪共 4420 个可用 Feature 位置。step3600 按位置重合度为 13.3945%，step12000 为 13.4599%，增加 0.0654 个百分点。各裁剪结果保存在 `fixed_real_overlap.json`。该指标使用全部真实前缀，在线结果使用真实起点后模型实际生成的前缀，两者不能直接互换。

## 正式结果

| 模式 | 新增 token | 总秒数 | token/s | Position accepted/verified | Feature accepted/verified | 对同轮 Target 速度比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target 普通解码 | 18241 | 269.1857 | 67.7636 | — | — | 1.0000× |
| step3600 | 18339 | 182.7242 | 100.3644 | 7855/9170=85.6598% | 1432/6523=21.9531% | 1.4811× |
| step12000 | 18432 | 185.6122 | 99.3038 | 7839/9216=85.0586% | 1341/6607=20.2967% | 1.4654× |

每组为 36 次运行。Target 和 step3600 在同一个裁剪/前缀/种子组合下分别自然 EOS，输出 321、419 token；其余运行均输出 512 token，step12000 没有提前 EOS。总 token 数因此不同，使用实际 token/s 计算速度，不直接比较总秒数。总外层耗时与原内层耗时的差值分别约 0.0132、0.0198、0.0190 秒。

step12000 相对 step3600 吞吐为 0.98943，即本轮点估计低约 1.06%；Feature 接受率低 1.6564 个百分点。三个种子组的新/旧速度比分别为 0.93746、1.01938、1.01344，方向不一致。这里只有一个场景，不能把点估计解释为跨场景的确定退化或确定改善。

| 真实前缀 token | step3600 P | step12000 P | step3600 F | step12000 F | step3600 速度 | step12000 速度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 90.0716% | 88.9648% | 24.3870% | 24.3353% | 1.6040× | 1.5878× |
| 256 | 85.3272% | 86.1654% | 23.6904% | 20.1162% | 1.4987× | 1.4538× |
| 1024 | 81.5755% | 80.0456% | 17.7233% | 16.3180% | 1.3626× | 1.3714× |

两版在 1024-token 前缀上的 P/F 接受率均低于 64-token 前缀；吞吐还受到前缀执行成本、缓存长度及生成路径影响。本轮仍取得约 1.47× 的 token 解码加速，但续训未带来总体额外收益。

固定真实前缀的 Feature 对齐水平明显低于此前 Teacher 测试轨迹上的约 26.5%–26.9%。它提示当前重建样本上的分布覆盖仍有缺口；无法据此判断全部 VFront 场景的平均表现，也无法区分重建质量、数据处理及模型容量各自的影响。本轮没有训练、替换任何模型，也没有渲染质量结论。

本地原始结果副本位于 `patches/feature_target_v2/real_results/`，其中 `summary.json` 为汇总，`online.json` 为全部逐次结果，`fixed_real_overlap.json` 为固定前缀评价，`plan.json` 为数据来源及配置。交接入口与实验记录已更新，并刷新本机状态快照。
