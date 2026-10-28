# 高接受率 D8 在真实 VFront 输入上的评价

2026-10-14。用户要求将此前高 Feature 接受率的 D8 GPT 直接作为草稿模型，重新测量接受率与加速比。本轮只评价既有权重，不进行训练。沿用 `/root/workspace/GaussianGPT`、Git `031d916`、RTX 4090 与原 Python 环境；开始时没有其他训练或评价作业，GPU 为 2 MiB、利用率 0%。

## 权重与历史口径

实际权重为 `draft_a/runs/feature_acceptance_20261014/adapt_d8_acceptance/step_3600.pt`。模型 8 层、512 维、8 个注意力头，37617680 参数。服务器权重存在、数值有限，`selected.json` 指向 step3600。此前从 ASE 预训练权重出发，使用原始 Target 的完整 Feature 分布 CE、Position/EOS 标签及正式采样策略 TV 续训。

历史结果来自同目录 `test_online_acceptance_3600.json`：Teacher 测试轨迹的 0/64/256 前缀、Gamma2/4 合并后，Feature accepted/verified 为 8920/16353=54.5466%，Position 为 97.5978%。这属于 GPT 同时提出 P/F 时的 Feature 条件接受率，不能自动推广到真实 VFront 前缀或只提出 Feature 的接法。

## 本轮范围

复用上一轮同一个真实 3D-FRONT 重建场景、四个不同 VFront 裁剪，真实前缀 64/256/1024 token，种子组 34701/34702/34703，各最多生成 512 token，FP32，temperature=0.9、top-p=0.9。五个原始裁剪去除重复 seed2，保留 0/1/3/4。重新读取的 token 与已有裁剪文件逐个相等。

四种模式为 Target 普通解码、step12000 轻量 Feature Draft 加现有 Position ranker、D8 只提出 Feature 加同一 ranker、D8 同时提出 P/F。前两种 Feature 接法在共用循环中使用 P1/F1/P2 候选，K=2 指两个 Position；GPT 联合模式复用历史精确解码入口，Gamma=2 指两个 token 候选，其轮次起点可能为 P 或 F。两种提议长度含义分别记录，不能将 K 和 Gamma 混用。正式共 4 裁剪 × 3 前缀 × 3 种子组 × 4 模式 = 144 次。

共用外层 CUDA 同步计时包含输入输出验证、缓存分配及释放、前缀处理、提案、Target 验证、拒绝处理与采样，排除模型加载、数据准备及所有训练。吞吐用总实际生成 token/总秒数，速度比以同轮 Target 为分母。accepted/verified 和 accepted/proposed 分开汇总。

只提出 Feature 的两种模型还在全部 4420 个相同真实 Feature 前缀上比较分布重合度。D8 同时提出 P/F 时，在线前缀及分母会变化，不能把这一静态指标直接当作在线接受率。

## 验证与状态

原始 Target 与 D8 将分别验证逐 token、分块和缓存退回概率；相同真实前缀下，Feature 接法验证普通与细分计时输出及计数一致；联合 GPT 入口验证相同种子重复运行计数一致，以及 Target 向自身提案全部接受。选定权重保持原样，测试结果不用于重新选模。

入口为远端 `draft_a/feature_d8_real_eval.py`，复用新增 GPT 参数的 `draft_a/feature_real_eval.py`，本地源码位于 `patches/feature_target_v2/`。结果目录为 `draft_a/runs/feature_d8_real_eval_20261014/`。短运行已完成：一个裁剪、64-token 前缀、32-token 续写，四种模式均正常结束，数值与短运行验证通过。正式 144 次均已完成。

固定真实前缀评价已完成：4420 个位置上，D8 Feature 分布重合度为 52.3789%，step12000 MLP 为 13.4599%。Target 与 D8 逐 token、分块及缓存退回的最大总变差分别为 1.4149e-6、4.1127e-6，均通过既有 0.01 容差。

数据只有一个真实场景，裁剪之间可能共享几何内容，不提供跨场景置信区间。速度结论仅覆盖 token 解码，不能代表完整三维生成和渲染。

## 正式结果

| 模式 | Position accepted/verified | Feature accepted/verified | token/s | 对同轮 Target 速度比 | 相对同轮 MLP 吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target 普通解码 | — | — | 69.0931 | 1.0000× | — |
| step12000 MLP 加 ranker | 7839/9216=85.0586% | 1341/6607=20.2967% | 100.0689 | 1.4483× | 1.0000× |
| D8 Feature 加 ranker | 7966/9216=86.4366% | 2863/5376=53.2552% | 93.6945 | 1.3561× | 0.9363× |
| D8 联合 P/F，Gamma2 | 5896/6822=86.4263% | 4203/7730=54.3726% | 66.2975 | 0.9595× | 0.6625× |

每种模式运行 36 次。Target 普通解码有一次自然 EOS：crop3、256-token 前缀、seed34703，实际输出 321 token；其余三种模式均输出每次 512 token。Target 总生成 18241 token，用时 264.0063 秒；三种草稿各生成 18432 token，用时分别为 184.1931、196.7244、278.0195 秒。使用实际吞吐归一化比较。

D8 Feature 接法相对 MLP 的 Feature 接受率增加 32.9586 个百分点，吞吐低 6.37%。它将 Target 调用数从 9288 降至 7639，但需执行 D8 Transformer 前向与独立 KV 的前缀处理、推进和退回，整体实测时间仍增加。当前实现中，更高接受率没有换来更高总吞吐。不能只按 Target 调用数判断加速。

D8 Feature 三个种子组的相对 MLP 吞吐分别为 0.97653、0.95106、0.88253，均低于同组 MLP；联合 GPT 分别为 0.68126、0.66929、0.63679。样本来自单个场景，这些变化仅作当前条件下的描述。

| 真实前缀 token | D8 Feature P | D8 Feature F | D8 Feature 速度 | D8 联合 P | D8 联合 F | D8 联合速度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 91.5039% | 56.5194% | 1.4837× | 93.2969% | 57.8242% | 0.9942× |
| 256 | 86.8490% | 54.7499% | 1.3858× | 81.8969% | 55.2071% | 0.9476× |
| 1024 | 80.9570% | 48.3560% | 1.2260× | 84.4397% | 50.1916% | 0.9389× |

相同固定真实前缀的 D8 Feature 重合度 52.3789% 与在线总体 53.2552%/54.3726% 分开报告：前者固定真实历史和 Position，后者使用各解码方式实际到达的前缀，且 Feature 分母受前序接受情况影响。

## 验证、文件与剩余范围

六组 Feature 接法的普通与细分计时输出及计数一致；三个联合 GPT 重复运行计数一致。三种前缀下 Target 向自身提案均为 22/22 接受，原始 Target 与 D8 数值验证通过。全部 144 次正式运行正常结束。MLP 和 Target 的生成、提案及接受计数完整复现上一轮真实输入评价，速度使用本轮结果。

模型权重、原始 Target、Position ranker 和真实 token 文件均未改变。作业已退出，GPU 最终为 2 MiB、利用率 0%。本轮只增加评价能力，没有训练或替换正式模型，也没有安装依赖。

本地原始产物位于 `patches/feature_target_v2/d8_real_results/`，含 `plan.json`、`online.json`、`summary.json`、`fixed_real_overlap.json`、`numerical_validation.json` 和 `progress.json`。远端同名 JSON 位于运行目录。新增源码为 `feature_d8_real_eval.py`，修改共用 `feature_real_eval.py` 以支持 GPT Feature、联合 GPT、固定真实前缀分布比较和动态模式汇总；原 MLP 评价默认参数保留。`.codex/AI_START_HERE.md`、`docs/project/EXPERIMENTS.md` 和自动生成状态快照已更新。

结论仅覆盖这一真实重建场景、FP32 和当前两种提案调度方式。没有跨场景平均收益、生产 BF16 速度、三维质量或完整场景总耗时结论。
