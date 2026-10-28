# D8 联合 P/F 的 Gamma 2、4、6 对照

2026-10-14。用户要求增加 D8 同时提出 P/F 的 Gamma 至 4 和 6。本轮同进程重测 Gamma2 与 Target，以排除不同测速轮次的绝对吞吐波动。沿用远端 `/root/workspace/GaussianGPT`、Git `031d916`、RTX 4090 和原 Python 环境。旧 SSH 会话失效后已重新连接；启动前 GPU 为 2 MiB、利用率 0%，没有其他训练或评价作业。

权重为 `draft_a/runs/feature_acceptance_20261014/adapt_d8_acceptance/step_3600.pt`，原始 Target 为 `checkpoints/gpt_vfront.ckpt`。模型不训练、不重新选型，权重与采样规则保持原样。

真实输入沿用一个 3D-FRONT 重建场景的四个不同 VFront 裁剪，crop_seed 为 0/1/3/4，重复 seed2 排除。重新读取 token 与既有 `real_crops.pt` 逐个相等。真实前缀 64/256/1024 token，种子组 34701/34702/34703，每次最多生成 512 token，FP32，temperature=0.9、top-p=0.9。

Gamma 指连续提出的 token 个数，候选可从 P 或 F 开始。三种 Gamma 均使用既有 `feature_acceptance_eval.decode`，首次拒绝后未用到的提案不进入 verified。主指标 accepted/verified，另存 accepted/proposed，记录每轮平均接受长度、平均实际产出、Draft/Target 调用数以及分前缀和分种子结果。

Target 普通解码沿用上一轮相同入口。四种模式在每个裁剪/前缀/种子组合内轮转运行，使用配对种子，共 4×3×3×4=144 次正式运行。外层计时前后 CUDA 同步，包含输入输出验证、缓存分配释放、前缀处理、GPT 提案、Target 验证、拒绝处理与采样，排除加载与数据准备。速度由实际总 token/总秒数汇总，再除以同轮 Target 吞吐。

先运行 Target 和 D8 的逐 token、分块及缓存退回数值验证，三档 Gamma 与三档前缀下验证短生成计数可重复，且 Target 向自身提案全部接受。真实数据只有一个场景，不提供跨场景置信区间，不把三个重复种子解释为三个独立场景。

入口为远端 `draft_a/feature_d8_gamma_eval.py`，本地源码副本为 `patches/feature_target_v2/feature_d8_gamma_eval.py`。结果目录为 `draft_a/runs/feature_d8_gamma_eval_20261014/`。短运行与正式 144 次评价均已完成，进程 397892 已退出。


## 正式结果

每种模式 36 次运行，所有速度比使用本轮 Target 的 68.3204 token/s。

| Gamma | Position accepted/verified | Feature accepted/verified | token/s | 对 Target 速度比 | 对 Gamma2 吞吐 | 每轮接受 token | 每轮实际产出 token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 5896/6822=86.4263% | 4203/7730=54.3726% | 66.2428 | 0.9696× | 1.0000× | 1.2098 | 2.2080 |
| 4 | 6988/8050=86.8075% | 4611/8588=53.6912% | 62.3531 | 0.9127× | 0.9413× | 1.7871 | 2.7843 |
| 6 | 7687/8685=88.5089% | 4855/9053=53.6286% | 55.6218 | 0.8141× | 0.8397× | 2.1229 | 3.1198 |

Gamma4 和 Gamma6 相对 Gamma2 的吞吐分别低 5.87% 和 16.03%。三个种子组都得到 Gamma2 > Gamma4 > Gamma6 的速度顺序。四份裁剪也都得到相同顺序。当前三档中 Gamma2 最快，整体仍低于 Target 普通解码。

较大的 Gamma 减少了 Target 调用，但 D8 逐 token 提案次数与未使用的候选显著增加。下面按实际输出长度归一化，避免自然 EOS 造成总调用数的误读。未使用比例是 (proposed-verified)/proposed；已被拒绝的 token 仍计入 verified。

| Gamma | Target 调用总数 | Draft 调用总数 | 每千输出 token 的 Target 调用 | 每千输出 token 的 Draft 调用 | 提案总数 | 未使用候选 | 未使用比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 8384 | 20601 | 454.86 | 1117.68 | 16685 | 2133 | 12.78% |
| 4 | 6527 | 27359 | 361.15 | 1513.81 | 25889 | 9250 | 35.73% |
| 6 | 5944 | 36010 | 322.48 | 1953.67 | 35281 | 17542 | 49.72% |

Gamma6 约一半提案未进入实际验证；Target 调用更少，但每次会对更长候选块前向计算。对应 Target 输入 token 总数由 Gamma2 的 41161 增加至 Gamma4 的 48508、Gamma6 的 57317，包含真实前缀处理与验证输入。调用节省不足以抵消 Draft 提案和更长验证块的成本。这里使用调用及输入计数解释实测差异，没有单独测量各阶段 GPU 时间。

三档 Feature 条件接受率约为 54.37%、53.69%、53.63%，没有随 Gamma 增加而上升。Position 条件接受率有所变化；模型权重相同，生成前缀与到达后续候选的条件不同，不能据此声称单步预测能力提高。每轮实际产出还包含拒绝后的 Target 采样或全部接受后的额外 token，因此高于每轮接受长度。

## 分前缀结果

| 真实前缀 token | Gamma2 速度比 | Gamma4 速度比 | Gamma6 速度比 | Gamma2 Feature 接受率 | Gamma4 Feature 接受率 | Gamma6 Feature 接受率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.0048× | 0.9831× | 0.8946× | 57.8242% | 57.0097% | 56.6589% |
| 256 | 0.9718× | 0.9052× | 0.7935× | 55.2071% | 53.3938% | 53.1281% |
| 1024 | 0.9355× | 0.8589× | 0.7666× | 50.1916% | 50.6980% | 51.1089% |

64-token 前缀的 Gamma2 为 1.0048×，接近 Target；长前缀的三档均更慢。该单场景结果不足以确定跨场景或其他精度的最优 Gamma。

## 长度、EOS、验证与产物

Target 实际生成 18241 token、266.9921 秒；Gamma2 为 18432 token、278.2492 秒；Gamma4 为 18073 token、289.8494 秒；Gamma6 为 18432 token、331.3807 秒。Target 在 crop3、256-token 前缀、seed34703 输出 321 token 后自然 EOS。Gamma4 在 crop1、256-token 前缀、seed34702 输出 153 token 后自然 EOS，其余各次均输出 512 token。不同 Gamma 消耗随机数的方式不同，不要求使用同一 seed 得到相同轨迹。

144 条运行记录唯一且完整，汇总计数、接受率、吞吐与速度比已用本地原始 JSON 重新计算验证。Gamma2 和 Target 共 72 次的生成、提案、接受、轮数及 Target 调用计数完整复现上一轮。速度使用本轮计时，不能拼接上一轮绝对吞吐。

三档前缀 × 三档 Gamma 的九组短验证均通过：重复运行计数一致，Target 向自身提案全部接受。Target 和 D8 的逐 token、分块及缓存退回最大总变差分别为 1.4149e-6、4.1127e-6。源码语法验证通过，作业正常退出；结束时 GPU 为 2 MiB、利用率 0%。

本地原始产物为 `patches/feature_target_v2/d8_gamma_results/`，包含 `plan.json`、`online.json`、`summary.json`、`numerical_validation.json`、`progress.json` 及下载的压缩包。远端日志为 `draft_a/runs/feature_d8_gamma_eval_20261014.log`。新增源码只为本轮评价入口，原始 Target、D8 权重、既有解码模块与真实 token 文件未修改。交接入口、实验索引及生成状态快照已更新。

结论仅覆盖一个真实重建场景的四份裁剪、FP32、当前缓存与调度实现的 token 解码。未测生产 BF16、跨场景平均收益或完整三维重建与渲染耗时。本轮无需继续扩大 Gamma；若后续探索长提案，需先降低提案成本或减少拒绝后的无效候选计算。
