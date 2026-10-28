# 同优化Target下的既有投机方法实测结果

2026-10-21。十个开发来源的220次主比较完成；Ranker+MLP仍最快，相对同优化Target为1.450584×，95%来源区间[1.370351,1.523298]。EAGLE、Medusa、Hydra和D8没有反超；相对同优化Target的2×尚未达到。

## 主比较：10来源、512-token预算

RTX4090、batch1、FP32 Target、highest matmul、TF32关闭、temperature/top-p=0.9/0.9；64/256真实场景前缀，seed43101。11法均使用显式BMM+动态CUDA Graph Target。每法独立持久runtime，首次构造、KV分配、图准备和捕获、前缀处理、草稿、验证、恢复、采样均进入同步外层墙钟；权重加载单列。每法独立16-token预热runtime丢弃。

速度比按配对吞吐比计算，来源内取对数均值后对来源等权；10000次来源重采样给95%区间。高于1表示快于同样优化后的Target。区间覆盖来源变化，未估计全部随机种子及训练不确定性。

| 方法 | /同优化Target | 95%来源区间 | 共同满512 | 合并吞吐 token/s |
| --- | ---: | --- | ---: | ---: |
| Ranker + step12000 MLP | 1.451× | [1.370, 1.523] | 1.451× | 314.83 |
| Ranker-only | 1.371× | [1.338, 1.397] | 1.371× | 301.10 |
| D8 Feature + Ranker | 0.899× | [0.876, 0.923] | 0.899× | 197.52 |
| EAGLE v3 Feature γ=3 | 0.818× | [0.798, 0.837] | 0.818× | 179.55 |
| EAGLE v4 Draft KV复用 γ=3 | 0.779× | [0.759, 0.798] | 0.779× | 171.00 |
| Medusa LowRank | 0.725× | [0.642, 0.774] | 0.768× | 168.85 |
| Hydra Conditional | 0.681× | [0.604, 0.729] | 0.722× | 158.77 |
| EAGLE v3 Feature γ=5 | 0.670× | [0.649, 0.691] | 0.670× | 146.96 |
| EAGLE v2 Joint γ=3 | 0.549× | [0.544, 0.554] | 0.549× | 121.07 |
| D8 Joint | 0.492× | [0.471, 0.512] | 0.492× | 108.82 |

优化Target合并吞吐为219.83 token/s。合并吞吐是总token/总墙钟，与来源等权比值属于不同汇总量，不可直接把此列相除替代主比值。MLP/Ranker-only为约1.058×，其余每个候选/MLP的来源区间上界均低于1。

Medusa/Hydra各一条在5token自然EOS，原body为4token并真实记录一次EOS，适配器把该EOS加入规范输出。它们的共同满512子集为19对、10来源，其余方法为20对、10来源。两种口径的排序保持。

## 相同条件下，执行优化怎样改变算法比

另做3来源、64/256前缀、seed42101、256-token预算、原/优化配对，共132次。每种方法6/6输出及已记录计数一致。下表每列均相对该列自己的Target，不跨轮拼接绝对速度。

| 方法 | 原执行/原Target | 优化执行/优化Target |
| --- | ---: | ---: |
| Ranker + step12000 MLP | 1.671× | 1.277× |
| Ranker-only | 1.423× | 1.310× |
| D8 Feature + Ranker | 1.474× | 0.848× |
| EAGLE v3 Feature γ=3 | 1.358× | 0.815× |
| EAGLE v4 Draft KV复用 γ=3 | 1.348× | 0.794× |
| Medusa LowRank | 0.842× | 0.781× |
| Hydra Conditional | 0.829× | 0.699× |
| EAGLE v3 Feature γ=5 | 1.199× | 0.636× |
| EAGLE v2 Joint γ=3 | 0.748× | 0.548× |
| D8 Joint | 0.928× | 0.503× |

这轮Target自身提速2.804154×，EAGLE v3 γ3整条路径提速1.683367×，MLP为2.143210×。候选接受计数保持相同，投机方法相对Target的收益缩小。原Target有一个配对在165token自然EOS；共同满256子集各5对，结论方向保持。

## 图复用与实际工作量

主比较所有方法都实际执行图，所有普通前向回退均为每请求一次的长prefill；每法20次，未见其他回退原因，所有disabled_graphs为空。排除双方本请求新增图的共同子集仅作辅助，保留各请求全部墙钟，不扣任何时间。

| 方法 | 无新增图子集 /Target | 配对/来源数 |
| --- | ---: | ---: |
| Ranker + step12000 MLP | 1.484× | 18/9 |
| Ranker-only | 1.386× | 18/9 |
| D8 Feature + Ranker | 0.902× | 19/10 |
| EAGLE v3 Feature γ=3 | 0.814× | 18/9 |
| EAGLE v4 Draft KV复用 γ=3 | 0.775× | 18/9 |
| Medusa LowRank | 0.724× | 19/10 |
| Hydra Conditional | 0.684× | 16/9 |
| EAGLE v3 Feature γ=5 | 0.666× | 18/9 |
| EAGLE v2 Joint γ=3 | 0.546× | 18/9 |
| D8 Joint | 0.479× | 18/10 |

这些子集同样没有EAGLE/Medusa的速度反超。详细图形状、调用、图准备与KV容量见[运行诊断](optimized_main_run01/runtime_diagnostics.json)。计数字段与接受率见[计数说明](COUNTER_MEANINGS.md)及[主比较计数汇总](optimized_main_run01/counter_summary.json)。低接受率、草稿重放和较宽验证块可以解释工作量；本轮没有测出每项CPU/GPU耗时份额。

## LFQ纯BOS补测的限制

lfq_bos_run01完成18/18：Target、MLP、LFQ，3种子44101–44103、512token、原/优化两执行，场景token未使用。审计7/8项通过，唯一失败项为输出和计数重放：Target3/3、MLP2/3、LFQ0/3。三法16token预热均一致。MLP首分歧suffix索引61，LFQ三个种子首分歧索引147/179/391。该轮速度仅保存在原始记录，不列为有效比较结论；不能用BOS三种子推断多场景效果。

LFQ BOS的单种子同历史复核已结束：bos_replay_verify_run02先原样重现seed44101的旧输出，再沿213次相同Target输入及相同截短事件执行图前向，213次全部用图、共13图。203次LFQ草稿输入状态全部匹配到参考Target的具体call/row。Target最大policy TV=0.0126780681，hidden最大行L2相对误差0.0013210566；Target policy支持集发生变化，首次TV超过1e-5在call2/绝对位置3。BF16 Draft q最大TV=0.0214848574，支持集也变化。这些数值显著大于真实前缀run02验证范围，不能把LFQ BOS视为数值同等。该探针未追踪抽样决策，尚未定位具体首分歧决策或具体算子，不把LFQ速度列为有效结果。原版、图版dtype均为FP32 Target，BF16仅用于原LFQ草稿。完整元数据见[bos_replay_verify_run02/metadata.json](bos_replay_verify_run02/metadata.json)。

首次probe run01在写入进度时因重复status参数结束；参考追踪保留。只改独立诊断脚本后在新source与新out run02完成57.490秒的复核，旧产物未覆盖。run02原始张量在远端/root/autodl-tmp/gaussiangpt_target_methods_20261021/bos_replay_verify_run02/reference_trace.pt（51,002,452字节）与raw_outputs.pt（101,092,880字节）；本机保存元数据及driver日志，原始大张量未复制到本机。最后GPU2MiB/0%，所有本轮进程已退出，原生产及权重保留。


## 验证与证据范围

- 运行接口验证run02覆盖T1/T2/T3/T4/T6、typed/full读出、pre/post norm hidden、跨512/1024桶及截短后合法Feature替换，最大policy TV为1.33086e-6。run01保留，其扰动样例的输入类型不合法，不用于合法换Feature结论。
- 参考短输出12次、全方法短输出44次、3来源配对132次各自独立审计8/8；主比较220次独立审计8/8，其中原/优化输出项因仅optimized而不适用，不能当作新增逐位重放证据。
- 实现沿用本项目已有EAGLE、Medusa/Hydra线性或条件提案，不代表论文全部树形功能。权重未重训，未尝试所有历史权重节点。代表配置与未纳入原因见[METHOD_INVENTORY.md](METHOD_INVENTORY.md)。
- 主比较为10个开发来源、一个新seed、512token窗口；没有新方法超过MLP，未启动30独立来源追加确认，也没有本轮新增完整三维质量评价。

## 文件与运行位置

新增源码位于本机`patches/target_methods_20261021/`，结果与文档位于本目录。实际运行隔离源码为远端`/root/autodl-tmp/gaussiangpt_target_methods_20261021/source/prefix_smoke_run02/`；所有运行目录为同级下的`runtime_verify_run01/02`、`reference_smoke_run01`、`all_methods_smoke_run01`、`paired_pilot_run01`、`optimized_main_run01`、`lfq_bos_run01`。实际命令完整保存在各目录pipeline.json/plan.json，输出与逐行计数在records.jsonl。

原工程`/root/workspace/GaussianGPT`及原权重、数据、旧结果保持原样；未安装依赖、未提交Git。

数值复核的详细边界与对齐证据见[BOS_REPLAY_RESULT.md](BOS_REPLAY_RESULT.md)。对应隔离诊断源码分别为远端source/bos_replay_verify_run01与source/bos_replay_verify_run02；原始大张量保留远端，本机只保存元数据及run02日志。

本轮新增9个Python文件位于patches/target_methods_20261021：evaluate_methods.py、method_adapters.py、metrics.py、audit_methods.py、counter_summary.py，及runtime下dynamic_target_graph.py、verify_runtime.py、diagnose_runtime.py、verify_bos_replay.py。项目记忆更新为.codex/AI_START_HERE.md、docs/project/EXPERIMENTS.md、RESEARCH_SOLUTIONS.md、LOGIC_VALIDATION.md、OPERATIONS.md，并由原脚本刷新PROJECT_STATUS.md。生产工程及旧权重未改。
