# 实验执行状态

当前：十来源开发主测240次、组合正确性36次、三十来源组合确认240次完成；纯备用分支三十来源180次完成；全部GPU任务退出。最终结论以RESULTS.md为准。

## 对象与位置

本机主交接区：`/Users/lief/Documents/ChatGPT/GaussianGPT`。

新增源码：`patches/spec_next_20261022/`。新增计划、方法解释、原始JSON与复算：`docs/project/artifacts/spec_next_20261022/`。本机没有可运行Target或PyTorch，GPU测量在已确认的远端主环境执行。

远端主工程：`/root/workspace/GaussianGPT`。原Target/VQ/Ranker/MLP权重沿用，生产默认、原Jupyter与TensorBoard保留。

远端本轮根目录：`/root/autodl-tmp/gaussiangpt_spec_next_20261022/`。

| 目录 | 用途与状态 |
|---|---|
| source_run01 | 初始诊断与有限解码；调试过程保留 |
| source_run02 | 开发主测、五节点/混合验证的实际执行源码 |
| source_run03 | 快速混合、组合、三十来源确认的实际执行源码 |
| prob_verify_run01 | 缺少导入的失败记录，保留 |
| prob_verify_run02 | 96次概率实现对照完成 |
| candidates_verify_run01 | 数值比较与204次解码完成 |
| rescue_smoke_run01 | 8次备用路径有限对照完成 |
| reuse_zero_run01 | 8次零权重对照完成 |
| main_run01 | 十来源240次开发主测完成 |
| combined_verify_run01 | 36次组合对照完成 |
| confirmation_run01 | 三十来源240次组合确认完成 |
| confirmation_rescue_run01 | 新种子三十来源180次纯备用确认完成 |

上述source_run02/03在相应测量启动后不再修改。组合验证的实际脚本另存raw/combined_verify_run01/verify_combined_executed.py；本机最新verify_combined.py增加了图调用/完整度判定，相应补充要求已经用已存记录独立复算。解码器不受这一差别影响。

旧feature_tree依赖：`/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04`，其reference/runtime继续复用。新runner的--source指向该依赖，并非本轮source_run03自身；本轮脚本目录优先载入新的decoder。

## 运行边界

FP32、TF32关闭、最高矩阵精度、temperature/top-p=.9/.9、容量2048。每臂独立持久运行时；原Target普通执行；首图时间进入同步外层解码。权重载入在计时外，完整模型载入秒数另记。独立预热不复用于正式请求。

确认组采用另外三十来源、64/256前缀、seed50101；混合权重.25不再调整。数据历史用于旧方法测试，不能宣称全项目从未使用。早EOS和满512分别统计，没有新的三维端到端倍率。

未开始新训练、MultiTap、W4或动态预算选择。全部结果保留，失败不被后续成功目录覆盖。

纯备用确认的可选source_scope标签漏传，原plan中默认development；实际crops和三十来源ID均来自stage_e_final，与组合确认来源相同。原plan与命令不改，真实范围说明见source_scope_note.json。此差别只涉及标签，未影响数据、种子、模型、计时或方法。

## 本轮源码清单

| 文件 | 实际用途 |
|---|---|
| diagnose_position.py | 相同前缀的Position分布诊断 |
| diagnose_feature_reuse.py | 历史Feature policy混合诊断 |
| prob_position_decoder.py | 归一Ranker概率提案与精确残差 |
| position_rescue_runtime.py | 五节点备用树与实际路径提交 |
| position_rescue_decoder.py | 点提案拒绝后的备用继续执行 |
| feature_reuse_decoder.py | 独立概率副本的混合实现 |
| feature_reuse_fast_decoder.py | lerp与拥有的概率张量切片 |
| combined_decoder.py | 备用继续与快速混合组合 |
| evaluate_probability.py | 概率实现有限对照 |
| evaluate_candidates.py | 原Target/W1/备用/混合的同轮计时 |
| evaluate_combined.py | 原Target/W1/快速混合/组合的同轮计时 |
| verify_candidates.py | 直接数值、缓存与204次解码对照 |
| verify_combined.py | 组合36次对照与逐次概率算术记录 |
| cost_diagnosis.py | 历史成本与初始分布诊断复算 |
| summarize_runs.py | 来源统计、子集与配对汇总 |

最终环境与逐文件对照：environment_final.json、measured_source_run02_comparison.json、measured_source_run03_comparison.json。GPU2MiB/0%，无计算进程，原服务保留。可用候选为纯备用Position，Feature混合不替换W1。
