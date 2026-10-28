# 原 Target 参照与下一步方法结论

同轮120次测量完成。主比较已统一为原Target=1倍，当前最快仍为Ranker+MLP；新方法筛选完成，尚无新候选在线速度。

| 当前完整路径 | 相对原Target | 95%来源区间 |
|---|---:|---:|
| 原Target普通执行 | 1.000 | 参照 |
| Target执行优化 | 3.153 | [2.953,3.272] |
| Ranker+MLP + Target/提案图 | 4.637 | [4.112,5.053] |
| 当前EAGLE v3 + Target/提案图 | 2.986 | [2.685,3.229] |
| 当前EAGLE v4 KV复用 + Target/提案图 | 2.888 | [2.620,3.097] |
| Medusa LowRank + Target/提案图 | 2.454 | [2.276,2.559] |

上述倍率包含执行优化与投机解码的合计收益；不能据此称算法自身达到同等倍率。原Target和各候选在同一轮、相同真实前缀、相同随机种子下直接配对，未用历史倍率相乘。各方法使用自己的随机采样过程，不要求不同算法生成相同token序列。

## 测量与验证

十个开发来源、64/256前缀、seed44101、512-token上限，共20条件×6配置。RTX4090、FP32、TF32关闭、batch1、temperature/top-p均0.9。运行404.864秒，模型权重载入单列。同步外层墙钟包括runtime创建、KV分配、prefill、首次图创建、提案、Target验证、恢复和采样；每配置独立runtime跨请求复用，预热runtime丢弃，各配置轮换顺序。

倍率按来源内配对token/s的log均值，再对来源等权取几何平均；以来源bootstrap10000次给区间。原Target和优化Target各有19次512token、1次191token；MLP为19次512、1次443，其余各20次512。所有实际生成长度均保留。两臂共同满512的MLP/EAGLEv3/v4/Medusa相对原Target为4.709/2.971/2.885/2.457，排序保持；该子集会按生成长度筛选，仅作敏感性分析。

原/优化Target各20/20输出、计数完全相同；四候选与此前提案图主轮各20/20输出、计数相同。全部120条输出满足P/F类型与单调位置规则。原Target没有图调用，其余每次有相应Target图，四候选每次有提案图。两类独立CPU复算共10项通过。直接复制的12份提案及参照源码与上一轮逐字节相同，只有本轮测量和分析入口新增。

严格无新增Target及提案图子集各18对中，四候选相对原Target为4.888/3.133/3.019/2.545。首个建图请求全部为00-0064-00；优化Target为4.145秒，MLP/EAGLEv3/v4/Medusa分别3.307/6.025/6.133/5.568秒，首次创建成本已进入主结果。该子集用于说明复用状态，不代替主表。

EAGLE图覆盖范围仍为T1递推与Feature读出，T>1和长历史走普通执行。MLP只捕获神经Feature提案，CPU Ranker留在原路径。Medusa捕获heads.raw。采样、接受、恢复与Python控制尚未整体图化。本轮未追加数值对照前向，以此前同输入、同输出验证和本轮80条重放作为依据。

## 下一步选择

优先保留强Position Ranker，在低接受率Feature处建立W2分支，取得各分支P2概率与真实Target耗时；随后再判断W4是否值得。Sequoia式验证是现有作者代码最易借用的起点，UniVer为后续多分支与多步验证候选；CaDDTree的成本思想适合预算选择，但其块扩散分解假设不能直接外推到当前MLP。DFlash需新结构、数据和训练，列为投入更大的路线。详细比较、论文与源码依据见[NEXT_METHODS.md](NEXT_METHODS.md)。

当前MLP Position/Feature接受率92.53%/36.97%，EAGLE Feature20.97%，Medusa P1拒绝99.24%，见[轨迹分析](TRACE_SCREEN.md)。完整类型化后缀没有重复，普通动态后缀检索暂时降级。已有多候选CPU接受质量不能代替在线速度，尚未启动树验证、DFlash或WTV训练。

## 操作对象与限制

本机新增源码在`patches/original_target_next_20261021/`，新增结果在本目录；远端隔离目录为`/root/autodl-tmp/gaussiangpt_original_target_next_20261021/`，实际测量源码为`source_run01/`，数据为`main_run01/`。生产工程`/root/workspace/GaussianGPT`、权重和原服务保留。未安装依赖、训练、提交Git或替换生产默认。

复算入口为`python3 patches/original_target_next_20261021/analyze_original_target.py <运行目录> --budget 512 --output-dir <新目录>`。逐次数据、运行命令、图调用和模型路径见[main_run01](main_run01/)，独立复算见[independent_audit.json](main_run01/independent_audit.json)。

本轮仍限10个已用开发来源和单一种子，512-token上限，尚无新独立来源或新完整三维生成评价。新候选的分布一致性、分支KV成本和实际速度均待验证。
