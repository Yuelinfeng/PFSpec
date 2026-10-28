# 三种提案侧专用图结果

三种方法均获得新增执行收益，排序保持：Ranker+MLP最快，为同优化Target的1.489254倍；EAGLE v3接近Target，v4与Medusa仍较慢。同优化Target下的2倍目标未达。接受算法与权重保持原样；这些新增倍率来自提案侧执行优化。

## 完整外层计时

10开发来源、64/256真实前缀、seed44101、每次最多512新token，共20组条件、9执行臂、180次。RTX4090，PyTorch2.8.0+cu129，FP32最高精度、TF32关闭，batch1，temperature/top-p均0.9。来源等权几何平均，按来源重采样10000次得到95%区间。

各执行臂持有独立Target及提案运行时，预热图丢弃。首次运行时建立、KV分配、图创建、前缀处理、提案、验证、恢复、采样和临时hook都进入同步外层墙钟；权重加载单列。整轮历时474.813秒。

| 方法 | 仅Target图 / 同优化Target | 加提案图 / 同优化Target | 提案图新增倍率及95%区间 |
| --- | ---: | ---: | --- |
| Ranker+MLP | 1.439277 | 1.489254 | 1.034724 [1.025808, 1.042434] |
| EAGLE v3 γ=3 | 0.830013 | 0.960764 | 1.157529 [1.129139, 1.176499] |
| EAGLE v4 KV复用 γ=3 | 0.788927 | 0.927083 | 1.175119 [1.138767, 1.198895] |
| Medusa LowRank | 0.761582 | 0.777153 | 1.020445 [1.013615, 1.026278] |

加提案图后相对共同Target的95%区间：
- Ranker+MLP：[1.412520, 1.564826]。
- EAGLE v3 γ=3：[0.922285, 0.998533]。
- EAGLE v4 KV复用 γ=3：[0.891677, 0.962514]。
- Medusa LowRank：[0.766174, 0.785551]。

Target有1条191-token自然EOS；MLP两臂同一条443-token自然EOS；其他候选均满512。共同满512子集仍保持同一排序：MLP18对为1.505781，EAGLE v3/v4和Medusa各19对为0.956621/0.922968/0.776409。

## 复用图与首次创建

严格复用子集要求该方法的两执行臂及共同Target三臂均已建立运行时、本请求无新增Target图、无新增提案图且实际使用提案图。每种方法均18对、9来源。

| 方法 | 提案图新增倍率 | 相对同优化Target |
| --- | ---: | ---: |
| Ranker+MLP | 1.038206 | 1.516174 |
| EAGLE v3 γ=3 | 1.170915 | 0.974637 |
| EAGLE v4 KV复用 γ=3 | 1.192539 | 0.938478 |
| Medusa LowRank | 1.022968 | 0.781818 |

EAGLE v3在严格复用子集相对Target为0.974637，95%区间[0.941810,1.007190]，与Target的差异尚不明确。各方法首次图创建请求的新增倍率依次为0.975148/0.948038/0.920566/0.977092，均低于1；完整结果表已包含它们。每种只有1条首次创建请求，不作总体启动时间估计。

## 专用图覆盖与验证

| 方法 | 新增图数量 | 实际神经图调用 | 保留的普通执行 |
| --- | ---: | --- | --- |
| Ranker+MLP | 1 | 3789 | Ranker、采样、接受与恢复控制 |
| EAGLE v3 γ=3 | 5 | 12886次递推 + 4308次Feature读出 | 长历史及T>1，4308次 |
| EAGLE v4 KV复用 γ=3 | 5 | 13788次递推 + 4308次Feature读出 | 长历史及T>1，3910次 |
| Medusa LowRank | 1 | 5112 | Ranker、采样、接受与恢复控制 |

MLP捕获FeatureStateDraft.forward；Medusa捕获heads.raw；EAGLE捕获T1 Draft及一维Feature读出头，固定KV临时缓冲按缓存对象与位置同步，连续递推可复用。EAGLE的T>1仍走普通执行，因此当前结果对应这些实际覆盖范围。所有候选无捕获失败或停用图。

最终all_verify_run01合并短测18次：MLP43、Medusa64次同输入前向，EAGLE v3/v4各149/157次递推与51次读出头对照；输出、policy和KV最大差0。EAGLE独立固定输入142操作覆盖128/256/512读桶、T2普通回退、缓存回退后的合法Feature替换；原始tensor独立CPU复算12/12通过。

正式主测四候选各20/20配对输出与计数一致，总80/80；两臂Target调用、图形状及缓存事件一致。主测独立复算9/9通过，主代理从原始逐次记录另算12组倍率全部匹配；EAGLE v3/v4主输出也20/20相同。数值对照仅在前置短测开启，主测速没有额外数值前向。有限输入的逐位一致不能推广至任意输入，已有纯BOS数值限制仍保留。

## 位置、复现与范围

- 本机源码：`patches/proposal_graphs_20261021/`，含evaluate_proposals.py、proposal_adapters.py、mlp_medusa_graph.py、eagle_graph.py、verify_eagle.py、analyze_proposals.py。reference中9份旧执行源码与上一轮副本逐字节一致。
- 远端固定执行源码：`/root/autodl-tmp/gaussiangpt_proposal_graphs_20261021/source_final_run01/`；正式数据：同根目录`main_run01/`，完整命令在pipeline.json。主分析使用本机最终analyze_proposals.py。
- 本机正式原始记录、summary.json、audit.json和independent_ratios.json位于本目录`main_run01/`。数值原始tensor与独立复算位于`eagle_fixed_input_run02/`。
- 同步更新的本机交接文件：`.codex/AI_START_HERE.md`、`docs/project/ARCHITECTURE.md`、`EXPERIMENTS.md`、`RESEARCH_SOLUTIONS.md`、`LOGIC_VALIDATION.md`、`OPERATIONS.md`；`docs/project/PROJECT_STATUS.md`由原脚本生成。
- 原工程`/root/workspace/GaussianGPT`、原权重和默认执行路径保留，HEAD为031d916，开始/结束均4181条工作区记录。未训练、安装、提交Git或操作既有Jupyter/TensorBoard。结束GPU2MiB、0%，本轮GPU进程全部退出。
- 结果限这10个开发来源、一个seed、batch1和最多512新token；未扩独立来源、完整自然EOS或三维质量评价，不作这些范围的结论。

复算命令：

```bash
python3 patches/proposal_graphs_20261021/analyze_proposals.py docs/project/artifacts/proposal_graphs_20261021/main_run01 --summary-output /tmp/proposal_summary_new.json --audit-output /tmp/proposal_audit_new.json
```

失败与过渡试验保留：dense_verify_run01为Medusa数值监测视图维度问题；eagle_verify_run01为Attention接口差异；eagle_verify_run02图创建失败、实际图0，不作图性能结果；eagle_verify_run03为仅T1首版正确性验证。eagle_fixed_input_run01因相对路径工作目录失败，run02已处理。所有开启数值对照的短测速度均不用于主表。
