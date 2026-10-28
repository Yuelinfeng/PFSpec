# GaussianGPT：192k与五节点最终结果，供网页端接续R1

## 本包的请求

原192000步训练与486次评价已全部结束，请依据真实结果确认后续有限诊断唯一开发Feature MLP参照。我们建议沿用原step12000；12k/24k/48k/96k/192k同轮接受率和速度没有可靠额外收益。请明确权重路径或精确步数，继续现有Round01-R1的最多480条p/q、总计96条分支条件、2/4-token成本诊断；新增训练0，总预算最多1 GPU小时。无需再制定完整惰性W2实现或36/180矩阵。

## 阅读顺序

1. `feature_extend192/FINAL_RESULTS.md`、`final_validation.json`、`FINAL_EVIDENCE_REVIEW.md`：完整结论、主代理验证和Luna复算。
2. `feature_extend192/final_raw/evaluation/summary.json`、`records.jsonl`、`plan.json`、`verification.json`：全部486次真实结果，含输出token、同步外围时间、来源区间与计数。
3. `feature_extend192/final_raw/training/history.json` 与 `analysis/`：121节点固定验证和31节点稀疏在线曲线。两类来源、种子和预算不同，分别解释。
4. `round01/`：当前概率证明、同长度调度、源码对应、R1补充说明和最新CPU结果；`research_code/`为实际新增代码全文，`ADDED_RESEARCH_CODE.patch`为对应新增差异。
5. `source_current/`：当前13份远端源码只读副本，包含此前11份以外新增的Ranker依赖。

## 已确认的结论与范围

全部486次完成，186条曲线、300条五节点主比较、五节点各60条。精确96k/192k的Feature接受率35.7257%/35.2191%；相对12k来源速度0.99894× [0.98029,1.01723] / 0.99895× [0.98063,1.01620]，没有可靠额外提速。固定验证重合度上升很小，没有突然跃升。

本轮没有Target-only对照，不能用本轮PF吞吐和历史Target拼成新加速比。更长训练结果只用来确定Motivation与后续参照；尚未达到2×。不同Feature权重的输出长度/EOS已列出，相同预算与相同实际输出长度的描述性结果均支持上述判断。精确192k与验证选择step190800分别保留，五节点使用精确权重。

R1数学补充：第二接受子集服从normalize(min(r1,q2))，完整第二阶段混合才服从r1。无支持q2的首拒事件贡献零，合法触发率按P(P1接受)×E[R1×L]处理；不能先丢弃无支持事件再乘全体首拒率。来源×前缀缺测保留缺失。有限分支乐观节省是点估计，不能称严格上界或自动停止依据。来源区间只重采样来源级代入均值，不包含上下文和分支抽样误差。

概率精确2207组p/q、60463组三元组合误差0，4096维binary64最大概率误差9.54e-18；44条调度轨迹通过。2/4-token默认4条件配对CPU模拟为72次额外前向、76次截短；真实CUDA/KV/RoPE和性能待验证。完整合成NPZ输入管线已于03:48在既有远端Python3.10.21/NumPy2.2.6、CUDA不可见的CPU环境通过，覆盖零残差、无支持、极小正质量、混合来源覆盖、分支权重和配对/恢复耗时；初次合成KV时间字段不一致已仅调整测试输入并复跑。记录在 `round01/results/npz_cpu_validation.json`。这些CPU结果不能证明2×。

## 文件与运行范围

原始结果来自 `/root/autodl-tmp/gaussiangpt_feature_extend192_20261019/`；现有Target、VQ、Ranker、原step12000与所有历史权重保持原样。包内没有大权重、原数据、凭据或私钥。新诊断只写独立研究目录，不修改生产解码器；所有加载、预热、分支和测时共用1 GPU小时，不能每个命令各计一小时。

网页原对话：https://chatgpt.com/g/g-p-6ab0f2354ef88191af84c1dd97a89d90-gaussiangpt/c/6ad1b79f-7bc4-83e8-a9d1-d41896fb2aed
