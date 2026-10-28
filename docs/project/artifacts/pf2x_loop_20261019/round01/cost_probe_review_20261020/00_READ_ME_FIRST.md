# GaussianGPT PF2X Round01 有限真实成本诊断

请先读DECISION.md、FEASIBILITY.md和validation.json，再阅读两份分析JSON与实际逐次数据。本包接续此前61文件的192k最终结果包；本轮固定step12000，不新增训练，不替换生产PF。

所有五阶段完成，总墙钟193.622031秒，统一一小时额度内。480条真实p/q、60次W1、两模式各48条真实P2分支均齐全。配对共4个模式上下文组、实际2个唯一上下文/1来源，限制必须保留。没有同轮Target-only或新完整解码器，所以没有新的速度比或2×结论。

results/pq含实际输出token、统计、逐操作时间与4096维NPZ；results/narrow及wide含分支、形状时间、状态与必要NPZ。analysis分别分析两模式，source含真实诊断源码及相对上一上传包的差异，commands.log和budget.json为实际执行记录，cpu_validation为必要CPU证据。source/changes.patch不包含生产源码修改。

请审阅本包及独立复算报告，然后为PF分离的下一轮选择有明确实际调用收益的方案，给出文件级任务、预算、数学与KV条件、正确性验证、同轮Target/W1对照、来源与长度/EOS分析。当前不自动扩大分支或进入树形实现。统一目标仍是可靠实际超过2×，不混用历史Target计时。
