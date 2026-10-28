# 本轮结果入口

2026-10-21。当前Target采用显式BMM和动态CUDA Graph后，既有EAGLE、Medusa、Hydra和D8没有超过Ranker+MLP。十个开发来源、220次512-token主比较中，MLP/同优化Target为1.450584× [1.370351,1.523298]，Ranker-only为1.371252×；EAGLE v3 γ3为0.817906×、Medusa为0.725175×。相对同优化Target的2×尚未达到。

先读[完整结果](RESULTS.md)。所有主倍率都以同样优化的Target为参照；3来源原/优化配对与10来源主比较分开列出，首次图创建计入同步外层计时。共同满预算及双方均未新建图的子集也没有新的方法反超。

## 证据与实现

- [主比较summary](optimized_main_run01/summary.json)、[逐行原始记录](optimized_main_run01/records.jsonl)、[独立审计](optimized_main_run01/audit.json)、[图诊断](optimized_main_run01/runtime_diagnostics.json)、[计数汇总](optimized_main_run01/counter_summary.json)。
- [同轮原/优化配对summary](paired_pilot_run01/summary.json)、[独立审计](paired_pilot_run01/audit.json)，11法各6/6输出与计数一致。
- [方法与权重清单](METHOD_INVENTORY.md)、[适配语义](METHODS_DESIGN.md)、[运行数值范围](RUNTIME_DESIGN.md)、[计数口径](COUNTER_MEANINGS.md)、[图复用解释](RUNTIME_INTERPRETATION.md)。
- [计划](PLAN.md)、[实际状态](EXECUTION_STATUS.md)。新源码为本机patches/target_methods_20261021，远端隔离结果为/root/autodl-tmp/gaussiangpt_target_methods_20261021。

LFQ纯BOS18次已完成，但原/图输出重放未一致，不能把它的速度作为已验证结论。同输入同历史复核run02已完成：213次Target图重放，最大policy TV0.012678、BF16草稿q TV0.021485，均有概率支持集变化，未达到真实前缀数值验证范围。具体算子与首次抽样分歧尚未确定，不继续扩大本轮。读[审计](lfq_bos_run01/audit.json)、[数值记录](bos_replay_verify_run02/metadata.json)、[数值解释](BOS_REPLAY_RESULT.md)和[实际状态](EXECUTION_STATUS.md)，勿覆盖失败证据或重复训练。

本轮沿用项目已有适配与权重，没有新增训练，没有替换生产源码和权重，没有安装依赖或提交Git。
