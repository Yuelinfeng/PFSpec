# 同前缀成本实验执行状态

2026-10-21。本轮全部完成，保留静态 W1，未启动外部测试或在线预算选择。最终结果见 [完成报告](RESULTS.md)。

## 已完成

- 相关工作有界比对：九篇原始论文、一篇综述，见 `NOVELTY_REVIEW.md`。通用进展/成本目标和学习式动态选树已有研究，不据此主张新的一般定理。
- 隔离单轮探针：完整 P/F 前缀上的 K1/W1/W2，首版及最终版各 32 项 GPU 对照通过。最终版为远端 `source_run03/`；对应 `smoke_run03.json`。
- 采集程序 12 行真实烟测完成，独立 JSON 计数复算通过，见 `matched_smoke_run02/` 和 `smoke_audit.json`。
- 原 `source_run02/` 的采集入口导入错误保留于 `matched_smoke_run01.log`，该次未采到样本。`source_run03/` 改用实际提供该函数的模块。

## 已结束的正式测量

远端 `/root/autodl-tmp/gaussiangpt_recovery_model_20261021/dev_run01/`。39 个真实前缀状态、10 个来源、8 个种子、K1/W1/W2、3 个计时重复，2808/2808 行完成，诊断计时关闭。另 `components_run01/` 为十来源20前缀、相同八种子、三个动作的一次分项测量，480/480 行完成。两组 `pipeline.json` 均为 complete，不重复启动。

采集代码位于同根目录 `source_run03/`，通过 `--source /root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04` 引用已测 W1 运行时；这两个路径用途不同。主工程 `/root/workspace/GaussianGPT`，解释器 `/root/miniconda3/envs/gaussiangpt/bin/python`。

## 结论与边界

两矩阵独立复算通过，分项与主测480/480输出相同。来源留出恢复模型/W1为0.987458×，与分别拟合R/C的39个动作相同；直接R/C为1.010234×且来源区间含1。正式统计为 `dev_analysis_v2.json`，旧版随机抽样问题保留在旧结果并已在最终代码处理。八模型点估计独立复算相符。

没有稳定开发收益，因此未启动未参与本轮选择的外部来源或在线实现。BOS-only 输入不在本次探针范围，所有前缀至少一对且长度为偶数；本次无EOS。生产、权重和已有服务保留，没有新神经网络训练。最终六文件源码为远端 `source_final_run04/`，实测三文件与本机逐字节相同。结束时GPU2MiB/0%，HEAD031d916、4181条工作区记录，原TensorBoard/JupyterLab保留。
