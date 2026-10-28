# Round03生成长度、EOS与三维结果

本轮没有新增Token生成、自然EOS评价、Target-only运行、W1运行、VQ解码或三维重建。所有新增计算均为既有R1分布上的CPU离线分析；GPU使用为0，完整网络训练与新解码均未运行。

R1输入资料中有60次既有W1运行，每次预算64 Token：59次生成满64 Token，1次在第5个Token以EOS结束。记录位于 `../round01/cost_probe_raw/run01/collection/w1_runs.jsonl` 与对应Profile中。它们属于原step12000 W1的历史输入资料，只用于描述本次离线数据来源；不能当作Round03空间残差、q校准或邻居混合的长度、EOS或质量结果。

本次的 `cpu_run02/rows.jsonl` 和 `q_outputs.npz` 记录的是分布与重合度，未包含任何新生成序列。留出分数不能说明自然结束长度、三维形状、输出质量或实际解码速度。若本轮后续不再推进，以上项目均应记录为未运行。
