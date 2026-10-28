# Round03实际验证

真实环境为既有远端Python3.10 / NumPy2.2.6，CUDA_VISIBLE_DEVICES为空；代码不导入Torch。本机没有NumPy，未安装依赖。所有模型原文件保持原样，仅向远端独立Round03/source目录传入CPU工具。

首次feature接口自测、离线合成自测、NumPy整数JSON专项自测均通过。覆盖六方向与边界、完整P/F和EOS、因果邻居可见性、极小正质量、关闭残差时原q逐项保留、所有允许结构的等参数量，以及真实NumPy拟合与推理。

480行首次计算的退出码为1：entropy_group_counts.low/high的NumPy布尔求和产生int64，最终JSON写入失败。已生成rows.jsonl、q_outputs.npz和fit_parameters.npz。首次代码和错误日志保存在failed_summary_write_01；首次部分产物保留于远端cpu_run01。只增加int转换和专项自测，所有数据、训练参数、来源、选择网格及随机seed不变。

技术重放输出cpu_run02，退出码0。validate_cpu_results.py逐项比较首次和重放的全部NPZ数组，全部相等；逐行JSON字节相等。再从保存概率独立归一化、计算480行重合度和三组逐来源均值，最大差1.1102230246251565e-16；停止判定一致。完整结果见primary_validation.json。

这是同一设置的技术重放，未据第一次结果改变方法。summary中的一次留出评分描述每次程序内部流程，不能解释为本轮实际仅计算一次。CPU耗时记录包含失败和重放，不省略失败运行。

未验证：CUDA/FP32在线空间头、GPU新增提案延迟、完整候选解码、同轮Target-only速度、自然EOS长度分布、三维质量或2×。CPU探针使用float64拟合与概率计算；上述缺项没有用历史结果替代。
