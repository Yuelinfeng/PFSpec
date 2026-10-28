# 12k与48k Ranker+MLP实际对照

用户最新要求先实际比较两者速度，本任务优先于网页下一轮实验。当前准备执行，尚无新结果。原训练和486次评价已完成；远端RTX4090无计算进程，0%/2MiB。远端主工程HEAD031d916、4181项已有变更保留。

仅更换Feature MLP精确权重：原selected.pt元数据step12000，与旧续训目录48000.pt元数据step48000，两者结构一致；已用CPU读取确认，未初始化CUDA。Ranker、原Target、原Feature head、PF-K2解码、FP32 highest与TF32关闭、温度/top-p0.9保持相同。10个既有开发来源、64/256前缀、seed_base48701/48702/48703，每次512预算，两个权重共120次。运行顺序按上下文交替，完整同步外层解码时间包含前缀、缓存、提案、验证、恢复和采样，加载与预热单列。

原Luna成本代理正准备独立runner；主代理负责源码/参数验收及远端执行。仅新增独立工具与结果目录，不改生产、权重或旧产物。没有新训练或Target-only运行，不混用历史Target时长。总执行上限1800秒，含加载与短验证，超时保留实际结果。

当前SSH复用/tmp/gaussiangpt_research_loop_20261020_reconnect.sock。此次完成前不启动网页下一轮GPU任务，不重复启动本比较。

## 已启动真实GPU比较

2026-10-20 15:06:03（UTC+8）启动，远端进程36440；唯一目录/root/autodl-tmp/gaussiangpt_feature_ckpt_compare_20261020/run01。预热与每权重同seed16-token重放均通过，已进入120次矩阵。原Target/Ranker与两份精确权重已同进程加载，FP32 highest、TF32关闭，实际配置保存于pipeline.json。首条512-token运行3.562714秒，仅用于估计等待，不提前判断速度。

有效SSH复用/tmp/gaussiangpt_research_loop_20261020_reconnect.sock。原目录已有launch.json，禁止重复启动。等待矩阵完整后镜像records/summary/verification/plan/pipeline与运行日志，再以来源复算；只读监控不占用GPU。

## 已完成

全部120次完成，整体494.186408秒。48k/12k来源速度0.994623×，95%区间[0.980059,1.007992]；F接受率35.230457%/35.252955%。共同满512的58配对速度0.995938×。本次没有48k更快的证据，继续保留12k速度参照。完整结果、设置与限制见RESULTS.md，所有原始数据已镜像raw/。原源码和权重保留，GPU已释放。不要重复启动。

项目记忆已更新。结束时未找到原gaussiangpt-2自动化配置文件，未重新创建或启动任何新测量；后续任务按用户指示接续。
