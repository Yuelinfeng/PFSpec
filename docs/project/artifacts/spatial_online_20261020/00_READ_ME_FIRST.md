# N2 S3完整GPU解码比较

180次真实解码、60配对组已全部完成；当前空间版本没有额外速度收益。原Ranker+12k MLP保持速度参照，所有权重和生产源码保留。

本轮起止为2026-10-20T20:13:23.660927+08:00至2026-10-20T20:29:00.405165+08:00，总用时936.744238秒，低于1800秒；余下时间未用于新增实验。GPU结束后为空闲0%/2MiB。

RTX4090，原Target，FP32最高精度、TF32关闭，temperature/top-p均0.9。10个既有开发来源，64/256前缀，种子38101/38102/38103，每次最多512新token，允许自然EOS。旧MLP精确step12000，空间模型固定S3 seed29、3585参数、倍率1.0，无新训练。权重与数据路径见results/plan.json和raw/run01/pipeline.json。

先读DECISION.md，再读CUDA_PARITY.md、TIMING_REVIEW.md、INDEPENDENT_REVIEW.md与PRIMARY_VALIDATION.json。results含完整逐次记录、token序列、来源表、等长表、EOS长度表、区间与总预算。raw保持远端真实原始文件，code为实际运行代码及最终独立复算程序；source_before为原模块只读副本。WEB_PLAN_ORIGINAL.md保存网页原文，IMPLEMENTATION.md列出实际路径和六排列顺序选择。

主统计每配对先取吞吐比例对数，来源内平均，再来源等权几何平均。10000次来源重采样，种子381012。合并吞吐是另一项统计。58组共同满512覆盖10来源，其余两组旧PF分别自然EOS于75与51token；三方法全部实际输出均通过P/F、Position顺序、EOS、计数和参数配对验证。

本次来源已被用于开发；没有独立来源、完整自然EOS生成或三维质量评价，未达到可靠2倍目标。
