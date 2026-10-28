# 空间模型完整解码测速已完成

2026-10-20用户要求的真实比较全部结束：180次、60完整配对、10既有开发来源、两种前缀、三新种子、512-token预算；同RTX4090、原Target、FP32最高精度和0.9/0.9采样。无新训练。

唯一远端结果为/root/autodl-tmp/gaussiangpt_spatial_online_20261020/run01。20:13:23.661至20:29:00.405，总936.744238秒；pipeline complete，GPU结束后0%/2MiB。余下额度未使用，不重复启动。

原Ranker+12k MLP为123.7590 token/s，空间S3为119.3687，原Target为73.4555。来源等权旧PF/Target=1.681580，S3/Target=1.635601。S3/旧PF=0.972657 [0.952074,0.993289]；共同满512的58组覆盖10来源，比例0.961998 [0.945646,0.977927]。空间Feature接受率35.8153%，旧34.5251%，本轮没有额外提速。

主代理独立复算180实际序列、PF/Position/EOS和计数通过，全部速度点估计及区间一致1e-12。两个Luna复核报告已完成。12份现用生产源码与运行前逐字节一致，SOURCE_PRESERVATION.json记录具体文件。原权重、源码和历史保留。

先读00_READ_ME_FIRST.md、DECISION.md、CUDA_PARITY.md、TIMING_REVIEW.md、INDEPENDENT_REVIEW.md和PRIMARY_VALIDATION.json。当前停止扩展S3训练与临时调参。压缩包及原生Chrome提交状态见WEB_SUBMISSION.md，不重复上传。

原网页对话：
https://chatgpt.com/g/g-p-6ab0f2354ef88191af84c1dd97a89d90-gaussiangpt/c/6ad1b79f-7bc4-83e8-a9d1-d41896fb2aed
