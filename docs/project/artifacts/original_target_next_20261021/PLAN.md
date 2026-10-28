# 原 Target 参照与新投机方法筛选

用户要求将后续主比较设为原 Target。为避免跨轮相乘，另开同轮六配置测量：原 Target、执行优化 Target、Ranker+MLP 提案图、EAGLE v3 提案图、EAGLE v4 KV复用提案图、Medusa LowRank 提案图。后四项均使用已有执行优化 Target；原 Target 使用同一权重和普通执行入口。

沿用十个开发来源、64/256真实前缀、seed44101、512 token预算，合计120次。各配置交错轮换；首次图创建、KV分配、prefill、提案、验证、恢复和采样均计入同步外层墙钟。权重载入单列；runtime跨请求复用。FP32、TF32关闭、batch1、temperature/top_p均为0.9。主指标为来源等权的配对token/s几何平均与来源bootstrap区间，同时保留共同满预算子集和实际EOS长度。

源码：`patches/original_target_next_20261021/`。远端：`/root/autodl-tmp/gaussiangpt_original_target_next_20261021/source_run01/`；结果：同根目录`main_run01/`。生产源码与所有权重保留，不新增训练或依赖。

新方法筛选依据为当前P/F接受计数、已有480行多候选残差分析、论文正文和作者代码。原三名gpt-6-luna/max分别负责论文、作者代码和CPU统计；主代理负责同轮GPU测量、整合与验收。研究过程记录在`/tmp/ggpt_next_spec_20261021/`，最终证据摘要放入本目录。研究候选尚未完成新在线解码实现，论文速度不能写作本项目速度。
