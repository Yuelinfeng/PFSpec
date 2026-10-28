# 相关工作与贡献边界

本节提供阅读入口和设计边界，未进行穷尽式新颖性论证。项目已有 32 篇文献资料收藏；本包仅引用与当前方案最直接相关的论文，不复制整个 PDF 收藏。网页端若能联网，请阅读所用版本原文并核实更新；无法联网时明确这一限制。

| 论文 | 原始入口 | 已有内容与本项目关系 |
| --- | --- | --- |
| GaussianGPT | https://arxiv.org/abs/2603.26661 | 稀疏离散三维场景、Position/Feature 序列、3D RoPE 与 Gaussian 输出；本项目保留原 Target 与 VQ |
| Fast Inference from Transformers via Speculative Decoding | https://proceedings.mlr.press/v202/leviathan23a.html | 概率接受与剩余概率抽样，保持 Target 分布的基础 |
| EAGLE | https://proceedings.mlr.press/v235/li24bt.html | 使用 Target 特征的 Draft；不能把状态复用本身当作新的贡献 |
| EAGLE-2 | https://aclanthology.org/2024.emnlp-main.422/ | 上下文相关候选树；本项目需要另论证 P/F 结构与随机采样 |
| EAGLE-3 | https://arxiv.org/html/2503.01840 | 多层特征与训练中的多步模拟；本项目已有 v3 条件训练尝试 |
| Medusa / Hydra | https://proceedings.mlr.press/v235/cai24b.html / https://arxiv.org/abs/2402.05109 | 多预测头与条件化草稿，相关小型结构已在第二轮对照 |
| Sequoia | https://arxiv.org/html/2402.12374 | 候选树结构优化、随机验证与硬件成本；仅添加 P/F 类型标签的贡献强度需要谨慎评价 |
| XSpecMesh | https://proceedings.mlr.press/v306/chen26dn.html | 三维 Mesh 的多头提案、验证与蒸馏；质量保持措辞不自动等于本项目要求的随机分布保持，需逐算法比较 |
| FlashMesh | https://arxiv.org/html/2511.15618 | 利用 Mesh 层级与结构先验组织推测生成；已有三维结构化方法，需明确 Gaussian P/F 的额外研究内容 |
| DistillSpec | https://arxiv.org/abs/2310.08461 | Draft 分布蒸馏；分布监督本身属于借用方法 |
| RheoSampling | https://arxiv.org/html/2609.21827 | 近期研究随机候选、树扩展与裁剪概率的相互影响；本项目尚未实现或复现 |

潜在原创内容应集中于：空间条件如何改善低成本 Feature 提案；P/F 条件关系如何决定候选宽度与深度；如何在保持原 Target 分布的前提下，将实际时延用于结构化候选预算。上述主张均待验证。

借用组件应明确署名，例如标准拒绝采样、已有树验证算法、多步候选历史训练。源码融合、批量计算和缓存优化可以是必要实现，但需要测量证明其作用。新增模块数量不能作为贡献强度的替代指标。
