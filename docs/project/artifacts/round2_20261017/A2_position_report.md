# A2：历史 Feature embedding 对 P1 的作用

日期：2026-10-17。目的：把已知历史 Feature 直接送入 P1 头，评价无需额外 Target 前向的条件提案。该实验覆盖有限数据与训练预算，未进行完整方法复现。

## 实现与训练

两个 P1 头都读取上一 Position 的归一化 Target 状态，经 rank-466 残差变换与共享 Position 输出头预测下一位置。条件版本增加可训练的历史 Feature embedding。位置递增限制在归一化概率之前生效。没有读取额外 Target Feature 后状态，也没有未来位置状态输入。

复用同一 40 条轨迹与完整轨迹划分、16,348/2,044 个位置监督行；10 轮、AdamW lr=1e-4/wd=0.01、seed 20261018，监督为实际下一 Position 的 CE。按验证 top-1 选择权重。两个模型参数量不同，不能把差异完全归因于单一条件变量。

| P1 头 | 可训练参数 | 最佳验证 top-1 |
| --- | ---: | ---: |
| position_state_only | 954,368 | 17.56% |
| with_pending_feature | 1,538,304 | 15.95% |

## 448 个相同真实上下文

| 方案 | P1 重合度 | top-1 命中 |
| --- | ---: | ---: |
| position_state_only | 1.610% | 1.786% |
| with_pending_feature | 1.682% | 1.562% |
| changed_historical_feature | 2.032% | 1.786% |

changed_historical_feature 在推理时把历史 Feature ID 加 113 后按词表取余，仅用于观察输入敏感性。真实 Feature 输入没有表现出明确优势；不能把 A1 中“额外处理 Feature 后隐藏状态”的大幅改善归结为简单增加 embedding 即可获得。

## 小规模在线筛选

从原开发集前三个来源取 64-token 前缀、单运行种子、256-token 上限，四方法共 12 条测量。新 P1 替换原 Medusa-style P1，F1/P2 继续使用原独立头；完整概率与接受/残差验证流程保持一致。计数统一包含自然 EOS。

| 方法 | 运行数 | 微平均 token/s | P1 接受/已验证 |
| --- | ---: | ---: | ---: |
| target | 3 | 74.384 | — |
| original_medusa | 3 | 61.535 | 4/292 |
| position_state_only | 3 | 62.489 | 13/288 |
| with_pending_feature | 3 | 61.690 | 6/291 |

三条来源测量只作筛选，不给正式区间或跨场景优劣结论。本次条件版本未改善完整路径速度，暂未扩大独立 P1 训练、添加候选重排器或进入正式三维确认。

## 尚未解决的问题

训练域与真实域差异仍大；历史 Feature 的有效表示、是否复用 Target token embedding、监督规模及优化预算尚未分开评价。本实验不证明 Feature 条件无效，也没有确定低 P1 接受率的唯一原因。当前继续保留空间 Ranker，避免用额外 Target 前向掩盖提案成本。
