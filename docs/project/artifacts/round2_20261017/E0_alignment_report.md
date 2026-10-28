# E0：已有 EAGLE 的状态与精度核验

日期：2026-10-17。复用 e46_hprev_dual_20ep/last.pt 与原八条验证轨迹；FP32 重算 16,360 行，无新增训练。

## 结果

- 保存张量类型：{'torch.float32': 13}。参数加载、草稿输出与共享输出头运算均为 FP32。BF16 是原训练 AMP 设置，先前“BF16 checkpoint”的称呼不准确。
- 本次完整 softmax 平均重合度：3.7056%。
- 同一批状态应用 temperature=0.9、top-p=0.9 后：2.7763%。
- TF32 matmul：False；matmul precision：highest。

## token 边界

hidden_prev_raw 来自同一 token 的倒数第二个 Transformer 块，与“前一个 token 的最后状态”含义不同。实际监督为 source_index=t，已知 token x[t+1]，预测归一化状态 h[t+1]，再通过共享头预测 x[t+2]。

代码中的 P2F/F2P 标记隐藏状态从哪个 token 跨到哪个 token。P2F=1 对应提案 Position；F2P=2 对应提案 Feature。下表按实际提案类型命名。

| 实际提案 | source index 区间 | 行数 | softmax 重合度 | top-p=0.9 重合度 |
| --- | --- | ---: | ---: | ---: |
| Position | 0_63 | 256 | 6.915% | 7.120% |
| Feature | 0_63 | 248 | 11.540% | 8.864% |
| Feature | 64_255 | 768 | 9.074% | 6.819% |
| Position | 64_255 | 768 | 2.271% | 2.368% |
| Feature | 256_1023 | 3072 | 6.722% | 4.791% |
| Position | 256_1023 | 3072 | 0.613% | 0.660% |
| Feature | 1024_plus | 4088 | 5.816% | 4.054% |
| Position | 1024_plus | 4088 | 0.238% | 0.252% |

## 范围与后续

已有策略等价与缓存数值一致性记录仍作为历史正确性证据；本次测的是 teacher-forced 提案质量，没有新增 EAGLE 原权重在线速度结果。表中 source index 为轨迹位置，不能称为独立在线前缀长度实验。

原 EAGLE 的草稿 RoPE 使用一维递增索引；Target 隐藏状态已包含原三维位置处理。这里确认实际实现，没有证明一维草稿位置方式导致性能下降。首个有效转移样本数量有限，不能据此评价任意长前缀下新建缓存的首步性能。

B1-E 另行实现共享 Position 的小型单步特征预测，使用两 token 因果块读取历史 Feature 与候选 Position；它覆盖特征预测机制，不代表完整 EAGLE 复现。递归、动态树和多层融合暂未加入。
