# 连续残差采样：纯CPU概率验证结果

2026-10-20。实施者为gpt-6-luna/max子代理 `/root/pf2x_evidence_review`，主代理已阅读脚本并独立重跑通过。

新增脚本为 `patches/pf2x_round01/verify_multicandidate_cpu.py`，结果为本目录 `results/probability_cpu.json`。实际命令：

```sh
python3 patches/pf2x_round01/verify_multicandidate_cpu.py
```

小词表用Python `fractions.Fraction` 精确分数计算，覆盖225组p/q1和2125组独立q2基础分布组合。逐分支展开首候选、拒绝后的第二候选及最终残差的输出质量，最终分布与p的最大误差为0，归一化误差为0。

边界包括首候选点质量、p/q支持集不同、排除首候选后无可用质量、首阶段零残差、第二阶段零残差、两次拒绝及极小正残差；极小案例的首、次残差质量分别为 `1/2^60` 与 `1/2^61`，未用正阈值丢弃质量。

4096维部分使用种子20261020生成合成Target与Draft logits，采用Python binary64，两种q2策略均完成分布展开。最大最终概率误差为 `9.540979117872439e-18`，最终归一化误差为0。宽支持q2仅从同一组合成Draft logits构造，不读取Target分布。

结果支持这套理想概率关系及本CPU脚本的计算；实际模型FP32/CUDA、KV、RoPE、EOS和解码状态仍未验证。本轮没有新模型输出和速度数据，不能把这些误差表述为GPU实现正确性或2×证据。

网页端已依据之前的源码审阅撤回直接实现惰性W2的安排。新版完整诊断任务仍在生成。本CPU结果包准备好后，等网页当前回复完成再补交，不中途重复发送。
