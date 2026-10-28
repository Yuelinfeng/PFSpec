# 组合验证复核

实测快照 `raw/combined_verify_run01` 收录 6 个上下文、36 条动作记录，覆盖 2 个来源和 prefix 0、64、256；6 个动作在每个上下文各有一条记录。JSONL 每行均可解析，所有记录均标记序列有效。

逐行比较确认，W1 与 rescue、W1 与 combined weight 0、fast reuse 0.25 与 combined 0.25、combined 0.25 首次与复跑，在 6 个上下文里的生成 token、EOS 标记和语义计数都相同。完整运行计数可以不同，因为替代 Position 节点会改变实际评估的节点数和恢复调用数。

Position rescue 的恢复调用差额与命中数逐上下文相等。各动作累计如下：

| 动作 | recovery_calls | position_rescue_hits |
|---|---:|---:|
| w1 | 22 | 0 |
| rescue | 16 | 6 |
| combined0 | 16 | 6 |
| fast25 | 28 | 0 |
| combined25_first | 21 | 7 |
| combined25_repeat | 21 | 7 |

summary 中的 26 次命中是 rescue、combined0、combined25 首次和 combined25 复跑四组记录的合计。

实际图前向计数由每条记录的 runtime delta 汇总，并与各臂末条记录的累计值交叉核对：

| 执行臂 | Target runtime | Proposal | Tree | Rescue |
|---|---:|---:|---:|---:|
| w1 | 31 | 132 | 132 | 0 |
| rescue | 25 | 132 | 75 | 57 |
| fast_reuse | 38 | 131 | 131 | 0 |
| combined | 87 | 394 | 223 | 171 |

四臂均有真实 Target 与 Tree 图前向；rescue 与 combined 也各有独立 Rescue 图前向。总数分别按记录汇总，末条累计统计一致。

混合概率计算的每次调用标量记录显示：

| 动作 | 调用数 | 最大 TV | 最大质量差 | 最大质量误差（普通/lerp） | 支持集变化 |
|---|---:|---:|---:|---:|---:|
| fast25 | 131 | 1.4e-08 | 1.19e-07 | 3.58e-07 / 2.38e-07 | 0 |
| combined25_first | 131 | 1.85e-08 | 1.19e-07 | 3.58e-07 / 2.38e-07 | 0 |
| combined25_repeat | 131 | 1.85e-08 | 1.19e-07 | 3.58e-07 / 2.38e-07 | 0 |

三组各 131 次；支持集均相同。TV、质量差和每个输入的质量误差由保存的逐次标量记录重新汇总。概率向量本身未保存在该目录，因此没有从原张量重算。

`verify_combined_executed.py` 是当时实际运行的快照。与当前 `verify_combined.py` 的差别在于补报 Target、Tree、Rescue 图前向计数，并要求 6 个上下文、36 行、6 组比较齐全且 Tree/Rescue 确实执行图前向。解码器文件和采样路径没有变化；本审阅按当时保存的每行计数和末条累计值独立复核。

复核状态：通过。机器可读结果见 [combined_verification_audit.json](combined_verification_audit.json)。
