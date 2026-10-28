# GaussianGPT 联合 Position/Feature 草稿训练说明

日期：2026-10-06（Asia/Shanghai）  
用途：给第一次参与训练的同学说明本轮联合 P/F 草稿如何训练、为什么这样训练，以及如何判断结果是否值得接入解码器。

2026-10-08 阅读提示：本文为此前教学方案，候选长度、损失系数与 A/B/C/D 名称不等同于 S13 实际配置。S13a/b 与真实联合结果已记录；当前暂缓新增实验，见 [阶段工作汇报](STAGE_REPORT_2026-10-08.md)。未来方案以 [联合训练计划第 17 节](JOINT_DRAFT_TRAINING_PLAN.md#17-s13-结果后的-s14-更新方案参考研究报告) 为参考，恢复实验后再决定实施范围。

## 1. 我们要训练什么

GaussianGPT 的 token 按下面的顺序排列：

```text
[Position_0, Feature_0, Position_1, Feature_1, ...]
```

Position token 说明一个 Gaussian 位于什么位置，Feature token 说明这个位置对应的离散场景特征。

现有大模型称为 Target，较小的 D8 称为 Draft。Target 负责最终生成分布，Draft 负责提前猜测一小段 token。若 Draft 猜得足够接近 Target，Target 可以一次处理多个候选 token，从而减少逐 token 计算。

本轮目标是训练一个可以连续提出：

```text
[P1, F1, P2, F2, P3, F3, P4, F4]
```

的 Draft。最终要看的结果是完整生成速度，同时要求 P/F 接受过程保持 Target 的原始概率分布。

## 2. 直接结论：从已有 D8 继续训练

本轮采用下面的路线：

```text
已有较好的 F-aware D8
        ↓
加入 Target 和 Draft 生成的真实前缀
        ↓
同时蒸馏 Position 和 Feature 分布
        ↓
从一对 P/F 逐步增加到四对 P/F
        ↓
用真实联合解码测量接受率和速度
```

不建议从头训练一个全新的 Draft，原因有三点：

1. 现有 D8 已经有较高的 Position 接受率，可以保留这部分能力；
2. 现有 F-aware 训练已经提高了 Feature 分布重合度；
3. 新一轮真正需要补充的是 Draft 自己生成前缀后的表现，以及连续 P/F 候选的接受情况。

GaussianGPT 原论文的联合 teacher-forcing 训练仍然作为基础。它负责让共享 Transformer 学会 P/F 交错序列。本轮在这个基础上加入更接近实际解码过程的蒸馏数据和目标。

如果记录中的 F-aware D8 权重无法使用，先按原论文方式训练一个联合 P/F D8，再执行本文后续步骤。两种情况下都保持相同的词表、序列化、3D RoPE 和 token 类型限制。

## 3. 为什么原论文训练还不够

原论文训练时，模型通常看到真实前缀：

```text
真实 P0 → 真实 F0 → 真实 P1 → 真实 F1
```

实际联合 Draft 会看到自己刚刚提出的 token：

```text
已提交历史 → Draft P1 → Draft F1 → Draft P2 → Draft F2
```

这两种前缀会逐渐产生差别。特别是 Feature 会影响下一次 Position，所以一个早期 Feature 偏差可能继续影响后面的 Position。

因此训练时必须让 Draft 在自己的 P/F 前缀上接受 Target 的软分布指导。只在真实 token 前缀上训练，无法充分覆盖联合解码中真正困难的状态。

## 4. 每条训练样本如何产生

固定 Target 和 VQ-VAE 参数，只更新 Draft。每条样本从一个已经提交的历史 (H) 开始：

```text
H = [P0, F0, P1, F1, ...]
```

然后让 Draft 逐 token 产生候选：

```text
第 1 步：Draft 提出 P1
第 2 步：在 P1 条件下提出 F1
第 3 步：在 P1,F1 条件下提出 P2
第 4 步：在 P1,F1,P2 条件下提出 F2
```

在每个位置保存：

```text
Draft 分布 q_j
Draft 采样出的候选 y_j
Target 在同一个候选前缀上的分布 p_j
token 类型：Position 或 Feature
```

第 j 步的两个分布分别是：

\[
q_j = Q_D(y_j\mid H,y_{<j}),
\]

\[
p_j = P_T(y_j\mid H,y_{<j}).
\]

关键要求是 Target 和 Draft 必须看到同一个前缀。计算 Feature 分布时，条件中必须使用 Draft 提出的 Position：

\[
p^F_j = P_T(F_j\mid H,P_j^{draft}).
\]

这样训练得到的 Feature 分布才对应实际联合解码。

## 5. 训练数据的三种来源

第一类是原始 tokenized scene。它保持模型原来的生成能力，防止继续训练后忘记基础序列规律。

第二类是 Target 轨迹。Target 从真实历史出发生成 P/F 序列，Draft 学习这些较稳定的条件分布。

第三类是 Draft 轨迹。当前 Draft 自己生成候选，Target 在同一候选前缀上提供分布。它专门覆盖 Draft 自己犯错以后会进入的状态。

第一轮可以使用以下比例作为起点：

```text
20% 原始 tokenized scene
40% Target 轨迹
40% Draft 轨迹
```

这只是实验起点。最终比例要由真实 P/F 接受率和速度决定。

遇到 Target 拒绝 Draft 候选时，还要保存拒绝位置和之后真正提交的前缀。后续训练使用真正提交的 Position/Feature 条件继续展开，不能把已经被舍弃的后缀当作真实历史。

## 6. 模型结构如何安排

第一版保持现有 D8 结构：

```text
共享 D8 Transformer
        ↓
现有输出层
        ↓
按当前位置使用 Position/Feature 合法 token mask
```

这相当于一个共享主干，同时学习两种 token。Position 和 Feature 不需要训练成两个完整模型，因为下一步 Position 需要读取前一步 Feature，两个任务必须共享同一段历史表示。

如果只改变训练数据和损失后，Feature 接受率仍然偏低，再测试轻量类型分支：

```text
h_shared
   ├─ Position 轻量分支
   └─ Feature 轻量分支
```

两个分支只负责最后的类型专用输出，主干和历史状态仍然共享。分支从零贡献开始，使新模型的初始分布接近旧 D8，便于比较训练前后的变化。

## 7. 损失函数用直白的话解释

我们希望 Draft 的概率分布接近 Target，而不是只让 Draft 猜中一个最高概率 token。

Position 分布损失：

\[
L_P = KL(p_T^P\|q_D^P)
\]

Feature 分布损失：

\[
L_F = KL(p_T^F\|q_D^F)
\]

第一轮总损失可以写成：

\[
L = \lambda_P L_P + \lambda_F L_F + \lambda_{CE}L_{CE}.
\]

其中：

- (L_P)：让 Draft 的 Position 选择接近 Target；
- (L_F)：让 Draft 的 Feature 选择接近 Target；
- (L_{CE})：保留原始 token 的交叉熵能力；
- (lambda_P,lambda_F,lambda_{CE})：控制三类目标的权重。

建议从下面的权重开始：

```text
lambda_P = 1.0
lambda_F = 1.0 或 2.0
lambda_CE = 0.1～0.5
```

Feature 接受率是当前短板，可以先测试 `lambda_F=2.0`。如果 Position 接受率下降，再比较 `lambda_F=1.0`。

第二阶段加入接受率相关的辅助项：

\[
Overlap(p,q)=\sum_v\min(p(v),q(v)),
\]

\[
L_{acc}=1-Overlap(p,q).
\]

该量越小，Target 和 Draft 的概率质量越接近，单个候选被接受的机会通常越高。它应当作为辅助目标，不能取代交叉熵和 KL。

训练时的概率计算使用合法 token 集合和 temperature=0.9。部署评价再使用完整的 temperature=0.9、top-p=0.9 策略。训练软分布与实际部署分布要分别记录。

## 8. 训练课程：由短到长

联合 P/F 候选过长时，早期错误会连续累积。因此按下面顺序增加候选长度：

| 阶段 | 主要候选长度 | 目的 |
|---|---:|---|
| A | 1 个 token | 先恢复单步 Position/Feature 分布 |
| B | 1 对 P/F | 让 Feature 学会读取当前 Position |
| C | 2 对 P/F | 学会在 Draft Feature 之后继续预测下一 Position |
| D | 4 对 P/F | 训练实际联合验证长度 |
| E | 8 个 token | 只在前面稳定后加入 |

可以按更新次数执行：

```text
1–512 步：一对 P/F 为主
513–1024 步：一对和两对混合
1025–2048 步：一对、两对和四对混合
```

每一阶段都保留短候选，避免长候选训练损害第一步接受率。

## 9. 训练结束后的联合解码

训练完成后，联合推理按下面顺序运行：

```text
Draft 提出 P_i
Target 验证 P_i
    ├─ P_i 接受：Draft 再提出 F_i
    │              Target 验证 F_i
    │              F_i 接受后提交 P_i,F_i
    │              F_i 拒绝时由 Target 采样 Feature residual
    └─ P_i 拒绝：由 Target 采样 Position residual
                   在新 Position 条件下重新取得 Feature
```

第一次拒绝后，后面的 Draft 候选全部丢弃。只有 Target 接受的 token 或 Target residual 采样得到的 token 才能写入最终历史。

对每个候选 token 使用：

\[
\min\left(1,\frac{p_T(y)}{q_D(y)}\right)
\]

来决定是否接受。拒绝时从 Target 与 Draft 概率差形成的 residual 分布采样。这样才能保持 Target 的原始生成分布。

## 10. 四个实验版本

| 版本 | 初始化 | 主要前缀 | 作用 |
|---|---|---|---|
| A | 当前 F-aware D8 | Target 轨迹 | 判断继续蒸馏能带来多少收益 |
| B | 同一 D8 起点 | Target/Draft 混合 | 解决训练前缀与部署前缀的差别 |
| C | 同一 D8 起点 | Draft 轨迹为主 | 强化错误状态和拒绝状态 |
| D | C 的最好权重 | C + overlap 辅助项 + 轻量类型分支 | 进一步提高 P/F 接受 |

A、B、C 要从同一份 D8 权重开始，并使用相同训练步数、有效监督数量和采样策略。D 只有在 C 已经提高真实 Feature 接受率后才运行。

## 11. 如何判断训练成功

每个版本都要记录：

- Position 实际接受率；
- Feature 实际接受率；
- P/F pair 接受率；
- 第 1、2、4、6、8 个候选仍连续通过的比例；
- 每轮实际提交 token 数；
- Draft 生成耗时；
- Target 调用次数；
- rejection 后的重新采样耗时；
- 完整生成 tokens/s。

Feature overlap 只能说明两个 Feature 概率分布接近。它不能直接代表 Feature 已经被真实联合解码接受。

最终模型的选择顺序是：

```text
先确认概率与 cache 状态正确
再看 P/F 实际接受率
再看连续接受长度
最后看完整生成速度
```

如果 Feature overlap 提高但实际 Feature 接受率没有提高，停止增加 Feature 损失权重。如果 P/F 接受率提高但完整速度仍低于普通自回归，说明 Draft 计算或拒绝处理成本过高，暂时保留 Position-only 路径。

## 12. 本轮推荐执行顺序

```text
1. 读取并确认现有 F-aware D8 权重和对应配置
2. 用旧解码器做零步评价，记录 P/F 分布和速度
3. 用 K=1 的 Target/Draft 前缀开始训练 128 步
4. 确认 loss、梯度和验证指标正常后，继续到 512 步
5. 加入两对候选，训练到 1024 步
6. 加入四对候选，训练到 2048 步
7. 在固定验证前缀上比较 A/B/C
8. 只对最好的版本加入 overlap 辅助项或轻量类型分支
9. 用真实联合解码测量 P/F 接受、cache 状态和完整速度
```

本轮最重要的判断是：Draft 自己产生的 P/F 前缀是否让后续候选更容易被 Target 接受，以及这份接受收益是否足以覆盖 Draft 计算成本。

## 13. 当前证据与已知限制

项目已有记录显示：

- 256-step F-aware D8 的 Position 接受率约为 `98.24%`；
- 同一实验的 Feature overlap 约为 `0.7037`；
- 另一份真实联合运行的 Feature 接受率约为 `42.15%`，pair 接受率约为 `39.84%`；
- 该联合运行的完整速度约为普通自回归的 `0.89–0.92×`。

这些数字来自不同运行，不能拼成一次完整实验。它们说明当前需要改善的是“Draft 自己的 P/F 前缀下的真实 Feature 接受率”，而不是继续单独追求 Feature overlap。

本轮训练方案仍有三个待确认问题：

1. Draft 自生成前缀蒸馏能否提高第二至第四对的接受率；
2. P/F 轻量分支能否在不增加太多耗时的情况下改善 Feature；
3. 联合接受长度提高后，完整速度能否超过普通自回归。

在这些问题得到实验数据前，Position-only 解码仍然是稳定的运行路径，联合 P/F 草稿属于研究版本。
