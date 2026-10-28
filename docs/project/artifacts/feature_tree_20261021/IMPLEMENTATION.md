# Feature W2 实现与数值范围

沿用现有 Position Ranker、step12000 Feature MLP 和 Target 权重。W1 提出一个 Feature，W2 提出两个不同的 Feature；没有新增训练。两个分支共用 Position 提案 P2，但分别使用自己的 Target P2 概率、后续 Feature 概率、隐藏状态与 KV。

## 候选与精确采样

第一候选来自既有温度与 top-p 处理后的 q。后续候选从 q 中移除已选 token 并重新归一化；原支持全部用完时，从尚未选择的 token 中均匀采样。每个候选都保留其实际抽样分布 q_j。

当 P1 被接受后，Feature 的剩余分布从 r_0=p 开始。候选 x_j 按 min(1,r_j[x_j]/q_j[x_j]) 接受；拒绝后更新 r_{j+1}=normalize(max(r_j-q_j,0))。所有候选都被拒绝时，从最终 r 抽样。这条规则对每个候选使用完整 q_j，避免只减去被拒 token 的点概率。小词表精确枚举覆盖普通分布、支持耗尽、全拒绝和 EOS 类型边界；远端 PyTorch RNG 顺序用例也已通过，共 9 项。

## 树与缓存

物理节点顺序为 `[pendingF,P1,F_a,P2_a,F_b,P2_b]`，逻辑深度为 `[0,1,2,3,2,3]`。注意力只能读取共享历史与自身祖先。RoPE 与 learned position 使用逻辑位置；Position 限制使用该路径最近的 Position。

验证只写暂存槽，不移动缓存逻辑位置。P1 拒绝时提交 `[0]`；Feature 全拒绝时提交 `[0,1]`；某分支的 P2 拒绝或接受时分别提交其前三或四个祖先节点。连续路径省去 KV 搬运，第二分支先暂存再复制到连续位置。恢复调用从提交后的位置继续。图捕获失败会直接报错，不能静默切换执行方式。

## 已测数值范围

`numeric_run03` 保留原逐元素容差失败：840 项中 464 项通过，368 个隐藏状态比较和 8 个 KV value 比较未通过；320 个 Target logits 比较全部通过，最大 policy TV 为 9.0078e-7，支持集相同。该轮尚未给出隐藏状态的相对误差与 MLP q 对照，不能据此直接认定隐藏状态相同。

`numeric_run05` 使用逐节点 RoPE 实现，取同一真实场景的 64/256-token 前缀，覆盖 W1/W2、普通执行与图重放、每条分支的提交长度 1–4 和不同合法 token 的缓存续接。每个分支另有普通 Target 与相同显式 BMM 因果前向参照；实际 BMM 图调用数和无替代执行的状态均写入原记录。真实 MLP 对照使用相同 pending Feature、合法下一 Position 和节点隐藏状态。

该轮 1,742 项重放观察全部满足相应数值要求。Target logits 保留逐元素 atol/rtol=2e-5，同时要求 policy TV≤1e-5 和支持集相同；MLP q 要求全有限、归一化误差≤1e-5、TV≤1e-5 和支持集相同。隐藏状态与 KV 要求全有限，整体相对 L2 与最大绝对误差除以参考张量最大绝对值均≤2e-6；逐元素 allclose 单列诊断。最大概率 TV=1.15744e-6，相对 L2=1.63893e-6，最大相对幅值误差=1.94737e-6；逐元素 allclose 仍有 440 项未通过，记录完整保留。

这些是两个有限前缀上的数值证据，未覆盖任意输入、纯 BOS 或窄注意力窗口。重放观察含相同分支在不同提交路径下的重复运行，不能当作独立场景样本。概率原张量保存在远端 `numeric_run05/raw_policy_pairs.pt`，独立 CPU 复算另存。

批量 RoPE 版只将逐节点 `rope_embedder` 调用与拼接合并成一次调用。`rope_batch.json` 的 CUDA 六组形状均逐位相同；六节点的普通准备外层均时约 0.268 ms→0.035 ms。这是局部算子成本，完整收益以第二轮解码为准。该版本的 `numeric_run06` 再次通过相同 1,742 项，最大误差摘要与 run05 相同；`numeric_audit_run06.json` 从原张量独立重算 688 条 Target 与 160 条 MLP q，12 项概率审计通过。隐藏状态与 KV 的相对误差没有独立原张量复算。

## 运行位置

本机源码：`patches/feature_tree_20261021/`。首轮数值与解码使用远端 `/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run03/`；批量 RoPE 版使用 `source_run04/`，运行数据在同一根目录下。生产主工程为 `/root/workspace/GaussianGPT`；本轮只在隔离目录运行候选。
