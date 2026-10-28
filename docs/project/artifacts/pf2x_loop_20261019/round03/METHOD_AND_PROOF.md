# Round03 六向邻域 Feature 探针

## 本轮范围

此文件记录零 GPU 阶段使用的小型 CPU 探针。新增实现位于 `patches/pf2x_round01/pf_round03_feature.py`。它读取 Round01 保存的 R1 上下文和 `raw_logits` / `q1` / `p_f`，为候选 P 查询已提交历史中的六面 Feature 邻居，并允许将可学习 logits 残差与无邻居对照作来源隔离比较。

本机接口自测后，主代理在已有 NumPy CPU 环境对480行R1记录完成一次来源分组分析：288行拟合、96行调参、96行留出评价。留出空间残差相对原 q1 增加0.1084个百分点，简单邻居频率混合增加1.3194个百分点，空间残差未达到3个百分点推进标准，故本轮停止。完整数据见 `cpu_run02/summary.json`；本轮GPU使用为0，没有开展网络训练或生产接入。拟合和推理接口延迟导入 NumPy，实现不导入 Torch，也不会创建 CUDA 上下文。此机未安装 NumPy，本机自测仅运行标准库部分。

## 已提交邻居语义

`CommittedNeighbors.from_tokens(context_prefix)` 接收实际历史 Token 序列，开头必须为 ID 12096。这个 ID 同时用于 BOS 与 EOS：首位解释为 BOS；其余位置只有尾部 12096 才解释为 EOS，且 EOS 前必须有完整 P/F 对。无 EOS 时允许末尾保留一个未配对 P，但该 P 不进入邻居表。只有序列里已经存在的完整 `(P,F)` 对会被记入；因此已确定、尚未写入 Target KV 的真实 F 仍可用，未提交候选和未来 Token 不会被造入表。重复 P、错误交替、非尾部 EOS 均报错。

P 坐标按 `utils/pos_tokens.py` 的行序映射：`P = x*400 + y*20 + z`，每轴范围为 0–19。`batch_features(position_ids)` 依据坐标差逐项查六个方向，顺序固定为 `(-x,+x,-y,+y,-z,+z)`，`delta_xyz` 表示邻居坐标减查询坐标。越界方向返回缺项，不绕到另一侧。查询不重排历史 Token，也不使用 xyz 或 xzy 扫描规则推导空间邻接。

接口返回 `neighbor_position_ids`、`neighbor_feature_ids`、`present` 和 `delta_xyz`，形状依次为 `[N,6]`、`[N,6]`、`[N,6]`、`[N,6,3]`。邻居 Feature 用全局 Token ID 8000–12095；缺项为 `-1`。残差模型也接受局部 ID 0–4095。R1 `context_prefix` 只包含候选 P 之前的实际历史，`p1_id` 单独作为查询位置；候选 F1 不会进入历史表。

## 有限参数探针

`DirectionalResidualProbe(mode="spatial"|"context_only", rank=4, neighbor_groups=8, logit_buckets=16)` 接收相同的 MLP `raw_logits` 和候选 P。两个模式都含 8 维上下文摘要与 48 维辅助输入，通过同形状的低秩隐藏层输出有限分组的 Feature logits 残差。空间模式的 48 维输入按方向与邻居 Feature 分组；对照模式只从候选坐标和 `raw_logits` 构造 48 维确定性上下文展开，不读取邻居列。两边使用相同模型结构与可学习参数数。

参数量为 `6*neighbor_groups*rank + 8*rank + rank + rank*logit_buckets + logit_buckets`。默认配置为 **308 个可学习参数**；受支持的有限配置为 rank `{2,4}`、邻居分组 `{4,8}`、输出分组 `{8,16}`，最大仍为308。输出分组内每类共享同一残差值，因此这是用于筛选的低容量模型，不能表达每个 Feature 类的独立变化。

空间输入把六方向各自的邻居 Feature ID 分成等宽组：4096 类依 `id*neighbor_groups//4096` 编码；无邻居为零向量。上下文摘要由候选 P 的归一坐标及 `raw_logits` 的标准差、顶端值、顶端间距、前8项与前32项均值构成。所有摘要都不读取 Target `p_f`。输出分组将4096类连续切分；默认16组，每组256类共用残差值。结果只适用于这项受限表达，负结果不能排除更细粒度或不同形式的空间条件。

调用 `.fit(raw_logits, position_ids, neighbor_feature_ids, target_p, ...)` 时，`target_p` 只用于拟合交叉熵标签。调用 `.residual_logits(raw_logits, position_ids, neighbor_feature_ids)` 时没有 `target_p` 参数。每次 `.fit` 按对象固定 seed 重新初始化参数；拟合代码默认100步、学习率0.03、L2系数0.001。R1 流程应仅把预先指定的 fit 来源传入拟合；候选配置和停止步数只按 tune 来源选择，留出来源只做一次最终离线评价。拟合交叉熵使用温度0.9的完整 softmax；它只是训练目标，不能替代实际 top-p 后的重合度比较。

## q 与拒绝余量

`apply_feature_policy` 在残差乘数为0且提供 `original_q` 时，验证原 q 的有限性与概率和后逐值复制；不会重新归一化。因此即使 FP32 q 的概率和与1相差少量舍入误差，关闭残差时仍与原 q 每一项相同。启用残差时按原始 MLP logits 加乘数后的残差，再按原温度0.9、top-p0.9规则形成提案分布。Target `p_f` 与 Target policy 不被修改。

对归一化 Target 分布 `p` 与提案分布 `q`，接受重合量为 `α=Σ_i min(p_i,q_i)`。拒绝余量为 `r_i=max(p_i-q_i,0)`，其总量 `M=Σ_i r_i=1-α`；若 `M>0`，拒绝时抽样分布为 `r_i/M`。代码不把极小的正余量截为零；若 `M=0`，返回零向量和0，不生成虚构的拒绝抽样分布。零余量下不得发生拒绝事件。

## 已有工作与结果解释

本项目已有 `docs/project/artifacts/pf2x_loop_20261019/round03/source_before/feature_spatial_delta.py` 空间路线，使用前一位置、状态、历史 Feature 和坐标差；它没有从已提交历史读取六面邻居 Feature。本探针测试的是一个不同且更小的离线输入形式，不代表空间条件化此前从未尝试。

Round01 的同一批480条样本已经用于温度/top-p和邻居频率混合探索。当前 offline 流程按6个拟合、2个调参、2个留出来源组织，只保证当前拟合和调参未用最后两来源；这两来源先前已用于方法选择，不能视为全新留出集。当前分数、按源变化、区间及成本筛选记录在 `cpu_run02/summary.json` 与 `COST_AND_TIMING.md`，有限探针不是数学严格上界。

## 本机验证与待办

本机执行：

```bash
python3 -B -m py_compile patches/pf2x_round01/pf_round03_feature.py
python3 -B patches/pf2x_round01/pf_round03_feature.py --self-test
python3 -B patches/pf2x_round01/pf_round03_feature.py --help
```

标准库自测覆盖 P 编码、固定方向、立方体边界、EOS 与完整配对、尾部未配对 P、历史增加 F 前后的邻居可见性、零残差时 q 原样保留、拒绝余量质量及 `2^-60` 级正质量。NumPy 如可用，自测还会运行极小的实际拟合/批量推理，并验证无邻居对照不受邻居输入影响。本机因无 NumPy，拟合/推理已标记跳过；主代理已在既有环境完成 NumPy接口自测与真实CPU筛选，结果见 `feature_cpu_validation.json` 与 `cpu_run02/summary.json`。GPU使用为0。

此实现不包含 R1 分布读取 CLI、来源拆分、阈值报告、完整网络训练或在线解码。offline Agent 已直接使用本文件提供的邻域与拟合接口，没有另写六向邻居提取。离线留出重合度不等于实际解码提速；本轮已按既定结果停止，不进入 GPU 阶段。
