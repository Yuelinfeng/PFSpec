# GaussianGPT 当前方案的初步逻辑验证

## 2026-10-23 动作成本与新选择器链路核验

新增33状态515动作1545重复，cache、共享Target运行对象和捕获输出恢复；12组有/无探针对照一致。成本范围仅限所列操作，不称完整解码最优值。9540组CPU完整/轻量输入等价，280正式记录输出/接受/调用一致，140跨进程配对一致。12在线记录2166决策与CPU重算一致，158次非空备用与实际树节点ID一致。9个仅余2token的终轮按can_tree条件不进入观察流，输出完整。EOS未单独预测、主概率误差与首次图费用均列为模型边界。见[完整报告](artifacts/recovery_value_evidence_20261023/RESULTS.md)。

## 2026-10-23 动态执行与真实逐轮动作验证

[完整记录](artifacts/recovery_value_execution_20261023/RESULTS.md)。13项CPU用例及解码/拟合/在线工具自测通过；1908开发context三模式动作相同、评分误差≤1e-12。252正式GPU记录对W1输出与语义相同，固定0/W1、固定1/旧单备用及三动态版本的执行计数相同；两进程、解码/完整请求交叉比对一致。12条独立在线诊断覆盖202实际决策状态，三组606次逐轮ID/输入对照及1212次评分对照通过，新路径深拷贝和诊断快照均0调用。主代理直接原始trace复算404配对状态通过。14成本上下文/16标签读取实际预热状态，4观察请求与W1输出/调用相同。84NPZ/504PNG有效，微小浮点和像素差异仍存在。

## 2026-10-23 恢复价值关键证据实验完成

[完整结果](artifacts/recovery_value_20261023/RESULTS.md)。20新来源组、64/128前缀、512上限，旧10来源开发、模型8/2留源，参数固定后两进程800解码及400完整三维请求完成。value/单备用解码0.932238 [0.928435,0.936197]，完整请求0.941442 [0.933740,0.947926]；相对概率成本和同数量Ranker区间跨1，未得到来源外强对照优势。单备用/W1来源等权解码1.013946、完整请求1.008566；完整请求累计129.007296→128.991790秒，仅省0.015506秒，累计比1.000120 [0.982643,1.014994]，不概括为稳定端到端净收益。

1200记录、960输出/语义/调用对照、600跨轮与400跨阶段对照通过；400NPZ与2400PNG有效，无空白视图，Gaussian/像素有极小差异。解码Luna独立复算通过；其最终E2E任务额度用尽无交付，主代理从原始记录直接另算四项点值/区间，与主统计完全相同，未将未交付工作计为证据。神经权重和生产入口保留。RTX4090结束2MiB/0%，无计算进程，远端HEAD031d916/4181记录、原权重大小及821/822服务保持。原始记录在本机artifacts/recovery_value_20261023、源码patches/recovery_value_20261023，远端/root/autodl-tmp/gaussiangpt_recovery_value_20261023；matrix_progress为complete，不重复启动。当前动态价值选择只作为负结果和边界，v4正文仍以固定恢复预计算为主；单模型、单卡、有限chunk与512上限不变。

## 2026-10-23 v3数学与四轮记录复核

用户CPU脚本在内存替换唯一写文件目标后重放，与原JSON一致；仅覆盖两个接受率反例算术及固定B的随机q恒等式，不覆盖GPU缓存、任意状态数值、成本预测或逐状态误差界。五命题的必要条件及σ(H)统计可测性/实际计算预算的区别见[审阅](artifacts/v3_review_20261023/THEORY_REVIEW.md)。完整H与固定θ相同时不能构造两个不同p。

四轮输出与语义480/480相同、跨四轮每臂120case相同；来源等权点值由两个程序及直接算术一致。原243条顺序提示已定位为正常轮换与二次来源偏移，分析器改按prompts实际序号核验，倒置配对仍报差异；新完整性差异0，原记录不改。详见rescue_repeat_20261022/analysis_review_run02及root_review_20261023.json。

## 2026-10-22 备用树与历史Feature混合验证

13状态、8来源的五节点/普通执行对照：78组类型logits、78组hidden、13组实际备用路径K/V；最高policy TV3.640074e-6，hidden相对L2最高1.592530e-6，支持集一致，符合已有相对张量与policy标准。逐元素allclose有121/182张量组未满足，日志保留，未宣称逐位相同。204次候选解码及36次组合对照通过；后者393次lerp/普通混合比较支持相同，maxTV1.851004e-8。

正式纯备用开发60/60、三十来源确认60/60输出与语义接受计数相同，恢复差额分别480/388，逐对等于备用命中。组合与快速混合三十来源60/60相同、差额380。小词表输出律最大误差5.6e-17；条件残差概率与整轮备用联合概率分别说明。没有保存完整hidden/KV原张量供本机独立重算，CPU审阅复算保存标量与计数。纯BOS仅验证相应耦合路径，不等于消除历史全局数值问题。详见[验证与结果](artifacts/spec_next_20261022/RESULTS.md)。

2026-10-21 同前缀单轮探针最终32项GPU对照通过，K1预算2、W1/W2预算4且max_rounds=1，输出与原解码器首轮相符，完整16token输出和共有计数相同。2808行主测与480行分项的独立JSON复算通过，重复输出一致、1872树轮与936一对轮的提交/Target调用关系成立，诊断与主测480/480输出一致。实际执行的三文件与本机逐字节相同。分析按来源留一，只使用前缀与动作计划形状；训练成本时的真实恢复次数在留出预测中由预测Erec替代。最终重采样使用标准随机数，旧版区间不采用。原始数据无EOS，BOS-only不在范围；局部进展率与本机评分计时不充当在线或端到端速度。见[验证与结果](artifacts/recovery_model_20261021/RESULTS.md)。

2026-10-21 原Target同轮比较120次通过资料与序列验证，原/优化Target输出计数20/20一致，四提案图方法跨轮输出计数80/80一致。主表改为原Target参照，系统与投机收益合计解释；当前MLP最快。后续Feature分支树须单独验证祖先注意力、深度位置、稀疏Position单调规则、分支KV提交与精确随机残差，不能把CPU多候选接受质量当在线速度，也不能直接套用CaDDTree块扩散分解假设的最优性结论。细节见[本轮结果](artifacts/original_target_next_20261021/RESULTS.md)与[下一方法](artifacts/original_target_next_20261021/NEXT_METHODS.md)。

2026-10-21 正式提案图主测也已完成180次、80/80输出计数配对一致、Target图/缓存事件一致，独立复算9/9通过；独立原始倍率重算匹配。2026-10-21 提案图验证完成：最终all_verify_run01为1开发来源、64/256前缀、64token、9臂共18次；四候选普通/图提案各2/2输出与计数相同，提案数值验证全部通过，Target图/缓存事件与调用配对一致，独立复算9/9通过。EAGLE独立同历史eagle_fixed_input_run02覆盖142操作、读桶128/256/512、T2回退与回退后合法Feature替换，hidden、policy与KV最大差0，缓存位置一致；完整tensor存raw_outputs.pt。数值监测关闭后才进入速度主测。有限输入验证不证明任意输入逐位一致，纯BOS的既有数值限制仍保留。详见artifacts/proposal_graphs_20261021。

2026-10-21 同Target执行比较完成：运行接口run02覆盖T1/T2/T3/T4/T6、typed/full logits、pre/post norm hidden、跨读桶、截短后合法Feature替换，最大policy TV1.33086e-6。EAGLE消费post-finalnorm hidden，MLP/LFQ消费pre-finalnorm，Medusa/Hydra保持原hook中的最终归一化。11法3来源原/优化配对66/66输出与记录计数一致；10来源主测只有优化执行，不能把该项不适用视为新增输出一致证据。所有方法真实调用图，主测回退仅每请求一次长prefill，无disabled图。LFQ BOS18次中Target3/3、MLP2/3、LFQ0/3配对一致，该轮7/8审计，仅重放项失败，同历史213次图前向复核最大Target policy TV0.012678、BF16草稿q TV0.021485并有支持集变化，BOS图路径不满足此前真实前缀数值范围，不排名、不用于同等采样结论。有限重放一致不证明所有输入逐位一致。见[验证范围与结果](artifacts/target_methods_20261021/RESULTS.md)。

2026-10-19 Feature EAGLE v4：11 项 `test_pf_eagle` 单元测试通过，覆盖持久 Draft KV 下 γ=3/5 的随机种子输出一致、拒绝后的 KV 长度及局部坐标与全上下文 RoPE 坐标一致。三种普通运行中，v3/v4 每个种子的输出、P/F 提案及接受计数逐项相同；分项运行的 profile replay 也与未插装输出一致。`py_compile` 通过。持续缓存仅由显式参数启用；完整质量与多场景测试未开展。详见 [v4 执行记录](FEATURE_EAGLE_V4_2026-10-19.md)。

2026-10-19 Feature EAGLE v3：九项测试通过，覆盖密集 KL、无真实标签时不计算 CE、末尾 F 已消费/待消费的递推与增量一致性，以及 Target 最终归一化前后两类状态。真实诊断中，新 EAGLE 使用最终 RMSNorm 后状态；旧 PF 对照使用其原接口要求的归一化前 P 状态。配对 F2 比较将 v2 重放到 v3 生成的同一候选历史；条件采样与实际到达的在线 F 接受分别报告。108 次在线比较、九路径分项重放及 13 次三维导出完成；v2 参照输出各 12 次复现历史结果。原始数据和限制见 [v3 执行记录](FEATURE_EAGLE_V3_2026-10-19.md)。

2026-10-17 成本路线 B1 验证：56 条轨迹与 448 次单轮重放的输出、分支和关键计数保持一致。精确 Feature 标签使用完整分布重合度，连续 P₁/F₁ 接受概率为首个 Position 的 Target 概率乘以该重合度；EOS 单列。概率补采的时间不进入速度结论，速度复用相同前缀的原计时。预测器按来源留一，Target 验证概率与接受结果均不作为输入；Feature 草稿统计单列为已经支付 Draft 成本的信息。坐标字段与原工程位置 ID 转换一致。未实施在线策略切换，也未验证其长期分布、缓存行为与完整生成收益。见 [本轮报告](artifacts/feature_value_20261017/full/FEATURE_VALUE_REPORT.md)。

2026-10-17 新验证入口：[A1–A3 候选成本报告](artifacts/candidate_cost_20261017/full/CANDIDATE_COST_REPORT.md)记录现用 PF-K2 的分支时间。56 组分项与普通运行的输出逐 token 一致，前轮关键计数全部复现；5,122 个主比较轮次及 896 个单轮对照的推进量、Target 调用关系均相符，EOS 短验证通过。只对隔离副本增加观测点，原正式采样器保持原样。当前路径已包含 Feature 提案，下文较早的 Position-only 描述按原日期理解。时间相减估计仅适用于已观测路径；不同策略的长期上下文变化与动态切换的采样、缓存行为尚未验证。

2026-10-08 阅读提示：本文按原日期保留各路径的验证范围。后续已出现真实联合 P/F 原型及 S13 结果，早期“Feature 只由 Target 采样”的描述限于 Position-only 路径。当前暂缓新增实验，综合结论与未解决事项见 [阶段工作汇报](STAGE_REPORT_2026-10-08.md)。

更新时间：2026-10-04。本文结合 Autodl 上 `model/gpt/position_selective_speculative.py` 的设计说明与统计结构，筛选适合当前解码器的提速路线。

## 结论先行

当前实现**只 draft position token，feature token 始终由 target 采样**。源码头部说明拒绝修正只用于 position；统计结构有 position proposals/accepts/rejects，而 feature 侧只有 `feature_observations` 和 `feature_overlap_sum`。因此低 feature overlap 是 draft 输出的诊断值，**不是 feature proposal 的拒绝率，也不会直接触发 pair 回滚**。

初筛后的近期顺序是：核验 position 的 exact correction 和 KV cache；拆分 draft、target block、rejection 重算和同步耗时；再决定 D4/D6 draft 对照或 position-only on-policy distillation。Feature distillation 可保留为模型诊断或质量实验，但只有它能改善 position proposal 或端到端成本时，才算加速方案。

Autodl 的文件头和统计结构已核对；本轮没有逐行审完函数主体。因此下面确认的是当前设计边界，corrected position 的实际 cache 执行路径仍要通过源码审查和小分布验证。

## 1. 已有数据应如何解释

项目记录中，位置实际接受率约为 0.922，position 分布 overlap 约为 0.938，feature 分布 overlap 约为 0.382；端到端 speedup 约为 1.06–1.15×，不同运行存在波动。

当前算法提议的是 position，所以影响 speculative accept/reject 的直接量是 `P_T(position | prefix)` 与 `Q_D(position | prefix)` 的重合度及实际 position 接受率。Feature overlap 来自 draft 对 feature logits 的旁路诊断；feature 实际从 target 分布采样。不能把 position 接受率和 feature overlap 相乘，也不能用 feature overlap 推断 pair rejection。

这些数据说明 position proposal 已有较高一致性，但当前 speedup 仍小。最需排查的是 D8 每轮成本、target 两 token block 的实测效率，以及 position rejection 后额外 target/cache 工作。低 feature overlap 只有在研究“把 feature 也纳入 draft”的新路线时才是直接指标，而那会改变当前算法结构和成本模型。

## 2. 当前解码器的正确性逻辑

场景 token 流按 `BOS, P_1, F_1, P_2, F_2, ...` 交替。当前设计说明给出的 steady-state target block 是：

```text
target cache through P_{i-1}
    + [F_{i-1}, draft(P_i)]
    -> logits for P_i and F_i | draft(P_i)
```

若 draft position 被接受，target block 第二行已经给出 `F_i` 的 target 分布，随后直接从 target 采样 feature。若 position 被拒绝，则按 target/draft 的最终 position policy 做 residual correction，回滚候选 position 的 target cache，把 corrected position 输入 target，再从得到的 feature logits 采样。Draft cache 必须同步到 target 实际提交的 `[P_i, F_i]`，再产生下一轮 position proposal。

因此当前实现无需“position 接受、feature 拒绝”的 partial acceptance：feature 不由 draft 提议，没有 feature rejection。把 feature 改为 draft proposal 后，才需要针对条件分布 `P(f | prefix,p)` 做 exact correction，并在 corrected feature 后丢弃基于旧 feature 生成的后续 draft；这属于另一条算法路线，应单独立项和验证。

## 3. 候选方案筛选

| 候选方向 | 判断 | 原因与进入条件 |
| --- | --- | --- |
| 核验 position residual correction、target/draft cache 同步 | **必须先做，P0** | 当前唯一的 speculative token 是 position。检查最终采样分布、拒绝 residual、corrected position 回写、target feature bonus 和 EOS 路径。 |
| 分项 wall-clock profiling | **保留，P0** | 高 position acceptance 与 1.06–1.15× speedup 并存，说明接受率不能解释全部成本。拆 draft forward、target 两 token block、拒绝后的 target 重算、cache 和同步时间。 |
| D4/D6 draft 对照 | **保留，P1** | 当前 D8 的 position acceptance 已约 0.92。更小 draft 可能降低 proposal 成本；必须测它损失多少 position overlap，以及端到端净收益。 |
| Position-only acceptance-aware/on-policy distillation | **保留，P1** | 训练目标只需重点匹配真实 target feature 之后的 position 分布。可在部署 policy 上比较 position `sum min(P,Q)` 与实际接受率；corrected sampling 仍保留 exactness。 |
| Feature distillation | **降为诊断或间接路线** | `q_F` 不参与当前 proposal 和 correction。它可能通过共享参数间接影响 `q_P`，但只有 position acceptance 或 wall-clock 实测改善，才能支持其提速价值。 |
| Feature-only 动态分支 / pair partial acceptance | **排除当前解码器的近期方案** | 当前没有 feature proposal 或 feature rejection，不能修复现有路径上的拒绝问题。若改成 draft feature，则需重新审查成本和 exactness。 |
| 固定增加 position draft 深度 | **排除为直接扩展** | position 与 target-sampled feature 交替；下一位置 proposal 依赖中间实际 feature。不能像连续文本 token 那样直接延长 draft block，除非树同时覆盖中间 feature 状态，分支成本可能很高。 |
| 多位置候选树 | **暂缓** | 可以研究同一已知 prefix 下的 position 候选，但要证明 target 验证、3D RoPE、cache 分支和候选概率修正的总成本低于单 proposal。它不能绕过下一步 feature 由 target 采样这一依赖。 |
| 加大 draft（例如 D12） | **只作容量对照** | 可能提高 position overlap，也会增加每轮 proposal 成本和显存。先测试更小 draft 和 profiling；没有净成本模型前不应默认扩大。 |
| Codebook 邻居 / 渲染近似救援 | **排除 exact 主线** | 直接用邻居替代 target residual 会改变目标分布。对当前 position-only proposal 也没有直接作用；可以另做近似质量路线。 |
| Lookahead / Jacobi / REST | **不作当前主线** | 这些方法依赖的复用、重复结构或迭代收敛条件尚未在 Gaussian position/feature 序列上得到证明。先用现有 position speculative baseline 建立净成本，再决定是否值得做控制实验。 |
| Medusa/EAGLE 多头 | **延期** | 多步 position head 必须跨过 target-sampled feature；普通 future-token head 不能直接假设这些 feature 已知。需设计条件 head/树并重训，先排除 D8 proposal 成本是否才是问题。 |
| 改 codebook / hierarchical VQ | **长期方向** | 会改变 VQ-VAE、tokenization、GPT 训练和渲染质量，属于表示层研究，不是当前 position rejection 的小修复。 |
| Render-aware loss 作为 exact 加速目标 | **不成立** | 渲染质量改善不等于 position proposal 分布更接近 target，也不保证 target 分布不变。可以作为单独质量目标，仍需 position exact correction。 |

## 4. GaussianGPT 的结构约束

1. Position token 携带空间坐标并参与 3D RoPE。不同坐标可成为并列候选，但每条路径必须保留自己的坐标、RoPE 和前缀状态，不能把路径合并或重排。
2. Position proposal 依赖此前已提交的 feature。当前 feature 来自 target，因此单轮 proposal 不能直接延伸到多个未来 position；每个中间 feature 都会引入一次 target 条件状态。
3. Feature code id 是 VQ 离散码，不等同于自然语言语义标签。codebook 邻近、Gaussian 属性距离或渲染相似度都不能代替 position target/draft correction。
4. 当前解码器按设计在模型 mask 和 temperature/top-k/top-p policy 后执行 position correction。loss、overlap 和接受率比较必须采用相同 policy 分布。
5. 当前实现声明 batch size 为 1，因为 rejection 会导致 cache 长度分叉；任何批量化优化需单独处理不同样本的提交长度。

## 5. 最小验证顺序与停止条件

### V0：完成当前 position sampler 源码审计

逐行确认 Autodl 上的实现是否兑现文件头所述设计：

- `P_T` 与 `Q_D` 是否对应相同的 slot/hard masks、temperature 和 top-k/top-p 后的最终 position 分布；
- position rejection 是否按归一化 `[P_T - Q_D]_+` 采样；
- speculative position cache entry 是否被回滚，corrected position 是否重新输入 target；
- target feature 是否只从正确 position 条件下的 logits 采样；
- draft cache 是否以 target 实际提交的 `[P_i,F_i]` 同步；
- EOS、零概率项和 token type mask 是否覆盖。

### V1：小分布 exactness 验证

用小词表 position 分布构造 target `P` 和 draft `Q`，覆盖 position accept/reject、零概率和 EOS。重复采样并检查 `(position, target-sampled feature)` 联合经验分布符合 target。模拟 cache 状态时确认拒绝位置从前缀移除、corrected position 与对应 feature 被提交。此项验证 correctness，不代表 GPU 加速。

### V2：成本拆分与 position proposal 实验

固定 teacher、VQ-VAE、seed、256-token 上限、warmup、temperature 0.9、top-p 0.9 和 GPU，多 seed 对比 baseline、现有 D8、D4/D6。记录：

- `alpha_pos = position_accepts / position_proposals`；`overlap_pos` 单独记为 position 分布上的 `sum_v min(P_T(v),Q_D(v))`；
- `q_F` feature overlap 作为旁路诊断单独保留，不与 position acceptance 混写；
- draft forward、target 2-token verify block、position rejection 后的 target rerun、cache、sampling/synchronization 时间、forward 次数、峰值显存和端到端吞吐。

比较 `T_spec = T_draft + T_target_verify + T_reject_rerun + T_cache + T_sampling/sync + T_other` 与相同输出长度的 baseline。接受率是诊断量，不是终点：若 D4/D6 降低 draft 时间且 position acceptance 损失很小，优先采用较小 draft；若 position rejection 是主要额外成本，再试 position-only on-policy distillation。

### 停止条件

- Exactness 或 cache 前缀验证失败：不跑性能对比。
- `q_F` overlap 改善但 position acceptance 和端到端时间未改善：不把 feature distillation 扩成加速主线。
- position acceptance 提升但总 wall-clock 未改善：回到分项成本，不继续扩大蒸馏。
- 多候选树减少拒绝却增加总 wall-clock、显存或 cache 复杂度：停止树路线。
- 近似方法改变 target 输出分布：从 exact 结果中剥离，独立报告渲染与场景质量。

## 6. 当前裁决

当前要解决的不是 feature rejection，因为现有解码器没有 feature proposal。近期主线应为 **V0 position correction/cache 审计 → V1 exactness → V2 draft/target 成本拆分 → D4/D6 与 position-only distillation**。Feature overlap 仍有研究价值，但需要先证明它能预测或改善 `q_P`，不能直接当作当前 speedup 的瓶颈指标。

本文件是初步逻辑筛选。函数主体审计、更新后的多 seed profiling 和小 draft 对照完成后，应继续修订判断。

## 7. Selective LM head 与 compact sampling 验证（2026-10-04）

### 7.1 代码与运行时检查

远端 `model/gpt/position_selective_speculative.py` 应用 selective LM head 和 compact sampling patch 后，通过了 `python -m py_compile`。最小 validator 记录：模型词表 `12098`，Position 起点 `8000`，Feature 大小 `4096`，EOS `12096`；完整/selected 合法 logits 的最大绝对误差分别约为 `6.6757e-6` 和 `4.7684e-6`，mask 外有限值计数为 0。32-token engine smoke 产生 16 对 proposal、14 次 Position accept，EOS=false，token stream/cache 不变量通过。

### 7.2 sampling distribution 等价

对带有相同 token-type mask 的 synthetic logits，将全词表 policy 与 compact support sampling 后恢复到全局 token id 的分布逐元素比较：

| token type | top-p | 恢复分布 `max_abs` | 概率和 |
| --- | ---: | ---: | ---: |
| Position + EOS | 0.9 | 0.0 | 1.0 |
| Position + EOS | 1.0 | 0.0 | 1.0 |
| Feature | 0.9 | 0.0 | 1.0 |
| Feature | 1.0 | 0.0 | 0.99999994 |

这说明在给定 logits、temperature 和 top-p policy 下，先压缩合法 support 再采样与全词表 masked sampling 数值等价。它不等价于固定 seed 的逐 token 输出：compact tensor 的形状会改变随机数消耗顺序；端到端 exactness 仍应使用经验分布或联合采样测试，而不是比较单条 token stream。

### 7.3 正确性边界

compact sampling 只删除已知为 `-inf` 的无效 token，并在 correction 时使用同一合法 support。它没有改变 Position residual correction 的公式、target feature 的采样来源或 cache rollback；因此属于实现级加速，不增加一个新的 acceptance decision。若未来让 Feature 成为 draft proposal，必须重新证明条件 Feature distribution、partial rejection 和 corrected-prefix cache，不能从本节的 support 等价性直接外推。

### 7.4 性能验证要求

已完成的 128/1024/2048/4096 token endpoint 对比支持约 5.8–8.4% 的初步吞吐收益，但 4096 只有两个公平的固定长度 seed，且 benchmark runner 保存在远端 `/tmp` 临时文件。额外的 seed7/8/9 运行出现单侧提前 EOS，已作为 EOS/鲁棒性控制保存，未混入固定长度均值。合入默认路径前，应将 runner 固化到 canonical e2e harness，补齐严格的长度/ EOS 报告、重复方差、GPU synchronization 边界，以及 token-type/渲染质量回归；同时保留 selective-head-only 作为消融，区分 LM head 优化与 compact sampling 的增量贡献。

## 8. Feature shadow acceptance 原型（2026-10-04）

### 8.1 接入位置与 cache 逻辑

远端源码现在在 Position proposal 已通过后，使用 Target 给出的条件 Feature logits `p_F(F_i | H_i,P_i)`，并从 Draft 的第一行 update logits 得到 `q_F(F_i | H_i,P_i)`。Draft update 的输入是已经提交的 `[P_i,F_i]`，因果 mask 保证第 0 行不读取第二个输入 Feature，因此该行可以回看当前 Feature 条件分布；第 1 行仍用于下一位置分布。这样既没有增加 Draft forward，也没有改变 Target 采样 Feature 和两侧 cache 的提交顺序。

shadow 分支用独立 RNG 从 `q_F` 采样候选，按照同一 compact policy 调用 ratio acceptance helper，并记录 `feature_shadow_accepts`、`pair_shadow_accepts`、接受概率和 overlap。最终 `output` 仍追加 Target 采样出的 Feature，Draft 仍同步这个真实提交的 `[P_i,F_i]`，所以现有 cache invariant 不变量继续适用。

### 8.2 数学与运行时证据

小分布 self-test 的目标概率为 `[0.1,0.2,0.7]`，经验输出为 `[0.09882,0.20186,0.69932]`，理论 overlap=0.55，经验接受率=0.55172。F-aware 端到端两次运行均记录 Position 119/128、Feature shadow 60/119、pair shadow 60/128；wall-clock ratio 分别为 1.02490x 和 1.01728x。没有 `draft_feature_probe` 阶段，说明共享 `q_F` 路径已经生效。源码 `py_compile`、cache invariant check 和 256-token engine smoke 均通过。

### 8.3 正确性边界

当前统计中的 `feature_shadow_accepts` 不是最终 Feature acceptance：shadow 候选不会替换 Target Feature，也不会触发 Feature residual correction。因此它可以回答“如果使用当前 q_F，条件 Feature proposal 的接受率大约是多少”，不能声称已经实现联合无损 speculative decoding，也不能把 0.46875 的 pair shadow alpha 用作端到端收益。

真实联合路径需要额外的状态机：先在 Target verify 前提出 `F_i~q_F`；若 Position 接受且 Feature 接受，提交 Draft pair；若 Feature 被拒绝，从归一化 `[p_F-q_F]_+` 采样 corrected Feature，并丢弃基于候选 Feature 的后续 Draft 状态；若 Position 被拒绝，先按 Position residual 修正，再在 corrected Position 条件下重新取得 Feature 分布。每条路径都要验证 Target/Draft cache 前缀、EOS、token-type mask 和 `[P,F]` 联合经验分布。

## 9. 真实联合 Feature acceptance 原型（2026-10-04）

远端实验模块 `model/gpt/joint_feature_speculative.py` 已将上述三条状态路径落地：Draft 先提出 Position，再执行 `draft_feature_probe` 得到条件 `q_F`；Target 的两 token block 同时产生 `p_P` 和候选 Position 条件下的 `p_F`；Position rejection 重新 forward corrected Position 的 Feature logits，Feature rejection 使用 `[p_F-q_F]_+` residual。每个完整 pair 都检查两侧 cache 长度，256-token/128-pair 运行没有触发 invariant 错误。

两次重复运行的接受计数均为 Position `121/128=94.53%`、Feature `51/121=42.15%`、联合 pair `51/128=39.84%`，Feature policy overlap 为 `0.41386`。端到端 baseline/joint 吞吐分别为 `72.725/66.804 tok/s`（`0.9186x`）和 `73.150/65.109 tok/s`（`0.8901x`）。因此机制和 cache 逻辑已验证，但性能为负：额外 `draft_feature_probe` 的成本超过了联合 acceptance 带来的摊销。`feature_shadow_*` 字段在现有 dataclass 中沿用旧命名，解释这些 JSON 时应按“真实 joint acceptance”读取，不能与 S7.3 的 retrospective shadow 统计混称。

50,000 次小词表联合 Monte Carlo 进一步得到 Target joint 分布最大绝对误差 `0.00284`、L1 误差 `0.00772`，`JOINT_FEATURE_MATH_SELFTEST` 通过。该测试只覆盖 ratio/residual 数学组合；真实模型仍需在多 seed 上比较 joint sampler 与 Target autoregressive sampler 的边际、pair 频率和渲染质量。

该结果还没有提供大规模联合分布 exactness 的统计证明；下一次必须在小词表/小序列上将真实 joint sampler 与 Target autoregressive sampler 的 Position、Feature 边际和 pair 频率对齐，并把这一测试放在性能 benchmark 之前。当前裁决是保留原型作为 correctness reference，不接入默认 Position-only 路径。

## 8. Structured Position micro-draft 的 exactness 前置条件

近零成本查表模型目前只在真实 Target prefix 上预测 Position gap 的 mode，因此它验证的是序列结构可预测性，不是 speculative decoder 的分布等价。真实接入前必须满足以下条件：

- 将每个结构化 context 的 gap Counter 归一化为合法的 `Q_D^P`，并在相同 temperature/top-k/top-p policy 下与 Target `P_T^P` 比较；
- 明确 gap 的合法范围、首个 Position、序列边界和 EOS，并把不存在的 gap 从 support 中屏蔽；
- 对 Position accept、Position residual correction 和 EOS 做小词表经验分布测试；
- 在 Position rejection 后确认 Target cache 只保留 proposal 前缀，corrected Position 后的 Feature 仍由 Target 重新采样；
- 对实际 lookup、support 映射、同步和 correction 单独计时，再与 D8 draft forward 比较。

因此 `85.04%` top-1、`2.58` 个 K=4 连续命中和 `<1 μs` CPU lookup 只能作为进入 adapter 实验的门槛证据。它们不能替代 `sum_v min(P_T,Q_D)`、实际 acceptance、联合输出分布和端到端 wall-clock 验证。

### 8.1 归一化表与 residual correction 单元结果

远端 `scripts/structured_position_exactness.py` 已将 h=5、低 1 位结构化表转换为合法的 `Q_D^P`。在验证 corpus 上，平均经验 overlap=`0.84520`、真实下一 Position 的平均 `Q_D^P` 概率=`0.78636`；这两个值使用的是 Target rollout 的经验条件分布，不能替代 Target neural `P_T^P`。Synthetic 分布测试中，100,000 次 ratio/residual correction 的 acceptance=`0.44762`、理论 overlap=`0.45`，L1=`0.00340`、最大绝对误差=`0.00170`。一个 82 次观测的真实 context 中，经验 acceptance=`0.86612`、理论 overlap=`0.86697`，L1=`0.00324`、最大绝对误差=`0.00121`。

由此只确认了 proposal 归一化和 residual correction 的数学路径；接入 engine 后还必须用 Target logits 计算 `sum_v min(P_T,Q_D)`，验证 Position rejection 后的 Target/Draft cache 前缀、EOS 和 token-type support，并在固定长度多 seed 上测 wall-clock。该单元测试通过不能被解释为已有真实 speculative acceptance 或 3.67x 加速。

### 8.2 结构化 Position adapter 的端到端逻辑检查（2026-10-05）

端到端 runner 在 Target neural verification 中使用结构化 Q_D^P，并保留原 engine 的 residual correction、rollback、Target Feature sampling 和 cache 提交顺序。固定 256 token/128 pair 的运行中，128 个 Position proposal 有 115 个被接受、13 个被拒绝；两侧 cache invariant 检查 128/128 通过，输出长度为 256，未发生 EOS 提前结束。adapter 只维护 Position proposal，feature_shadow_accepts=0、pair_shadow_accepts=0，所以没有把未实现的 Feature acceptance 混入正确性结论。

这验证了三件事：查表 proposal 能被映射到合法 Position support；Position rejection 后仍能沿既有 corrected-prefix 路径继续生成；Target-sampled Feature 没有被 draft 候选替换。它尚未验证 1024/2048/4096 长序列的统计稳定性，也没有证明结构化表和 Target logits 的联合分布完全相同；这些需要固定长度、多 seed 的输出分布与渲染回归。端到端 1.36685x 是本次 runner 的真实 wall-clock 结果，不能替换为理想 block-level 3.67x。

### 8.3 长序列 cache 与 acceptance 检查（2026-10-05）

固定长度扩展覆盖 1024、2048、4096 token。三次运行的 Position proposal/accept/reject 为 512/459/53、1024/925/99、2048/1860/188；cache invariant 检查分别为 512/512、1024/1024、2048/2048，terminated_on_eos 均为 false。Position rejection 触发的 Target cache rewind 次数与 reject 数一致，没有出现 corrected Position 后的 Feature 条件错位。

长度扩展只证明当前 seed 和这张结构化表的状态机在更长 prefix 上不崩溃，不能证明跨场景泛化或多 seed 稳定性。lookup 计时从约 146.6 μs/query 增加到 244.5 μs/query，后续要将这部分从 Python host 路径迁移或融合，分别测 table lookup、support mask、Target verify、rejection rerun 和同步时间，避免把 host overhead 隐藏在总 speedup 中。


### 8.4 多 seed scaling 的图形化逻辑检查（2026-10-05）

远端 profiling/s8_structured_position/grid/ 中的 12 个运行（seed=0/1/2，长度=256/1024/2048/4096）均返回成功，输出长度与请求一致。图形化汇总 multiseed_scaling.png 同时展示了 speedup、Position acceptance、两条吞吐曲线和 lookup 成本，便于人工检查异常趋势。

逻辑检查结论：

- speedup 的每个 seed/长度点均大于 1，均值约 1.36–1.40x；没有发现某一长度或某一 seed 退化到 baseline 以下。
- Position acceptance 在 1024–4096 token 约 0.905–0.911，和单 seed 长序列结果一致；256 token 的跨 seed 波动较大，符合较小 proposal 数导致的统计方差。
- lookup 成本随长度增长，且与 structured 吞吐下降同向；这支持将 host lookup/support mapping 作为下一轮 profiling 的重点。
- 本轮验证的是 Position-only exact-correction 路径；Feature proposal、joint pair acceptance、跨场景输出分布与渲染质量仍未覆盖。图形结果不能替代这些边界测试。

该检查确认当前 adapter 的 cache/rollback 路径在多 seed、长序列 grid 中没有显性逻辑异常，但进入默认路径前仍需多场景复测以及 GPU/fused lookup 对照。


### 8.5 正式 Draft proposer 合约与双路径回归（2026-10-05）

`scripts/selftest_draft_proposer.py` 通过：Neural D8 proposer 和 structured proposer 都只在合法 Position/EOS support 上返回 proposal；structured history 在 commit、EOS 和 rewind 后保持同步，reset 能清空状态。engine 的 syntax compile 通过。

正式 structured proposer 的 256-token smoke 中，Target 仍是唯一的 Feature 采样来源，Position rejection 继续走既有 residual correction 和 Target cache rollback。128 次 cache invariant 检查全部通过，115 次接受、13 次拒绝，输出长度为 256，未提前 EOS；`feature_shadow_accepts` 与 joint pair acceptance 没有被启用。无 proposer 的 D8 路径同版本回归通过，证明 proposer 注入没有破坏旧路径的 cache grammar。

这一步只确认接口、support、EOS、commit/rewind 以及真实 Target verification 的 cache 前缀逻辑；structured smoke 的 1.39922x 是单 seed、单场景、256-token wall-clock evidence。多 seed/长度 grid 仍使用此前 canonical artifact，跨场景、GPU/fused lookup、cascade 和 F-aware/joint 路线尚未在本节中声明完成。

### 8.6 GPU/vectorized lookup 的 exactness 与端到端逻辑检查（2026-10-05）

第三阶段的 lookup 对照使用同一张结构化表和同一合法 support，先把各实现的输出恢复到相同的全局 Position token 维度，再比较概率张量。CUDA benchmark 明确在计时前后同步；`tensor`、`vectorized` 以及 vectorized batch size 1/8/32/128 相对 Python 基线的最大绝对误差和绝对误差和都为 0。该检查覆盖整数 context code、backoff 选择、gap support 映射、EOS 位置和 batch 输出形状，未发现实现级分布漂移。

formal proposer E2E 的三种模式都得到 128 次 proposal、115 次接受、13 次拒绝；Target cache invariant 检查 128/128 通过，输出长度为 256，`terminated_on_eos=false`。三次运行中 Target Feature 仍由 target 采样，`feature_shadow` 和 joint `[P,F]` acceptance 没有打开，所以 lookup 改动没有改变 correction 或 cache 提交顺序。

需要保留的边界：`vectorized.query_batch` 本身的 batch exactness 已通过，但 engine 现在逐 proposal 调用，不能用 batch microbenchmark 证明端到端收益；当前 vectorized 实现仍有逐行的 host 端选择和小 tensor 构造。checkout 中没有编译并注册的 fused CUDA extension，fused 分支因此只记录 unavailable。下一次若接入 engine 级 batch 或 fused kernel，必须重复本节的全局 support exactness、EOS、rejection rollback 和 cache invariant 检查。

### 8.7 多 seed/长度 Position-only 验证（2026-10-05）

第四阶段完成 `seed=0/1/2` 与 `256/1024/2048/4096` token 的 12 个 structured + 12 个 D8 配对单元。固定配置为 RTX 4090、warmup=128、`temperature=0.9`、`top_p=0.9`、CUDA、同一 Target/Draft/VQ-VAE checkpoint；structured 使用 Python lookup，表为 `h=5, feature_bits=1, Morton bit=1`。所有任务返回码均为 0。

逻辑门结果：

- structured speedup 的 12/12 单元均大于 1，范围 1.30586–1.46463x，总体均值 1.36149x；
- structured acceptance 范围 0.89648–0.99219，长度均值为 256: 0.93750、1024: 0.90951、2048: 0.90527、4096: 0.91097，没有系统性下降；
- structured 输出在 12/12 单元达到请求长度，structured cache invariant 12/12 通过；
- D8 最低 speedup 为 0.96349（seed=2、1024），所以 D8 的 `all_cells_gt_1` gate 为 false；
- D8 seed=2 的 2048 和 4096 route 在 1753 token 合法 EOS，D8 提前 EOS 计数为 2；其余 D8 route 达到请求长度；
- 将 EOS route 的合法终止条件纳入检查后，D8 cache invariant 也为 12/12 通过。非 EOS 要求 `checks=proposals=generated_pairs`；EOS 要求 `checks=generated_pairs` 且 `proposals∈{generated_pairs, generated_pairs+1}`。

这一轮的 `scene_mode` 是 `eos_seed_trajectory`，不是独立场景测试。当前 runner 没有 prompt/scene 参数，且 baseline 使用 `seed+10`、route 使用 `seed+11`，所以 baseline 与 route 不共用同一 RNG stream；发生 EOS 时实际 token 数必须和 speedup 一起报告。该限制解释了 D8 两个长度不匹配单元，也阻止把本轮结果写成跨场景泛化或相同输出序列的证明。

汇总脚本修正了 D8 `terminated_on_eos` 的嵌套字段读取，并将 EOS 最后一次 proposal 的 cache 规则显式化；修正前的“D8 无 EOS”和 cache fail 是解析器误判，不能作为实验结论。结果图和原始 JSON/CSV 位于远端 `profiling/s9_multiscene/`，重建 GIF 仍是 S8 的现有轨迹，未新增独立场景渲染。

因此本阶段只批准 **structured Position micro-draft → cascade 评估**：其收益、acceptance 和 cache 逻辑在当前冻结轨迹矩阵上稳定。跨场景输入、相同输出长度对照以及重建质量回归仍是 cascade 前必须补齐的实验边界；Feature-aware 和 joint `[P,F]` 路径继续保持关闭。


### 8.8 structured→D8 cascade proposer 合约与阶段门槛（2026-10-05）

`CascadePositionProposal` 复用同一个 structured support、EOS 处理、`commit`、`rewind` 和 `reset` 合约；它先查询结构化 Position 分布，再按 top-1 probability 或 entropy 选择 structured proposal，低置信度时返回 canonical D8 proposal。Target correction 没有按 proposal 来源分叉，仍由 engine 统一决定 Position accept、residual correction、Target cache rollback 和 Feature sampling。

固定 256-token/128-pair smoke 中，structured、top1=0.5、top1=0.9、entropy≤1.0 和 D8 五条路径分别完成 128/128 cache invariant checks；Position accepted 分别为 115、115、122、115、113。top1=0.9 触发 18 次 D8 fallback，entropy≤1.0 触发 4 次，说明 fallback 分支可达且没有破坏 cache 前缀。形式合约 selftest 与 `py_compile` 均通过。

当前实现的关键限制是 `CascadePositionProposal.requires_draft_forward=True`。这使 engine 在调用 cascade 之前对每个 proposal 都运行 D8 forward，因此 top1=0.5 虽然 0 次 fallback，仍然为每一步支付 D8 成本；top1=0.9 和 entropy 路径也支付完整的 eager D8 成本。实际 route throughput 为 73.743、76.173、73.872 tok/s，低于 structured-only 的 95.571 tok/s。由此得到的阶段判定为：

- **正确性门槛：通过** —— support/EOS、fallback、rollback 和 128/128 cache invariant 均通过；
- **wall-clock 门槛：未通过** —— cascade 没有优于 structured-only，不能进入 F-aware D8 或 joint `[P,F]`。

下一次修改必须保持 Target correction 不变，并把 D8 forward 改成 lazy fallback：先用结构化 proposal 计算置信度，只有低置信度时才补做一次 D8 forward，再统一进入已有 ratio acceptance。完成后需重复本节的五路对照、多 seed、cache invariant 和实际输出长度检查。


### 8.9 lazy cascade 的 deferred replay 与 cache 检查（2026-10-06）

lazy 实现把低置信度标记留在 `PositionProposal.metadata`，engine 在同一 proposal 层创建临时 D8 cache，并用当前已提交的 `[BOS] + output` replay 得到 D8 Position logits；重试时只替换 proposer 的 Position 分布，Target correction 和后续 commit/rewind 仍沿原路径执行。高置信度结构化 proposal 不触发 replay。

在 256-token/128-pair smoke 中，top1 阈值 `.5` 的 `proposal` 为 `structured=128`、`neural_fallback=0`，top1 阈值 `.9` 为 `structured=67`、`neural_fallback=61`；两条路径均完成 128 次 cache invariant 检查。`.9` 路径 Position 为 127 accepted、1 rejected，`.5` 为 115 accepted、13 rejected；EOS、support、rollback 和 Target-sampled Feature 均保持合法。

该检查覆盖了“结构化 proposal 延迟到 D8 replay 后再重新提案”的状态流，但没有证明 replay 的 wall-clock 可以摊销。`.9` 的 route 只有 1.04909x，而无回退 `.5` 为 1.37455x，说明 fallback 比例和 replay 成本是端到端门槛；不得因为 deferred replay 的 cache 逻辑通过就提前接入 joint `[P,F]`。

### 8.10 Position policy-KL support 与训练路径验证（2026-10-06）

Position policy-KL 只作用于合法 Position/EOS 行：

- Position support 为 `0..7999`，EOS 为 `12096`；
- 对每个有效 target 行，Teacher 与 Draft 都按同一 support 取 logits，再用当前 temperature/top-p policy 得到分布；
- support 外的 Position ID 在权重大于零时直接报错，避免把 Feature 行混入 Position KL；
- `position_rows` 记录实际参与 KL 的行数，`position_policy_overlap` 记录概率重叠度。

实现复用了已有 forward 输出，没有增加 target 或 draft 的完整前向；Target verification、Feature 采样、residual correction、cache rollback 和提交顺序保持原状。因此这项训练指标不会改变解码器的 cache 状态，也不会隐式启用联合 `[P,F]` acceptance。

验证证据：

- 两个训练脚本 `py_compile` 通过；
- 两个 CLI help 都包含 `--position-policy-kl-weight`；
- RTX 4090 短跑完成 `global_step=2`，训练/验证 Position rows 都为 129；
- 训练 Position KL/overlap 为 `0.27861869/0.97100383`，验证为 `0.70167381/0.86094409`；
- 输出 `summary.json`、`run_config.json` 和日志均已写入远端 S12 目录。

当前范围只覆盖 2 step 的运行链路，尚未比较不同权重、多个 seed、完整训练后的 Position acceptance、Feature shadow acceptance、端到端吞吐或重建质量。独立 Feature/logit projection 与真实联合路径仍需单独验证。


### 8.11 F-aware D8 矩阵验证（2026-10-06）

七个矩阵运行均按请求完成：五个 8 step 运行和两个 16 step 运行的 `global_step` 分别为 `[8, 8, 8, 8, 8, 16, 16]`。带有 Position policy-KL 的运行均报告 129 个有效 Position 行；带有 Feature KL 的运行均报告 128 个有效 Feature 行。`lambda_P=0.3` 的验证 Position policy overlap 落在 0.86142–0.86169，和 2 step 短跑的 0.86094 一致。日志中没有出现 support、EOS 或 token grammar 错误，七个 `summary.json` 都能被汇总脚本读取。

这轮只验证训练指标路径、权重写入、有效行数和结果文件输出。decoder acceptance、缓存前缀、端到端 wall-clock、Draft 成本和重建结果尚未覆盖，因此不能据此声称 F-aware D8 已经带来加速。Feature policy overlap 是训练分布指标；它与 Feature shadow 统计和真实 joint `[P,F]` acceptance 保持分开，后续报告必须继续分别命名。

### 8.12 F-aware checkpoint 解码验证（2026-10-06）

S12.1 的两个 16 step checkpoint 已接入同一 shadow 解码入口，加载、前向和输出均成功。配置为 CUDA、256 token、warmup=32、seed=1、`temperature=0.9`、`top_p=0.9`、`feature_shadow=True`。候选与对照分别为：

- `/root/workspace/GaussianGPT/profiling/s12_faware_matrix/onpolicy_lambdaF03_lambdaP03_16step/checkpoints/last.ckpt`
- `/root/workspace/GaussianGPT/profiling/s12_faware_matrix/onpolicy_lambdaF0_lambdaP0_16step/checkpoints/last.ckpt`

两次运行都得到 Position `113/128`、Feature shadow `54/113`，生成长度为 256，`terminated_on_eos=false`。候选 route 时间为 `3612.0473 ms`、吞吐 `70.8739 token/s`，相对同次 Target AR 为 `1.02164x`；对照 route 时间为 `3633.8838 ms`、吞吐 `70.4480 token/s`，相对同次 Target AR 为 `0.99905x`。Target cache rewind 计数在候选日志中为 15；本次 smoke 没有单独重算全部 cache invariant，因此只把它作为回退路径的观测值。

日志没有出现 Position/EOS support、token grammar 或 checkpoint 读取错误。候选与对照的接受计数完全相同，说明这一次短跑没有观察到 F-aware 训练带来的接受改善。Feature shadow 仍是旁路统计，不代表真实联合 `[P,F]` 接受；当前证据不足以恢复联合路径，也不足以证明端到端收益可重复。

原始结果位于 `/root/workspace/GaussianGPT/profiling/s12_faware_matrix/eval_lambdaF03_lambdaP03_16step.json`、`/root/workspace/GaussianGPT/profiling/s12_faware_matrix/eval_lambdaF0_lambdaP0_16step.json` 和 `/root/workspace/GaussianGPT/profiling/s12_faware_matrix/decoder_shadow_compare.json`。后续需要多 seed、多长度、额外场景以及完整 cache invariant 与重建回归；本轮未新增 GIF。

## 8.14 PF执行优化独立解码的配对与数值验证（2026-10-21）

独立确认集含30个场景，每场景覆盖prefix 64/256与1个seed，四模式共240次、60组配对，每次最多512个新token。`independent_audit.json`复算通过：计划和实跑数量一致，原/候选Target的60组输出及计数一致，原/候选PF亦为60组完全一致；来源等权速度比及区间与`summary.json`相符。候选PF相对原Target为`4.519564 [4.228452, 4.720021]`，相对原PF为`2.613919 [2.506082, 2.672991]`，相对同优化Target为`1.442534 [1.372647, 1.495168]`。

单个256-token前缀上的Target T=1/2/4语义验证均保持logits与采样支持一致，最大policy TV分别为`5.02e-7`、`6.50e-7`、`9.09e-7`；pre-finalnorm hidden每行相对L2最大值分别为`2.28e-7`、`7.51e-7`、`8.50e-7`。同一真实序列上的Feature MLP输入`h_prev`支持一致、相对L2为`1.39e-7`，4096类q的支持一致且TV为`8.72e-7`。逐元素hidden allclose在T=2/4未通过，记录为诊断项；语义判定依照有限值、相同形状、支持和分布距离结果通过。该数值证据仅覆盖一个前缀样例。

模型权重于矩阵计时前加载；候选运行时初始化、缓存分配及图创建计入对应外层解码时长，32-token预热另行运行。自然EOS补测为3场景、3组配对、12次解码，全部EOS结束且没有预算截断；8项独立审计全通过。原/候选Target与原/候选PF分别3/3组输出和计数相同。候选PF相对原Target、原PF、同优化Target的来源等权速度比分别为`3.630147 [2.791982, 4.246229]`、`2.209333 [1.987015, 2.373086]`、`1.273811 [1.019382, 1.443080]`；只有3个来源，区间较宽。

三维有效性只覆盖一个候选PF自然EOS样例：153个Position与153个Feature token解码为11,286个Gaussian，六个`384×384`视图的RGB和alpha均有限并已保存，alpha coverage为`0.050615`。`view_00`存在非空内容但形体稀疏，只支持单chunk解码和渲染有效，不能说明重建质量提高。旧`full3d_validity_run01`的一对一数量假设失败记录保持不变；`run02`所用VQ路径三级上采样、总因子8，不能认为Gaussian与Position数量固定相等。数据见[natural EOS审计](artifacts/pf2x_breakthrough_20261021/natural_eos_run01/independent_audit.json)、[run02元数据](artifacts/pf2x_breakthrough_20261021/full3d_validity_run02/metadata.json)及[视图0](artifacts/pf2x_breakthrough_20261021/full3d_validity_run02/views/view_00.png)。原512-token独立矩阵与单前缀语义数值材料仍见[原汇总](artifacts/pf2x_breakthrough_20261021/independent_confirm_run01/summary.json)、[原审计](artifacts/pf2x_breakthrough_20261021/independent_confirm_run01/independent_audit.json)和[原数值验证](artifacts/pf2x_breakthrough_20261021/independent_confirm_run01/verification.json)。

### 8.13 F-aware checkpoint 多 seed/长度解码验证（2026-10-06）

验证入口与 S12.2 相同，比较两个 16 step checkpoint：F0.3/P0.3 候选和 F0/P0 对照。矩阵覆盖 seed 0/1/2、请求长度 256/1024/2048/4096，共 24 个运行；配置为 CUDA、`temperature=0.9`、`top_p=0.9`、`warmup=min(128, length/2)`、`feature_shadow=True`。聚合文件为 `/root/workspace/GaussianGPT/profiling/s12_faware_matrix/decoder_grid_summary.json`。

验证结果：

- 候选 route speedup 按长度为 `1.01668/1.03987/1.03951/1.09685`，对照为 `1.02119/1.04518/1.03844/1.09708`；逐 seed 差值没有固定方向。
- 候选 Position 接受比例为 `0.91406/0.95898/0.96973/0.97579`，对照为 `0.91406/0.95964/0.96940/0.97755`，两条路径的 Position 行为基本一致。
- 候选 Feature shadow 接受比例在 2048/4096 为 `0.40466/0.42046`，对照为 `0.40103/0.40416`；这只是旁路观测，不能替代 joint `[P,F]` acceptance。
- 4096 请求的实际长度为候选 `3319/3731/2339`、对照 `3319/4096/2339`；EOS 字段没有一条为 true，长度差异按 JSON 原值保留，未作 EOS 解释。
- 记录中的 cache 计数随实际运行长度增长，256/1024/2048 的均值为 128/512/1024；4096 候选/对照均值为 1564.33/1625.33。这里只核对字段存在和数量趋势，没有逐条重新验证全部 cache invariant。

因此，F-aware 训练在这组矩阵中没有带来稳定 wall-clock 改善；略高的 Feature shadow 也不足以支付联合 Feature proposal 的额外成本。当前结论是保留该 checkpoint 作为研究对照，继续沿 Position-only structured 路线；真实 joint `[P,F]` 需要等 Feature acceptance、等长输出和端到端吞吐同时改善后再做。矩阵没有覆盖额外场景和重建 GIF。


## 2026-10-21 Feature 树路径与采样验证

采样模块9项用例通过：小词表精确输出律、无放回支持耗尽、全拒绝、EOS类型与W1 RNG状态。最终真实两前缀覆盖W1/W2、普通与图执行、各分支提交长度1–4及恢复，共1742项语义对照通过。Target logits仍要求逐元素容差与policy支持相同/TV≤1e-5；MLP q要求全有限、归一化和支持相同/TV≤1e-5；隐藏状态/KV采用全有限与两种相对误差≤2e-6，440项逐元素allclose未通过仍单列保留。最大概率TV为1.15744e-6。

CPU子代理从保存概率与logits复算688条Target和160条MLP q，12项通过，未独立复算隐藏状态与KV原张量。两轮解码各72条由另一标准库脚本复算，序列顺序、完整配对、吞吐、来源等权倍率与区间通过；W1/MLP各18/18输出计数相同，跨轮72/72相同。复算含重复路径观察，不视为1742个独立场景。范围仅两个真实前缀和三个开发场景，纯BOS、窄窗及任意输入仍未覆盖。详见[实现与数值范围](artifacts/feature_tree_20261021/IMPLEMENTATION.md)与[结果](artifacts/feature_tree_20261021/RESULTS.md)。
