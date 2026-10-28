# Position Transition G0：在线评价入口设计

本轮只完成 CPU 规划器与评价接口，未访问远端、未载入模型，也未使用 GPU。已读取本地 [source manifest](source_manifest.json)：50条来源按16 train、4 val、10 dev、20 confirmation 划分；20个 confirmation 来源在历史 Stage E/F 中出现过，本轮不用于选参。G3 使用按 scene_id 升序排列的前3个 dev 来源，G4 使用全部10个 dev 来源。模型权重及 B/C 权重的实际运行路径仍须由主代理提供；脚本不填入推测路径。

## 本地源码对应

只读依据为 [source_before](source_before/) 内的归档源码与既有 [空间在线 runner](../../../../patches/spatial_embedding_20261020/compare_online.py)。当前快照中的 `draft_a.pf_feature_unified.decode(target, adapter, ranker, ranks, prompt, budget, seed)` 会返回 `(stats, output)`；`MLPAdapter.propose(prefix, p1, h_prev, rng)` 沿用现有统一解码器。B/C 只需提供同一 adapter 生命周期：`prepare(prompt, capacity)`、`propose(prefix, p1, h_prev, rng)`、`rewind(keep)`、`close()`。adapter 的 `propose` 先执行原 MLP 一次，再产生 B 或 C 的残差，并对4096类 Feature 执行一次 temperature 0.9、top-p 0.9 policy。Target、Ranker、位置校验、采样与恢复仍走同一 decoder。

Target-only 使用 `draft_a.pf_ranker_eval.decode(..., k=0, ...)`。归档函数已构造 `output = prefix[length:]`，但返回值只有统计；`expose_target_output()` 只复制函数源码，把返回改为 `(stats, output)`，不动归档或生产文件。

统一 PF decoder 会在 Target 验证期间用 `CaptureLastBlock` 取得正常解码所需的 Hidden 状态。正式计时不运行额外 teacher 收集器、不保存 `hF/hP` 教师数据，也不开启 teacher Hook。计时边界由 `timed_decode()` 提供：外部 CUDA 同步、`perf_counter`、完整方法调用、CUDA 同步。调用函数应包含 KV 创建、真实前缀预填、Ranker、adapter、Target 验证、残差抽样、缓存恢复和 close；四种方法共用相同边界。

## 固定矩阵

矩阵方法次序为 T、W1、B、C；前缀取64/256，配对种子为39101/39102，组内四法共用实际 prompt 和 seed。

| 阶段 | 来源数 | 前缀×种子 | 配对组 | 每组方法数 | 调用数 | 每次新 Token 上限 | 总墙钟上限 |
|---|---:|---:|---:|---:|---:|---:|---:|
| G3 | 3个由主代理确认的既有开发来源 | 2×2 | 12 | 4 | 48 | 256 | 900秒 |
| G4 | 10个由主代理确认的既有开发来源 | 2×2 | 40 | 4 | 160 | 512 | 1500秒 |

按来源、前缀、种子的顺序建立配对组；四种循环次序依次轮换。G3 中每种方法在每个调用位置各出现3次，G4 各出现10次。该安排平衡每种方法先后位置。G3 到 G4 沿用网页计划中的固定两个 seed；实际来源顺序由主代理提供。

G3 停止条件按计划执行：C/W1 配对吞吐点估计低于1.05时不进入G4。G4 的继续条件为 C/W1 至少1.08且 C/Target 至少1.90；任一不满足即停止。来源等权区间会报告，但不替代网页计划中的点估计条件。G5 的来源确认与三维渲染不属于本轮实现。

## 零残差重放

G3 前的接口为一个由主代理提供的来源、前缀0/64/256、生成上限16/64、固定seed 39101。每个组合依序运行 W1、B残差置零、C残差置零，共18次。前缀0使用 `[12096]`；其余前缀使用 `[12096] + 实际 P/F Token[:prefix]`，长度分别为65和257。各 prompt 必须以完整 P/F 对结束。

B/C 仍执行与正式版本相同的 MLP/网络路径，只把残差倍率设为0；q 应复用旧 MLP logits 的原 policy。相同 prompt、seed 和预算下，逐 Token 输出、Target调用及输入量、Ranker调用、Position/Feature 提案/验证/接受计数、round 计数和缓存恢复 keep trace 必须与 W1 一致。比较结果保存 `equal` 与逐字段差异。速度数据不参与此项通过判定。

`capture_rewind_trace(adapter)` 只包装该 adapter 实例的 `rewind(keep)`。统一 decoder 在 Target 缓存截短后以相同 keep 值调用 adapter，因此 trace 能记录拒绝恢复位置，且不改全局 decoder。prefix0 的最初 Target Position/Feature 生成发生在完整提案循环前，重放仍比较完整输出和总计数。

## 记录与统计接口

`timed_decode()` 将完整 prompt 与完整生成 Token 列表写进每条 JSON 行，同时记录方法、source ID、prefix、seed、预算、执行位置、EOS、最后 Token、实际长度、外层同步 wall、统计计数和 `rewind_keep_trace`。EOS ID 为12096，只允许出现在输出末尾。每完成一个方法调用就 flush 一行，供后续保留已完成记录。

G3/G4 的来源等权主指标为：在每个配对组计算 log(TPS候选/TPS参照)，先对来源内的四个 prefix/seed 组合求平均，再对来源等权平均并指数化。报告 W1/T、B/W1、C/W1、C/T；另列各来源值、来源 bootstrap 95%区间及共同满预算子集。共同满预算子集对相同输出长度以外层时间比计算。合并 Token/秒作为次要口径，同时给出自然 EOS 数和实际输出长度分布。bootstrap 固定10000次、seed 381012，作为此实现的可重复统计设置。

缓存恢复次数用 `verified - accepted` 推得，并标明推导口径；精确 keep trace 单独保留。输出完整 token ID，不用 token 数替代序列。teacher 数据收集、teacher 捕获与额外分项同步均不在正式方法计时中。

## CLI 与运行依赖

当前 CLI 只写计划 JSON，不载入模型：

```bash
python patches/pf_position_transition/eval_transition.py \
  --phase g3 \
  --source-manifest docs/project/artifacts/position_transition_20261020/source_manifest.json \
  --out <new-plan-json>

python patches/pf_position_transition/eval_transition.py \
  --phase zero-replay \
  --source-manifest docs/project/artifacts/position_transition_20261020/source_manifest.json \
  --seed 39101 \
  --out <new-replay-plan-json>
```

G4 使用 `--phase g4`。脚本读取manifest中的split与scene_id，仅取dev来源；计划结果记录实际manifest文件位置。模型权重与 B/C checkpoint 字段会显式留空，等待主代理提供真实路径。`execute_matrix()` 与 `execute_zero_replay()` 通过 prompt provider、方法回调和 CUDA 同步回调接入主代理确认后的运行环境；方法回调需返回 `stats`、完整 `output_tokens` 和 `rewind_keep_trace`。这些回调与真实远端模型路径仍待主代理提供，当前没有声称线上入口已接通。

只读依据路径：`docs/project/artifacts/position_transition_20261020/source_before/` 与 [SOURCE_SPLIT.md](SOURCE_SPLIT.md)。本地归档中可见旧 Target/Ranker/MLP 的接口；已选 B/C 权重和实际运行环境路径仍由主代理统一提供。
