# 运行图统计说明

`all_methods_smoke_run01` 含一个来源、prefix 64/256、11 种方法、两种执行方式，共 44 条记录；每条输出 32 token，未遇到 EOS。CPU 独立复算显示 11 种优化方式都实际重放了 CUDA Graph；每种方式的两个前缀请求都新增了图，因而本次没有可用于比较稳定执行速度的配对请求。

图按 query token 数和 read bucket 分别建立。prefix 64 使用 read bucket 128，prefix 256 使用 read bucket 512。`runtime_created=false` 只说明 Python runtime 已存在；该行若仍有 `new_graph_events`，仍然包含图构建时间，所以稳定子集必须同时要求 `runtime_created=false`、新增图事件为空，并确认至少发生一次 graph forward。

| 方法 | 图数 | Graph forward 次数与 query 长度 | 回退 | 捕获 / 图预热时间 | 优化 / 原执行吞吐比 |
| --- | ---: | --- | ---: | ---: | ---: |
| target | 4 | T1×64 | 2 | 748 / 121 ms | 0.787 |
| pf_mlp | 5 | T4×23、T2×2、T1×2 | 2 | 979 / 160 ms | 0.439 |
| ranker_only | 3 | T2×32、T1×2 | 2 | 581 / 93 ms | 0.741 |
| eagle_v3_g3 | 8 | T4×22、T3×4、T1×3 | 2 | 1,605 / 254 ms | 0.337 |
| eagle_v3_g5 | 8 | T6×24、T5×3、T3×1、T1×3 | 2 | 1,551 / 250 ms | 0.395 |
| eagle_v4_reuse_g3 | 8 | T4×22、T3×4、T1×3 | 2 | 1,547 / 273 ms | 0.339 |
| eagle_v2_joint_g3 | 8 | T4×28、T3×4、T1×30 | 2 | 1,604 / 249 ms | 0.518 |
| medusa_lowrank | 6 | T4×30、T2×2、T1×32 | 2 | 1,172 / 187 ms | 0.635 |
| hydra_conditional | 6 | T4×30、T2×2、T1×32 | 2 | 1,197 / 187 ms | 0.631 |
| d8_pf | 3 | T4×21、T1×2 | 2 | 589 / 93 ms | 0.653 |
| d8_joint | 4 | T3×31、T2×2；全词表读出 | 2 | 798 / 127 ms | 0.673 |

每种方式的两次回退都是长 prefill 的 `unsupported_query_tokens`，每次运行另有一次原始 forward；其余 32-token 生成调用都经过图，`disabled_graphs` 为空。各方法的 KV cache 都是 201,326,592 bytes；11 个 runtime 在测量期间分别保留缓存，合计 2,214,592,512 bytes。

两种前缀的图共捕获 12,370 ms、图预热 1,994 ms、准备 40 ms；优化臂外层墙钟总计 19.531 秒。捕获、图预热和准备都发生在同步外层解码计时内。第一次 16-token 预热使用另一批 runtime，随后释放；它们的图没有供正式记录复用。因 smoke 的两个 prefix 又落在不同 read bucket，不能用它估计图建立后的稳定收益。

这组小样本里，原执行吞吐第一是 `pf_mlp` 的 102.1 token/s，优化执行吞吐第一是 `ranker_only` 的 65.7 token/s；所有方法的优化 / 原执行配对比都低于 1。这个排序主要受两次请求都新建图的成本影响，不能用来判断稳定重放后的算法排序。每种方法只有两个配对请求且来自单一来源，来源抽样区间无法计算；该结果只证明运行路径已覆盖。

正式结果仍以所有同步外层墙钟为主，完整计入 runtime 创建、KV cache、图创建、prefill、草稿、验证、恢复和采样；不从主指标中扣除首次成本。优化执行 / 原执行的稳定子集要求优化臂无新图且发生过图重放，给出每方法的配对数、来源数和来源等权几何均值。优化方法 / 优化 Target 的稳定子集更严格，要求两臂都无新图且都发生过图重放。样本不足时明确标为不可计算；子集为空时不要以全量结果推断稳定收益。

为解释方法排序，正式附表宜按方法列出：T1/T2/T3/T4/T5/T6 与全词表路径的实际 graph forward 次数、各 read bucket 的图数、`graph_return_hidden` / `graph_full_forward` 次数、prefill 与其他回退数、图准备 / 预热 / 捕获耗时、KV cache 字节数。再与方法自己的 target/draft 调用数、提案 / 验证 / 接受数、接受长度、输出长度及 EOS 数并列；前一组描述执行路径，后一组描述算法工作量。

纯 CPU 汇总入口：

```bash
python patches/target_methods_20261021/runtime/diagnose_runtime.py \
  --run-dir docs/project/artifacts/target_methods_20261021/all_methods_smoke_run01 \
  --out docs/project/artifacts/target_methods_20261021/all_methods_smoke_run01/runtime_diagnostics_recomputed.json
```

脚本从既有 `plan.json`、`summary.json`、`records.jsonl` 和可选 `warmup.json` 读取数据，写出新的 `runtime_diagnostics.json`。输出保留优化执行 / 原执行结果与 `summary.json` 中优化方法 / 优化 Target 比率，并分别计算各自稳定配对子集；所有比率仍用同步外层墙钟，并带配对数和来源数。

## paired_pilot_run01

正式试跑记录完整：132 条运行，11 种方法、两种执行方式、3 个开发来源、prefix 64/256、各 1 个 seed，预算 256。每个方法主算法比都有 6 个配对、3 个来源；下表采用来源等权几何均值和来源抽样 95% 区间。稳定列只使用方法优化臂与优化 Target 臂都没有 runtime 新建、没有新增图事件且至少发生一次图重放的共同请求。

| 方法 / 优化 Target | 主算法比 [95% 区间] | 共同稳定比 [95% 区间] | 稳定配对 / 来源数 |
| --- | ---: | ---: | ---: |
| pf_mlp | 1.277 [1.138, 1.413] | 1.353 [1.296, 1.413] | 4 / 2 |
| ranker_only | 1.310 [1.281, 1.342] | 1.359 [1.307, 1.432] | 5 / 3 |
| eagle_v3_g3 | 0.815 [0.732, 0.919] | 0.860 [0.804, 0.919] | 4 / 2 |
| eagle_v3_g5 | 0.636 [0.593, 0.678] | 0.648 [0.638, 0.657] | 3 / 2 |
| eagle_v4_reuse_g3 | 0.794 [0.737, 0.883] | 0.824 [0.770, 0.883] | 4 / 2 |
| eagle_v2_joint_g3 | 0.548 [0.542, 0.555] | 0.551 [0.547, 0.555] | 4 / 2 |
| medusa_lowrank | 0.781 [0.770, 0.795] | 0.774 [0.770, 0.780] | 5 / 3 |
| hydra_conditional | 0.699 [0.594, 0.777] | 0.737 [0.733, 0.740] | 3 / 2 |
| d8_pf | 0.848 [0.837, 0.856] | 0.844 [0.837, 0.852] | 4 / 2 |
| d8_joint | 0.503 [0.422, 0.619] | 0.452 [0.421, 0.488] | 4 / 3 |

稳定配对子集共 40 条，单方法有 3–5 条，来源数为 2–3。主算法比的六个配对全部纳入；Target 有一次在 165 token 结束，其余五次和所有草稿方法均达到 256 token。主算法比保留实际输出长度；稳定列的配对数和区间应与主结果分开呈现。

11 个优化 runtime 都有实际图重放。Target 捕获 6 个图，重放 1,445 次；另有 6 次原始 forward，均为长 prefill 的 `unsupported_query_tokens` 回退。按 Target 总 forward 调用 1,451 次计，图覆盖 1,445 次（99.59%），生成 token 对应的 Target 调用全部经图。各方法回退数也都是 6，原因只有上述长 prefill；11 个 runtime 的 `disabled_graphs` 均为空。

| 方法 | 捕获图数 | Graph forward 次数 | 捕获 / 图预热时间 |
| --- | ---: | ---: | ---: |
| target | 6 | 1,445 | 1,223 / 189 ms |
| pf_mlp | 7 | 676 | 1,444 / 221 ms |
| ranker_only | 6 | 846 | 1,225 / 187 ms |
| eagle_v3_g3 | 10 | 668 | 2,098 / 331 ms |
| eagle_v3_g5 | 12 | 677 | 2,381 / 384 ms |
| eagle_v4_reuse_g3 | 10 | 668 | 1,998 / 319 ms |
| eagle_v2_joint_g3 | 10 | 1,526 | 1,976 / 321 ms |
| medusa_lowrank | 7 | 1,531 | 1,399 / 225 ms |
| hydra_conditional | 10 | 1,526 | 2,001 / 318 ms |
| d8_pf | 7 | 578 | 1,375 / 217 ms |
| d8_joint | 4 | 730 | 832 / 126 ms |

`d8_joint` 的 730 次图 forward 都走全词表读出；EAGLE 的 `return_hidden=True` 返回 final norm 后的 hidden，同时 runtime 另向 hook 发布 pre-finalnorm hidden，分别为 v3 g3 668 次、v3 g5 677 次、v4 668 次、v2 1,526 次。11 个方法各自仍计入首次建图时间，主算法比没有扣除捕获或图预热。

## optimized_main_run01

这轮为 optimized-only：220 条记录对应 11 种方法、10 个开发来源、prefix 64/256、每个条件 1 个 seed、预算 512。没有原执行臂，因此不计算优化执行 / 原执行比率。下表主比率取 `summary.json` 的优化方法 / 优化 Target 对照，稳定列按两臂都没有 runtime 新建、没有新增图事件且都至少发生一次图重放筛选；数字均为来源等权几何均值和来源抽样 95% 区间。

| 方法 / 优化 Target | 主比率 [95% 区间] | 主配对 / 来源数 | 共同稳定比 [95% 区间] | 稳定配对 / 来源数 |
| --- | ---: | ---: | ---: | ---: |
| pf_mlp | 1.451 [1.370, 1.523] | 20 / 10 | 1.484 [1.426, 1.539] | 18 / 9 |
| ranker_only | 1.371 [1.338, 1.397] | 20 / 10 | 1.386 [1.368, 1.402] | 18 / 9 |
| eagle_v3_g3 | 0.818 [0.798, 0.837] | 20 / 10 | 0.814 [0.794, 0.834] | 18 / 9 |
| eagle_v3_g5 | 0.670 [0.649, 0.691] | 20 / 10 | 0.666 [0.644, 0.687] | 18 / 9 |
| eagle_v4_reuse_g3 | 0.779 [0.759, 0.798] | 20 / 10 | 0.775 [0.755, 0.795] | 18 / 9 |
| eagle_v2_joint_g3 | 0.549 [0.544, 0.554] | 20 / 10 | 0.546 [0.543, 0.550] | 18 / 9 |
| medusa_lowrank | 0.725 [0.642, 0.774] | 20 / 10 | 0.724 [0.642, 0.773] | 19 / 10 |
| hydra_conditional | 0.681 [0.604, 0.729] | 20 / 10 | 0.684 [0.599, 0.733] | 16 / 9 |
| d8_pf | 0.899 [0.876, 0.923] | 20 / 10 | 0.902 [0.877, 0.929] | 19 / 10 |
| d8_joint | 0.492 [0.471, 0.512] | 20 / 10 | 0.479 [0.459, 0.496] | 18 / 10 |

`pf_mlp` 与 `ranker_only` 的主比率和共同稳定比都高于 1；其余八种方法均低于 1。Target 20 次都生成满 512 token；Medusa 与 Hydra 各有 1 次在 5 token 后结束，其余输出均为 512 token。主比率按实际生成长度计算。共同稳定子集总计 180 个方法配对，单方法有 16–19 个配对、9–10 个来源，适合作为主结果的补充。

Target 有 8 个已捕获图，graph forward 10,240 次；20 次长 prefill 各有一次 `unsupported_query_tokens` 回退并走原始 forward。10,260 次 Target forward 中有 10,240 次重放图，覆盖率 99.81%，生成 token 的 Target 调用全部经图。其余 10 个 runtime 也都有图调用；每种方法的同类 prefill 回退均为 20 次，没有其他回退原因，所有 `disabled_graphs` 均为空。

| 方法 | 捕获图数 | Graph forward 次数 | 捕获 / 图预热时间 |
| --- | ---: | ---: | ---: |
| target | 8 | 10,240 | 1,527 / 240 ms |
| pf_mlp | 9 | 4,207 | 1,781 / 280 ms |
| ranker_only | 8 | 5,514 | 1,543 / 251 ms |
| eagle_v3_g3 | 11 | 4,871 | 2,107 / 342 ms |
| eagle_v3_g5 | 12 | 4,608 | 2,366 / 377 ms |
| eagle_v4_reuse_g3 | 11 | 4,871 | 2,199 / 374 ms |
| eagle_v2_joint_g3 | 11 | 10,159 | 2,165 / 381 ms |
| medusa_lowrank | 9 | 9,681 | 1,800 / 286 ms |
| hydra_conditional | 12 | 9,713 | 2,439 / 378 ms |
| d8_pf | 9 | 3,767 | 1,722 / 279 ms |
| d8_joint | 5 | 4,660 | 1,032 / 159 ms |

此轮共建立 105 个图；捕获 20.680 秒、图预热 3.349 秒，都在对应解码请求的同步外层墙钟内。`d8_joint` 的 4,660 次图 forward 都走全词表读出；EAGLE 的 `return_hidden=True` 返回 final norm 后的 hidden，同时 runtime 另向 hook 发布 pre-finalnorm hidden，数量与各自 graph forward 相同。详细逐方法数据见 [runtime_diagnostics.json](optimized_main_run01/runtime_diagnostics.json)。
