# GaussianGPT Feature EAGLE v4 性能诊断（2026-10-19）

## 结论

本轮选择了持久复用 Draft KV，减少 Feature EAGLE 每轮重放整段历史的计算。每 1,024 个新 Token 的 Draft 输入，从 v3 诊断运行的 345,971 个降到 v4 的 2,847 个，减少 99.18%；三种子普通运行的 P/F 提案数、接受数和完整输出逐 token 相同。

这些变化没有形成稳定的端到端提速。三次普通运行的吞吐中位数，v3 为 96.39 token/s，v4 为 95.15 token/s；三次合计吞吐从 v3 的 96.60 到 v4 的 95.42 token/s，下降 1.22%，三个同种子差异为 +0.94%、-3.11%、-1.28%。计时运行中，Draft 重放减少约 431 ms，提交已接受状态增加约 698 ms，v4 总时间比同条件 v3 多约 281 ms。当前保留旧 PF 作为最快方案，持久 Draft KV 只通过显式参数启用。

## 范围与实现

目标模型、Position Ranker、1L-256 Feature Draft、γ=3、采样和 Target 接受流程保持不变；没有训练 GaussianGPT、VQ-VAE、Ranker 或 Draft。v4 的改变是可选的 Draft 状态缓存：保留此前已提交的 Target 前缀状态；提案被拒绝后截短缓存；接受后仅将新增且已提交的 Token 与 Target 隐藏状态追加到 Draft KV。坐标根据局部 Position/Feature 后缀重建。

| 文件 | 修改内容 | 性能关联 |
| --- | --- | --- |
| `speculative/pf_engine.py` | `decode(..., reuse_draft_cache=False)` 新增可选持久 Draft KV，记录 Draft 输入量及状态追加量；补充 Ranker、P/F Draft、Feature 抽样、Target 验证、恢复阶段计时。 | 不改变默认路径；启用时减少历史重放，但每轮需处理已提交状态。 |
| `eval_pf_draft.py` | 增加 `--profile-budget`、`--reuse-draft-cache`，按硬件能力记录 FA3 或 PyTorch SDPA 路径；保存预热、主测和 profile 参考重放。 | 支持对照和分项测时；最终两批数据统一记录 RTX 4090 实际使用的 PyTorch SDPA 路径。 |
| `test_pf_eagle.py` | 增加持久 KV 与完整历史重放的输出一致性测试，以及局部坐标与全上下文坐标的一致性测试。 | 覆盖接受后追加状态、拒绝后截短缓存及坐标对应关系。 |

源代码运行位置：远端 `/root/workspace/GaussianGPT/`；本地源码镜像：`/Users/lief/Documents/ChatGPT/GaussianGPT/patches/eagle_pf_v2/`。v3 执行报告：[`FEATURE_EAGLE_V3_2026-10-19.md`](FEATURE_EAGLE_V3_2026-10-19.md)。Target 权重为 `/root/workspace/GaussianGPT/checkpoints/gpt_vfront.ckpt`；Feature 权重为 `/root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/rollout/selected.pt`；旧 PF 权重为 `/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt`。

## 测试条件与复现

在 NVIDIA RTX 4090 上，固定同一真实场景、256-token 前缀、FP32、batch 1、温度 0.9、top-p 0.9。普通吞吐使用三个种子 `34901, 34902, 34903`，每次生成恰好 1,024 个 Token，均未提前 EOS；每个方法另有一次 16-token 预热。分项计时使用 seed `51697`、1,024 个 Token。CSV 收录 54 次运行：普通测量、预热、未插装的 profile 参考重放和插装 profile；中位吞吐只由普通测量计算。报告不计模型加载、场景解码或渲染。

远端命令使用 `/root/miniconda3/envs/gaussiangpt/bin/python`，在 `/root/workspace/GaussianGPT` 执行：

```bash
/root/miniconda3/envs/gaussiangpt/bin/python eval_pf_draft.py \
  --training /root/autodl-tmp/gaussiangpt_eagle_pf_v2_20261019/training_extended \
  --feature-only \
  --feature-checkpoint /root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/rollout/selected.pt \
  --gammas 1 3 \
  --output /root/autodl-tmp/gaussiangpt_feature_eagle_v4_20261019/final_after \
  --scenes 1 --prefixes 256 --seeds 34901 34902 34903 \
  --budget 1024 --profile-budget 1024 --old-pf --reuse-draft-cache
```

v3 对照使用同一命令，输出目录为 `/root/autodl-tmp/gaussiangpt_feature_eagle_v4_20261019/final_before`，移除 `--reuse-draft-cache`。完整运行日志分别在 `/root/autodl-tmp/eagle_v4_final_before_20261019.log` 与 `/root/autodl-tmp/eagle_v4_final_after_20261019.log`；每个输出目录含 `records.json`、`summary.json`、`profiles.json`、`warmups.json` 和 `profile_references.json`。

RTX 4090 的计算能力低于 FA3 加载器要求的 sm90，因此 v3 与 v4 实际都使用 PyTorch SDPA。旧记录保存的 `flash-attn` 是元数据标签错误，实际注意力实现没有切换。

## 普通吞吐与接受计数

下表采用执行 v4 的同一批次数据；吞吐中位数和 wall 秒中位数分别从三个逐次结果计算。接受率按接受/已验证候选计算。旧 PF 的逐次记录没有与 EAGLE 同定义的 `verify_calls`，所以其普通运行推进量留空；单次分项运行记录了 360 次 Target 验证调用。

| 方法 | token/s 中位数 | 1,024 Token wall 秒中位数 | P 接受/验证 | F1 接受/验证 | F2 接受/验证 | Token/Target 验证 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ranker-only γ=1 | 102.92 | 9.950 | 1,476/1,536 (96.09%) | 0/0 | 0/0 | 2.000 |
| v3 γ=3 | 96.39 | 10.624 | 1,474/1,536 (95.96%) | 314/1,166 (26.93%) | 0/0 | 2.514 |
| 旧 PF | 119.09 | 8.598 | 1,450/1,536 (94.40%) | 334/1,118 (29.87%) | 0/0 | — |
| v4 γ=3 | 95.15 | 10.762 | 1,474/1,536 (95.96%) | 314/1,166 (26.93%) | 0/0 | 2.514 |

合并 3 次完整 1,024-token 运行后，Target、Ranker-only、v3、旧 PF、v4 的吞吐依次为 69.80、101.78、96.60、117.03、95.42 token/s。v4 相对 v3 的三个同种子吞吐变化为 +0.94%、-3.11%、-1.28%；输出序列和 P/F 计数逐种子完全一致。当前样本不支持 v4 提速。F2 为 0/0，因本配置每轮最多提出一个 Feature 候选。

## 分项计时

分项计时使用另一次 1,024-token 运行，所有 EAGLE 阶段在同步设备后按 CPU wall 计时；`其他` 为未列阶段和计时残量。普通吞吐不插入逐阶段同步。旧 PF 由其原有 stage profiler 记录 host 秒；CUDA event 数值与 host 秒分开，不能相加。

| 方法 | Ranker ms | Draft P ms | Draft F ms | Feature 抽样 ms | Target 验证 ms | 恢复 ms | 其他 ms | 分项 wall ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ranker-only γ=1 | 126.6 | 0 | 0 | 0 | 9,011.5 | 527.2 | 930.1 | 10,595.4 |
| v3 γ=3 | 226.4 | 384.2 | 748.7 | 172.5 | 7,787.4 | 299.1 | 1,950.6 | 11,568.9 |
| 旧 PF | 117.2 | — | — | — | 6,513.7 | — | — | 7,808.9 |
| v4 γ=3 | 226.6 | 388.9 | 770.5 | 173.7 | 7,805.8 | 303.4 | 2,180.6 | 11,849.5 |

v3 的 Target 验证占分项 wall 的 67.3%，v4 为 65.9%；v4 与 v3 的 Target 验证调用次数相同，均为 438。旧 PF 的 profiler 在这一条样本中记录 360 次 Target 验证、1,438 个验证输入 Token。计时数据提示验证路径占比最大，缓存改动本身没有减少验证工作。Draft 全历史重放由 431.9 ms 降至 1.2 ms，但提交已接受状态增加 698.3 ms；总计时反而增加 280.6 ms。计时插装不同，阶段结果用于定位，不作为各方案普通吞吐差异的精确归因。

## 目标分布与质量范围

v4 沿用原有 Target/Draft 实际分布抽样、按概率比接受及 residual 抽样逻辑，本轮没有改动 Target 概率或 Feature 候选的接受规则。新增测试与三种子逐次输出比较均确认，持久 KV 路径与 v3 输出一致。跨不同方法的随机采样序列没有用于准确性结论。三维解码、渲染质量和多场景正式评价未评测。

## 后续

1. 旧 PF 继续作为当前最快路线；v4 缓存参数保持关闭。
2. 若继续优化，先分析不同解码器的 Target 验证批次长度、调用数与单次耗时，再决定是否需要更改提案策略。
3. 当前不追加训练；本轮只测了一个场景和三个种子，结论适用于此开发比较范围。

本地网页审阅包：[`feature_eagle_v4_review.zip`](artifacts/feature_eagle_v4_review.zip)。
