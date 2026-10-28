# GaussianGPT Feature EAGLE v4 — Web Review

## 1. 目标与结论

- 目标：说明 Feature EAGLE v3 γ=3 慢于旧 PF 的成本，并验证一项针对性缓存改动。
- 结论：旧 PF 继续作为当前最快方案；持久 Draft KV 保留为显式可选路径，当前不启用。
- 关键结果：三次普通 1,024-token 测量的 v4 中位吞吐为 95.15 token/s，v3 为 96.39 token/s。合并吞吐分别为 95.42/96.60 token/s；配对种子变化为 +0.94%、-3.11%、-1.28%。三次合并吞吐下降 1.22%，没有端到端提速证据。

## 2. 实际阅读的代码与修改位置

| 文件 | 函数/区域 | 修改前状态 | 本次改动 | 性能关联 |
| --- | --- | --- | --- | --- |
| `speculative/pf_engine.py` | `decode`、`append_actual_cache_tokens` | 每轮从完整已消费历史重建 Draft KV。 | 增加默认关闭的 `reuse_draft_cache`，保留已提交 Draft 状态；拒绝时截短，接受后用 Target 隐藏状态及局部坐标追加。增加分阶段计时与 Draft 输入计数。 | profile 输入 2,847 对 345,971，减少 99.18%；状态提交 698.3 ms。 |
| `eval_pf_draft.py` | 参数解析、`run`、输出记录 | profile 运行的参考重放和预热没有逐次保存。 | 增加 `--profile-budget`、`--reuse-draft-cache`，按 `HAS_FA3` 记录实际 Attention 路径，保存预热和 profile 参考重放。 | 54 行 CSV 覆盖每个已记录预热、普通测量、参考重放和 profile。 |
| `test_pf_eagle.py` | KV 复用测试 | 没有跨 γ 与坐标重放比较。 | 新增持久缓存与完整历史重放的输出一致测试，以及局部坐标一致测试。 | 覆盖接受后追加、拒绝后截短和坐标对应关系。 |

v3 报告：`docs/project/FEATURE_EAGLE_V3_2026-10-19.md`。Target 权重：`/root/workspace/GaussianGPT/checkpoints/gpt_vfront.ckpt`；Feature Draft：`/root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/rollout/selected.pt`；旧 PF：`/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt`。本轮新增训练步数为 0。

## 3. 测试条件

- GPU：NVIDIA GeForce RTX 4090；Target 为 24 层模型；精度 FP32。
- Attention 路径：PyTorch SDPA fallback。最终两批 JSON 的 `attention_backend` 字段相同；运行代码根据 `HAS_FA3` 记录，RTX 4090 未加载 FA3。
- Batch=1，前缀长度 256，temperature=0.9，top-p=0.9，同一场景 `003ac11d-2abc-44f8-9836-4354e7dfa543`。普通运行使用 seeds `34901, 34902, 34903`，每个方法另有 16-token warmup；profile 与未插装参考重放使用 seed `51697`、1,024 token。
- AR wall 只计自回归 token 生成，不含模型加载、VQ 解码、场景导出或渲染。54 条 CSV 分为 steady-state、warmup、profile_reference 和 profile；性能中位数只使用 steady-state 三次记录。

远端目录 `/root/workspace/GaussianGPT`，解释器 `/root/miniconda3/envs/gaussiangpt/bin/python`。实际 v4 命令：

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

v3 对照移除 `--reuse-draft-cache`，输出目录改为 `/root/autodl-tmp/gaussiangpt_feature_eagle_v4_20261019/final_before`。日志分别为 `/root/autodl-tmp/eagle_v4_final_before_20261019.log` 与 `/root/autodl-tmp/eagle_v4_final_after_20261019.log`；逐次 JSON 各在相同目录的 `records.json`、`warmups.json`、`profile_references.json`、`profiles.json`。

## 4. 真实对照结果

下表使用 v4 批次的三次普通运行。吞吐与 wall 秒中位数分别根据逐次结果计算，F 接受率为 `F1_accepted / F1_verified`。F2 为 0/0，因为配置只提出一个 Feature 候选。

| 方法 | token/s 中位数 | 1,024 token wall 秒中位数 | P接受/验证 | F1接受/验证 | F2接受/验证 | 新 token / Target 验证 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ranker-only γ=1 | 102.92 | 9.950 | 1,476/1,536 = 96.09% | 0/0 | 0/0 | 2.000 |
| v3 γ=3 | 96.39 | 10.624 | 1,474/1,536 = 95.96% | 314/1,166 = 26.93% | 0/0 | 2.514 |
| 旧 PF | 119.09 | 8.598 | 1,450/1,536 = 94.40% | 334/1,118 = 29.87% | 0/0 | — |
| v4 γ=3 | 95.15 | 10.762 | 1,474/1,536 = 95.96% | 314/1,166 = 26.93% | 0/0 | 2.514 |

合并吞吐：Target 69.80、Ranker-only 101.78、v3 96.60、旧 PF 117.03、v4 95.42 token/s。全部 27 次普通运行都生成 1,024 token，未提前 EOS。旧 PF 普通记录没有同定义的 `verify_calls`；其单次 profile 记录 360 次 Target 验证/1,024 token。

## 5. 性能时间分解与最主要瓶颈

下表来自单条独立 1,024-token profile。EAGLE 阶段在 CUDA 同步边界内按 CPU wall 计时，`其他` 是未列阶段和残量。旧 PF 行只填该实现记录的 host 秒；CUDA event 与 host 秒分开保存，未做相加。

| 方法 | Ranker ms | Draft P ms | Draft F ms | Feature 抽样 ms | Target 验证 ms | 恢复 ms | 其他 ms | Profile wall ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ranker-only γ=1 | 126.6 | 0.0 | 0.0 | 0.0 | 9011.5 | 527.2 | 930.1 | 10595.4 |
| v3 γ=3 | 226.4 | 384.2 | 748.7 | 172.5 | 7787.4 | 299.1 | 1950.6 | 11568.9 |
| 旧 PF | 117.2 | — | — | — | 6513.7 | — | — | 7808.9 |
| v4 γ=3 | 226.6 | 388.9 | 770.5 | 173.7 | 7805.8 | 303.4 | 2180.6 | 11849.5 |

Target 验证占 v3/v4 profile wall 的 67.3%/65.9%。二者各 438 次验证。旧 PF profile 记录 360 次 Target 验证、1,438 个验证输入 Token。Draft 输入降至 2,847，但状态提交时间高于节省的历史重放时间；缓存路径没有减少 Target 验证工作。

## 6. 实际改动与效果关联

- `reuse_draft_cache=False` 是默认值；本次只在 v4 命令打开。
- 三个配对种子的 v3/v4 输出逐 token 一致，P/F 提案、接受、拒绝和 Bonus 计数相同。v4 Feature 接受率为 314/1166 = 26.93%。
- Draft 历史输入减少 99.18%；分项 wall 中 Draft 全历史重放为 431.9 ms，v4 为 1.2 ms；v4 提交状态花 698.3 ms。普通吞吐中位数下降 1.28%。
- 无新增训练、依赖、Target 前向结构或权重。远端 `python -m unittest test_pf_eagle -v`：11 项通过；本地及远端 `py_compile` 通过。

## 7. 分布与质量边界

v4 沿用 Target 与 Draft 的 temperature/top-p policy、实际 q/p 抽样、按概率比接受及 residual 分布抽样；本次没有更改接受逻辑。测试覆盖随机采样输出一致、γ=3/5 KV 长度及局部坐标。不同方法间不以随机输出逐 Token 相同作为要求。本轮未做 3D 解码、渲染质量或独立质量评测。

## 8. 尚未解决的问题与下一步

1. v4 的 Target 验证占 profile wall 约 66%；缓存路径没有改变这一成本。
2. 旧 PF 中位吞吐比 v4 高 25.2%；v4 不应替换旧 PF。
3. 本轮只覆盖一个场景、一个前缀和三个普通吞吐种子，结论适用于此开发测量范围；本轮不追加训练。
