# Feature EAGLE γ=5 定向实验（2026-10-19）

## 任务与设置

用户要求尝试 γ=5。本轮在 v3 已测 γ=5 的基础上，继续训练一个侧重第二处 Feature 的候选，比较新旧权重在 γ=3/5 下的在线表现。候选顺序为 P1、F1、P2、F2、P3。F1/F2 接受率均为真正到达该位置后的 accepted/verified，不含 Target 额外生成的 Feature。

Target 使用 `checkpoints/gpt_vfront.ckpt`，Ranker 使用原有 `ranker_trained_on_61001_61016.joblib`；Feature 网络维持 1L、256 宽、4 heads、1,572,864 个训练参数。原始 Target、Ranker、验证算法、采样温度 0.9、top-p 0.9 和完整历史的 Draft 状态重建方式保持原样。模型与算子使用 FP32，运行 GPU 为 RTX 4090。

起点权重为远端 `/root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/rollout/selected.pt`，新产物独立保存于 `/root/autodl-tmp/gaussiangpt_feature_eagle_gamma5_20261019/`。旧 PF 与此前全部研究权重均保留。

## 实际训练

沿用 v3 的 2,220 条条件样本：16 个训练来源、3 个验证来源，相互无重合。真实 P/F 样本提供有效 Feature CE；Ranker 条件和递推样本仅使用原始 Target 在相同候选历史上的分布监督。两处 Feature 之间的预测状态停止梯度传播，沿用 v3 递推实现。

本轮只增加 F2 分布损失权重与递推样本比例两个可选参数，默认值保持此前训练方式。双 Feature 样本的 KL 为 `(KL_F1 + 2 × KL_F2) / 3`；总损失为 `0.25 × 单处样本损失 + 0.75 × 双处样本损失`。CE 权重 0.1，P/F 隐状态损失权重 0.1，KL 权重 1。每批 8 条，AdamW 学习率 0.00002，最多 1,200 次更新；每 200 次更新重新生成 256 条当前 Draft 轨迹并取得 Target 标签。

每 100 次更新进行验证。另以起始 v3 生成 64 条固定双 Feature 验证轨迹，选模分数为 `0.5 × 单处重合度 + (F1重合度 + 2 × F2重合度) / 6`。在线比较开始后不再按其结果调整训练。

训练入口：

```bash
python train_pf_draft.py --feature-v3 train --v3-stage rollout \
  --data /root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/conditioned_dense_data \
  --init /root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/rollout/selected.pt \
  --output /root/autodl-tmp/gaussiangpt_feature_eagle_gamma5_20261019/training \
  --steps 1200 --lr 0.00002 --hidden-weight 0.1 --validate-every 100 \
  --v3-f2-weight 2 --v3-rollout-fraction 0.75 --v3-pool 256 --v3-refresh 200
```

共完成 1,200 次更新，训练主体耗时 272.18 秒，选择第 900 次更新的权重。第 1,200 次更新仍保存为 `last.pt`，在线运行使用第 900 次更新权重。

| 同一固定验证集 | 起始 v3 | 第 900 次更新 |
| --- | ---: | ---: |
| 单处 Feature 重合度 | 28.1878% | 28.5604% |
| 递推 F1 重合度 | 29.4450% | 30.2788% |
| 递推 F2 重合度 | 32.9231% | 34.2012% |
| 单处 KL | 2.21230 | 2.19164 |
| 递推 F1 KL | 1.72741 | 1.70253 |
| 递推 F2 KL | 1.59465 | 1.57557 |

上述重合度来自本轮相同的固定验证轨迹，不能作为在线接受率。原 v3 报告的双 Feature 验证轨迹由更早的起点生成，与本表不直接比较。

## 修改对象与验证

四个源码文件同步到远端 `/root/workspace/GaussianGPT/`，本地副本位于 `patches/eagle_pf_v2/`：

- `speculative/pf_distill.py`：增加各深度 KL 记录、F2 权重、递推损失比例和对应选模公式。
- `train_pf_draft.py`：提供 `--v3-f2-weight` 与 `--v3-rollout-fraction`，原默认训练方式继续可用。
- `eval_pf_draft.py`：提供指定 γ 列表和参照权重名称，结果保存权重路径；对指定 γ 逐一验证贪心输出。
- `test_pf_eagle.py`：验证默认 KL 一致性及双倍 F2 权重公式。

九项核心测试通过，包含实际采样分布、增量状态、候选长度 1–5、EOS、待消费 Feature 与训练梯度。四个文件的 Python 语法验证通过。修改前源码保存在远端实验目录的 `source_before/`；本地变更文本为 `artifacts/feature_eagle_gamma5_20261019/source_changes.diff`。

Target 核心、Draft 网络结构、Ranker 与 `speculative/pf_engine.py` 本轮没有修改，没有安装依赖或提交权重文件。

## 在线比较方法

使用三个既有开发来源，每个来源取 64/256 两种真实前缀，种子为 34701、34702、34703、34704，每次最多生成 256 个新 token。六条路径为 Target、起始 v3 γ=3/5、新候选 γ=3/5、旧 PF，共 144 次。每组轮换路径顺序，16-token 预热不计入结果。吞吐按各路径实际生成总 token 除以总耗时汇总，包含前缀处理、状态重建、提案、验证与恢复；模型加载不计入。

开发来源与训练、验证来源隔离，但已用于此前研究，不能称为新的正式确认数据。原有保留测试来源未用于训练、选模或本轮评价。分项计时另取一条 64-token 轨迹，并验证开启计时后输出逐 token 相同。

```bash
python eval_pf_draft.py \
  --training /root/autodl-tmp/gaussiangpt_eagle_pf_v2_20261019/training_extended \
  --feature-only \
  --feature-checkpoint /root/autodl-tmp/gaussiangpt_feature_eagle_gamma5_20261019/training/selected.pt \
  --reference-feature /root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/rollout/selected.pt \
  --reference-label feature_v3 --gammas 3 5 \
  --output /root/autodl-tmp/gaussiangpt_feature_eagle_gamma5_20261019/online \
  --scenes 3 --prefixes 64 256 --seeds 34701 34702 34703 34704 \
  --budget 256 --old-pf
```

## 最终在线结果

144/144 次运行完成，每条路径 24 次。接受计数如下，吞吐单位为 token/s：

| 路径 | F1 接受/到达 | F1 接受率 | F2 接受/到达 | F2 接受率 | 吞吐 | 相对 Target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target | — | — | — | — | 72.814 | 1.000× |
| 原 v3 γ=3 | 588/2294 | 25.63% | — | — | 97.434 | 1.338× |
| 新权重 γ=3 | 646/2229 | 28.98% | — | — | 98.680 | 1.355× |
| 原 v3 γ=5 | 517/2058 | 25.12% | 127/472 | 26.91% | 87.237 | 1.198× |
| 新权重 γ=5 | 543/2159 | 25.15% | 136/492 | 27.64% | 87.788 | 1.206× |
| 旧 PF | 780/2045 | 38.14% | — | — | 124.493 | 1.710× |

本轮种子数量和生成预算与 v3 报告不同，表中接受率与上轮 128-token 结果不直接比较。新 γ=5 的 F2 增加 0.7355 个百分点，F1 近乎相同；总体吞吐仅增加 0.6317%。新 γ=5 相对原 v3 γ=3 为 0.9010×，三个来源分别为 0.8723×、0.9261×、0.9034×，均较慢。新权重 γ=3 相对原 v3 γ=3 为 1.0128×，本轮没有据此替换 v3 权重。

### 提前 EOS 与同长度子集

原 v3 γ=5 在来源 `030345aa-f00e-4922-b50f-b56055702045` 的两个组合提前 EOS：前缀 64、seed34701 生成 203 token；前缀 256、seed34703 生成 25 token。新 γ=5 在同来源前缀 256、seed34704 生成 225 token 后 EOS。其余运行均为 256 token。主表使用各方法实际 token/实际时间，提前 EOS 会改变上下文长度构成。

作为补充，去除上述三个完整的场景×前缀×seed 组合，保留六条路径全部达到 256 token 的 21 组。原 v3 γ=5 / 新 γ=5 / 原 v3 γ=3 的吞吐分别为 88.675 / 87.697 / 97.852 token/s；新 γ=5 相对原 γ=5 为 0.9890×。该子集是在运行结束后按长度选取的描述性比较。两种汇总均没有显示 γ=5 值得替换 γ=3，不能把主表 0.63% 的差异视为可靠提速。

### 递推成本

新 γ=5 共提出 2,368 个 F2，其中 492 个走到接受判定，占 20.78%；其余提案的 Draft 计算已经发生。相对新 γ=3，每次验证的平均推进量只从 2.525 增至 2.563 token。额外递推带来的推进量尚未覆盖其计算成本。

64-token 分项单例中，新 γ=3 / 新 γ=5 的 Draft P 状态更新为 20.24 / 38.46 ms，F 状态更新为 37.49 / 54.79 ms；总时间为 641.09 / 675.35 ms。该单例只说明所测轨迹上的成本分布，不作为全池速度估计。六条路径的计时重放均逐 token 复现普通输出。真实 Target 的分块分布最大总变差为 2.1663e-6；γ=3/5 贪心输出与 Target 一致。

## 产物与结论

远端实验根目录 `/root/autodl-tmp/gaussiangpt_feature_eagle_gamma5_20261019/` 包含 `training/`、`online/`、`selected_weights.json` 与 `source_before/`。训练选中的第 900 次更新权重保留为研究候选。日志为 `/root/autodl-tmp/eagle_gamma5_train_20261019.log` 和 `/root/autodl-tmp/eagle_gamma5_online_20261019.log`。

本地小型证据在 [artifacts/feature_eagle_gamma5_20261019/](artifacts/feature_eagle_gamma5_20261019/)：训练配置、初始与逐次验证、选中权重说明、完整在线逐次记录、汇总、分项计时、数值验证、按来源与同长度子集统计，以及源码变更文本。完整权重仍在远端。

结论：完成 γ=5 定向训练和同轮比较，F2 验证重合度的小幅改善仅对应很小的在线接受率变化，没有获得可靠额外速度收益。研究 EAGLE 继续保留原 v3 γ=3，整体速度仍以旧 PF 更高。本轮未新增三维导出，token 解码结果不能宣称完整流程或质量改善。来源仅三个、开发数据曾被历史研究使用、单个训练种子、F2 到达数仍有限；本轮没有开展更长训练或扩大网络。
