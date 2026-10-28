# GaussianGPT 架构说明

## 2026-10-23 实验性概率树与同状态操作探针

新隔离目录patches/recovery_value_evidence_20261023使用纯Python浅树推理，从正式轻量上下文预测主/备用概率，按读长度操作费用选择零/单备用。Target标签只进入训练和观察记录。operation_probe为可选诊断回调；测量验证、非空路径提交、单步Target及EOS提交，退出恢复cache、树图、共享线性运行对象及捕获输出。正式计时不启用该回调。策略EOS质量按未命中费用近似，首次建图没有单独估值；原生产入口不变。结构合法与实际收益分开论证，见[结果](artifacts/recovery_value_evidence_20261023/RESULTS.md)。

## 2026-10-23 实验解码器的正式与观察路径

恢复价值实验的adaptive_decoder现在将正式选择输入与观察记录分开：正式路径仅构造标量、候选与图状态，空候选跳过回调；观察路径保留完整前缀、标签和事件。固定0/1动作与动态动作使用同一解码器和通用恢复运行对象，原W1/快速单备用保留参照入口。selection.py支持reference与compiled两种等价路径，compiled在创建时预读线性参数；模型改变后需重建选择器。成本解析使用实际计时状态，价值输入仍限定在验证前。未替换生产入口。实现与验证见[本轮报告](artifacts/recovery_value_execution_20261023/RESULTS.md)。

2026-10-21 新增三种提案侧专用图，隔离源码在patches/proposal_graphs_20261021。MLP捕获FeatureStateDraft前向，Medusa LowRank捕获heads.raw；输入复制到固定缓冲，输出在下一次同形状调用前消费。EAGLE捕获单token Draft递推与一维Feature读出头，长历史/T>1保留普通执行；按读长度分桶，固定staging KV从原缓存同步，连续同缓存递推可复用，回退/普通前向使同步状态失效。原decoder仍负责采样、接受、拒绝、缓存截短和Ranker。每个执行臂独立持有Target图与提案图，原权重和生产默认保留；首次建图进入外层同步墙钟。若外部同位置重写Draft KV，调用invalidate_cache(cache)。详见artifacts/proposal_graphs_20261021/PLAN.md。

2026-10-19 Feature EAGLE v4 为 `speculative/pf_engine.py` 增加可选持久 Draft KV。每轮按 Target 已消费前缀截短或追加 Draft 状态，局部重建新增 token 的坐标；默认关闭时沿用 v3 全历史重放。三种子运行表明 Draft 输入显著减少，但追加状态时间抵消节省，未形成稳定端到端收益。详见 [v4 性能诊断](FEATURE_EAGLE_V4_2026-10-19.md)。

2026-10-19 Feature EAGLE v3 沿用 v2 网络与验证，新增 `speculative/pf_distill.py`，在原训练入口提供条件样本、密集 Feature KL 与两处 Feature 递推。训练区分真实末尾 F 已由 Target 消费、或尚待消费；后一种先由 Draft 更新该 F 状态。相邻预测节点停止梯度传播，当前节点独立学习；Target 和 Ranker 参数不变。在线新增 F1/F2 与首次拒绝槽位计数，接受算法没有重写。详见 [v3 执行记录](FEATURE_EAGLE_V3_2026-10-19.md)。

2026-10-19 新增 EAGLE v2：`GPT.forward(return_hidden=True)` 可同时返回原 logits 和最终 RMSNorm 状态。`speculative/pf_draft.py` 共享 1L-256 因果模块，按实际 P/F token 和四维坐标递推；Joint 输出 P/F，Feature-only 只输出 F，但两类输入都参与状态更新。`speculative/pf_engine.py` 共用一次 Target 候选验证、实际提案概率的接受采样，以及待消费 token 的恢复逻辑。详见 [实现记录](EAGLE_PF_V2_2026-10-19.md)，本地源码副本为 `patches/eagle_pf_v2/`，运行版本仍在远端主工程。

## 1. 原版目标与数据流

本地 `GaussianGPT` 目录目前保存的是项目交接层，不是完整源码 checkout。下面的架构和实验说明来自对 Autodl 工作区与 GitHub 原版的核对；路径只有在源码同步到本地后才可直接执行。

原版 GaussianGPT 的核心任务是学习 3D Gaussian Splatting 场景的离散生成分布。一个场景先经过 VQ-VAE 编码成有空间结构的 token 序列，GPT 按自回归方式预测下一个 token，生成结果再由 VQ-VAE 解码为 Gaussian 参数并渲染成点云或图像。

```text
3D-FRONT / PhotoShape 场景
        │
        ▼
VQ-VAE / sparse 3D CNN encoder
        │  位置 token + 特征 token
        ▼
tokenized scene chunks
        │
        ▼
GPT with 3D RoPE, autoregressive next-token prediction
        │
        ├─ generate_chunks.py / complete_chunks.py / generate_scene.py
        ▼
VQ-VAE decoder → Gaussian Splatting renderer
        │
        ▼
scene point cloud / orbit render / evaluation artifact
```

VQ-VAE 训练阶段同时关注稀疏 3D 特征重建和 Gaussian re-render loss。`tokenize_dataset.py` 保持 encoder 参数不变，把场景写成 token 流，GPT 训练则使用这些 token 流做交叉熵。位置 token 决定 3D 结构和扫描顺序，特征 token 表示该位置的 Gaussian 语义/外观信息，因此 token 不是普通的一维文本词表。

## 2. 代码目录职责

| 目录 | 作用 |
| --- | --- |
| `model/vqvae/` | 稀疏 3D VQ-VAE、编码器、解码器和 Gaussian 重建相关模块 |
| `model/gpt/` | GPT 主干、3D RoPE、采样和 position-selective speculative 原型 |
| `serialization/` | 位置/特征 token 的序列化、反序列化和数据约定 |
| `scripts/` | 训练、tokenize、生成、渲染、测速和诊断入口 |
| `conf/` | Hydra 数据、模型、训练和实验配置 |
| `data/`, `data_splits/` | 本地数据目录与 train/val 列表；大文件不应提交 |
| `checkpoints/` | 可直接使用的基础 checkpoint；需以状态文档中的清单为准 |
| `profiling/` | 训练日志、checkpoint、端到端测速 JSON 和路线实验 |
| `outputs/` | Hydra 运行目录、渲染和临时输出 |
| `fig/` | 论文/报告图表或可视化资源 |
| `.codex/` | AI 交接说明、项目配置和状态刷新脚本 |

## 3. 数据与坐标约定

3D-FRONT/ASE 使用 z-up 坐标，PhotoShape 使用 y-up 坐标。数据配置通过 `conf/data/tokenized_ase_s2.yaml` 等文件指定数据集、split、token 上限和 chunk 策略。修改序列化或坐标变换时必须同步核对 decoder、renderer 和数据预处理，否则训练 loss 可能正常但渲染方向错误。

VQ-VAE token 保存的是 latent 网格整数坐标。默认配置有 3 个下采样阶段、每阶段缩小 2 倍，`grid_size=0.025` 米，因此 20³ latent chunk 对应名义边长 `20 × 8 × 0.025 = 4` 米。解码时先将 latent 坐标乘以 stride，再把连续 `coord_offset` 加到体素坐标上，最后乘以 `grid_size` 得到 Gaussian 中心；`coord_offset` 使用恒等映射，没有裁剪到单元格范围。因此 `[0,4]` 表示网格覆盖范围，不是 Gaussian 中心必须满足的硬限制。实现见远端 `model/vqvae/cnn/configurable_vqvae.py` 的 `decode_idxs`，参数见 `conf/model/vqvae_cnn.yaml`，偏移表示见 `conf/internal_representations.py`。

当前 S2 配置把 `max_tokens` 设为 16000，`min_chunk_occupancy` 为 0.3，chunk 起点约束为 `[null, null, -1]`。这些约束会直接影响序列长度、显存和 draft/teacher 的可比性。

## 4. GPT 与 draft 模型

原版 teacher 配置在 `conf/model/gpt.yaml` 一类配置中定义词表、序列化和 3D RoPE。当前 S2 draft 配置保持相同的词表、序列化和上下文长度，只缩小 Transformer 宽度：

| 配置 | embedding | layers | heads | context |
| --- | ---: | ---: | ---: | ---: |
| `gpt_draft_d4.yaml` | 256 | 4 | 4 | 16384 |
| `gpt_draft_d6.yaml` | 384 | 6 | 6 | 16384 |
| `gpt_draft_d8.yaml` | 512 | 8 | 8 | 16384 |
| `gpt_draft_d12.yaml` | 768 | 12 | 12 | 16384 |

draft 必须和 teacher 使用相同 token grammar；只缩小网络容量，不能改变位置/特征 token 的交替关系或 RoPE 位置解释，否则 speculative acceptance 的统计没有意义。

## 5. S3 与 position-selective speculative decoding

S3 的 feature distillation 在保留标准 token-level cross entropy 的同时，对 feature-token 行加入 teacher→draft 的 KL 约束，使用温度 `T²` 缩放，并记录 feature CE/KL、top-1 agreement、raw/policy overlap。on-policy 版本先保持 teacher 参数不变并采样 trajectory corpus，再训练 D8 student，避免只在 teacher-forced 前缀上对齐。

`model/gpt/position_selective_speculative.py` 当前只用 draft 提出 position；feature 始终由 target 采样。位置被拒绝时按 target/draft 的最终 position policy 做 residual correction，并回滚/重建相应 KV cache。统计中 `position_accepts/rejects` 用于实际接受率，`position_overlap` 衡量 position 分布重合；`feature_overlap` 来自 draft 的 feature logits，仅是诊断值，不是 feature 接受率，也不表示当前解码器会因 feature rejection 回滚。

关键不变量：

1. draft 和 target 的 token grammar、BOS/EOS 与位置编码必须兼容，缓存各自对应正确的输入前缀；
2. Target 验证会暂时消费候选 P，拒绝后须移除该候选并重新取得最终 P 条件下的 F 分布；
3. 完整一对提交后，Target 缓存长度为输出长度，神经 Draft 为输出长度加 BOS；Target 的尾 F 在下一轮消费；
4. 接受使用最终概率比率，拒绝使用归一化 `max(p-q,0)`，不能只按 top-1 决定；
5. speedup 应注明长度、采样参数、预热、硬件、随机轨迹和同步方式；同 seed 不保证两条路线输出相同。

## 6. 环境与复现风险

当前记录的环境包括 Python 3.10.21、PyTorch 2.8.0+cu129、MinkowskiEngine 0.5.9、gsplat 1.5.0、PyTorch3D 0.7.8、FlashAttention 2.8.2、Lightning 2.5.2 和 RTX 4090。README 提醒 MinkowskiEngine 编译建议 GCC ≤13，而当前记录的 GCC 为 14.3；重建环境或重新编译扩展前必须单独验证这一兼容性。


## 7. Draft proposer 接口（2026-10-05）

Position-selective engine 现在可以通过正式 proposer 接口取得 Position proposal：

```text
DraftProposal.propose_position(state)
DraftProposal.commit(token)
DraftProposal.rewind(state)
```

`DraftProposalState` 携带 Position history、最近 Feature、可选的 D8 `q_position_logits`、合法 Position token ids、EOS、sampling policy、随机数生成器和当前 step。engine 统一执行：

```text
proposer q_D^P
        ↓
Target p_T^P
        ↓
ratio acceptance / residual correction
        ↓
Target Feature commit + cache invariant
```

`NeuralDraftProposal` 只封装 D8 的 logits proposal，保留 Draft prefill/update forward；`StructuredPositionProposal` 调用结构化 table adapter，跳过 Draft Transformer forward，仅维护 Draft cache 的 BOS/位置 bookkeeping。两者都经过同一 Target acceptance/correction/rollback 路径，因此可以直接比较 proposal 成本、acceptance 和 wall-clock。Feature proposer 与 joint `[P,F]` 状态机仍不属于默认接口路径。

### 7.1 Structured lookup 实现层

`StructuredPositionGapDraft` 的正式接口保留三种 lookup mode，runner 通过 `STRUCTURED_LOOKUP_MODE` 选择：

```text
python      → tuple/dict backoff，兼容基线
tensor      → integer context code + device gap/count tensors
vectorized  → integer code + query_batch 的 batched support mapping
```

三种实现都先选择相同的 backoff table，再将有效 gap 计数归一化为原始 Position 概率。正式 proposer 将其取对数并应用 temperature/top-k/top-p，接受函数使用重新处理后的 q。`vectorized` 已提供 batch API，但当前 engine 的 `propose_position` 是单 query 调用；要取得 batch microbenchmark 的收益，必须先把 engine 的 proposal 调度改成批量。已有材料未记录 fused CUDA extension 的注册，fused lookup 仍属于后续方案。

### 7.2 直接源码核对后的实现说明（2026-10-08）

完整算法见 [D8 与结构化 Position 实现汇报](POSITION_ONLY_IMPLEMENTATION_REPORT_2026-10-08.md)。本轮通过 Computer Use 直接读取远端源码，确认查表键使用最近 gap 和最后一个 Position id 的低位，当前没有使用 F token。源码字段 `morton_low_bits` 的实际计算为对 Position id 按位与，没有额外 Morton 编码调用。

查表路线跳过 Draft Transformer 前向，当前仍加载兼容 D8 对象并分配缓存；其 F 诊断使用零 logits 占位，不能解释成查表模型的 F 预测质量。当前共用引擎只提出一个 P，离线 K=4 mode 指标没有成为四位置生成策略。

已读取的 `run_structured_position_e2e.py` 通过 profiler 对纯 Target 每次前向进行 CUDA 同步，候选引擎则关闭逐次前向计时；两条路径的输出头范围也不同。引用速度时须注明入口与计时条件，本轮没有重新测量这些差异。


## 2026-10-21 Feature W1/W2 树解码原型

隔离源码 `patches/feature_tree_20261021/` 新增 `tree_decoder.decode` 与 `TargetTreeRuntime`。树节点为 pendingF、共享P1及每分支的F/P2；注意力读取共享历史和自身祖先，RoPE与位置限制按逻辑路径计算。验证暂存KV，提交一条祖先路径后才推进逻辑位置；第二分支先暂存再压入连续槽。Feature候选无放回抽样，每次拒绝都更新完整剩余分布。沿用既有Ranker和12k MLP，不新增训练。批量RoPE版与逐节点版在实测中保持输出和业务计数相同。该原型未进入生产默认，机制与数值范围见[实现说明](artifacts/feature_tree_20261021/IMPLEMENTATION.md)。
