# GaussianGPT 作者训练流程还原与草稿训练衔接

日期：2026-10-28。研究对象为本次 RTX 4090D 实例使用的官方 `gpt_vfront.ckpt` / `vqvae_vfront.ckpt`，源码阅读版本为 `031d9160622f73dae69324a25d656ce4eb086524`。

## 1. 结论与证据范围

可以还原作者的模型结构、样本构造、主要优化设置及公开权重保存时的训练状态。原始预处理场景、完整训练命令与训练时的源码版本没有保存在这对 checkpoint 中，无法仅由权重恢复完整历史。

本次直接从服务器用 CPU 内存映射读取两份 checkpoint 的超参数、优化器参数组、学习率调度器和进度字段；原始记录见 [checkpoint_training_metadata.json](artifacts/author_training_20261028/checkpoint_training_metadata.json)。未运行训练、GPU 前向或新的性能测量。

来源分为四类：checkpoint 中实际保存的字段；指定版本公开源码的行为；论文对整体实验的描述；本文针对草稿模型提出的建议。下文分别标明，不把当前 YAML 的默认值当作历史训练记录。

官方[权重清单](https://kaldir.vc.cit.tum.de/gaussiangpt/README.md)将 `vfront` 配对列为在 3D-FRONT 上训练，将 `both` 配对列为 3D-FRONT+ASE 预训练后在 3D-FRONT 微调。我们当前使用 `vfront` 配对。论文也描述了混合训练与微调实验，但不能由此把当前权重认作 `both` 版本。

## 2. 作者的完整数据与训练链路

| 阶段 | 输入与处理 | 输出与来源 |
| --- | --- | --- |
| 场景拟合 | 从 3D-FRONT 多视图图像与深度构造初始点云，使用简化的 Scaffold-GS/Voxel-GS 拟合每个场景 60,000 次；场景保留世界尺度，体素边长 2.5 cm，关闭 anchor 增密，SH degree 上限为 1 | 每场景的 Gaussian 属性。作者在 [Issue #2 回复](https://github.com/nicolasvonluetzow/GaussianGPT/issues/2#issuecomment-4869165139)列出了这些设置，并说明 scale 使用 `2 * voxel_size * sigmoid()` |
| VQ-VAE | 用稀疏三维 CNN 压缩 Gaussian 网格，再用 LFQ 离散化；主要监督来自多视图颜色重渲染、VGG 感知损失、占用预测和量化辅助项 | 已训练编码器、解码器及 4,096 类离散特征；见[论文 3.1](https://arxiv.org/html/2603.26661v2#S3.SS1)与 [GaussianVQVAE](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/model/gaussian_vqvae.py#L152-L240) |
| 离线编码 | 保持 VQ-VAE 参数不变，对完整场景进行编码；可在 Gaussian 空间先做四种 z 轴旋转，再各做镜像/不镜像，共八种变体 | 每份文件保存 `coords: [N,3]` 和 `feature_ids: [N,1]`，训练时再裁空间块；见 [tokenize_dataset.py](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/tokenize_dataset.py#L385-L438)与[保存格式](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/tokenize_dataset.py#L605-L609) |
| GPT | 从场景 token 文件随机选取局部空间块，转为 P/F 交替序列；用真实前缀预测下一个 token，交叉熵同时覆盖 P、F 和结束符 | 场景自回归模型；见 [GaussianGPT.general_step](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/model/gaussian_gpt.py#L109-L186) |

作者明确回复过暂时没有发布预处理数据集的计划。[作者回复](https://github.com/nicolasvonluetzow/GaussianGPT/issues/2#issuecomment-4835101141) 原始拟合脚本和完整相机生成设置仍有公开请求：[Issue #4](https://github.com/nicolasvonluetzow/GaussianGPT/issues/4)、[Issue #6](https://github.com/nicolasvonluetzow/GaussianGPT/issues/6)。因此，安装成功和拥有权重，还不足以从头重复作者的全部数据准备。

## 3. 这对公开权重实际保存了什么

| 项目 | `gpt_vfront.ckpt` | `vqvae_vfront.ckpt` |
| --- | --- | --- |
| 保存轮次 | `epoch=99`，即第 100 轮 | `epoch=19`，即第 20 轮 |
| 更新次数 | `global_step=50300` | `global_step=10080` |
| 配置中的总轮数 | 100 | 20 |
| 每卡场景数 | 4 | 1 |
| 梯度累积次数 | 4 | 2 |
| 数值精度 | `bf16-mixed` | `bf16-mixed` |
| Lightning 版本 | 2.5.2 | 2.5.2 |
| 优化器状态 | Muon + AdamW | AdamW，weight decay 为 0 |
| 调度 | cosine，无学习率预热，最终为初值的 0.1 | 1,008 次预热，随后 9,072 次 cosine，最终 `1e-5` |
| 梯度裁剪配置 | `algorithm=null`，未启用 | `algorithm=none`，未启用 |

进度字段是在保存回调中取得；表中表述为保存位置，不据此推断 checkpoint 保存后的其他运行历史。

GPT 为 24 层、宽度 1,024、16 个注意力头及 16 个 KV 头，上下文为 16,384；未使用 value embeddings，稀疏 P/F 模式，`xyz` 顺序。VQ-VAE 有 3 个下采样阶段，输入体素 0.025 m，每阶段缩小 2 倍，latent 间距为 0.2 m，LFQ 类数为 4,096。

论文报告 GPT 使用 4 张 GH200、有效批量 64；VQ-VAE 使用 4 张 RTX A6000、场景有效批量 8。结合 checkpoint 中的每卡批量与累积次数，分别为 `4×4×4=64` 和 `1×2×4=8`。GPU 数量来自论文，checkpoint 的超参数没有单独保存原始机器清单。[论文 4.2](https://arxiv.org/html/2603.26661v2#S4.SS2)

## 4. 训练样本如何构造

公开 split 文件包含 4,025 个 3D-FRONT 训练场景 ID、447 个验证场景 ID；ASE 列表包含 21,254 个训练 ID、2,362 个验证 ID。这是仓库名单数量，实际参与训练的文件数还会受到缺失文件和编码失败影响。

当前公开 GPT 数据配置为 `chunk_shape=20`、`chunk_origin=[null,null,-1]`、`min_chunk_occupancy=0.3`、`max_chunk_attempts=10`。因此它从完整 latent 场景里随机选 x/y 起点、固定 z 起点，取 20³ 个潜在位置的局部块，对应名义边长 4 m。最多尝试十个候选，遇到占用率达标的候选便使用，否则取已尝试候选中占用最多的块。占用率按非空 latent 位置数除以 8,000 计算。[数据配置](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/conf/data/tokenized_vfront.yaml)、[采样实现](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/data/common.py#L833-L896)

旋转和镜像在编码前作用于 Gaussian 场景，方向相关的旋转属性也随之变化。对已有离散 Feature ID 直接重新排列，不能保证等价于重新编码八个变体。数据加载器启用 `load_augmented_tokens` 时会展开文件中的旋转变体；八份变体仍对应一个原始场景。划分训练和验证时应先按场景 ID 分组，再产生变体，避免同一房间出现在两边。[变体选择实现](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/data/common.py#L750-L776)

块内先将坐标减去块起点，再按 `xyz` 排序，z 变化最快。位置 ID 为 `P=400*x+20*y+z`，范围 0–7,999；Feature 类 ID 为 0–4,095，放进 GPT 序列时加 8,000，得到 8,000–12,095。BOS 与 EOS 共用 12,096，PAD 为 12,097。[位置转换](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/utils/pos_tokens.py#L6-L40)、[词表构造](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/model/gpt/gpt.py#L28-L46)

训练序列及监督关系为：

```text
输入：BOS, P0, F0, P1, F1, …
标签： P0, F0, P1, F1, …, EOS
```

满占用时最多 8,000 对 P/F，即 16,000 个正文 token；16,384 的上下文可以容纳整块及特殊符号。批处理保留每条序列的有效长度，EOS 后标签改为 `-1`，不计入损失。[移位监督与补齐](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/model/gpt/gpt.py#L143-L165)

训练使用普通的逐 token 交叉熵，当前原版没有额外给 Feature 加权或加入 Teacher 蒸馏项。P 槽只允许位置和 EOS，F 槽只允许 Feature；实现上是同一个输出矩阵的不同有效行。logits 先以 `15*tanh(logits/15)` 处理，再应用类型遮罩。位置单调限制在有 KV cache 的采样路径中执行，标准训练交叉熵没有应用这项限制。[实际损失与遮罩](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/model/gpt/nanochat_gpt.py#L1010-L1056)

论文对该目标的原文是 “Training is performed with teacher forcing”。这表示用真实前缀监督下一 token，并不表示原论文在蒸馏一个较小模型。[论文 3.2](https://arxiv.org/html/2603.26661v2#S3.SS2) `temperature=0.9`、`top_p=0.9` 属于生成与渲染评价设置，没有进入原版交叉熵计算。

## 5. GPT 的实际优化设置

`training.lr=1e-4` 虽保存在 checkpoint 中，当前 GPT 的 `configure_optimizers` 会调用 nanochat 的分组优化器；这个字段没有作为各参数组的初始学习率使用。[优化器调用](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/model/gaussian_gpt.py#L188-L225)

| 参数组 | 优化器 | 当前 T24 权重保存的初始学习率 | 其他设置 |
| --- | --- | ---: | --- |
| 输出矩阵 | AdamW | 0.0034641016 | betas=(0.8,0.95)，eps=1e-10，weight decay=0 |
| token embedding | AdamW | 0.1732050808 | 同上 |
| residual 缩放参数 | AdamW | 0.005 | 同上 |
| 输入 skip 参数 | AdamW | 0.5 | betas=(0.96,0.95)，eps=1e-10，weight decay=0 |
| Transformer 矩阵 | Muon | 0.02 | beta2=0.95，Newton–Schulz 迭代 5 次 |

输出矩阵与 embedding 的基准学习率为 0.004 和 0.2，乘以 `(width/768)^(-0.5)`。Muon 的 momentum 在前 300 次更新由 0.85 增至 0.95。配置 weight decay=0.1 还会乘 `(12/depth)^2`，T24 的初始实际值为 0.025，并线性减至零；保存的末端值约为 `4.97e-7`。所有组随 cosine 学习率调度到初值的十分之一。[分组实现](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/model/gpt/nanochat_gpt.py#L900-L1008)、[momentum 和衰减](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/model/gaussian_gpt.py#L248-L269)

这会影响小草稿的设置：D8 宽度 512 时，这两组初始学习率分别约为 0.004899、0.244949；若继续用原始 weight decay 字段 0.1，实际 Muon 衰减变为 `0.1*(12/8)^2=0.225`。这些是源码行为的算术结果，不代表已经验证为草稿的合适超参数。

## 6. 当前默认配置与历史记录的差别

| 项目 | 当前公开默认值 | 公开 checkpoint 记录 | 影响 |
| --- | --- | --- | --- |
| GPT weight decay 字段 | 0.2 | 0.1 | 直接使用默认 YAML 无法复现这项历史值 |
| VQ-VAE 总轮数 | 500，源码注释称故意设大 | 20 | 不能把默认轮数当作作者真实训练时长 |
| VQ-VAE 解码剪枝调度 | `prune_thresh_schedule=true` | false | 从头训练 VQ-VAE 前需明确采用哪版行为 |
| VQ-VAE 属性内部表示 | 当前类名与实现已整理 | checkpoint 保存了较早的表示类名 | 应从权重恢复配套配置，避免直接以新默认值替代 |
| 量化辅助损失 | 当前 LFQ wrapper 使用库内部辅助项，再乘外层 `commit_weight` | 同时保存了 `commit_weight=0.25`、`entropy_weight=0.1` | 当前 wrapper 没有显式读取顶层 `entropy_weight`，原训练依赖版本和精确组合仍需额外资料 |
| 训练数据设置 | 当前 YAML 有路径、块采样与变体配置 | checkpoint 仅保存 model/training 两组配置 | 当前 YAML 支持还原机制，不能证明原训练每个 data override 的值 |

VQ-VAE 的损失构成可由论文及代码确认；若要求逐项重做原 VQ-VAE 训练，还需要原版本依赖及配置。对于当前草稿任务，可以保留作者 VQ-VAE，继续使用现有 token 语义。

公开验证数据也使用相同的随机块采样类，所以 `shuffle=false` 不意味着每次取到同一个空间块。草稿比较宜额外保存固定、独立来源的验证块，确保各权重消费相同前缀；这属于本项目的实验建议。[数据类参数](https://github.com/nicolasvonluetzow/GaussianGPT/blob/031d9160622f73dae69324a25d656ce4eb086524/data/vfront_dataset.py#L1082-L1098)

## 7. 如何用于我们的草稿

### 完整自回归 D8

可以复用作者的 tokenizer、空间块选择、序列化、RoPE、类型遮罩和移位交叉熵，缩小 Transformer。我们已有 D8 配置为 8 层、宽度 512、8 个注意力头及 8 个 KV 头，上下文仍为 16,384，P/F 词表保持。[现有配置](../../.codex/remote-source/conf/model/gpt_draft_d8.yaml)

有真实 Gaussian/token 语料时，先按作者数据链路学习真实 P/F 序列，再在同一前缀上加入原始 Target 分布蒸馏，并纳入 Draft 实际提出候选后的前缀。前一阶段复用原作者的自回归训练，后一阶段为草稿接受率目标增加监督。蒸馏的权重与温度需要由独立验证确定，本文没有把某个系数当作已测最优值。

如果缺少真实训练场景，可先建立较多独立 BOS 生成和真实前缀补全的 Teacher 轨迹语料。这条路线能训练草稿，但其数据性质应标为 Teacher 采样语料；它不能证明重复了作者的原始数据训练。

### 当前轻量 Feature Draft

当前 `FeatureStateDraft` 为约 286 万参数，输入包含 Target 状态、上一 Feature 与候选 Position，并复用 Target 的 Feature 输出权重。作者 `train_gpt.py` 只读取 token 序列，无法直接为这个模型提供所需的状态监督。

应在相同 P/F 历史下，用原始 Target 采集模型所需隐藏状态及完整 4,096 类 Feature 分布；确保标签对应实际候选 Position 消费后的同一历史，再用现有 Feature 训练入口学习。已有 [Target 完整分布训练](FEATURE_TARGET_V2_2026-10-14.md)实现了原始 T=1 分布 CE 与采样策略 TV；[192k 续训记录](FEATURE_MLP_EXTEND192_2026-10-19.md)显示同一批语料增加更新次数后的固定验证改善有限。这些既有结果支持把新研究优先放在来源覆盖、空间块与候选历史的数据条件上，尚无新训练结果证明其中哪项会提高最终速度。

### 数据准备与评价顺序

1. 沿用已确认配对的 Target/VQ-VAE，确认可用真实 Gaussian/token 数据与 Teacher 轨迹的位置；按来源场景先划分训练、验证、测试。
2. 真实数据通过原 VQ-VAE 编码并产生八种变体；训练时采随机 20³ 空间块。Teacher 采样数据保留完整 P/F、BOS/EOS 含义与采样配置。
3. 完整 D8 使用序列 CE 并增加相同候选历史下的 Target 蒸馏；轻量 Feature Draft 复用现有隐藏状态采集与训练入口。
4. 先验证 token 类型、坐标、标签位置和 EOS，再比较独立来源的 P/F 重合度、在线接受数及完整生成吞吐。更低 CE 不直接等于更快生成。

此前默认 4×4 样例只有 76 个 token，阶段 PNG/GIF 是可视化产物，不构成训练集。`enable_bootstrap=false no_empty_columns=true` 产生的逐列结果也采用了不同采样条件，不能不加区分地混作标准 Teacher 轨迹。

单卡有效批量为 `每卡批量×累积次数`；作者 GPT 的有效批量 64 不能通过单卡仍使用 `4×4` 得到。小草稿可通过更小的每卡批量及更多累积达到目标批量，但 24 GB 下的完整上下文显存与训练速度尚未测量；学习率调度还应使用该任务实际更新次数。

## 8. 仍然缺少什么

- 作者原始拟合 Gaussian 场景、预先渲染的图像/深度及完整相机生成设置。
- 当时完整 Hydra 数据配置、确切训练命令、依赖版本与训练源码提交号。
- 此 RTX 4090D 工程中已配置好的完整训练语料路径；本次只读取了工程内目录及公开权重，没有全盘搜索其他用户数据。
- 新草稿在扩展语料下的训练与在线结果。本文给出研究结论与衔接方案，没有启动训练或验证收益。

本轮本地新增本报告与 checkpoint 元数据 JSON，并更新交接入口和研究方案索引。远端只读取了权重和目录信息，GPU 读数为 0%/2 MiB，没有修改服务器源码、配置或权重。

研究路径为 `research-router → github-local` 与 `research-router → academic-local → academic-evidence`。论文证据来自 arXiv v2 正文 HTML；源码来自指定提交的官方仓库临时浅副本，Issue 回复通过 GitHub 连接器读取。当前工具和 CLI 帮助未提供项目指定的 `multi_agent_v1` 派发入口，本轮由主会话完成对照，没有改用其他子代理接口。
