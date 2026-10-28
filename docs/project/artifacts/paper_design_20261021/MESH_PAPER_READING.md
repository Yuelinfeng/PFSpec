# Mesh 自回归与 Speculative Decoding 文献研读

日期：2026-10-23  
范围：公开可访问的 mesh speculative decoding、speculative mesh generation 及其相邻 mesh 自回归工作。本文只纳入两篇直接工作和两篇相邻工作；详细方法、结果和限制均来自正文 PDF，摘要只用于定位。

## 结论先行

在本轮公开检索范围内，截至 2026-10-23，已确认的直接 mesh speculative decoding 论文有两篇：

1. **FlashMesh**：CVPR 2026，CVF 官方开放页面标注为 accepted version，论文集页码 27052–27061；这是本轮确认已录用并公开刊载的直接工作。
2. **XSpecMesh**：arXiv:2507.23777v2，2025-08-06 更新；本轮确认它是完整预印本，尚未从可访问的官方记录中取得 venue 或 decision，因此状态写作“未确认已录用”。

本轮检索未获得 `SpecMesh`、`MeshSpec` 等名称变体对应的另一篇直接论文。这个结论限定于本记录末尾列出的公开平台、关键词和可访问页面，不作全网唯一性断言。FlashMesh 已经把“结构感知的 mesh speculation + 顶点共享一致性处理”写成明确贡献，因此后续论文需把 `structure-aware speculative mesh decoding`限定为已有叙事，并把差异落到 Gaussian latent P/F 的信息条件、接受规则、分布性质或实测成本上。

## 四篇论文身份

| 类别 | 论文、作者 | 日期与会议状态 | 官方页面与正文 PDF | 正文下载与抽取路径 |
|---|---|---|---|---|
| 直接 | **FlashMesh: Faster and Better Autoregressive Mesh Synthesis via Structured Speculation**；Tingrui Shen, Yiheng Zhang, Chen Tang, Chuan Ping, Zixing Zhao, Le Wan, Yuwang Wang, Ronggang Wang, Shengfeng He | 2026；CVPR 2026，已录用并在 CVF Open Access 页面公开，pp. 27052–27061 | [CVF 页面](https://openaccess.thecvf.com/content/CVPR2026/html/Shen_FlashMesh_Faster_and_Better_Autoregressive_Mesh_Synthesis_via_Structured_Speculation_CVPR_2026_paper.html)；[CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_FlashMesh_Faster_and_Better_Autoregressive_Mesh_Synthesis_via_Structured_Speculation_CVPR_2026_paper.pdf)；[arXiv 2511.15618](https://arxiv.org/abs/2511.15618) | PDF：`/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/FlashMesh_CVPR2026.pdf`；正文抽取：`/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/FlashMesh_CVPR2026.txt` |
| 直接 | **XSpecMesh: Quality-Preserving Auto-Regressive Mesh Generation Acceleration via Multi-Head Speculative Decoding**；Dian Chen, Yansong Qu, Xinyang Li, Ming Li, Shengchuan Zhang | 2025-07-31 首次提交，2025-08-06 v2；arXiv 预印本，录用状态未确认 | [arXiv v2 页面](https://arxiv.org/abs/2507.23777v2)；[arXiv PDF](https://arxiv.org/pdf/2507.23777)；[代码仓库](https://github.com/CD-link/XSpecMesh) | PDF：`/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/pdfs/01_XSpecMesh_Quality-Preserving_Auto-Regressive_Mesh_Generation_Acceleration_via_Multi-Head_Speculative_Decoding.pdf`；正文抽取：`/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/01_XSpecMesh_Quality-Preserving_Auto-Regressive_Mesh_Generation_Acceleration_via_Multi-Head_Speculative_Decoding.txt` |
| 相邻 | **MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers**；Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, Matthias Nießner | 2024；CVPR 2024，已录用并在 CVF Open Access 页面公开，pp. 19615–19625 | [CVF 页面](https://openaccess.thecvf.com/content/CVPR2024/html/Siddiqui_MeshGPT_Generating_Triangle_Meshes_with_Decoder-Only_Transformers_CVPR_2024_paper.html)；[CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Siddiqui_MeshGPT_Generating_Triangle_Meshes_with_Decoder-Only_Transformers_CVPR_2024_paper.pdf)；[arXiv 2311.15475](https://arxiv.org/abs/2311.15475) | PDF：`/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/MeshGPT_CVPR2024.pdf`；正文抽取：`/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/MeshGPT_CVPR2024.txt` |
| 相邻 | **MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization**；Yiwen Chen, Yikai Wang, Yihao Luo, Zhengyi Wang, Zilong Chen, Jun Zhu, Chi Zhang, Guosheng Lin | 2025；ICCV 2025，已录用并在 CVF Open Access 页面公开，pp. 13922–13931 | [CVF 页面](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_MeshAnything_V2_Artist-Created_Mesh_Generation_with_Adjacent_Mesh_Tokenization_ICCV_2025_paper.html)；[CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_MeshAnything_V2_Artist-Created_Mesh_Generation_with_Adjacent_Mesh_Tokenization_ICCV_2025_paper.pdf)；[arXiv 2408.02555](https://arxiv.org/abs/2408.02555) | PDF：`/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/MeshAnythingV2_ICCV2025.pdf`；正文抽取：`/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/MeshAnythingV2_ICCV2025.txt` |

### 会议身份边界

FlashMesh 和 MeshGPT 的 CVF 页面都写明页面版本与 accepted version 一致，并给出会议论文集页码；MeshAnything V2 的 CVF 页面给出 ICCV 2025 论文集信息。XSpecMesh 的 arXiv 页面只确认版本、作者和提交日期。本轮尝试读取 `https://openreview.net/forum?id=7s68UdT65x` 时遇到验证挑战并返回 403，未取得 venue 或 decision；XSpecMesh 在本文中保持“arXiv 预印本、录用状态未确认”的身份。

## 1. FlashMesh：最直接的已发表 mesh speculative decoding

### 问题与表示

作者从 mesh 自回归生成的逐 token 延迟出发，将 mesh 描述为具有 face、point、coordinate 三个层级的结构，并指出 Hourglass Transformer 会在这些层级间压缩和展开特征。论文的研究问题是：如何在保留自回归精度的同时，让层级结构支持多个未来 token 的并行预测。上述问题、表示和 Hourglass 差异见 §1，PDF pp. 1–2（印刷页 27052–27053），[官方 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_FlashMesh_Faster_and_Better_Autoregressive_Mesh_Synthesis_via_Structured_Speculation_CVPR_2026_paper.pdf)。

### 机制与数学

FlashMesh 把推理组织为 `predict–correct–verify`。`predict` 阶段由原始 Hourglass Transformer 产生 main token，SP-Block 从当前层特征产生未来 draft token，HF-Block 将高层 speculative feature 上采样后与当前层 KV cache 融合。对输入位置 `n`，正文明确写出 backbone 产生 `x_{n+1}`，SP/HF 模块并行产生 `x_{n+2:n+D+1}`。SP-Block 的式 (1)、上采样式 (2)、HF-Block 的式 (3)–(5) 位于 §3.1，PDF pp. 3–5（印刷页 27054–27056），[官方 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_FlashMesh_Faster_and_Better_Autoregressive_Mesh_Synthesis_via_Structured_Speculation_CVPR_2026_paper.pdf)。

`correct` 阶段在 point level 加 label head，把点分为历史点、新点、同批重复点；同批重复点按顶点重合和最近新点处理，之后按 z-y-x 重新排序。坐标交叉熵、标签交叉熵及总损失分别见式 (6)–(8)，§3.2，PDF p. 5（印刷页 27056），[官方 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_FlashMesh_Faster_and_Better_Autoregressive_Mesh_Synthesis_via_Structured_Speculation_CVPR_2026_paper.pdf)。这一步直接使用 mesh 邻接与顶点共享信息，已经占据“structure-aware correction”这一表述空间。

`verify` 阶段把 main token 与校正后的 draft token送入下一次 backbone forward。backbone 对候选位置重新计算 token，作者将 draft 与重新计算结果比较；原文称找到“latest match”，接受到该位置并重新处理后续段。该过程见 §3.3、Figure 5，PDF pp. 5–6（印刷页 27056–27057），[官方 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_FlashMesh_Faster_and_Better_Autoregressive_Mesh_Synthesis_via_Structured_Speculation_CVPR_2026_paper.pdf)。所读正文没有充分区分末次匹配与连续匹配前缀，故这里不进一步断言它严格在首处不一致时停止，也不替作者推出精确概率性质。

数学边界需要单独保留：正文给出了模块计算式和交叉熵损失，也给出了匹配式 token 验证流程；本次通读未见标准 speculative sampling 所需的目标分布等价证明或残差抽样公式。因此，论文的“质量提升”应写作实测结果，输出分布保持仍属于未确认事项。证据位置为 §3.1–§3.3，PDF pp. 3–6，[官方 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_FlashMesh_Faster_and_Better_Autoregressive_Mesh_Synthesis_via_Structured_Speculation_CVPR_2026_paper.pdf)。

### 贡献证据与限制

实验在 H20 GPU 上报告 CD、HD、BBox-IoU、TPS 和 Speed-up。表 1 中，FlashMesh 接入 Meshtron-1B 后为 180.4 TPS、1.83x；接入 Meshtron-2B 后为 136.6 TPS、2.03x，同时 CD/HD/BBox-IoU 分别为 0.089/0.198/0.949。实验设置与表 1 位于 §4.1–§4.2，PDF pp. 6–7（印刷页 27057–27058），[官方 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_FlashMesh_Faster_and_Better_Autoregressive_Mesh_Synthesis_via_Structured_Speculation_CVPR_2026_paper.pdf)。

消融表 2 将 Meshtron-1B 的 95.5 TPS 依次提升到 SP-Block 的 109.7、SP+HF 的 176.5、SP+HF+Correction 的 180.4；表 3 显示 draft 过多时正确验证数量比例下降且计算成本上升，18–15 配置达到 180.4 TPS 和 1.83x。证据位于 §4.4，PDF pp. 7–8（印刷页 27058–27059），[官方 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_FlashMesh_Faster_and_Better_Autoregressive_Mesh_Synthesis_via_Structured_Speculation_CVPR_2026_paper.pdf)。

原表存在需保留的数字差异：表1／表3的Meshtron-1B为98.6 TPS，表2同名配置为95.5 TPS；表3的9–9配置为139.9 TPS／1.52x，直接除以98.6约为1.419。这里逐表记录作者结果，不将它们合并为一套自行推算的统一统计。该差异也说明正式论文需要明确速度定义和汇总对象。

§5 的限制写明模型仍受自回归早期预测错误敏感性的影响，后续工作方向包括混合解码和更明确的几何先验。该限制位于 PDF p. 8（印刷页 27059），[官方 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_FlashMesh_Faster_and_Better_Autoregressive_Mesh_Synthesis_via_Structured_Speculation_CVPR_2026_paper.pdf)。这说明 FlashMesh 的结构处理与 Gaussian latent P/F 的错误传播分析仍需分开论证。

### 对论文写作的借鉴

FlashMesh 的有效结构是：先说明 mesh 的层级表示和逐 token 成本，再把并行预测、mesh-specific 处理、backbone 验证拆成三个有名称的阶段；随后用模块消融把每个阶段和速度、质量指标逐项对应。对当前论文可借鉴“表示差异 -> 机制输入 -> 验证量 -> 速度/质量实测”的顺序。FlashMesh 已经完成 mesh 结构感知这一层叙事，当前工作若要差异化，应明确 Gaussian latent P/F 的候选信息、随机轨迹和接受量，并由实验给出对应证据。该判断基于 §1、§3、§4、§5，PDF pp. 1–8，[官方 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_FlashMesh_Faster_and_Better_Autoregressive_Mesh_Synthesis_via_Structured_Speculation_CVPR_2026_paper.pdf)。

## 2. XSpecMesh：最直接的多头 mesh speculative decoding 预印本

### 问题与表示

XSpecMesh 把 mesh 自回归生成拆成离散 mesh 序列化、Transformer 自回归预测和反序列化三个环节；正文说明顶点与面会变成 token 序列，backbone 按因果掩码同时计算多个位置的预测分布。问题、表示和式 (1)–(2)见 §3，PDF p. 3，[arXiv PDF](https://arxiv.org/pdf/2507.23777)。

### 机制与数学

设当前接受位置为 `s`，backbone 的最终 hidden state 为 `h_s`。式 (3) 用 `softmax(W^(0) h_s)`得到下一 token 分布；第 `d` 个 cross-attention decoding head 用式 (4)预测位置 `s+d+1`，再从 `p^(0)_{s+1}, p^(1)_{s+2}, ..., p^(D)_{s+D+1}`采样多个候选。公式和模块定义见 §4.1，PDF p. 4，[arXiv PDF](https://arxiv.org/pdf/2507.23777)。

验证规则是：backbone 对候选序列重新计算分布；从当前位置开始，只要 `p_i^(0)(x_i) > δ` 就接受该 token；首个不满足阈值的位置起，使用 backbone 与 decoding heads 的分布重新采样后续候选。算法 1 位于 §4.2，PDF pp. 4–5；正文还给出多头交叉熵损失式 (5)–(6)和带 backbone 保持项的第二阶段损失式 (7)，§4.3，PDF p. 5。[arXiv PDF](https://arxiv.org/pdf/2507.23777)。

这里应保留精确边界：阈值接受 `p_i^(0)(x_i)>δ`与首个拒绝位置重采样是论文正文的实际规则；本次通读未见标准 residual correction 形式或严格输出分布等价证明。论文将质量保持作为实验主张，阈值规则的性质应限定在正文给出的经验结果内。证据位于 §4.2–§4.3，PDF pp. 4–5，[arXiv PDF](https://arxiv.org/pdf/2507.23777)。

### 贡献证据与限制

作者以 BPT 为 backbone，在 Objaverse 约 10K 个 shape 子集上训练 decoding heads 与 LoRA。表 2 报告 BPT 的 CD/HD/US/平均延迟为 0.1165/0.2223/37%/257.6s，XSpecMesh 为 0.1168/0.2261/36%/151.4s。实验设置和表 2 位于 §5.1–§5.3，PDF pp. 5–7，[arXiv PDF](https://arxiv.org/pdf/2507.23777)。

表 3 中，cross-attention heads + LoRA 配置的 SCR 为 2.021、单步延迟 47.83ms、speedup 为 1.71x，CD/HD 为 0.1168/0.2261。头数消融在 `D=4` 达到 1.71x；阈值消融选择 `δ=0.5`作为速度与 CD/HD 的折中。证据位于 §5.4、表 3、Figure 5，PDF pp. 7–8，[arXiv PDF](https://arxiv.org/pdf/2507.23777)。

附录 §7.8 写明方法仍受 BPT backbone 能力限制；附录 §7.5–§7.7 给出与 BPT 形状、拓扑和速度的比较。证据位于 PDF pp. 13–15，[arXiv PDF](https://arxiv.org/pdf/2507.23777)。

### 对论文写作的借鉴

XSpecMesh 适合借鉴“先定义 token 序列和 backbone 条件，再给多头分布式候选，最后把接受规则和重采样流程写成算法”的组织方式。它的实验把接受头数、阈值、head 类型、LoRA 训练阶段分别做消融，能直接对应机制选择。对当前论文可借鉴这种证据排列，但 Gaussian latent P/F 是否具有同样的 token 条件关系，需要单独测量；XSpecMesh 的 BPT、mesh token 和阈值结果应限定为 mesh 实验结果。判断依据为 §3–§5、§7.8，PDF pp. 3–8、13，[arXiv PDF](https://arxiv.org/pdf/2507.23777)。

## 3. MeshGPT：mesh 表示和 decoder-only 自回归基础

### 问题与表示

MeshGPT 将三角 mesh 组织为三角形序列 `M=(f_1,...,f_N)`；顶点按 z-y-x 排序，面按最低顶点索引等规则排序，每个三角形的直接表示含 9 个坐标值。作者指出直接坐标 token 序列长度为 `9N`，且缺少邻近三角形几何信息。上述表示和问题位于 §3.1，PDF pp. 3–4（印刷页 19617–19618），[官方 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Siddiqui_MeshGPT_Generating_Triangle_Meshes_with_Decoder-Only_Transformers_CVPR_2024_paper.pdf)。

### 机制与数学

作者用 graph convolution encoder 从面及其邻域提取几何特征，再用 residual vector quantization 形成每个面的多级 codebook token。式 (3)–(5)定义量化 token 与解码 mesh；在 `D<9`时，序列长度 `|T|=DN` 小于直接坐标表示的 `9N`。随后 decoder-only Transformer 以自回归 next-index 目标训练，式 (6)最大化各 codebook index 的条件概率。证据位于 §3.1–§3.2，PDF pp. 3–4（印刷页 19617–19618），[官方 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Siddiqui_MeshGPT_Generating_Triangle_Meshes_with_Decoder-Only_Transformers_CVPR_2024_paper.pdf)。

生成阶段从 start token 开始，按 beam sampling 生成 codebook index，decoder 还原三角形；作者说明输出初始可能包含邻面重复顶点，之后用近邻顶点合并得到 mesh。该流程见 §3.2，PDF p. 4（印刷页 19618），[官方 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Siddiqui_MeshGPT_Generating_Triangle_Meshes_with_Decoder-Only_Transformers_CVPR_2024_paper.pdf)。本文没有 speculative proposal、backbone verification 或首个拒绝位置重采样机制，因此属于表示与自回归基础工作。

### 贡献证据与限制

表 1 在 ShapeNet 的 Chair、Table、Bench、Lamp 上报告 MeshGPT 的 COV、MMD、1-NNA、FID、KID 和生成 mesh 大小；例如 Chair 的 COV/MMD/FID/KID 为 43.28/3.29/18.46/0.010，Table 为 45.68/2.36/6.24/0.002。结果位于 §4.1–§4.2，PDF pp. 5–6（印刷页 19619–19620），[官方 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Siddiqui_MeshGPT_Generating_Triangle_Meshes_with_Decoder-Only_Transformers_CVPR_2024_paper.pdf)。

表 3 的消融显示去掉 learned tokens、encoder features、pretraining、sequence compression 或 per-vertex quantization都会降低 Chair 结果；正文还写明自回归采样单个 mesh 约需 30–90 秒，场景级生成仍受序列长度和上下文限制。证据位于 §4.2.1、Limitations，PDF p. 8（印刷页 19622），[官方 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Siddiqui_MeshGPT_Generating_Triangle_Meshes_with_Decoder-Only_Transformers_CVPR_2024_paper.pdf)。

### 对论文写作的借鉴

MeshGPT 的写法值得保留：先固定 mesh 的序列化约定，再解释直接坐标序列的长度和几何缺失，接着引入 codebook、量化和 next-index 目标，最后用表示消融验证每个选择。对 Gaussian latent P/F 论文，类似结构可以帮助读者区分“生成对象是什么”和“如何加速它”；它提供 mesh 表示证据，Gaussian latent P/F 的 speculative decoding 机制仍需独立实证。依据为 §3–§4、Limitations，PDF pp. 3–8，[官方 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Siddiqui_MeshGPT_Generating_Triangle_Meshes_with_Decoder-Only_Transformers_CVPR_2024_paper.pdf)。

## 4. MeshAnything V2：相邻面 token 化与序列长度证据

### 问题与表示

MeshAnything V2 关注 Artist-Created Mesh 的序列化成本。传统表示先按 z-y-x 排序顶点，再按顶点索引排序面，每个面使用三个顶点；因此顶点序列长度为面数的三倍，且共享顶点重复出现。AMT 在相邻面可用时只追加一个顶点；无法找到相邻面时插入特殊 token `&` 并从未编码面继续。该规则和 Algorithm 1 位于 §3.1，PDF pp. 3–5（印刷页 13924–13926），[官方 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_MeshAnything_V2_Artist-Created_Mesh_Generation_with_Adjacent_Mesh_Tokenization_ICCV_2025_paper.pdf)。

论文还定义 `$` vertex-swap token，使下一面可使用前一面的第一和最后顶点，从而扩大相邻面搜索范围。V2 直接使用离散坐标作为 token index，并把 `&` 加入 token 表；点云条件经过 encoder 后作为 Transformer prefix，mesh token 序列由 decoder-only Transformer 自回归生成。上述机制位于 §3.1–§3.2，PDF pp. 5–6（印刷页 13926–13927），[官方 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_MeshAnything_V2_Artist-Created_Mesh_Generation_with_Adjacent_Mesh_Tokenization_ICCV_2025_paper.pdf)。

### 贡献证据与限制

表 1 的 AMT 消融中，V2 的 S Ratio 为 0.497，去掉 AMT 的版本为 1.000；同时 CD 为 0.874 对 0.895、ECD 为 4.721 对 4.832、NC 为 0.933 对 0.924。表 1 位于 §4.3，PDF p. 7（印刷页 13928），[官方 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_MeshAnything_V2_Artist-Created_Mesh_Generation_with_Adjacent_Mesh_Tokenization_ICCV_2025_paper.pdf)。

表 2 比较多种 token 化方法：直接三顶点方法的 S Ratio 为 1.000，AMT 为 0.492，AMT(Swap) 为 0.455；AMT 的 CD/ECD/NC 为 2.348/19.33/0.904，AMT(Swap) 为 2.517/19.86/0.913。作者在 §3.1 和 §4.3 说明 AMT 平均缩短序列约一半，并将 V2 的最大生成面数从 800扩展到1600。证据位于 PDF pp. 5–8（印刷页 13925–13929），[官方 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_MeshAnything_V2_Artist-Created_Mesh_Generation_with_Adjacent_Mesh_Tokenization_ICCV_2025_paper.pdf)。

### 对论文写作的借鉴

MeshAnything V2 的论文结构把“序列长度”和“序列规律性”拆成两个问题，再用 token 化规则、可逆解码、消融表和不同 token 化对照逐项支撑。它适合作为当前论文处理候选长度、结构约束和解码成本的相邻参考。它没有 draft head、backbone verification 或接受概率实验，因此 AMT 的序列压缩数字应限定为 mesh token 化结果，Gaussian speculative decoding 的加速数字仍需单独测量。依据为 §3.1–§4.3，PDF pp. 5–8，[官方 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_MeshAnything_V2_Artist-Created_Mesh_Generation_with_Adjacent_Mesh_Tokenization_ICCV_2025_paper.pdf)。

## Gaussian latent P/F 与 mesh 顶点/面序列的边界

| 比较项 | 当前主纲的 Gaussian latent P/F | 本轮四篇 mesh 工作 |
|---|---|---|
| 生成对象 | 以项目定义的 Gaussian latent P/F 为生成单元；本记录不重新定义 P/F | 明确的顶点、面、坐标或三角形 codebook token |
| 结构关系 | 需要依据主纲和数据确认 P/F 的条件依赖、重复模式和错误传播 | FlashMesh 使用 face/point/coordinate 层级、顶点共享与 z-y-x 顺序；XSpecMesh 使用 BPT 的 mesh token 序列 |
| 候选处理 | 需测量 Gaussian token 的候选分布、接受量、随机轨迹和额外计算 | FlashMesh 采用匹配式 token 验证；XSpecMesh 采用阈值接受与首个拒绝位置重采样 |
| 质量证据 | 当前文献不会替代 Gaussian 数据 | mesh 论文报告 CD、HD、BBox-IoU、用户评价、TPS 或序列压缩率 |
| 可直接借鉴的部分 | 论文组织、候选长度消融、接受规则的明确定义、速度与质量分开测量 | 表示定义、算法流程、模块消融和限制段落的写法 |
| 需要数据后才能判断的部分 | Gaussian latent P/F 是否具有 mesh 顶点共享式结构、是否适合结构处理、是否保持所需输出分布 | 四篇论文没有给出 Gaussian latent P/F 上的实验 |

因此，当前证据只支持“mesh 文献提供相邻设计和写作参照”。mesh 的顶点共享、面序列接受率和 TPS 数字尚未获得迁到 Gaussian latent P/F 的数据依据；FlashMesh 的 structure-aware 贡献也应保留其原有归属。后续若主Agent需要保留跨域联系，应以 Gaussian 数据上的候选接受率、额外 forward 成本、质量指标和随机轨迹证据为依据。

## 最接近工作差异表

| 工作 | 生成表示与主模型 | 加速机制 | 候选接受/质量性质 | 正文实证 | 与 Gaussian latent P/F 的距离 |
|---|---|---|---|---|---|
| FlashMesh，CVPR 2026 | face/point/coordinate 三层 mesh token；Hourglass Transformer | SP-Block + HF-Block，多层多头预测；mesh 顶点共享校正 | backbone 重算后按匹配位置接受；正文未给标准目标分布等价证明 | Meshtron-1B 1.83x，Meshtron-2B 2.03x；表 1–3，§4，PDF pp. 6–8；[PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_FlashMesh_Faster_and_Better_Autoregressive_Mesh_Synthesis_via_Structured_Speculation_CVPR_2026_paper.pdf) | 最近的结构感知 mesh 直接工作；Gaussian P/F 的条件依赖和 token 结构尚未对应 |
| XSpecMesh，arXiv v2 | BPT 的 mesh token；decoder-only backbone + cross-attention heads | 多头预测多个后续 token；两阶段 decoding-head/LoRA 训练 | `p_i^(0)(x_i)>δ` 接受，首个拒绝位置重采样；正文未给标准 residual correction 证明 | 1.71x，SCR 2.021，`δ=0.5`折中；表 2–3、Figure 5，§5，PDF pp. 5–8；[PDF](https://arxiv.org/pdf/2507.23777) | 机制上最近；阈值与 BPT 依赖需要在 Gaussian P/F 上重新定义 |
| MeshGPT，CVPR 2024 | 三角形序列；graph convolution + RQ codebook + GPT decoder | 无 speculative decoding；sequence compression | 自回归 next-index；无候选验证 | ShapeNet 表 1、表示消融表 3，§4，PDF pp. 5–8；[PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Siddiqui_MeshGPT_Generating_Triangle_Meshes_with_Decoder-Only_Transformers_CVPR_2024_paper.pdf) | 提供离散 mesh 表示基础，未涉及 Gaussian P/F |
| MeshAnything V2，ICCV 2025 | 顶点/面序列；AMT、`&`、`$`、点云 prefix | token 化压缩与局部邻接顺序；无 speculative decoding | 自回归生成并屏蔽无效 token；无 speculative acceptance | S Ratio 0.497、面数上限 1600；表 1–2，§4，PDF pp. 7–8；[PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_MeshAnything_V2_Artist-Created_Mesh_Generation_with_Adjacent_Mesh_Tokenization_ICCV_2025_paper.pdf) | 提供序列长度与结构规律性参照，未证明跨表示迁移 |

## 可借鉴的论文写作结构

1. **先锁定生成对象和序列化规则。** 先说明 token 代表什么、条件输入是什么、一步生成覆盖哪些位置，再讨论并行化。
2. **把瓶颈变成可测量问题。** 分开报告序列长度、单步 forward 成本、接受 token 数、完整样本延迟和质量指标。
3. **用算法和公式对应真实操作。** 明确候选从哪里来、backbone 何时重算、何时接受、何时重新采样；公式只覆盖正文实际执行的量。
4. **把速度、质量和分布性质分开写。** mesh 论文的 TPS、CD、HD 和 BBox-IoU 属于不同证据；实测质量不等于概率分布性质。
5. **消融必须对应设计选择。** FlashMesh 对 SP/HF/Correction 和 draft 数量做消融；XSpecMesh 对 head 类型、LoRA、head 数和阈值做消融；MeshAnything V2 对 token 化规则和序列长度做消融。
6. **限制段落给出适用边界。** FlashMesh 保留早期错误敏感性，XSpecMesh 受 BPT 能力限制，MeshGPT 受自回归采样时间和上下文限制；当前 Gaussian 论文应单独报告 P/F 数据的边界。

## 检索记录：词、成功原文、路径与失败

### 实际检索词

`mesh speculative decoding`；`speculative mesh generation`；`mesh autoregressive speculative decoding`；`SpecMesh`；`MeshSpec`；`XSpecMesh`；`FlashMesh`；`structure-aware mesh speculation`；`topology-aware mesh decoding`；`block-wise mesh decoding`；`MeshAnything`；`MeshAnything V2`；`MeshGPT`；`MeshXL`；`Meshtron`；`autoregressive mesh generation`；`vertex face sequence`。

### 成功的原文 URL 与本地路径

| 论文 | 成功原文 URL | 原文下载路径 | 正文抽取路径 |
|---|---|---|---|
| FlashMesh | [CVF 页面](https://openaccess.thecvf.com/content/CVPR2026/html/Shen_FlashMesh_Faster_and_Better_Autoregressive_Mesh_Synthesis_via_Structured_Speculation_CVPR_2026_paper.html)；[CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_FlashMesh_Faster_and_Better_Autoregressive_Mesh_Synthesis_via_Structured_Speculation_CVPR_2026_paper.pdf)；[arXiv](https://arxiv.org/abs/2511.15618) | `/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/FlashMesh_CVPR2026.pdf` | `/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/FlashMesh_CVPR2026.txt` |
| XSpecMesh | [arXiv v2 页面](https://arxiv.org/abs/2507.23777v2)；[arXiv PDF](https://arxiv.org/pdf/2507.23777) | `/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/pdfs/01_XSpecMesh_Quality-Preserving_Auto-Regressive_Mesh_Generation_Acceleration_via_Multi-Head_Speculative_Decoding.pdf` | `/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/01_XSpecMesh_Quality-Preserving_Auto-Regressive_Mesh_Generation_Acceleration_via_Multi-Head_Speculative_Decoding.txt` |
| MeshGPT | [CVF 页面](https://openaccess.thecvf.com/content/CVPR2024/html/Siddiqui_MeshGPT_Generating_Triangle_Meshes_with_Decoder-Only_Transformers_CVPR_2024_paper.html)；[CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Siddiqui_MeshGPT_Generating_Triangle_Meshes_with_Decoder-Only_Transformers_CVPR_2024_paper.pdf)；[arXiv](https://arxiv.org/abs/2311.15475) | `/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/MeshGPT_CVPR2024.pdf` | `/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/MeshGPT_CVPR2024.txt` |
| MeshAnything V2 | [CVF 页面](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_MeshAnything_V2_Artist-Created_Mesh_Generation_with_Adjacent_Mesh_Tokenization_ICCV_2025_paper.html)；[CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_MeshAnything_V2_Artist-Created_Mesh_Generation_with_Adjacent_Mesh_Tokenization_ICCV_2025_paper.pdf)；[arXiv](https://arxiv.org/abs/2408.02555) | `/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/MeshAnythingV2_ICCV2025.pdf` | `/Users/lief/.codex/skills/research-router/records/papers/2026-10-23/gaussiangpt-paper-outline/mesh_sources/MeshAnythingV2_ICCV2025.txt` |

### 失败与未确认事项

- `SpecMesh`、`MeshSpec` 的 arXiv 标题检索没有得到可核对的论文标题命中；本记录不把名称变体当作论文。
- FlashMesh 初始按猜测的 CVF PDF 文件名访问失败，随后从 CVF 官方 HTML 页面取得页面内给出的正式 PDF URL并成功下载。
- MeshAnything V2 初始猜测的 CVF 页面大小写路径访问失败，随后通过 CVF 的 `MeshAnything` 查询结果取得正确页面和 PDF URL。
- XSpecMesh 的 OpenReview 页面 `https://openreview.net/forum?id=7s68UdT65x` 及其公开 API 在本轮返回验证挑战/403；没有得到 venue、decision 或录用日期。XSpecMesh 保持“arXiv 预印本、录用状态未确认”。
- 部分 Crossref 请求遇到 SSL EOF；Google Scholar 本轮对应结果文件为空。论文身份和正文结论均未依赖这些失败来源。
- `MeshXL`、`Meshtron`、`MeshAnything` 等名称进入检索词和相邻发现范围；遵照本任务范围，正文详细研读只保留 MeshGPT 与 MeshAnything V2 两篇相邻工作。

检索路径：`research-router -> academic-local -> academic-evidence`。本轮只新增本文档；没有改动源码、理论文档、主大纲或生成状态页。

主代理整合后，公共资料目录由临时位置移入Skill稳定资料区，本文路径同步更新，未复制PDF。方法结论以表中所选官方PDF及XSpecMesh v2为准；原先取得的arXiv版本与相邻发现资料保留作为版本来源记录，不额外计入四篇正文研读。MeshAnything V2官方页内PDF路径的`with`大小写已直接核对。
