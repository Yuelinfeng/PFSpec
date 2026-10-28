# GaussianGPT 阶段工作汇报

整理日期：2026-10-08（Asia/Shanghai）  
实验记录截止：2026-10-07 的 S13.2；更早的 S2/S3 按已有记录纳入。  
当前状态：暂缓新增实验，完成阶段整理；S14 在本报告原始材料中属于后续方案。  
文档用途：汇总项目背景、已完成工作、实际结果、失败原因、代码与产物位置，以及尚未解决的问题。

本报告覆盖当前工作区能够找到的项目文档、源码参考和补丁。远端性能数字来自历史实验记录，本轮没有连接 Autodl 重新读取原始 JSON，也没有运行训练、推理测速或渲染。文中“已完成”“已通过”描述的是对应历史阶段的记录；本轮直接完成的核对另行说明。未留在这些材料中的工作无法据此补全。

同日追加说明：用户随后授权 Computer Use，已直接阅读远端 P-only 引擎、proposer、查表适配器、建表脚本和运行入口，并读取 S8 表评价 JSON。详见 [两种 Position Only 方法的实现汇报](POSITION_ONLY_IMPLEMENTATION_REPORT_2026-10-08.md)。此前性能矩阵仍沿用历史记录，未重新测量。远端列表已出现 S14 评价/overlap 目录，尚未阅读内容；本文关于 S14 的原方案描述不能用于断言远端当前没有相关工作。

## 1. 阶段结论

当前最有证据支持的加速方向是**结构化 Position micro-draft**：用低成本位置查表替代每轮 D8 Transformer 提议，继续由 Target 生成 Feature。S9 的 12 个种子与长度组合中，结构化路线相对同次 Target 自回归生成的吞吐比均大于 1，平均为 **1.36149×**，范围为 **1.30586–1.46463×**。这一结果限于现有起始方式、权重和轨迹设置，尚不构成独立场景上的泛化结论。[依据：实验记录 S9](EXPERIMENTS.md)

神经草稿训练已从常规 D8、Target 轨迹蒸馏推进到 Draft 自生成轨迹蒸馏。它们能够改变分布指标，但没有证明持续训练就能带来稳定的额外吞吐。S13b 的真实联合 P/F 路线在三个种子上均慢于纯 Target，平均吞吐比 **0.9104×**；Position 接受率约 **94.79%**，条件 Feature 接受率约 **42.31%**，成对接受率约 **40.10%**。现阶段不支持按相同配方继续增加训练步数。[依据：实验记录 S13.1–S13.2](EXPERIMENTS.md)

这一阶段的主要积累包括：明确 P/F 条件依赖；实现 Position-only 概率接受与拒绝后采样；建立阶段计时和分布诊断；完成选择性输出计算、紧凑采样、结构化查表、统一 proposer、cascade、Feature 旁路评价与真实联合原型；用实验排除了几条看似可行但成本不合算的路线。

| 工作方向 | 当前证据 | 阶段判断 |
| --- | --- | --- |
| D8 Position-only | 早期单次约 1.06–1.15×；S9 矩阵平均 1.03719× | 保留神经草稿参照；收益受草稿计算成本限制 |
| 结构化 Position 查表 | S8.4 与 S9 两批种子/长度评价均为正收益 | 当前优先保留的加速候选；仍需独立场景和质量评价 |
| 选择性 LM head、compact sampling | 完整路线约 1.058–1.084×；compact 部分增量约 1.51% | 有实际工程收益，但未消除 Target 主计算成本 |
| Target 轨迹与 Draft 轨迹蒸馏 | 部分 KL/overlap 和 P 接受改善；速度收益不稳定 | 保留研究结果和权重，不以训练损失直接决定部署 |
| 真实联合 P/F | S7.4 约 0.89–0.92×；S13b 约 0.91× | 当前草稿质量及提议成本不足 |
| 多对完整 token 提议 | 已测 K=1/2/4 均低于纯 Target 吞吐 | 保留负对照，缺少继续扩大 K 的性能依据 |
| GPU tensor/vectorized 查表 | 分布一致；单 query 端到端慢于 Python | 保留实现与对照，批量微测量收益尚未进入引擎 |
| eager/lazy cascade | 状态流程可运行；有 D8 回退时未胜过纯查表 | 保留原型，不据此替换结构化路线 |
| S14、类型分支、Target 状态辅助草稿 | 已有方案，未见对应完整实施结果 | 本轮全部暂缓 |

不同表格中的速度比来自不同日期、脚本和条件，不能直接拼成同一次横向竞赛。本文优先保留各次运行内的对照关系。

## 2. 核查范围与证据来源

### 2.1 本轮直接核对的对象

| 对象 | 实际位置 | 本轮能够确认的内容 |
| --- | --- | --- |
| 本地主工作区 | `/Users/lief/Documents/ChatGPT/GaussianGPT` | 交接文档、补丁和项目配置存在；Git 分支为 `main`，尚无提交 |
| 项目入口及状态 | `README.md`、`.codex/AI_START_HERE.md`、`docs/project/PROJECT_STATUS.md` | 原阅读顺序、状态生成范围、历史工作区说明 |
| 长期项目文档 | `docs/project/` 中原有 9 份 Markdown | 架构、S2–S13 记录、方法历史、研究候选、训练方案及操作约定 |
| 早期源码参考 | `.codex/remote-source/` | 模型与训练配置、P-only 引擎、计时脚本、序列化和环境记录；该目录被 Git 忽略 |
| 本地补丁 | `patches/target_head/`、`patches/structured_sd/` | 选择性输出/紧凑采样脚本、联合块引擎与验证入口的源码内容 |
| 研究材料 | `/Users/lief/Downloads/deep-research-report.md` | 阅读摘要、路线排序与结尾建议；只用作历史方案来源 |

本地早期源码参考约 704 KiB，包含模型与配置，但缺少后续结构化 proposer、cascade 和 S13 训练实现。它可以说明当时的设计，无法确认 2026-10-08 的远端运行版本。

本地根目录未找到完整的 `model/`、`scripts/`、`conf/`、`serialization/` 运行工程，未找到模型权重或原始性能 JSON。旧状态页记录时间为 2026-10-07 13:34:52 +08:00；其扫描对象是根目录中的运行资产，不包含 `.codex/remote-source/` 参考副本。因此“根目录没有源码”和“本地存在部分源码参考”可以同时成立。

### 2.2 三种证据分别使用

1. **直接文件证据**：本轮实际读取的配置、源码和补丁。能够确认内容存在；没有据此声称模型已在本机运行。
2. **历史实验记录**：`EXPERIMENTS.md` 中的计数、速度、参数和远端路径。能够归纳此前结论；原始产物完整性和远端当前状态未在本轮复核。
3. **方案与解释**：研究报告、训练计划和本文的综合判断。它们说明为什么选择某方向，不能替代实施结果。

本地没有源码提交历史，无法生成覆盖此前全部开发工作的 commit 清单。也没有完整作业账单、GPU 小时、逐次运行日期和全量训练曲线，因此本文不估算累计人时、总算力成本或项目开始日期。

## 3. 原项目机制与本阶段目标

### 3.1 从场景到生成结果

历史架构记录的流程是：3D 场景经稀疏 3D VQ-VAE 编码，得到位置与离散特征组成的序列；GPT 逐 token 生成；VQ-VAE decoder 将 token 还原为 Gaussian 表示，之后由渲染器输出场景图像。[依据：架构说明](ARCHITECTURE.md)

```text
3D-FRONT / ASE / PhotoShape 等数据
    → 稀疏 VQ-VAE 编码
    → [P0, F0, P1, F1, …]
    → 带 3D RoPE 的自回归 GPT
    → token 解码为 Gaussian 场景
    → 场景渲染与质量评价
```

P 是稀疏空间位置 token，F 是该位置的离散潜在特征。当前稀疏配置中，两类 token 按对交替；排除终止符等特殊项后，数量占比接近各一半。已有“约 90%”的 Position 指标描述接受或匹配情况，不能解释为 Position 占序列的 90%。

早期配置参考给出的 Target 为 24 层、宽度 1024、16 个注意力头，上下文长度 16384。Position ID 为 `0..7999`，Feature ID 为 `8000..12095`，EOS 为 `12096`；位置坐标参与 3D RoPE。生成时还要满足位置顺序与合法集合限制，不能仅依赖静态的 token 类型区间。

3D-FRONT/ASE 使用 z-up，PhotoShape 使用 y-up。S2 数据配置为 `max_tokens=16000`、`min_chunk_occupancy=0.3`、`chunk_origin=[null,null,-1]`。坐标、词表、序列顺序和上下文能力必须与 Target 保持一致，才能解释草稿接受率。

### 3.2 本阶段优化的是哪段时间

已完成工作主要优化 GPT 的自回归 token 生成。文中 tokens/s 和 speedup 大多对应完整解码循环，包括草稿、Target 验证、采样、拒绝处理、缓存和调度；它们通常不含模型加载、VQ-VAE 场景解码和最终渲染。

因此，报告中的 1.36× 表示所测 token 生成路径的吞吐比。现有材料不足以把它换算为完整房间或多 chunk 场景的总时间收益，也不足以推断图像质量变化。

### 3.3 Position-only 的条件关系

当前 P-only 路线中，下一位置依赖已经提交的真实 Feature：

```text
已提交历史 H → 提议 P_i → Target 验证 P_i 并产生 F_i 分布
             → 接受 P_i，或拒绝后重新采样 P_i
             → Target 在最终 P_i 条件下采样 F_i
             → 提交 [P_i,F_i] → 进入下一轮
```

同一历史下，Target 和 Draft 的最终采样概率分别记为 p、q。候选 y 的接受概率为 `min(1,p(y)/q(y))`；拒绝后从归一化的 `max(p-q,0)` 采样。两者须使用正确的合法 token 集合和部署采样参数。平均理论接受概率为 `sum(min(p,q))`，实测接受率则来自实际接受次数。

这一机制允许替换 proposer，同时继续使用同一 Target 分布。拒绝后需舍弃旧候选对应的缓存，并根据最终提交的位置重新取得 Feature 分布。结构化查表、D8 和 cascade 的比较建立在这条共同路径上。

## 4. 工作演进总表

| 阶段 | 记录日期 | 已完成的主要工作 | 结果与变化 |
| --- | --- | --- | --- |
| S2 | 未单列完整日期 | D4/D6/D8/D12 配置；canonical D8 38 epoch 权重 | 建立神经草稿基础；仅 D8 有本文所需的完整使用记录 |
| S3 与早期 P-only | 至 10-03 | Feature 蒸馏、Target 轨迹训练、分布诊断、Position-only 解码 | P 接受较高但速度收益有限，开始拆解成本 |
| 阶段计时 | 10-03 前后 | 1024/2048/4096 token 分项计时 | 主要成本来自 Draft 和 Target 前向 |
| S4 | 10-03 | 共同 temperature 的敏感性实验 | 接受率随温度变化；同时改变生成目标分布 |
| S5 | 10-04 | 完整 P/F 多对提议负对照 | K=1/2/4 均未达到纯 Target 吞吐 |
| S6 | 10-04 | 选择性 LM head、compact sampling | 数值与短运行验证通过，有小幅工程收益 |
| S7 | 10-04 | F-aware 短训、Feature 旁路统计、真实联合 P/F | 旁路复用 q_F 可行；真实联合受 F 质量及额外前向限制 |
| S8 | 10-04 至 10-05 | 位置 gap 表、概率适配、E2E、统一接口、查表实现对照 | 结构化低成本提议成为当前主要正结果 |
| S9 | 10-05 | 12 个 structured 与 12 个 D8 单元 | 结构化平均 1.36149×；实际仍为种子轨迹测试 |
| S10 | 10-05 | eager structured→D8 cascade | 即便不回退也运行 D8，抵消查表收益 |
| S11 | 10-06 | lazy D8 回放及再次提议 | 不回退恢复收益；回退较多时仍慢于纯查表 |
| S12 | 10-06 | Position policy-KL、7 个短训、24 个解码结果 | 训练指标有轻微变化，未见稳定额外吞吐 |
| S13a/b | 10-07 | Draft 自生成轨迹蒸馏及三种子旁路评价 | P/F 指标没有同时超过 D8 |
| S13.2 | 10-07 | S13b 真实联合 P/F 三种子评价 | 平均 0.9104×，不支持延续同配方加长训练 |
| 阶段整理 | 10-08 | 汇总证据、厘清旧计划、记录当前暂缓状态 | 本报告与入口文档更新；没有新增实验 |

后文所有历史实验数字均以 [实验记录](EXPERIMENTS.md) 为主要来源；方法解释同时参考 [方法历史](DRAFT_METHODS_HISTORY.md)、[架构](ARCHITECTURE.md) 和 [逻辑验证记录](LOGIC_VALIDATION.md)。

## 5. S2、S3 与早期性能定位

### 5.1 小型 Transformer 草稿

四个草稿配置保持相同词表、稀疏序列、3D RoPE 和 16384 上下文，仅调整容量。

| Draft | 宽度 | 层数 / 注意力头 | 本阶段证据 |
| --- | ---: | ---: | --- |
| D4 | 256 | 4 / 4 | 配置存在；没有完整吞吐对照可供本报告判优 |
| D6 | 384 | 6 / 6 | 配置存在；没有完整吞吐对照可供本报告判优 |
| D8 | 512 | 8 / 8 | canonical 38 epoch 权重被后续多轮使用 |
| D12 | 768 | 12 / 12 | 配置存在；没有完整吞吐对照可供本报告判优 |

S2 配置为 38 epochs、每批 4 个 scene、梯度累积 4、bf16 mixed，每 10 epoch 渲染 4 个 sample。早期配置注释与优化器实现说明采用 Muon + AdamW 的参数分组；不能仅修改普通 `training.lr` 就假定所有 GPT 学习率随之变化。

可复用 D8 路径见第 14 节。路径末尾为 epoch37、step50502，本文沿用“canonical D8”名称，不推断其余容量均已完成同等训练。

### 5.2 Feature 蒸馏和 Target 轨迹训练

S3 在 token 交叉熵之外加入 Feature KL，记录 CE、KL、top-1 agreement、raw overlap 与 policy overlap。Target 轨迹版本先采样语料，再训练 D8；分布诊断脚本用于区分词表类型、熵和匹配程度。

同一 seed=5 的记录显示：

| 草稿及长度 | P 实际接受率 | Feature overlap | Draft 双 token 时间 | Target 双 token 时间 | 模型成本公式估计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D8，1024 | 97.27% | 0.3876 | 6.4889 ms | 16.8301 ms | 1.1407× |
| Target 轨迹蒸馏 256 step，1024 | 98.24% | 0.7037 | 6.5937 ms | 17.0964 ms | 1.1424× |
| 同权重，2048 | 97.85% | 0.6107 | 6.5138 ms | 16.9148 ms | 1.1529× |
| Target 轨迹 pilot 64 step，1024 | 93.16% | 0.3384 | 6.8000 ms | 17.5779 ms | 1.0964× |

256-step 权重确实改变了分布，64-step 权重未达到同样效果。但 Feature 此时由 Target 采样，Feature overlap 上升不会直接少运行一次 Target。表中末列只计算模型耗时项；另一次 256-token wall-clock 为 1.1369×，仅有单 seed 且两路径使用不同 seed offset。

### 5.3 为什么较高接受率没有带来大幅加速

seed=5、warmup=128 的阶段计时给出：

| 输出 token 数 | AR 的 P/F 两步合计 | Draft 更新 | Target 双 token 验证 | 拒绝后的 Target 单步 | P 接受率 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 26.7618 ms | 6.5024 ms | 16.8440 ms | 13.2853 ms | 97.27% |
| 2048 | 26.7277 ms | 6.4756 ms | 16.7586 ms | 13.1039 ms | 97.66% |
| 4096 | 28.7837 ms | 6.6688 ms | 17.6637 ms | 15.0392 ms | 96.92% |

按 `AR两步 / [Draft更新 + Target验证 + (1-P接受率)×拒绝重算]` 估计，加速约 1.13–1.16×。P 拒绝只占约 2%–3%，加权重算约 0.3–0.5 ms/对；Draft 每轮约 6.5–6.7 ms 的固定成本更突出。

已计时的非模型部分中，Feature 采样约 0.42 ms/对、P 后处理约 0.35 ms/对、Draft P 采样约 0.31 ms/对、Feature 诊断约 0.30 ms/对。缓存回退每次约 0.035–0.040 ms，且只在拒绝时发生。由此形成的判断是优先降低模型与提议成本；这些分项没有覆盖所有调度，也不能直接相加作为完整生成耗时。

### 5.4 S4 温度实验

在 D8、seed=5、1024 token、top-p=0.9 下，共同 temperature 为 0.7/0.9/1.1 时，P 接受率分别为 98.83%/97.27%/85.55%，模型公式估计分别为 1.1644×/1.1407×/1.0614×。它说明采样集中度影响接受率；温度同时改变 Target 分布，因此不能把较低温度的数值当作同一生成任务下的等价提速。

## 6. S5–S6：多对负结果与输出计算优化

### 6.1 完整 P/F 多对提议

S5 使用低层缓存 Transformer 连续提出完整 P/F，再由 Target 验证。128-token 记录如下：

| 每轮 pair 数 K | temperature | 相对纯 Target 吞吐 | token 接受率 |
| ---: | ---: | ---: | ---: |
| 1 | 1.0 | 0.533× | 0.487 |
| 2 | 1.0 | 0.571× | 0.398 |
| 4 | 1.0 | 0.434× | 0.213 |
| 2 | 0.1 | 0.795× | 0.630 |
| 4 | 0.1 | 0.891× | 0.563 |

记录中的 token 合法性与缓存状态均通过，但端到端吞吐全部低于纯 Target。Draft 的中间 F 改变后续 P 的条件，连续候选容易积累分布偏差；额外 Draft 前向和最终被拒候选的验证也需要时间。

这一结果限定的是当前实现、权重和成本。对同一候选历史逐步计算正确概率，联合提议在数学上仍可实现；不能把条件依赖解释为所有联合方法都不可行。

### 6.2 3.67× 数值的实际含义

历史 Target block 计时：K=1 约 22.76 ms，K=2 整块约 23.26 ms，K=4 整块约 24.76 ms。按每 pair 摊销，K=4 约为 6.19 ms，`22.76/6.19≈3.67`。

本轮直接读取的 `.codex/remote-source/scripts/profile_structured_block_cost.py` 使用合成 token、短 KV 缓存和 `1+2K` 输入行；不同 K 对应 3/5/9 行，包含边界处理。因此 3.67× 仅描述该微测量的单位 pair 前向效率，没有计入真实草稿生成、接受、拒绝、缓存管理及输出长度。它既不能替代完整解码倍率，也不表示已获得 3 倍以上场景生成速度。

### 6.3 选择性 LM head 与 compact sampling

S6 只计算当前 token 类型合法的输出行，并在采样时去掉已被 mask 排除的词表项。Position 紧凑集合为 8000 个位置加 EOS，Feature 为 4096 个 ID；采样结果再映射回全局 token ID。

| 请求长度 | seeds | Target 平均 tok/s | 优化路线平均 tok/s | 吞吐比 |
| ---: | --- | ---: | ---: | ---: |
| 128 | 5/6/7 | 72.8046 | 77.0298 | 1.0580× |
| 1024 | 5/6/7 | 72.0142 | 76.7853 | 1.0663× |
| 2048 | 5/6/7 | 73.6560 | 78.6005 | 1.0671× |
| 4096 | 5/6 | 69.1468 | 74.9337 | 1.0838× |

128-token 仅选择性 head 的路线为 75.8818 tok/s；加入 compact sampling 后为 77.0298 tok/s，增量约 1.51%。上述总倍率包含整个优化路线，不能全部归因于 compact sampling。

历史数值验证中，合成 logits 的紧凑与全词表采样分布最大差为 0；真实模型合法 logits 误差约 `6.7e-6/4.8e-6`，mask 外无有限值。32-token 引擎短运行完成 16 对、14 次 P 接受，状态正常。

限制包括：性能 runner 当时位于远端 `/tmp`；4096 仅两个等长种子；额外 seed7/8/9 出现单侧提前 EOS，已从等长均值中排除。概率等价也不要求同 seed 输出逐 token 相同，因为不同张量形状可能改变随机数消耗。

## 7. S7：从 Feature 诊断到真实联合接受

### 7.1 F-aware 短训练

S7.2 使用 Target 生成的 8 条、每条 1024 token 轨迹，训练 128 step，Feature KL 权重为 0.3。第一次因验证间隔 64 大于 8 个训练 batch 未进入训练，调整为 8 后完成。验证 Feature policy overlap 为 0.34996，raw overlap 为 0.38900，top-1 agreement 为 0.20410。

同一 256-token runner 的即时对照中，seed1 的 canonical/候选速度比为 1.0228×/1.0440×，seed2 为 1.0612×/1.0549×。候选 P 接受数略高，但速度改善方向不一致。两个种子的速度比均值约为 1.0420× 与 1.0494×，不足以据此确定额外训练收益。

另一次 Feature diagnostic 在输出 KL 时遇到 `None` 格式化异常。历史记录没有给出后续完整解决结果，因此该入口的当前可用性仍待确认。

### 7.2 Feature shadow：测量候选，不提交候选

S7.3 在 P 已接受的条件下，从 Draft 的 q_F 采样旁路候选并计算接受统计；最终输出的 F 仍由 Target 采样。实现复用已提交 `[P_i,F_i]` 的 `draft_update_2` 第 0 行 logits，第 1 行继续提供下一 P 的分布。因果性保证第 0 行不读取后面的 F 输入，因此可以取得该位置的 q_F，无需额外探测前向。

| 指标 | 两次重复的记录 |
| --- | ---: |
| P 接受 | 119/128 = 92.97% |
| 条件 F 旁路接受 | 60/119 = 50.42% |
| 成对旁路接受 | 60/128 = 46.88% |
| Feature policy overlap | 0.48695 |
| 完整路线速度比 | 1.02490×、1.01728× |

这一步提供了条件 F 接受的观测办法，去掉此前约 5 ms/event 的额外探测成本。但观测发生在真实 P/F 已提交之后，不能把这种复用方式直接视为提前提出 F 的免费计算。旁路接受也不代表 F 已进入输出。

### 7.3 真实联合 P/F 原型

S7.4 新增 `model/gpt/joint_feature_speculative.py`，在 Target 验证前先提出 P 与 F。P 被拒绝时回退并根据新 P 取得 F；P 接受、F 拒绝时从 Feature 残差概率采样；两者均接受则提交草稿 pair。

两次 256-token 运行的计数相同：P 为 121/128，条件 F 为 51/121，成对为 51/128。历史表中的参考/联合吞吐为 72.725/66.804 与 73.150/65.109 tok/s，速度比为 0.9186×、0.8901×。额外 `draft_feature_probe` 和约 42% 的条件 F 接受没有形成净收益。

小词表联合 Monte Carlo 运行 50,000 次，最大绝对误差 0.00284、L1 误差 0.00772。它支持两级概率接受与拒绝后采样的数学实现，仍无法替代真实模型的独立场景、长序列分布与渲染评价。

旧记录对 S7.4 的参考路线名称存在表述差异：正文提到 Position-only，其他材料称其为普通自回归。由于本轮未取得原始 runner 输出，本文保留“历史参考吞吐”的称呼，不用这两次运行做跨方法的精确比较。S13.2 则明确使用纯 Target 作为对照。

## 8. S8–S9：结构化 Position micro-draft

### 8.1 位置序列规律与查表模型

S8 复用 Target 轨迹语料：32 条训练、4 条验证，每条 2048 sparse token。只从 P 序列取相邻位置的正 gap，以最近 h 个 gap 和当前位置的低位 Morton/Z-order 特征构造查询上下文；按长历史、短历史、特征级和全局级依次回退。

训练/验证 gap 熵记录为 1.8317/1.7023 bit。选择 h=5、低 1 位特征后，验证 mode top-1 为 3480/4092=85.04%，K=4 连续命中均值 2.57969，全部四个位置命中比例 54.48%，表项数 1325。h=4 为 84.68% top-1、728 项；h=6 为 84.995%、2166 项，扩大历史没有带来清楚的额外收益。

上述数值衡量给定真实前缀的预测能力。后续 adapter 将 gap 计数归一化为 q_P，在合法 Position/EOS 上执行概率接受与拒绝后采样，才得到真实神经 Target 下的接受率。纯 CPU 查表约 0.7186 μs/query，也不能代替引擎中的设备同步和 support 映射成本。

S8.1 的 synthetic 100,000 次采样中，理论 overlap 0.45、经验接受率 0.44762，输出 L1 0.00340；另一个观测 82 次的真实 context 中，理论 overlap 0.86697、经验接受率 0.86612，L1 0.00324。这些验证支持概率路径，可以进入真实 Target 引擎。

### 8.2 从单次 E2E 到多种子长度评价

S8.2 的首次真实运行使用 RTX 4090、seed1、256 token、warmup32：纯 Target 为 71.225 tok/s，结构化路线为 97.354 tok/s，速度比 1.36685×。P 接受 115/128，拒绝 13 次，缓存状态验证 128/128 通过；F 始终来自 Target。

S8.3 的单种子长度扩展在 1024/2048/4096 上得到 1.34220×/1.34804×/1.35691×，均生成完整请求长度。S8.4 随后覆盖 seed0/1/2 和四个长度，共 12 个 endpoint：

| 请求长度 | 速度比均值 ± std | P 接受率均值 ± std | lookup 均值 |
| ---: | ---: | ---: | ---: |
| 256 | 1.39624 ± 0.03090× | 0.93750 ± 0.03984 | 133.43 μs/query |
| 1024 | 1.39089 ± 0.03490× | 0.90951 ± 0.01218 | 167.60 μs/query |
| 2048 | 1.36011 ± 0.01345× | 0.90527 ± 0.00498 | 206.46 μs/query |
| 4096 | 1.36546 ± 0.01149× | 0.91097 ± 0.00294 | 253.32 μs/query |

历史记录保存了速度、接受率、吞吐与 lookup 的 PNG/PDF，以及一个 4096-token 轨迹的 orbit GIF。本轮未重新观看这些远端图像；它们属于已记录的产物，不构成本轮新增视觉评价。S8.4 的收益在该轨迹设置中跨 seed 和长度保持为正，独立场景泛化仍待评价。

### 8.3 统一 proposer 接口

S8.5 将 runner 中的适配方式接到正式接口：`propose_position(state)`、`commit(token)`、`rewind(state)`。`NeuralDraftProposal` 保留 D8 前向，`StructuredPositionProposal` 使用查表，Target 验证、采样、拒绝处理和缓存提交继续由统一引擎管理。

正式 structured 的 256-token 短运行速度比为 1.39922×，P 接受 115/128；无 proposer 的旧 D8 路线同版本为 1.04136×，P 接受 113/128。两者缓存验证均通过。该阶段证明接口迁移可运行，单次速度不替代多种子矩阵。

### 8.4 GPU tensor 与 vectorized 查表

S8.6 的 python/tensor/vectorized 实现采用同一表和回退规则。512 个 CUDA query 的概率输出，以及 batch=1/8/32/128 的输出，相对 Python 的最大与总绝对差均为 0。

| 实现 | 引擎内 lookup 均值 | 256-token 速度比 | 实际收益 |
| --- | ---: | ---: | --- |
| Python | 129.064 μs/query | 1.36503× | 当前可比较的最快模式 |
| tensor | 534.570 μs/query | 1.33324× | 单 query 未降低端到端成本 |
| vectorized | 658.683 μs/query | 1.33043× | 引擎仍逐 query 调用，批量接口没有发挥作用 |

独立微测量中 vectorized batch=8/32/128 每 query 约为 120.628/83.460/75.420 μs，说明批量场景存在收益空间。当前引擎没有对应调度，fused CUDA 实现也未注册。不能据此声称 GPU 查表已经加速完整路线。

在该 256-token Python 运行中，记录的 lookup 只占约 0.63% 总时间。S8.4 的 lookup 随长度增长值得追踪，但仅凭增长趋势不足以判定它是完整路线的最大成本。

### 8.5 S9 的 structured 与 D8 矩阵

S9 固定 RTX 4090、Target/VQ-VAE/canonical D8、temperature/top-p=0.9、warmup128、Python lookup，运行 seed0/1/2 × 长度256/1024/2048/4096。每个路线有 12 个单元，并各有同配置的纯 Target 对照。

| 请求长度 | structured 速度比 | structured P 接受率 | D8 速度比 | D8 P 接受率 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1.38617 ± 0.06483× | 0.93750 | 1.04733 ± 0.02501× | 0.91406 |
| 1024 | 1.37055 ± 0.01901× | 0.90951 | 1.02437 ± 0.04618× | 0.87565 |
| 2048 | 1.33807 ± 0.00462× | 0.90527 | 1.02051 ± 0.03819× | 0.88046 |
| 4096 | 1.35117 ± 0.00499× | 0.91097 | 1.05654 ± 0.03299× | 0.88323 |

structured 12/12 单元为正收益，均达到请求长度，总体速度比均值 1.36149×。D8 均值 1.03719×，最低单元为 0.96349×；seed2 的 2048/4096 请求均在 1753 token 合法 EOS，这两行属于非等长运行。

目录名为 `s9_multiscene`，实际 `scene_mode=eos_seed_trajectory`；runner 当时没有独立 scene/prompt 输入。纯 Target 使用 seed+10，route 使用 seed+11。该矩阵覆盖不同随机轨迹及长度，不能写成三个独立场景，也不能要求两条路径输出同一序列。

### 8.6 这一方向的阶段贡献

结构化路线的主要收益来自移除神经 Draft 的固定前向成本。即使 P 接受率略低于部分 D8 运行，总成本仍更低。它提供了一条明确的研究参照：未来神经草稿需要同时与纯 Target 和这条低成本路线比较，单独提高接受率不足以证明价值。

## 9. S10–S11：结构化与 D8 cascade

S10 将结构化表和 D8 包装成 `CascadePositionProposal`，以 top-1 probability 或熵决定是否使用神经提议。Target 接受与拒绝后的采样继续使用统一路径。

| 版本与阈值 | P 接受 | D8 fallback | 256-token 速度比 |
| --- | ---: | ---: | ---: |
| S10 structured-only | 115/128 | — | 1.3302× |
| S10 eager，top1≥0.5 | 115/128 | 0 | 1.0137× |
| S10 eager，top1≥0.9 | 122/128 | 18 | 1.0369× |
| S10 eager，entropy≤1.0 | 115/128 | 4 | 1.0089× |
| S10 D8 | 113/128 | — | 1.0157× |
| S11 lazy，top1≥0.5 | 115/128 | 0 | 1.37455× |
| S11 lazy，top1≥0.9 | 127/128 | 61 | 1.04909× |

eager 版本每轮先运行 D8，即使 fallback=0 仍支付草稿成本。lazy 版本在低置信度时才创建 D8 cache，并回放已提交历史取得 logits。无回退时恢复纯查表量级的收益；较多回退时 P 接受提高，但 replay 和 D8 更新抵消节省。

这些结果为同一 seed 的短运行，缓存状态均通过。eager 与 lazy 的 fallback 数来自各自实际轨迹，不能把 18 与 61 当作相同历史上的策略差值。当前没有证据显示 cascade 优于 structured-only，不作为优先部署方向。

## 10. S12：Position policy-KL 与 F-aware 矩阵

### 10.1 新训练接口

S12 将 `position_policy_kl_weight` 加入 Feature 蒸馏的两条入口。在 Position `0..7999` 加 EOS `12096` 的集合中，用 temperature/top-p=0.9 计算 Teacher/Draft policy，复用已有前向 logits，并记录 Position KL、overlap 和有效行数。

2-step GPU 短运行使用训练2条、验证1条256-token轨迹、seed2401、bf16，P/F KL 权重均为0.3。训练/验证 Position KL 为0.27861869/0.70167381，overlap 为0.97100383/0.86094409，有效 P 行均为129。该阶段确认损失与输出记录可运行。

### 10.2 七个短训练

S12.1 固定 canonical D8 起点、相同小语料和采样参数，完成五个8-step、两个16-step运行。

| λF / λP / steps | val Feature KL | val Feature policy overlap | val Position KL | val Position policy overlap |
| --- | ---: | ---: | ---: | ---: |
| 0 / 0 / 8 | 2.24902 | 0.21318 | — | — |
| 0.1 / 0.3 / 8 | 2.24791 | 0.21339 | 0.70129 | 0.86148 |
| 0.3 / 0 / 8 | 2.24661 | 0.21340 | — | — |
| 0.3 / 0.3 / 8 | 2.24498 | 0.21354 | 0.70097 | 0.86142 |
| 1.0 / 0.3 / 8 | 2.24545 | 0.21376 | 0.70036 | 0.86169 |
| 0 / 0 / 16 | 2.24030 | 0.21569 | — | — |
| 0.3 / 0.3 / 16 | 2.23276 | 0.21608 | 0.66137 | 0.86158 |

Feature 指标变化很小，0.3/0.3/16 留作解码候选。各配置的 total loss 包含不同权重的附加项，数值定义随 λ 改变，不能把候选的总损失较高直接解释为基础任务能力下降；需要比较同定义的分项与实际解码。

### 10.3 单次旁路评价与24个解码结果

S12.2 的候选/对照均为 P 接受113/128、F旁路接受54/113。候选与对照吞吐为70.8739/70.4480 tok/s，各自相对同次 Target 为1.02164×/0.99905×。单次候选略快，接受计数完全相同。

S12.3 再比较两个权重，覆盖三个seed、四个长度，共24个JSON：

| 请求长度 | 0.3/0.3 候选速度比 | 0/0 对照速度比 | 候选/对照 P 接受率 | 候选/对照 F旁路接受率 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1.01668 ± 0.01651× | 1.02119 ± 0.02751× | 0.91406 / 0.91406 | 0.36937 / 0.36654 |
| 1024 | 1.03987 ± 0.01225× | 1.04518 ± 0.02746× | 0.95898 / 0.95964 | 0.39262 / 0.39307 |
| 2048 | 1.03951 ± 0.01270× | 1.03844 ± 0.01633× | 0.96973 / 0.96940 | 0.40466 / 0.40103 |
| 4096 | 1.09685 ± 0.02058× | 1.09708 ± 0.03610× | 0.97579 / 0.97755 | 0.42046 / 0.40416 |

速度差随 seed 与长度改变方向，未见稳定训练增益。4096请求的实际输出为候选3319/3731/2339、对照3319/4096/2339，EOS字段均未明确为true。长度不足原因尚未确认，4096行只保留描述性统计，不作为严格等工作量比较。

该阶段没有新增独立场景或重建GIF，也没有逐条重算所有缓存状态。当前结果支持保留权重作为研究对照。

## 11. S13：Draft 自生成轨迹蒸馏及真实联合结果

### 11.1 S13a：改变前缀来源

S13a 从 canonical D8 出发，让 Draft 生成8条512-token训练轨迹和2条验证轨迹，Target 在同一 Draft 历史上提供软分布。训练128step，λF=λP=0.3，seed2401，bf16。

选中权重的验证 Feature KL 为0.65265、Feature policy overlap为0.58502，Position policy KL为0.28940、overlap为0.97421。256-token、warmup32、三个seed的实际旁路评价为：

| seed | D8 P接受 | S13a P接受 | D8 F旁路接受 | S13a F旁路接受 | D8/S13a速度比 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 118/128 | 124/128 | 36/118 | 43/124 | 1.0209 / 1.0358× |
| 1 | 113/128 | 124/128 | 56/113 | 41/124 | 1.0057 / 1.0608× |
| 2 | 120/128 | 118/128 | 72/120 | 39/118 | 1.0105 / 1.0129× |

合并计数后，P接受从351/384=91.41%升至366/384=95.31%；F旁路接受从164/351=46.72%降至123/366=33.61%。seed0的F旁路比例实际上升，其余两个下降。因此“合并后下降”成立，“三个seed均下降”与表格不符。

### 11.2 S13b：增加数据长度和Feature权重

S13b 从 S13a 所选权重继续，用该 Draft 重新生成16条1024-token训练轨迹和4条验证轨迹；λF增至0.6，λP保持0.3，seed2402，完成512step。按最低验证Feature KL=0.59038，选中step128权重。

| seed | P接受 | F旁路接受 | 速度比 | 路线吞吐 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 124/128 | 49/124 | 1.0293× | 76.23 tok/s |
| 1 | 110/128 | 38/110 | 0.9850× | 71.25 tok/s |
| 2 | 125/128 | 36/125 | 1.0501× | 77.71 tok/s |

合并P接受359/384=93.49%，F旁路123/359=34.26%。与S13a相比，Feature旁路略升、P接受下降；仍未同时超过canonical D8。增加语料和Feature权重没有建立稳定的速度优势。

S13a与S13b的轨迹来源、数量和长度不同，其验证KL不能解释为同一验证集上的严格进步幅度。S13b的配置文字曾未同步描述轨迹来源；历史记录按实际student权重路径说明其来自S13a，而非重新从canonical D8开始。

### 11.3 S13.2：实际提交Draft Feature

远端 `generate_joint_feature` 使用S13b的step128权重，在验证前提出P与F；拒绝时执行对应残差采样。固定纯Target、CUDA、256生成token、warmup32、temperature/top-p=0.9，每个seed运行一次：

| seed | P接受 | P接受后的F接受 | 成对接受 | 相对纯Target吞吐 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 116/128 = 90.63% | 49/116 = 42.24% | 49/128 = 38.28% | 0.8929× |
| 1 | 126/128 = 98.44% | 56/126 = 44.44% | 56/128 = 43.75% | 0.9206× |
| 2 | 122/128 = 95.31% | 49/122 = 40.16% | 49/128 = 38.28% | 0.9177× |
| 合并计数/速度均值 | 364/384 = 94.79% | 154/364 = 42.31% | 154/384 = 40.10% | 0.9104× |

平均联合吞吐约65.3 tok/s，纯Target约71.8 tok/s。这里的F计数是真实联合接受，虽然JSON仍沿用`feature_shadow_accepts`等旧字段名。读取字段时必须同时确认实际生成入口。

S13b训练完成512step，最终评价使用所选step128；不能把这组解码结果称为step512权重的表现。它与S7.4约42%的条件F接受接近，说明现有前缀训练与权重调整没有解决联合路径的主要性能短板。

### 11.4 三轮神经草稿的综合判断

| 比较对象 | 合并P接受率 | 合并F旁路接受率 | 已确认的用途 |
| --- | ---: | ---: | --- |
| canonical D8，S13旁路对照 | 91.41% | 46.72% | 原始神经草稿参照 |
| S13a，旁路评价 | 95.31% | 33.61% | P匹配改善，F旁路退化 |
| S13b，旁路评价 | 93.49% | 34.26% | F旁路较S13a略高，仍低于D8 |
| S13b，真实联合评价 | 94.79% | 此处F实际接受为42.31% | 联合吞吐低于纯Target |

旁路与真实联合会访问不同前缀并消耗不同随机数，不能把33%/34%的旁路值与42%的联合值直接解释成同一条件下的改善。已支持的结论是：仅以最低验证Feature KL选择权重，尚未选出有稳定联合速度收益的草稿。

## 12. 研究梳理与方案文档的实际产出

本阶段还完成了研究候选筛选、正确性边界梳理、联合训练方案和面向初学者的说明。它们帮助确定实验顺序，但其中的建议参数和速度目标需与实测表分开理解。

| 已形成的文档工作 | 主要内容 | 当前使用方式 |
| --- | --- | --- |
| 架构与操作记录 | VQ-VAE→token→GPT→Gaussian流程；坐标、采样、环境与运行入口 | 解释原项目和恢复条件 |
| 研究路线梳理 | 蒸馏、动态候选、Target多头、层级token、Feature表示与渲染目标 | 保留候选及适用前提；没有逐项实施 |
| 逻辑验证记录 | Position-only概率路径、缓存、EOS、类型限制及统计边界 | 按各节日期读取，早期限制需要结合后续结果 |
| 方法历史 | 汇总D8、蒸馏、旁路、联合、查表和输出优化 | 内容主要截至10-05；本报告补充S9–S13 |
| 联合训练计划 | 同起点对照、Target/Draft前缀、P/F轻量分支、课程及完整吞吐评价 | 前16节为历史方案，第17节为S13结果后的建议 |
| 联合训练说明 | 给新参与者解释前缀、损失和接受过程 | 教学用途，版本与实际训练参数有差异 |
| 外部研究材料 | 建议低成本P-only优先、再蒸馏、再评估Target状态复用 | 只记录其对项目排序的影响，不把目标区间当作实验结果 |

此前方案中的A/B/C名称在不同文档中并不完全一致。正式联合计划的A是继续Target前缀蒸馏，B是Target/Draft混合前缀，C是加入类型分支；教学说明另有四组版本。S13实际是canonical D8→Draft轨迹128step→候选轨迹512step的连续训练，未完成上述同起点、同预算的A/B/C全套对照。

S14建议再使用A/B/C名称，含义是“on-policy蒸馏 / 加拒绝状态回放 / 再加样本选择”。引用时需注明方案日期及版本，避免把三套标签当作同一实验。

### 12.1 已论证但未完成实施的方向

| 方向 | 已有准备 | 缺少的实际结果 |
| --- | --- | --- |
| 至少三个独立场景的P-only评价 | S8/S9种子与长度矩阵、场景范围说明 | scene/prefix输入、真实场景划分和逐场景结果 |
| fused lookup、引擎批量查表 | tensor/vectorized等价与微测量 | 注册的fused实现或引擎实际批量调用、E2E净收益 |
| P/F类型专用轻量分支 | 残差分支设计，宽度64的建议 | 相同起点的初始化、训练和吞吐对照 |
| 一对→两对→四对课程 | 候选长度、监督位置预算和评价指标定义 | 完整课程运行及连续接受深度曲线 |
| 拒绝状态数据 `D_reject` | 拟保存拒绝候选、残差结果及后续真实状态 | 采集实现、实际语料和训练结果 |
| 接受目标与样本选择 | P/F分项损失、熵和难度分组设计 | 与纯蒸馏同预算的效果对照 |
| Target hidden state辅助草稿 | 信息可用顺序和状态管理建议 | hidden state接口、训练权重和性能结果 |
| Target多头、树、层级token或codebook改动 | 研究方向及限制 | 本项目中的完整实现、训练和质量证据 |

历史材料中出现的1.4–1.8×、1.8–2.8×或2–4×等区间属于研究目标；本文只采用项目实测结果。论文条目与外部材料的引用本轮没有逐篇重新核验，不用于声称GaussianGPT已经达到其他模型的收益。

## 13. 已有验证、统计限制与文档勘误

### 13.1 已积累的验证证据

| 验证对象 | 历史记录的结果 | 能支持的范围 |
| --- | --- | --- |
| Position概率接受与拒绝后采样 | 理论overlap0.55，经验接受0.55172 | 小词表数学路径 |
| 结构化表归一化与残差采样 | 100,000次synthetic及真实context验证通过 | 查表输出成为合法proposal的数学路径 |
| 联合P/F概率组合 | 50,000次Monte Carlo，最大误差0.00284 | 小词表联合实现；无法替代真实长序列评价 |
| 选择性输出与compact policy | logits误差在记录精度范围；synthetic分布差为0 | 相同mask与采样参数下的数值实现 |
| 三种查表实现 | tensor/vectorized相对python输出差为0 | 对应表、support和batch设置 |
| proposer接口 | support、EOS、commit、rewind、reset自测通过 | 对应接口与状态流程 |
| S8/S9输出与缓存 | structured各矩阵达到请求长度；缓存记录通过 | 已测随机轨迹及长度 |
| cascade状态 | fallback可达，256-token缓存验证通过 | 短运行的提议选择和回退路径 |
| S12训练入口 | 2step与7个短训练完成，权重和行数写入 | 小语料损失计算与运行链路 |

本地 `patches/structured_sd/scripts/benchmark_structured_sd.py` 还实现了串行/块前向比较、greedy输出一致性和拒绝后KV回放比对。这些代码存在，但当前没有与该版本对应的完整结果文件；因此本文将其列为验证能力，未将它们额外计作本轮通过项目。

### 13.2 本轮发现并统一的解释

| 问题 | 本报告采用的表述 | 处理位置 |
| --- | --- | --- |
| README曾称联合训练尚未启动 | S13a/b已记录完成；S14未实施，现阶段暂缓实验 | README、项目入口、方案状态说明 |
| 方法历史与教学文档早于S13 | 保留原日期，新增阅读提示；当前综合判断以本报告及S13记录为准 | 两份历史说明顶部 |
| S13a正文称三个seed的F旁路均下降 | seed0上升，seed1/2下降；合并比例下降 | 实验记录S13.1及本报告 |
| S9目录名暗示多场景 | 实际为EOS起始的不同seed轨迹 | 本报告第8节；原统计范围保留 |
| S8.4与S9倍率不同 | 两批运行分别呈现，不拼接均值 | 本报告第8节 |
| 3.67×容易被当作完整收益 | 合成输入、短缓存的Target块前向摊销 | 本报告第6节 |
| Feature overlap、旁路、实际接受混用 | 按运行路径、前缀及分母分别命名 | 本报告第7、10、11节 |
| S12总损失含不同λ项 | 不据总量跨权重判定任务能力 | 本报告第10节 |
| S13b完成512step但评价step128 | 训练预算与实际加载权重分别记录 | 本报告第11节和权重索引 |
| 不同方案共用A/B/C名称 | 按文档日期与含义识别；S13不算完整A/B/C | 本报告第12节 |

S7.4参考路线名称、S12长序列停止原因、旧诊断脚本异常是否解决等事项缺少原始运行证据。本轮未替它们补写确定结论。

### 13.3 仍影响结论的限制

- 每个seed多数只运行一次；seed间标准差描述当前三条轨迹差异，不能代替同配置重复计时的不确定度或跨场景置信区间。
- 速度比的均值与“总输出token/总时间”的吞吐汇总定义不同。本文保留源记录的聚合方式，未用四舍五入后的表值重造汇总。
- P、F和成对接受的分母不同。条件F通常只在P已接受时计数；合并比例按计数求比，不能简单平均各seed比例。
- 不同随机数流与自然EOS会改变实际工作量。输出长度不同的单元需另作解释，不能把较短总时间直接称为相同工作量提速。
- 已有单个GIF与有限状态自测不足以证明独立场景质量保持。尚无覆盖主要路线、相同设置的PSNR/SSIM/LPIPS等质量总表。
- 早期参考源码不含后续模块；本地Git也无已提交源码历史。当前不能精确还原每个结果对应的远端代码版本。
- 本轮未确认Autodl当前GPU进程、作业、服务、权重完整性或数据可读性。暂缓本对话新增实验没有被表述成远端所有任务已停止。

## 14. 可复用资产与位置索引

以下远端相对路径均以历史主工程 **`/root/workspace/GaussianGPT`** 为根。本轮只核对本地文件的存在与内容，远端路径根据实验记录列出，当前可读性待恢复工作时确认。索引不要求复制或清理大文件。

### 14.1 基础与草稿权重

| 对象 | 远端相对路径 | 记录的用途与限制 |
| --- | --- | --- |
| Target | `checkpoints/gpt_vfront.ckpt` | 主要生成及Teacher权重 |
| VQ-VAE | `checkpoints/vqvae_vfront.ckpt` | token对应的场景解码权重 |
| 损坏Target文件 | `checkpoints/gpt_vfront.ckpt.corrupt` | 历史已记不可用，排除实验输入 |
| canonical D8 | `profiling/s2/d8_canonical_38ep/logs/s2_d8_canonical_38ep/lightning_logs/version_0/checkpoints/val_loss_monitor_epoch_37_step_50502.ckpt` | 后续神经草稿主要起点 |
| S3较好256step | `profiling/s3/s3_onpolicy_feature_distill_32x2048_256step/checkpoints/onpolicy_feature_kl_best_epoch07_step256.ckpt` | 高Feature overlap记录；未证明稳定联合速度收益 |
| S3 pilot64 | `profiling/s3/s3_onpolicy_feature_distill_pilot_64step/checkpoints/onpolicy_feature_kl_best_epoch07_step64.ckpt` | 早期负对照 |
| S7短训128 | `profiling/s3/faware_lambda03_128b/checkpoints/onpolicy_feature_kl_best_epoch15_step128.ckpt` | 旁路及初版真实联合评价用权重 |
| S12候选16step | `profiling/s12_faware_matrix/onpolicy_lambdaF03_lambdaP03_16step/checkpoints/last.ckpt` | 训练矩阵候选，解码无稳定额外收益 |
| S12对照16step | `profiling/s12_faware_matrix/onpolicy_lambdaF0_lambdaP0_16step/checkpoints/last.ckpt` | 相同更新数对照 |
| S13a所选128step | `profiling/s13_draft_prefix/run_d8_draft_rollout_128step/checkpoints/onpolicy_feature_kl_best_epoch15_step128.ckpt` | S13b继续训练的实际起点 |
| S13b所选128step | `profiling/s13_draft_prefix/run_candidate_rollout_fkl06_512step/checkpoints/onpolicy_feature_kl_best_epoch07_step128.ckpt` | 512step预算内选中的权重，S13.2真实联合评价对象 |

### 14.2 语料、结果、日志与图像

| 阶段 | 远端目录或主要文件 | 保存的内容 |
| --- | --- | --- |
| 早期P-only | `profiling/e2e_wallclock_seed2.json`、`profiling/e2e_real_seed1.json`、`profiling/e2e_fair_baseline_seed1.json` | 三个早期吞吐与接受记录 |
| 阶段计时 | `profiling/route_a/stage_detail_seed5_{1024,2048,4096}.json` | 模型与后处理阶段计时 |
| S3权重比较 | `profiling/route_a/decomp_seed5_1024.json`、`feature_distill_1024.json`、`feature_distill_2048.json`、`feature_distill_pilot64_1024.json` | 此行后三个文件同在`profiling/route_a/`；不同草稿计时与分布 |
| S3单次E2E | `profiling/e2e_wallclock_faware256_seed1.json` | F-aware256step的短输出速度记录 |
| S3/S8共用语料 | `profiling/s3/s3_onpolicy_feature_distill_32x2048_256step/teacher_trajectory_corpus.pt` | 32训练、4验证的Target轨迹 |
| S5 | `profiling/route_a/structured_k*.json` | 多对完整token负对照及低温对照 |
| S6 | `profiling/route_a/compact_sampling_e2e_{128,1024,2048,4096}.txt` | 选择性输出与紧凑采样性能文本 |
| S7 | `profiling/s3/faware_lambda03_128b/` | 训练、旁路和联合runner；部分原始结果文件名未在材料中单列 |
| S8表 | `profiling/s8_structured_position/h5_b1/structured_zlow3_model.json`、`structured_zlow3_eval.json`、`structured_exactness.json` | 此行三个文件同目录；gap表、离线评价及概率验证 |
| S8种子矩阵 | `profiling/s8_structured_position/grid/`、`multiseed_summary.json`、`multiseed_scaling.png`、`multiseed_scaling.pdf` | 此行汇总和图像位于`profiling/s8_structured_position/` |
| S8 lookup | `profiling/s8_structured_position/lookup_variants.json` | 三种查表实现的数值和成本 |
| 已有渲染 | `profiling/s8_structured_position/visual4096/orbit/gif/scene_000000__idx0000.gif` | 一个已有轨迹的orbit GIF |
| S9 | `profiling/s9_multiscene/phase4_matrix_summary.json`、`phase4_matrix.csv`、`phase4_speedup.png`、`phase4_acceptance.png` | 此行四个文件同目录；structured/D8矩阵 |
| S9日志 | `logs/s9_multiscene/matrix_runner.log` | 矩阵运行记录 |
| S10 | `profiling/s10_cascade/`、`logs/s10_cascade/` | eager cascade结果与日志 |
| S11 | `profiling/s11_cascade_lazy/cascade.json`、`profiling/s11_cascade_lazy_structured/cascade.json` | lazy fallback与纯结构化阈值运行 |
| S11日志 | `logs/s11_cascade_lazy.log`、`logs/s11_cascade_lazy_structured.log` | 两次lazy运行日志 |
| S12训练 | `profiling/s12_faware_matrix/matrix_summary.json`及各配置子目录 | 7个短训练的配置、汇总和权重 |
| S12解码 | `profiling/s12_faware_matrix/decoder_shadow_compare.json`、`decoder_grid/`、`decoder_grid_summary.json` | 此行对象同在S12目录；短运行对照及24个结果 |
| S13a训练 | `profiling/s13_draft_prefix/run_d8_draft_rollout_128step/` | 配置、Draft语料、汇总及权重 |
| S13b训练 | `profiling/s13_draft_prefix/run_candidate_rollout_fkl06_512step/` | `run_config.json`、`draft_trajectory_corpus.pt`、`logs/train.log`、`summary.json`及权重 |
| S13解码 | `profiling/s13_draft_prefix/e2e/` | D8/S13a旁路、`candidate_s13b_seed{0,1,2}_e2e_shadow.json`、`joint_s13b_seed{0,1,2}.json`及runner |

大括号表示记录中的多文件命名方式；通配路径只说明范围，不表示本轮已列出远端全部文件。

### 14.3 本地源码参考、补丁与远端模块

| 功能 | 本地或历史远端位置 | 当前证据 |
| --- | --- | --- |
| Target/Draft配置 | 本地`.codex/remote-source/conf/model/gpt*.yaml`及`conf/training/gpt_draft_s2*.yaml` | 文件存在并已读取关键参数 |
| P-only引擎 | 本地`.codex/remote-source/model/gpt/position_selective_speculative.py` | 早期模块含compact与阶段计时，未含后续proposer/shadow接口 |
| 模型、缓存、序列化 | 本地`.codex/remote-source/model/gpt/`、`serialization/`、`utils/` | 可作早期实现参考，未证明与远端最新版本相同 |
| 早期测速与数学自测 | 本地`.codex/remote-source/scripts/` | 已核对块计时、fair wrapper与数学脚本 |
| 选择性输出补丁 | 本地`patches/target_head/apply_selective_lm_head.py`、`apply_compact_sampling.py`、`validate_selective_head.py` | 文件存在；本轮没有运行修改脚本 |
| 联合块引擎参考 | 本地`patches/structured_sd/model/gpt/structured_speculative.py` | 顺序候选接受、首拒绝后丢弃后缀、额外Target提交等代码存在 |
| 联合块验证/测速 | 本地`patches/structured_sd/scripts/benchmark_structured_sd.py` | 验证与计时入口存在，依赖完整工程和权重 |
| S3训练与诊断 | 远端`profiling/s3/s3_feature_distill_train.py`、`s3_onpolicy_feature_distill_train.py`、`s3_feature_alignment_diagnostic.py` | 历史实施记录，本地未同步 |
| 结构化表与适配 | 远端`scripts/structured_position_lookup.py`、`structured_position_exactness.py`；`model/gpt/structured_position_adapter.py` | 以S8记录为证据 |
| proposer与主引擎 | 远端`model/gpt/draft_proposal.py`、`position_selective_speculative.py` | 以S8.5、S10、S11记录为证据 |
| structured runner与自测 | 远端`scripts/run_structured_position_e2e.py`、`selftest_draft_proposer.py`、`benchmark_structured_lookup_variants.py` | 路线、接口和lookup验证入口 |
| 真实联合pair引擎 | 远端`model/gpt/joint_feature_speculative.py::generate_joint_feature` | S7.4、S13.2使用；与本地联合块引擎需区分 |
| S13训练入口 | 远端`profiling/s13_draft_prefix/s13_draft_prefix_train.py` | Draft语料生成与联合蒸馏入口 |

本地配置与补丁只作为资产索引。本轮未把参考副本移成第二套运行工程，也未应用补丁。

### 14.4 环境与版本记录

本地`.codex/remote-source/gaussiangpt_versions.txt`记录远端曾用Python3.10.21、PyTorch2.8.0+cu129、CUDA12.9、Torchvision0.23.0+cu129、MinkowskiEngine0.5.9、gsplat1.5.0、PyTorch3D0.7.8、FlashAttention2.8.2、Lightning2.5.2与RTX4090。

它描述历史运行环境，没有验证当前远端环境。本轮也未安装或编译依赖。架构文档记录的GCC兼容性顾虑需在重建扩展时单独核实；仅整理文档无需重建环境。

历史交接入口使用`Yuelinfeng/GaussianGPT_SD`作为运行参照，研究文档另引用原项目仓库。两者在本轮没有进行远端分支或提交对应，因此恢复工作时应先确认实际checkout，不能用本地文档Git状态替代源码版本。

## 15. 暂缓后的工作状态与恢复顺序

### 15.1 当前保留的成果

已保留的主要成果是P-only概率与状态路径、分项性能定位、结构化查表的正结果、神经草稿的不同训练方式、真实联合的负结果，以及文档化的代码和产物索引。数据同时说明：提议成本、接受质量和Target验证效率需要共同衡量，提升某一个指标不能自动得到完整收益。

本轮暂缓新增实验；没有远端停止命令、持续运行任务或数据清理操作。用户后续恢复实验时，优先复用已确认的主工程和资产，不按旧文档立即展开大规模训练。

### 15.2 恢复时的顺序

1. 读取实际Autodl主工程、Git状态、GPU任务、数据划分、权重和已有结果，确认当前运行对象及可用资产。
2. 解决scene/prefix来源与输出长度记录，建立相同条件下的纯Target和现有结构化P-only对照；扩展到独立场景。
3. 同时记录生成吞吐、实际长度、终止原因、Target/提议成本、状态正确性和场景重建质量。
4. 若现有路线满足实际需求，保留该路线。若仍有速度缺口，按成本确定是Target计算、support映射、提议质量或其他因素。
5. 只有证据显示神经草稿能解决具体缺口，且收益有机会覆盖额外计算时，再实施S14的小规模数据与损失对照。
6. 只有真实联合接受长度和完整速度均改善后，再恢复多对联合提议或更复杂的类型分支。

这是后续决策顺序，本轮没有启动上述评价或训练。

### 15.3 优先解决的遗留事项

| 优先级 | 问题 | 需要的证据或产出 |
| --- | --- | --- |
| 高 | 实际远端版本与产物是否仍可用 | 主工程、源码改动、权重和结果可读性记录 |
| 高 | 缺少独立场景评价 | 可靠场景标识、划分和同设置多场景结果 |
| 高 | S12的4096输出不足原因不明 | 原始JSON、终止逻辑及每轮生成长度 |
| 高 | 速度与质量评价尚未连成完整场景报告 | token生成、VQ解码、渲染总时间及质量对照 |
| 中 | 不同wrapper、随机数流及计时定义 | 统一实际长度、warmup、计时边界和聚合口径 |
| 中 | Feature诊断旧异常、联合字段沿用旧名 | 实际脚本状态与运行入口说明 |
| 中 | GPU查表批量优势未进入引擎 | 真实调度及成本占比，决定是否值得实现 |
| 低 | 类型分支、Target状态辅助、多尺度表示 | 仅在前述结果形成具体需求后再投入 |

## 16. 本轮整理交付与验证记录

本轮交付为一份阶段主报告，实验总表与资产索引作为文内附录内容保留。项目阅读入口和历史方案顶部增加当前状态说明，S13a的一处计数解释按原表调整。原始数据、模型、脚本、依赖和远端服务均没有变更。

实际新增或修改的文档均位于 `/Users/lief/Documents/ChatGPT/GaussianGPT`：

| 文件 | 本轮内容 |
| --- | --- |
| `docs/project/STAGE_REPORT_2026-10-08.md` | 新增完整阶段报告；续写先前初稿 |
| `README.md` | 加入报告入口，说明S13已完成及S14暂缓 |
| `.codex/AI_START_HERE.md` | 更新当前任务、源码参考范围和恢复实验的阅读顺序 |
| `docs/project/EXPERIMENTS.md` | 加入整理状态，调整S13a逐seed的解释 |
| `docs/project/DRAFT_METHODS_HISTORY.md` | 新增原日期与后续结果的阅读提示 |
| `docs/project/JOINT_PF_DRAFT_TRAINING_GUIDE.md` | 标明教学方案与实际训练配置的区别 |
| `docs/project/JOINT_DRAFT_TRAINING_PLAN.md` | 明确当前暂缓状态及S14未实施 |
| `docs/project/RESEARCH_SOLUTIONS.md` | 标明历史候选与研究目标的适用范围 |
| `docs/project/LOGIC_VALIDATION.md` | 标明早期P-only描述与后续联合路径的范围 |

文档验收范围包括：S2–S13覆盖、原表关键数值及计数比例、引用文件存在性、Markdown表格结构、文档措辞、当前状态与历史方案的时间关系。验收不会把历史PASS重新宣称为本轮模型测试。

2026-10-08文档核对结果：S8/S9、S12及S13表格与源记录的关键字段、合并比例和速度均值共116项比对通过；报告引用与新增报告入口的25处本地文件链接均可解析。所有本轮修改文档的表格列数与代码块配对正常，主报告和项目入口的新增正文措辞符合当前规则。表内远端路径、通配路径与文献页面未作在线可读性验证。

本文没有构造跨场景置信区间，没有重新生成图像，也没有为已缺失的原始数据补造测量。现有证据不足的事项已列入第13及15节。

生成状态页的扫描逻辑已读取。本次仅修改静态文档，没有改变代码、权重、性能结果或运行环境，因此未运行状态刷新脚本；原生成页继续保留其2026-10-07时间。

本地状态脚本目前只读取根目录的环境、权重与`profiling/e2e*.json`，性能字段只支持一种早期结构，不能覆盖后续嵌套结果。本文已用人工索引补充材料范围；脚本兼容性改进保留为后续维护事项，本轮未改业务或工具代码。

## 17. 阅读索引

- [两种 Position Only 方法的实现汇报](POSITION_ONLY_IMPLEMENTATION_REPORT_2026-10-08.md)：同日直接读取远端源码后的独立算法说明。
- [实验记录](EXPERIMENTS.md)：逐阶段事实、运行条件、计数及远端路径。
- [架构说明](ARCHITECTURE.md)：原流程、token与坐标、引擎和proposer职责。
- [方法历史](DRAFT_METHODS_HISTORY.md)：截至原整理日期的方法说明。
- [研究路线](RESEARCH_SOLUTIONS.md)：已整理的论文线索和适用性判断。
- [逻辑验证记录](LOGIC_VALIDATION.md)：概率、状态和统计边界，按日期读取。
- [联合训练计划](JOINT_DRAFT_TRAINING_PLAN.md)：历史A/B/C设计与S13后的S14建议。
- [联合训练说明](JOINT_PF_DRAFT_TRAINING_GUIDE.md)：教学材料，执行参数以实际配置为准。
- [操作说明](OPERATIONS.md)：运行资产、复现和维护约定。
- [本地生成状态](PROJECT_STATUS.md)：只反映本地根目录扫描事实。
- [项目接手入口](../../.codex/AI_START_HERE.md)：当前暂缓状态与阅读顺序。
