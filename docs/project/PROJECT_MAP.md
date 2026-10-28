# GaussianGPT 项目资产地图

本文按本机现有文档整理路径与依赖；本轮未连接远端。每项后附可追溯的本地记录。

## 四个位置

| 位置 | 用途与边界 | 依据 |
| --- | --- | --- |
| 本机文档工作区：`/Users/lief/Documents/ChatGPT/GaussianGPT` | 保存交接文档与局部源码参考；不是远端完整工程。本机状态快照没有 `model/`、`scripts/` 等完整源码目录，也没有权重索引。 | [交接入口](../../.codex/AI_START_HERE.md)、[状态快照](PROJECT_STATUS.md)、[目录审查范围](GAUSSIANGPT_DIRECTORY_REVIEW_2026-10-15.html#summary) |
| 远端主工程：`/root/workspace/GaussianGPT` | 完整源码、基础权重及实验脚本所在位置；目录审查以此为根目录。 | [目录审查](GAUSSIANGPT_DIRECTORY_REVIEW_2026-10-15.html#summary)、[架构说明](ARCHITECTURE.md) |
| 外部数据根目录：`/root/workspace/datasets/GaussianGPT` | 与主工程分开的数据位置。目录记录引用了其下的 `ASE_tokens_base_vqvfront`；本机文档没有给出该数据根目录的完整清单。 | [目录审查的边界与清单](GAUSSIANGPT_DIRECTORY_REVIEW_2026-10-15.html#files)、[多场景计划的数据划分](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md) |
| 结果与数据根目录：`/root/autodl-tmp/pf_multiscene_pilot_20261015/` | 多场景试验、裁剪、阶段 E/F 输出和分析文件的位置，与主工程分开。 | [多场景计划](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)、[阶段 F 输出记录](FULL_3D_OUTPUT_2026-10-15.md) |

## 当前主方法与资产

当前主方法仍为旧 step12000 轻量 PF。统一循环组合 Position ranker 与轻量 FeatureState，在原始 Target 下运行 P/F；阶段 E 评价的正式方法为静态 K2。依据：[计划中的资产与方法](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md#使用的资产与操作范围)、[阶段 E 正式记录](EXPERIMENTS.md)、[目录审查中的代码用途与权重链](GAUSSIANGPT_DIRECTORY_REVIEW_2026-10-15.html#files)。

| 角色 | 远端精确路径 | 作用 | 依据 |
| --- | --- | --- | --- |
| P/F 统一方法源码 | `/root/workspace/GaussianGPT/draft_a/pf_feature_unified.py` | 提供 MLP 与 GPT Feature 接口的 K2 解码循环，并调用 Position ranker。 | [目录审查代码清单](GAUSSIANGPT_DIRECTORY_REVIEW_2026-10-15.html#files)、[多场景计划](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md#使用的资产与操作范围) |
| 多场景评价入口 | `/root/workspace/GaussianGPT/draft_a/pf_multiscene/run_final_evaluation.py`；底层评价脚本 `/root/workspace/GaussianGPT/draft_a/pf_multiscene/feature_real_eval_multiscene.py` | 调度正式裁剪生成、FP32 评价和汇总；底层脚本在真实场景裁剪上运行在线评价。 | [交接入口](../../.codex/AI_START_HERE.md)、[目录审查代码清单](GAUSSIANGPT_DIRECTORY_REVIEW_2026-10-15.html#files) |
| 完整三维评价与流程入口 | `/root/workspace/GaussianGPT/draft_a/pf_multiscene/full_3d_eval.py`、`/root/workspace/GaussianGPT/draft_a/pf_multiscene/run_full_3d_pipeline.py` | 生成完整输出、解码 Gaussian、渲染视图，并串联正式补全与 BOS 输出。 | [阶段 F 文件记录](FULL_3D_OUTPUT_2026-10-15.md#文件与运行对象)、[目录审查代码清单](GAUSSIANGPT_DIRECTORY_REVIEW_2026-10-15.html#files) |
| 原始 Target | `/root/workspace/GaussianGPT/checkpoints/gpt_vfront.ckpt` | Target 概率来源与解码验证模型。 | [多场景计划资产表](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md#使用的资产与操作范围)、[阶段 F 设置](FULL_3D_OUTPUT_2026-10-15.md) |
| VQ-VAE | `/root/workspace/GaussianGPT/checkpoints/vqvae_vfront.ckpt` | 将场景编码为 token，并把生成 token 解码为 Gaussian。 | [多场景计划资产表](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md#使用的资产与操作范围)、[架构说明](ARCHITECTURE.md) |
| Position ranker | `/root/workspace/GaussianGPT/profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib` | 为 P 提案提供排序器；正式多场景计划沿用该权重。 | [多场景计划资产表](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md#使用的资产与操作范围)、[目录审查权重记录](GAUSSIANGPT_DIRECTORY_REVIEW_2026-10-15.html#files) |
| 轻量 Feature 权重 | `/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt` | FeatureState 全局第 12000 步权重；为正式 PF 方法使用的轻量 F。 | [多场景计划资产表](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md#使用的资产与操作范围)、[目录审查权重链](GAUSSIANGPT_DIRECTORY_REVIEW_2026-10-15.html#files)、[阶段 F 设置](FULL_3D_OUTPUT_2026-10-15.md) |

依赖顺序：外部数据与准备好的场景 token 进入 Target/VQ-VAE 流程；Position ranker 与轻量 Feature 权重接入统一 P/F 循环，Target 提供分布与验证；阶段 E 使用预先确定的场景裁剪评价方法，阶段 F 读取正式裁剪运行完整生成，再由 VQ-VAE 解码、导出 Gaussian 并渲染。项目 token 与解码关系见[架构说明](ARCHITECTURE.md)，阶段 E/F 的资产与产出见[研究计划](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)和[阶段 F 记录](FULL_3D_OUTPUT_2026-10-15.md)。

## 阶段 E/F 正式记录

下表仅列本地记忆与正式输出记录中明确写出的运行组合；没有补写未记载的命令参数。

| 阶段 | 已记录的运行组合 | 正式输出位置 |
| --- | --- | --- |
| E：多场景固定窗口评价 | T、P、PF、PF-D8、GPT-D8 五模式；FP32、temperature/top_p=0.9；真实前缀 64/256/1024、最多续写 512 token、三个 seed。30 个测试场景、56 条唯一裁剪，共 2,160/2,160 次。 | `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/fp32_test_full/online.json`；`/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/fp32_test_full/multiscene_summary.json`；状态 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/pipeline.json`；裁剪 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/final_test_crops.pt`；方法与设置 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/method_selection.json`。依据：[多场景计划](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)、[实验索引](EXPERIMENTS.md)。 |
| F：完整单 chunk 输出 | Target、P-only、轻量 PF 三模式；FP32、真实前缀 256 token、temperature/top_p=0.9、正文上限 16,000；30 场景×3 seed（51701–51703）共 270 份，另有 BOS seed 51801–51830 共 90 份。 | 根目录 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_f_full3d/`；正式输出 `test_completion/`、`bos_generation/`；状态 `pipeline.json`；统计 `analysis/summary.json`、`analysis/test_samples.csv`、`analysis/test_scenes.csv`；视频 `analysis/test_completion_30scenes_seed51701_6views.mp4`、`analysis/bos_30trajectories_6views.mp4`。依据：[阶段 F 正式输出与统计](FULL_3D_OUTPUT_2026-10-15.md)、[多场景计划阶段 F](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)。 |

阶段 F 的统计与两条视频均已完成。全图 PSNR/SSIM 仅描述与真实重建参考的差异。远端已有 30 场景参考图及 270 份生成 PNG/NPZ/token 文件；LPIPS 软件可用，但没有可见区域遮罩，且全视图比较会混入随机续写。三个模式在 90 个场景×seed 组中的 256-token 前缀均相同。多 Chunk 连续性尚无结果；该批只覆盖一个 20×20×20 latent chunk。坐标映射和质量比较口径见[阶段 F 记录](FULL_3D_OUTPUT_2026-10-15.md)。

## 历史方法导航

以下仅作资料入口，不在此重复实验结论。

| 方法 | 导航 |
| --- | --- |
| D8 | [真实输入评价](FEATURE_D8_REAL_EVAL_2026-10-14.md)、[阶段 E/F 对照与资产](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md) |
| D4 | [D4 联合训练与评价记录](D4_JOINT_TRAINING_2026-10-14.md) |
| LFQ、F1/F2、K2/K3 | [LFQ 联合候选评价](PF_JOINT_EVALUATION_2026-10-13.md)、[Feature 接受率实验](LFQ_FEATURE_ACCEPTANCE_2026-10-13.md) |
| Medusa | 远端历史目录 `profiling/baselines_5way/b3_medusa_aligned/`、`b3_medusa_conditional/`、`b3_medusa_linear/`；可在[目录审查清单](GAUSSIANGPT_DIRECTORY_REVIEW_2026-10-15.html#files)按目录名定位，方案脉络见[研究方案 §2–3](RESEARCH_SOLUTIONS.md#2-相关论文与可迁移启示)。 |
| EAGLE | 远端历史目录 `profiling/eagle/results/`；可在[目录审查清单](GAUSSIANGPT_DIRECTORY_REVIEW_2026-10-15.html#files)按目录名定位，相关路线见[研究方案 §2](RESEARCH_SOLUTIONS.md#2-相关论文与可迁移启示)及[逻辑筛选 §3](LOGIC_VALIDATION.md#3-候选方案筛选)。 |
| Route-A | 远端历史目录 `profiling/route_a/`；成本拆分与早期记录见[实验记录](EXPERIMENTS.md)和[目录审查清单](GAUSSIANGPT_DIRECTORY_REVIEW_2026-10-15.html#files)。 |
| S2 / S3 | [S2 小模型线](EXPERIMENTS.md#s2小模型-draft-训练线)、[S3 蒸馏与 on-policy 线](EXPERIMENTS.md#s3feature-distillation-与-on-policy-对齐)。 |

## 未解决事项

- 本次只读查看了远端阶段 F 的结果、参考图、样本 token 与相关配置；未枚举远端全工程或完整数据目录。阶段 F 生成入口当前未见运行进程。远端工程已有大量已暂存与未跟踪内容，本次未更动，后续不得把它们当作可清理对象。
- `/root/workspace/datasets/GaussianGPT` 的完整子目录、当前数据版本与各数据对象用途，现有本地文档未逐项列出。
- Stage F 的可见区域图像质量、LPIPS 与多 Chunk 连续性仍无结果；当前证据及比较限制见[阶段 F 记录](FULL_3D_OUTPUT_2026-10-15.md#坐标范围与图像比较口径)。
