# 完整单 chunk 输出：阶段 F

2026-10-16。阶段 F 的单 chunk 正式生成、质量与耗时汇总、并排视频均已完成。阶段 E 的 30 来源结果仍支持旧 step12000 轻量 PF 作为主方法。

使用原始 `checkpoints/gpt_vfront.ckpt`、`checkpoints/vqvae_vfront.ckpt`、既有 Position ranker 及 `draft_a/runs/feature_target_continue_20261014/selected.pt`。T、P-only、轻量 PF 三模式；最终方法与旧 PF 相同，重复模式已去除。FP32、temperature/top_p=0.9、共同正文上限 16,000 token。自然 EOS 后停止；达到上限的输出保留为截断，不补 EOS。

## 输入、处理与产出

输入为准备好的真实 256-token 前缀；单场景采用裁剪清单中的第一条唯一裁剪。预定 seed 为 51701/51702/51703。30 场景、三 seed、三模式，共 270 份补全输出。BOS 从空正文开始，seed 为 51801–51830，三模式共 90 份输出。这些 BOS 轨迹单独统计，不当作真实场景数量。

每份产出包含生成 token、自然结束标记、Gaussian NPZ、六张 384×384 PNG 和分项耗时 JSON。坐标沿用 20×20×20 latent chunk 的局部坐标，物理边长 4 米；相机围绕相同的局部范围固定，所有模型使用同一组视角。该轮覆盖单 chunk 完整输出，多 chunk 场景扩展仍未实施。

从已准备好的前缀开始计时，包含 AR、缓存、输入输出验证、token 保存、VQ 解码、Gaussian 导出、六视角渲染及 PNG 保存。模型加载与预热单列；参考真实 chunk 的解码和渲染在测量之外。每个输入内轮转模式顺序，同一 GPU 顺序执行。解码时沿用已有三维试运行的尺度范围 1e-4–0.05 米，记录尺度超出比例与非有限数值。

完整输出等待时间按实际结束比较，同时报告生成长度、自然 EOS 比例、截断数量和 token 吞吐。不同模式同 seed 的随机路径可以不同；较长输出可能抵消每 token 的速度优势，因此固定窗口速度与完整等待时间分别报告。

参考图为同一裁剪完整真实 token 经 VQ-VAE 重建的局部 chunk。全图 PSNR/SSIM 仅描述与这份参考的差异，生成区域允许多种合理结果。全图指标不能单独说明生成质量等同。

## 两场景试运行结果

两场景分别为 `003ac11d-2abc-44f8-9836-4354e7dfa543` 与 `030345aa-f00e-4922-b50f-b56055702045`，均为第一条唯一裁剪、seed51701。

| 场景 | 模式 | 正文 token，含前缀 | AR 秒 | 到全部 PNG 与 Gaussian 的总秒 |
| --- | --- | ---: | ---: | ---: |
| 003ac11d | T | 1,892 | 22.6050 | 23.2611 |
| 003ac11d | P-only | 2,350 | 19.2760 | 19.9883 |
| 003ac11d | PF | 3,442 | 25.5220 | 26.6071 |
| 030345aa | T | 2,088 | 25.0443 | 25.7079 |
| 030345aa | P-only | 1,858 | 15.5760 | 16.0620 |
| 030345aa | PF | 2,266 | 17.5212 | 18.1765 |

六份均自然 EOS，Gaussian 与渲染数值均有效。六个共同相机矩阵的旋转正交性验证通过。普通 T/P 解码仅在内存中扩展返回值以取得已提交 token；未修改原解码文件。T/P 两组短运行与原入口的提案计数、接受计数、调用次数、长度、EOS 一致，重复输出逐 token 一致。轻量 PF 沿用既有返回输出的统一入口。

第一个场景 PF 输出较长，总等待时间高于 T；第二个场景 PF 总时间更短。完整来源与 seed 汇总见下文。

## 文件与运行对象

远端工程 `/root/workspace/GaussianGPT`，Python 为 `/root/miniconda3/envs/gaussiangpt/bin/python`。本地源码副本：

- `patches/pf_multiscene/full_3d_eval.py`：完整生成与三维输出。
- `patches/pf_multiscene/run_full_3d_pipeline.py`：开发试运行完成后依次执行正式补全和 BOS。

远端源码对应 `draft_a/pf_multiscene/`。远端结果根目录：`/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_f_full3d/`。

- `source_validation.json`：普通解码返回输出的相容性与相机验证。
- `development_pilot/`：6/6 成功，含全部输出及 `all_samples.png`。
- `pipeline.json`：顺序流程及分析结果状态，`complete / analysis_complete`。
- `test_completion/`：270 份正式补全输出及逐次记录。
- `bos_generation/`：90 份 BOS 输出及逐次记录。
- `analysis/`：完整统计、CSV、两条 MP4 和首帧图。

## 正式全长结果（2026-10-16）

正式真实前缀补全为 30 场景×3 seeds×3 模式，共 270/270 成功；BOS 为 30 seeds×3 模式，共 90/90 成功。全部自然 EOS，无截断或失败。所有输出均完成 Gaussian 导出与六视角 PNG 保存。运行设置为 FP32、256-token 真实前缀、temperature/top_p=0.9、正文上限 16,000；总时间从准备好的前缀计至 NPZ 与六张 PNG 写出，不含模型加载和预热。

### 真实前缀补全

| 模式 | 平均完整输出秒 | 中位数秒 | 平均新生成 token | AR token/s | 全流程 token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target | 46.07 | 51.60 | 3,050 | 67.56 | 66.23 |
| P-only | 31.37 | 32.13 | 2,987 | 98.01 | 95.26 |
| 轻量 PF | 29.52 | 31.08 | 3,087 | 107.94 | 104.59 |

速度比以同一场景的三个 seed 配对，先在场景内计算三个 seed 速度比的几何平均，再对 30 个场景等权计算；95% 区间以场景为单位进行 10,000 次百分位重采样。

| 对照 | 场景等权几何平均 | 95% 场景重采样区间 | 快于参考的场景 | 全样本总耗时比 |
| --- | ---: | ---: | ---: | ---: |
| P-only / Target | 1.467× | [1.266, 1.731] | 25/30 | 1.469× |
| 轻量 PF / Target | 1.569× | [1.318, 1.852] | 27/30 | 1.561× |
| 轻量 PF / P-only | 1.070× | [0.912, 1.252] | 18/30 | 1.063× |

两种提议模式相对 Target 的区间均高于 1。轻量 PF 相对 P-only 的区间跨过 1，当前数据支持其点估计更快，尚不足以确认稳定的增量收益。

| 模式 | Position 接受/已核验 | Position 接受/提议 | Feature 接受/已核验 | Feature 接受/提议 |
| --- | ---: | ---: | ---: | ---: |
| P-only | 91.715% | 91.715% | — | — |
| 轻量 PF | 91.720% | 55.698% | 23.256% | 21.329% |

接受率是 90 个样本的逐样本比率算术平均；每条摘要没有保留每类 token 的完整分母，因此这里不将其称为按 token 加权的总接受率。阶段 E 的 33.26% Feature 接受率来自固定窗口评价，样本路径和统计口径不同，单独保留。

平均 Gaussian 数量为 Target/P-only/轻量 PF 的 218,986/212,133/222,091；尺度上限裁切比例为 8.74%/8.84%/8.63%。alpha 覆盖率为 31.93%/31.15%/32.25%。全 chunk 对真实 VQ 重建的平均 PSNR 为 11.41/11.67/11.46 dB，SSIM 为 0.7302/0.7383/0.7321。该全图指标混合已知前缀与随机生成区域，只能描述差异，不用于判定质量相同；本轮没有 LPIPS 或可见区域掩码评价。

### 坐标范围与图像比较口径

默认 VQ-VAE 配置中每个 latent 轴有 20 个位置、下采样 stride 为 8、体素边长为 0.025 米，对应名义 4 米局部范围。解码 Gaussian 中心时，先将 latent 整数坐标乘以 stride，再加上连续 `coord_offset`，最后乘以体素边长。`coord_offset` 使用恒等映射，未裁剪到 `[0,1]` 或其他固定区间，所以 Gaussian 中心可落在名义范围之外。

远端 `test_completion/references/` 有 30 个场景的参考重建，每场景保存六个固定视角和 `metrics.json`。逐份读取参考统计后，25/30 个参考场景的 Gaussian 中心已超出 `[0,4]`；全体参考中心的轴向最小值为 `[-0.1846, -0.1550, 0.0159]` 米，最大值为 `[4.1703, 4.1810, 3.3373]` 米。生成结果的中心范围也会超过对应参考场景的包围范围。这与解码器连续偏移及不同续写结果相符，单凭中心坐标越界不能判定输出损坏或质量不佳。

真实补全输出与参考图的素材已经配齐：每个场景有六张参考 PNG；270 个输出样本各有六张 PNG、Gaussian NPZ 和 `tokens.pt`。另对 90 个场景×seed 组逐一读取三个模式的 `tokens.pt`，它们声明的前缀长度均为 256，三个模式的前 256 个 token 完全相同。这只确认了条件输入一致，不代表生成区域的三维质量。

远端 `gaussiangpt` 环境可导入 `lpips` 与 `torchmetrics`。目前没有 LPIPS 数值，也没有已保存的可见区域遮罩。直接对全视图计算 LPIPS 会同时计入固定前缀与随机续写，不能单独解释为生成质量；若后续需要该指标，应先定义可复现的可见区域遮罩和比较方式。BOS 输出没有对应场景参考图。多 chunk 连续性与接缝仍无数据，本轮不据单 chunk 结果外推。

### BOS 与视频

90 份 BOS 输出全部自然 EOS。平均完整输出时间为 Target/P-only/轻量 PF 的 71.00/48.55/40.33 秒。按 30 个 seed 配对，轻量 PF 相对 Target 为 1.791× [1.588, 2.023]，相对 P-only 为 1.220× [1.094, 1.365]。BOS 结果是无条件随机轨迹，不替代真实场景结论。

主视频覆盖全部 30 个测试场景的预定首个 seed 51701；BOS 视频覆盖全部 30 个 seed。每个画面按模式分行、六个共同固定视角分列，显示实际测得的端到端秒数。每个场景画面停留两秒以便查看，视频播放时长不等于模型生成所花的累计时间。

- [真实场景六视角并排视频](artifacts/stage_f_full3d/analysis/test_completion_30scenes_seed51701_6views.mp4)
- [BOS 六视角并排视频](artifacts/stage_f_full3d/analysis/bos_30trajectories_6views.mp4)
- [完整汇总 JSON](artifacts/stage_f_full3d/analysis/summary.json)、[逐样本 CSV](artifacts/stage_f_full3d/analysis/test_samples.csv)、[分场景 CSV](artifacts/stage_f_full3d/analysis/test_scenes.csv)

本地可复现汇总脚本为 `patches/pf_multiscene/analyze_full_3d.py`，远端副本位于 `/root/workspace/GaussianGPT/draft_a/pf_multiscene/`。已知 token 前缀的一致性已读取现有样本确认；已知区域图像质量、LPIPS 和多 chunk 接缝/连续性仍未测量，完整房间扩展也未实施。
