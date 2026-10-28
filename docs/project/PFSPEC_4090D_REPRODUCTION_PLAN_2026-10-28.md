# PFSpec 主要成果复现：计划与结果

日期：2026-10-28。依据 `RANKER_MLP_REPRODUCTION_2026-10-28.md` 制定；用户已确认以 4090D 服务器为目标。源文档描述的研究工程位于另一台 5090 服务器。

## 完成结果

材料已从 5090 复制到 4090D 的 `/root/autodl-tmp/pfspec_repro_4090d_20261028/`；两份官方权重通过 `project/checkpoints` 复用当前工程的文件。代码、Ranker、step12000 MLP、Feature 输出权重及裁剪输入来自源文档列明的现有资产。独立副本中的 `tree/reference/method_adapters.py` 有五处工程绝对路径适配；原工程未修改。迁移包在本机与目标服务器的 SHA-256 一致。

依赖安装到该目录的 `deps/`，不覆盖原环境；scikit-learn 1.7.2、numba 0.68.0、llvmlite 0.50.0、joblib 1.6.0、threadpoolctl 3.7.0、cloudpickle 3.1.2 已完成导入。Feature 损失自测通过。`validate_02/` 中 Target、Position-only、W1、Rescue 在 64 和 256 前缀的普通执行与图执行对照全部通过；短运行有实际图执行调用。`pilot_01/` 为 3 来源试跑，最终状态以该目录的 `progress.json` 和 `summary.json` 为准。5090 上原有多场景任务仍在运行，本次未启动该设备的 GPU 计算。

30 来源主要比较位于 `main_01/`，240/240 次完成，RTX 4090D、FP32、TF32 关闭、temperature/top-p 均为 0.9、64/256 前缀、seed 57101、最多 512 token。来源等权解码吞吐比：W1/优化 Target 为 1.553996，95% 来源区间 [1.471569, 1.619142]；Rescue/优化 Target 为 1.571931，[1.478594, 1.646461]；Rescue/W1 为 1.011542，[0.999612, 1.020557]。W1 和 Rescue 分别在 29/30 个来源快于优化 Target；Rescue 相对 W1 的区间包含 1，当前测量未确认其稳定增益。Target/W1/Rescue 自然 EOS 分别为 3、3、3 条；其余请求到达 512 token 上限。实际输出长度不同，结果是此评价条件下的解码吞吐，不代表相同输出长度下的固定倍率。

W1 与 Rescue 在 59/60 个相同来源、前缀和种子组合中生成了逐 token 相同的输出。`23-64-57101` 在第 339 个新生成 token 首次出现差异，两边均生成到 512 token 且未 EOS；该样本的后续历史因此分开。相同种子不保证随机解码逐 token 一致；本批也没有重放该单样本到足以解释差异的概率和随机数记录，不能据此宣称逐轨迹一致。

自然 EOS 的 `01-64-57101` W1 结果已解码和渲染：187 个新 token、总正文 250 token、11,407 个 Gaussian、六个 384×384 视角；各视角像素和 Gaussian 数值有限。预览在本目录的 `artifacts/pfspec_4090d_reproduction_20261028/render_pilot_01/contact_sheet.png`，结构化结果在同目录下的 `main_01/`、`pilot_01/`、`validate_02/` 与 `render_pilot_01/`。服务器完整运行目录保留原始日志和结果。

## 复现目标与原始条件

复现 Position Ranker、Feature MLP、W1 和 Rescue 的现有推理能力，并在相同评价条件下测量它们与 Target 的速度。先使用现有模型，模型重训作为另一个独立任务。

4090D 原工程为 `/root/autodl-tmp/GaussianGPT`，当前分支保留作者源码与阶段帧改动；Target 和 VQ-VAE 官方权重在该工程的 `checkpoints/`。迁移前，Ranker、Feature MLP、完整比较入口和评价裁剪都不在 4090D 项目中，环境也缺少 `scikit-learn`、`numba` 和 `joblib`。原始材料路径、模型大小及训练方法以源文档第 1、3、4 节为准。

## 已完成的执行顺序

1. **读取源资产。** 从 5090 原工程复制比较所需 Python 源码、Ranker、step12000 MLP、Feature 输出权重及 30 来源裁剪；5090 多场景队列继续运行，没有调用它的 GPU。
2. **建立隔离运行目录。** 在 4090D 创建 `/root/autodl-tmp/pfspec_repro_4090d_20261028/`，复用原工程两份官方权重；依赖装入独立 `deps/` 目录。仅对迁移副本的 `tree/reference/method_adapters.py` 适配五处工程路径。
3. **完成加载与短流程。** Feature 损失自测通过；Target、Position-only、W1、Rescue 在 64 和 256 前缀上的普通执行与图执行共 8 组对照通过，图调用计数非零。
4. **完成试跑和主要比较。** 3 来源 24 条试跑完成；30 来源 240 条正式比较完成。详见本节结果与限制。
5. **完成代表性三维输出。** 使用一条自然 EOS 的 W1 输出完成 VQ-VAE 解码、Gaussian 保存和六视角渲染。没有进行新的训练或多 Chunk 场景运行。

## 评价口径

历史 4090 结果中，W1/优化 Target 为 1.551487，Rescue/优化 Target 为 1.575053，Rescue/W1 为 1.015189，范围与样本条件见 `artifacts/pf_comparison_20261025/RESULTS.md`。4090D 新结果单独报告，不能预先假定相同倍率。Rescue 额外验证备用位置，需同时记录备用命中、恢复调用、Target 输入行数和完整轮次耗时。Token 吞吐与完整场景耗时分别报告。

普通执行与图执行对照、P/F 合法集合、拒绝后的状态、EOS 和有限精度是正确性评价范围。不同方法即使使用同一种子也可能得到不同序列；分布保持由算法和实现共同决定，需要结合概率与状态验证，不能靠渲染相似度单独证明。

## 当前限制

未重训 Ranker 或 MLP，也未在 4090D 重做 270 份补全与 90 份 BOS 的全长图像/几何评价；本轮只对一个自然 EOS 样本执行六视角渲染。单个固定 seed 的 60 组不能用于评估训练 seed 或场景总体变化；Rescue 单样本分歧仍需专门解释后，才能主张逐轨迹保持。
