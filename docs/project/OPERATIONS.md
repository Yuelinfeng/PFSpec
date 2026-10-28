# GaussianGPT 操作与复现手册

## 2026-10-23 新恢复价值开发任务已结束

远端独立目录/root/autodl-tmp/gaussiangpt_recovery_value_evidence_20261023；action_collect/control两个长度段、dev_timing_run01/02、dev_online_run01全部complete，不重复启动。对应本机源码patches/recovery_value_evidence_20261023、记录docs/project/artifacts/recovery_value_evidence_20261023。正式运行器保存在artifact的source_used，后续增加可选逐轮诊断但选择模型未变。两进程加在线506.015秒<600，成本/配对88.085秒<300，总运行器594.100秒。GPU结束2MiB/0%、无计算进程，HEAD031d916/4181记录、原三权重大小与821/822服务保持；无安装或部署。复现参数见各run的plan.json，先读RESULTS.md。

## 2026-10-23 动态路径维护完成与复现位置

本机源码patches/recovery_value_20261023，远端新增/root/autodl-tmp/gaussiangpt_recovery_value_execution_20261023/source；旧/root/autodl-tmp/gaussiangpt_recovery_value_20261023/source保留。run_execution_review.py串行完成168解码、84完整请求、4成本观察；verify_online_execution.py另完成12条逐轮诊断，该时间不用于测速。完整命令在各plan.json与progress.json。只用既有gaussiangpt环境、4090、原权重，无安装/训练/生产替换。结束GPU2MiB/0%、无计算进程，HEAD031d916、4181记录、原三权重大小及821/822服务保持。CPU测试在本机运行，三维实物留远端，原始记录与报告已下载至[本机结果目录](artifacts/recovery_value_execution_20261023/RESULTS.md)；输出目录不可重复使用。

## 2026-10-23 恢复价值关键证据实验完成

[完整结果](artifacts/recovery_value_20261023/RESULTS.md)。20新来源组、64/128前缀、512上限，旧10来源开发、模型8/2留源，参数固定后两进程800解码及400完整三维请求完成。value/单备用解码0.932238 [0.928435,0.936197]，完整请求0.941442 [0.933740,0.947926]；相对概率成本和同数量Ranker区间跨1，未得到来源外强对照优势。单备用/W1来源等权解码1.013946、完整请求1.008566；完整请求累计129.007296→128.991790秒，仅省0.015506秒，累计比1.000120 [0.982643,1.014994]，不概括为稳定端到端净收益。

1200记录、960输出/语义/调用对照、600跨轮与400跨阶段对照通过；400NPZ与2400PNG有效，无空白视图，Gaussian/像素有极小差异。解码Luna独立复算通过；其最终E2E任务额度用尽无交付，主代理从原始记录直接另算四项点值/区间，与主统计完全相同，未将未交付工作计为证据。神经权重和生产入口保留。RTX4090结束2MiB/0%，无计算进程，远端HEAD031d916/4181记录、原权重大小及821/822服务保持。原始记录在本机artifacts/recovery_value_20261023、源码patches/recovery_value_20261023，远端/root/autodl-tmp/gaussiangpt_recovery_value_20261023；matrix_progress为complete，不重复启动。当前动态价值选择只作为负结果和边界，v4正文仍以固定恢复预计算为主；单模型、单卡、有限chunk与512上限不变。

## 2026-10-23 v3只读研究与重复结果接续

本次读取Downloads三文件，新增本机artifacts/v3_review_20261023审阅、功能对照与可用性材料；原下载文件未改。只读恢复SSH并确认此前四轮均complete，重新收齐原始记录后CPU复算，没有新增GPU工作、训练、安装或部署。远端主工程HEAD031d916、4181条工作区记录、三权重大小和821/822服务保持，GPU2MiB/0%、无计算进程。两个旧重复入口与远端测量副本逐字节相同。

实际维护patches/rescue_repeat_20261022/analyze_repeats.py的prompt轮换及来源偏移判读，audit_independent.py涵盖全部图创建字段；旧analysis保留，当前产物analysis_review_run02。v3后续GPU和训练建议没有执行。当前入口[v3决定](artifacts/v3_review_20261023/DECISION.md)和[四轮复核](artifacts/rescue_repeat_20261022/RESULTS_REVIEW_20261023.md)。

## 2026-10-22 论文实验已完成

贡献入口patches/spec_next_20261022/evaluate_attribution.py，远端spec_next_20261022/source_paper_next，直接调用旧source_run02和feature_tree_20261021/source_run04；paper_attribution_run01完整240次，外层769.16秒。端到端扩展原patches/w1_e2e_20261021/run_e2e.py及audit_outputs.py/analyze_e2e.py，远端w1_e2e_20261021/source_rescue_paper，rescue_paper_run01完整60次，外层236.80秒。总1005.96<1200，未运行新训练、自然16000或更多GPU补测。

本机输入w1_e2e_20261021/inputs/rescue_paper_prompts.json固定原前10来源20输入，不能按结果选择来源；全部原始token、六阶段计时、统计和CPU产物对照已下载，大Gaussian与360PNG留远端。产物对照按原行序和坐标一一对应分别报告；使用现有SciPy的cKDTree，无新依赖安装。原HEAD031d916/4181记录、权重/Ranker、821/822服务保持，GPU2MiB/0%，任务结束不再派发。完整命令保存在两组plan/progress/pipeline和artifacts/spec_next_20261022/PAPER_NEXT.md。

## 2026-10-22 旧5.056×配置已按用户要求复跑

直接执行既有/root/autodl-tmp/gaussiangpt_spec_next_20261022/source_run02/evaluate_candidates.py，依赖feature_tree_20261021/source_run04，命令只替换输出位置为同实验根reproduce_5056_run01。原target/w1/rescue三配置、30来源、64/256、seed51101、512token和容量2048保留，60份prompt与旧轮相同。180complete，备用/原Target5.090087×、备用/W1=1.015750×；外层607.18秒。已下载六份完整记录和日志，复用patches/spec_next_20261022/summarize_runs.py完成统计，主代理直接读取原始记录独立复算，旧结果不覆盖。

完整命令与范围见artifacts/spec_next_20261022/REPRODUCE_5056.md。旧入口继续保留单备用五节点实现，不用source_widths替代；本轮没有远端业务代码修改、训练或生产替换。结束GPU2MiB/0%、无计算进程，主工程HEAD031d916/4181记录、原权重/Ranker与821/822服务保持。任务已完成，不再重复派发。

## 2026-10-22 备用数量比较已完成

连接恢复后确认旧widths_main未派发，按原1200秒额度首次正式跑完360次，另补同图暖态成本8组；总GPU进程1008.297937秒。原/root/autodl-tmp/gaussiangpt_spec_next_20261022目录的widths_main、widths_numeric_run03、widths_costs均complete且本机记录已下载。主测evaluate/decoder/runtime与本机逐字节相同；probe仅新增--optimized-recovery/--cost-repeats用于成本补测。原source_run02、主工程和权重保持。结果保留单备用，无扩宽新收益，不重跑或自动扩大。

CPU复算入口patches/spec_next_20261022/analyze_widths.py和audit_widths_root.py，参数与已保存结果见artifacts/position_widths_20261022/RESULTS.md。无GPU任务，结束2MiB/0%，HEAD031d916/4181记录、821/822服务及三份权重大小不变。失败日志全部保留，旧运行中标题仅为历次记录。

## 2026-10-22 备用数量实验

结束更新：widths_numeric_run03远端passed8组/36.073秒，完整资料尚未复制。widths_main尝试命令有850秒内部停止与880秒外层限时，SSH中断后实际派发状态未知；已结束失效本机连接，未改远端生产。恢复后先读原目录和进程，禁止重复派发。没有持续执行安排。详细预算与精确接续位置见artifacts/position_widths_20261022/EXECUTION_STATUS.md。

原四次重复已全部完成并下载，读取artifacts/rescue_repeat_20261022/suite.json、analysis/REPEAT_ANALYSIS.md及independent_audit.json，不重跑。原工程/root/workspace/GaussianGPT、原source_run02与三份权重保留，GPU开始空闲；本轮扩展patches/spec_next_20261022并上传同一远端根目录source_widths，依赖feature_tree_20261021/source_run04。当前SSH复用/tmp/ggpt_position_width_20261022.sock，凭据不写入源码或资料。

新的备用数量比较不训练、不调Feature、不替换默认。用evaluate_candidates.py的rescue1/rescue2/rescue3/rescue4显式选择固定数量上限，候选不足时缩小；--max-wall-seconds在新请求组前预留时间。所有新GPU调用累计墙钟上限1200秒，外层timeout作为有限执行边界，原始未完整组保留且不进入主统计。计划与当前状态见artifacts/position_widths_20261022/PLAN.md、EXECUTION_STATUS.md。widths_smoke已结束20次，数值和完整六臂主测待完成；不要根据旧“正在执行”状态重复启动。

## 2026-10-22 新候选执行结束

本机新增patches/spec_next_20261022与docs/project/artifacts/spec_next_20261022，远端/root/autodl-tmp/gaussiangpt_spec_next_20261022。source_run02用于main_run01和confirmation_rescue_run01，source_run03用于combined_verify_run01和confirmation_run01；测量后未改执行源码。--source继续引用feature_tree_20261021/source_run04依赖。概率失败run01与成功run02都保留。

240开发、240组合确认、180纯备用确认均complete，新增候选只有纯备用显示小幅净提速，混合不替换现用W1。全部GPU进程退出，GPU2MiB/0%，HEAD031d916、4181工作区记录、Target/VQ/MLP大小与原821/822服务保留。无安装、部署或权重改动。可选source_scope标签漏传与最终probe补充判定均有独立说明；原plan/记录不改。复现入口、15文件清单和数据口径见[执行状态](artifacts/spec_next_20261022/EXECUTION_STATUS.md)，不要重复已完成矩阵。

2026-10-21 同前缀恢复成本实验全部退出：远端 `/root/autodl-tmp/gaussiangpt_recovery_model_20261021/` 的dev_run01为2808行主测、components_run01为480行诊断，均complete。GPU实际源码source_run03，`--source`引用此前feature_tree的source_run04；最终六文件交接版source_final_run04包含分析与独立复算工具。本机 `patches/recovery_model_20261021/` 与 `docs/project/artifacts/recovery_model_20261021/` 保存源码与原始记录。正式CPU统计为dev_analysis_v2.json；旧版区间不采用。恢复模型没有稳定选择收益，未运行外部测试、在线预算策略或新神经网络训练，不重复主矩阵。GPU2MiB/0%，原工程、权重与服务保留。详见[完成报告](artifacts/recovery_model_20261021/RESULTS.md)。

2026-10-21 W1端到端42次已结束，隔离目录 `/root/autodl-tmp/gaussiangpt_w1_e2e_20261021/`，计时源码 `source_run01/run_e2e.py` 调用此前已测 `gaussiangpt_feature_tree_20261021/source_run04/`。`main_run01/` 保存完整token、Gaussian NPZ、六视图和逐次记录；本机 `patches/w1_e2e_20261021/` 有计时、CPU复算与实物读取入口，结果见 [端到端报告](artifacts/w1_e2e_20261021/RESULTS.md)。42份NPZ与252张PNG读取通过，固定512组4.334555×，自然EOS组4.869038×，分母为原Target，两组独立统计。模型加载单列、首次建图计时、运行时随后复用。没有安装或训练；GPU已退出，原工程、权重和服务保留，不重复已完成矩阵。

2026-10-21 原Target参照六配置120次已完成，隔离目录`/root/autodl-tmp/gaussiangpt_original_target_next_20261021/`，实际GPU源码`source_run01/`、结果`main_run01/`，不重复启动。CPU分析入口`patches/original_target_next_20261021/analyze_original_target.py`，加`--output-dir <新目录>`可复算而保留原输出；主表分母为原Target。本轮没有新训练或依赖，生产、权重与原服务保留。新方法研究见[下一步](artifacts/original_target_next_20261021/NEXT_METHODS.md)，新树验证尚未开始。

2026-10-21 提案侧图实验使用既有远端工程/root/workspace/GaussianGPT与gaussiangpt环境，隔离目录/root/autodl-tmp/gaussiangpt_proposal_graphs_20261021；固定执行源码source_final_run01，main_run01为10开发来源、64/256、seed44101、512token、180次。main_run01已完成180次，pipeline.json为complete，不重复启动。独立CPU复算入口为本机patches/proposal_graphs_20261021/analyze_proposals.py，参数为本机结果目录；写出summary.json和audit.json。已有结果不覆盖，可用--summary-output和--audit-output指定新JSON。仅Target图与Target加提案图使用相同Target实现，EAGLE普通版及缓存复用版各自配对；首次捕获完整计时。未训练或安装依赖，生产与权重保留。

2026-10-21 旧投机方法复测：生产工程/root/workspace/GaussianGPT、原权重和数据保留；新源码副本在patches/target_methods_20261021，远端隔离目录/root/autodl-tmp/gaussiangpt_target_methods_20261021。主132次配对、220次优化比较均完成，不重复启动；固定执行源码source/prefix_smoke_run02，命令保存在各pipeline.json。LFQ BOS18次已结束但输出重放不一致，单种子同输入同历史数值复核run02也已完成，测得Target policy TV0.012678和草稿q TV0.021485，BOS路径不进入有效排名；原始大张量留远端，本机有元数据。所有本轮GPU任务已退出，最后2MiB/0%，不替换生产或继续训练。当前细节见[接续入口](artifacts/target_methods_20261021/00_READ_ME_FIRST.md)与EXECUTION_STATUS.md。使用既有gaussiangpt环境，无新增依赖。

## 2026-10-21 Ranker+MLP执行优化复现入口

本轮已完成240次独立确认、12次自然EOS及单chunk三维验证，不重复运行。完整结果、计时范围及适用限制见[本轮阅读入口](artifacts/pf2x_breakthrough_20261021/00_READ_ME_FIRST.md)。源码仅在隔离目录，生产默认入口未替换，Target/VQ/Ranker及12k MLP权重保持原样。

远端原工程`/root/workspace/GaussianGPT`，现有解释器`/root/miniconda3/envs/gaussiangpt/bin/python`。用于独立确认和自然EOS的源码快照为`/root/autodl-tmp/gaussiangpt_pf2x_breakthrough_20261021/source/independent_confirm/`，渲染脚本为`source/full3d_validity_run02/verify_full3d_output.py`。若另有复现任务，先确认GPU作业，使用全新输出目录；以下`REPRO_NEW`只表示新目录，不要覆盖已完成结果。

```bash
cd /root/workspace/GaussianGPT
/root/miniconda3/envs/gaussiangpt/bin/python \
  /root/autodl-tmp/gaussiangpt_pf2x_breakthrough_20261021/source/independent_confirm/evaluate_runtime.py \
  --project /root/workspace/GaussianGPT \
  --crops /root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/final_test_crops.pt \
  --feature-checkpoint draft_a/runs/feature_target_continue_20261014/selected.pt \
  --out /root/autodl-tmp/gaussiangpt_pf2x_breakthrough_20261021/REPRO_NEW \
  --implementation explicit_bmm --dynamic-graph --graph-cache-capacity 1024 \
  --source-scope independent --sources 30 --prefixes 64,256 \
  --seeds 1 --seed-start 39101 --budget 512
```

自然EOS使用同一入口，改为`--sources 3 --prefixes 256 --budget 16000 --graph-cache-capacity 16384`及另一新目录。真实已完成结果分别为`independent_confirm_run01/`、`natural_eos_run01/`，3D为`full3d_validity_run02/`；旧run01失败保留。长输出3GiB KV是容量16384的单池，不包含模型和图额外显存。图只用于短T1/T2/T4，长prefill走原前向；语义验证须实际出现graph_forward，不能仅凭退出码。

本机独立复算脚本无需GPU：

```bash
cd /Users/lief/Documents/ChatGPT/GaussianGPT
python3 patches/pf2x_breakthrough_20261021/runtime/audit_completed_run.py \
  docs/project/artifacts/pf2x_breakthrough_20261021/independent_confirm_run01 \
  --output /tmp/gaussiangpt_pf2x_audit_new.json
```

该输出必须不存在。当前已保存的独立复算为`independent_audit_v2.json`，自然EOS为其目录下`independent_audit.json`，两者均8/8通过。最终状态脚本仍为`python3 .codex/scripts/update_status.py`。最后GPU2MiB、0%，本轮作业结束；Jupyter和TensorBoard保留，未安装依赖。

2026-10-19 PF研究循环已按用户授权启用当前对话心跳 `gaussiangpt-2`，15分钟一次；正常进行时不发送进度通知。用户要求操控电脑上的Chrome网页端GaussianGPT对话，不使用内置浏览器；当前通过 `cua.getApp('com.google.Chrome')` 原生窗口可操作。每轮先网页方案、后Luna/max子代理实施、再打包上传结果。用户已批准本任务改用当前collaboration子代理接口，最多3个并行。详情与网页地址见 [循环记录](PF_2X_RESEARCH_LOOP_2026-10-19.md)。

2026-10-19 原MLP继续至192000：使用既有gaussiangpt环境，独立后台入口为 `/root/autodl-tmp/gaussiangpt_feature_extend192_20261019/run_extension.py`，训练完成后自动串行评价；无须持续连接SSH。读取该目录 `pipeline.json` 和 `training/progress.json` 判断实际阶段；完成后使用精确 `96000.pt`、`192000.pt`，不能以 `training/selected.pt` 替代。不要重复启动已有任务。原数据、主工程源码与现用PF权重保留，详情见 [续训安排](FEATURE_MLP_EXTEND192_2026-10-19.md)。

2026-10-19 Feature EAGLE v4 评测使用远端 `/root/workspace/GaussianGPT` 与既有 `gaussiangpt` 环境。EAGLE 路径新增 `--reuse-draft-cache`（默认关闭）及 `--profile-budget`；v4 在单场景 1,024-token 三种子测量中减少 Draft 历史输入，但中位吞吐低于 v3。命令、checkpoint、完整逐次 JSON、测试结论和网页审阅包路径见 [v4 执行记录](FEATURE_EAGLE_V4_2026-10-19.md)。不重训或替换旧 PF 权重。

2026-10-19 γ=5 实验已结束，产物在 `/root/autodl-tmp/gaussiangpt_feature_eagle_gamma5_20261019/`。原训练入口增加 `--v3-f2-weight`（默认 1）与 `--v3-rollout-fraction`（默认 0.5）；评测入口增加 `--gammas` 与 `--reference-label`。本轮使用 2 / 0.75、1,200 次更新，选择 step900。γ=5 仍慢于原 v3 γ=3，研究候选保留，主路径保持原样。设置、计数与提前 EOS 的影响见 [γ=5 定向记录](FEATURE_EAGLE_GAMMA5_2026-10-19.md)。

2026-10-19 Feature EAGLE v3 使用同一工程和环境，原训练脚本增加 `--feature-v3 diagnose/prepare/train`，原评测脚本支持显式 Feature 权重与 v2 对照。研究权重为远端 `/root/autodl-tmp/gaussiangpt_feature_eagle_v3_20261019/rollout/selected.pt`；主数据、配置、训练曲线、逐次在线记录与三维结果均留在同一目录。首批试验为 `single/`，主结果为 `single_dense/` 与 `rollout/`，诊断以 `diagnostic_before_final/` 和 `diagnostic_after/` 为准。完整命令及计时范围见 [v3 执行记录](FEATURE_EAGLE_V3_2026-10-19.md)，不要用早期诊断试算替代最终对照。

2026-10-19 的 EAGLE v2 使用远端既有 `gaussiangpt` 环境，新增 `train_pf_draft.py`、`eval_pf_draft.py`、`generate_pf_scene.py` 和 `test_pf_eagle.py`。完整命令、数据与权重位置见 [EAGLE v2 实现记录](EAGLE_PF_V2_2026-10-19.md)。本地只有源码副本；请在 `/root/workspace/GaussianGPT` 运行，使用新的输出目录，保留已有权重与实验产物。

## 1. 每次接手先做什么

```bash
cd /Users/lief/Documents/ChatGPT/GaussianGPT
python .codex/scripts/update_status.py
git status --short
```

然后查看生成的 `docs/project/PROJECT_STATUS.md`，确认 HEAD、分支、环境、checkpoint 和 profiling JSON 是否与任务描述一致。不要仅凭目录名猜测 checkpoint 是否可用。

当前本地目录是文档交接工作区，尚未放入 Autodl 源码和模型产物。因此状态快照中的源码、环境和 checkpoint 为空是预期结果；把源码 checkout 同步进来后再执行一次刷新，训练命令才会在本机有意义。

## 2. 原版链路入口

具体参数以 `conf/` 为准，常见入口如下：

```bash
# VQ-VAE 训练
python train_ae.py --config-name=<vqvae-config>

# 冻结 encoder，写出 token 流
python tokenize_dataset.py --config-name=<tokenize-config>

# GPT 训练
python train_gpt.py --config-name=<gpt-config>

# 从 token/chunk 生成并解码场景
python generate_chunks.py --config-name=<generation-config>
python complete_chunks.py --config-name=<generation-config>
python generate_scene.py --config-name=<generation-config>
python decode_scene.py --config-name=<generation-config>
```

仓库中可能存在不同版本的 Hydra 参数名；先运行 `--help` 或检查对应脚本，再提交长时间作业。训练前确认数据 split、坐标系、`max_tokens`、checkpoint 和输出目录。

## 3. S2/S3 与 profiling 操作

推荐先做短验证，再做长任务：

```bash
python scripts/selftest_position_selective_math.py
python -m py_compile model/gpt/position_selective_speculative.py \
  scripts/profile_position_selective_sd.py scripts/run_e2e_fair.py \
  scripts/run_e2e_wallclock.py
```

端到端 profiling 必须显式记录 teacher、VQ-VAE、draft checkpoint、token 数、warmup、seed、temperature、top-p、device 和计时边界。baseline 与 selective 应在同一进程条件下运行，避免 GPU lazy initialization、缓存或数据加载差异造成假 speedup。

## 4. 状态刷新脚本

`.codex/scripts/update_status.py` 是无第三方依赖的状态生成器，扫描以下动态事实：

- Git HEAD、分支、upstream、工作区变更；
- `gaussiangpt_versions.txt` 中的 Python/CUDA/PyTorch/扩展和 GPU 记录；
- `checkpoints/` 与 `profiling/` 中的 checkpoint 路径和大小；
- `profiling/e2e*.json` 中 baseline/selective 吞吐和 speedup；
- 当前未提交文件索引。

运行：

```bash
python .codex/scripts/update_status.py
```

它只更新 `docs/project/PROJECT_STATUS.md`。架构解释和实验结论应修改静态文档，不要把它们塞进脚本输出。

## 5. 产物与安全边界

- checkpoint、渲染图、Hydra 输出和 profiling 日志通常很大，优先保留路径、用途和验证结论。
- 看到 `.corrupt`、未完成训练或没有对应配置的 checkpoint 时，先标注为不可复用，不要自动加载。
- 任何重新编译 MinkowskiEngine、FlashAttention、gsplat 或 PyTorch3D 的操作都应记录编译器、CUDA 和 PyTorch 版本。
- 改动 serialization、坐标变换、cache 或 token grammar 后，必须同时做数学/语法自检和最小生成验证。


## 2026-10-21 Feature 树隔离运行已结束

本机源码 `patches/feature_tree_20261021/`；远端 `/root/autodl-tmp/gaussiangpt_feature_tree_20261021/`。首轮实际源码source_run03，最终已测source_run04，交接源码source_final_run05仅另补审计入口与方法描述。main_run01/main_run02各72次已完成，不重复启动；旧数值失败目录保留，最终数值为numeric_run06。CPU复算入口audit_tree_results.py与audit_tree_numeric.py；命令和真实输入路径见[执行状态](artifacts/feature_tree_20261021/EXECUTION_STATUS.md)。

结束HEAD031d916、工作区记录数4181，与任务开始相同；GPU2MiB/0%，原TensorBoard821与JupyterLab822保留。没有安装、训练、Git提交或生产默认替换。W2无可靠净提速，未进入W4；后续先读结果，不把局部RoPE时长比当完整解码倍率。

# 2026-10-25 网页版研究交接包

固定单备用Position方案的交接包已生成：[下载ZIP](artifacts/web_handoff_w1_rescue_20261025/GaussianGPT_W1_PositionRescue_5p056_v1_20261025.zip)。解压后先读`00_START_HERE.md`和`02_VERSION_AND_RESULTS.md`；`01_FOR_WEB_CHATGPT.md`可作为阅读请求。版本标识GGPT-W1-RESCUE-R1-20261025。

包含176份原件及6份说明、清单与验证文件，共182个文件、3907772字节。原件涵盖本地源码、首次确认、原配置复跑、四进程重复、贡献分解、完整三维请求、备用数量比较，以及后续动态选择的结果边界。所有原件复制前后字节相同；50份Python语法、90份JSON及2309行JSONL读取通过；ZIP全量读取和内容对照通过。七项主要倍率从原始记录重新计算，与已有报告一致。

本包用于研究阅读和CPU数据复算。当前本地源码含后续备用数量扩展，首次5.056076倍的精确实测入口仍在远端source_run02，版本关系已明确说明。完整主工程、权重、原始场景数据及三维大产物未随包提供。没有新增GPU、训练、依赖安装或远端操作；原推理源码和实验记录保留。本次仅整理静态资料，生成状态页保留原时间。

## 2026-10-28 AutoDL 教程复现

RTX 4090D 实例新增独立 `gaussiangpt` Conda 环境，并在 `/root/autodl-tmp/GaussianGPT` 完成权重加载、单块生成、俯视图、180 帧轨迹 GIF、4×4 场景解码及 10 张阶段帧的 GIF。默认 4×4 参数只得到 4/16 个非空列；逐列模式 `enable_bootstrap=false no_empty_columns=true` 得到 16/16 个非空列。相同 seed 的 token 序列和列长度与阶段帧改动前完全一致，稀疏现象符合原有采样路径表现；CSDN 记录没有列占用统计，无法确认作者那次结果。阶段帧路径、与原产物的对照结果、版本和权重位置见[AutoDL 复现记录](GAUSSIANGPT_AUTODL_2026-10-28.md)。

