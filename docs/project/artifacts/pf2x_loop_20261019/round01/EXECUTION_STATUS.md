# PF分离2×研究：首轮执行状态

更新：2026-10-20，台北时间。

当前有效计划为 [WEB_PLAN_R1_ORIGINAL.md](WEB_PLAN_R1_ORIGINAL.md)，31,519字节，已由Chrome下载保存。R1取代旧版惰性W2完整实现及36/180次在线比较。现在执行零GPU概率验证、同长度调度模拟与有限诊断脚本准备；真实GPU诊断须等原训练与486次评价全部结束。

网页对话：<https://chatgpt.com/g/g-p-6ab0f2354ef88191af84c1dd97a89d90-gaussiangpt/c/6ad1b79f-7bc4-83e8-a9d1-d41896fb2aed>

## 原训练状态

03:43原流程为 `complete / analysis_complete`，精确192000步及全部486次评价完成。真实结果已镜像 `../../feature_extend192_20261019/final_raw/`（24份），主代理逐次复算通过。96k/192k在线F接受率为35.7257%/35.2191%，相对12k来源速度为0.99894× [0.98029,1.01723] / 0.99895× [0.98063,1.01620]。没有可靠额外提速，建议保留step12000作唯一开发参照，结果包已于03:49通过Chrome上传发送，网页正在解压审阅，完整确认待接收。

全部五节点共同达到256上限的58组仍约0.999倍速度。EOS、来源区间、实际token和本轮缺少Target-only对照等限制均见 `../../feature_extend192_20261019/FINAL_RESULTS.md` 与 `final_validation.json`。不混用历史Target时间宣称新加速比。原源码和权重未改动；尚未开展GPU诊断。

本次复用三名Luna：资料代理只读复算完整结果；成本代理仅新增完整NumPy输入测试脚本；调度代理只读审阅模型接口。以上与此前已完成的数学、调度验证任务不同，不重复旧任务。原五节点结果包上传网页并确认参照后，才执行总计最多1 GPU小时诊断。

## 已确认的R1范围

1. 现在完成CPU概率证明、同预算调度轨迹、源码位置对应及诊断脚本准备，新增训练0。
2. 原训练与486次评价全部结束后，汇总精确12000/24000/48000/96000/192000、长度/EOS、计时与来源区间，先上传网页确认唯一开发MLP参照。
3. 之后仅在10个已有开发来源收集最多480条p/q条件、最多96条第二分支P2条件，并测量Target两/四Token、KV恢复及相关成本。总计最多1 GPU小时。
4. P2使用正确P1缓存与被接受Feature条件；不能沿用旧Feature分支的logits。二次接受子集按 `min(r1,q2)/β` 加权，其与二次拒绝分支的整体混合才服从r1。网页已确认，见 [补充记录](WEB_CONFIRMATION_R1_ADDENDUM.md)。
5. 整合结果上传同一网页对话，再选择停止分支转空间条件MLP、W1短段调度或真正并行分支设计。当前不实现W2生产解码器、树形Attention，不做新训练，不使用确认来源。

原Target、GPU、FP32、temperature=0.9、top-p=0.9及相同生成预算保持一致。完整计时包括缓存、前缀、提案、验证、恢复、采样；加载与预热单列。正式2×需要来源等权速度与95%区间下界均超过2，以及未参与方法选择的来源支持。自然EOS、完整Token输出与三维有效性分别报告。

## 三名Luna子代理的实际产物

固定 `gpt-6-luna/max`，最多三名并行；接口例外已经用户授权，无需再次询问。任务均限本机指定文件，不访问服务器或网页，不使用GPU。

- `/root/pf2x_evidence_review`：已完成 `verify_multicandidate_cpu.py`、`results/probability_cpu.json`、[PROBABILITY_PROOF.md](PROBABILITY_PROOF.md)。主代理已阅读并独立重跑当前脚本：9组配置覆盖2/3/4/8类别，2207组p/q、60463组三元组合，精确误差0；4096维binary64最大最终概率误差9.54e-18。另有100000条合成辅助抽样。结果不代表CUDA/FP32、真实KV或性能。
- `/root/pf2x_schedule_review`：已完成 `schedule_trace.py`、`SOURCE_MAP.md`、`SCHEDULE_TRACES.md`、`results/schedule_traces.jsonl`。主代理阅读完整脚本并复跑44条轨迹通过，另核对每条轨迹候选标签唯一；36条非EOS满预算、8条EOS、20条尾部K1。6-token构造路径中，W1-K2为两次Target调用、输入长度[4,4]，惰性W2为三次、[4,2,2]。这是控制流与符号缓存记账，不验证实际token-ID、Ranker数值排序、RoPE或耗时。
- `/root/pf2x_cost_probe_prepare`：三个诊断入口与 `PROBE_PREPARATION.md` 的首版已完成。保存逐次W1输出/计数、未来同种子短回放，并保留原Target数值和布局要求。只读审阅提出的无支持q2权重与极小正质量问题已处理：合法触发率使用 `P(P1接受)×E[R1×L]`，全体首拒率另存；零R1行合法质量为0；首拒按δ加权，正质量不再以1e-15截断。无支持事件比例按R1统计，合法缺样单元保留缺失，乐观节省点估计不视为严格上界或停止依据。
- 主代理已阅读上述改动，并实际运行三个CLI帮助、三脚本语法解析及262组浮点枚举自测，均通过，最大误差约1.39e-16；真实运行记录在 `results/preparation_cpu_validation.json`。本机Python3.12和系统Python3.9均无NumPy，未安装依赖或仿造接口；完整合成NPZ输入的 `analyze()` 尚未执行，留待既有远端环境的CPU验证。
- 现已复用同一成本代理补充R1内的计时准备：仅编辑 `measure_target_shapes.py`、`analyze_cost_probe.py`、`PROBE_PREPARATION.md`，默认4个已有条件、各形状预热1次、8个交替顺序配对。统一从处理完P1的KV位置开始并逐次恢复，固定输入token，不新增q2或独立分支样本，额外调用和恢复单列。该项已完成并由主代理阅读及复跑验收：默认4条件共8次预热、32个配对、72次新增前向、76次截短；每个上下文明确计为预热2次、配对16次。复跑发现的回调记录分类问题已在同一函数路径处理并验证。原单次形状统计不接收重复行，恢复耗时单列，调用计数含预填及全部实际诊断前向。记录在 `results/timing_preparation_cpu_validation.json`；仍无GPU运行。`/root/pf2x_schedule_review` 已完成该段的只读调用顺序设计，无文件修改。


脚本统一位于本机 `patches/pf2x_round01/`，文档与结果位于本目录。当前远端源码和生产解码器均无修改，没有新的性能结果。为核对真实Ranker依赖，主代理另从远端只读补取 `bench_fast_hgb.py`、`eval_lean_ranker_cpu.py`，当前 `source_current/` 共13份完整源码；原上传包保留其当时11份不覆盖。

## 已上传与保存的材料

- 初始交流包在 `../round00/GaussianGPT_PF2X_Round00_2026-10-19.zip`：149文件、2274007字节，23:59通过Chrome上传，网页确认读取。仅含当时训练快照，不能代替最终结果。
- 原方案存档 [WEB_PLAN_ORIGINAL.md](WEB_PLAN_ORIGINAL.md)：20851字节。惰性W2的短Target调用可能拆分W1下一轮工作；只读源码审阅见 [REVIEW_BEFORE_IMPLEMENTATION.md](REVIEW_BEFORE_IMPLEMENTATION.md)。网页已接受该疑问并撤回原实现矩阵。
- `GaussianGPT_PF2X_Round01_Preimplementation_Review.zip`：14文件、73188字节，约00:32上传；内含11个现用源码的只读副本与实施前审阅。`bench_gaussiangpt_pf_block2_cow.py` 实际位于远端工程根目录。
- `GaussianGPT_PF2X_Round01_CPU_Probability.zip`：5文件、17183字节，约00:53上传。该包保留初版225组p/q、2125组三元组合，不覆盖。网页已确认解压并表示独立复跑主要数值一致；这属于网页报告，本机当前脚本另经Codex重跑。
- 当前CPU补充与后续调度结果尚未重新打包上传；待整合后与有限诊断结果共同交回网页，避免重复发送碎片材料。

## 下一动作

既有本地数学与调度CPU验收已完成；不重复旧验证。当前先完成结果复算、上传五节点最终包并确认开发权重，随后使用既有远端环境完成NumPy输入分析与模型/CUDA验证，再运行R1有限诊断。每轮必须保留实际差异、设置、命令、日志、逐次结果、来源汇总与限制。

## 03:49实际提交与CPU进展

最终五节点结果包61文件、702978字节已上传原网页并发送，网页回复正在解压阅读；见 `WEB_SUBMISSION_192K_FINAL.md`，勿重复上传。主代理已在既有远端Python3.10.21/NumPy2.2.6且CUDA不可见环境执行完整NPZ合成输入验证，通过；首次测试输入KV时间字段与真实格式不一致，仅调整该合成输入并复跑通过，见 `results/npz_cpu_validation.json`。远端隔离脚本目录为 `/root/autodl-tmp/gaussiangpt_pf2x_round01_20261020/`。GPU仍未使用。

资料Luna完成 `../../feature_extend192_20261019/FINAL_EVIDENCE_REVIEW.md`，独立复算点估计一致；以来源计数组合精确加权枚举得到的区间与原10000次抽样端点略有差别，分别保留。调度Luna确认接口在完整工程路径下静态相符，实际GPU重放尚待进行；主代理已确认真实Target/VQ/head/Ranker文件存在，主工程HEAD仍031d916。成本Luna现只新增共同3600秒运行编排，避免每个子命令单独获得一小时。


## 2026-10-20 03:56 网页完整批准

原文已通过Chrome原生下载并保存为 `WEB_APPROVAL_192K_ORIGINAL.md`（9518字节）。唯一开发MLP为精确step12000；64新token、60次W1采集、最多480条p/q；narrow/wide各最多48个独立分支，各2个已有配对上下文，每个8组。所有阶段共用3600秒，新增训练0，生产解码器不改。网页要求预算到期时保留已完成采集；当前由同一成本Luna补充有限保存/退出处理，尚未启动GPU。

主代理核对采集入口时发现W1逐次JSONL调用把单个对象传给列表接口；该问题已交成本Luna处理，并补充纯CPU验证。实际运行配置为 `run01_config.json`。完成接口与同seed验证后才继续采集；完成有限诊断后向原网页提交结果，不自动实施新方法。


## 2026-10-20 04:08 首次真实GPU诊断

保存与完整W1对象输出已由主代理验证；远端真实NumPy快照先发现漏传np参数，补充后两次快照均通过，reservoir不被修改。见 `results/snapshot_cpu_validation.json`。首个GPU任务已启动，共用预算04:08:01.948至05:08:01.948（本地UTC+8）；实际远端输出为 `/root/autodl-tmp/gaussiangpt_pf2x_round01_20261020/run01`。先加载与同seed短回放，随后最多480pq、两模式各48分支及各2配对条件，最后CPU分析。没有重复原训练评价。后续若有失败，保留该目录，沿用同一绝对截止时间，不重新获取一小时。


## 2026-10-20 R1有限诊断完成

已依Chrome原网页完整批准沿用step12000，新增训练0。04:08至04:11完成60次W1、480条真实p/q、narrow/wide各48个真实P2分支和两模式各2个配对上下文组，总墙钟193.622秒。两种模式β为15.2197%/14.3099%，有利事件每轮点估计8.5644%/5.7112%；来源区间未涵盖上下文或分支抽样误差。2/4-token配对耗时接近，配对仅2个唯一上下文、1个来源；没有新完整解码或同轮Target-only，不能宣称2×。原始结果位于首轮 `cost_probe_raw/run01/`，审阅包资料在 `cost_probe_review_20261020/`。下一步完成Luna独立复算与上传原网页，取得新一轮方案后再开展实验；不要重复采集或消耗剩余额度追加测量。


## 2026-10-20约04:22 真实诊断已提交网页

两名Luna独立概率/计时复算通过，新增 `INDEPENDENT_REVIEW.md` 和 `TIMING_REVIEW.md`；主代理验收后压缩64文件、11276116字节，通过原Chrome上传发送。原生AX已确认附件及请求发送，网页正在生成下一轮方案，尚未取得完整新计划。不要重复发送、重跑采集或自动追加GPU。提交见 `WEB_SUBMISSION_COST_PROBE.md`。
