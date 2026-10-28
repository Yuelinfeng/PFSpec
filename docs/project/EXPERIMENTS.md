# GaussianGPT 实验记录与证据

## 2026-10-27 已有全长证据进入论文主稿

本轮没有新增 GPU 测量。阶段 F 已完成的 270 次真实前缀补全和 90 次 BOS 请求重新纳入当前论文：360 行均自然 EOS、无截断，基础轻量 P/F 的完整请求时间比分别为 1.569348978、1.790705485。两份 CSV 的全部状态、分时求和及相对 Target 的点值与既有汇总相符，区间沿用原报告，未重新抽样。完整结果见 [阶段 F 说明](FULL_3D_OUTPUT_2026-10-15.md)、[补全记录](artifacts/stage_f_full3d/analysis/test_samples.csv)和[BOS 记录](artifacts/stage_f_full3d/analysis/bos_samples.csv)。

该批沿用 step12000 与基础短 P/F 路径，使用较早执行后端，未加入 Rescue；据此调整此前“缺少全长三维证据”的概述。当前完整 PFSpec 的共享后端及 Rescue 全长增量仍需单独评价。正文 5.7、表 5、附录 C.7 已包含自然 EOS 结果与已有六视角图；方法 3.2 和附录 C.6 补齐 Ranker、Feature 网络及训练细节。运行源码、权重、原始结果和场景图像均保持。

## 2026-10-26 机制分析与实验口径对应

本次没有新增实验。 引言采用用户修订的两类计算需求结构，约1.8%标注为四轮重复；1181→793来自既有单备用对照，新增文献不计作已测方法。Introduction与Related Work已加入当前正文，动机使用既有W1/W2与阶段E结果，文献比较不扩大已测方法范围。T_init明确包含BOS启动生成和其他循环外成本，状态s的条件概率不将未来随机结果视为已知。当前[正文](artifacts/paper_v4_20261023/PAPER_MAIN.md)第2节为正式Method，第3节以含EOS的式A、B、C刻画推进、两类Position恢复与备用复用，并分别陈述分布保持和耦合轨迹保持。恢复计数不代替时间，局部成本关系不视为独立预测定理。

有限请求采用 \(N=N_{\rm boot}+\sum N_r\)、\(v=G^{request}/\bar C\)，区分完整输出摊入轮次与循环本身的推进；rounds只统计启动后的循环。原Target和优化Target的既有rounds=0不用于N/R，直接使用N/T。W1/W2最终18对无BOS启动的输入仍支持1.043680827/1.086795177=0.960328910；完整轮次成本包含初始化，不能解释成纯循环体耗时。

RQ1使用W1/W2、阶段E PF-D8/轻量MLP和已完成的共同后端比较；RQ2使用Position-only、Ranker、W1与已训练输入消融；RQ3使用W1/Rescue配对及同轮3.240487×1.557471×1.016788≈5.131692。不同批次、速度分母和阶段计时分别报告。Feature-only未完成。摘要、原始记录、运行源码、权重和环境未改。

## 2026-10-25 同执行条件的方法与结构比较

[完整结果](artifacts/pf_comparison_20261025/RESULTS.md)。30既有来源、64/256前缀、seed57101、512token，8方法480次全部完成；PFSpec/优化Target为1.575053 [1.487180,1.644946]，无备用W1为1.551487，EAGLE v3适配版0.966838，Medusa低秩0.780244，Hydra条件0.719455。通用适配版保留各自采样流程，使用共同Target后端。Rescue/W1为1.015189 [1.004406,1.022387]，60对输出与接受相同，少357次恢复且等于命中，净输入增加3658。共同满512子集55对另列，不替代主值。

同入口另完成10开发来源、256token、18配置360次，Feature结构均复用同一W1循环。B0/B1同训练轨迹和更新预算、三种训练seed分列，参数量未严格匹配；B1-E/B0为1.064/1.046/1.016，仅第二种子的区间不含1。无Target状态/同预算完整版0.893863 [0.832795,0.962154]，去候选Position与历史Feature未显示明确独立速度收益。所有权重复用，没有新训练；初次B1-E图捕获错误通过提前准备同一固定遮罩处理，原/新前向输出相同，随后48组普通/图执行对照通过。源码`patches/pf_comparison_20261025`，远端同名独立根；GPU任务结束，原工程、权重与服务保留。当前适配版本速度不推广为官方原方法的普遍排名，摘要保持定稿。

## 2026-10-23 恢复价值：新估计器与真实动作成本

原八训练/两容量选择来源拟合64棵深度2概率树，备用来源等权MSE .039218→.035144；操作计时33状态、515动作、1545重复，含真实提交、恢复与EOS路径。十旧来源64/256、seed60201、512上限，七方式两进程280记录通过输出/语义/调用核验；新价值/固定1=.993432，对概率规则1.000437区间跨1，对同数量Ranker .998760。相同1474备用节点，新价值命中236、Ranker254。额外12在线记录2166决策与158实际备用ID一致。属于旧来源开发评价，没有来源外或新增完整三维收益；完整记录见[结果](artifacts/recovery_value_evidence_20261023/RESULTS.md)。

## 2026-10-23 动态执行路径维护与同入口对照

[完整结果](artifacts/recovery_value_execution_20261023/RESULTS.md)。三开发来源、64/256、seed58101/58102、512上限，原W1/原单备用/同入口固定0/同入口固定1/旧动态/轻量路径配原选择器/轻量路径配快速选择器，168解码、84完整请求，两独立进程。新/旧动态来源等权速度为1.058868/1.046277；同入口动态/固定1为0.990502/0.992057。代码额外开销确实影响归因，当前策略仍未显示优势。252正式记录、12在线诊断、4成本观察通过；无模型训练或参数选择。旧20来源记录保留，不能把三开发来源结果泛化到来源外。

## 2026-10-23 恢复价值关键证据实验完成

[完整结果](artifacts/recovery_value_20261023/RESULTS.md)。20新来源组、64/128前缀、512上限，旧10来源开发、模型8/2留源，参数固定后两进程800解码及400完整三维请求完成。value/单备用解码0.932238 [0.928435,0.936197]，完整请求0.941442 [0.933740,0.947926]；相对概率成本和同数量Ranker区间跨1，未得到来源外强对照优势。单备用/W1来源等权解码1.013946、完整请求1.008566；完整请求累计129.007296→128.991790秒，仅省0.015506秒，累计比1.000120 [0.982643,1.014994]，不概括为稳定端到端净收益。

1200记录、960输出/语义/调用对照、600跨轮与400跨阶段对照通过；400NPZ与2400PNG有效，无空白视图，Gaussian/像素有极小差异。解码Luna独立复算通过；其最终E2E任务额度用尽无交付，主代理从原始记录直接另算四项点值/区间，与主统计完全相同，未将未交付工作计为证据。神经权重和生产入口保留。RTX4090结束2MiB/0%，无计算进程，远端HEAD031d916/4181记录、原权重大小及821/822服务保持。原始记录在本机artifacts/recovery_value_20261023、源码patches/recovery_value_20261023，远端/root/autodl-tmp/gaussiangpt_recovery_value_20261023；matrix_progress为complete，不重复启动。当前动态价值选择只作为负结果和边界，v4正文仍以固定恢复预计算为主；单模型、单卡、有限chunk与512上限不变。

## 2026-10-23 四轮重复的独立复核与v3材料审阅

既有四进程960解码/480配对/120唯一case复算一致：单备用/W1=1.0180919，95%来源区间约[1.0148,1.0213]；四进程1.018037/1.018846/1.019219/1.016268，28/30来源总体更快。输出与接受计数480/480相同、恢复8400→5492，差额2908逐对等于命中。首图子集0.875832，未剔除。今天没有重跑GPU，新分析按真实prompt轮换与来源顺序判定，差异0，原243条提示保留。详见[最终复核](artifacts/rescue_repeat_20261022/RESULTS_REVIEW_20261023.md)。

v3算术反例成立，在线估值和新颖性仍需实际证据；k1..4与端到端的旧结果已有，不能按文稿的未完成列表重复启动。逐轮概率/候选/纠正事件记录仍缺，39固定前缀仅供离线探针。详见[v3决定](artifacts/v3_review_20261023/DECISION.md)。

## 2026-10-22 论文贡献分解与单备用端到端完成

同轮原Target/优化Target/W1/单备用30来源×64/256×seed51101×512token、240次60配对完成。来源等权吞吐分解3.240487×1.557471×1.016788=5.131692，单备用/优化Target1.583618。原/优化Target60输出计数相同，W1/备用60输出相同；旧三臂180输出与全部counters相同。单备用恢复1181→793、388命中省388调用。正式算法与三个性质已写入paper_design_20261021/THEORY_NOTES.md顶部，标准抽样与局部成本条件不主张全局最优。

固定前10来源20输入三配置端到端60次完成：全量备用/原Target4.307862 [3.867624,4.660647]，共同满512的19配对4.185458；备用/W1全量1.001848 [.962270,1.024833]，未显示稳定端到端新增收益。原Target10240token，W1/备用9837token且各1EOS，实际长度差异保留。60份NPZ/360PNG有效，20/20token相同；Gaussian行序不同，坐标一一对应最大距离5.977205e-7、最大字段相对L2 3.048302e-7，PNG变动通道最高0.00881619%，非逐位相同。两GPU进程累计1005.96秒<1200，GPU释放，原权重/服务/工程保持，无新训练。完整资料见[本轮报告](artifacts/spec_next_20261022/PAPER_NEXT.md)。

## 2026-10-22 原5.056×配置复跑完成

用户明确要求复现旧单备用配置，直接用既有spec_next/source_run02与feature_tree/source_run04，原30来源×64/256×seed51101×512token×target/w1/rescue共180次完成。60份prompt逐项与旧轮相同，FP32/TF32关闭、temperature/top-p=.9/.9、容量2048、16token独立预热和方法循环顺序保留，首图仍计入。

来源等权全量W1/原Target=5.011159、备用/原Target=5.090087 [4.593280,5.421318]、备用/W1=1.015750 [1.001101,1.024756]；旧备用/原Target5.056076保留，相对差+0.672674%。恢复1181→793、命中388、备用节点3875/拒绝858，接受及输出总量与旧轮相同。完整解码矩阵605.295049秒，外层607.18秒，GPU退出且原权重/服务/主工程不变。详见[同配置复跑](artifacts/spec_next_20261022/REPRODUCE_5056.md)及其原始记录、两套分析与独立对照。无新训练和三维端到端结果，不将子集倍率替代全量主值。

## 2026-10-22 备用数量比较结束，无扩宽净收益

30来源×64/256×seed53101×512token上限，原Target/W1/备用1..4六臂360次complete。来源等权对原Target：W1=4.854534，备用1/2/3/4=4.905899/4.869986/4.834639/4.813895；备用2/3/4对单备用=.992680/.985475/.981246，均无可靠改善。无新图54配对/28来源，后三者/单备用=.995905/.998315/.999621，不因剔除冷图得到新收益。满512的54配对单备用/原Target=5.176860仅为等数量子集，全部主值保持4.905899，旧5.056076属于另轮seed51101。

四备用各60/60输出、EOS、长度和接受语义计数与W1相同；恢复1051→709/656/647/647；第二/第三/第四多省53/9/0调用，增加1441/200/3节点。4052适用轮中只有200轮有第三备用，3轮有第四；备用图4/8/12/14个，首次成本进入测量。单备用本轮全量/W1=1.010581区间跨1，满512和暖态各1.016330/1.018024；此前四进程重复证据保留。

数值与成本均8条passed，构造合法位置仅用于路径覆盖。成本补测同DynamicTargetGraph有300次T1图前向，恢复暖态3.45–3.67ms，不用旧非图174–187ms推导在线速度。累计GPU进程1008.297937秒<1200，全部GPU退出；未改生产/权重/Feature机制或新训练。完整原始资料及独立复算见artifacts/position_widths_20261022/RESULTS.md，不重复矩阵。

## 2026-10-22 单备用重复收益已确认，备用数量实验进行中

结束更新：1..4备用扩展已通过20次短解码对照与8组分支GPU数值对照；数值run03远端完成，完整记录下载因网络故障未完成。原Target/W1/备用1..4六臂主测尝试后SSH失效，是否派发及完成状态均未取得证据，尚无扩宽新倍率。恢复后先读取原结果和进程，不重复启动。详见artifacts/position_widths_20261022/EXECUTION_STATUS.md。

原四次独立进程串行重复全部完成，960条解码、480个配对、120唯一case，覆盖30既有来源、64/256前缀、两个新种子。单备用/W1来源等权倍率1.0180919，独立复算95%区间约[1.0148,1.0213]；四进程倍率1.01627..1.01922。输出与语义计数480/480一致，跨重复各臂120/120 case输出一致，恢复差额2908=命中2908。首图子集备用/W1约.8758，冷请求仍可能变慢。原始记录和分析位于artifacts/rescue_repeat_20261022，未重跑、没有新的原Target或端到端倍率。

本轮按用户方案比较原Target/W1/备用1..4，无新训练、Feature扩宽或动态策略，新增GPU累计墙钟不超过1200秒。复用原源码和工程，最多5/6/7/8节点；候选不足时数量缩小，实际形状分别建图。39个已有教师前缀CPU累计质量分析提示第二备用增加质量，第三/第四在该有限资料中没有增加；短64token对照20次输出和接受计数相同，恢复差额等于命中。数值、成本及512token正式结果尚待完成，详见artifacts/position_widths_20261022。

## 2026-10-22 备用Position有小幅提速，Feature混合尚无可靠净收益

十来源开发240次：备用/W1=1.018645 [1.005513,1.028114]，60/60输出相同，恢复1314→834。另三十来源组合240次：Feature接受/验证33.2333%→34.5925%，来源等权差+0.9497pp、95%[-1.6660,+3.5001]pp；快速混合/W1=.990377，组合/W1=1.004189，均未显示可靠净收益。组合/快速混合=1.013946，输出60/60相同、恢复1134→754。

读出后追加同三十来源seed51101、64/256、512上限的原Target/W1/纯备用三臂180次：备用/原Target=5.056076 [4.553989,5.386374]，备用/W1=1.017481 [1.002716,1.026950]。29/30来源更快，60/60输出与alpha计数相同、388次备用命中省388恢复。共同满512且无新图55配对29来源为1.023267 [1.019071,1.027431]。这些区间为主代理seed104212；另有Luna/max seed402101独立复算。首图计入，权重加载/VQ/渲染不计入。全部完成，不把5.056称作纯算法倍率或新端到端结果。详见[结果与限制](artifacts/spec_next_20261022/RESULTS.md)。

2026-10-21 用户将主比较改为原Target：已另做同轮六配置、10开发来源、64/256前缀、seed44101、512token上限，共120次。来源等权MLP/EAGLEv3/EAGLEv4复用/Medusa对原Target为4.6365/2.9861/2.8876/2.4537倍，包含执行优化与投机解码合计收益；Target执行优化单独为3.1527倍。原/优化Target20/20输出计数相同，四候选新旧主轮80/80相同，独立CPU复算10项通过。首项研究建议为保留Ranker的Feature W2分支真实验证成本，再考虑W4、UniVer及成本预算；DFlash需新模型，尚未训练。普通完整token后缀在已有生成后缀中零重复，优先级降低。详见[本轮结果](artifacts/original_target_next_20261021/RESULTS.md)与[方法筛选](artifacts/original_target_next_20261021/NEXT_METHODS.md)。原Target口径已超过2倍；下一项需要证明比现有MLP进一步提速。

2026-10-21 三种提案侧专用图实测完成：10开发来源、64/256前缀、seed44101、512token、180次；MLP/EAGLE v3/EAGLE v4 KV复用/Medusa LowRank的提案图新增倍率为1.034724/1.157529/1.175119/1.020445，相对同优化Target由1.439277/0.830013/0.788927/0.761582变为1.489254/0.960764/0.927083/0.777153。2倍未达，MLP仍最快。四候选各20/20输出和计数一致，Target调用、图形状与缓存事件一致，独立复算9/9通过。首次图创建进入同步外层墙钟；三臂均无新增图的18对、9来源子集中MLP为1.516174，EAGLE v3为0.974637且区间跨1，v4与Medusa为0.938478/0.781818。EAGLE仅T1递推与Feature读出头图化，T>1仍普通执行。无训练、生产默认和权重保留，GPU已释放。详见[完整结果](artifacts/proposal_graphs_20261021/RESULTS.md)。

2026-10-21 同优化Target下旧方法再评价：11配置、10开发来源、64/256真实前缀、seed43101、512token、220次主比较完成。来源等权MLP/同优化Target=1.450584× [1.370351,1.523298]，Ranker-only1.371252×，EAGLE v3γ3 0.817906×、v3γ5 0.670129×、v4复用0.779104×、v2Joint0.548611×，Medusa0.725175×、Hydra0.681434×、D8-PF0.899345×、D8Joint0.491742×。没有方法超过MLP，2×未达。另3来源132次原/优化配对11法各6/6输出计数一致；主测220次独立审计8项通过，其中单执行的原/优化项不适用。LFQ BOS18次输出重放未一致；同历史213次图前向复核最大Target policy TV0.012678、草稿q TV0.021485，概率支持集有变化，不报告LFQ有效速度。完整结果见[本轮结果](artifacts/target_methods_20261021/00_READ_ME_FIRST.md)。

2026-10-20 用户要求的新12k/48k Ranker+MLP实测完成：10开发来源、64/256前缀、3新种子、512预算、120次AB/BA对照，FP32原解码完整外层计时。48k/12k来源等权速度0.994623× [0.980059,1.007992]；F接受率35.230457%/35.252955%。共同满512的58配对速度0.995938×。本次没有复现48k接受率或速度优势，保留12k参照。原生产与权重不变。详细依据见[实际比较结果](artifacts/feature_ckpt_compare_20261020/RESULTS.md)。

2026-10-20 PF2X Round03止于零GPU筛选：已有480条p/q按来源6/2/2分开；308参数空间探针留出重合度从36.878741%到36.987119%，仅+0.108377个百分点，低于简单邻居混合+1.319351个百分点与预设3个百分点。未进入完整网络训练或在线评价，GPU=0。两条诊断来源此前已参与网页探索，有限分组探针的负结果不排除全部空间条件方法。首次JSON保存失败与同配置重放均保留，概率/参数/逐行结果完全一致。真实资料见[Round03结果](artifacts/pf2x_loop_20261019/round03/00_READ_ME_FIRST.md)。

2026-10-20 PF2X Round02分组输出微测完成：736次前向、728次恢复、288配对，688数值比较通过；四token配对中位仅降0.4875%，全样本均值无改善，一/二token略慢，按计划停止，未进入完整解码。真实GPU墙钟65.529621秒。原训练至192000及486次五节点评价全部完成，没有接受率突然跃升或可靠额外速度，主方法继续沿用step12000。后文“新节点尚无结果”是先前训练发起记录，以最终[192k结果](artifacts/feature_extend192_20261019/FINAL_RESULTS.md)为准。

2026-10-19 用户再次要求原MLP继续至96000、192000：从精确48000追加144,000次更新，复用原400/50条轨迹与CE+2TV；新建一次AdamW，学习率1e-6→1e-7，96000处不重启，训练seed24690。每1200步保存完整验证，训练后自动进行186次中间曲线与300次五节点同轮对照；新节点尚无结果。独立目录为 `/root/autodl-tmp/gaussiangpt_feature_extend192_20261019/`，设置与实际状态见 [新续训记录](FEATURE_MLP_EXTEND192_2026-10-19.md)。原工程源码及旧权重保留，前一轮停止建议以用户最新要求为准。

2026-10-19 原 Feature MLP 长续训完成：从 step12000 追加 36,000 次更新，精确 step24000/48000 均保存，186 次曲线及 180 次三节点在线对照完整。十个开发来源上，12000/24000/48000 的 Feature 接受率为 34.0312%/33.7806%/35.6000%，吞吐为 116.5768/116.0411/116.7613 token/s。48000/12000 来源等权速度比为 1.0021× [0.9850,1.0189]，接受率差 +1.5688 个百分点 [-1.0329,4.3738]，没有可靠额外提速或突然跃升的证据。完整验证重合度从 26.7549% 平缓升到 27.4553%，最优节点为 46800。复用原数据与 CE+2TV，AdamW 重新初始化一次；一个训练 seed，既有开发来源，未评价新来源或三维画质。保留原 step12000 PF，不追加相同训练。原始记录复算、相同长度子集与设置见 [续训记录](FEATURE_MLP_EXTEND_2026-10-19.md)。

2026-10-19 Feature EAGLE v4 性能诊断：在 RTX 4090、真实场景前缀 256、FP32、温度/top-p 0.9、三个 seed 各 1,024 token 的同设置下，加入可选 Draft KV 复用。profile 中 Draft 输入由 345,971 降至 2,847 个 token（-99.18%），P/F 接受计数和输出保持一致。普通吞吐中位数由 v3 的 96.39 降至 95.15 token/s；三次合并吞吐由 96.60 降至 95.42 token/s（-1.22%），未显示加速。Target 验证占 v3/v4 插装 wall 的 67.3%/65.9%，新增状态提交成本高于节省的历史重放时长。旧 PF 为 117.03 token/s，仍最快。缓存复用保持可选且默认关闭；不追加训练。详见 [v4 执行记录](FEATURE_EAGLE_V4_2026-10-19.md) 与 [网页审阅包](artifacts/feature_eagle_v4_review.zip)。

2026-10-19 γ=5 定向尝试完成：在 v3 上增加 F2 KL 权重与双 Feature 递推比例，训练 1,200 次、选择 step900。144 次开发运行中新 γ=5 F1/F2 为 543/2159、136/492，吞吐 87.788 token/s；原 v3 γ=5 为 87.237，原 v3 γ=3 为 97.434。全部达到相同生成长度的 21 组中，新/原 γ=5 为 0.989×，没有可靠额外提速，保留原 v3 γ=3 和旧 PF。九项测试与真实分块、贪心、计时重放验证通过。详见 [γ=5 定向记录](FEATURE_EAGLE_GAMMA5_2026-10-19.md)。

2026-10-19 Feature EAGLE v3 完成：19 个既有真实场景、36 个裁剪扩为 2,220 条真实/Ranker 条件样本，按来源划分。以 Feature KL 为主要信号，主训练 1,600 次更新，加 800 次两处 Feature 递推；分别选 step1400/step500。108 次在线运行中 γ=3 的 F 接受为 182/518=35.14%，v2 为 117/542=21.59%；吞吐 97.545/86.382 token/s，提升 1.129×。旧 PF 为 115.628 token/s，仍最快。γ=5 的 F1/F2 为 136/520、30/116；未显示在线 F2 提升。两个场景八次自然 EOS 导出通过；一个相同 1,024 新 token 的三维单例中，v2/v3γ3/旧 PF 为 11.951/9.748/8.842 秒。九项测试通过，旧权重保留。详见 [v3 执行记录](FEATURE_EAGLE_V3_2026-10-19.md)。

2026-10-19 EAGLE v2：用户最新 Joint P/F 与 Ranker + Feature-only 方案完成代码、同预算训练和开发比较。A/B 均为 1L-256、1,572,864 参数，各 4,000 次更新、913,302 个状态监督位置，再追加 100 次真实候选短 rollout。三个既有开发来源、64/256 前缀、两个随机种子共 108 次运行中，Target 为 71.424 token/s，B γ=1/3/5 为 1.342×/1.213×/1.101×，B γ=4 为 1.130×；A γ=1/3/5 为 0.820×/0.739×/0.669×，旧 PF 为 1.622×。B γ=3 仍慢于 Ranker-only，A 的 P 接受率约 2%，当前没有证据支持替换旧 PF。六项核心测试通过，真实分块策略最大总变差约 2.17e-6。训练数据仅 19 场景且不含自然 EOS，尚缺独立来源确认。源码与结果见 [EAGLE v2 实现记录](EAGLE_PF_V2_2026-10-19.md)。

2026-10-17 Gap 条件选择路线 G5 完成：D1 10 个新开发来源共完成 180 次 512-token 和 90 次 2,048-token 测量。主要指标 Gap/PF-K2 为 1.018× [0.990,1.048]，6/10 来源更快；未达到预设的 1.02 点估计及区间下界高于 1.00 的条件，按计划停止在 D1，不开展 T 来源确认和完整三维生成。512-token Gap/PF-K2 为 1.009× [0.981,1.036]。Gap/P-only 在长、短窗口分别为 1.136× [1.113,1.161]、1.185× [1.158,1.212]；固定 PF-K2/P-only 也分别达到 1.116× [1.094,1.136]、1.174× [1.141,1.210]，所以本轮没有证明间隔选择持续优于固定 PF-K2。详见 [G5 结果](artifacts/gap_selection_20261017/G5_report.md) 与[阶段结论](artifacts/gap_selection_20261017/G8_final_decision.md)。

2026-10-17 基于 Position 间隔的策略选择计划 G0–G2 已完成：确认 Position Ranker 的首个 Position 提案为确定性输出，并核对 B1 序列化 ID 间隔；真实 Target 上的固定路径、路由切换、EOS、缓存状态和两步玩具分布测试通过。G2 在 56 个已用开发前缀、10 个来源、8 个种子上完成 1,344 次单轮测量。Gap 相对固定 PF-K2 的来源等权比为 0.996× [0.989,1.002]，暂未显示单轮净收益；相对 P-only 为 1.221× [1.159,1.275]。当前不支持 Gap 规则优于固定 PF-K2，连续在线结果待 G3。完整结果、运行记录和范围说明见 [Gap 计划 G0–G2](artifacts/gap_selection_20261017/G2_report.md)；原始评测留在同目录。

2026-10-17 基于 Position 间隔的策略选择计划 G3 完成：56 个开发前缀、10 个来源上完成 672 次 512-token 连续生成，另在 10 个来源上完成 90 次 2,048-token 长窗口生成。512-token 的 Gap/PF 为 1.006× [0.991,1.022]，尚不能确认收益；长窗口为 1.026× [1.008,1.044]，7/10 来源更快。按预设路线继续 G4 机制消融和 G5 新来源稳定性评测；两种窗口仍属于 D0 开发来源，不能作为独立确认。详见 [G3 连续生成结果](artifacts/gap_selection_20261017/G3_report.md)。

2026-10-17 G4 归因实验完成：在同一 D0 来源上，512-token 的 Gap/PF 为 0.998× [0.979,1.015]，未显示快于固定 PF；Gap 相对 PF 使用率相近的随机选择为 1.026× [1.010,1.040]。2048-token 的 Gap/PF 为 1.036× [1.017,1.054]，Gap/随机选择为 1.055× [1.034,1.075]。反向分配在两种窗口均明显变慢，额外间隔判断的耗时影响低于测量分辨率。结果支持继续以新来源验证较长窗口的收益；不能外推为独立确认或短窗口提速。详见 [G4 归因实验结果](artifacts/gap_selection_20261017/G4_report.md)。

2026-10-17 成本路线 B1 完成：56 条 PF 轨迹、5,122 轮中补采 5,078 个完整候选的精确 Feature 重合度，另完成 448 次固定前缀 PF 单轮重放，均与原输出和计数一致。时间仅复用 A1–A3 的 896 次低干扰单轮数据，整体即时效率比仍为 1.141× [1.085,1.195]。首个候选 Position 间隔为 1 / 大于 1 的组分别为 1.190× [1.117,1.261]、0.970× [0.934,1.015]。精确 Feature 重合度的第 10/50/90 百分位为 3.80%/29.60%/67.96%。固定岭回归按来源留一，生成前特征预测重合度 R²=0.264，加入草稿统计为 0.509；额外时间预测 R² 为负，弱于均值参照。当前支持上下文差异，未证明动态选择的净收益；没有新训练、在线策略或完整三维输出。详见 [Feature 条件收益报告](artifacts/feature_value_20261017/full/FEATURE_VALUE_REPORT.md)，远端 /root/autodl-tmp/gaussiangpt_feature_value_20261017/。

2026-10-17 候选成本 A1–A3 完成：十个开发场景、28 个前缀组合、FP32、最多 256 新 token。168 次主比较中，普通 PF/P-only 来源等权为 1.171× [1.130,1.211]；56 组分项运行复现原输出及前轮关键计数。56 个相同上下文、八个种子、两动作共 896 次单轮中，PF 平均推进 2.482 token、耗时 22.801 ms；P-only 为 2.000 token、20.973 ms，即时效率比 1.141× [1.085,1.195]。P2 拒绝及实际补算均为 113 次；少量计时点复测的整段处理为 1.553/116.107 s，占 1.337%，免费删除的乐观时间相减为 1.0136×。另有八次同步子集与八次普通交错对照；少量计时点吞吐比为 0.998×，逐操作计时为 0.975×。Feature 保留，暂缓单纯提高 P2 接受率的独立 Head；更便宜的提案机制另行比较。没有训练新模型、运行新正式测试或三维输出。见 [成本报告](artifacts/candidate_cost_20261017/full/CANDIDATE_COST_REPORT.md)，原始结果在同目录；远端 /root/autodl-tmp/gaussiangpt_candidate_cost_20261017/。

2026-10-17 多头与特征草稿第二轮首批完成：A0/A1/E0 确认旧 Medusa/Hydra 自行提出 Position，历史末尾 Feature 未进入 P1；训练与在线 token 边界相符，已测状态归一化和位置顺序一致。448 个真实上下文中 P1 重合度为 1.566%/1.407%。四种共享 Position 的 Feature 结构完成三个训练种子共 12 个模型；首种子七路径 392 次在线开发比较中，旧 PF 来源等权为 1.624× [1.557,1.682]，新结构中 B1-E 最高，为 1.506× [1.450,1.558]，仍慢于旧 PF。另完成 56 次分项测量、448 上下文的三种子离线比较及 A2 两头训练和 12 次短测。K2 提交量与调用次数关系在 280 条 PF 记录中一致，旧 PF 的 P2 未接受数至多对应 1.92% Target 调用。EAGLE 保存张量确认为 FP32，训练 AMP 为 BF16；16,360 行 FP32 top-p=0.9 重合度为 2.776%。当前保留旧 PF；尚未开展候选树、跨轮递归和新方案正式三维确认。细节与范围见 [第二轮执行记录](ROUND2_EXECUTION_2026-10-17.md)。

2026-10-17 下一轮计划完成：步骤 0–7 的计划工作已结束，涵盖采样与缓存验证、性能拆分、P/F 消融、输入变化诊断、Medusa-style/Hydra-style 对照和提案方式选择。步骤 4 的短程训练对照没有稳定超过 step12000，未采用新权重；步骤 6 根据分层结果使用固定 PF，未运行在线动态切换。步骤 7 在 30 个新来源上完成 666 次 Token 评测和 270 次完整三维输出。Token 微平均吞吐 Target/P-only/PF 为 71.996/102.685/119.729 token/s；来源组几何平均 PF/Target 为 1.674× [1.642, 1.706]、PF/P-only 为 1.170× [1.155, 1.186]。完整流程平均耗时为 46.35/31.37/29.41 s；来源组速度比 PF/Target 为 1.530× [1.340, 1.721]，PF/P-only 为 1.058× [0.958, 1.167]，后者尚不确定。质量指标只覆盖完整 chunk，尚无生成区域遮罩；预训练资料缺少场景 ID 清单，无法确认预训练来源与确认来源完全隔离。详见 [下一轮计划执行记录](NEXT_ROUND_RESULTS_2026-10-17.md)。远端原始数据位于 `/root/autodl-tmp/gaussiangpt_next_round_20261017/`。

2026-10-16 Profiling 数据整理：远端 P1 五次原始运行已定位，AR 中位时间为 66023.570 ms，Core 为 66311.287 ms，逐运行 AR/Core 占比中位数为 99.5653586%。Core 包含 Prefill、AR、token 后处理和 VQ 解码，不含输入压缩、加载、KV 分配及渲染。独立 AR 窗口的 Target forward 占比为 96.682793%，两项不相乘推导完整流程 forward 占比。

本次还复算 P3/P4/P5 汇总、Adapter 分项及阶段 F 全部 360 条样本，并整理阶段 E 摘要。P7 局部延迟表将 32 个分段横坐标均写为 8192，报告使用 P3 原始 `context_length_end`；局部延迟从上下文 513 的 13.47747 增至 8192 的 30.54179 ms/token。完整结果、六张图、CSV/JSON 与复算入口见 [Obsidian Profiling 数据汇报](</Users/lief/workspace/obsidian_md/投机解码/GaussianGPT-specific Speculative Decoding Design/GaussianGPT Profiling 数据汇报.md>)。本次未运行新训练、GPU 实验或渲染，状态快照未刷新。

2026-10-15 阶段 E 完成与阶段 F 接续：

阶段 E 正式完成：30 场景全部制作成功，56 条唯一裁剪、63,328 个正文 token，64/256 前缀各 56 条、1024 前缀 32 条；按三个 seed、五模式得到 2,160/2,160 次 FP32 评价。微平均速度比 T/P/PF/PF-D8/GPT-D8 为 1/1.43491/1.69675/1.45756/0.94020；来源组几何平均为 1/1.43773/1.71250/1.46646/0.93815。轻量 PF 区间 [1.68044,1.74402]，30/30 来源组快于 Target；GPT-D8 有 24/30 来源组低于 1×。P-only、PF、PF-D8、GPT-D8 的 P 接受率为 92.4408%/92.6410%/92.5066%/78.9677%；后三者 F 为 33.2582%/55.4712%/55.5221%。接受率为 accepted/verified。微平均以全池 token/时间计算，几何平均对来源组等权，二者不混用；正式微平均 1.69675× 与开发微平均 1.68765× 同口径。该结果不用于重新选择模型，主方法保持旧 step12000 轻量 PF。

正式证据为远端 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/fp32_test_full/{online.json,multiscene_summary.json}`，状态 `stage_e_final/pipeline.json` 为 complete，进入 ready_for_full_3d_stage。服务器在本轮开始时 GPU 空闲。阶段 F 接下来先在两个开发场景接通完整生成、VQ 解码、Gaussian 输出与固定视角渲染；T、P-only、轻量 PF 三模式去除重复最终权重。阶段 E 的固定窗口速度仍属于 token 解码结果，尚不能宣称完整流程速度或三维质量等效。


阶段 F 完成。真实补全 270/270、BOS 90/90 均成功且自然结束。完整输出从准备好的输入计至 Gaussian NPZ 与六张 PNG 保存；条件为真实前缀 256、正文上限 16,000、FP32、temperature/top_p=0.9，覆盖单个 20×20×20 latent chunk。真实补全中 P-only/Target 端到端场景等权速度比为 1.467× [1.266,1.731]，轻量 PF/Target 为 1.569× [1.318,1.852]；轻量 PF/P-only 为 1.070× [0.912,1.252]，区间跨过 1。轻量 PF 的 P/F 接受/已核验为 91.720%/23.256%。全图 PSNR/SSIM只作描述，不代表生成质量相同。完整统计、方法口径、图像质量限制与视频见 [阶段 F 记录](FULL_3D_OUTPUT_2026-10-15.md)。远端状态为 `complete / analysis_complete`，GPU 作业已结束。

2026-10-15 P/F 多场景计划执行：阶段 A/B 已完成，10 个开发场景形成 19 条唯一裁剪。FP32 五模式开发评价完成 810/810 条，12 项数值与重复运行验证通过。Target 为 71.96 token/s；十来源组等权速度比：P-only 1.4357× [1.4264,1.4457]，轻量 PF 1.6882× [1.6561,1.7192]，PF-D8 1.4434× [1.4241,1.4620]，GPT-D8 0.9737× [0.9614,0.9856]。轻量 PF 的 P/F 接受率为 91.78%/33.40%；PF-D8 的 Feature 接受率为 55.49%，但速度更低。29 个场景×前缀组合均由轻量 PF 胜出，当前不做动态调度。

BF16 五条短测完成，但 Target 与 D8 的分块/逐 token 分布最大总变差分别为 0.0282/0.0632，高于 0.01 容差，未开展正式 BF16 速度比较。真实来源适配集 19/20 个场景重建成功，形成 36 条唯一裁剪、42,236 个真实 token；一个原始 OBJ 索引越界，保留来源与日志。288 条 Target 续写全部完成，生成 554,705 token，45 条自然 EOS（15.625%）；含真实前缀的正文共 600,740 token。

D2 首对两组各 3,600 步训练与 810 次开发评价全部完成。原始 Target 监督，A 仅旧数据，B 为旧/适配来源 70/30。旧/A/B 的在线 F 已验证接受率为 33.40%/32.92%/34.00%，来源组等权速度为 1.6869×/1.6782×/1.6897×。B/旧速度比 1.00168，区间 [0.99219,1.01034]；B/A 为 1.00688，[0.99368,1.02163]，尚无可靠额外提速。固定真实前缀重合度 B 为 26.8020%、A 为 25.2824%，来源组等权差 1.5151 个百分点，[1.2440,1.8037]，10/10 个来源组改善。分布匹配改善与在线收益分别保留，不追加训练 seed，正式主方法继续使用旧 step12000 轻量 PF。

该阶段 E 启动快照中，阶段 F 尚未开始。30 个独立测试场景制作进程 PID 430778；后续任务 PID 431988 自动接续裁剪、五模式短测与最多 2,700 次 FP32 正式评价。选择在方法结果生成前写入 `stage_e_final/method_selection.json`；状态 `stage_e_final/pipeline.json`，正式输出 `stage_e_final/fp32_test_full/`。新增 `patches/pf_multiscene/run_final_evaluation.py`，汇总脚本新增集合名称参数；已有 810 次数值汇总重放一致，计数排除旧开发场景，CPU 验证通过。

详细结果与路径见 [多场景执行记录](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)。D2 证据位于远端 `/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_d2_adaptation/stage_c/dev_pair_seed42001/`，包含逐次结果、来源组区间和 `development_decision.json`；原始模型与新 A/B 权重分别保存。阶段 C 的五模式开发结果仍在 `stage_c/fp32_dev_full/`，不同轮次的绝对速度不拼接。

2026-10-14 后续规划：用户同意研究方向并要求规划，已完成 [P/F多场景研究计划](PF_MULTISCENE_RESEARCH_PLAN_2026-10-14.md)。先以两个场景确认数据制作与完整渲染，再开展10开发场景、五个固定模式的同条件比较；根据开发证据开展P-only/K2动态选择或轻量F真实来源适配，最终在至少30独立测试场景和完整三维流程上评价。首轮可选适配训练集20场景，原始Target监督；测试不用于选模。FP32与可用BF16分开报告，区间按来源场景分组计算。此项为规划记录，未新增训练、测速或渲染结果。

2026-10-14 D4首轮：已按计划完成8600步直接Target训练，累计35986829个预测位置、17992103个F位置。120次验证集比较选择step7400，216次真实前缀评价及10组分项计时均完成。D4 Gamma2/4/6的P接受率为19.2623%/20.5253%/20.2518%，F为4.4105%/4.1906%/4.0446%，速度比0.6328×/0.4918×/0.4006×；同轮D8 Gamma2为0.9670×，轻量Feature加Position ranker为1.4630×。D4提案成本为D8的61.74%–65.69%，但每轮接受长度仅约0.13–0.14，尚无加速收益。固定真实F重合度为5.3666%，D8为52.3789%。全部数值、恢复、计数和汇总复核通过，原权重及语料保留，GPU空闲；随机初始化D4与预训练D8起点不同，只覆盖一个真实场景。详见 [D4执行记录](D4_JOINT_TRAINING_2026-10-14.md)，原 [首轮计划](D4_JOINT_TRAINING_PLAN_2026-10-14.md) 保留规划时状态。远端结果目录 `draft_a/runs/d4_joint_20261014/` 已完成，无需重复启动。

2026-10-14 D8 Gamma 对照：用户要求将 D8 联合 P/F 的 Gamma 提升至 4 和 6。同轮 Target 与 Gamma2/4/6 共 144 次真实前缀评价已完成，Feature 接受率为 54.3726%/53.6912%/53.6286%，Position 为 86.4263%/86.8075%/88.5089%，速度比为 0.9696×/0.9127×/0.8141×。每轮接受长度增至 1.2098/1.7871/2.1229，但较长提案吞吐下降，三个种子组和四份裁剪的速度顺序均为 Gamma2 > Gamma4 > Gamma6。模型权重未修改，数值与九组短验证通过，Gamma2/Target 的 72 次计数复现上一轮，GPU 空闲。只有一个真实场景，结果限于当前 FP32 token 解码。详见 [Gamma 对照记录](FEATURE_D8_GAMMA_2026-10-14.md)。

2026-10-14 高接受率 D8 真实输入对照：使用 `adapt_d8_acceptance/step_3600.pt`，真实四裁剪、64/256/1024 前缀、三个种子组，四模式共 144 次已完成。D8 Feature 加原 ranker 的 P/F 接受率为 86.4366%/53.2552%，速度 1.3561×；D8 联合 P/F Gamma2 为 86.4263%/54.3726%、0.9595×。同轮轻量 MLP 为 85.0586%/20.2967%、1.4483×。固定真实 Feature 前缀的 D8 分布重合度 52.3789%，明显高于 MLP 的 13.4599%；D8 实际 Feature 接受率提高，但吞吐低于 MLP。只有一个真实场景，使用本轮同条件比较，不能和历史 Teacher 测速直接拼接。数值及缓存验证通过，原始权重保留，GPU 空闲。详见 [D8 真实输入评价记录](FEATURE_D8_REAL_EVAL_2026-10-14.md)。

2026-10-14 真实前缀评价：一个真实 3D-FRONT 重建场景的四个不同 VFront 裁剪，64/256/1024 token 前缀、三个种子组、最多续写 512 token，三组共 108 次运行已完成。step3600 P/F 接受率 85.6598%/21.9531%，速度 1.4811×；step12000 为 85.0586%/20.2967%，速度 1.4654×。新/旧吞吐点估计为 0.98943，三个种子组的方向不一致；不能据此断言跨场景退化。4420 个相同完整真实 Feature 前缀上的重合度仅从 13.3945% 升至 13.4599%，续训收益很小。计时采用两条路径共用外层范围；真实前缀的速度不可与此前 Teacher 测速直接拼接。数据来源、去重、自然 EOS 和分前缀结果见 [真实 VFront 评价记录](FEATURE_REAL_EVAL_2026-10-14.md)。模型权重保留，GPU 空闲。

2026-10-14 续训：按用户要求，直接 Target 的轻量 Feature Draft 从 step3600 增加到 step12000，新增 8400 步、17556882 个 Feature 监督位置。完整验证重合度增加 0.3487 个百分点；全部 50 条测试轨迹相同前缀增加 0.3708 个百分点，区间 [0.2230,0.5276]，43/50 条轨迹改善。108 次在线对照中，新 Feature 接受率 32.1659%，同轮 step3600 为 32.9079%；速度 1.7363× 对 1.7404×，新/旧区间 [0.97596,1.01747]，未见稳定的额外在线收益。训练分布对齐仍有小幅提升，最后 1200 步验证增幅约 0.0058 个百分点。正式权重保留，作业已结束。配置、曲线及统计限制见 [继续训练记录](FEATURE_TARGET_CONTINUE_2026-10-14.md)，远端结果目录 `draft_a/runs/feature_target_continue_20261014/`。

2026-10-14：用户要求旧轻量 Feature Draft 直接学习原始 Target，并复用 GPT 成功训练的完整分布 CE、全部可用位置、低学习率续训与正式采样策略 TV。完整三阶段 3600 步与 108 次在线对照已完成，选定 step3600。新模型 F 接受率 32.9079%，同轮旧 MLP 为 29.5130%；差值 3.3949 个百分点，95% 区间 [0.8673,5.5180]。新速度为 1.7294×，旧为 1.7092×；新/旧区间 [0.98589,1.03544]，尚不支持稳定额外加速。正式权重保留。运行对象、数据范围、训练记录及旧迁移实验与本版本的差异见 [直接 Target 训练记录](FEATURE_TARGET_V2_2026-10-14.md)。原始产物为远端 `draft_a/runs/feature_target_full_v2_20261014/`。

2026-10-13 后续评价：Draft-A v2 顺序任务已完成，普通 Draft-A 在线速度比最高 0.874×。LFQ 联合候选的独立测试与五组轮转解码测速也已完成；F1 测试按轨迹增加 0.1002 个百分点，F2 在两个新提议器前缀来源上分别增加 0.2089、0.1305 个百分点。24 seed 在线速度比为父 K2 1.7307×、联合 K2 1.6940×、父 K3 1.6610×、联合 K3 1.6067×；联合候选没有取得速度收益，正式权重不替换。完整配置、区间与在线统计见 [联合候选独立评价](PF_JOINT_EVALUATION_2026-10-13.md)。以下启动状态为此前记录。

2026-10-13 Feature Draft 更新：CPU 输出头误差分析与三组同预算训练对照已完成，保留正在运行的 Draft-A v2。完整验证集的联合分布重合度为 28.9468%；原目标与新增输出头监督均未超过父模型。轨迹均衡采样的最佳按位置重合度为 28.9654%，按轨迹仅增加 0.0163 个百分点，尚无明显收益。正式权重未替换，GPU 测速未运行。配置与证据见 [Feature Draft 接受率实验](LFQ_FEATURE_ACCEPTANCE_2026-10-13.md)。随后两组 F1/F2 联合递归 CPU 对照完成各 4 轮；joint 第 2 轮 F2 按来源与轨迹重合度增加 0.4441 个百分点，F1 按轨迹下降 0.0200 个百分点。已保留并实际载入成对候选，未读取测试集，未形成新增加速比结论。见 [联合递归训练记录](PF_JOINT_RECURSIVE_2026-10-13.md)。

2026-10-13 更新：用户已授权恢复实验，扩大 Draft-A 的 Teacher corpus 至 500 条。远端 `draft_a/runs/rollout500_20261013/` 已启动语料生成及后续顺序任务；包括 5000 步 CE 训练、独立集合 overlap 和四档提案长度的在线解码对照。原 43 条语料实际含 34 条自然 EOS。模型结构和 Target 参数保持不变。配置、产物位置、已完成短测试与限制见 [Draft-A v2 实验记录](DRAFT_A_ROLLOUT_V2_2026-10-13.md)。以下旧日期的暂缓描述保留为历史记录。

2026-10-08 状态：暂缓新增实验，完成阶段性整理。S2—S13 的工作演进、实验总表、资产索引和统计限制见 [阶段工作汇报](STAGE_REPORT_2026-10-08.md)。本轮整理没有新增模型训练、性能测量或渲染结果。

本文记录当前本地实验线、可复用 checkpoint 和已经实际测量的结果。生成的最新文件索引见 [`PROJECT_STATUS.md`](PROJECT_STATUS.md)。

面向当前瓶颈的论文检索和候选方案见 [`RESEARCH_SOLUTIONS.md`](RESEARCH_SOLUTIONS.md)；实现前的正确性审查、优先级和停止条件见 [`LOGIC_VALIDATION.md`](LOGIC_VALIDATION.md)。其中不少方案仍是待验证假设。

2026-10-19 EAGLE v2 三维验证：同一实际 256-token 前缀上，Target、B γ=3、A γ=3 均生成到自然 EOS，完成 VQ 解码、Gaussian NPZ 和六视角 PNG。含导出总时间分别为 24.483/20.443/31.925 秒，完整 body 长度分别为 1,892/1,878/1,862 token。单例用于确认完整流程，未开展等质量统计；模型加载不计时。见 [完整输出数据](artifacts/eagle_pf_v2_20261019/full3d_results.json)。

## S2：小模型 draft 训练线

S2 的目标是得到能和原版 teacher 对齐、又能降低单步推理成本的 draft GPT。四个配置只改变 Transformer 容量，保留相同词表、serialization、3D RoPE 和 16384 context：

| draft | embedding | layers / heads | 当前用途 |
| --- | ---: | ---: | --- |
| D4 | 256 | 4 / 4 | 快速 smoke / 最小容量对照 |
| D6 | 384 | 6 / 6 | 中间容量对照 |
| D8 | 512 | 8 / 8 | 当前主要 speculative draft |
| D12 | 768 | 12 / 12 | 接近 teacher 的容量对照 |

标准 S2 配置 `conf/training/gpt_draft_s2.yaml` 使用 38 epochs、`batch_scenes=4`、`accumulate_grad_batches=4`、bf16 mixed precision、每 10 个 epoch 做 render、每次 4 个 sample，采样温度 0.9、top-p 0.9。smoke 配置只跑 1 epoch；runtime pilot 用于快速验证训练入口。

当前可复用的主要 D8 checkpoint：

```text
profiling/s2/d8_canonical_38ep/logs/s2_d8_canonical_38ep/lightning_logs/version_0/checkpoints/val_loss_monitor_epoch_37_step_50502.ckpt
```

teacher 与 VQ-VAE 的基础 checkpoint 通常为 `checkpoints/gpt_vfront.ckpt` 和 `checkpoints/vqvae_vfront.ckpt`。`checkpoints/gpt_vfront.ckpt.corrupt` 已知不可用，不能当作实验输入。

## S3：feature distillation 与 on-policy 对齐

- `s3_feature_distill_train.py` 在 canonical CE 上增加 feature-token 行的 teacher→draft KL，记录 feature CE、KL、top-1 agreement 和 raw/policy overlap。
- `s3_onpolicy_feature_distill_train.py` 先让冻结 teacher 用温度 0.9、top-p 0.9 采样 trajectory corpus，再训练 D8 student；它要求有效序列至少包含两个 token，并固定 student embedding width 为 512。
- `s3_feature_alignment_diagnostic.py` 用于检查 raw/policy 分布、熵、KL、TV、overlap、top-1、correct-type mass 和条件 type overlap，适合在改变蒸馏权重或温度后先做诊断。

当前解释：只有 position token 是 speculative proposal，位置分布一致性直接决定接受率。Feature token 由 target 采样，draft 的 feature overlap 只作诊断；它可能通过共享参数间接影响 position logits，但当前记录没有证明这种影响。因此只看整体 token accuracy 会掩盖结构和外观两类 token 的差异。

## Position-selective speculative decoding

数学自检已通过：

```text
python scripts/selftest_position_selective_math.py
POSITION-SELECTIVE MATH SELFTEST: PASS
theoretical overlap = 0.550000
empirical acceptance = 0.551720
```

端到端测试使用同一个 teacher、VQ-VAE、D8 draft、256 tokens、warmup 32、seed、CUDA、temperature 0.9 和 top-p 0.9。已有 JSON 证据：

| 测试 | baseline tok/s | selective tok/s | 约 speedup | 位置统计 |
| --- | ---: | ---: | ---: | --- |
| `profiling/e2e_wallclock_seed2.json` | 74.29 | 78.88 | 1.06× | 128 pairs，118 position accepts，10 rejects，实际 position accept rate=0.9219 |
| `profiling/e2e_real_seed1.json` | 73.42 | 78.02 | 1.06× | 同一基准设置 |
| `profiling/e2e_fair_baseline_seed1.json` | 66.51 | 76.68 | 1.15× | 更严格的公平 baseline |

在 `e2e_wallclock_seed2.json` 中，记录的位置分布 overlap 约 0.9376，feature 分布 overlap 约 0.3823；实际 position accept rate 为 0.9219。Feature overlap 的上下文和 policy 口径还应从原始 JSON 核实；由于 feature 由 target 采样，这个值不能单独解释 position rejection 或端到端瓶颈。三组 speedup 有运行波动，不能把单次 1.15× 当作稳定结论；后续应固定多 seed 并报告均值、方差和端到端 wall-clock。

## 已知限制与下一步优先级

1. 用 `s3_feature_alignment_diagnostic.py` 记录 feature logits 的 KL/overlap，但当前 feature 由 target 采样，这些指标不直接进入 speculative acceptance；若评估 S3 对加速的价值，必须同时检查它是否改善 position logits/acceptance。
2. 做多 seed、相同 warmup 和相同 CUDA 状态的 e2e benchmark，分项记录 D8 proposal、target block、position rejection 后重算、cache 和 Python/CUDA 调度时间。
3. 若 position acceptance 提升但 wall-clock 不升，或 q_F overlap 改善而 position acceptance 不变，检查主要成本并停止把 feature distillation 当作直接加速方案。
4. 任何新实验都要记录配置、checkpoint、seed、硬件、JSON 路径和结论，并运行状态刷新脚本。

## Fine-grained profiling instrumentation

远端 `model/gpt/position_selective_speculative.py` 已加入 profiling-only 的 `stage_timing` 字段，并通过 `python -m py_compile`。每个阶段记录样本数、均值、中位数和 p95；现有 `timing` 仍按 `_forward` kind 记录同步后的模型前向耗时。

当前阶段名为 `draft_position_sampling`、`target_position_postprocess`、`target_cache_rewind`、`target_feature_sampling_postprocess`、`pair_commit_prepare`、`draft_feature_diagnostic` 和 `cache_invariant_checks`。其中 sampling/postprocess 阶段包含对应的 policy 概率处理、采样与必要的 `.item()` 同步；feature diagnostic 只诊断 q_F，不代表 feature acceptance。

`target_position_postprocess` 与 `target_feature_sampling_postprocess` 测量的是共享 target 双 token forward 之后各自的处理，不是独立的 (C_{V,P})、(C_{V,F}) GPU 前向成本。当前 target 仍用一次 `target_verify_2` 同时产生 P/F logits；没有额外执行 target forward 来拆分两类 logits 的 GPU 时间。各阶段也不覆盖循环里的全部 Python 调度，因此分项和不应被当作完整端到端延迟。

在远端 JupyterLab 上完成了 1024/2048/4096 token 的 seed 5 profiling。三档使用相同的 teacher、D8 draft、VQ-VAE checkpoint，warmup 128、temperature/top-p 均为 0.9、CUDA；GPU 为 NVIDIA GeForce RTX 4090 (24,564 MiB)。原始输出保存在 `profiling/route_a/stage_detail_seed5_{1024,2048,4096}.json`。

| 生成 token | T_P (ms) | T_F (ms) | D_2 (ms) | V_2 (ms) | C_1 (ms) | position accept | 公式预测 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 13.5690 | 13.1928 | 6.5024 | 16.8440 | 13.2853 | 498/512 = 97.27% | 1.1287x |
| 2048 | 13.5412 | 13.1865 | 6.4756 | 16.7586 | 13.1039 | 1000/1024 = 97.66% | 1.1354x |
| 4096 | 14.5672 | 14.2165 | 6.6688 | 17.6637 | 15.0392 | 1985/2048 = 96.92% | 1.1609x |

这里的 speedup 是按 `(T_P+T_F)/(D_2+V_2+(1-alpha_P)*C_1)` 得到的模型项估算，不是端到端 wall-clock 实测。初步看，draft 双位置更新约 6.5-6.7 ms，target 双 token 验证约 16.8-17.7 ms；相较之下，位置拒绝率约 2.3%-3.1%，拒绝后单位置 target correction 的代价虽高 (13.1-15.0 ms)，加权到每对提议上的惩罚仍约 0.3-0.5 ms。即使位置接受率略有下降，公式预测值随序列增长而升至约 1.16x，主要反映此估算中的 target AR 单步基准耗时略增；仍需端到端测量确认。

该组是一个 seed 的单次运行；JSON 的阶段 p95 描述此序列内各事件的延迟分布，不代表多次运行之间的方差。跨长度的微小变化不应解释成稳定的长度趋势。

三个 JSON 已包含 `stage_timing` 的 count/mean/median/p95。下表给出 mean/p95 (ms/event)；除 cache rewind 外，count 等于提议对数，cache rewind 的 count 等于 position rejects。完整统计仍保留在各 JSON 中。

| 阶段 | 1024 mean/p95 (n) | 2048 mean/p95 (n) | 4096 mean/p95 (n) |
| --- | ---: | ---: | ---: |
| draft position sampling | 0.3150/0.4022 (512) | 0.3108/0.3999 (1024) | 0.3105/0.3979 (2048) |
| target position postprocess | 0.3595/0.4119 (512) | 0.3560/0.4147 (1024) | 0.3532/0.4260 (2048) |
| target cache rewind | 0.0400/0.0446 (14) | 0.0367/0.0451 (24) | 0.0351/0.0405 (63) |
| target feature sampling/postprocess | 0.4174/0.4468 (512) | 0.4169/0.4427 (1024) | 0.4161/0.4431 (2048) |
| pair commit prepare | 0.0310/0.0372 (512) | 0.0308/0.0361 (1024) | 0.0309/0.0361 (2048) |
| draft feature diagnostic | 0.3003/0.3148 (512) | 0.3007/0.3239 (1024) | 0.2969/0.3118 (2048) |
| cache invariant checks | 0.0220/0.0226 (512) | 0.0219/0.0225 (1024) | 0.0218/0.0225 (2048) |

在已计时的非-forward 阶段中，最大单项是 target feature sampling/postprocess (约 0.42 ms/对)，其次为 target position postprocess (约 0.35 ms/对)、draft position sampling (约 0.31 ms/对) 和不参与 acceptance 的 draft feature diagnostic (约 0.30 ms/对)。cache rewind 只在拒绝时发生，每次约 0.035-0.040 ms；pair commit 和 invariant check 更小。与 target 双 token forward 的 16.8-17.7 ms、draft 双位置 forward 的 6.5-6.7 ms 相比，结果支持当前延迟主要由模型计算构成，验证后处理/cache 管理不是首要瓶颈。若运行时不需要 feature overlap 诊断，可考虑关闭 q_F diagnostic，但其收益上限仅为约 0.30 ms/对。

该结论仅限当前插桩的阶段：它们不覆盖循环内所有 Python 调度，也没有把共享 target forward 拆成独立的位置/feature GPU forward。因此各阶段不可直接相加当作完整 wall-clock，预测 speedup 也仍需真实端到端测量验证。

## S3 F-aware 草稿的直接验证（2026-10-03）

为验证可迁移的 feature-aware/on-policy distillation 思路，在不改变 target、VQ-VAE、采样参数和 position-selective 解码逻辑的条件下，只替换 draft checkpoint。所有结果使用同一台 NVIDIA GeForce RTX 4090、CUDA、seed=5、temperature=0.9、top-p=0.9、warmup=128、`max_tokens=1024`；因此这里比较的是草稿训练状态对 position proposal 和 feature diagnostic 的影响，而不是新的解码算法。

| draft | checkpoint | tokens | position accept | position overlap | feature overlap | D2 (ms) | V2 (ms) | 公式预测 speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D8 baseline | `profiling/s2/d8_canonical_38ep/logs/s2_d8_canonical_38ep/lightning_logs/version_0/checkpoints/val_loss_monitor_epoch_37_step_50502.ckpt` | 1024 | 498/512 = 97.27% | 0.9720 | 0.3876 | 6.4889 | 16.8301 | 1.1407x |
| on-policy feature distill, 256 step | `profiling/s3/s3_onpolicy_feature_distill_32x2048_256step/checkpoints/onpolicy_feature_kl_best_epoch07_step256.ckpt` | 1024 | 503/512 = 98.24% | 0.9862 | 0.7037 | 6.5937 | 17.0964 | 1.1424x |
| on-policy feature distill, 256 step | 同上 | 2048 | 1002/1024 = 97.85% | 0.9835 | 0.6107 | 6.5138 | 16.9148 | 1.1529x |
| on-policy feature distill pilot, 64 step | `profiling/s3/s3_onpolicy_feature_distill_pilot_64step/checkpoints/onpolicy_feature_kl_best_epoch07_step64.ckpt` | 1024 | 477/512 = 93.16% | 0.9295 | 0.3384 | 6.8000 | 17.5779 | 1.0964x |

原始 profiling JSON 为 `profiling/route_a/decomp_seed5_1024.json`、`profiling/route_a/feature_distill_1024.json`、`profiling/route_a/feature_distill_2048.json` 和 `profiling/route_a/feature_distill_pilot64_1024.json`。256-step checkpoint 在 1024 tokens 时相对 baseline 将 feature overlap 提高约 0.316、position acceptance 提高 0.98 个百分点；在 2048 tokens 时仍保持 97.85% 的实际接受率。当前引擎只对 Position 做 speculative acceptance，Feature 仍由 Target 采样，所以 q_F overlap 不会直接减少 `target_verify_2` 的调用次数。与此同时，D2/V2 测得略慢；模型项预测加速从 baseline 的 1.1407x 增至 1024-token 的 1.1424x、2048-token 的 1.1529x，按 overlap 估算的 2048-token 上界为 1.1561x。64-step pilot 同时损伤 position acceptance、feature overlap 和单步耗时，说明短训练 checkpoint 不能作为有效的 F-aware 草稿。

端到端 wall-clock 复核使用 `profiling/e2e_wallclock_faware256_seed1.json`：`run_e2e_fair` wrapper 固定同一 teacher/VQ-VAE、CUDA、`max_tokens=256`、warmup=32 和 temperature/top-p=0.9，仅替换为上述 256-step F-aware draft。baseline 为 69.6680 tokens/s，selective 为 79.2048 tokens/s，测得 wall-clock speedup 为 1.1369x；Position 接受 127/128 = 99.22%。这是单 seed 结果，且 wrapper 内 baseline/selective 使用不同的 seed offset，因此用于确认量级，不替代多 seed fair benchmark。

这组结果把方案分成两个层次：

1. F-aware/on-policy distillation 已证明能改变草稿分布，而且在训练充分时能小幅改善 Position proposal；它适合作为后续 joint/structured speculation 的草稿训练目标。
2. 在当前“只 speculative Position、Target 每轮验证 P/F 两个 token”的实现中，单纯替换为 F-aware draft 不是有意义的加速方案。主要瓶颈仍是 `target_verify_2`（约 17 ms/对），不是 q_F 的诊断或 Python 后处理。

因此下一阶段应优先实现能实际减少 Target verification 次数或降低每次验证成本的机制：例如按 position acceptance/不确定度自适应 proposal block，或让 Feature proposal 进入一个经过证明的联合 acceptance 路径；F-aware distillation 应作为这些机制的草稿训练基础，而不是单独的终点。实现任何联合 Feature acceptance 前，必须先补充 token-type 条件分布、rollback 和 exactness 测试，不能把 feature overlap 直接当作无损接受率。

## S4 采样温度敏感性与自适应 proposal 动机（2026-10-03）

在同一 D8 draft、同一 teacher/VQ-VAE、seed=5、1024 tokens、warmup=128、top-p=0.9 下，只改变 target 与 draft 的共同 temperature，得到：

| temperature | position accept | feature overlap | D2 (ms) | V2 (ms) | 公式预测 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7 | 506/512 = 98.83% | 0.4055 | 6.4052 | 16.5672 | 1.1644x |
| 0.9（baseline） | 498/512 = 97.27% | 0.3876 | 6.4889 | 16.8301 | 1.1407x |
| 1.1 | 438/512 = 85.55% | 0.3163 | 6.5641 | 16.9002 | 1.0614x |

对应 JSON 为 `profiling/route_a/temp07_1024.json` 和 `profiling/route_a/temp11_1024.json`。低温度提高 proposal 集中度并减少拒绝，高温度则相反；这验证了 acceptance 对 proposal 不确定度敏感。但因为 temperature 同时改变了最终 Target 的采样分布，不能把 0.7 的绝对 speedup 直接解释为同一生成目标下的无损收益。更合理的迁移方向是从 draft logits 估计位置级不确定度，在固定 Target 分布下自适应决定 proposal block 大小、是否跳过低收益 proposal，或选择更严格的候选截断；所有这些策略都应以 exactness/self-test 和相同 Target 分布的 wall-clock 对比为准。

## S5 全 token 结构化投机的负对照（2026-10-04）

为了检验能否把普通文本 speculative decoding 直接迁移到 GaussianGPT，远端 RTX 4090 上新增了 scripts/structured_speculative_k2.py 原型。它使用低层 cached transformer，在每轮一次提出多个完整 [Position, Feature] pair，再由 Target 统一验证；实现了 cache 截断、回滚、稀疏 token 合法性和最终 cache 不变量检查。该路线作为负对照，不改变当前 canonical position-only engine 的默认行为。

| pairs/round | temperature | baseline tok/s | structured tok/s | wall-clock speedup | token acceptance | cache/stream |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.0 | 49.408 | 26.353 | 0.533x | 0.487 | pass |
| 2 | 1.0 | 50.691 | 28.924 | 0.571x | 0.398 | pass |
| 4 | 1.0 | 49.675 | 21.553 | 0.434x | 0.213 | pass |
| 2 | 0.1 | 50.730 | 40.329 | 0.795x | 0.630 | pass |
| 4 | 0.1 | 52.155 | 46.451 | 0.891x | 0.563 | pass |

原始结果位于远端 profiling/route_a/structured_k1_128.json、structured_k2_128.json、structured_k4_128.json、structured_k2_temp01_128.json 和 structured_k4_temp01_128.json；32-token smoke 位于 structured_k2_smoke.json。所有运行均保持 valid_token_stream=true 和 cache_invariant=true。这证明实现的 token grammar 与 cache 回滚在测试范围内自洽，但尚未替代独立的逐分布 exactness 采样检验，因此不能把它表述为完整的无损证明。

结果说明，增加候选 pair 数不能抵消当前问题。temperature=1 时，K=1/2/4 的接受率从 0.487 降到 0.398/0.213，端到端速度分别只有 0.533x/0.571x/0.434x；temperature=0.1 虽将 K=4 接受率提高到 0.563，仍只有 0.891x。失败原因不是单一的 Python 验证开销：draft 需要额外生成无法看到中间 Target Feature 的 pair，Target 还要验证大量最终会被拒绝的 token；例如低温 K=4 的两次运行中，Target/Draft token 数为 243/243 和 203/203，draft forward 约 1.52–1.92 s，而 structured 总时间仍约 2.42–3.19 s。

架构上，当前 canonical decoder 的依赖是 P_i -> Target-sampled F_i -> P_{i+1}。普通文本式的连续多 token draft 假设后续 token 可以由 draft 自己的前缀递推，而这里的下一个 Position 条件包含 Target 才知道的 Feature；直接把多个 pair 串成一个验证块会积累条件分布失配。这解释了为什么全 token route 的正确性检查可以通过、但接受率和 wall-clock 仍然不划算。

因此该实验把后续边界固定为：F-aware/on-policy distillation 继续作为提高 q_P 的训练基础；主要算法应做不跨越 Target-sampled Feature 的 position-level 自适应 proposal budget，或通过 target-native/shared-trunk head 减少每次 target_verify_2 的计算；验证和 cache exactness 必须与端到端 tok/s 一起报告。继续单纯增大 full-token draft 或 pair 数没有证据支持。

### 历史 K=2/K=4 Target block 摊销结果的含义

此前记录的 K=2/K=4 测试测量的是 **Target block 前向的并行摊销**，不是完整 speculative decoder 的端到端加速。K=1 的单 pair Target forward 约为 `22.76 ms`；K=2 的 block 总 forward 约为 `23.26 ms`，折算为 `23.26/2=11.63 ms/pair`，因此相对 K=1 的 pair-level 计算加速为：

```text
S_2 = 22.76 / 11.63 = 2 * 22.76 / 23.26 ≈ 1.96x
```

K=4 的 block 折算约为 `6.19 ms/pair`，对应总 block 时间约 `4*6.19=24.76 ms`，因此：

```text
S_4 = 22.76 / 6.19 = 4 * 22.76 / 24.76 ≈ 3.67x
```

这个 3.67x 的含义是：四个 pair 的 Target verification 在一个 batch/block 中并行执行时，相对于四次串行 K=1 Target forward 的理想计算摊销收益。它没有计入 draft 生成、pair acceptance、rejection correction、cache 分支与回滚、sampling、同步和最终输出长度；由于 GaussianGPT 的条件链是 `P_i → Target-sampled F_i → P_{i+1}`，它也不能直接解释为当前 decoder 的端到端 speedup。后续 structured speculative 的端到端结果因此必须单独报告。

## S6 选择性 LM head 与 compact sampling（2026-10-04）

在完成阶段 profiling 后，先实现两个不改变 target policy 的推理优化：

1. Position 行只计算合法的 Position 区间和 EOS，Feature 行只计算合法的 Feature 区间；这是 selective LM head，避免对另一个 token type 的无效词表位置做线性投影。
2. 在采样前把已经由 mask 置为 `-inf` 的无效词表位置物理压缩掉。Position 的 compact support 是 `0..7999` 加 EOS（8001 个 id），Feature 的 compact support 是 `8000..12095`（4096 个 id）。采样得到 compact index 后再映射回全局 token id；residual correction 也在同一 compact support 中执行。

这两个改动都位于已确定 token-type mask 和 temperature/top-p policy 之后。它们不会跳过 `target_verify_2`，也不会改变 Position-only acceptance 的条件链；收益来自减少 LM head 投影和 sampling/normalization 的无效工作。代码和可重复应用脚本位于 `.codex/remote-source/model/gpt/position_selective_speculative.py`、`patches/target_head/apply_selective_lm_head.py` 和 `patches/target_head/apply_compact_sampling.py`。远端源码通过 `python -m py_compile` 检查，运行时 smoke 也通过。

### 端到端 wall-clock

远端 Autodl 的 NVIDIA GeForce RTX 4090（24,564 MiB）上，固定 teacher `checkpoints/gpt_vfront.ckpt`、VQ-VAE `checkpoints/vqvae_vfront.ckpt`、D8 draft、CUDA、`temperature=0.9`、`top_p=0.9`，对 baseline 与 selective 使用相同 seed 和输出上限。下表的 selective 同时包含 selective LM head 与 compact sampling；吞吐是完整生成路径的 tokens/s。

| 生成上限 | seeds | baseline mean | selective mean | speedup | position accepts |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 5/6/7 | 72.8046 | 77.0298 | 1.0580x | 181/192 = 94.27% |
| 1024 | 5/6/7 | 72.0142 | 76.7853 | 1.0663x | 1481/1536 = 96.42% |
| 2048 | 5/6/7 | 73.6560 | 78.6005 | 1.0671x | 2929/3072 = 95.35% |
| 4096 | 5/6 | 69.1468 | 74.9337 | 1.0838x | 3887/4096 = 94.90% |

128-token 的同一 benchmark 在只启用 selective LM head、尚未压缩 sampling support 时为 baseline 72.8052、selective 75.8818 tokens/s，speedup 1.0424x。加入 compact sampling 后 selective 均值为 77.0298 tokens/s，相对 head-only endpoint 再提升约 1.51%，总 speedup 提高约 1.56 个百分点。1024 和 2048 token 的总收益稳定在约 6.6–6.7%；4096 token 的两 seed 均值为 8.38%，但 seed 间为 6.08% 和 10.69%，所以只能作为长序列的初步证据，不能据此声称严格的长度单调趋势。

原始远端输出保存在 `profiling/route_a/compact_sampling_e2e_128.txt`、`compact_sampling_e2e_1024.txt`、`compact_sampling_e2e_2048.txt` 和 `compact_sampling_e2e_4096.txt`。实际 runner 是远端 `/tmp/bench_selective_head*.py` 下的临时 benchmark，后续应将同样的配置迁移到项目的 canonical e2e harness，再补齐 4096 的第三个 seed 和重复运行方差。

另外运行了 4096 上限的 EOS 控制：seed7 中 selective 在 3357 token 结束而 baseline 达到 4096；seed8 中 baseline 在 3223 token 结束而 selective 达到 4096；seed9 中 baseline 在 3767 token 结束而 selective 达到 4096。这三次的输出长度不相同，不能和固定 4096 长度的吞吐直接平均，因此均保留为 EOS/鲁棒性记录，不并入上表。它们说明长序列 benchmark 必须同时报告实际长度，或在 canonical harness 中提供成对的固定长度测量。

### 等价性与边界

使用同一组带 mask 的 synthetic logits，分别对 Position（含 EOS）和 Feature，在 `temperature=0.9`、`top_p=0.9/1.0` 下比较原始全词表 sampling distribution 与 compact support 恢复后的全局分布：四种设置的 `max_abs` 均为 0.0，概率和的差异仅为 float32 舍入（最大约 `6e-8`）。这证明压缩没有改变给定 logits 的 policy 分布。真实模型 validator 还得到 selected logits 的最大绝对误差约 `6.7e-6`/`4.8e-6`，无 mask 外有限值；32-token engine smoke 为 16 对、14 次 Position accept，cache/stream 正常。

这里验证的是数值分布等价与运行时不变量，不是固定 seed 下逐 token 输出完全相同：压缩后的 multinomial 张量形状不同，随机数消耗顺序可能不同。因此端到端结果应解释为同一 target policy 下的统计吞吐比较；若要声称经验联合分布不变，还需在多个 seed 上比较 token-type、Position/Feature 边际和渲染指标。

这组结果支持把 compact sampling 作为低风险工程优化合入实验分支，但它没有改变当前瓶颈判断：target 双 token forward 仍是主要成本，compact sampling 只贡献约 1–2% 的增量吞吐。后续主线仍应优先研究减少 target verification 次数或降低其计算量的 position-level 自适应 budget、F-aware 草稿训练与 target-native/shared-trunk verification。

## S7 Feature-aware draft 与 micro-draft 实验设计（2026-10-04）

基于 profiling 中 `C_D≈6.5–6.7 ms`、`C_{target_verify_2}≈16.8–17.7 ms` 的结果，下一轮实验拆成“提高条件匹配”和“降低草稿成本”两条可独立验证的路径。

| 路线 | 部署时是否提出 Feature | 主要改动 | 首要验收指标 |
| --- | --- | --- | --- |
| Feature-aware auxiliary draft | 否 | 共享 trunk 增加 Target-sampled Feature CE/KL 辅助目标 | `q_P` overlap、Position acceptance、端到端 tok/s |
| n-gram/lookup micro-draft | 否 | 结构化 `[P,F]` key、最长匹配与 backoff，只输出 Position proposal | lookup latency、acceptance、`C_D`、端到端 tok/s |
| n-gram → D8 cascade | 否 | 高置信度查表命中时跳过 D8，否则回退 | 触发比例、总 wall-clock、质量与 exactness |
| 联合 `[P,F]` proposal | 是 | 新的 pair acceptance、Feature correction 和 cache 路径 | 联合分布 exactness、cache、端到端 tok/s |

Feature-aware 训练的第一版应使用 Target rollout 中真实采样的 Feature 作为条件，保留现有 Position-only decoder。micro-draft 的第一版使用 Position-only n-gram，按 `[P_i,F_i]` 或最近两组 pair 建 key，并逐级 backoff；原始 token-id 直接 n-gram 只作为稀疏性对照。所有版本都固定 teacher、VQ-VAE、采样温度、top-p、warmup、seed 和输出长度，并同时记录 `C_D`、lookup/hit latency、`alpha_P`、`sum min(P_T,Q_D)`、rejection correction、token stream/cache invariant 和完整 tokens/s。

micro-draft 的成本门槛可用下式预筛：`C_micro + (1-alpha_micro)C_reject < C_D8 + (1-alpha_D8)C_reject`。按当前 profiling，若 lookup 几乎为零，Position acceptance 约 0.50–0.55 以上就值得进入真实端到端测试；该数值只是成本估计，不能替代 GPU synchronization 后的实测。

### S7.1 下一轮 F-aware 训练矩阵

现有 S3 已经给出第一条证据：在相同 Target/VQ-VAE/解码器条件下，256-step on-policy feature distill 将 1024-token 的 feature overlap 从 `0.3876` 提高到 `0.7037`，Position acceptance 从 `97.27%` 提高到 `98.24%`；但当前 decoder 仍只接受 Position，尚未测联合 pair acceptance。下一轮实验固定 D8 容量和部署 policy，比较：

| 版本 | prefix 来源 | Feature/Logit 解耦 | 目标 |
| --- | --- | --- | --- |
| A | 静态 teacher-forcing | 否 | 现有 S3 baseline |
| B | Target-sampled on-policy | 否 | 测量部署前缀收益 |
| C | Target-sampled on-policy | 否，加入 policy KL | 测量分布逼近 |
| D | Target-sampled on-policy | 是，独立 feature/logit projection | 复现 FSPAD-like partial alignment |

每个版本至少使用 `lambda_F ∈ {0, 0.1, 0.3, 1.0}`、固定 `temperature=0.9`、`top-p=0.9`、warmup=128，并在 seed 5/6/7、1024/2048 tokens 上记录：Feature policy overlap、Position policy overlap、实际 Position acceptance、联合 pair overlap、rejection correction 次数、draft/target wall-clock、cache invariant 和渲染质量。联合 pair overlap 只在独立的 pair sampler 中报告，不把当前 `feature_overlap` 误写成 Feature acceptance。

进入 K=4 Target block 实验的门槛是：D 版本相对 C 同时改善 `q_F` 与 `q_P`，并在小规模联合 sampler 中通过分布级 exactness；如果只改善 Feature overlap 而没有联合 pair 或 wall-clock 收益，停止扩大 Feature 分支，转回 Position-only cascade。

### S7.2 λF=0.3 on-policy 短跑结果（2026-10-04）

本轮先用现有 `s3_onpolicy_feature_distill_train.py` 做低成本权重筛选，训练数据来自 Target 实际采样的 8 条、每条 1024 token 的轨迹；训练 128 step、`lambda_F=0.3`、验证间隔 8。第一次启动因 Lightning 配置 `val_check_interval=64` 大于 8 个训练 batch 而未进入训练，随后改为 8 重跑成功。远端输出目录为 `profiling/s3/faware_lambda03_128b/`，checkpoint 为 `checkpoints/onpolicy_feature_kl_best_epoch15_step128.ckpt`。

验证结果：`val_feature_policy_overlap=0.34996`、`val_feature_raw_overlap=0.38900`、`val_feature_top1_agreement=0.20410`。这表明短跑只产生有限的 Feature 分布移动，不能与此前 256-step on-policy 结果的约 0.70 overlap 相比；当前共享输出头的 Feature KL 仍不足以形成可接受的 Feature proposal。

同一远端、同一 256-token wall-clock 脚本做了两个 seed 的即时对照。seed1：canonical 接受 113/128、`speedup_wall_tokens_per_s=1.0228x`，新 draft 接受 119/128、`1.0440x`；seed2：canonical 接受 120/128、`1.0612x`，新 draft 接受 122/128、`1.0549x`。新 draft 的接受数略高，但 wall-clock 没有稳定提升，两个 seed 的 speedup 均值约为 canonical `1.0420x`、新 draft `1.0494x`，差异仍在单次运行噪声量级，远低于 Target block 的理论摊销收益。Feature alignment diagnostic 在输出 Feature KL 时触发 `None` 格式化异常，不过已经得到 Position raw/policy overlap 0.9278/0.9332；该脚本需要修复后再作为正式诊断入口。

结论：on-policy Feature 蒸馏方向有初步正信号，但当前训练长度和共享 head 结构不足以支撑 Feature 接受。下一轮优先延长有效训练并加入独立 Feature/Logit projection；只有当 Feature policy overlap、joint pair overlap 和 wall-clock 同时改善时，才进入 K>1 的联合接受实验。

### S7.3 Feature shadow acceptance 原型与共享 q_F（2026-10-04）

为把“Feature logits 只是诊断”推进到可测量的 proposal/acceptance 层，远端 `model/gpt/position_selective_speculative.py` 增加了 `feature_shadow` 路径。它在 Position 已接受的条件下，使用同一套 temperature/top-k/top-p policy，从 Draft 的 `q_F` 采样候选，并按 Target/Draft 的 ratio acceptance 规则计算 Feature 接受、pair 接受和 residual correction 统计；默认 Position-only 输出与 Target Feature 采样保持不变，因此这一步是 shadow，而不是新的输出分布。

关键实现是复用必需的 `draft_update_2` forward：输入为已提交的 `[P_i,F_i]`，其第 0 行在因果 mask 下正好是 `q_F(F_i | H_i,P_i)`，第 1 行继续产生下一位置分布。因而 Feature shadow 不再额外执行 `draft_feature_probe`，也不改变 Draft/Target cache 的稳态提交顺序。远端源码通过 `py_compile`，并且没有残留 `q_feature_shadow_logits` 或实际 `draft_feature_probe` 阶段。

使用 `profiling/s3/faware_lambda03_128b/checkpoints/onpolicy_feature_kl_best_epoch15_step128.ckpt`、同一 Target/VQ-VAE、RTX 4090、256 tokens、128 pairs 的端到端运行得到：

| 指标 | 结果 |
| --- | ---: |
| Position acceptance | 119/128 = 0.92969 |
| Feature shadow proposals | 119 |
| Feature shadow accepts | 60/119 = 0.50420 |
| Pair shadow accepts | 60/128 = 0.46875 |
| Feature policy overlap | 0.48695 |
| Mean shadow acceptance probability | 0.49273 |
| baseline / selective throughput | 72.182 / 73.980 tokens/s |
| wall-clock ratio | 1.02490x |

同一运行中 `feature_shadow_acceptance` 的均值约 0.827 ms/event，额外的 `draft_feature_probe` 阶段已消失；但这是单次 256-token 测量，1.0249x 只能作为代码路径和成本回归的初步结果，不能替代多 seed 结论。`scripts/selftest_position_selective_math.py` 也通过：理论 overlap=0.55，经验接受率=0.55172。

随后立即用同一 runner 和同一配置重复一次，接受统计完全一致（Position 119/128、Feature shadow 60/119、pair shadow 60/128），baseline/selective 为 73.248/74.514 tokens/s，wall-clock ratio=1.01728x。两次结果落在 1.017–1.025x，说明共享 q_F 后不会再引入此前的约 5 ms/event 探测开销，但当前端到端增益仍处于单次运行噪声范围。

这一步解决了“没有 Feature acceptance 观测”的问题，同时保留了无损 Position-only 输出。它还没有解决“Feature 被接受后真正写入输出、被拒绝后从 Target residual 修正并回滚”的问题。要进入真实联合路径，必须在 Target verify 前从同一条件前缀提出 `F_i~q_F`，然后分别处理 `(P accepted, F accepted)`、`(P accepted, F rejected)` 和 `P rejected` 三条 cache 分支，并用 `[P,F]` 联合经验分布做 exactness 验证。

### S7.4 真实联合 `[P,F]` proposal/acceptance 原型（2026-10-04）

按照 S7.3 的边界，远端新增实验模块 `model/gpt/joint_feature_speculative.py` 和 runner `profiling/s3/faware_lambda03_128b/run_e2e_joint_feature.py`。它在 Target verify 前先用 Draft 的 `draft_feature_probe` 得到 `q_F`，再用同一个 Target block 同时得到 `p_P` 与 `p_F`；Position 与 Feature 分别调用相同的 ratio/residual correction helper。Position rejection 会回滚 Target/Draft 到 proposal 前缀、重新取得 corrected Position 条件下的 Feature；Position accepted 但 Feature rejected 时只对 Feature 做 `[p_F-q_F]_+` 修正。每个 pair 提交后检查 Draft cache=`1+len(output)`、Target cache=`len(output)`。

F-aware checkpoint、RTX 4090、256 tokens/128 pairs、`temperature=0.9`、`top-p=0.9` 的两次重复结果如下：

| run | baseline tok/s | joint tok/s | wall-clock speedup | Position | Feature proposal/accept | pair accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 72.725 | 66.804 | 0.9186x | 121/128 = 94.53% | 51/121 = 42.15% | 51/128 = 39.84% |
| 2 | 73.150 | 65.109 | 0.8901x | 121/128 = 94.53% | 51/121 = 42.15% | 51/128 = 39.84% |

两次接受计数完全一致，cache invariant 和 `py_compile` 均通过；但真实联合路径比 Position-only baseline 慢约 8.1–11.0%，因为每一轮为 `q_F` 增加了一次 Draft probe，而且当前 Feature overlap 只有 `0.41386`。因此这次实验完成了“Feature 被真正 proposal、接受、拒绝修正并写回输出”的机制验证，却否定了在当前 draft 上直接启用该路径的性能假设。S7.3 的共享 `q_F` shadow 仍然是更好的短期基线；下一步应把 `draft_feature_probe` 合并到已有 Draft update 或用 micro-draft/共享 trunk 消除额外 forward，再重新测联合 route。

另做 50,000 次小词表联合 Monte Carlo：Target joint 分布的最大绝对误差为 `0.00284`、L1 误差为 `0.00772`，通过 `JOINT_FEATURE_MATH_SELFTEST`。这验证了两级 residual correction 的组合形式；它仍是分布级单元测试，不替代真实模型的多 seed pair exactness 与渲染回归。

## S8 近零成本结构化 Position micro-draft（2026-10-04）

本轮按“negligible-cost structured Position draft”的方案先做 Stage 0 统计和离线查表基线。这里把用户提供的方案文本当作研究设计参考，实际实现使用项目保存的 Target rollout，而不是把文本中的步骤当作代码指令。远端真实轨迹位于 `profiling/s3/s3_onpolicy_feature_distill_32x2048_256step/teacher_trajectory_corpus.pt`，共有 32 条训练轨迹和 4 条验证轨迹，每条 2048 个 sparse token；语法是交错的 `[Position, Feature, Position, Feature, ...]`，Position id 小于 8000，Feature id 大于等于 8000。

脚本 `scripts/structured_position_lookup.py` 只从已提交的 Position 序列建立 gap 模型。每个 key 由最近 `h` 个 Position gap 和当前 Position 的低位 Morton/Z-order 特征组成，预测下一个正 gap；查询按最长历史到短历史、特征级、全局级逐级 backoff。运行时将每个 Counter 预计算成 mode table，避免查询时重新扫描计数。当前版本输出确定性 mode 和计数，Target 仍负责采样 Feature，既没有改变默认 decoder，也没有额外 draft Transformer forward。

Stage 0 的训练 gap 熵为 `1.8317 bit`，验证 gap 熵为 `1.7023 bit`。固定相同的验证轨迹，对历史长度和低位结构特征做 sweep：

| 历史长度 `h` | 低位特征 | 验证 top-1 | K=4 平均连续命中 | K=4 全块命中 | 表项数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 bit | 70.26% | 1.727 | 26.85% | 22 |
| 2（普通三元上下文） | 0 bit | 78.15% | 2.139 | 40.07% | 95 |
| 3 | 3 bit | 81.94% | 2.410 | 46.66% | 704 |
| 4 | 1 bit | 84.68% | 2.569 | 53.37% | 728 |
| 5 | 1 bit | **85.04%** | **2.580** | **54.48%** | 1325 |
| 6 | 1 bit | 84.995% | 2.582 | 54.27% | 2166 |

暂定配置为 `h=5, feature_bits=1`：验证集 top-1 为 `3480/4092=0.85044`，K=4 的平均连续命中为 `2.57969`，四个位置全部命中的比例为 `0.54485`，预计算表有 1325 项，Python CPU microbenchmark 的查表均值为 `0.7186 μs/query`。`h=4, feature_bits=1` 的查表约 `0.6905 μs/query` 且 top-1 为 `0.84677`；`h=6` 没有稳定提高 top-1，却显著增加表项，因此不继续扩大历史长度。`h=3, feature_bits=3` 的查表约 `0.6434 μs/query`，但命中率明显低于 h=4/h=5 配置。

这里的 `top-1` 是在真实 Target prefix 上用 mode 预测下一 gap 的 teacher-prefix 命中；K=4 指标是连续 mode 命中代理，不能直接写成 speculative acceptance，也没有使用 `sum_v min(P_T,Q_D)` 的 exact overlap。当前 artifact 还需要把计数归一化成合法的 `Q_D^P`，加入 BOS/边界/EOS 与 token mask，并接入 Target-vs-q 的 residual correction 后，才能测真正的 Position acceptance。查表 benchmark 还是 CPU/Python 小循环，不能代替同步后的 GPU 端到端成本；因此本轮没有声称 3.67x 或任何完整生成 speedup。

需要特别记录一项可重复性修正：早期探索输出的 `83.04%` 结构化 top-1 在按固定脚本、固定 32/4 轨迹重跑时没有复现。本节只采用可重跑的保守数字；早期数字不作为实验结论。

远端 artifact 保存在：

- `profiling/s8_structured_position/h5_b1/structured_zlow3_eval.json`
- `profiling/s8_structured_position/h5_b1/structured_zlow3_model.json`
- 对照配置 `profiling/s8_structured_position/h4_b1/` 和 `profiling/s8_structured_position/`

下一步是写一个独立的 Position micro-draft adapter：保留完整 gap 计数形成归一化 `Q_D^P`，在合法 Position/EOS support 上执行 exact acceptance/residual correction，先通过小词表分布测试和 Target cache 前缀测试，再接入现有 Position-only engine 做 1024/2048/4096 的 fair wall-clock。默认 D8 路径在这些测试通过前保持不变。

### S8.1 归一化 proposal 与 residual correction 前置验证（2026-10-04）

在接入生成引擎前，已将 h=5、低 1 位配置的 gap 计数归一化成合法 Position proposal `Q_D^P`，并在同一验证轨迹上做经验 overlap 估计。训练表在 4 条验证轨迹的 4092 个 Position transition 上得到的平均经验 overlap 为 `0.84520`，真实下一 Position 在 `Q_D^P` 下的平均概率为 `0.78636`。这两个数仍然是 corpus 条件分布代理：验证集的经验 `P` 不是 Target neural logits，因此不能直接称为模型 acceptance。

残差修正单元测试也已通过。对 synthetic `P=[0.1,0.2,0.7]`、`Q=[0.55,0.30,0.15]` 做 100,000 次采样，经验 acceptance=`0.44762`，理论 overlap=`0.45`，输出 L1=`0.00340`、最大绝对误差=`0.00170`。在验证轨迹中选取一个出现 82 次的真实结构化 context，`Q_D^P` 与该 context 的经验 `P` 的理论 overlap=`0.86697`，经验 acceptance=`0.86612`，输出 L1=`0.00324`、最大绝对误差=`0.00121`。这证明 ratio/residual 实现的采样数学正确，但没有证明它已匹配 Target logits。

远端 exactness 脚本为 `scripts/structured_position_exactness.py`，结果 artifact 为 `profiling/s8_structured_position/h5_b1/structured_exactness.json`。下一步仍需把同一 adapter 接入 Position-only engine，测 Target logits 下的真实 `sum_v min(P_T,Q_D)`、cache rollback 和 1024/2048/4096 token wall-clock；在此之前不改变默认路径。

### S8.2 结构化 Position adapter 的首轮端到端验证（2026-10-05）

已将 h=5、低 1 位结构特征的查表 proposal 接到现有 Position-only speculative engine。远端新增 model/gpt/structured_position_adapter.py 和 scripts/run_structured_position_e2e.py；adapter 直接输出合法 Position support 上的 Q_D^P，不运行 Draft Transformer，只通过 store_tokens/advance 维护 Draft cache 的 token/position 不变量。Target 仍采样真实 Feature，Position rejection 仍使用原有 ratio acceptance、residual correction 和 rollback，因此该实验测的是完整 Target 校正路径，而不是离线命中率。

固定 Target/Draft/VQ-VAE、seed=1、RTX 4090、warmup=32、256 sparse tokens（128 pairs）后，得到：

| 路径 | elapsed | 吞吐 | 相对 Target AR |
| --- | ---: | ---: | ---: |
| Target autoregressive baseline | 3594.230 ms | 71.225 tok/s | 1.000x |
| structured Position adapter | 2629.573 ms | 97.354 tok/s | **1.36685x** |

本次共有 128 个 Position proposal，接受 115 个、拒绝 13 个，实测 alpha_P=0.8984375；同一 Target policy 下记录的 Position overlap 为 0.8945407。生成长度为 256 token、128 pair，cache invariant 检查 128 次均通过，未触发 EOS 或异常回滚。adapter 被调用 129 次，端到端 Python lookup 计时均值约 122.6 μs/query；离线纯查表的 0.7186 μs/query 仍不能直接代表 engine 中的 host/device 与 support 映射成本。当前 feature_shadow 和 joint pair acceptance 均关闭，Target Feature 只作为真实输出采样，因而本结果是 Position-only 的实际收益，不是联合 [P,F] 加速结果。

这次运行首次证明：在保持无损 Position correction 和 cache rollback 的前提下，近零成本结构化 draft 可以在真实 Target 验证路径上取得正 wall-clock 收益。1.36685x 不能与早先只计算 Target block 摊销的 3.67x 混用；后者不含 draft、采样、rollback 和 cache 成本。当前结果仍是 256-token、单 seed 的 smoke evidence，且 adapter 由 runner 接管 draft forward，尚未成为默认生产路径。下一步应先把 adapter 接口固化到 engine，再用同一输入和固定长度补跑 1024/2048/4096、多 seed，并单独报告 table lookup、Target verify、rejection rerun 与 sampling/synchronization 成本。

### S8.3 结构化 Position adapter 的长度扩展（2026-10-05）

沿用 S8.2 的 Target/Draft/VQ-VAE、h=5/低 1 位查表、seed=1、temperature/top-p、warmup 和 RTX 4090，只替换固定输出长度，得到以下单 seed fair endpoint：

| 目标 token 数 | Target AR tok/s | structured tok/s | wall-clock speedup | alpha_P | overlap_P | lookup μs/query |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 71.225 | 97.354 | 1.36685x | 0.89844 | 0.89454 | 122.6 |
| 1024 | 73.964 | 99.274 | 1.34220x | 0.89648 | 0.89961 | 146.6 |
| 2048 | 72.058 | 97.137 | 1.34804x | 0.90332 | 0.90455 | 193.8 |
| 4096 | 67.182 | 91.160 | 1.35691x | 0.90820 | 0.90903 | 244.5 |

四个长度都生成了完整目标长度，terminated_on_eos=false；Position proposal/accept/reject 分别为 128/115/13、512/459/53、1024/925/99、2048/1860/188，两侧 cache invariant 检查分别为 128、512、1024、2048 次且均通过。结构化 adapter 的 wall-clock 收益在这组固定 seed 上稳定在 1.34–1.37x，说明收益主要来自移除 Draft Transformer，而不是短序列启动偶然性。lookup 计时随序列变长升到约 244.5 μs/query，仍显著低于一次神经 Draft forward，但它包含 Python/host support 映射，正式实现仍应把表和合法 support 放到 GPU 或 fused kernel 中。

这些是单 seed、固定长度的 scaling evidence，不是多 seed 置信区间，也没有把 Feature 纳入 proposal。下一步应先补 seed 2/3 和不同场景的长度复测，再评估高置信度 table hit → D8 fallback cascade；在多 seed 和场景外测试通过前，不把 1.35x 写成最终模型指标。


### S8.4 多 seed scaling 与结果图人工检查（2026-10-05）

在同一 Target/Draft/VQ-VAE checkpoint、同一场景和同一 h=5/低 1 位结构化 Position policy 下，补跑了 seed=0、1、2 三个 seed，覆盖 256/1024/2048/4096 token，共 12 个有效 endpoint。seed=1 的 256-token 结果在独立 grid 输出目录中重跑，避免早期默认输出文件被覆盖。所有运行返回码为 0，并生成了完整目标长度。

| 目标 token 数 | speedup 均值 ± std | alpha_P 均值 ± std | lookup 均值（μs/query） |
| ---: | ---: | ---: | ---: |
| 256 | 1.39624 ± 0.03090 | 0.93750 ± 0.03984 | 133.43 |
| 1024 | 1.39089 ± 0.03490 | 0.90951 ± 0.01218 | 167.60 |
| 2048 | 1.36011 ± 0.01345 | 0.90527 ± 0.00498 | 206.46 |
| 4096 | 1.36546 ± 0.01149 | 0.91097 ± 0.00294 | 253.32 |

结果图 multiseed_scaling.png 和 PDF 版本将四类信息放在同一张图中：每个 seed 的 speedup 曲线及均值/标准差、Position acceptance、baseline/structured 吞吐、lookup 成本。人工观察得到：

- 三个 seed、四个长度的 speedup 都大于 1，均值保持在 1.36–1.40x；2048 token 有轻微低谷，4096 token 恢复到与 2048 接近的水平，没有看到随长度持续崩溃。
- acceptance 在长序列稳定在约 0.905–0.911。256 token 的均值较高且标准差较大，这是因为每次只有 128 个 proposal，短序列统计方差明显。
- lookup 从约 133 μs/query 增长到 253 μs/query，增幅约 1.90x；这与结构化吞吐随长度下降的方向一致，说明 Python/host 查表、support 映射和同步仍是当前 micro-draft 的主要可优化成本之一。
- 这轮只覆盖同一场景和 checkpoint；它证明的是 seed/长度方向的稳定性，不是跨场景泛化，也没有打开 Feature proposal 或 joint [Position, Feature] acceptance。

远端原始 JSON 位于 profiling/s8_structured_position/grid/，汇总为 profiling/s8_structured_position/multiseed_summary.json；图像为 profiling/s8_structured_position/multiseed_scaling.png 和 .pdf。当前结果支持继续把该 adapter 作为约 1.35x 的 Position-only micro-draft 基线，同时优先迁移 GPU/fused lookup，再做多场景复测。


### S8.5 正式 Draft proposer 接口与双路径回归（2026-10-05）

按计划把 runner 接管的结构化 adapter 固化为 engine 的 Draft proposer 接口。远端新增/修改：

- `model/gpt/draft_proposal.py`：`DraftProposalState`、`PositionProposal`、`NeuralDraftProposal` 和 `StructuredPositionProposal`；
- `model/gpt/position_selective_speculative.py`：engine 支持构造器或 `generate(..., draft_proposer=...)` 注入 proposer。结构化 proposer 不运行 Draft Transformer，只维护 Draft cache 的 BOS/位置 bookkeeping；D8 proposer 保留原始 Draft forward；
- `scripts/run_structured_position_e2e.py`：结构化路径改用正式接口；
- `scripts/selftest_draft_proposer.py`：小词表合约自测。

接口的状态流为 `propose_position(state)` → Target position ratio acceptance/correction → `commit(token)` 或 `rewind(state)` 后再 `commit(token)`。EOS 保持在合法 Position support 中，Position rejection 不改变 Target correction 和 rollback 逻辑；本阶段没有打开 Feature proposal 或 joint `[P,F]` acceptance。

同一远端环境、seed=1、warmup=32、256 tokens（128 pairs）下，正式 structured proposer 的 smoke 为：

| 路径 | elapsed | 吞吐 | Position acceptance |
| --- | ---: | ---: | ---: |
| Target AR | 3632.696 ms | 70.471 tok/s | — |
| structured Position proposer | 2596.227 ms | 98.605 tok/s | 115/128 = 0.89844 |

对应 wall-clock throughput speedup 为 **1.39922x**；13 次 rejection、cache invariant 检查 128/128 通过，输出 256 tokens 且未提前 EOS。查表统计为 128 calls、平均 **122.012 μs/query**，table_count=1325，Position history 长度=128。该结果是正式接口 smoke/regression evidence，不替代已有 seed=0/1/2、256/1024/2048/4096 的多 seed canonical grid，也不构成跨场景结论。

无 proposer 的原 D8 runner 也在同一代码版本回归通过：256 tokens、113/128 Position acceptance、cache checks=128，Target AR 73.783 tok/s，D8 76.834 tok/s，wall-clock speedup=1.04136x。该数字受单次运行噪声影响，结论是旧路径仍可运行并保持原 engine correction/cache 语义。

`python3 -m py_compile` 覆盖 engine、proposer、structured runner 和 selftest；formal proposer contract selftest 覆盖 neural/structured support、EOS、commit、rewind、reset，结果为 PASS。后续按计划进入 GPU tensor/vectorized/fused lookup 前，保留这两条路径作为可比较基线。

### S8.6 GPU tensor/vectorized lookup 对照（2026-10-05）

第三阶段把结构化 Position adapter 的查表实现拆成三个可比较模式，并保留同一张 `h=5, feature_bits=1` 表、同一合法 Position/EOS support 和同一 backoff 规则：

- `python`：原始 Python tuple/dict backoff 与逐 gap support 映射，作为兼容基线；
- `tensor`：整数 context code、预缓存 gap/count tensor 和按 device 的 lazy cache，单 query 用 tensor support mapping；
- `vectorized`：整数 context code 加批量 support mapping，提供 `query_batch`，但当前 engine 仍逐 proposal 调用它；
- `fused`：当前 checkout 没有已注册的 CUDA extension，因此只记录为 unavailable，没有伪造 kernel 结果。

远端脚本为 `scripts/benchmark_structured_lookup_variants.py`，artifact 为 `profiling/s8_structured_position/lookup_variants.json`。512 个固定 mixed-length query 在 CUDA 上比较，所有输出先恢复到相同的 `[N, 8000]` Position 全局 token 形状再做 exactness：`tensor` 和 `vectorized` 相对 `python` 的 `max_abs_diff=0.0`、`sum_abs_diff=0.0`；`vectorized` 的 batch size 1/8/32/128 也全部为 0。

| lookup 模式 | 单 query 均值（μs） | 备注 |
| --- | ---: | --- |
| Python dict | 271.677 | 兼容基线 |
| GPU tensor | 279.352 | 单 query 没有降低调度成本 |
| vectorized | 430.818 | 单 query 包装开销更高 |
| vectorized batch=1/8/32/128 | 443.425 / 120.628 / 83.460 / 75.420（每 query） | 批量增大后才显示收益 |

同一 formal proposer engine 的 256-token smoke 仍然保持 115/128 Position accepted、13 rejects、128/128 cache invariant checks、256 token 输出且没有提前 EOS。每次运行内的 baseline 与 structured 是成对测量，结果如下：

| lookup 模式 | Target AR ms / tok/s | structured ms / tok/s | wall-clock speedup | lookup 均值（μs/query） | 估算 lookup 占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python | 3574.188 / 71.625 | 2618.397 / 97.770 | **1.36503x** | 129.064 | 0.63% |
| tensor | 3645.768 / 70.218 | 2734.517 / 93.618 | **1.33324x** | 534.570 | 2.50% |
| vectorized | 3640.333 / 70.323 | 2736.199 / 93.560 | **1.33043x** | 658.683 | 3.08% |

占比按 `lookup_mean_us × 128 / structured_elapsed_ms` 估算，只表示 adapter 记录的 lookup 均值，不替代完整 stage profiler。三条路径的 acceptance、rollback 数和 cache 结果一致，说明整数编码和 device tensor 迁移没有改变 Target correction 语义；但 tensor/vectorized 没有改善端到端 wall-clock。原因是正式 engine 目前每轮只发出一个 query，vectorized batch API 没有被使用，同时 tensor/device 路径仍承担每 query 的调度与同步开销。当前应继续把 `python` 保留为默认可比路径，下一次优化聚焦 engine 级 batch 调用、host-free support mapping 或注册 fused kernel；不能把 microbenchmark 的 batch 加速外推为已有 E2E 加速。

这一步满足“查表实现必须先过 exactness 和真实 wall-clock 再进入下一阶段”的门槛，但没有完成 fused lookup，也没有改变默认 D8 或 structured proposer 的 correction/cache 接口。后续可以在保留这三种实现的前提下进入跨场景、多 seed 验证；若先做性能优化，应优先让 engine 真正调用 `query_batch`，再重新测查表占比。

### S9 多 seed/长度 Position-only 验证（2026-10-05）

第四阶段完成了 `seed={0,1,2} × max_tokens={256,1024,2048,4096}` 的完整矩阵，共 12 个 structured 单元与 12 个 D8 单元；每个单元都有同配置的 Target AR baseline。固定条件为 `gpt_vfront.ckpt`、`vqvae_vfront.ckpt`、canonical D8、RTX 4090、`temperature=0.9`、`top_p=0.9`、warmup=128、CUDA、Python lookup；结构化表为 `h=5, feature_bits=1, Morton bit=1`。

当前 runner 没有独立 prompt/scene 参数，因此汇总中的 `scene_mode` 明确写为 `eos_seed_trajectory`：三组 seed 是从 EOS 起始的冻结随机轨迹，不是三个独立场景。seed policy 保持 runner 现状：baseline 使用 `seed+10`，route 使用 `seed+11`，所以 route 可能提前 EOS，必须同时报告实际输出长度。

按请求长度聚合（均值 ± seed population std）：

| 请求长度 | structured speedup | structured alpha_P | D8 speedup | D8 alpha_P | D8 提前 EOS | D8 route 长度 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 1.38617 ± 0.06483 | 0.93750 ± 0.03984 | 1.04733 ± 0.02501 | 0.91406 ± 0.02300 | 0/3 | [256] |
| 1024 | 1.37055 ± 0.01901 | 0.90951 ± 0.01218 | 1.02437 ± 0.04618 | 0.87565 ± 0.08221 | 0/3 | [1024] |
| 2048 | 1.33807 ± 0.00462 | 0.90527 ± 0.00498 | 1.02051 ± 0.03819 | 0.88046 ± 0.10284 | 1/3 | [1753, 2048] |
| 4096 | 1.35117 ± 0.00499 | 0.91097 ± 0.00294 | 1.05654 ± 0.03299 | 0.88323 ± 0.10317 | 1/3 | [1753, 4096] |

structured 路径 12/12 个单元 speedup 均大于 1，单元范围 **1.30586–1.46463x**，总体均值 **1.36149x**；Position acceptance 范围 **0.89648–0.99219**，没有随长度持续下降。structured route 在全部 12 个单元生成请求长度，cache invariant 全部通过。D8 总体 speedup 均值 **1.03719x**，最低单元为 seed=2、1024 token 的 **0.96349x**，不能把 D8 写成每个单元都加速。seed=2 的 2048/4096 D8 route 在 1753 token 合法 EOS；这两个单元保留在汇总中，但不是相同输出长度的严格对照。

汇总脚本修正了两个判定边界：D8 的 `terminated_on_eos` 位于 route 的 `stats` 内；EOS 终止时允许最后一次 proposal 使 `proposals=generated_pairs+1`，而 cache checks 必须等于 `generated_pairs`。修正后 12/12 structured 与 12/12 D8 cache invariant 均通过，12/12 配对任务返回码为 0，所有请求长度均满足“达到请求长度或合法 EOS”。

远端 artifacts：`/root/workspace/GaussianGPT/profiling/s9_multiscene/phase4_matrix_summary.json`、`phase4_matrix.csv`、`phase4_speedup.png`、`phase4_acceptance.png`，以及 `/root/workspace/GaussianGPT/logs/s9_multiscene/matrix_runner.log`。现有重建 GIF 继续使用 `/root/workspace/GaussianGPT/profiling/s8_structured_position/visual4096/orbit/gif/scene_000000__idx0000.gif`；由于没有独立场景输入接口，本轮不把它写成跨场景重建结论。该矩阵满足 structured micro-draft 进入 cascade 评估的门槛；跨场景泛化仍需先扩展 runner 的 scene/prompt 输入。


### S10 structured→D8 cascade proposer smoke（2026-10-05）

第五阶段把结构化 adapter 和 canonical D8 包装为正式的 `CascadePositionProposal`，并在 runner 中增加 `STRUCTURED_PROPOSAL_MODE`、`STRUCTURED_CASCADE_METRIC` 与 `STRUCTURED_CASCADE_THRESHOLD`。`top1` 使用结构化表的 top-1 probability 判断置信度；`entropy` 使用 raw Shannon entropy（nats）；低置信度样本回退到 `NeuralDraftProposal`。无论 proposal 来源如何，Target verification、Position ratio acceptance、residual correction、rollback 和 Feature commit 都继续走同一 engine 路径。

固定同一远端环境、RTX 4090、seed=1、warmup=128、256 tokens（128 pairs）、temperature=0.9、top_p=0.9、Python lookup，得到：

| proposer | Target tok/s | route tok/s | wall-clock speedup | Position accepted | D8 fallback | structured fraction | cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| structured | 71.8480 | 95.5714 | **1.3302x** | 115/128 | — | — | 128/128 |
| cascade top1≥0.5 | 72.7500 | 73.7434 | 1.0137x | 115/128 | 0 | 1.0000 | 128/128 |
| cascade top1≥0.9 | 73.4589 | 76.1729 | 1.0369x | 122/128 | 18 | 0.8594 | 128/128 |
| cascade entropy≤1.0 | 73.2191 | 73.8715 | 1.0089x | 115/128 | 4 | 0.9688 | 128/128 |
| D8 | 73.9961 | 75.1585 | 1.0157x | 113/128 | — | — | 128/128 |

所有路径都完成 256 token 输出，cache invariant 均为 128/128。top1=0.9 的 fallback 计数为 18，entropy≤1.0 的 fallback 计数为 4，说明置信度门控确实会选择 D8；但当前 cascade 将 `requires_draft_forward=True`，engine 在每个 proposal 前都先执行 D8 forward，即使随后仍采用 structured proposal。于是 cascade 的端到端吞吐只有 73–76 tok/s，明显低于 structured-only 的 95.57 tok/s；单次 top1=0.9 只略高于本次 D8 route，不能视为稳定收益。

远端 artifact 位于 `/root/workspace/GaussianGPT/profiling/s10_cascade/`，日志位于 `/root/workspace/GaussianGPT/logs/s10_cascade/`。本轮通过的是 proposer 接口、合法 support/EOS、fallback 计数以及 Target/Draft cache 检查；性能门槛未通过，因此不进入 F-aware D8 训练。下一步应先在 proposal 层实现 lazy draft-forward：高置信度结构化 proposal 不支付 D8，只有低置信度样本才触发一次 D8 fallback，并重新测同一组 wall-clock、acceptance 和 cache 指标。


### S11 lazy structured→D8 cascade smoke（2026-10-06）

为避免 eager cascade 在每个 proposal 前都支付 D8 forward，`CascadePositionProposal.requires_draft_forward` 已改为 `False`。engine 收到低置信度结构化 proposal 后才创建临时 D8 cache，对 `[BOS] + 已提交 token` 做一次 replay，重新得到 `q_position_logits`，随后再次调用同一个 cascade proposer；Target verification、ratio acceptance、residual correction、rollback 和 Feature commit 没有分叉。代码通过 `py_compile`，`PYTHONPATH=. python scripts/selftest_draft_proposer.py` 继续为 PASS。

固定 RTX 4090、同一 Target/D8/表、Python lookup、seed=1、warmup=128、256 tokens（128 pairs）、temperature=0.9、top_p=0.9，远端运行：

| lazy cascade | Target tok/s | route tok/s | wall-clock speedup | Position accepted | D8 fallback | structured fraction | cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top1≥0.5 | 73.8136 | 101.4608 | **1.37455x** | 115/128 | 0 | 1.0000 | 128/128 |
| top1≥0.9 | 73.3402 | 76.9404 | 1.04909x | 127/128 | 61 | 0.5234 | 128/128 |

`top1≥0.9` 的 proposer 记录 `score_mean=0.93257`、`score_min=0.61949`、`structured=67`、`neural_fallback=61`；`.5` 记录 `structured=128`、`neural_fallback=0`。这证明 deferred metadata、D8 replay 和再次提案的状态流可达，且没有破坏 Target/Draft cache grammar。`.5` 结果与 structured-only 的约 1.33–1.40x 区间一致；`.9` 虽然比同次 Target AR 略快，但仍远低于 structured-only，且一次完整 replay 加后续 D8 增量更新的成本会随着 fallback 比例上升。因此 lazy cascade 的 correctness gate 通过，稳定 wall-clock gate 仍未通过，不能把 cascade 写成当前默认路线。

远端 artifacts：`/root/workspace/GaussianGPT/profiling/s11_cascade_lazy/cascade.json`、`/root/workspace/GaussianGPT/profiling/s11_cascade_lazy_structured/cascade.json`；日志分别为 `/root/workspace/GaussianGPT/logs/s11_cascade_lazy.log` 和 `/root/workspace/GaussianGPT/logs/s11_cascade_lazy_structured.log`。报告中 route 仍写在 `structured_position` 节点，具体 proposer 统计位于该节点的 `proposal`，cache/stage 统计位于 `stats`。

本轮结论是保留 lazy cascade 作为可复现的 correctness reference，默认实际加速路线仍为 structured Position micro-draft；下一阶段转入 F-aware D8 的独立训练/评估，不接入 joint `[P,F]` acceptance。

### S12 F-aware D8 的 Position policy-KL 接口与 GPU 短跑（2026-10-06）

本轮把 Position policy-KL 加入两条 F-aware D8 训练入口，并保留原有 Position-only 解码流程。远端改动位于：

- `profiling/s3/s3_feature_distill_train.py`：增加 `position_policy_kl_weight` 参数、Position/EOS support 的 KL 计算、重叠度与有效行数指标；
- `profiling/s3/s3_onpolicy_feature_distill_train.py`：增加同名 CLI 参数，并写入模型配置与 `run_config.json`；模型损失复用上一个脚本的实现。

当权重为正时，训练目标使用 Position ID `0..7999` 与 EOS `12096` 的 compact support。Teacher 与 Draft 从各自同一次 forward 的 logits 取出相同 support，按 `temperature=0.9`、`top_p=0.9` 生成 policy，再计算 `F.kl_div`；总损失增加 `lambda_P * position_policy_kl`。若目标序列出现 support 外的 Position ID，训练会直接报错；权重为零时保持旧训练行为。新增日志为 `distill/{stage}_position_policy_kl`、`distill/{stage}_position_policy_overlap` 和 `distill/{stage}_position_rows`。Target verification、residual correction、cache rollback 和 Feature 采样没有改动，也没有打开联合 `[P,F]` acceptance。

远端执行了 256-token、2 条训练轨迹、1 条验证轨迹、2 step 的 RTX 4090 GPU 短跑，配置为 `lambda_F=0.3`、`lambda_P=0.3`、seed=2401、bf16 mixed。结果目录为 `/root/workspace/GaussianGPT/profiling/s12_faware_matrix/onpolicy_lambda03_policykl_2step/`，日志为 `/root/workspace/GaussianGPT/logs/s12_faware_matrix/onpolicy_lambda03_policykl_2step.log`。

| 指标 | train | validation |
| --- | ---: | ---: |
| Position policy KL | 0.27861869 | 0.70167381 |
| Position policy overlap | 0.97100383 | 0.86094409 |
| Position rows | 129 | 129 |
| total loss | 2.57889867 | 4.93278122 |

`run_config.json` 记录的两个 KL 权重均为 `0.3`，`global_step=2`，`summary.json` 保存了上述指标。两条脚本均通过 `python -m py_compile`，两个 CLI help 都能看到 `--position-policy-kl-weight`。这次短跑只确认 support、损失合成、指标写入和 GPU 前向可运行，不能用来声称 Position acceptance、wall-clock 或重建质量已经改善。

下一步仍是按 `lambda_F` 与 `lambda_P` 的矩阵运行完整训练，再比较 Position acceptance、Feature shadow acceptance、Draft 成本和端到端吞吐；独立 Feature/logit projection 与真实联合 `[P,F]` 路径继续保持在后续阶段。


### S12.1 F-aware D8 训练矩阵（2026-10-06）

在 S12 接口短跑通过后，使用同一 RTX 4090 和同一数据轨迹完成七个远端短训练，用于筛选 `lambda_F` 与 `lambda_P` 的下一轮候选。固定条件为：

- Teacher：`/root/workspace/GaussianGPT/checkpoints/gpt_vfront.ckpt`；VQ-VAE：`/root/workspace/GaussianGPT/checkpoints/vqvae_vfront.ckpt`；student：`/root/workspace/GaussianGPT/profiling/s2/d8_canonical_38ep/logs/s2_d8_canonical_38ep/lightning_logs/version_0/checkpoints/val_loss_monitor_epoch_37_step_50502.ckpt`；
- `trajectory_length=256`，训练轨迹 2 条，验证轨迹 1 条，`seed=2401`；
- `temperature=0.9`，`top_p=0.9`，`precision=bf16-mixed`，`lr_scale=0.01`，`lr_final_frac=0.1`；
- 五个 8-step 配置和两个 16-step 配置，均由 on-policy 入口运行。

远端汇总文件为 `/root/workspace/GaussianGPT/profiling/s12_faware_matrix/matrix_summary.json`。七个结果目录都位于该目录下，名称为 `onpolicy_lambdaF0_lambdaP0_8step`、`onpolicy_lambdaF01_lambdaP03_8step`、`onpolicy_lambdaF03_lambdaP0_8step`、`onpolicy_lambdaF03_lambdaP03_8step`、`onpolicy_lambdaF10_lambdaP03_8step`、`onpolicy_lambdaF0_lambdaP0_16step` 和 `onpolicy_lambdaF03_lambdaP03_16step`。每个目录保留 `run_config.json`、`summary.json` 和训练日志。

| 配置目录 | lambda_F | lambda_P | global_step | val total | val Feature KL | val Feature policy overlap | val Position KL | val Position policy overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F0/P0/8 | 0.0 | 0.0 | 8 | 4.03948 | 2.24902 | 0.21318 | — | — |
| F0.1/P0.3/8 | 0.1 | 0.3 | 8 | 4.47679 | 2.24791 | 0.21339 | 0.70129 | 0.86148 |
| F0.3/P0/8 | 0.3 | 0.0 | 8 | 4.71337 | 2.24661 | 0.21340 | — | — |
| F0.3/P0.3/8 | 0.3 | 0.3 | 8 | 4.92208 | 2.24498 | 0.21354 | 0.70097 | 0.86142 |
| F1/P0.3/8 | 1.0 | 0.3 | 8 | 6.49385 | 2.24545 | 0.21376 | 0.70036 | 0.86169 |
| F0/P0/16 | 0.0 | 0.0 | 16 | 4.03512 | 2.24030 | 0.21569 | — | — |
| F0.3/P0.3/16 | 0.3 | 0.3 | 16 | 4.90078 | 2.23276 | 0.21608 | 0.66137 | 0.86158 |

16 step 对照中，`lambda_F=0.3, lambda_P=0.3` 的 Feature KL 从 2.24030 降至 2.23276，Feature policy overlap 从 0.21569 升至 0.21608；Position policy overlap 为 0.86158。总损失从 4.03512 升至 4.90078，说明当前收益主要体现在训练分布指标，仍需用解码测量确认是否值得付出额外损失。8 step 结果给出同方向但幅度很小的 Feature 指标变化；`lambda_F=1.0` 的总损失明显更高，Feature KL 没有带来相称改善。

因此当前把 `lambda_F=0.3, lambda_P=0.3` 保留为下一轮研究候选，把 `lambda_F=1.0` 暂不列为首选。矩阵尚未测量 decoder acceptance、Draft forward 时间、端到端 wall-clock、缓存状态或重建 GIF，Feature policy overlap 也不等同于解码接受率。远端 checkpoint 和训练产物没有同步到本地，也没有提交到本地 Git。

### S12.2 F-aware checkpoint 解码 shadow smoke（2026-10-06）

为把 S12.1 的 16 step 候选放回实际解码器，使用同一个评估入口分别运行 `lambda_F=0.3, lambda_P=0.3` 候选和 `lambda_F=0, lambda_P=0` 对照。评估脚本为 `/root/workspace/GaussianGPT/profiling/s3/faware_lambda03_128b/run_e2e_joint_shadow_faware.py`，两次运行都加载成功。

固定条件为 CUDA、`MAX_TOKENS=256`、`WARMUP=32`、`seed=1`、`temperature=0.9`、`top_p=0.9`，并打开 `feature_shadow=True`。候选 checkpoint 为 `/root/workspace/GaussianGPT/profiling/s12_faware_matrix/onpolicy_lambdaF03_lambdaP03_16step/checkpoints/last.ckpt`，对照 checkpoint 为 `/root/workspace/GaussianGPT/profiling/s12_faware_matrix/onpolicy_lambdaF0_lambdaP0_16step/checkpoints/last.ckpt`。汇总文件为 `/root/workspace/GaussianGPT/profiling/s12_faware_matrix/decoder_shadow_compare.json`。

| 路径 | Target AR 时间（ms） | route 时间（ms） | route 吞吐（token/s） | 相对吞吐 | Position 接受 | Feature shadow 接受 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F0.3/P0.3/16 候选 | 3690.2231 | 3612.0473 | 70.8739 | 1.02164x | 113/128 | 54/113 |
| F0/P0/16 对照 | 3630.4164 | 3633.8838 | 70.4480 | 0.99905x | 113/128 | 54/113 |

两条路径都生成 256 个 token，均未提前 EOS；候选 route 在这一次短跑中快约 0.6%，但 Position 接受和 Feature shadow 接受完全相同。报告中的 Feature shadow 只用于诊断，仍然没有启用真实联合 `[P,F]` 接受。该结果是单场景、单 seed、单长度的随机样本，不能据此判断 F-aware 训练已经带来稳定收益。此次短跑也没有生成新的重建 GIF。

下一步应在候选与对照上补齐多 seed、1024/2048/4096 token 和至少一个额外场景的解码测量，同时记录 Draft forward、Target verification、cache 回退和实际输出长度；只有 Feature 接受与端到端吞吐同时出现可重复改善，才重新评估联合 `[P,F]` 路径。


### S12.3 F-aware checkpoint 多 seed/长度解码矩阵（2026-10-06）

在 S12.2 的 256-token smoke 后，把 `lambda_F=0.3, lambda_P=0.3` 的 16 step 候选和 `lambda_F=0, lambda_P=0` 的 16 step 对照放入同一解码入口，覆盖 `seed={0,1,2}` 与请求长度 `{256,1024,2048,4096}`，共 24 个 JSON。运行使用 CUDA、`temperature=0.9`、`top_p=0.9`、`warmup=min(128, length/2)`，并打开 `feature_shadow=True`。原始文件在 `/root/workspace/GaussianGPT/profiling/s12_faware_matrix/decoder_grid/`，聚合文件为 `/root/workspace/GaussianGPT/profiling/s12_faware_matrix/decoder_grid_summary.json`。

按请求长度聚合如下；吞吐和 speedup 为三个 seed 的均值，括号内为 seed population std；`alpha_P` 是 Position 接受比例，`alpha_F^shadow` 只表示旁路统计。

| 请求长度 | F0.3/P0.3 route token/s | F0.3/P0.3 speedup | F0.3/P0.3 alpha_P | F0.3/P0.3 alpha_F^shadow | F0/P0 route token/s | F0/P0 speedup | F0/P0 alpha_P | F0/P0 alpha_F^shadow |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 72.7661 | 1.01668 ± 0.01651 | 0.91406 | 0.36937 | 72.5848 | 1.02119 ± 0.02751 | 0.91406 | 0.36654 |
| 1024 | 73.3488 | 1.03987 ± 0.01225 | 0.95898 | 0.39262 | 74.5420 | 1.04518 ± 0.02746 | 0.95964 | 0.39307 |
| 2048 | 74.5051 | 1.03951 ± 0.01270 | 0.96973 | 0.40466 | 74.5748 | 1.03844 ± 0.01633 | 0.96940 | 0.40103 |
| 4096 | 73.6743 | 1.09685 ± 0.02058 | 0.97579 | 0.42046 | 73.5848 | 1.09708 ± 0.03610 | 0.97755 | 0.40416 |

Position 接受几乎相同；候选的 Feature shadow 在 2048 和 4096 上略高，但仍没有进入真实联合接受。候选相对对照的逐 seed speedup 差值在 `-0.03256` 到 `+0.01845` 之间，route token/s 差值在 `-2.5217` 到 `+2.3342` 之间，方向随 seed 和长度变化，因此没有可复现的端到端收益。

4096 请求的实际 `generated_tokens` 按 JSON 原值保留：候选为 `3319/3731/2339`，对照为 `3319/4096/2339`（seed 0/1/2）。这些行的 EOS 字段均未置为 true，部分字段为空；本轮不把长度不足归因于 EOS，也不把它们当作等长吞吐对照。所有记录都写入了 `cache_check_count` 和 `target_cache_rewind`：256/1024/2048 的 cache 计数均值为 128/512/1024，4096 的候选和对照均值分别为 1564.33 和 1625.33；本轮没有逐条重新验证全部 cache invariant。

这组结果只支持继续保留 F-aware checkpoint 作为研究对照。Feature shadow 是旁路诊断，未改变 Target correction、Feature 采样或 rollback 流程，也没有启用真实 joint `[P,F]` acceptance。当前不进入联合路径，且本轮没有新增重建 GIF 或额外场景。

### S13.1 Draft 自生成轨迹的联合 Position/Feature 蒸馏（2026-10-07）

本轮从 canonical D8 出发，复制 S3 on-policy 训练入口到远端 `profiling/s13_draft_prefix/s13_draft_prefix_train.py`，将轨迹生成者改为当前 Draft。Target 仍提供同一 Draft 前缀上的软目标。首轮生成 8 条、每条 512 token 的训练轨迹和 2 条验证轨迹，使用 `temperature=0.9`、`top_p=0.9`、seed=2401、bf16 mixed；Feature KL 与 Position policy KL 权重均为 0.3，训练 128 步。

首轮最优 checkpoint 为 `/root/workspace/GaussianGPT/profiling/s13_draft_prefix/run_d8_draft_rollout_128step/checkpoints/onpolicy_feature_kl_best_epoch15_step128.ckpt`。验证集 Feature KL 为 0.65265、Feature policy overlap 为 0.58502，Position policy KL 为 0.28940、Position policy overlap 为 0.97421。训练语料保存在 `profiling/s13_draft_prefix/run_d8_draft_rollout_128step/draft_trajectory_corpus.pt`，配置与验证汇总保存在同一目录。

随后用同一评测入口对照 canonical D8 与该 checkpoint。固定 Target、VQ-VAE、CUDA、256-token 上限、32-token warmup、`temperature=0.9`、`top_p=0.9`，seed 为 0、1、2；每个 seed 的 D8 与候选使用相同的随机种子偏移规则。远端 JSON 位于 `/root/workspace/GaussianGPT/profiling/s13_draft_prefix/e2e/`，名称为 `d8_seed{0,2}_e2e_shadow.json`、`d8_e2e_shadow.json`，以及对应的 `candidate_seed{0,2}_e2e_shadow.json`、`candidate_e2e_shadow.json`。

| seed | D8 Position 接受 | 候选 Position 接受 | D8 Feature shadow | 候选 Feature shadow | D8 speedup | 候选 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 118/128 = 0.9219 | 124/128 = 0.9688 | 36/118 = 0.3051 | 43/124 = 0.3468 | 1.0209x | 1.0358x |
| 1 | 113/128 = 0.8828 | 124/128 = 0.9688 | 56/113 = 0.4956 | 41/124 = 0.3306 | 1.0057x | 1.0608x |
| 2 | 120/128 = 0.9375 | 118/128 = 0.9219 | 72/120 = 0.6000 | 39/118 = 0.3305 | 1.0105x | 1.0129x |

三个 seed 合并后，Position 接受由 D8 的 351/384 = 0.9141 升至候选的 366/384 = 0.9531；Feature shadow 由 164/351 = 0.4672 降至 123/366 = 0.3361。合并后的 Position 接受改善，Feature shadow 比例下降；逐 seed 看，seed0 的 Feature shadow 从 36/118 升至 43/124，seed1 和 seed2 下降。speedup 单次变化范围较大，不足以据此判断稳定的端到端增益。

本评测的 Feature shadow 只作旁路统计；现有解码器仍由 Target 采样 Feature，没有提交 Draft Feature，因此这些数字不能称为真实 Feature 接受率。首轮损失虽同时监督 Position 与 Feature，Position/Feature 在真实解码上下文中的表现仍未同时改善。

据此启动 S13b：从上述 S13a 最佳 checkpoint 继续，使用该 Draft 重新生成 16 条训练轨迹和 4 条验证轨迹，每条 1024 token；Feature KL 权重设为 0.6，Position policy KL 权重保持 0.3，训练上限 512 步，seed=2402，bf16 mixed。远端运行摘要中的 `student_checkpoint` 指向 S13a checkpoint；轨迹由该权重生成。配置摘要里描述轨迹来源的文字未同步，本记录按权重路径和训练语料实际来源书写。

S13b 完成 512 步，最佳 checkpoint 选择指标为验证 Feature KL，记录值为 0.59038，选中 step128：`/root/workspace/GaussianGPT/profiling/s13_draft_prefix/run_candidate_rollout_fkl06_512step/checkpoints/onpolicy_feature_kl_best_epoch07_step128.ckpt`。训练配置、语料、日志、汇总分别保存在同一实验目录下的 `run_config.json`、`draft_trajectory_corpus.pt`、`logs/train.log`、`summary.json`。

用该最佳 checkpoint 沿用相同的 Target、VQ-VAE、CUDA、256-token 上限、32-token warmup、`temperature=0.9`、`top_p=0.9`，在 seed 0、1、2 上各测一次。结果写入 `/root/workspace/GaussianGPT/profiling/s13_draft_prefix/e2e/candidate_s13b_seed{0,1,2}_e2e_shadow.json`。

| seed | Position 接受 | Feature 影子接受 | speedup | 路由 tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 124/128 = 0.9688 | 49/124 = 0.3952 | 1.0293x | 76.23 |
| 1 | 110/128 = 0.8594 | 38/110 = 0.3455 | 0.9850x | 71.25 |
| 2 | 125/128 = 0.9766 | 36/125 = 0.2880 | 1.0501x | 77.71 |

三种子合计，S13b 的 Position 接受为 359/384 = 0.9349，处于 D8 的 0.9141 与 S13a 的 0.9531 之间；Feature 影子接受为 123/359 = 0.3426，略高于 S13a 的 0.3361，仍低于 D8 的 0.4672。三个 speedup 从 0.9850x 到 1.0501x，波动较大，当前数据不能证明稳定的吞吐提升。

判断：提高 Feature KL 权重并增加训练轨迹长度后，Position 与 Feature 的影子指标仍未同时超过 D8。S13a/b 的主要模型选择依据是验证集 Feature KL，不能代表联合路径的实测收益。下一步需要将真实 P/F 接受、连续接受长度和完整吞吐纳入权重选择。

### S13.2 S13b 权重的真实联合 P/F 评测（2026-10-07）

对 S13b 最佳权重运行远端已有的 `model/gpt/joint_feature_speculative.py` 中 `generate_joint_feature`。该路径在 Target 验证前同时提出 P 与 F，按各自分布执行接受与 residual correction；当 Feature 被拒绝时使用 Target residual 样本继续生成。统计结构仍保留 `feature_shadow_accepts` 等旧字段名，但此处记录的是实际联合路径中的接受计数。

固定 Target `checkpoints/gpt_vfront.ckpt`、CUDA、256 个生成 token、32 个 warmup token、`temperature=0.9`、`top_p=0.9`，seed 为 0、1、2。Draft 为 S13b step128 权重：`/root/workspace/GaussianGPT/profiling/s13_draft_prefix/run_candidate_rollout_fkl06_512step/checkpoints/onpolicy_feature_kl_best_epoch07_step128.ckpt`。每个 seed 运行一次：

| seed | Position 接受 | Feature 接受（P 已接受时） | P/F 成对接受 | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 116/128 = 0.9063 | 49/116 = 0.4224 | 49/128 = 0.3828 | 0.8929x |
| 1 | 126/128 = 0.9844 | 56/126 = 0.4444 | 56/128 = 0.4375 | 0.9206x |
| 2 | 122/128 = 0.9531 | 49/122 = 0.4016 | 49/128 = 0.3828 | 0.9177x |

合并计数为 Position `364/384 = 0.9479`、Feature `154/364 = 0.4231`、成对 `154/384 = 0.4010`。三次 speedup 均低于 1，均值约 `0.9104x`；平均吞吐约为联合路径 `65.3 tok/s`、纯 Target `71.8 tok/s`。结果文件位于远端 `/root/workspace/GaussianGPT/profiling/s13_draft_prefix/e2e/joint_s13b_seed{0,1,2}.json`，运行脚本和日志在同目录下。

结论：以“提升精确联合 P/F 解码吞吐”为目标，S13b 当前效果不够好。它的 P 接受已较高，主要短板是条件 F 接受约 42%，且提出 F 带来的计算成本超过接受收益。相较 S7.4 的旧联合结果（P `121/128`、F `51/121`、pair `51/128`），这次三种子合计指标接近；小样本不支持 S13b 带来可复现的联合性能提升。这个判断针对解码速度，不代表本轮评估过场景重建质量。

因此不继续按 S13b 配方单纯增加步数。下一轮扩展已测结构化 P-only 路线到多场景，保持 F 由 Target 采样。若结果满足实际需求，就沿用现有路线，不启动新训练；若仍有吞吐缺口，先分析原因，只有确认神经草稿能解决该问题且有净收益时，才考虑加入拒绝状态回放、按 P/F 难度选择样本的训练。方法细节见 [联合 P/F 草稿训练计划第 17 节](JOINT_DRAFT_TRAINING_PLAN.md#17-s13-结果后的-s14-更新方案参考研究报告)。


## 2026-10-20 N2空间embedding CPU实验已完成

8个3585参数模型、1200次更新及480条已有记录的留源筛选完成，独立逐行复算通过。Tune中空间模型相对旧q提高0.879037个百分点，相对最佳对照提高0.707884个百分点，一个来源略降，未达预设继续条件；Hold普通重合度旧q36.878741%、空间37.672828%、固定简单混合38.198092%。停止当前模型，GPU=0，无在线接受率或新加速结论。真实原始输入、全部权重/q/训练记录、技术失败记录及最终复算见docs/project/artifacts/spatial_embedding_20261020/00_READ_ME_FIRST.md。实际源码在patches/spatial_embedding_20261020，生产及原权重保留。不要重复拟合或自动扩大，后续提交状态以该目录WEB_SUBMISSION.md为准。



## 2026-10-20 N2 S3真实GPU速度比较

用户要求补测CPU空间候选的净速度。现有S3 seed29与原Ranker+step12000 MLP、同轮原Target在RTX4090上完成180次解码，10开发来源×64/256前缀×三种子×三方法，FP32最高精度、TF32关闭、temperature/top-p均0.9、最多512新token。无训练，所有空间处理完整计时。

旧PF/S3/Target合并吞吐分别123.7590/119.3687/73.4555 token/s。来源等权加速比分别1.681580/1.635601/1.0000；S3相对旧PF为0.972657 [0.952074,0.993289]。共同满512的58组覆盖10来源，S3/旧PF为0.961998 [0.945646,0.977927]。Feature接受率从3486/10097=34.5251%到3747/10462=35.8153%，仍未抵消额外成本。旧PF两次自然EOS于75/51token，其他输出均满512；等长分析保持较慢结论。

本轮936.744秒完成，480行CUDA概率、两长度零残差/非零模型重放与180实际输出独立复算通过；原生产12份源码逐字节一致，权重保留。来源已用于开发，没有独立来源、完整自然EOS或三维质量结论，不达2倍目标。保留原PF，停止扩大当前S3。资料见[完整结果](artifacts/spatial_online_20261020/00_READ_ME_FIRST.md)，真实远端/root/autodl-tmp/gaussiangpt_spatial_online_20261020/run01。

## 2026-10-21 位置条件双阶段状态转移止于G2

按用户任务书执行G0/G1/G2，2400条真实条件、4800次B/C更新完成。C相对旧12k MLP的Val overlap仅+0.008281个百分点，相对等成本B为+1.284795个百分点，两个Val来源下降，未满足继续标准。提案增量0.465364ms；没有运行G3–G5，无新完整解码速度结论。教师概率、零残差重放和保存权重CPU验证通过。原生产与权重保留。完整设置、逐条数据与限制见artifacts/position_transition_20261020/DECISION.md，网页提交状态见同目录WEB_SUBMISSION.md。


## 2026-10-21 Ranker+MLP执行优化：Attention微测完成

用户授权当前对话制定方案，使用gpt-6-luna/max持续研究真实2×。本轮先测已有4090上FP32最高精度、TF32关闭的Attention实现。默认SDPA、MATH、显式BMM各覆盖ctx64/256/768/2048与T1/2/4；普通执行和CUDA Graph各12形状180条计时，Q/K归一化及KV转置布局沿用实际形状。相对CPU FP64最大概率/输出误差2.25e-7/8.94e-7。

普通T4下短ctx64/256的显式BMM约89微秒，慢于默认约57微秒；ctx2048显式约89微秒、默认约279微秒。图内T4在ctx64/256/768/2048的显式实现为9.60/11.95/17.42/30.26微秒，默认22.61/49.20/110.98/274.48微秒。事件区间可能含主机提交间隙，结果仅描述固定张量的局部Attention。

完整Target Graph探针与四模式解码入口已在本地准备，尚未运行真实模型GPU；SSH连接约01:26失效，现有密钥认证被拒，已请求用户恢复连接或告知已有凭据位置。不得把局部算子倍率写成完整解码倍率。生产源码、原Target/VQ、Ranker和12k权重均保留。本轮资料见artifacts/pf2x_breakthrough_20261021/PLAN.md与EXECUTION_STATUS.md，远端目录/root/autodl-tmp/gaussiangpt_pf2x_breakthrough_20261021。

## 2026-10-21 Ranker+MLP执行优化：独立来源完整解码确认

在30个独立场景、prefix 64/256、每组1个seed上完成四模式同轮比较，共60个配对、240次解码；每次最多生成512个新token。来源等权速度比为：候选PF/原Target `4.519564 [4.228452, 4.720021]`，候选PF/原PF `2.613919 [2.506082, 2.672991]`，候选PF/同优化Target `1.442534 [1.372647, 1.495168]`。前两项包含Attention执行路径与CUDA Graph收益；相同执行路径下PF的增益为1.44倍，当前证据不支持将超过2倍的全部增益归给PF算法。

独立审计确认240次记录、60组配对与30个场景齐全；原/候选Target及原/候选PF各自60组输出和计数逐项相同。权重在计时矩阵前加载；候选运行时建立、KV缓存分配及首次图创建都计入对应的外层解码时长，另有32-token预热运行不计入正式矩阵。该矩阵限于512-token预算和单seed。明细见[汇总](artifacts/pf2x_breakthrough_20261021/independent_confirm_run01/summary.json)、[独立审计](artifacts/pf2x_breakthrough_20261021/independent_confirm_run01/independent_audit.json)与[数值验证](artifacts/pf2x_breakthrough_20261021/independent_confirm_run01/verification.json)。

自然EOS补测覆盖3个独立场景、3组配对、四模式共12次；12次均由EOS结束且未触及16000-token上限。来源等权速度比为候选PF/原Target `3.630147 [2.791982, 4.246229]`、候选PF/原PF `2.209333 [1.987015, 2.373086]`、候选PF/同优化Target `1.273811 [1.019382, 1.443080]`。8项审计全部通过；原/候选Target及原/候选PF分别3/3组输出与计数相同。样本仅3个场景，区间较宽，需按小样本解读。详见[natural EOS汇总](artifacts/pf2x_breakthrough_20261021/natural_eos_run01/summary.json)与[审计](artifacts/pf2x_breakthrough_20261021/natural_eos_run01/independent_audit.json)。

单chunk三维有效性样例来自候选PF的自然EOS输出：153个Position与153个Feature token解码为11,286个Gaussian；六个`384×384`视图的RGB与alpha均有限并已保存，alpha coverage为`0.050615`。`view_00`有非空内容，但结构较稀疏；这仅证明该样例可完成解码和渲染，不构成重建质量提升证据。旧`full3d_validity_run01`失败记录中的1:1数量假设仍原样保留为历史错误；`run02`确认此VQ路径含三级上采样、总因子8，Gaussian数不能按Position数作固定1:1推断。详见[run02元数据](artifacts/pf2x_breakthrough_20261021/full3d_validity_run02/metadata.json)、[Gaussian文件](artifacts/pf2x_breakthrough_20261021/full3d_validity_run02/gaussians.npz)及[视图0](artifacts/pf2x_breakthrough_20261021/full3d_validity_run02/views/view_00.png)。


## 2026-10-21 Feature W2 完整解码已完成

两轮各72次，3开发场景×64/256前缀×三种子×四臂，FP32 highest、TF32关闭、.9/.9、每次512token，全部满预算。最终批量RoPE版MLP/W1/W2相对原Target为4.708736/4.966189/4.769174倍；W2/MLP=1.012835 [0.987925,1.036427]，W2/W1=0.960329 [0.938755,0.988246]。W2没有可靠净提速证据，停止扩宽，未运行W4。W1/MLP=1.054675 [1.048754,1.062945]，收益来自执行实现；各轮18/18输出与共有计数一致。

Feature任一候选接受率35.74297%→42.88544%，Target调用3716→3665只减少1.37%，输入token16901→23024增加36.23%。最终1742项语义数值对照与12项CPU概率审计通过，440项逐元素allclose诊断保留；两轮72/72跨轮输出和计数相同。没有新增独立场景、完整自然EOS或三维质量评价。源码、原记录、复算与限制见[本轮结果](artifacts/feature_tree_20261021/RESULTS.md)。


## 2026-10-21 W1 端到端计时完成

同一 W1 源码增加VQ、六视角渲染、Gaussian NPZ与PNG保存，42/42次完成。固定512组复用原18输入，来源等权总时长倍率4.334555× [3.899431,4.576120]，同轮token生成倍率4.874125×；端到端累计秒为原Target134.007141、W1 31.594370，总时长商4.241488。另三个测试场景自然EOS六次全部完成，来源等权倍率4.869038× [3.361697,6.610150]，累计135.913043→26.616866秒，总时长商5.106275；自然组输出8969→6157token，token吞吐倍率3.559610，完整等待时间包含长度差异。权重加载单列，未测输入编码，首次运行时与图创建计时，随后复用。42份NPZ和252张PNG实物读取通过；固定两臂各18/18、自然各3/3与旧轮输出一致，Luna/max独立复算一致。仅单局部chunk与三个来源，未评价质量等价。详见[端到端结果](artifacts/w1_e2e_20261021/RESULTS.md)。此前旧自然EOS记录的实际预算为16000，已据原plan.json同步文字。
