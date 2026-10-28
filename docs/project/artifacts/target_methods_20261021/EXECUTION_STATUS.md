# 当前状态：主比较与LFQ数值复核结束

11配置主比较220次、原/优化配对132次及LFQ纯BOS18次均已结束。MLP相对同优化Target1.450584×，其他方法没有反超，2×未达。主比较与配对复算通过；LFQ BOS重放未一致，追加同历史数值复核也测出概率支持集变化，因此不排名。最后GPU2MiB/0%，原生产及权重保留。以下保留实际过程记录；最终依据见末尾“最终状态”和RESULTS.md。

## 历史过程记录

新用户任务是在同样优化后的Target上重新测EAGLE、Medusa及其他已有投机方法。已确认同一远端主工程、Python3.10.21/Torch2.8.0+cu129/4090，无已有GPU任务；原源码、权重及Jupyter/TensorBoard保留。三个原Luna/max子代理分别开展方法清单、decoder适配和动态图泛化。当前尚无本轮新速度结果。方案及计时范围见PLAN.md。


首批计划纳入12个不同配置：Target、Ranker+MLP、Ranker-only、EAGLE Feature γ3/γ5、EAGLE Draft KV复用γ3、EAGLE Joint γ3、Medusa lowrank、Hydra conditional、PF-D8、D8 Joint、LFQ scratch PF。是否进入实测以实际加载及精确decoder可运行为准。EAGLE原return_hidden为最终norm后的状态，MLP与Medusa旧hook仍需保留norm前状态；运行实现正扩展T1至T8、完整logits和两种hidden接口。主比较单独保留每方法runtime，避免其他方法代付图创建。


运行接口数值验证run02已实际通过：T1/T2/T3/T4/T6、typed与full logits、return_hidden后norm状态、pre-norm hook及512/1024跨桶，共7次图前向、6个图；唯一普通前向为长prefill。最大policy TV1.33086e-6。截短到511后保持Position3867不变，将真实Feature8435换为8436，原/图状态相符。run01保留，仅其扰动样例把输入Position换为Feature，不能作为合法换Feature证据；runtime本身未因此改变。后续正式比较只优化Target的图内Attention，不全局替换Attention函数，以保留D8等Draft原执行。


参考入口烟测reference_smoke_run01完成12/12次：Target、PF-MLP、Ranker-only，单场景64/256前缀、32token预算、普通/优化两执行，8项独立复算通过。短输出首次图创建成本占主导，此轮只验证集成，不用它排名。现已启动all_methods_smoke_run01：11法、同前缀与预算，共44次；22次16token独立预热的old/new输出与计数已全部一致。Medusa/Hydra和LFQ旧函数把自然EOS记在stats但不写输出列表，新adapter仅在真实终止时把已生成EOS补到返回suffix，并保留raw_body_tokens/raw_eos_count；采样循环未改。LFQ另作纯BOS三方法补测，不混入真实前缀主比较。


11法短输出验证all_methods_smoke_run01已完成44/44，结果已复制本机，独立复算正在进行。paired_pilot_run01已启动：3开发来源、64/256前缀、seed42101、256token、11法普通/优化配对共132次。正式使用隔离source/prefix_smoke_run02；22次独立预热完成。


paired_pilot_run01已完成132/132，11法各6/6同种子输出与计数一致。全量同步墙钟下旧/新算法比分别：MLP1.67116/1.27726，Ranker-only1.42333/1.30995，EAGLE v3g3 1.35761/0.81499，Medusa0.84200/0.78146。只是3来源256预算小试，不能代替主结果。optimized_main_run01现已启动，10开发来源、64/256前缀、seed43101、512token预算，11法均使用同优化Target，共220次。


paired_pilot_run01本机独立审计8/8通过；runtime_diagnostics.json已生成。只有Target在一个前缀自然EOS，原/新都为165token，其他方法满256；所以共同满预算子集各5对。Target图1445次、原前向6次均为长prefill；所有runtime的disabled_graphs为空。3来源中执行收益Target2.80415×、EAGLE v3g3 1.68337×、PF-MLP2.14321×。接受计数不变，相对同执行Target的算法比发生变化。


optimized_main_run01已完成220/220，初步来源等权速度MLP1.450584×、Ranker-only1.371252×、EAGLE v3g3 0.817906×、Medusa0.725175×，所有10个非Target配置区间上界均低于2。结果复制中，等待独立审计与无新增图子集诊断。LFQ纯BOS补测lfq_bos_run01已串行启动：Target/MLP/LFQ、3种子44101–44103、512token、原/优化两执行共18次。


主测独立审计8/8通过（optimized-only的old/new项不适用，真实配对依据来自pilot）。共同无新增图子集MLP18对/9来源=1.483876×、EAGLE v3g3同18对/9来源=0.814455×；所有非MLP均慢于MLP。Medusa/Hydra共同满512各19对，速度0.768/0.722×，仍低于Target。

LFQ BOS run01完成18/18，但未满足输出与计数重放一致：Target3/3、MLP2/3、LFQ0/3；首分歧MLP seed44101第61个suffix索引，LFQ三个种子索引147/179/391，均为Feature值差异。16token预热一致。保留整轮原始记录，暂不把LFQ速度列为有效结论，正在只读定位并准备同输入/同历史数值验证。GPU退出后2MiB/0%，原Jupyter保留；未安装依赖。


## 最终状态

LFQ BOS的单种子同历史复核已结束：bos_replay_verify_run02先原样重现seed44101的旧输出，再沿213次相同Target输入及相同截短事件执行图前向，213次全部用图、共13图。203次LFQ草稿输入状态全部匹配到参考Target的具体call/row。Target最大policy TV=0.0126780681，hidden最大行L2相对误差0.0013210566；Target policy支持集发生变化，首次TV超过1e-5在call2/绝对位置3。BF16 Draft q最大TV=0.0214848574，支持集也变化。这些数值显著大于真实前缀run02验证范围，不能把LFQ BOS视为数值同等。该探针未追踪抽样决策，尚未定位具体首分歧决策或具体算子，不把LFQ速度列为有效结果。原版、图版dtype均为FP32 Target，BF16仅用于原LFQ草稿。完整元数据见[bos_replay_verify_run02/metadata.json](bos_replay_verify_run02/metadata.json)。

首次probe run01在写入进度时因重复status参数结束；参考追踪保留。只改独立诊断脚本后在新source与新out run02完成57.490秒的复核，旧产物未覆盖。run02原始张量在远端/root/autodl-tmp/gaussiangpt_target_methods_20261021/bos_replay_verify_run02/reference_trace.pt（51,002,452字节）与raw_outputs.pt（101,092,880字节）；本机保存元数据及driver日志，原始大张量未复制到本机。最后GPU2MiB/0%，所有本轮进程已退出，原生产及权重保留。

本轮原/图主要方法比较已完成；2×未达，当前MLP仍最快。LFQ纯BOS因数值差异不排名；后续若继续需要独立解决BOS启动的数值稳定性，不能直接宣称同等采样或设为生产默认。本轮不新增训练、不替换原源码或权重。
