# Ranker+MLP执行优化：2×目标已取得实测证据

2026-10-21，台北时间。固定原Target参照下，新PF在30个独立场景的来源等权速度比为4.519564×，95%区间[4.228452,4.720021]；相对原Ranker+step12000 MLP为2.613919×。三个gpt-6-luna/max负责成本、解码与运行实现，主代理整合并串行安排GPU。主计划见[PLAN.md](PLAN.md)。

通用执行优化也使Target自身更快。同优化Target下，新PF为1.442534×；本轮主要收益来自执行方式，尚无算法自身达到同优化Target两倍的证据。

## 实际结果

独立确认使用stage_e_final/final_test_crops.pt的全部30场景，与开发10场景零重合；prefix64/256、每个条件一种子、512新token预算，共60配对条件、240次解码。保持同一4090、FP32最高精度、TF32关闭、temperature=0.9、top_p=0.9，四模式轮换运行。

| 对照 | 来源等权速度比 | 来源重采样95%区间 |
|---|---:|---:|
| 新PF / 原Target | 4.519564× | [4.228452, 4.720021] |
| 新PF / 原PF | 2.613919× | [2.506082, 2.672991] |
| 新PF / 同优化Target | 1.442534× | [1.372647, 1.495168] |
| 同优化Target / 原Target | 3.133074× | [3.081829, 3.164061] |

两臂都生成满512token的55配对条件中，新PF/原Target为4.564797× [4.362947,4.724507]。全部长度结果按真实生成token数与同步解码墙钟计算吞吐，再对每来源等权汇总；不同方法可能自然结束于不同长度。置信区间用来源重采样10000次、种子381012。原始记录及独立复算在[independent_confirm_run01](independent_confirm_run01/summary.json)与[independent_audit_v2.json](independent_confirm_run01/independent_audit_v2.json)，8/8项通过。

原/新PF的60/60组输出与接受计数完全相同，原/新Target也为60/60。候选实际调用42579次图前向、13个图；120次普通前向全部为长prefill，没有异常回退。它保留原Ranker、Feature MLP、采样及拒绝恢复逻辑，通过动态CUDA Graph、稳定KV地址与短查询显式BMM减少Target执行成本。

测量runtime在正式解码开始后创建，首次KV分配、图预热与创建、后续新长度桶的图创建均已计入外层同步墙钟。独立32token预热所用runtime被丢弃。测量runtime跨候选请求及两个候选方法复用，图创建合计约2.910秒，已经包含于实测，不再另加或扣除；每请求重新创建runtime的部署方式未测。权重加载、数值验证和独立预热单列，倍率不包含VQ解码与渲染。

## 自然EOS与三维验证

[natural_eos_run01](natural_eos_run01/summary.json)使用同一实现、排序前三个独立场景、prefix256、一种子和16000token上限，12/12次自然结束，无预算截断。PF新token长度为1233/1889/3035，Target为1637/3205/4127；原/新两类分别3/3组逐token与计数相同。新PF/原Target为3.630147× [2.791982,4.246229]，新PF/原PF为2.209333× [1.987015,2.373086]，新PF/同优化Target为1.273811× [1.019382,1.443080]。独立复算8/8项通过。只有三个场景，不能代替30场景长输出统计。

[full3d_validity_run02](full3d_validity_run02/metadata.json)取独立确认中第一条自然EOS的新PF输出：153组Position/Feature经原VQ解码得到11286个Gaussian，所有参数、RGB和alpha均有限，六张384×384视图已保存。主代理查看view_00，内容可见但结构稀疏；这是单chunk有效性证据，没有新增质量提升结论。VQ包含三层上采样，latent Position与最终Gaussian数量没有固定1:1对应。原脚本错误假设1:1的run01保留为失败记录。原helper已有scale裁剪，本例裁剪比例9.70%，沿用原行为。

## 实际文件与操作位置

本机源码目录：`/Users/lief/Documents/ChatGPT/GaussianGPT/patches/pf2x_breakthrough_20261021/`。

- `attention_override.py`、`runtime/dynamic_target_graph.py`：隔离Attention与动态图实现。
- `evaluate_runtime.py`：原Target、原PF、新PF、同优化Target四模式实测入口。
- `runtime/verify_dynamic_target_graph.py`：真实混合T、跨长度桶及截短换Feature验证。
- `runtime/verify_full3d_output.py`、`runtime/audit_completed_run.py`：VQ渲染验证与独立统计复算。
- `attention/bench_sdpa_candidates.py`、`attention/bench_sdpa_candidates_graph.py`、`runtime/pf_target_graph_probe.py`：前期成本探针。
- `residual/analyze_residual_feasibility.py`：480条既有p/q的CPU分析；未把模拟接受质量当成速度。

本机资料与原始结果：`/Users/lief/Documents/ChatGPT/GaussianGPT/docs/project/artifacts/pf2x_breakthrough_20261021/`。PLAN、COST_DESIGN、MODEL_DESIGN、RUNTIME_DESIGN、EXECUTION_STATUS及本文件记录方案、实现和证据；全部实测目录及旧失败记录保留。

本机项目记忆已更新：`.codex/AI_START_HERE.md`、`docs/project/EXPERIMENTS.md`、`RESEARCH_SOLUTIONS.md`、`LOGIC_VALIDATION.md`、`OPERATIONS.md`，并用现有脚本刷新`PROJECT_STATUS.md`。

远端新增隔离源码与实测结果位于`/root/autodl-tmp/gaussiangpt_pf2x_breakthrough_20261021/`；前期固定Target探针位于`/root/autodl-tmp/pf_target_graph_probe_20261021/`。实际主工程为`/root/workspace/GaussianGPT`。未改生产源码、Target/VQ/Ranker/MLP权重或用户数据，未安装依赖，未提交Git；优化尚未设为生产默认入口。最后GPU为2MiB、0%，原Jupyter和TensorBoard保持运行。

## 数值依据与适用范围

真实T1/T2/T4、512→1024跨桶、缓存截短并换Feature验证实际使用图前向，最大policy TV为1.14e-6、hidden每行相对L2为1.68e-6；Feature q另有真实前缀诊断。接近零的hidden分量曾使逐元素allclose失败，旧失败保留；后续要求支持集一致、Target/q TV≤1e-5、hidden有限同形且每行相对L2≤1e-5，逐元素结果保留作诊断。60配对和三组长输出的完全一致是本次实测证据，不能证明任意输入逐位相同。

本轮只测一台4090、batch=1及当前模型，未覆盖其他设备或并发。短输出第一次承担图创建时可能低于2×；30来源平均并不保证每次请求达到2×。长输出KV容量16384占约3GiB，实际最多覆盖4127新token，未实测上下文上限。所有提前失败的探针、普通前向烟测和SSH中断记录继续保留，详细历程见[EXECUTION_STATUS.md](EXECUTION_STATUS.md)，不要重复运行旧失败实验或已无收益的训练。
