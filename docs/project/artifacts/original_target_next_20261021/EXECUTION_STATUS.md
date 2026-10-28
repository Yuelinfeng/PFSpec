# 执行状态

状态：complete。120/120同轮GPU测量、主统计和10项独立CPU复算全部完成。GPU运行404.864秒，无新增训练、依赖或权重。结束时远端主工程HEAD为031d9160622f73dae69324a25d656ce4eb086524，工作区记录仍为4181条；GPU为2MiB、0%，原TensorBoard与Jupyter服务保留。

主表统一以原Target为参照；MLP4.636505倍、EAGLEv3 2.986100倍、EAGLEv4复用2.887617倍、Medusa2.453741倍。此前历史结果与源码保留，未通过历史倍率相乘产生新结论。

本机源码`patches/original_target_next_20261021/`，结果`docs/project/artifacts/original_target_next_20261021/`。远端隔离根目录`/root/autodl-tmp/gaussiangpt_original_target_next_20261021/`；实际测量源码`source_run01/`、原始结果`main_run01/`、运行日志`run01.log`。

新增GPU入口`evaluate_original_target.py`，新增CPU统计`analyze_original_target.py`，新增轨迹筛选`screen_trace_methods.py`；其余12份py为上一轮逐字节相同的提案与参照源码。生产源码与权重未替换。

三名原Luna/max已完成论文、作者代码与CPU统计任务。研究路径research-router → academic-local / academic-evidence / github-local，七篇论文正文、五份仓库源码副本，五份临时源码副本均已清理。研究路由与经验记录在`/Users/lief/.codex/skills/research-router/records/`，临时研究摘要在`/tmp/ggpt_next_spec_20261021/`，无凭据。

下一项仅为已形成的研究建议：保留Ranker+MLP，先实现Feature W2分支验证与真实成本测量；新树解码、UniVer、DFlash、WTV均未开始在线实验。不要把论文速度或CPU接受质量写成本项目新加速比。
