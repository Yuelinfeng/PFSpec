# 2026-10-23 最终复核

四轮均已结束；以[最终复核](RESULTS_REVIEW_20261023.md)及analysis_review_run02为当前解释。今天没有重跑GPU，旧完整测量和243条旧分析提示保留。

# 四次重复验证的运行状态

2026-10-22 本会话重新读取远端 suite.json：四次均 complete，960 条记录已全部保存到本机 raw/repeat_01..04；没有重跑 GPU。两个独立程序复算总体备用/W1=1.0180919，95% 来源区间约[1.0148,1.0213]，各进程倍率1.01627..1.01922。480/480输出与语义计数一致，120个唯一case各臂跨四次输出也一致；恢复差额2908等于命中2908。首图子集较慢，详见 analysis/REPEAT_ANALYSIS.md 和 independent_audit.json。原分析列出的243条顺序差异全部为实际轮换及来源偏移，未见输出、计数或完整度错误；另补独立说明。

下方“正在执行”是启动时记录，当前四次已结束。远端 GPU2MiB/0%，主工程HEAD031d916、4181条工作区记录、原821/822服务与三份权重大小保留。新任务为备用数量比较，另有独立1200秒预算；旧结果不乘入新原Target倍率。

正在执行固定四次独立进程，GPU串行，无新训练、调参、生产部署或原Target重测。详细设计见PLAN.md。新CLI两种子52101/52102、三十既有来源、64/256前缀、512token上限，每次240解码，总960解码/480重复配对/120唯一case。

远端根目录/root/autodl-tmp/gaussiangpt_rescue_repeat_20261022；suite.json记录每次子进程PID、命令与状态。suite初始PID89035，四次源组偏移0/8/15/23、方法列表w1,rescue/rescue,w1交替。source_run01中只有薄入口与串行驱动；实际算法和计时继续加载spec_next_20261022/source_run02/evaluate_candidates.py及同目录备用实现，旧源码不改。

本机新增代码patches/rescue_repeat_20261022，资料docs/project/artifacts/rescue_repeat_20261022。run_repeat.py仅包装_prompt_rows；run_suite.py顺序启动四个全新进程。analyze_repeats.py与audit_independent.py分开复算，不把480重复观测作为480独立来源。

正在复用SSH /tmp/ggpt_rescue_repeat_20261022.sock。密码不写入文件。连接若结束只恢复查看，不重复启动已有suite。按suite.json及各repeat_XX/pipeline.json判断实际状态；只有标记complete且文件复制完成后才做正式分析。

环境起始HEAD031d916、4181工作区记录、GPU2MiB/0%、原821/822服务与原权重保留，详见environment_start.json。本批只确认备用/W1的增量，旧5.056倍率不与新倍率相乘，无新VQ/渲染端到端结果。
