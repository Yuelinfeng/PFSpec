# 实际执行命令与资源

G0使用已有远端Python环境，CUDA_VISIBLE_DEVICES为空，实际执行transition_model.torch_cpu_self_test；4种模型参数数目、零残差logits、概率有限性与归一通过。代码语法与来源元信息另在本机读取，无新依赖安装。

G1唯一进程入口：/root/miniconda3/envs/gaussiangpt/bin/python /root/autodl-tmp/gaussiangpt_position_transition_20261020/source/run_stage1.py --project /root/workspace/GaussianGPT --manifest /root/autodl-tmp/gaussiangpt_position_transition_20261020/source/source_manifest.json --out /root/autodl-tmp/gaussiangpt_position_transition_20261020/g1_run01 --deadline-unix 原g1_budget.json记录值。

外层进程与脚本共用同一900秒绝对截止，含加载/采集/概率重算/成本/零残差重放。原Target与旧MLP只加载一次。G1的MLP提案计时为同输入完整提案64配对、4预热配对、16个真实条件，old/new顺序交替；每次同步CUDA。以C平均增量≤0.5ms选择main，否则只试一次narrow；目标为0.25ms。正式速度不使用带教师捕获的采集外围时间。

零残差在prefix0/64/256、16/64预算下逐token/接受与调用计数/Target实际输入/rewind位置比较，另覆盖预算2/4和历史已知EOS条件。历史EOS条件仅验证实现，不用作新性能样本。

G1续接命令与原截止见raw/g1_resume_budget.json，run_stage1.py新增--reuse-teacher，读取g1_run01已完整保存的数据，输出g1_run02。没有再次采集。

G2实际统一入口：source/run_stage2.py --root /root/autodl-tmp/gaussiangpt_position_transition_20261020 --deadline-unix raw/g2_budget.json记录值。脚本顺序执行B17/B29/C17/C29，每次1200更新，参数与损失见实际源码和每次progress.json。2400秒额度内四次训练均完成。

保存权重CPU复算入口：CUDA_VISIBLE_DEVICES为空，source/export_review.py --root /root/autodl-tmp/gaussiangpt_position_transition_20261020。使用原环境Torch/NumPy，无新增GPU调用。网页包只包含紧凑导出，完整teacher_labels.npz与teacher.pt保留原远端目录。
