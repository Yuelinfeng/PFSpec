# A1–A3 候选成本测量

本目录只增加独立测量与分析入口，不修改正式解码器，不训练模型。

## 文件

- `branch_probe.py`：从当前共享 K2 解码函数生成带观测点的副本。沿用原提案、随机数、验证、残差采样和缓存操作。
- `run_cost_study.py`：先验证输出与计数一致，再运行 168 次固定窗口 token 生成、896 次相同上下文单轮对照及补算同步子集。
- `run_repair_light.py`：56 次只保留少量计时点的 P₂ 拒绝处理复测，另有 8 次交错普通运行。
- `analyze_cost_study.py`：验证逐轮计数关系、复算成本与速度、输出报告及静态图。

## 执行对象

远端主工程为 `/root/workspace/GaussianGPT`，新文件与结果保存在 `/root/autodl-tmp/gaussiangpt_candidate_cost_20261017/`。原 Target、VQ、Ranker 和 step12000 路径沿用前轮入口中的真实资产。

运行环境为 `/root/miniconda3/envs/gaussiangpt/bin/python`。在主工程目录依次运行以下命令；已有逐条结果会被识别并继续，正式记录不得删除后重跑选优。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/gaussiangpt/bin/python -u /root/autodl-tmp/gaussiangpt_candidate_cost_20261017/run_cost_study.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/gaussiangpt/bin/python -u /root/autodl-tmp/gaussiangpt_candidate_cost_20261017/run_repair_light.py
/root/miniconda3/envs/gaussiangpt/bin/python /root/autodl-tmp/gaussiangpt_candidate_cost_20261017/analyze_cost_study.py
```

固定窗口 token 生成的时间包含缓存分配、前缀计算与释放。相同上下文单轮时间从状态准备完毕后开始，不包含这些准备工作。两种时间范围分别解释。

P₂ 乐观收益采用已观测处理区间的时间相减，其他路径保持原样；这项估计不能代替可部署算法的实测速度。
