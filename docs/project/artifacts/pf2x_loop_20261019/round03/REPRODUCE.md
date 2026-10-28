# 已有结果与复算入口

实际执行命令、环境、退出码和墙钟见commands.jsonl及各execution.json。最终来源分布在cpu_run02/q_outputs.npz；其原始p/q部分保留FP32值转成float64后的精确数值，raw logits以float32保存。cpu_run02/fit_parameters.npz仅包含两个308参数CPU探针，完整生产权重未纳入此包。

code目录包含两个新增CPU工具，唯一原方法改动是新增工具文件，没有修改生产解码器。source_before为11份当时读取的原源码；prior_spatial_summary.json保留已有空间位移失败结果。serialization_change.diff为首次与成功版本之间的实际差异，failed_summary_write_01/source保留首次代码。source_manifest.json记录本次全部来源分组。

以下是包内重新计算的参考命令，本轮没有再次执行这些命令。先在已有NumPy环境解压，并从保存数组恢复原R1向量文件，原R1 JSONL已在r1_inputs：

```python
import numpy as np
with np.load('cpu_run02/q_outputs.npz', allow_pickle=False) as a:
    np.savez_compressed('r1_inputs/vectors.npz',
        p_f=a['p_f_raw_float64'].astype(np.float32),
        q1=a['q1_raw_float64'].astype(np.float32),
        raw_logits=a['raw_logits_float32'])
```

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python -B code/pf_round03_offline.py \
  --collection-dir r1_inputs --split-manifest source_manifest.json --out replay
```

不需要重算才能阅读结果。cpu_run02/rows.jsonl包含480行来源、候选P、邻居数、每方法重合度和支持量；summary.json包含选择网格、三组逐源结果、来源区间、数值差与计时。primary_validation.json已记录本轮技术重放的全部数组一致性及独立汇总。

首次部分输出仍保存在既有远端/root/autodl-tmp/gaussiangpt_pf2x_round03_20261020/cpu_run01；为避免重复大数组，本包只附最终完整概率文件与首次对比结果。首次代码、错误日志和所有实际命令均保留。没有新在线解码、Target-only或自然EOS产物。
