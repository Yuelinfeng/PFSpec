# 实际运行命令

运行位置为服务器原工程`/root/workspace/GaussianGPT`，新代码和产物保存在独立目录。以下GPU命令本轮已经启动一次；不要在原任务运行期间重复执行。

```bash
timeout --signal=TERM --kill-after=5s 1800s env \
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/miniconda3/envs/gaussiangpt/bin/python -u \
  /root/autodl-tmp/gaussiangpt_spatial_online_20261020/code/compare_online.py \
  --project /root/workspace/GaussianGPT \
  --crops /root/autodl-tmp/pf_multiscene_pilot_20261015/stage_c/dev_crops.pt \
  --feature-checkpoint /root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt \
  --n2-artifact-root /root/autodl-tmp/gaussiangpt_spatial_embedding_20261020/run01/experiment02 \
  --out /root/autodl-tmp/gaussiangpt_spatial_online_20261020/run01 \
  --deadline-seconds 1800
```

命令通过nohup启动，外部进程40834，日志为`/root/autodl-tmp/gaussiangpt_spatial_online_20261020/runner.log`。程序实际起止时间和各阶段耗时以`raw/run01/pipeline.json`及`summary.json`为准。外部timeout提供同一任务的最长墙钟限制。

GPU任务前，使用同一远端Python、`CUDA_VISIBLE_DEVICES=''`进行真实CPU API与全部480条已保存分布验证；结果分别为`raw/cpu_api_validation.json`与`raw/cpu_saved_distribution.json`。这两项未使用GPU。

独立复算使用本机标准Python，读取完整镜像后的原始逐次数据：

```bash
python3 patches/spatial_embedding_20261020/validate_online.py \
  --run docs/project/artifacts/spatial_online_20261020/raw/run01
```

实际三个新增程序与网页示意路径的对应关系见IMPLEMENTATION.md。源码副本保存在source_before，仅供核对；原Target、MLP、Ranker、VQ及既有空间权重不复制改写到生产目录。
