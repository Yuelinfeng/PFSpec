# Feature 残差块深度实验

保留现有 FeatureStateDraft，新增 3 个、5 个完整残差块，形成总计 1/4/6 块。各块拥有独立状态压缩和残差网络，共用两个 token embedding 与固定 Target Feature 输出权重。新增块输出层置零；训练前输出与原模型相同。

`block_model.py` 定义模型及旧权重扩展；`experiment.py` 复用原训练目标、轨迹抽样和完整验证集选择，执行三个等量续训组。`online.py` 将所选模型接入现有 W1 解码。`pipeline.py` 等待训练完成，依次执行解码等价验证、在线比较、前向耗时测量和汇总。`measure.py` 按已验证 Feature 计接受率，并以来源为单位计算配对区间。

新增块的状态归一化与压缩层从原块复制；残差分支内部使用 PyTorch 默认随机初始化，最后一层置零。两个 token embedding 共用原参数。原块与新增块全部参与训练，固定 Feature 输出权重保持不变。

本轮不改 Target、Position Ranker、采样规则或 Rescue。原 step12000 另外保留为在线参考；原单块的等量续训用于区分结构变化与追加训练。

## 运行位置

4090D 上代码位于 `/root/autodl-tmp/pfspec_mlp_blocks_20261028/`，既有推理环境为 `/root/autodl-tmp/pfspec_repro_4090d_20261028/`。下列命令在该布局下使用：

```bash
cd /root/autodl-tmp/pfspec_repro_4090d_20261028/project
export PYTHONPATH=/root/autodl-tmp/pfspec_mlp_blocks_20261028:/root/autodl-tmp/pfspec_repro_4090d_20261028/project:/root/autodl-tmp/pfspec_repro_4090d_20261028/tree:/root/autodl-tmp/pfspec_repro_4090d_20261028/deps
/root/miniconda3/envs/gaussiangpt/bin/python \
  /root/autodl-tmp/pfspec_mlp_blocks_20261028/experiment.py \
  --root /root/autodl-tmp/pfspec_mlp_blocks_20261028/train_01 \
  --data-plan /root/autodl-tmp/pfspec_mlp_blocks_20261028/data_plan.json \
  --data-root /root/autodl-tmp/pfspec_mlp_blocks_20261028/training_data/data \
  --initial draft_a/runs/feature_target_continue_20261014/selected.pt \
  --steps 1200 --eval-every 400
/root/miniconda3/envs/gaussiangpt/bin/python \
  /root/autodl-tmp/pfspec_mlp_blocks_20261028/pipeline.py
```

重复运行必须更换训练与在线输出位置；`pipeline.py` 中的训练目录和评价输出目录需同步调整。已有输出不覆盖。权重和约 9.5 GB 监督数据保留在服务器，不纳入 Git。

完整设置与结果见项目文档 `docs/project/FEATURE_RESIDUAL_BLOCKS_2026-10-28.md`。
