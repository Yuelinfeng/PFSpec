# 单步 Feature Transformer

复用此前 B1-E 两 token、单层 Transformer，宽度 256、4 个注意力头、FFN 512。输入为已提交上下文的 Target 状态、上一 Feature 与候选 Position。Target embedding 和 Feature 输出权重保持不变；不维护跨轮 Draft KV。`feature_model.py` 仅将固定因果遮罩提前分配，便于 CUDA Graph 执行。`legacy_models.py` 为旧定义的原样副本。

`train.py` 使用原三种子 B0/B1-E 所选权重，在当前 400/50 条完整轨迹数据上各追加 1200 步。MLP 与 Transformer 共用数据顺序、损失和学习率。此前的 10 轮训练搜索预算相同，所选轮数由各自验证结果决定。本轮参数量没有刻意匹配。

训练和评价从服务器已有推理环境导入公共实现；无需安装新依赖。运行位置为 `/root/autodl-tmp/pfspec_single_step_transformer_20261028/`，解释器为既有 `gaussiangpt` 环境。

```bash
cd /root/autodl-tmp/pfspec_repro_4090d_20261028/project
export PYTHONPATH=/root/autodl-tmp/pfspec_single_step_transformer_20261028:/root/autodl-tmp/pfspec_repro_4090d_20261028/project:/root/autodl-tmp/pfspec_repro_4090d_20261028/tree:/root/autodl-tmp/pfspec_repro_4090d_20261028/deps
/root/miniconda3/envs/gaussiangpt/bin/python \
  /root/autodl-tmp/pfspec_single_step_transformer_20261028/train.py \
  --root /root/autodl-tmp/pfspec_single_step_transformer_20261028 \
  --out /root/autodl-tmp/pfspec_single_step_transformer_20261028/train_01 \
  --data-plan /root/autodl-tmp/pfspec_mlp_blocks_20261028/train_01/add0/data_plan.json
/root/miniconda3/envs/gaussiangpt/bin/python \
  /root/autodl-tmp/pfspec_single_step_transformer_20261028/pipeline.py
```

旧模型应先存入 `initial_direct/b1_training/<seed>/{b0,b1_e}/selected.pt`。`prepare_assets.py` 可从原 GPT 读取共用权重，要求新输出路径，并与已有 Feature 输出权重逐元素比对。本轮已完成此读取，保存在 `fixed_assets.pt`。

`pipeline.py` 等待训练完成，依次执行 14 组普通/图解码对照、420 次在线比较、batch1 前向耗时和来源配对汇总。训练种子保留在同一来源内，不将其当作更多独立场景。再次运行需同步调整训练、验证和在线输出位置，既有结果不覆盖。

详情见 `docs/project/SINGLE_STEP_TRANSFORMER_2026-10-28.md`。原 Target、Ranker 与 step12000 MLP 均保留；生成权重、监督数据和原始数值记录留在服务器，不提交到 Git。
