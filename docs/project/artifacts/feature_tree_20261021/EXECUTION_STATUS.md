# Feature 树任务状态

本轮实现、两轮完整解码与复算已完成。W2 对当前 MLP 未显示可靠净提升，停止扩宽，W4 和新训练均未运行。GPU 任务已结束。

## 实际对象

本机：`/Users/lief/Documents/ChatGPT/GaussianGPT/patches/feature_tree_20261021/`，结果：`docs/project/artifacts/feature_tree_20261021/`。复用原三名 gpt-6-luna/max，职责分别为树运行时、解码/采样、统计/独立数值审计；主代理整合与 GPU 测量。最终统计子代理额度不足，主代理完成实际结果的另一路标准库复算。

远端主工程：`/root/workspace/GaussianGPT`；隔离根目录：`/root/autodl-tmp/gaussiangpt_feature_tree_20261021/`；Python：`/root/miniconda3/envs/gaussiangpt/bin/python`，3.10.21、PyTorch 2.8.0+cu129、RTX 4090。Feature 权重为主工程 `draft_a/runs/feature_target_continue_20261014/selected.pt`，沿用现有 step12000；Target、Ranker 与 VQ 权重沿用原配置，没有替换。

`source_run03/` 对应 numeric_run05、smoke_run01、main_run01；`source_run04/` 仅把 RoPE 节点调用合并，对应 rope_batch.json、numeric_run06、main_run02。`source_final_run05/` 保存最终交接源码：与 source_run04 相比仅新增结果/概率审计脚本和准确的树方法描述信息，推理逻辑未再改变。

## 实测及历史记录

- `sampling_law.json`：9/9 通过，包含 CPU PyTorch generator 状态对照。
- `numeric_run01`：文件传输尚未完成，导入 reference 失败，未加载模型；该目录保留。
- `numeric_run02`：测试试图从 top-p 单点分布构造另一个正概率 token，夹具终止，没有最终数值结论。
- `numeric_run03`：840 项、464 项通过；隐藏状态逐元素比较和少量 KV value 未过，原失败记录保留。
- `numeric_run04`：增加相同 BMM 与 MLP q 对照；最终判断以补足全有限和归一化要求的 run05/06 为准。
- `numeric_run05`、`numeric_run06`：各 1742/1742 语义项通过，各保留 440 项逐元素 allclose 诊断；原概率张量仅在远端对应目录 `raw_policy_pairs.pt`。
- `smoke_run01`：2 输入×4 臂×64 token，8/8 完成，W1/MLP 输出及共有计数 2/2 相同。
- `main_run01`、`main_run02`：各 72/72，全部 512 token；各轮 W1/MLP 18/18 相同，跨轮 72/72 输出与业务计数相同。
- `numeric_audit.json`、`numeric_audit_run06.json`：子代理 CPU 复算概率；`independent_audit_run01.json`、`independent_audit_run02.json`：主代理从原记录复算完整解码。

所有阶段目录保留，没有用最终成功记录覆盖早期失败。首轮与最终轮分别汇总，没有把重复输入称为独立新场景。

## 复现命令

在远端使用对应 Python 和已经保留的 source_run04：

```bash
OMP_NUM_THREADS=1 /root/miniconda3/envs/gaussiangpt/bin/python \
  /root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04/evaluate_tree.py \
  --project /root/workspace/GaussianGPT \
  --crops /root/autodl-tmp/pf_multiscene_pilot_20261015/stage_c/dev_crops.pt \
  --out /root/autodl-tmp/gaussiangpt_feature_tree_20261021/new_run \
  --methods target,pf_mlp,tree_w1,tree_w2 --sources 3 --prefixes 64,256 \
  --seeds 3 --seed-start 45101 --budget 512
```

`new_run` 必须尚不存在。完整矩阵已完成，后续会话不要无理由重复运行。无需再次 GPU 测量即可复算：

```bash
python3 patches/feature_tree_20261021/audit_tree_results.py \
  docs/project/artifacts/feature_tree_20261021/main_run02 \
  --previous-run-dir docs/project/artifacts/feature_tree_20261021/main_run01 \
  --out /tmp/feature_tree_result_audit.json
```

概率审计脚本接受 `--json`、`--raw`、`--out`，必须使用远端现有 PyTorch 解释器并映射到 CPU；详见其 CLI。run06 概率复算约在本机时区 14:06:53–14:06:59 完成，显式计算线程为 1；后续脚本还显式设置交互线程为 1。计时矩阵中没有同时启动其他 GPU 工作。

## 结束状态

远端主工程 HEAD 为 `031d9160622f73dae69324a25d656ce4eb086524`，工作区记录数仍为 4181；本轮未写生产源码或权重。GPU 2 MiB、0%；原 TensorBoard PID 821 与 JupyterLab PID 822 保留。没有安装依赖、训练、提交 Git 或替换生产默认。项目状态页已按交接规则刷新。
