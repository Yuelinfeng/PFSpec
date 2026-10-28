# 运行入口、源码关系与缺失资产

## 实测源码的关系

首次5.056076×与复跑5.090087×使用远端 `/root/autodl-tmp/gaussiangpt_spec_next_20261022/source_run02`，依赖 `/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04` 和主工程 `/root/workspace/GaussianGPT`。

本包 `original/patches/spec_next_20261022` 是当前本地副本，后来支持备用数量1–4。其固定一个备用的路径参加了后续六臂测量；不能把整份文件当作首次测量时的逐字节副本。历史 `measured_source_run02_comparison.json` 记录当时五个文件相同，不是今天重新完成的比对。Feature tree本地副本也包含后来补充的验证工具。包内代码用于阅读与条件具备时的复现，不声称拥有全套独立运行资产。

关键文件：

- `original/patches/spec_next_20261022/position_rescue_decoder.py`：备用命中、残差抽样与真实路径提交。
- `original/patches/spec_next_20261022/position_rescue_runtime.py`：备用树验证运行时。
- `original/patches/spec_next_20261022/evaluate_candidates.py`：三臂及扩宽比较。
- `original/patches/spec_next_20261022/evaluate_attribution.py`：原Target、优化Target、W1、单备用分解。
- `original/patches/feature_tree_20261021/tree_decoder.py`、`tree_runtime.py`、`reference/runtime/dynamic_target_graph.py`：W1与Target图执行。
- `original/patches/w1_e2e_20261021/run_e2e.py`：完整三维请求。
- `original/patches/rescue_repeat_20261022/analyze_repeats.py`：四进程来源统计；当前使用analysis_review_run02的最终结果。

## 历史环境和配置

RTX4090；Python3.10.21；PyTorch2.8.0+cu129；FP32 highest；TF32关闭；batch1；temperature/top-p均0.9；相关确认实验缓存容量2048。主工程历史Git版本为 `031d9160622f73dae69324a25d656ce4eb086524`，但远端存在大量工作区内容，单靠该版本号不能恢复完整运行工程。驱动历史记录550.120。完整环境记录附于原件目录。

## 未随包提供的运行资产

| 对象 | 历史路径 | 记录大小/说明 |
|---|---|---|
| Target | /root/workspace/GaussianGPT/checkpoints/gpt_vfront.ckpt | 3421157765字节 |
| VQ-VAE | /root/workspace/GaussianGPT/checkpoints/vqvae_vfront.ckpt | 2115020643字节 |
| MLP | /root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt | 11466893字节；step12000 |
| Ranker | /root/workspace/GaussianGPT/profiling/target_eval/results/ranker_holdout_81001_81016/ranker_trained_on_61001_61016.joblib | 310544字节 |
| 场景token | /root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/final_test_crops.pt | 原输入数据；逐请求prompt已随包 |
| 主工程与依赖 | /root/workspace/GaussianGPT | model、draft_a、序列化、渲染依赖未完整提供 |
| 三维大产物 | /root/autodl-tmp/gaussiangpt_w1_e2e_20261021/rescue_paper_run01 | 包内只提供逐请求记录、相机和历史产物读取结果 |

表中大小来自历史环境记录；本次没有重新验证远端文件内容。包内没有登录密码、私钥或会话凭据。

## 历史原配置复跑命令

下面命令供理解运行入口。需要远端工程及上述依赖；输出目录应另取新名称。

```bash
/root/miniconda3/envs/gaussiangpt/bin/python   /root/autodl-tmp/gaussiangpt_spec_next_20261022/source_run02/evaluate_candidates.py   --project /root/workspace/GaussianGPT   --source /root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04   --crops /root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/final_test_crops.pt   --out /root/autodl-tmp/gaussiangpt_spec_next_20261022/reproduction_new   --methods target,w1,rescue --sources 30 --prefixes 64,256   --seeds 1 --seed-start 51101 --budget 512 --cache-capacity 2048
```

## CPU数据复算

JSON与JSONL可使用标准Python读取。以下是包内既有分析程序的用法，需要其Python依赖；本次整理没有安装依赖或运行GPU推理。输出写到新目录，保留原结果。

```bash
python3 original/patches/spec_next_20261022/summarize_runs.py   --run original/docs/project/artifacts/spec_next_20261022/raw/confirmation_rescue_run01   --reference target --output /tmp/ggpt_review_first.json

python3 original/patches/rescue_repeat_20261022/analyze_repeats.py   original/docs/project/artifacts/rescue_repeat_20261022/raw/repeat_01   original/docs/project/artifacts/rescue_repeat_20261022/raw/repeat_02   original/docs/project/artifacts/rescue_repeat_20261022/raw/repeat_03   original/docs/project/artifacts/rescue_repeat_20261022/raw/repeat_04   --output-dir /tmp/ggpt_repeat_review_new
```
