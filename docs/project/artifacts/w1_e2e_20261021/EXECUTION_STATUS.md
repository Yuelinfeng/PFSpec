# W1 端到端测量

状态：42/42 次运行完成；独立 CPU 复算和保存产物读取均通过，GPU 工作已结束。固定 512-token 组的来源等权端到端倍率为 4.334555×；自然结束组为 4.869038×，后者包含输出长度差异。完整说明见 [实测结果](RESULTS.md)。

主工程为远端 `/root/workspace/GaussianGPT`，HEAD `031d9160622f73dae69324a25d656ce4eb086524`，工作区 4181 条记录。启动前 RTX 4090 占用 2 MiB、利用率 0%，原 TensorBoard 与 JupyterLab 保留。本轮没有安装依赖或训练模型。

实际调用的 W1 源码为 `/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04/`，与此前 4.966× 对应同一实现。新计时入口为 `patches/w1_e2e_20261021/run_e2e.py`，远端副本为 `/root/autodl-tmp/gaussiangpt_w1_e2e_20261021/source_run01/`；结果为同根目录 `main_run01/`。远端保留全部 Gaussian、token 和图片，本机保存 JSON、逐次记录与日志。不要重复运行已完成矩阵。

固定组复用 `feature_tree_20261021/main_run02/prompts.json` 的 18 个输入，共 36 次。自然结束组复用 `pf2x_breakthrough_20261021/natural_eos_run01/prompts.json` 的三个输入，共六次。两组各自对比原 Target 和 W1，轮换先后顺序，独立统计。

计时从已就绪的 token 前缀开始，到保存 token、Gaussian NPZ 与六张 384×384 PNG 结束，包含首次请求的运行时与图创建、KV 分配、生成、序列验证、VQ 解码、渲染及文件写入。权重加载、预热、输入获取和结果 JSON 写入单列。预热的运行时丢弃，正式运行时在同组请求之间复用。

固定组每次生成 512 token，可能还没有完整结束。自然结束组上限 16000 token；输出长度和 Gaussian 数量可能不同，完整请求时长与每秒 token 速度分别统计。本轮只覆盖单个局部 chunk，未测原始场景编码或多 chunk 拼接，有限数值与非空视图不代表视觉质量相同。

本机复算命令：

```sh
python3 patches/w1_e2e_20261021/analyze_e2e.py docs/project/artifacts/w1_e2e_20261021/main_run01 --previous docs/project/artifacts/feature_tree_20261021/main_run02/records.jsonl --natural-previous docs/project/artifacts/pf2x_breakthrough_20261021/natural_eos_run01/records.jsonl
```

实物读取入口为 `audit_outputs.py`，在远端对结果目录执行，已确认 42 份 NPZ 和 252 张 PNG。独立 Luna/max 子代理复算的总倍率、总耗时商、自然组吞吐与阶段求和均一致。结束时远端 HEAD 和 4181 条工作区记录保留，GPU 2 MiB、0%，原服务保留。
