# Round02 GPU 微测数值复核

## 结论

`micro_run01` 在 `cuda:0`、`torch.float32` 上完成。8 个真实上下文来自 4 个来源，前缀类别为 64 与 256，各 4 个。`probe_rows.jsonl` 记录 736 次 Target 调用，尝试数与完成数一致；另记录 728 次缓存恢复和 688 条数值比较，688 条均通过。`paired_target.jsonl` 含 288 组唯一配对结果，三个输入形状 1、2、4 各 96 组。

分组路径在 688 条数值比较中的最大 logits 绝对差为 `1.93119049e-5`，低于 `2e-5` 容差；有限值掩码差异为 0，NaN 数为 0。最大策略概率绝对差为 `3.57627869e-7`，最大总变差为 `3.21044354e-7`，均低于 `5e-5` 容差。比较到的最终 Transformer block hidden、RoPE、有效长度内 K/V 的最大绝对差均为 0；cache position、有效 token、`cache_seqlens` 一致。原始路径与未修改 `GPT.forward` 的 logits 对照最大差为 0。

## 计数、状态与覆盖

元数据和汇总的 Target 调用数一致：prefill 8、reference 24、奇数起点 prefill 8、奇数起点 reference 8、warmup 96、paired 576、奇数起点 grouped/original 各 8，共 736。数值比较由 576 条 paired 模式记录、96 条 warmup 记录和 16 条奇数起点记录组成。全部 `metrics.passed` 为 true。

奇数起点的 8 个上下文均将预测类型记录为 `feature`；original 与 grouped 各 8 次数值比较通过，保留位置均从 `keep_before` 前进 1。EOS 只按输出支持集核对：共 1,656 行，Position 行的 EOS 在 936/936 行有限，Feature 行的 EOS 在 720/720 行均被屏蔽，`eos_policy_valid` 全部为 true。这说明掩码符合记录的类型规则，不表示采样到了自然 EOS。

728 条恢复记录按配对、warmup、reference、奇数起点和清理阶段汇总后，恢复位置都与对应保存位置一致。576 次 paired 恢复分别覆盖形状 1、2、4 各 192 次，`kv_position_after` 均等于配对记录的 `kv_keep`。恢复时间单独记录，728 条均标记为不计入前向时间；288 组配对记录也标记 `restore_costs_separate=true`。

运行期间 `float32_matmul_precision` 为 `highest`，`cuda_matmul_allow_tf32=false`，autocast 关闭；`cudnn_allow_tf32` 在运行期间关闭。运行前后记录的后端状态相同，运行后恢复为 `cudnn_allow_tf32=true`。设备与 dtype 字段分别为 `cuda:0` 和 `torch.float32`。

## 证据边界

微测仅覆盖真实上下文上的 Target 前向及缓存保存、恢复。它没有执行完整自回归生成、PF 接受或拒绝后的 token 重放、残差采样、固定随机流逐分支对照，也没有观察模型自然选出 EOS 并结束序列。EOS 字段记录的是有限性和掩码策略。

RoPE 比较记录 `shape_match` 与数值误差；记录没有标注坐标轴或三维坐标场景覆盖，因此无法据此确认三维坐标变化路径的覆盖程度。有效 K/V 比较只针对有效长度内切片，不涵盖无效尾部缓存的内容。

时序只统计 Target 前向。CUDA 事件中 grouped/original 中位时间比在形状 1、2、4 分别为 `1.004060`、`1.005837`、`0.995214`：形状 1、2 略慢，形状 4 约快 `0.48%`。这组诊断没有显示跨形状的稳定收益，也不代表端到端解码速度。

## 核对输入

- `micro_run01/probe_rows.jsonl`
- `micro_run01/paired_target.jsonl`
- `micro_run01/metadata.json`
- `micro_run01/summary.json`
- `micro_run01/progress.json`
