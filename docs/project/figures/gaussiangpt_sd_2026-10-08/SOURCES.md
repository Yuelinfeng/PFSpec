# 图表数据与范围

这些图是报告配图，由已有数值重绘，没有新增模型实验。正文及数据表提供与图相同的关键数值。

## AR 分项图

`ar_breakdown.csv` 原样复制自本机 `/Users/lief/Downloads/panel_c_ar_breakdown_data.csv`。配套元数据为 `/Users/lief/Downloads/panel_c_ar_breakdown_metadata.json`，原图为同目录 `panel_c_ar_breakdown_v2.png`，原件均未改动。

元数据标记上下文约 2048 token，窗口合计 13.628462576889433 ms/token。CSV 列名为 `median_ms_per_token`；图保留该导出统计，未声称完整场景平均时间。forward 阶段包含其内部同步，不另加同步分项。

元数据指向以下远端相对路径，根目录为 `/root/workspace/GaussianGPT`：

- `profiling/ar_breakdown/ar_phase_window_ctx2048_n128_r3/summary.json`
- `profiling/ar_breakdown/sdpa_cachepos_sync_window_ctx2048_n128_r3/summary.json`
- `profiling/ar_breakdown/p5_sampling_ablation_2048_r5/summary.json`

10 月 8 日制图时未读取上述原始 summary，因此原图仅作该窗口的 AR 内部分析。2026-10-16 已读取第一项原始 summary 及远端 `profiling/raw/p1_vfront_final.csv` 的五次记录，确认窗口 forward 占比 96.6828% 与 P1 AR/Core 占比 99.5654%。两项统计范围不同，不相乘。新图与原始附件见 [Profiling 数据汇报](</Users/lief/workspace/obsidian_md/投机解码/GaussianGPT-specific Speculative Decoding Design/GaussianGPT Profiling 数据汇报.md>)；本目录历史图未重绘。

原元数据的缓存位置读取消融列出两项时间 13.742485316470265 和 13.352033027331345 ms，差为 0.39045228913892 ms，另列节省时间为 0.4407947417348623 ms。由于数值不一致，新图不引用该消融。top-p 到 greedy 的消融改变目标分布，也不纳入本报告的同分布性能结论。

## S9 速度图

`s9_summary.csv` 转录自[阶段报告第 8.5 节](../../STAGE_REPORT_2026-10-08.md)与[实验记录](../../EXPERIMENTS.md)，属于历史汇总的本地副本，未伪装为本轮读取的原始运行 JSON。

历史原始对象为远端 `profiling/s9_multiscene/phase4_matrix_summary.json` 与 `phase4_matrix.csv`。RTX 4090、三种子、四请求长度、temperature/top-p=0.9、warmup=128。点为历史速度比均值，误差线为记录的 std；它们描述种子间差异，不是重复计时置信区间。图不连接不同请求长度的点，以免暗示完整连续函数。

后两个 D8 请求长度含 seed2 在 1753 token 的提前 EOS；图中以星号注明。structured 全部达到请求长度。所有输入为 BOS 起始的随机轨迹，不能描述成独立真实场景。当前读取的对照入口有同步和输出计算范围差异，历史 S9 的全部 runner 未重新核对，图用于阶段资料展示。

## 重绘与阅读

`render_figures.py` 使用 Python 标准库生成 SVG，再调用本机已有 Sharp 生成 PNG。图 1 用共同零起点的横条比较毫秒数，并直接标记占比；图 2 用点和误差线表示速度比，1× 为普通 Target 参照线。蓝色表示主要方法，橙色表示对照，形状同时区分，灰色表示上下文。图表只读，无交互状态和服务依赖。较窄屏幕可结合正文表格阅读数值。

运行 `python3 render_figures.py` 可在本目录重绘；脚本使用当前本机已经提供的 Node/Sharp 路径，没有安装新依赖。SVG 保留文字，PNG 用于 Markdown 显示。
