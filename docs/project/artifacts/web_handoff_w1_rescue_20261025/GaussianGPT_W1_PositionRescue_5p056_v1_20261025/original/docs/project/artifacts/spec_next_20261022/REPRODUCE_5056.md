# 5.056076× 配置复跑

## 当前状态

用户于2026-10-22明确要求复现旧5.056×配置。唯一一次原三配置矩阵已完成180/180次，得到备用/原Target **5.090087×**，旧值为5.056076×，相对差+0.672674%。输入与执行配置沿用旧轮，计时按本轮真实记录报告。本轮不训练、不调参、不扩宽，不替换生产代码、权重或服务；旧结果完整保留。

60份输入prompt已与原轮逐项相同；Python3.10.21、Torch2.8.0+cu129和RTX4090均与原轮相同。同步外层解码包含首图，采用30来源等权吞吐几何平均。运行器、现有统计程序与主代理直接读取原始记录复算得到相同主值；不额外建立重复统计程序。

## 实测结果

| 比较 | 旧轮 | 本轮 | 本轮95%来源区间 |
|---|---:|---:|---:|
| W1/原Target | 4.969208× | 5.011159× | [4.578822,5.307562] |
| 单备用/原Target | 5.056076× | 5.090087× | [4.593280,5.421318] |
| 单备用/W1 | 1.017481× | 1.015750× | [1.001101,1.024756] |

本轮单备用增量为1.575045%。这是相同配置的一次独立计时重复；没有更改配置来追求特定小数。区间按30来源重采样10000次、统计seed104212。旧轮与本轮区间都有来源差异，计时仍受系统时序影响；该相近复测支持旧配置约5倍token解码收益，不能解释为每次请求固定5.056倍。

三臂共同满512为56配对/30来源，备用/原Target5.133997×；三臂无新增图为58配对/29来源，5.330136×；两者兼具55配对/29来源，5.284593×。这些均是子集，主值仍为5.090087×。早EOS与冷态请求保留在全量结果中。

原Target输出29768token，W1/备用各30170token，各臂2条EOS。W1/备用恢复调用1181→793，Target调用12626→12238，388次备用命中省388次调用；额外备用节点3875，拒绝858次。P接受13903/15086、F接受3701/10345，两种投机配置均相同。主代理逐行对照确认180/180记录的pair_id、method、source、prefix、seed、budget、output_tokens、EOS、generated_tokens和全部counters相同；60份prompt完全相同。直接由输出数量与外层耗时重算主倍率，与运行器和现有统计程序相同。

## 完成范围与资产

180次矩阵正常完成，运行器历时605.295049秒，shell外层607.18秒，未达到900秒执行上限；外层数值包含Python启动和退出。`time -p`结果由本机SSH命令输出取得。模型加载10.847040秒，单独记录，不在主解码计时内。

原始六文件和完整日志已下载至`raw/reproduce_5056_run01`和`raw/reproduce_5056_run01.log`；六文件逐字节与远端相同。分析文件为`reproduction_analysis_target.json`、`reproduction_analysis_w1.json`，环境记录为`reproduction_environment_running.json`与`reproduction_environment_final.json`。原来源、旧记录、运行源码均保留。临时SSH连接已关闭。

结束GPU为2MiB/0%，无计算进程；HEAD031d916、默认4181条工作区记录和821/822服务保持，三份权重与Ranker大小与开始记录相同。source_run02的三个运行文件大小和修改时间均未改变。只新增实验资料并更新项目说明，不修改远端业务源码、权重、数据或生产配置。没有新增三维端到端、更多硬件或更多种子结果。

## 输入和执行

直接使用远端既有`/root/autodl-tmp/gaussiangpt_spec_next_20261022/source_run02/evaluate_candidates.py`，依赖既有`/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04`。原主工程为`/root/workspace/GaussianGPT`。

30来源来自`/root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/final_test_crops.pt`；前缀64/256，seed51101，每次上限512token，缓存容量2048，三配置target/w1/rescue，共180次和60完整配对。FP32、TF32关闭、temperature/top-p均0.9。使用原方法循环顺序与16-token独立预热；预热对象丢弃，正式首次建图计入同步外层解码时间。权重载入、VQ、渲染、保存不在解码主计时内。

输出目录为远端既有实验根中的`reproduce_5056_run01`，本机原资料目录中的`raw/reproduce_5056_run01`。启动前确认目录不存在，GPU为空闲4090，主工程HEAD与Target/VQ/MLP权重大小保持，TensorBoard821和Jupyter822保留。只运行一次完整矩阵，不按倍率重复择优。

外层运行最长900秒，正常完成后立刻结束。旧轮同矩阵约603秒；900秒用于避免异常运行无限占用。上一任务的1200秒预算已经结束，本轮耗时独立记录。首次命令因远端没有`/usr/bin/time`而在启动Python前退出，未创建结果目录、未使用GPU；随后使用shell内置`time -p`，失败文字保留在同一日志开头。

## 结果判断

本轮实际命令沿用旧`pipeline.json`，只更换输出位置：

```bash
time -p timeout --signal=TERM --kill-after=10s 900s \
  /root/miniconda3/envs/gaussiangpt/bin/python \
  /root/autodl-tmp/gaussiangpt_spec_next_20261022/source_run02/evaluate_candidates.py \
  --project /root/workspace/GaussianGPT \
  --source /root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04 \
  --crops /root/autodl-tmp/pf_multiscene_pilot_20261015/stage_e_final/final_test_crops.pt \
  --out /root/autodl-tmp/gaussiangpt_spec_next_20261022/reproduce_5056_run01 \
  --methods target,w1,rescue --sources 30 --prefixes 64,256 \
  --seeds 1 --seed-start 51101 --budget 512 --cache-capacity 2048
```

主值采用每对实际生成token数/同步外层耗时的吞吐比，来源内log比平均后跨30来源等权。按来源重采样10000次、统计seed104212，与旧主报告相同。分别记录全量、共同满512和无新增图子集，不用子集替代主值。

逐条对照旧轮输入、运行顺序、token输出、EOS、长度和语义计数；同轮W1/备用的恢复调用差额应等于备用命中。速度按新测量如实报告，5.056076×保留为旧值，不要求计时逐位相同。
