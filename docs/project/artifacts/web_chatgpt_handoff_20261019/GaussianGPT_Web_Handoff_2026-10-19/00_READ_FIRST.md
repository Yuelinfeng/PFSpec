# GaussianGPT 网页端研究交接包

资料截止：2026-10-19。用途：让网页端 ChatGPT 理解已完成研究，并协助设计下一阶段完整方法。

用户已确定：将当前最好的 Position Ranker + MLP 作为完成的 Motivation；下一步讨论论文级完整方案。接手时请沿用这个定位，保留对证据强度和新颖性的独立判断。

## 使用方法

上传这个 ZIP，将 [01_WEB_CHATGPT_PROMPT.md](01_WEB_CHATGPT_PROMPT.md) 的内容作为第一条消息。网页环境若不能展开 ZIP，请解压后先上传 [ALL_IN_ONE.md](ALL_IN_ONE.md)，再按问题补充 evidence 与 source 中的文件。不要声称已读取未能打开的附件。

## 阅读顺序

1. [01_WEB_CHATGPT_PROMPT.md](01_WEB_CHATGPT_PROMPT.md)：任务与期望输出。
2. [02_COMPLETED_WORK.md](02_COMPLETED_WORK.md)：已完成工作、关键结果与研究结论。
3. [03_CURRENT_METHOD.md](03_CURRENT_METHOD.md)：当前方法、输入状态、接受采样与源码对应。
4. [04_NEXT_RESEARCH.md](04_NEXT_RESEARCH.md)：下一步研究骨架、假设与实验顺序。
5. [05_EVIDENCE_MAP.md](05_EVIDENCE_MAP.md)：数字与文件的对应关系。
6. [06_RELATED_WORK.md](06_RELATED_WORK.md)：相关工作与贡献边界。
7. [07_LIMITATIONS.md](07_LIMITATIONS.md)：数据、统计、质量和环境限制。

ALL_IN_ONE.md 合并上述正文，便于一次读取；它与分篇文件内容重复，无需重复阅读。

## 包内资料的使用边界

evidence/reports 是已有报告的原文引用副本，保留各自日期和原始措辞。历史段落可能已被后续实验更新。其原工程链接未全部迁入本包，请使用 05_EVIDENCE_MAP.md 与 source_inventory.csv 定位已附文件。新整理正文负责说明当前状态。

evidence/data 包含精选的既有统计与逐次结果，source 包含相关源码快照。它们供阅读、复算和方法审阅使用；整个包不是可直接启动训练的完整工程。完整权重、数据集、全部渲染文件和所有历史日志未收入。未包含服务器账号密码、登录命令和私钥。

本次打包没有连接服务器，没有运行训练、GPU 测速、渲染或安装操作。远端状态仅依据既有记录，未在本轮重新确认。
