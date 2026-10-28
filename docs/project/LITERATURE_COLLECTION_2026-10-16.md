# 投机解码文献资料收集

日期：2026-10-16。用户要求先整体收集文献，向后续判断模型提供完整原始信息；本次交付未新增方法评价或实验建议。

## 资料入口

- [Obsidian 文献索引](</Users/lief/workspace/obsidian_md/投机解码/GaussianGPT-specific Speculative Decoding Design/文献资料/2026-10-16/00 文献索引.md>)。
- [模型交接说明](</Users/lief/workspace/obsidian_md/投机解码/GaussianGPT-specific Speculative Decoding Design/文献资料/2026-10-16/01 模型交接说明.md>)。
- [结构化目录与来源](</Users/lief/workspace/obsidian_md/投机解码/GaussianGPT-specific Speculative Decoding Design/文献资料/2026-10-16/catalog.json>)。
- [已有项目资料副本](</Users/lief/workspace/obsidian_md/投机解码/GaussianGPT-specific Speculative Decoding Design/文献资料/2026-10-16/project_context/README.md>)。

## 范围

共 32 篇，包含原汇报的三类相关工作，以及状态复用、多预测头、训练、概率理论、候选分配、综述和部分 2026 年论文。EAGLE 1/2/3、Medusa、Hydra、XSpecMesh、FlashMesh、Fast SceneScript 等均有独立资料卡。

每篇保存可取得的原始 PDF、可搜索文本、发表或 arXiv 页面、版本来源、内容页码入口和获取状态。正式版与预印本分开存放；独立附件保留原始压缩包及其中可读取的文本文件。全文合集与分段索引可供模型逐篇读取。

代码资料包含 EAGLE v1、EAGLE 当前分支、Medusa、Hydra、XSpecMesh、FlashMesh、MSD 和 Gumiho 的作者仓库记录。部分仓库只有 README 与展示资源，目录状态按本次获取的实际内容记录。EAGLE 当前分支的压缩包超过大小限制，已按记录的 Git 版本取得 67 份文本源码。

项目副本含已有说明、用户汇报、源码配置和部分运行记录，以及阶段 E 汇总、阶段 F 汇总及逐次耗时。副本中的历史解释按原始日期阅读，本次复制不代表重新确认其中全部主张。

## 实际操作对象

- Obsidian 资料根目录：`文献资料/2026-10-16/`，位于上述 GaussianGPT 目录内。
- 本地收集脚本与初始目录：[artifacts/literature_collection_20261016/](artifacts/literature_collection_20261016/)。
- 本项目入口：`.codex/AI_START_HERE.md` 新增资料链接。

未修改用户现有汇报正文、远端工程或模型权重，未启动训练、性能实验或渲染。生成状态页保持原样。

## 资料限制

已确认 32 篇均有可读取 PDF 和可搜索文本；包内 49 份 PDF 共 892 个物理页，数量包含不同版本和独立附录。8 份作者仓库或分支记录已保存，文本源码数量与所取得的目录记录一致。新建索引、交接说明和资料卡的本地链接均有效，275 份项目副本与来源字节一致。实际结果见 Obsidian 资料根目录的 `validation.json`。

论文 PDF 为图、表和数学公式的阅读依据；文本与关键词页码只辅助定位。部分正式发表信息未单独核实，部分附件获取失败，均在逐篇记录中保留。原文中的完整附录随 PDF 或独立附件保存；没有逐公式复核全部论文，也没有复现论文结果。

模型权重、完整训练数据、全部项目日志与渲染资产未复制。副本中的既有相对链接按原始目录解释，原始位置与包内位置见 `project_context/source_inventory.json`。
