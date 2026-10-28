# GaussianGPT 当前状态快照

> 由 `.codex/scripts/update_status.py` 自动生成。更新时间：`2026-10-28T10:59:33+08:00`
> 
> 稳定的架构与实验解释请看 `ARCHITECTURE.md`、`EXPERIMENTS.md` 和 `OPERATIONS.md`；本文件只保存易变事实。

## Git

- HEAD: `HEAD`
- 当前分支: `main`
- upstream: `fatal: no such branch: 'main'`
- 工作区改动数: `6`（tracked 修改/新增 `0`，未跟踪 `6`）

### 工作区摘要

- `?? .codex/`
- `?? .gitignore`
- `?? AGENTS.md`
- `?? README.md`
- `?? docs/`
- `?? patches/`

## 环境记录

```text
未找到 `gaussiangpt_versions.txt`。
```

## 源码工作区

当前目录是文档交接工作区，未检测到 model/scripts/conf/serialization 源码目录。

## 权重文件索引

| 路径 | 大小 |
| --- | ---: |
| （未找到） | — |

## 端到端 profiling 索引

吞吐单位为 tokens/s；speedup 为提议路线 / Target。

| JSON | Target tok/s | 提议路线 tok/s | speedup |
| --- | ---: | ---: | ---: |
| （未找到 `profiling/e2e*.json`） | — | — | — |

## 交接提示

- 先读 `.codex/AI_START_HERE.md`，再结合 `ARCHITECTURE.md`、`EXPERIMENTS.md` 和 `OPERATIONS.md` 判断任务类型。
- 运行或引用实验前，确认 teacher、VQ-VAE、draft checkpoint、seed、token 数、temperature、top-p、warmup 和 GPU。
- 当前状态脚本不会判断 checkpoint 是否语义可用；`.corrupt`、未完成或没有配置对应关系的文件仍需人工排除。
- 代码、配置、checkpoint 或环境变更后重新运行本脚本。
