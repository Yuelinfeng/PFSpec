#!/usr/bin/env python3
"""Refresh the AI handoff snapshot for the GaussianGPT workspace.

This script deliberately uses only the Python standard library so it can run
before the ML environment is activated.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "project" / "PROJECT_STATUS.md"


def run_git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"unavailable ({exc})"
    value = result.stdout.strip()
    if value:
        return value
    return result.stderr.strip() or "unavailable"


def read_versions() -> str:
    path = ROOT / "gaussiangpt_versions.txt"
    if not path.exists():
        return "未找到 `gaussiangpt_versions.txt`。"
    return path.read_text(encoding="utf-8", errors="replace").strip()


def checkpoint_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for base in (ROOT / "checkpoints", ROOT / "profiling"):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.ckpt")):
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
                rows.append((str(path.relative_to(ROOT)), f"{size_mb:.1f} MB"))
            except OSError:
                rows.append((str(path.relative_to(ROOT)), "unreadable"))
    return rows[:80]


def profiling_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for path in sorted((ROOT / "profiling").glob("e2e*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        baseline = data.get("baseline", {})
        selective = data.get("selective", {})
        base_tps = baseline.get("tokens_per_second")
        selective_tps = selective.get("tokens_per_second")
        speedup = None
        if isinstance(base_tps, (int, float)) and base_tps:
            speedup = selective_tps / base_tps if isinstance(selective_tps, (int, float)) else None
        rows.append(
            (
                str(path.relative_to(ROOT)),
                f"{base_tps:.2f}" if isinstance(base_tps, (int, float)) else "—",
                f"{selective_tps:.2f}" if isinstance(selective_tps, (int, float)) else "—",
                f"{speedup:.3f}×" if isinstance(speedup, (int, float)) else "—",
            )
        )
    return rows


def bullet_lines(values: Iterable[str], empty: str = "（无）") -> str:
    items = list(values)
    return "\n".join(f"- `{item}`" for item in items) if items else empty


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    head = run_git("rev-parse", "HEAD")
    branch = run_git("branch", "--show-current")
    upstream = run_git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    status_text = run_git("status", "--short")
    status_lines = [line for line in status_text.splitlines() if line.strip()]
    modified = [line for line in status_lines if not line.startswith("??")]
    untracked = [line for line in status_lines if line.startswith("??")]

    ckpts = checkpoint_rows()
    profiles = profiling_rows()
    source_dirs = [name for name in ("model", "scripts", "conf", "serialization") if (ROOT / name).exists()]
    source_state = "已检测到源码目录：" + ", ".join(source_dirs) if source_dirs else "当前目录是文档交接工作区，未检测到 model/scripts/conf/serialization 源码目录。"
    checkpoint_table = "\n".join(f"| `{path}` | {size} |" for path, size in ckpts)
    if not checkpoint_table:
        checkpoint_table = "| （未找到） | — |"
    profiling_table = "\n".join(
        f"| `{path}` | {base} | {selective} | {speedup} |"
        for path, base, selective, speedup in profiles
    )
    if not profiling_table:
        profiling_table = "| （未找到 `profiling/e2e*.json`） | — | — | — |"

    status = f"""# GaussianGPT 当前状态快照

> 由 `.codex/scripts/update_status.py` 自动生成。更新时间：`{timestamp}`
> 
> 稳定的架构与实验解释请看 `ARCHITECTURE.md`、`EXPERIMENTS.md` 和 `OPERATIONS.md`；本文件只保存易变事实。

## Git

- HEAD: `{head}`
- 当前分支: `{branch or "（detached HEAD）"}`
- upstream: `{upstream}`
- 工作区改动数: `{len(status_lines)}`（tracked 修改/新增 `{len(modified)}`，未跟踪 `{len(untracked)}`）

### 工作区摘要

{bullet_lines(status_lines[:30])}

## 环境记录

```text
{read_versions()}
```

## 源码工作区

{source_state}

## 权重文件索引

| 路径 | 大小 |
| --- | ---: |
{checkpoint_table}

## 端到端 profiling 索引

吞吐单位为 tokens/s；speedup 为提议路线 / Target。

| JSON | Target tok/s | 提议路线 tok/s | speedup |
| --- | ---: | ---: | ---: |
{profiling_table}

## 交接提示

- 先读 `.codex/AI_START_HERE.md`，再结合 `ARCHITECTURE.md`、`EXPERIMENTS.md` 和 `OPERATIONS.md` 判断任务类型。
- 运行或引用实验前，确认 teacher、VQ-VAE、draft checkpoint、seed、token 数、temperature、top-p、warmup 和 GPU。
- 当前状态脚本不会判断 checkpoint 是否语义可用；`.corrupt`、未完成或没有配置对应关系的文件仍需人工排除。
- 代码、配置、checkpoint 或环境变更后重新运行本脚本。
"""
    OUT.write_text(status, encoding="utf-8")
    print(f"updated {OUT}")


if __name__ == "__main__":
    main()
