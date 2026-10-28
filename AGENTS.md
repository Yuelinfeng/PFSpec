# GaussianGPT project instructions

Conversations in this project may delegate independent work to subagents when it is useful.

Use at most three concurrent subagent threads for this project. Prefer `gpt-6-luna` for spawned agents and use `max` reasoning effort by default. Keep shared-file edits coordinated through the primary conversation.

## 实验结果解释

实验结果不符合预期时，先核验代码实现、配置、数据流、抽样与缓存行为、计时范围及对照执行路径，再讨论方法的有效性。将已确认的实现问题、设置局限与方法证据分别说明；未完成实现核验时，不用负结果直接否定研究方向。修改执行代码后，先完成输出与动作等价验证，再进行性能测量。

## Project handoff and maintenance

- Before changing code, read `.codex/AI_START_HERE.md` and the current `docs/project/PROJECT_STATUS.md`.
- Treat `docs/project/ARCHITECTURE.md`, `docs/project/EXPERIMENTS.md`, `docs/project/RESEARCH_SOLUTIONS.md`, `docs/project/LOGIC_VALIDATION.md`, and `docs/project/OPERATIONS.md` as the canonical project memory for new AI sessions.
- Refresh the generated status snapshot with `python .codex/scripts/update_status.py` after changes to code, checkpoints, profiling results, or the environment.
- Keep stable architecture and experiment conclusions in the static documents; do not hand-edit the generated status file except to fix the updater.
- Do not commit generated checkpoints, render outputs, caches, or profiling artifacts unless a task explicitly requires them. Record their paths and interpretation in the project documents instead.
- When a code or experiment change alters behavior, update the relevant project document in the same change so a future AI can reproduce the reasoning.
