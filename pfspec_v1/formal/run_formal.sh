#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${PFSPEC_RUN_DIR:-/root/autodl-tmp/pfspec_formal_20261028}"
REPRO_DIR="${PFSPEC_REPRO_DIR:-/root/autodl-tmp/pfspec_repro_4090d_20261028}"
PYTHON_BIN="${PFSPEC_PYTHON:-/root/miniconda3/envs/gaussiangpt/bin/python}"
MODE="${1:-}"

case "$MODE" in
  prepare)
    if [[ $# -ne 3 ]]; then
      printf 'Usage: %s prepare CROPS.pt SCENE_IDS.txt\n' "$0" >&2
      exit 2
    fi
    mkdir -p "$RUN_DIR"
    "$PYTHON_BIN" "$SCRIPT_DIR/make_formal_prompts.py" --crops "$2" --source-list "$3" --out "$RUN_DIR/prompts"
    "$PYTHON_BIN" - "$2" "$3" "$RUN_DIR/warm_tokens.pt" <<'PY'
import sys
from pathlib import Path
import torch
sources = [s.strip() for s in Path(sys.argv[2]).read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
rows = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
for row in rows:
    sid = str(row.get('scene') or row.get('source') or Path(str(row['path'])).parents[1].name)
    if sid == sources[0]:
        torch.save({'tokens': torch.as_tensor(row['tokens']).cpu()}, sys.argv[3])
        break
else:
    raise ValueError('First selected scene was not found')
PY
    ;;
  A|B)
    if [[ $# -ne 1 ]]; then printf 'Usage: %s A|B\n' "$0" >&2; exit 2; fi
    common=(--project "$REPRO_DIR/project" --source "$REPRO_DIR/tree"
      --candidate-source "$REPRO_DIR/candidate" --fixed-prompts "$RUN_DIR/prompts/fixed_prompts.json"
      --warm-tokens "$RUN_DIR/warm_tokens.pt" --cache-capacity 2048)
    if [[ "$MODE" == A ]]; then
      "$PYTHON_BIN" "$SCRIPT_DIR/run_e2e.py" "${common[@]}" --natural-prompts "$RUN_DIR/prompts/natural_prompts.json" \
        --methods target,opt_target,ranker_only,w1,rescue --out "$RUN_DIR/run_A"
    else
      "$PYTHON_BIN" "$SCRIPT_DIR/run_e2e.py" "${common[@]}" --methods w1,w2 --fixed-only --out "$RUN_DIR/run_B"
    fi
    ;;
  summary)
    if [[ $# -ne 2 || ( "$2" != A && "$2" != B ) ]]; then
      printf 'Usage: %s summary A|B\n' "$0" >&2; exit 2
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/analyze.py" "$RUN_DIR/run_$2" --out "$RUN_DIR/run_$2/analysis.json"
    ;;
  resume|merge)
    if [[ $# -ne 2 || ( "$2" != A && "$2" != B ) ]]; then
      printf 'Usage: %s resume|merge A|B\n' "$0" >&2; exit 2
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/resume_formal.py" "$MODE" "$RUN_DIR/run_$2" --python "$PYTHON_BIN"
    ;;
  *)
    printf 'Usage: %s prepare CROPS.pt SCENE_IDS.txt | A | B | summary A|B | resume|merge A|B\n' "$0" >&2
    exit 2
    ;;
esac
