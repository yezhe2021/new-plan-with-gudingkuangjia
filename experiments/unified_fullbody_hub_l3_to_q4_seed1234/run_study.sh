#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python"
cd "$ROOT"

run() {
  echo "[$(date '+%F %T')] START $*"
  "$PY" -u unified_hub.py "$@"
  echo "[$(date '+%F %T')] DONE  $*"
}

echo "[$(date '+%F %T')] RESTART FORMAL PIPELINE WITH 2 EPOCHS"
run stage_a_resume
run stage_b_kl
run stage_b_hybrid
run evaluate
echo "[$(date '+%F %T')] ALL EXPERIMENTS COMPLETED"
