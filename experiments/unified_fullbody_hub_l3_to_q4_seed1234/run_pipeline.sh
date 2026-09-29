#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python"
cd "$ROOT"
mkdir -p logs

run() {
  echo "[$(date '+%F %T')] START $*"
  "$PY" -u unified_hub.py "$@"
  echo "[$(date '+%F %T')] DONE  $*"
}

run audit
run stage_a --smoke
run stage_b_kl --smoke
run stage_b_hybrid --smoke
run evaluate --smoke
echo "[$(date '+%F %T')] SMOKE COMPLETED"

run stage_a
run stage_b_kl
run stage_b_hybrid
run evaluate
echo "[$(date '+%F %T')] ALL EXPERIMENTS COMPLETED"
