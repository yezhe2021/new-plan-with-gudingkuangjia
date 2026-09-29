#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
cd "$ROOT"; mkdir -p logs
run() { echo "[$(date '+%F %T')] START $*"; "$PY" -u equivalence_audit.py "$@"; echo "[$(date '+%F %T')] DONE  $*"; }
run preflight
run capture_llama --smoke
run capture_gemma --smoke
run audit --smoke
run cleanup --smoke
echo "[$(date '+%F %T')] SMOKE COMPLETED"
run capture_llama
run capture_gemma
run audit
run cleanup
echo "[$(date '+%F %T')] ALL EXPERIMENTS COMPLETED"
