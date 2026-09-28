#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/pipeline.log") 2>&1

echo "[$(date '+%F %T')] CPU tests and dataset/alignment audit"
"$PY" "$ROOT/tests.py"
if [[ ! -f "$ROOT/runs/audit.json" ]]; then
  "$PY" -u "$ROOT/evaluate.py" audit
else
  echo "[$(date '+%F %T')] Reusing completed CPU dataset/alignment audit"
fi
echo "[$(date '+%F %T')] Llama3.2-3B -> Qwen3-4B smoke"
"$PY" -u "$ROOT/evaluate.py" llama --smoke
echo "[$(date '+%F %T')] Llama3.2-3B -> Qwen3-4B study"
"$PY" -u "$ROOT/evaluate.py" llama
echo "[$(date '+%F %T')] Gemma3-4B -> Qwen3-4B smoke"
"$PY" -u "$ROOT/evaluate.py" gemma --smoke
echo "[$(date '+%F %T')] Gemma3-4B -> Qwen3-4B study"
"$PY" -u "$ROOT/evaluate.py" gemma
echo "[$(date '+%F %T')] ALL FROZEN-WRITER EVALUATIONS COMPLETED"
