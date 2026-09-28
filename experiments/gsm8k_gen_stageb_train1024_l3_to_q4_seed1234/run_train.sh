#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/pipeline.log") 2>&1
echo "[$(date '+%F %T')] CPU configuration tests"
"$PY" "$ROOT/tests.py"
echo "[$(date '+%F %T')] GPU protocol/gradient smoke audits"
"$PY" -u "$ROOT/train_generation.py" audit --smoke
echo "[$(date '+%F %T')] Old MCQ Stage-B 0/32 reproduction"
"$PY" -u "$ROOT/evaluate_generation.py" preflight_old
echo "[$(date '+%F %T')] Training fresh Generation Stage-B"
"$PY" -u "$ROOT/train_generation.py" train
echo "[$(date '+%F %T')] Final train/test generation evaluation"
"$PY" -u "$ROOT/evaluate_generation.py" evaluate
echo "[$(date '+%F %T')] GENERATION-ONLY STAGE-B PILOT COMPLETED"
