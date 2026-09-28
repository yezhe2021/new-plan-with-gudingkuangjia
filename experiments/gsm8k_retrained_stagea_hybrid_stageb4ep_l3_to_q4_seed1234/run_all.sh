#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/pipeline.log") 2>&1
echo "[$(date '+%F %T')] START CPU tests"
"$PY" "$ROOT/tests.py"
echo "[$(date '+%F %T')] START two-sample GPU smoke"
"$PY" -u "$ROOT/smoke.py"
echo "[$(date '+%F %T')] START frozen GSM8K Stage-A, fresh four-epoch Hybrid Stage-B, and test128"
"$PY" -u "$ROOT/experiment.py"
echo "[$(date '+%F %T')] ALL PIPELINE STAGES COMPLETED"
