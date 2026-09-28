#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/pipeline.log") 2>&1

echo "[$(date '+%F %T')] START CPU tests"
"$PY" "$ROOT/tests.py"
echo "[$(date '+%F %T')] START protocol and gradient audits"
"$PY" -u "$ROOT/train.py" audit
echo "[$(date '+%F %T')] START shared-cache three-objective training"
"$PY" -u "$ROOT/train.py" train --objective all
echo "[$(date '+%F %T')] START official untouched test128 evaluation"
"$PY" -u "$ROOT/evaluate.py"
echo "[$(date '+%F %T')] GENERATION STAGE-B V2 ALL COMPLETED"
