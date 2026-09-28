#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
cd "$ROOT"
mkdir -p logs
test -f /home/yezhe/异构模型/fulltoken_sync_copydrop_v0_l3_to_q4_seed1234/runs/retrain/stage_a/selection.json
"$PY" -m py_compile retrain_fullsync.py fullsync_v0.py
echo "[$(date '+%F %T')] START Full-Sync Stage-B 4 epochs, 3072 train, 384 validation"
"$PY" -u retrain_fullsync.py stage_b
echo "[$(date '+%F %T')] START held-out test384"
"$PY" -u retrain_fullsync.py evaluate
echo "[$(date '+%F %T')] ALL STAGES COMPLETED"
