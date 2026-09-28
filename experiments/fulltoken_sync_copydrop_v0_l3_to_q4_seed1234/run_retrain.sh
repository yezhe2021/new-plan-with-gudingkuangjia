#!/usr/bin/env bash
set -euo pipefail
cd /home/yezhe/异构模型/fulltoken_sync_copydrop_v0_l3_to_q4_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
mkdir -p logs
"$PY" -u test_alignment.py
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
for stage in stage_a stage_b evaluate; do
  echo "[$(date '+%F %T')] START $stage"
  "$PY" -u retrain_fullsync.py "$stage"
  echo "[$(date '+%F %T')] DONE $stage"
done
echo "[$(date '+%F %T')] ALL STAGES COMPLETED"
