#!/usr/bin/env bash
set -euo pipefail
cd /home/yezhe/异构模型/fulltoken_sync_copydrop_v0_g3_to_q4_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python

"$PY" -u tests.py
"$PY" -u experiment.py audit

nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo "[$(date '+%F %T')] START Gemma3-4B -> Qwen3-4B Full-Sync training/evaluation"
"$PY" -u experiment.py all --smoke
echo "[$(date '+%F %T')] SMOKE COMPLETED; starting full study"
"$PY" -u experiment.py all
echo "[$(date '+%F %T')] ALL STAGES COMPLETED"
