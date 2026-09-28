#!/usr/bin/env bash
set -euo pipefail
cd /home/yezhe/异构模型/fulltoken_sync_copydrop_v0_q4_to_l3_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python

"$PY" -u tests.py
"$PY" -u experiment.py audit
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo "[$(date '+%F %T')] START Qwen3-4B -> Llama3.2-3B Full-Sync"
"$PY" -u experiment.py all --smoke
echo "[$(date '+%F %T')] SMOKE COMPLETED; starting full study"
"$PY" -u experiment.py all
echo "[$(date '+%F %T')] ALL STAGES COMPLETED"
