#!/usr/bin/env bash
set -euo pipefail
LLAMA_PID=18220
while kill -0 "$LLAMA_PID" 2>/dev/null; do
  echo "[$(date '+%F %T')] Waiting for Qwen-to-Llama PID $LLAMA_PID to finish"
  sleep 60
done
echo "[$(date '+%F %T')] Qwen-to-Llama finished; retrying Qwen-to-Gemma"
cd /home/yezhe/异构模型/fulltoken_sync_copydrop_v0_q4_to_g3_seed1234
exec bash run_pipeline.sh
