#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/logs"
while pgrep -f '/home/yezhe/异构模型/fulltoken_sync_copydrop_v0_q4_to_g3_seed1234/experiment.py' >/dev/null; do
  echo "[$(date '+%F %T')] Waiting for current Qwen-to-Gemma experiment" >> "$ROOT/logs/queue.log"
  sleep 60
done
echo "[$(date '+%F %T')] Starting frozen-writer benchmark suite" >> "$ROOT/logs/queue.log"
exec bash "$ROOT/run_pipeline.sh"
