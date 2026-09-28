#!/usr/bin/env bash
set -euo pipefail
WAIT_PID="${1:?usage: queue_memorysafe_restart.sh WAIT_PID}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  echo "[$(date '+%F %T')] Waiting for benchmark pipeline PID $WAIT_PID"
  sleep 60
done
echo "[$(date '+%F %T')] Starting memory-safe Qwen-to-Gemma rerun"
cd "$ROOT"
exec bash run_pipeline.sh
