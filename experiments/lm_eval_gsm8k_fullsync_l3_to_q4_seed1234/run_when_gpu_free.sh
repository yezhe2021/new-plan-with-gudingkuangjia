#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/logs"
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ "$FREE" =~ ^[0-9]+$ ]] && (( FREE >= 20000 )); then
    echo "[$(date '+%F %T')] GPU free=${FREE}MiB; starting pilot" >> "$ROOT/logs/launcher.log"
    exec bash "$ROOT/run_pilot.sh"
  fi
  echo "[$(date '+%F %T')] waiting: GPU free=${FREE:-unknown}MiB" >> "$ROOT/logs/launcher.log"
  sleep 60
done
