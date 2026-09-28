#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p logs
while true; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 24000 )); then
    break
  fi
  echo "[$(date '+%F %T')] GPU free=${free_mib:-unavailable} MiB; waiting for 24000 MiB"
  sleep 60
done
echo "[$(date '+%F %T')] GPU free=$free_mib MiB; starting Stage-B run"
exec bash run_stageb.sh >> logs/pipeline.log 2>&1
