#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"; mkdir -p logs
while true; do
  FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true)"
  if /home/yezhe/data/miniconda3/envs/attnkv/bin/python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1 \
      && [[ "$FREE" =~ ^[0-9]+$ ]] && (( FREE >= 25000 )); then
    echo "[$(date '+%F %T')] GPU ready: ${FREE} MiB free"
    exec bash run_pipeline.sh
  fi
  sleep 300
done
