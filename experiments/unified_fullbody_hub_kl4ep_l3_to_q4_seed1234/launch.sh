#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p logs

if pgrep -af "${ROOT}/unified_hub.py|${ROOT}/run_pipeline.sh" | grep -v grep >/dev/null; then
  echo "Pipeline is already running"
  exit 0
fi

nohup bash run_pipeline.sh > logs/pipeline.log 2>&1 &
echo $! > logs/pipeline.pid
echo "Started PID $(cat logs/pipeline.pid)"
