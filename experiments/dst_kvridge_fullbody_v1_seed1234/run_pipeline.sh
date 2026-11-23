#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>pipeline.lock
flock -n 9

python=/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

if [[ ! -s stage_a/summary.json ]]; then
    pid=$(cat stage_a.pid)
    echo "Waiting for the raw Ridge fit (PID $pid)"
    while [[ ! -s stage_a/summary.json ]]; do
        if ! kill -0 "$pid" 2>/dev/null || [[ $(ps -o stat= -p "$pid") == *Z* ]]; then
            [[ -s stage_a/summary.json ]] || { echo "Ridge fit stopped; see stage_a.log" >&2; exit 1; }
        fi
        sleep 10
    done
fi

"$python" -u run_stage_b.py --check
"$python" -u run_stage_b.py
exec "$python" -u evaluate_fixed.py --full
