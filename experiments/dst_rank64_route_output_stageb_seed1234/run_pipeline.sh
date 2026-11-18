#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python_exec=$1
shared_lock=../dst_rank64_gated_dynamic_latency_seed1234/batch2_quality/run.lock
exec 8>"$shared_lock"
flock 8
exec 9>pipeline.lock
flock -n 9
"$python_exec" -u train_stageb.py
"$python_exec" -u evaluate_stageb.py short
"$python_exec" -u evaluate_stageb.py long
