#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec 8>../dst_rank64_gated_dynamic_latency_seed1234/batch2_quality/run.lock
flock 8
exec 9>pipeline.lock
flock -n 9
"$1" -u long_memory.py oracle
"$1" -u train_recovery.py smoke
"$1" -u train_recovery.py train
"$1" -u long_memory.py recovered
"$1" -u train_recovery.py short
