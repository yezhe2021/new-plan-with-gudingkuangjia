#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec 8>../dst_rank64_gated_dynamic_latency_seed1234/batch2_quality/run.lock
flock 8
exec 9>pipeline.lock
flock -n 9
for phase in train short long; do
    "$1" -u run_rank128.py "$phase"
done
