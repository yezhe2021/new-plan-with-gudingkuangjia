#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/异构模型/openbookqa_b0_mixedhub_receiver_stageb_gemma_seed1234"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
MAX_USED_MIB=2000

cd "$ROOT"
while true; do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ "$used_mib" =~ ^[0-9]+$ ]] && (( used_mib <= MAX_USED_MIB )); then
        break
    fi
    printf '[%s] Waiting for GPU: used=%s MiB, threshold=%s MiB\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "$used_mib" "$MAX_USED_MIB"
    sleep 60
done

printf '[%s] GPU available; starting evaluate only\n' "$(date '+%Y-%m-%d %H:%M:%S')"
exec "$PYTHON" -u experiment.py evaluate
