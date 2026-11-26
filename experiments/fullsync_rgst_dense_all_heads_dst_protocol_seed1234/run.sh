#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
exec 9>run.lock
flock -n 9 || exit 1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HETEROFOLD_V100_SDPA=1
py=/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python
for layers in 14 20 24; do
    if "$py" -u run.py --offload-sender-layers "$layers"; then
        exit 0
    else
        code=$?
    fi
    if [ "$code" -ne 75 ]; then
        exit "$code"
    fi
    printf 'CUDA OOM with %s offloaded layers. Resuming from the last saved update.\n' "$layers"
done
exit 75
