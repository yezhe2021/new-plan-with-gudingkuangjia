#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
exec 9>evaluation.lock
flock -n 9 || exit 1
py=/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HETEROFOLD_V100_SDPA=1
"$py" -u evaluate.py base
until test -s inputs/heterofold.pt; do sleep 15; done
"$py" -u evaluate.py paper
