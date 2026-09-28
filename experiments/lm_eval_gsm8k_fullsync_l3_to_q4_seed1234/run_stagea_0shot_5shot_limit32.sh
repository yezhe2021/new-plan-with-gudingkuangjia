#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
COMMON=(
  --include_path "$ROOT/tasks"
  --tasks gsm8k_local
  --limit 32
  --batch_size 1
  --device cuda:0
  --log_samples
  --seed 1234
)

mkdir -p "$ROOT/results" "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/stagea_0shot_5shot_limit32.log") 2>&1

for SHOTS in 0 5; do
  TAG="stagea_${SHOTS}shot_limit32"
  echo "[$(date '+%F %T')] START ${TAG}"
  "$PY" "$ROOT/run_lm_eval.py" "${COMMON[@]}" --num_fewshot "$SHOTS" \
    --model fullsync_l3_to_q4 \
    --model_args "mode=fullsync_stagea,max_gen_toks=256,max_length=32768,audit_path=$ROOT/results/${TAG}_audit.jsonl" \
    --output_path "$ROOT/results/${TAG}"
done

echo "[$(date '+%F %T')] FULL-SYNC STAGE-A 0-SHOT/5-SHOT LIMIT32 COMPLETED"
