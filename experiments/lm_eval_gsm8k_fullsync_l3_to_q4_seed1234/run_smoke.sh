#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
mkdir -p "$ROOT/results/smoke2_hf" "$ROOT/results/smoke2_custom" "$ROOT/logs"
exec > "$ROOT/logs/smoke.log" 2>&1
COMMON=(--include_path "$ROOT/tasks" --tasks gsm8k_local --num_fewshot 0 --limit 1 --batch_size 1 --device cuda:0 --log_samples --seed 1234 --gen_kwargs max_gen_toks=32)
"$PY" -m lm_eval "${COMMON[@]}" --model hf \
  --model_args "pretrained=/home/yezhe/all_models/models/Qwen/Qwen3-4B,dtype=float16" \
  --output_path "$ROOT/results/smoke2_hf"
"$PY" "$ROOT/run_lm_eval.py" "${COMMON[@]}" --model fullsync_l3_to_q4 \
  --model_args "mode=native_qwen,max_gen_toks=256,max_length=32768,audit_path=$ROOT/results/smoke2_custom_audit.jsonl" \
  --output_path "$ROOT/results/smoke2_custom"
echo "SMOKE_COMPLETED"
