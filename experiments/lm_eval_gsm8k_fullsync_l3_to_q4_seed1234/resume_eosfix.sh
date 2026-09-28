#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
LM_ARGS=(--include_path "$ROOT/tasks" --tasks gsm8k_local --num_fewshot 0 --limit 10 --batch_size 1 --device cuda:0 --log_samples --seed 1234)
exec > >(tee -a "$ROOT/logs/pilot.log") 2>&1
"$PY" "$ROOT/run_lm_eval.py" "${LM_ARGS[@]}" --model fullsync_l3_to_q4 \
  --model_args "mode=native_qwen,max_gen_toks=256,max_length=32768,audit_path=$ROOT/results/qwen_native_custom_eager_eosfix_audit.jsonl" \
  --output_path "$ROOT/results/qwen_native_custom_eager_eosfix_0shot_limit10"
"$PY" "$ROOT/compare_native.py"
"$PY" "$ROOT/run_lm_eval.py" "${LM_ARGS[@]}" --model fullsync_l3_to_q4 \
  --model_args "mode=fullsync,max_gen_toks=256,max_length=32768,audit_path=$ROOT/results/fullsync_backend_eager_eosfix_audit.jsonl" \
  --output_path "$ROOT/results/fullsync_genb_eager_eosfix_0shot_limit10"
echo "[$(date '+%F %T')] EOS-FIXED CUSTOM NATIVE AUDIT AND FULL-SYNC PILOT COMPLETED"
