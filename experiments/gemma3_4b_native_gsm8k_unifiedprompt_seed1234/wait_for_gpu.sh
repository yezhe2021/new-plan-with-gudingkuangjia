#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/yezhe/异构模型/gemma3_4b_native_gsm8k_unifiedprompt_seed1234"
cd "$ROOT"
mkdir -p logs
while true; do
  if pgrep -af "gemma3_4b_native_gsm8k_unifiedprompt_seed1234/native_eval.py" | grep -v grep >/dev/null; then
    exit 0
  fi
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  printf '[%s] free=%s MiB threshold=25000 MiB\n' "$(date '+%F %T')" "$free_mib" >> logs/waiter.log
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 25000 )); then
    nohup bash run.sh > logs/pipeline.log 2>&1 &
    echo $! > logs/pipeline.pid
    printf '[%s] started pid=%s\n' "$(date '+%F %T')" "$(cat logs/pipeline.pid)" >> logs/waiter.log
    exit 0
  fi
  sleep 300
done
