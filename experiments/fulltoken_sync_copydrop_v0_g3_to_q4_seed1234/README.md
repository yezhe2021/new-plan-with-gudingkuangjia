# Gemma3-4B → Qwen3-4B Full-Sync Copy/Drop V0

Independent replication of the Llama3.2-3B → Qwen3-4B Full-Sync experiment,
changing only the Sender and the necessary Writer head geometry.

- Exactly the same balanced OpenBookQA, ARC-Challenge and MMLU-Pro IDs:
  1,024 train, 128 validation and 128 test per dataset; seed 1234.
- Standard `Question → Options → Answer:` prompt. Qwen keeps its native
  Question KV; every Qwen Options token receives one translated Gemma KV.
- Tokenizer-independent UTF-8 synchronized units, ordered copy/drop on the
  Qwen Options grid. Gemma uses its native pre-RoPE KV representation.
- Full34 HeadMix256 Writer, K/V-independent, bias-free, 34×4×256 to 36×8×128.
  Stage-A learns full-token KV reconstruction for four epochs (1,536 steps).
- Frozen Stage-A plus Residual64 Stage-B learns choice-only KL for two epochs
  (768 steps). Effective batch 8, 64-token microchunks, gradient clip 30.
- Checkpoints selected on validation. Test evaluated once after both stages.
- Source/target native KV tensors are temporarily retained in CPU RAM within
  the running process; they are never written to disk. Logs, checkpoints and
  per-sample choice outputs are kept in `runs/retrain/` or `logs/`.

Run `bash run_pipeline.sh`. It performs CPU tests and a full alignment audit,
then immediately runs a GPU smoke test followed by the full study.
Follow progress with `tail -f logs/pipeline.log` after launch.

The Qwen-native Oracle predictions are compared by test ID against the earlier
Llama Full-Sync experiment, so any Receiver/protocol drift is reported.
