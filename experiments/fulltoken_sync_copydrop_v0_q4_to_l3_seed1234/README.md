# Qwen3-4B → Llama3.2-3B Full-Sync Copy/Drop V0

Reverse-direction counterpart of the Full-Sync experiments.

- The same balanced OpenBookQA, ARC-Challenge and MMLU-Pro IDs: 1,024 train,
  128 validation and 128 test per dataset; seed 1234.
- Standard `Question → Options → Answer:` prompt. Llama retains native Question
  KV; every Llama Options token receives one translated Qwen KV.
- UTF-8 synchronized units with ordered copy/drop on the Llama Options grid.
- Bias-free Full36 MLP Writer, independent K/V and target layers/heads,
  mapping Qwen `[36,T,8,128]` to Llama `[28,T,8,128]`.
- Stage-A: four epochs / 1,536 steps of full-token KV reconstruction.
- Stage-B: frozen Stage-A plus Residual64, two epochs / 768 steps using
  choice-only KL. Effective batch 8; gradient clip 30.
- Native KVs are kept temporarily in CPU RAM during the process and are never
  persisted. Checkpoint selection uses validation; test runs once afterward.

Run `bash run_pipeline.sh`; it performs CPU tests, a full alignment audit, GPU
smoke, Stage-A, Stage-B and final evaluation in order.
