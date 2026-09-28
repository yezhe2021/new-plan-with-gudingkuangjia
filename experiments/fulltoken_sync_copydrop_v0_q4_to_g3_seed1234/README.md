# Qwen3-4B → Gemma3-4B Full-Sync Copy/Drop V0

Concurrent reverse-direction Full-Sync experiment.

- Same balanced 3,072/384/384 OpenBookQA, ARC-Challenge and MMLU-Pro samples.
- Standard prompt; Gemma retains native Question KV and receives translated KV
  for every Gemma Options token.
- UTF-8 synchronized units with ordered copy/drop on the Gemma token grid.
- Full36 HeadMix128 Writer maps Qwen `[36,T,8,128]` to Gemma
  `[34,T,4,256]`; independent bias-free K/V mappings.
- Stage-A: four epochs / 1,536 steps. Stage-B: frozen Stage-A plus Residual64,
  two epochs / 768 steps. Effective batch 8 and gradient clip 30.
- Native KV is temporarily retained in CPU RAM only and never persisted.

`bash run_pipeline.sh` runs tests, alignment audit, smoke, training and final
evaluation. It may run concurrently with the Qwen→Llama experiment when GPU
capacity permits.
