# Unified Full-Body Hub: Llama3.2-3B → Qwen3-4B

All tasks use one interface: every token before native Qwen `Answer:` is
provided by Sender→Full-Sync→Stage-A→Stage-B. Qwen token0 remains native.

- Stage-A: one Full28 MLP trained with KV reconstruction on 2,048 GSM8K plus
  1,024 each from OpenBookQA, ARC-Challenge and MMLU-Pro.
- Stage-B: two fresh Residual64 adapters trained from the same frozen Stage-A,
  with deterministic 50% GSM8K / 50% MCQ exposure per epoch.
- `universal_kl`: full-vocabulary trajectory KL only.
- `universal_hybrid`: the same KL + 0.1 gold-token CE + 0.1 EOS CE.
- HellaSwag is OOD evaluation only and never enters training or validation.

No native KV cache is persisted. Only model checkpoints, logs, metrics and
per-sample predictions are written.
