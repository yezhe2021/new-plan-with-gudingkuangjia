# Unified Full-Body Hub KL-only 4-epoch retraining: Llama3.2-3B → Qwen3-4B

All tasks use one interface: every token before native Qwen `Answer:` is
provided by Sender→Full-Sync→Stage-A→Stage-B. Qwen token0 remains native.

- Stage-A: one Full28 MLP retrained for 4 epochs with KV reconstruction on
  2,048 GSM8K plus 1,024 each from OpenBookQA, ARC-Challenge and MMLU-Pro.
- Stage-B: one fresh Residual64 adapter trained for 4 epochs, with deterministic
  50% GSM8K / 50% MCQ exposure per epoch.
- The only Stage-B objective is `universal_kl`: full-vocabulary trajectory KL.
- The Hybrid branch is intentionally omitted so the experiment isolates whether
  longer KL-only training improves GSM8K under the unchanged unified protocol.
- HellaSwag is OOD evaluation only and never enters training or validation.

No native KV cache is persisted. Only model checkpoints, logs, metrics and
per-sample predictions are written.
