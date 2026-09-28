# GSM8K Gemma3-4B → Qwen3-4B Full-Sync generation

Sender swap against the Llama3.2-3B → Qwen3-4B GSM8K 2048×4 run. The
question-only prompt, Qwen receiver, train/validation/test IDs, Hybrid
generation loss, and scoring stay fixed. The sender is Gemma3-4B and Stage-A
is freshly trained using its Full34 HeadMix256 writer.

- Stage-A: first 1024 training questions; 2 epochs; KV reconstruction;
  independent bias-free K/V; validation chooses the checkpoint.
- Stage-B: frozen Stage-A + freshly initialized Residual64; 2048 training
  questions; 4 epochs; effective batch 8; trajectory KL + 0.1 answer CE
  + 0.1 EOS CE. Validation only after epoch 4; test128 only at the end.
- Full UTF-8 synchronized copy/drop of Gemma question K/V to Qwen's question
  token grid. Qwen token0 remains native. No gold answer enters the sender.
- The Gemma Sender uses the existing FP32 native capture protocol.
- Native K/V remain in CPU RAM during the run and are not written to disk.

Run `bash run_all.sh`. Live log: `tail -f logs/pipeline.log`.
