# GSM8K-retrained Stage-A + Hybrid Stage-B

This experiment tests whether the earlier Stage-A domain mismatch caused the
GSM8K transfer gap. It preserves the Llama3.2-3B to Qwen3-4B Full-Sync
alignment and Full28 MLP architecture, but trains a fresh Stage-A on the exact
lm-eval GSM8K question prefix before training one fresh Hybrid Residual64.

- Exact old train1024 indices; next 128 shuffled train rows are validation.
- Official test is untouched and used once for final test128 generation.
- Stage-A: two epochs, KV reconstruction, LR 1e-3, effective batch 8.
- Stage-B: two epochs, trajectory KL + 0.1 CE + 0.1 EOS, LR 1e-4.
- Both stages use validation-only checkpoint selection.
- Final inference regenerates only the new Stage-A and new Hybrid conditions;
  old conditions are referenced from Generation Stage-B v2.

Run with `bash run_all.sh`; monitor with `tail -f logs/pipeline.log`.

See [`RESULTS.md`](RESULTS.md) for the selected checkpoints, test128 metrics,
comparison with the previous Generation Stage-B v2 run, and conclusion.
