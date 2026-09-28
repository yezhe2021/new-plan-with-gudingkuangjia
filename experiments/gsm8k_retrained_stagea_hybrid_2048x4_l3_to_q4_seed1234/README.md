# GSM8K Hybrid Stage-B: 2048 samples x 4 epochs

This experiment tests whether more distinct training problems improve the
Llama3.2-3B to Qwen3-4B Full-Sync Hybrid translator. It reuses and freezes
the GSM8K Stage-A epoch-2 checkpoint.

- Train: 2048 official GSM8K train questions, including the previous 1024.
- Validation: the same 128 training-split questions held out previously.
- Test: the same 128 untouched official-test questions.
- Hybrid Residual64: freshly initialized, four complete epochs, effective
  batch 8, learning rate 1e-4, 1024 optimizer steps total.
- Validation runs only after epoch 4. The epoch-4 checkpoint is selected by
  protocol, then evaluated on test128 once.
- Training records include loss, pre-clip gradient norm and clip indicator.
- Native and translated K/V are held in memory only; no persistent K/V cache.

Run with `bash run_all.sh`; view progress with `tail -f logs/pipeline.log`.

See [RESULTS.md](RESULTS.md) for the completed validation and test results.
