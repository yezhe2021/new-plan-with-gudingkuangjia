# GSM8K frozen Stage-A + four-epoch Hybrid Stage-B

This experiment isolates Stage-B training sufficiency. It reuses and freezes
the GSM8K-trained Stage-A epoch-2 checkpoint from the preceding experiment,
initializes a fresh zero-residual Hybrid Residual64 adapter, and trains it for
four complete epochs (512 optimizer steps).

- Same 1024 train / 128 validation / untouched official test128 split.
- Same Full-Sync Llama3.2-3B to Qwen3-4B cache protocol.
- Same effective batch 8, LR 1e-4, and Hybrid trajectory objective.
- Validation is run after every epoch; the final test is run only for the
  validation-selected checkpoint.
- Epoch 2 remains directly comparable with the preceding two-epoch run.
- No persistent native or translated K/V cache is written.

Run with `bash run_all.sh`; monitor with `tail -f logs/pipeline.log`.

See [RESULTS.md](RESULTS.md) for the completed validation trajectory and
official test128 comparison.
