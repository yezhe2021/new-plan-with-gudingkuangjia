# GSM8K Generation-only Stage-B Pilot

This independent experiment freezes the existing Llama3.2-3B to Qwen3-4B
Full-Sync Stage-A and trains a newly initialized Residual64 Stage-B using only
teacher-forced CE over the complete original GSM8K answer trajectory.

- Deterministic seed-1234 shuffle of 1,024 official train rows; manifest saved.
- No validation and no test-based checkpoint selection.
- Two epochs, effective batch 8, 256 optimizer steps, LR 1e-4, clip 30.
- Qwen native token0 plus translated question-prefix KV; no native Question KV.
- Sender, Qwen and Stage-A are frozen. Qwen forward retains autograd so loss
  gradients reach only the new Residual64 adapter.
- Standard `Question:\n...\n\nAnswer:` generation protocol; no constructed
  choices and no MCQ supervision.
- Native KVs and frozen Stage-A outputs live only in CPU RAM for the running
  process and are never persisted.

`bash run_train.sh` executes CPU tests, four protocol audits, training, a
32-row test generation evaluation, and a 32-row train generation diagnostic.
Follow with `tail -f logs/pipeline.log`.
