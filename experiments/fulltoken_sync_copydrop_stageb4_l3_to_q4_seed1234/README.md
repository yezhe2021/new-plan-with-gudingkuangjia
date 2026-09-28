# Full-Sync choice-KL Stage-B, four epochs

Llama3.2-3B to Qwen3-4B on the same balanced OpenBookQA,
ARC-Challenge and MMLU-Pro manifests as the original Full-Sync retraining.

The original selected Stage-A epoch-4 checkpoint is frozen. A new Residual64
Stage-B adapter is initialized with the original seed and trained for four
complete epochs: 3072 training examples, effective batch 8, 384 optimizer
steps per epoch and 1536 steps in total. Choice-only KL, learning rate 1e-4,
gradient clip 30, model/token alignment and validation selection are unchanged.
The 384 held-out test examples are evaluated once after checkpoint selection.

Run `bash run_stageb.sh` and view `logs/pipeline.log`. K/V caches are not saved.
If another job temporarily occupies the GPU, `bash wait_for_gpu.sh` starts the
same run when at least 24000 MiB is free; its wait log is `logs/queue.log`.

See [RESULTS.md](RESULTS.md) for the completed validation trajectory and
test384 results by dataset.
