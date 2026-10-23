# Grouped full-head translator v1

Independent Llama-3.2-3B → Qwen3-4B experiment, seed 1234. Four contiguous groups,
7 source layers → 9 target layers each. Per token and role, flatten all 7×8×128
source features → bias-free 7168→1024 Linear + GELU → independent bias-free
1024→1024 output for each target layer. K and V have separate networks; every
target head can depend on every source head in its group. No cross-group or
cross-token connections. Exactly 134,217,728 trainable parameters.

Stage A: existing parent MCQ 2048 train / 256 validation caches; 2 epochs,
batch 8, lr 1e-3, NMSE + cosine loss; select best epoch by validation loss.
Stage B: existing UFD four-task 2048 contexts × 4 epochs / batch 8 = 1024 steps,
lr 1e-4, same exact vocabulary forward KL, Native/student mixed prefixes,
64-token rollout cap and up to 8 supervised positions. Train the complete new
translator. Both LLMs are frozen. No Reader or extra per-head residual.
Use fixed final Stage-B checkpoint. This changes architecture and trainable
scope together; it is not a grouping-only or equal-parameter ablation.

Reuse pinned parent Full-Sync token/cache/position interfaces and data splits.
Evaluation reuses identity-checked Native and old UFD outputs; evaluates only
the new translator on the same six task subsets. GSM8K uses the existing 32
historical examples and 4096-token cap. MMLU-Pro remains a custom held-out split.
No inference latency or strict zero-prefill claim is made for this accuracy path.

`python -m unittest -v test_translator test_queue` runs CPU structural tests.
`python experiment.py prepare` pins source, config, data and parent identities.
`python run_pipeline.py` reports readiness without launching.
`python run_pipeline.py --launch-if-ready` is idempotent and starts a detached
pipeline only after the prior reward experiment's successful completion marker,
matching manifest and released pipeline lock, with ≥20 GiB GPU memory free and
≥28 GiB disk free. Failures and unexpected exits require review; no automatic
retry. A Codex heartbeat invokes the launcher after the dependency completes.

GPU smoke tests run before each training stage: real cached reconstruction,
Native cache parity, Native/student-prefix backward passes through all groups,
finite gradients, frozen Receiver. Smoke weights are discarded before formal
training. They are queued with training, not run concurrently with the prior job.
Checkpoints support manual resume after review; outputs stay in this directory.
Parent files, caches, checkpoints and running jobs are never edited or stopped.
