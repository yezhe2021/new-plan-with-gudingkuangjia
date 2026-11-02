# RN-KV structural ablations

Two arms distinguish architectural training ablations from the previous inference-only adapter removal.

| Arm | Trained parameters | Adapter | Answer routing at19–30 |
|---|---:|---|---|
| Original RN-KV (existing baseline) |409600|4 trained Rank16 residuals|Memory only; body blocked|
| `memory_only` |81920|Identity from initialization; never trained or used|Memory only; body blocked|
| `keep_both` |409600|Original4 Rank16 residuals|Both translated body and Memory|

“Adapter from the beginning not trained” and “only train Memory Tokens” are the same first arm. It recreates the original deterministic32×2560 prompt initialization and trains it from scratch. It does not load the jointly trained prompt and remove adapters afterward. Omitting adapters is exactly the function of the original zero-up adapter held fixed forever.

The default second arm also trains from scratch with the same budget. It restores body access **only** at19–30; Memory stays accessible there. Other layers continue to block Memory and retain body. Construction always reads all translated body and causal Memory; positions remain L..L+31 and suffix L+32. No changes to memory count, layer groups, positions, output fusion or learned gates.

## Matched training

Exact byte-identical original ARC1024+Hotpot1024 and validation64/task; same seed1234, batches, quotas, Task CE+0.5 Full KL,4epochs/batch8/1024 successful updates, AdamWlr1e-4/wd1e-4/clip1, fixed final epoch4 checkpoint. Native/gold/current-student prefixes use the original4/4/0→3/3/2→2/2/4 schedule. Only each arm's student-generated prefix and structural parameterization differ. Frozen Sender, Recolor, C and Qwen; gradients still flow through Qwen.

`ablation_train.py` follows the original training/checkpoint/validation implementation with arm-scoped files. `ablation_runtime.py` shares original computation, batch generator and AMP transaction handling, with child-owned Native prefix traces and loss logging. Each checkpoint includes arm identity. Failed batches back off scale and retry without committing optimizer updates or dropping examples. Initial/first4 states are saved, then every32 steps.

## Tests and evaluation

Use exactly the existing ARC/Hotpot64 and five new-task64 cohorts (448 questions/arm), preserving their scorers and baseline results. `ablation_evaluate.py` calls the original RN scorer for ARC/Hotpot and the frozen five-task scorer for the rest. Report each arm against full RN-KV and C with per-question changes/intervals. These cohorts have been inspected already; this is controlled structural diagnosis, not a new untouched confirmation benchmark.

CPU checks use a tiny real36-layer Qwen3: same initialization, memory-only equivalence to zero-up adapters, correct keep-both mask, real memory gradients and checkpoint gradient parity. GPU checks exercise the longest train row,4 consecutive real batches per arm, frozen parameter scopes and original trained RN baseline scorer parity. Smoke weights/updates are discarded before fresh training.

`run_pipeline.py launch` runs detached on server B: both GPU checks→memory-only training/evaluation→keep-both training/evaluation→combined report. Other paused experiments remain paused. Pause via `touch controls/pause_requested.json`; remove it and use `run_pipeline.py run` to resume. No automatic upload, additional tuning, GSM8K or ongoing client monitoring.

If the user explicitly selects inference-only keep-both, its `train` config flag is false: the same routing intervention instead loads the existing trained RN checkpoint, runs before memory-only training, and is labeled as a train/inference routing mismatch rather than a matched retraining ablation.

## Cross-server scheduling (2026-11-02)

At the user's request, keep-both moves to server A; memory-only continues uninterrupted on B with automatic evaluation. The original B two-arm orchestrator is replaced by `controls/host_b_memory_only_runner.py`, so it cannot duplicate the second training. A uses `controls/host_a_keep_both_runner.py` for only keep-both Smoke→1024 updates→seven-task evaluation. The experiment code/config/manifest/data remain identical under a task-owned `/hy-tmp/yezhe` mirror; pre-existing A model weights and attnkv are reused after identity/version checks.

A waits for verified assets and at least24GiB free GPU memory. Existing A GPU workloads are left alone. Status is `controls/server_a_status.json`, logs `logs/server_a_keep_both.log`. B's remaining workflow logs are `logs/server_b_memory_only.log` (the running training still writes its inherited pipeline log). Per-arm results live on their respective hosts until collected for the combined report; a host completing its local arm does not imply both arms are finished. Other paused experiments remain paused.

The user subsequently moved keep-both back to B after memory-only finished. A's unstarted waiting queue is terminated and disabled. `controls/host_b_keep_both_runner.py` now runs the second arm's checks→fresh1024-step training→seven-task evaluation→combined summary on B. All scientific code/config/data and the manifest remain unchanged. Log: `logs/server_b_keep_both.log`; operational state: `controls/server_b_keep_both_status.json`.
