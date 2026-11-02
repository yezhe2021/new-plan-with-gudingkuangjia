# RN-KV adapter-off inference ablation

Load the existing final RN-KV epoch4/step1024 checkpoint. Bypass the four construction-only Rank16 Memory adapters after blocks6/12/18/24 by returning the input hidden-state tensor exactly. Keep the jointly trained32 memory embeddings and all other weights, positions, masks, replacement layers, prompts and scoring fixed. No retraining and no native body access.

Evaluate on the same ARC64/HotpotQA64. Reuse immutable parent per-question C/full-RN-KV/Native baselines after checking enabled-model parity. Compare adapter-off vs full-RN-KV and vs C with paired question bootstrap intervals and rescued/harmed IDs. This answers how the already-trained model behaves when its adapters are bypassed; it does not establish whether training without adapters would work equally well.

Smoke verifies enabled-checkpoint predictions/scores/generation, one exact bypass per configured adapter per memory construction, unchanged prompt/weights/cache prefix and positions, unchanged memory K/V of layers1-6, and an actual change in later memory K/V. Checks use fixed first task rows and longest Hotpot row for correctness, never effect-based filtering. `run_pipeline.py launch` starts detached; logs are `logs/pipeline.log`. Pause with `touch controls/pause_requested.json` at a question boundary.
