# Reward-aligned Reader: single GSM8K pilot

Authorized on 2026-10-21. Freeze the existing Stage-A/UFD Stage-B and Qwen base;
continue only the existing rank16 full36 Q/O Reader for256 updates. This is a
feasibility pilot, not an attribution experiment or a paper reproduction.

Reuse the existing512 GSM8K training questions and their aligned Sender states.
Protect all parent train/dev questions from the128 evaluation questions. Report
the historical32 and previously unevaluated96 separately. Only the initialized
Reader and final reward Reader are evaluated. No additional trained baselines.

Training uses temperature1/top-p1 on-policy sampling, two responses/question,
eight questions/update. Binary correctness gates the native-versus-student
sampled-token log-probability difference. No additive KL or CE. Per-trajectory
token means include EOS. Truncations are censored and separately counted.

For V100 memory, replay uses64-token truncated BPTT, retaining every sampled
token's local gradient but detaching previous continuation caches. This is an
explicit approximation to full-sequence policy gradients. Teacher and rollout
passes have no gradients. Reader is disabled for Native cache/teacher and for
the fixed Writer. The transmitted KV prefix is never rewritten.

`run_pipeline.py` performs CPU/data/identity audit, waits for GPU capacity,
captures only missing new-test Sender states, runs mathematical and GPU smoke,
trains, evaluates with unchanged4096-token greedy protocol, and automatically
appends completion/failure to `/hy-tmp/yezhe/异构模型/EXPERIMENT_RESULTS.md`.
Final step256 is primary. No automatic retraining, test selection or upload.

Large checkpoints and Sender states remain on server B. The pipeline lock
prevents duplicate launches. Inspect status/processes before any restart.

Repair: the initial attempt stopped before its first optimizer update because
trajectory IDs (`tokens`) and replay token counts shared a record field name.
Replay counts now use `supervised_tokens`, with a disjoint-field assertion.
Preserve that failed attempt under `history/`; re-audit code, rerun GPU smoke,
and reuse Sender caches only under their original recorded manifest identity.
Training configuration, data and initialization are unchanged.
