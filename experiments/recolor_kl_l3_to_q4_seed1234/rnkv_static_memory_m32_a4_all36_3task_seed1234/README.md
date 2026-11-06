# Static Memory versus Dynamic Memory

Llama3.2-3B → frozen HeteroFold Recolor+C → Qwen3-4B.

The completed `rnkv_keep_both_all36_arc_hotpot_gsm_seed1234` is the dynamic reference. This experiment trains a fresh M32 + four Rank16 adapters at blocks6/12/18/24 (409600 trainable parameters), with identical initialization, exact prepared data, batch schedules, mixed-prefix rules, optimizer, losses and final-step selection. No Sender, Receiver, Recolor or C weights train.

Dynamic construction reads the fixed public header and the sample's translated body. Static construction uses the same Qwen/P/adapters pipeline but **only the fixed public header**, at canonical positions H..H+31. Canonical construction accepts no sample arguments. Its output KV is therefore shared by every input at fixed learned parameters. This is a trained static control, rather than body masking on a trained dynamic checkpoint.

The attachment step deterministically rephases the already-normalized, post-RoPE Memory K to recipient positions L..L+31. It neither applies k_norm twice nor changes V. Answer-time caches and masks retain the translated body and all32 Memory slots in all36 layers; suffix positions start at L+32. Canonical construction remains differentiable through the frozen Qwen, and is recomputed per training microexample. Evaluation caches it only while parameter versions match.

Data: ARC/HotpotQA/GSM8K,1024 train and64 validation each. Four epochs, batch8,1536 accepted updates, AdamW1e-4, task CE+.5full-vocabulary KL. Gold CE uses full targets. Native/Student rollout budget64, KL positions8. Eight fixed64-question test cohorts, QA64/GSM512 generation caps. Preserve benchmark labels and both GSM numeric scorers. Native prefix and completed baseline files are reused read-only with provenance.

`run_pipeline.py launch` starts a detached, locked pipeline: prepare → real tiny-Qwen CPU checks → dynamic endpoint parity for eight tasks → longest-example and consecutive-batch GPU smoke → fresh full training → automatic final evaluation and paired comparisons. Smoke updates never warm-start formal training. Checkpoints include optimizer/scaler/RNG and resume only accepted updates. Pause requests are honored at batch boundaries.

The contrast estimates the usefulness of sample-conditioned construction relative to a generic learned KV prefix. It does not by itself establish factual recovery, Native-state recovery, or absence of information in dynamic Memory if scores are similar. Canonical header-to-memory distance is fixed, unlike dynamic construction; this is part of constructing a strictly input-independent control. One seed and64 questions/task are exploratory.

No large disk cache is created. Other paused jobs remain paused. Results are written to `summary.json`, `RESULTS.md`, and `evaluation/<task>/per_sample/`.
