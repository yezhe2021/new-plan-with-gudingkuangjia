# Dynamic Memory on the frozen DST Stage2 base

This is a fresh matched comparison to `dst_stage2_static_memory_retrain_m32_a4_3task_seed1234`. Freeze the exact final DST Stage2 epoch4/step1536 checkpoint (`1d46e308fe1ddbf8297a67210b7f05df0f753842512c6779882477648b7d7cfc`), Sender and Qwen. Train only the same409600 parameters:32 prompt vectors and four Rank16 adapters after construction blocks6/12/18/24. Use identical fresh seed1234 weights, without loading any learned Memory checkpoint into formal training.

## Controlled change

Static construction reads only the fixed public header, then relocates fixed native-projected Memory K to the recipient coordinates. Dynamic construction instead uses the existing original `RNKV.construct` unchanged: Qwen processes P32 once at the actual positions L..L+31, with the **entire current-sample frozen DST body KV** as its past. It learns no token alignment and performs no token autoregression inside Memory construction. The first layer's Memory projections occur before reading the body; later-layer Memory states can depend on that body. Construction-only adapters are enabled as before. The final Memory KV comes from Qwen's own projections and native k_norm/RoPE at those actual positions; no static canonical cache or post-hoc K relocation is used.

The original complete RAW Full-Sync DST cache stays unchanged. All36 answer layers read body+32 Memory slots, the same causal masks and suffix positions L+32 as the trained Static control. Memory is constructed once per example; normal generated-token KV accumulates afterward. Native body is used only for teacher/baseline computation, never to construct the student's Memory at test time.

## Same data, optimization and scoring

Byte-identical3072 train/192 validation/512 test rows and identical batch/prefix schedules from the completed Static group. Train ARC/HotpotQA/GSM8K1024 each,64 validation each;4epochs,1536 accepted updates,effective batch8,AdamW1e-4,weight_decay1e-4,clip1. Full gold answer/rationale CE +0.5 full-vocabulary KL(Native||Student). Native/Student rollout cap64,max8 KL locations; gold CE is not truncated. Prefix quotas per8 are4/4/0,3/3/2,2/2/4,2/2/4 gold/native/student. Frozen Native prefixes reuse the same hashed original files. Failed AMP batches retry with the same RNG and no skipped examples.

Validation is diagnostic; final epoch4 is primary. Resume restores optimizer/scaler/RNG and accepted-update accounting. Test eight64-question cohorts using unchanged prompts, labels, likelihood scoring, QA64/GSM512 caps and both GSM numeric metrics. Compare DST alone, trained Static on DST, new trained Dynamic on DST, historical Recolor+C+Static and Native. Old zero-shot Static outcomes remain available in per-question baseline records.

## Checks and queue

CPU checks use a real tiny36-layer Qwen to verify that same-length changed body KV changes later-layer dynamic Memory while the equally weighted static control stays identical. They verify first-layer invariance, unchanged body, construction/answer causality, complete81-token gold supervision and checkpoint-gradient parity. GPU checks reproduce DST/Native/trained-Static predictions on one question per task. They verify exact fresh initialization, bitwise original dynamic-construction parity, controlled same-length body-value dependence, the longest training example of each task and four consecutive effective-batch8 updates. These updates are disposable; formal training restarts from the matched fresh initialization.

`python run_pipeline.py launch` runs a detached locked queue: prepare → scorer imports → CPU checks → endpoint parity → GPU smoke → fresh full training → final evaluation. Only Memory/adapters train; frozen DST/Sender/Receiver are checked after updates. No new large disk cache or checkpoint duplication is introduced. Pause requests are honored at batch/example boundaries. No unrelated jobs or automatic uploads are started.

Read `runner_status.json`, `status.json` and `logs/pipeline.log`. Final tables, paired bootstrap intervals and per-question changes appear in `RESULTS.md`, `summary.json` and `evaluation/`.

Dynamic-versus-static holds answer positions, access masks and allocated capacity constant; it tests the benefit of sample-conditioned construction under this budget. It does not by itself prove factual recovery or Native-state recovery. Comparisons to DST alone still include32-position shifts. Previously inspected64-question cohorts and one seed are exploratory; retain original labels including the known contradictory GSM952 row.
