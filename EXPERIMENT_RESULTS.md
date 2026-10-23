# Experiment results

Machine-generated from archived raw files by `tools/build_catalog.py`.
This index does not reinterpret or replace each experiment's original results.

## frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234

- Status: `completed` (archived result artifacts)
- Original path: `异构模型/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234`
- Recorded configuration: seed=1234
- Config: [config.json](experiments/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234/config.json)
- Summary: [gsm8k_summary.json](experiments/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234/runs/gemma/gsm8k_summary.json)
- Summary: [hellaswag_summary.json](experiments/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234/runs/gemma/hellaswag_summary.json)
- Summary: [longbench_v2_summary.json](experiments/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234/runs/gemma/longbench_v2_summary.json)
- Summary: [summary.json](experiments/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234/runs/gemma/summary.json)
- Summary: [summary.json](experiments/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234/runs/gsm8k_generation/llama/summary.json)
- Summary: [gsm8k_summary.json](experiments/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234/runs/llama/gsm8k_summary.json)
- Per Sample: [gsm8k_per_sample.jsonl](experiments/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234/runs/gemma/gsm8k_per_sample.jsonl)
- Per Sample: [hellaswag_per_sample.jsonl](experiments/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234/runs/gemma/hellaswag_per_sample.jsonl)
- Per Sample: [longbench_v2_per_sample.jsonl](experiments/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234/runs/gemma/longbench_v2_per_sample.jsonl)
- Per Sample: [per_sample.jsonl](experiments/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234/runs/gsm8k_generation/llama/per_sample.jsonl)
- Log: [gsm8k_generation.log](experiments/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234/logs/gsm8k_generation.log)
- Log: [gsm8k_generation_quick.log](experiments/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234/logs/gsm8k_generation_quick.log)
- Original report: [RESULTS.md](experiments/frozen_fullsync_newbenchmarks_l3_g3_to_q4_seed1234/RESULTS.md)

Automatically extracted metrics from the linked JSON:

| JSON field | Value |
|---|---:|
| `count` | 128 |
| `sender_full_native.correct` | 36 |
| `sender_full_native.accuracy` | 0.28125 |
| `qwen_full_native.correct` | 58 |
| `qwen_full_native.accuracy` | 0.453125 |
| `native_oracle.correct` | 58 |
| `native_oracle.accuracy` | 0.453125 |
| `stage_a.correct` | 35 |
| `stage_a.accuracy` | 0.2734375 |
| `stage_a.both_with_oracle_correct` | 26 |
| `stage_a.writer_only_correct` | 9 |
| `stage_a.oracle_only_correct` | 32 |

## fulltoken_sync_copydrop_stageb4_l3_to_q4_seed1234

- Status: `completed` (archived result artifacts)
- Original path: `fulltoken_sync_copydrop_stageb4_l3_to_q4_seed1234`
- Summary: [summary.json](experiments/fulltoken_sync_copydrop_stageb4_l3_to_q4_seed1234/runs/retrain/results/summary.json)
- Per Sample: [per_sample.jsonl](experiments/fulltoken_sync_copydrop_stageb4_l3_to_q4_seed1234/runs/retrain/results/per_sample.jsonl)
- Log: [pipeline.log](experiments/fulltoken_sync_copydrop_stageb4_l3_to_q4_seed1234/logs/pipeline.log)
- Log: [queue.log](experiments/fulltoken_sync_copydrop_stageb4_l3_to_q4_seed1234/logs/queue.log)
- Original report: [RESULTS.md](experiments/fulltoken_sync_copydrop_stageb4_l3_to_q4_seed1234/RESULTS.md)
- Checkpoint selection: [stage_b/selection.json](experiments/fulltoken_sync_copydrop_stageb4_l3_to_q4_seed1234/runs/retrain/stage_b/selection.json)

Automatically extracted metrics from the linked JSON:

| JSON field | Value |
|---|---:|
| `stage_a.correct` | 0.4427083333333333 |
| `stage_a.oracle_correct` | 0.7161458333333334 |
| `stage_a.count` | 384 |
| `stage_b.correct` | 0.6145833333333334 |
| `stage_b.oracle_correct` | 0.7161458333333334 |
| `stage_b.count` | 384 |

## fulltoken_sync_copydrop_v0_g3_to_q4_seed1234

- Status: `completed` (archived result artifacts)
- Original path: `异构模型/fulltoken_sync_copydrop_v0_g3_to_q4_seed1234`
- Recorded configuration: train_samples=3072, test_samples=384, stage_a_epochs=4, stage_b_epochs=2
- Config: [training_config.json](experiments/fulltoken_sync_copydrop_v0_g3_to_q4_seed1234/runs/retrain/training_config.json)
- Summary: [summary.json](experiments/fulltoken_sync_copydrop_v0_g3_to_q4_seed1234/runs/retrain/results/summary.json)
- Per Sample: [per_sample.jsonl](experiments/fulltoken_sync_copydrop_v0_g3_to_q4_seed1234/runs/retrain/results/per_sample.jsonl)
- Log: [noalign_full_gemma_tokens.log](experiments/fulltoken_sync_copydrop_v0_g3_to_q4_seed1234/logs/noalign_full_gemma_tokens.log)
- Log: [pipeline.log](experiments/fulltoken_sync_copydrop_v0_g3_to_q4_seed1234/logs/pipeline.log)
- Original report: [RESULTS.md](experiments/fulltoken_sync_copydrop_v0_g3_to_q4_seed1234/RESULTS.md)
- Checkpoint selection: [stage_a/selection.json](experiments/fulltoken_sync_copydrop_v0_g3_to_q4_seed1234/runs/retrain/stage_a/selection.json)
- Checkpoint selection: [stage_b/selection.json](experiments/fulltoken_sync_copydrop_v0_g3_to_q4_seed1234/runs/retrain/stage_b/selection.json)

Automatically extracted metrics from the linked JSON:

| JSON field | Value |
|---|---:|
| `stage_a.correct` | 0.4114583333333333 |
| `stage_a.oracle_correct` | 0.7161458333333334 |
| `stage_a.count` | 384 |
| `stage_b.correct` | 0.6380208333333334 |
| `stage_b.oracle_correct` | 0.7161458333333334 |
| `stage_b.count` | 384 |
| `by_dataset.arc_challenge.count` | 128 |
| `by_dataset.arc_challenge.stage_a_correct` | 66 |
| `by_dataset.arc_challenge.stage_b_correct` | 105 |
| `by_dataset.arc_challenge.oracle_correct` | 115 |
| `by_dataset.arc_challenge.both_stage_b_oracle_correct` | 102 |
| `by_dataset.arc_challenge.stage_b_only_correct` | 3 |

## fulltoken_sync_copydrop_v0_l3_to_q4_seed1234

- Status: `completed` (archived result artifacts)
- Original path: `异构模型/fulltoken_sync_copydrop_v0_l3_to_q4_seed1234`
- Recorded configuration: train_samples=3072, test_samples=384, stage_a_epochs=4, stage_b_epochs=2
- Config: [training_config.json](experiments/fulltoken_sync_copydrop_v0_l3_to_q4_seed1234/runs/retrain/training_config.json)
- Summary: [summary.json](experiments/fulltoken_sync_copydrop_v0_l3_to_q4_seed1234/runs/retrain/results/summary.json)
- Per Sample: [per_sample.jsonl](experiments/fulltoken_sync_copydrop_v0_l3_to_q4_seed1234/runs/retrain/results/per_sample.jsonl)
- Per Sample: [per_sample.jsonl](experiments/fulltoken_sync_copydrop_v0_l3_to_q4_seed1234/runs/study/per_sample.jsonl)
- Per Sample: [smoke_per_sample.jsonl](experiments/fulltoken_sync_copydrop_v0_l3_to_q4_seed1234/runs/study/smoke_per_sample.jsonl)
- Log: [alignment_audit.log](experiments/fulltoken_sync_copydrop_v0_l3_to_q4_seed1234/alignment_audit.log)
- Log: [evaluation.log](experiments/fulltoken_sync_copydrop_v0_l3_to_q4_seed1234/evaluation.log)

Automatically extracted metrics from the linked JSON:

| JSON field | Value |
|---|---:|
| `stage_a.correct` | 0.4427083333333333 |
| `stage_a.oracle_correct` | 0.7161458333333334 |
| `stage_a.count` | 384 |
| `stage_b.correct` | 0.6145833333333334 |
| `stage_b.oracle_correct` | 0.7161458333333334 |
| `stage_b.count` | 384 |

## fulltoken_sync_copydrop_v0_q4_to_g3_seed1234

- Status: `completed` (archived result artifacts)
- Original path: `/home/yezhe/异构模型/fulltoken_sync_copydrop_v0_q4_to_g3_seed1234`
- Recorded configuration: train_samples=3072, test_samples=384, stage_a_epochs=4, stage_b_epochs=2
- Config: [training_config.json](experiments/fulltoken_sync_copydrop_v0_q4_to_g3_seed1234/runs/retrain/training_config.json)
- Config: [training_config.json](experiments/fulltoken_sync_copydrop_v0_q4_to_g3_seed1234/runs/retrain_smoke/training_config.json)
- Summary: [summary.json](experiments/fulltoken_sync_copydrop_v0_q4_to_g3_seed1234/runs/retrain/results/summary.json)
- Per Sample: [per_sample.jsonl](experiments/fulltoken_sync_copydrop_v0_q4_to_g3_seed1234/runs/retrain/results/per_sample.jsonl)
- Per Sample: [per_sample.jsonl](experiments/fulltoken_sync_copydrop_v0_q4_to_g3_seed1234/runs/retrain_smoke/results/per_sample.jsonl)
- Log: [memorysafe_restart.log](experiments/fulltoken_sync_copydrop_v0_q4_to_g3_seed1234/logs/memorysafe_restart.log)
- Log: [pipeline.log](experiments/fulltoken_sync_copydrop_v0_q4_to_g3_seed1234/logs/pipeline.log)
- Checkpoint selection: [stage_a/selection.json](experiments/fulltoken_sync_copydrop_v0_q4_to_g3_seed1234/runs/retrain/stage_a/selection.json)
- Checkpoint selection: [stage_b/selection.json](experiments/fulltoken_sync_copydrop_v0_q4_to_g3_seed1234/runs/retrain/stage_b/selection.json)

Automatically extracted metrics from the linked JSON:

| JSON field | Value |
|---|---:|
| `stage_a.correct` | 0.6197916666666666 |
| `stage_a.oracle_correct` | 0.6380208333333334 |
| `stage_a.count` | 384 |
| `stage_b.correct` | 0.6302083333333334 |
| `stage_b.oracle_correct` | 0.6380208333333334 |
| `stage_b.count` | 384 |
| `by_dataset.arc_challenge.count` | 128 |
| `by_dataset.arc_challenge.stage_a_correct` | 106 |
| `by_dataset.arc_challenge.stage_b_correct` | 106 |
| `by_dataset.arc_challenge.oracle_correct` | 106 |
| `by_dataset.arc_challenge.both_stage_b_oracle_correct` | 98 |
| `by_dataset.arc_challenge.stage_b_only_correct` | 8 |

## fulltoken_sync_copydrop_v0_q4_to_l3_seed1234

- Status: `completed` (archived result artifacts)
- Original path: `异构模型/fulltoken_sync_copydrop_v0_q4_to_l3_seed1234`
- Recorded configuration: train_samples=3072, test_samples=384, stage_a_epochs=4, stage_b_epochs=2
- Config: [training_config.json](experiments/fulltoken_sync_copydrop_v0_q4_to_l3_seed1234/runs/retrain/training_config.json)
- Summary: [summary.json](experiments/fulltoken_sync_copydrop_v0_q4_to_l3_seed1234/runs/retrain/results/summary.json)
- Per Sample: [per_sample.jsonl](experiments/fulltoken_sync_copydrop_v0_q4_to_l3_seed1234/runs/retrain/results/per_sample.jsonl)
- Log: [pipeline.log](experiments/fulltoken_sync_copydrop_v0_q4_to_l3_seed1234/logs/pipeline.log)
- Checkpoint selection: [stage_a/selection.json](experiments/fulltoken_sync_copydrop_v0_q4_to_l3_seed1234/runs/retrain/stage_a/selection.json)
- Checkpoint selection: [stage_b/selection.json](experiments/fulltoken_sync_copydrop_v0_q4_to_l3_seed1234/runs/retrain/stage_b/selection.json)

Automatically extracted metrics from the linked JSON:

| JSON field | Value |
|---|---:|
| `stage_a.correct` | 0.6067708333333334 |
| `stage_a.oracle_correct` | 0.6197916666666666 |
| `stage_a.count` | 384 |
| `stage_b.correct` | 0.6197916666666666 |
| `stage_b.oracle_correct` | 0.6197916666666666 |
| `stage_b.count` | 384 |
| `by_dataset.arc_challenge.count` | 128 |
| `by_dataset.arc_challenge.stage_a_correct` | 111 |
| `by_dataset.arc_challenge.stage_b_correct` | 108 |
| `by_dataset.arc_challenge.oracle_correct` | 105 |
| `by_dataset.arc_challenge.both_stage_b_oracle_correct` | 96 |
| `by_dataset.arc_challenge.stage_b_only_correct` | 12 |

## gsm8k_fullsync_g3_to_q4_stagea_hybrid2048x4_seed1234

- Status: `completed` (archived result artifacts)
- Original path: `/home/yezhe/异构模型/gsm8k_fullsync_g3_to_q4_stagea_hybrid2048x4_seed1234`
- Recorded configuration: seed=1234, train_samples=2048, val_samples=128, test_samples=128, stage_a_epochs=2, stage_b_epochs=4
- Config: [config.json](experiments/gsm8k_fullsync_g3_to_q4_stagea_hybrid2048x4_seed1234/config.json)
- Summary: [selection_summary.json](experiments/gsm8k_fullsync_g3_to_q4_stagea_hybrid2048x4_seed1234/runs/selection_summary.json)
- Summary: [test128_summary.json](experiments/gsm8k_fullsync_g3_to_q4_stagea_hybrid2048x4_seed1234/runs/test128_summary.json)
- Per Sample: [test128_per_sample.jsonl](experiments/gsm8k_fullsync_g3_to_q4_stagea_hybrid2048x4_seed1234/runs/test128_per_sample.jsonl)
- Log: [launcher.log](experiments/gsm8k_fullsync_g3_to_q4_stagea_hybrid2048x4_seed1234/logs/launcher.log)
- Log: [pipeline.log](experiments/gsm8k_fullsync_g3_to_q4_stagea_hybrid2048x4_seed1234/logs/pipeline.log)
- Original report: [RESULTS.md](experiments/gsm8k_fullsync_g3_to_q4_stagea_hybrid2048x4_seed1234/RESULTS.md)
- Checkpoint selection: [hybrid/selection.json](experiments/gsm8k_fullsync_g3_to_q4_stagea_hybrid2048x4_seed1234/runs/hybrid/selection.json)
- Checkpoint selection: [stage_a/selection.json](experiments/gsm8k_fullsync_g3_to_q4_stagea_hybrid2048x4_seed1234/runs/stage_a/selection.json)

Automatically extracted metrics from the linked JSON:

| JSON field | Value |
|---|---:|
| `gemma_stage_a.count` | 128 |
| `gemma_stage_a.strict_accuracy` | 0.0 |
| `gemma_stage_a.flexible_accuracy` | 0.0 |
| `gemma_stage_a.first_hash_accuracy` | 0.0 |
| `gemma_stage_a_hybrid2048x4.count` | 128 |
| `gemma_stage_a_hybrid2048x4.strict_accuracy` | 0.0078125 |
| `gemma_stage_a_hybrid2048x4.flexible_accuracy` | 0.671875 |
| `gemma_stage_a_hybrid2048x4.first_hash_accuracy` | 0.0078125 |

## gsm8k_gen_stageb_train1024_l3_to_q4_seed1234

- Status: `completed` (archived result artifacts)
- Original path: `gsm8k_gen_stageb_train1024_l3_to_q4_seed1234`
- Recorded configuration: seed=1234, train_samples=1024, test_samples=32
- Config: [config.json](experiments/gsm8k_gen_stageb_train1024_l3_to_q4_seed1234/config.json)
- Summary: [summary.json](experiments/gsm8k_gen_stageb_train1024_l3_to_q4_seed1234/runs/preflight_old_stage_b/summary.json)
- Summary: [summary.json](experiments/gsm8k_gen_stageb_train1024_l3_to_q4_seed1234/runs/study/results/summary.json)
- Summary: [training_summary.json](experiments/gsm8k_gen_stageb_train1024_l3_to_q4_seed1234/runs/study/training_summary.json)
- Per Sample: [per_sample.jsonl](experiments/gsm8k_gen_stageb_train1024_l3_to_q4_seed1234/runs/preflight_old_stage_b/per_sample.jsonl)
- Per Sample: [test_per_sample.jsonl](experiments/gsm8k_gen_stageb_train1024_l3_to_q4_seed1234/runs/smoke/results/test_per_sample.jsonl)
- Per Sample: [train_per_sample.jsonl](experiments/gsm8k_gen_stageb_train1024_l3_to_q4_seed1234/runs/smoke/results/train_per_sample.jsonl)
- Per Sample: [test_per_sample.jsonl](experiments/gsm8k_gen_stageb_train1024_l3_to_q4_seed1234/runs/study/results/test_per_sample.jsonl)
- Log: [pipeline.log](experiments/gsm8k_gen_stageb_train1024_l3_to_q4_seed1234/logs/pipeline.log)
- Original report: [RESULTS.md](experiments/gsm8k_gen_stageb_train1024_l3_to_q4_seed1234/RESULTS.md)

Automatically extracted metrics from the linked JSON:

| JSON field | Value |
|---|---:|
| `test.count` | 32 |
| `test.qwen_native.correct` | 12 |
| `test.qwen_native.accuracy` | 0.375 |
| `test.native_cache_oracle.correct` | 12 |
| `test.native_cache_oracle.accuracy` | 0.375 |
| `test.stage_a.correct` | 0 |
| `test.stage_a.accuracy` | 0.0 |
| `test.old_mcq_stage_b.correct` | 0 |
| `test.old_mcq_stage_b.accuracy` | 0.0 |
| `test.new_generation_stage_b.correct` | 4 |
| `test.new_generation_stage_b.accuracy` | 0.125 |
| `train.count` | 32 |

## gsm8k_gen_stageb_v2_l3_to_q4_seed1234

- Status: `completed` (archived result artifacts)
- Original path: `gsm8k_gen_stageb_v2_l3_to_q4_seed1234`
- Recorded configuration: seed=1234, train_samples=1024, val_samples=128, test_samples=128
- Config: [config.json](experiments/gsm8k_gen_stageb_v2_l3_to_q4_seed1234/config.json)
- Summary: [training_summary.json](experiments/gsm8k_gen_stageb_v2_l3_to_q4_seed1234/runs/ce/training_summary.json)
- Summary: [training_summary.json](experiments/gsm8k_gen_stageb_v2_l3_to_q4_seed1234/runs/hybrid/training_summary.json)
- Summary: [training_summary.json](experiments/gsm8k_gen_stageb_v2_l3_to_q4_seed1234/runs/kl/training_summary.json)
- Summary: [test128_summary.json](experiments/gsm8k_gen_stageb_v2_l3_to_q4_seed1234/runs/test128_summary.json)
- Per Sample: [test128_per_sample.jsonl](experiments/gsm8k_gen_stageb_v2_l3_to_q4_seed1234/runs/test128_per_sample.jsonl)
- Log: [pipeline.log](experiments/gsm8k_gen_stageb_v2_l3_to_q4_seed1234/logs/pipeline.log)
- Original report: [RESULTS.md](experiments/gsm8k_gen_stageb_v2_l3_to_q4_seed1234/RESULTS.md)
- Checkpoint selection: [ce/selection.json](experiments/gsm8k_gen_stageb_v2_l3_to_q4_seed1234/runs/ce/selection.json)
- Checkpoint selection: [hybrid/selection.json](experiments/gsm8k_gen_stageb_v2_l3_to_q4_seed1234/runs/hybrid/selection.json)

Automatically extracted metrics from the linked JSON:

| JSON field | Value |
|---|---:|
| `llama_native.count` | 128 |
| `llama_native.strict_accuracy` | 0.0 |
| `llama_native.flexible_accuracy` | 0.5234375 |
| `llama_native.first_hash_accuracy` | 0.0 |
| `qwen_native.count` | 128 |
| `qwen_native.strict_accuracy` | 0.0 |
| `qwen_native.flexible_accuracy` | 0.5625 |
| `qwen_native.first_hash_accuracy` | 0.0 |
| `stage_a.count` | 128 |
| `stage_a.strict_accuracy` | 0.0 |
| `stage_a.flexible_accuracy` | 0.015625 |
| `stage_a.first_hash_accuracy` | 0.0 |

## gsm8k_retrained_stagea_hybrid_2048x4_l3_to_q4_seed1234

- Status: `completed` (archived result artifacts)
- Original path: `gsm8k_retrained_stagea_hybrid_2048x4_l3_to_q4_seed1234`
- Recorded configuration: seed=1234, train_samples=2048, val_samples=128, test_samples=128, stage_a_epochs=0, stage_b_epochs=4
- Config: [config.json](experiments/gsm8k_retrained_stagea_hybrid_2048x4_l3_to_q4_seed1234/config.json)
- Summary: [selection_summary.json](experiments/gsm8k_retrained_stagea_hybrid_2048x4_l3_to_q4_seed1234/runs/selection_summary.json)
- Summary: [test128_summary.json](experiments/gsm8k_retrained_stagea_hybrid_2048x4_l3_to_q4_seed1234/runs/test128_summary.json)
- Per Sample: [test128_per_sample.jsonl](experiments/gsm8k_retrained_stagea_hybrid_2048x4_l3_to_q4_seed1234/runs/test128_per_sample.jsonl)
- Log: [launcher.log](experiments/gsm8k_retrained_stagea_hybrid_2048x4_l3_to_q4_seed1234/logs/launcher.log)
- Log: [pipeline.log](experiments/gsm8k_retrained_stagea_hybrid_2048x4_l3_to_q4_seed1234/logs/pipeline.log)
- Original report: [RESULTS.md](experiments/gsm8k_retrained_stagea_hybrid_2048x4_l3_to_q4_seed1234/RESULTS.md)
- Checkpoint selection: [hybrid/selection.json](experiments/gsm8k_retrained_stagea_hybrid_2048x4_l3_to_q4_seed1234/runs/hybrid/selection.json)

Automatically extracted metrics from the linked JSON:

| JSON field | Value |
|---|---:|
| `gsm8k_stage_a.count` | 128 |
| `gsm8k_stage_a.strict_accuracy` | 0.0 |
| `gsm8k_stage_a.flexible_accuracy` | 0.0 |
| `gsm8k_stage_a.first_hash_accuracy` | 0.0 |
| `gsm8k_stage_a_hybrid2048x4.count` | 128 |
| `gsm8k_stage_a_hybrid2048x4.strict_accuracy` | 0.0 |
| `gsm8k_stage_a_hybrid2048x4.flexible_accuracy` | 0.5703125 |
| `gsm8k_stage_a_hybrid2048x4.first_hash_accuracy` | 0.0 |
| `previous_old_stage_a.count` | 128 |
| `previous_old_stage_a.strict_accuracy` | 0.0 |
| `previous_old_stage_a.flexible_accuracy` | 0.015625 |
| `previous_old_stage_a.first_hash_accuracy` | 0.0 |

## gsm8k_retrained_stagea_hybrid_stageb4ep_l3_to_q4_seed1234

- Status: `completed` (archived result artifacts)
- Original path: `gsm8k_retrained_stagea_hybrid_stageb4ep_l3_to_q4_seed1234`
- Recorded configuration: seed=1234, train_samples=1024, val_samples=128, test_samples=128, stage_a_epochs=0, stage_b_epochs=4
- Config: [config.json](experiments/gsm8k_retrained_stagea_hybrid_stageb4ep_l3_to_q4_seed1234/config.json)
- Summary: [selection_summary.json](experiments/gsm8k_retrained_stagea_hybrid_stageb4ep_l3_to_q4_seed1234/runs/selection_summary.json)
- Summary: [test128_summary.json](experiments/gsm8k_retrained_stagea_hybrid_stageb4ep_l3_to_q4_seed1234/runs/test128_summary.json)
- Per Sample: [test128_per_sample.jsonl](experiments/gsm8k_retrained_stagea_hybrid_stageb4ep_l3_to_q4_seed1234/runs/test128_per_sample.jsonl)
- Log: [gpu_wait.log](experiments/gsm8k_retrained_stagea_hybrid_stageb4ep_l3_to_q4_seed1234/logs/gpu_wait.log)
- Log: [launcher.log](experiments/gsm8k_retrained_stagea_hybrid_stageb4ep_l3_to_q4_seed1234/logs/launcher.log)
- Original report: [RESULTS.md](experiments/gsm8k_retrained_stagea_hybrid_stageb4ep_l3_to_q4_seed1234/RESULTS.md)
- Checkpoint selection: [hybrid/selection.json](experiments/gsm8k_retrained_stagea_hybrid_stageb4ep_l3_to_q4_seed1234/runs/hybrid/selection.json)

Automatically extracted metrics from the linked JSON:

| JSON field | Value |
|---|---:|
| `gsm8k_stage_a.count` | 128 |
| `gsm8k_stage_a.strict_accuracy` | 0.0 |
| `gsm8k_stage_a.flexible_accuracy` | 0.0 |
| `gsm8k_stage_a.first_hash_accuracy` | 0.0 |
| `gsm8k_stage_a_hybrid4ep.count` | 128 |
| `gsm8k_stage_a_hybrid4ep.strict_accuracy` | 0.0 |
| `gsm8k_stage_a_hybrid4ep.flexible_accuracy` | 0.5234375 |
| `gsm8k_stage_a_hybrid4ep.first_hash_accuracy` | 0.0 |
| `previous_old_stage_a.count` | 128 |
| `previous_old_stage_a.strict_accuracy` | 0.0 |
| `previous_old_stage_a.flexible_accuracy` | 0.015625 |
| `previous_old_stage_a.first_hash_accuracy` | 0.0 |

## gsm8k_retrained_stagea_hybrid_stageb_l3_to_q4_seed1234

- Status: `completed` (archived result artifacts)
- Original path: `gsm8k_retrained_stagea_hybrid_stageb_l3_to_q4_seed1234`
- Recorded configuration: seed=1234, train_samples=1024, val_samples=128, test_samples=128, stage_a_epochs=2, stage_b_epochs=2
- Config: [config.json](experiments/gsm8k_retrained_stagea_hybrid_stageb_l3_to_q4_seed1234/config.json)
- Summary: [selection_summary.json](experiments/gsm8k_retrained_stagea_hybrid_stageb_l3_to_q4_seed1234/runs/selection_summary.json)
- Summary: [test128_summary.json](experiments/gsm8k_retrained_stagea_hybrid_stageb_l3_to_q4_seed1234/runs/test128_summary.json)
- Per Sample: [test128_per_sample.jsonl](experiments/gsm8k_retrained_stagea_hybrid_stageb_l3_to_q4_seed1234/runs/test128_per_sample.jsonl)
- Log: [launcher.log](experiments/gsm8k_retrained_stagea_hybrid_stageb_l3_to_q4_seed1234/logs/launcher.log)
- Log: [pipeline.log](experiments/gsm8k_retrained_stagea_hybrid_stageb_l3_to_q4_seed1234/logs/pipeline.log)
- Original report: [RESULTS.md](experiments/gsm8k_retrained_stagea_hybrid_stageb_l3_to_q4_seed1234/RESULTS.md)
- Checkpoint selection: [hybrid/selection.json](experiments/gsm8k_retrained_stagea_hybrid_stageb_l3_to_q4_seed1234/runs/hybrid/selection.json)
- Checkpoint selection: [stage_a/selection.json](experiments/gsm8k_retrained_stagea_hybrid_stageb_l3_to_q4_seed1234/runs/stage_a/selection.json)

Automatically extracted metrics from the linked JSON:

| JSON field | Value |
|---|---:|
| `gsm8k_stage_a.count` | 128 |
| `gsm8k_stage_a.strict_accuracy` | 0.0 |
| `gsm8k_stage_a.flexible_accuracy` | 0.0 |
| `gsm8k_stage_a.first_hash_accuracy` | 0.0 |
| `gsm8k_stage_a_hybrid.count` | 128 |
| `gsm8k_stage_a_hybrid.strict_accuracy` | 0.0 |
| `gsm8k_stage_a_hybrid.flexible_accuracy` | 0.3984375 |
| `gsm8k_stage_a_hybrid.first_hash_accuracy` | 0.0 |
| `previous_old_stage_a.count` | 128 |
| `previous_old_stage_a.strict_accuracy` | 0.0 |
| `previous_old_stage_a.flexible_accuracy` | 0.015625 |
| `previous_old_stage_a.first_hash_accuracy` | 0.0 |

## lm_eval_gsm8k_fullsync_l3_to_q4_seed1234

- Status: `completed` (archived result artifacts)
- Original path: `lm_eval_gsm8k_fullsync_l3_to_q4_seed1234`
- Recorded configuration: seed=1234
- Config: [config.json](experiments/lm_eval_gsm8k_fullsync_l3_to_q4_seed1234/config.json)
- Summary: [fullsync_5shot32_metrics.json](experiments/lm_eval_gsm8k_fullsync_l3_to_q4_seed1234/results_flat/fullsync_5shot32_metrics.json)
- Summary: [fullsync_genb_eager_eosfix_metrics.json](experiments/lm_eval_gsm8k_fullsync_l3_to_q4_seed1234/results_flat/fullsync_genb_eager_eosfix_metrics.json)
- Summary: [llama_native_5shot32_metrics.json](experiments/lm_eval_gsm8k_fullsync_l3_to_q4_seed1234/results_flat/llama_native_5shot32_metrics.json)
- Summary: [llama_native_eager_metrics.json](experiments/lm_eval_gsm8k_fullsync_l3_to_q4_seed1234/results_flat/llama_native_eager_metrics.json)
- Summary: [llama_native_initial_metrics.json](experiments/lm_eval_gsm8k_fullsync_l3_to_q4_seed1234/results_flat/llama_native_initial_metrics.json)
- Summary: [qwen_custom_eager_eosfix_metrics.json](experiments/lm_eval_gsm8k_fullsync_l3_to_q4_seed1234/results_flat/qwen_custom_eager_eosfix_metrics.json)
- Per Sample: [fullsync_5shot32_samples.jsonl](experiments/lm_eval_gsm8k_fullsync_l3_to_q4_seed1234/results_flat/fullsync_5shot32_samples.jsonl)
- Per Sample: [fullsync_genb_eager_eosfix_samples.jsonl](experiments/lm_eval_gsm8k_fullsync_l3_to_q4_seed1234/results_flat/fullsync_genb_eager_eosfix_samples.jsonl)
- Per Sample: [llama_native_5shot32_samples.jsonl](experiments/lm_eval_gsm8k_fullsync_l3_to_q4_seed1234/results_flat/llama_native_5shot32_samples.jsonl)
- Per Sample: [llama_native_eager_samples.jsonl](experiments/lm_eval_gsm8k_fullsync_l3_to_q4_seed1234/results_flat/llama_native_eager_samples.jsonl)
- Log: [launcher.log](experiments/lm_eval_gsm8k_fullsync_l3_to_q4_seed1234/logs/launcher.log)
- Log: [pilot.log](experiments/lm_eval_gsm8k_fullsync_l3_to_q4_seed1234/logs/pilot.log)
- Original report: [RESULTS.md](experiments/lm_eval_gsm8k_fullsync_l3_to_q4_seed1234/RESULTS.md)

Automatically extracted metrics from the linked JSON:

| JSON field | Value |
|---|---:|
| `results.gsm8k_local.exact_match,strict-match` | 0.0625 |
| `results.gsm8k_local.exact_match_stderr,strict-match` | 0.04347552147751577 |
| `results.gsm8k_local.exact_match,flexible-extract` | 0.0 |
| `results.gsm8k_local.exact_match_stderr,flexible-extract` | 0.0 |

## grouped_headmix4_l3_to_q4_seed1234

```json
{
  "experiment": "grouped_headmix4_l3_to_q4_seed1234",
  "time": "2026-10-23 04:35:39",
  "status": "completed",
  "manifest": "/hy-tmp/yezhe/异构模型/grouped_headmix4_l3_to_q4_seed1234/manifest.json",
  "config": {
    "name": "grouped_headmix4_l3_to_q4_seed1234",
    "seed": 1234,
    "ufd_parent": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
    "mcq_parent": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
    "wait_for_experiment": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234",
    "parent_manifest_sha256": {
      "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234": "b5c8ddb4500933320e8f2262fa51215bcd6d60837e1cbccc59910154b7e1c2d6",
      "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739"
    },
    "architecture": {
      "source_layers": 28,
      "target_layers": 36,
      "source_heads": 8,
      "target_heads": 8,
      "source_dim": 128,
      "target_dim": 128,
      "groups": 4,
      "hidden_dim": 1024
    },
    "expected_parameters": 134217728,
    "stage_a_epochs": 2,
    "stage_b_epochs": 4,
    "stage_a_learning_rate": 0.001,
    "stage_b_learning_rate": 0.0001,
    "effective_batch": 8,
    "gradient_clip": 30.0,
    "weight_decay": 0.0,
    "chunk_tokens": 64,
    "checkpoint_every_steps": 64,
    "stage_a_selection": "Minimum epoch-end held-out reconstruction loss; 2048 MCQ train / 256 validation, same as old Stage-A",
    "stage_b_selection": "Fixed epoch4 step1024 last checkpoint, never test-selected",
    "stage_b_objective": "Existing UFD exact forward vocabulary KL, max64-token Native/student mixed prefixes and up to8 supervised positions; update the entire grouped translator, no Reader or per-head residual",
    "stage_b_train_counts": {
      "openbookqa": 512,
      "arc_challenge": 512,
      "gsm8k": 512,
      "mmlu_pro": 512
    },
    "evaluation_tasks": [
      "openbookqa",
      "arc_challenge",
      "hellaswag",
      "gsm8k",
      "mmlu_pro",
      "sciq"
    ],
    "monitor_validation_per_task": 16,
    "queue_min_free_gpu_gib": 20,
    "disk_reserve_gib": 20,
    "caveats": [
      "Architecture and trainable Stage-B parameter scope both change; not a grouping-only ablation.",
      "Reuse Full-Sync accuracy interface; no zero-prefill latency claim.",
      "Existing test sets are historical and MMLU-Pro is a custom split."
    ]
  },
  "results": {
    "openbookqa": {
      "qwen_native": {
        "n": 128,
        "correct": 103,
        "accuracy": 0.8046875
      },
      "new_ufd_stage_b": {
        "n": 128,
        "correct": 84,
        "accuracy": 0.65625
      },
      "grouped_translator": {
        "n": 128,
        "correct": 84,
        "accuracy": 0.65625
      }
    },
    "arc_challenge": {
      "qwen_native": {
        "n": 128,
        "correct": 111,
        "accuracy": 0.8671875
      },
      "new_ufd_stage_b": {
        "n": 128,
        "correct": 83,
        "accuracy": 0.6484375
      },
      "grouped_translator": {
        "n": 128,
        "correct": 81,
        "accuracy": 0.6328125
      }
    },
    "hellaswag": {
      "qwen_native": {
        "n": 64,
        "correct": 23,
        "accuracy": 0.359375
      },
      "new_ufd_stage_b": {
        "n": 64,
        "correct": 21,
        "accuracy": 0.328125
      },
      "grouped_translator": {
        "n": 64,
        "correct": 20,
        "accuracy": 0.3125
      }
    },
    "gsm8k": {
      "qwen_native": {
        "n": 32,
        "correct": 31,
        "accuracy": 0.96875
      },
      "new_ufd_stage_b": {
        "n": 32,
        "correct": 9,
        "accuracy": 0.28125
      },
      "grouped_translator": {
        "n": 32,
        "correct": 8,
        "accuracy": 0.25
      }
    },
    "mmlu_pro": {
      "qwen_native": {
        "n": 128,
        "correct": 52,
        "accuracy": 0.40625
      },
      "new_ufd_stage_b": {
        "n": 128,
        "correct": 24,
        "accuracy": 0.1875
      },
      "grouped_translator": {
        "n": 128,
        "correct": 44,
        "accuracy": 0.34375
      }
    },
    "sciq": {
      "qwen_native": {
        "n": 64,
        "correct": 57,
        "accuracy": 0.890625
      },
      "new_ufd_stage_b": {
        "n": 64,
        "correct": 58,
        "accuracy": 0.90625
      },
      "grouped_translator": {
        "n": 64,
        "correct": 57,
        "accuracy": 0.890625
      }
    }
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/grouped_headmix4_l3_to_q4_seed1234/evaluation/per_sample.jsonl"
}
```

## ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234

```json
{
  "experiment": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234",
  "timestamp": "2026-10-23 01:51:38",
  "status": "completed",
  "seed": 1234,
  "config": {
    "name": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234",
    "seed": 1234,
    "revision_reason": "Repair trajectory/diagnostic tokens field collision before any successful optimizer update. Hyperparameters and dataset unchanged. Preserve original failure and reuse its existing Sender caches with their original identity.",
    "source_cache_reuse_manifest_sha256": "57950947e95d9e4e94b9d2730a5df7ecc112c015a5da0a07ecfdfef198caeefd",
    "reader_parent": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234",
    "reader_parent_manifest_sha256": "e3cdafb48dca164e6193671c5271cf1772772aecb5e8300c7ddbbac98d04ee41",
    "reader_initial_sha256": "a757ce6e4559168ee227ec53e380f16df9f6f6cd2c82b02c556802fdcf6f9599",
    "writer_stage_a_sha256": "db1f295c502f27305a9d2ac410c47a1bdef4db7e0255f817483867b19d78cf0d",
    "writer_stage_b_sha256": "25942a14d77a2a36173cccf6653d0e86cef01bdfdbd082836395faa895ba995e",
    "updates": 256,
    "questions_per_update": 8,
    "rollouts_per_question": 2,
    "train_questions": 512,
    "evaluation_questions": 128,
    "historical_questions": 32,
    "learning_rate": 1e-05,
    "weight_decay": 0,
    "gradient_clip": 1.0,
    "temperature": 1.0,
    "top_p": 1.0,
    "initial_train_max_new_tokens": 1024,
    "extended_train_max_new_tokens": 2048,
    "smoke_questions": 2,
    "tbptt_tokens": 64,
    "formal_max_new_tokens": 4096,
    "checkpoint_every_steps": 16,
    "queue_min_free_gpu_gib": 20,
    "queue_retry_seconds": 60,
    "disk_reserve_gib": 20,
    "selection": "Fixed last step256 primary; no test/validation checkpoint selection or hyperparameter search.",
    "objective": "Only OPDVR gated sampled-token log-probability surrogate. Correct: max(0,logP_native-logP_student); incorrect: min(0,logP_native-logP_student). Gate fully detached. No additive KL, CE or format reward.",
    "normalization": "Mean over tokens within each completed trajectory, then divide by16 attempted trajectories/update. Truncated trajectories have zero loss, not negative reward. No group centering, reward normalization or gate clipping.",
    "gradient_approximation": "Complete trajectories determine correctness. Teacher/student replay identical sampled IDs; current64-token blocks retain their full gradient, previous continuation KV is detached at each block. This is truncated BPTT, not full-sequence policy gradient.",
    "training_length_rule": "Smoke4 sampled training trajectories at1024; any cap triggers2048 for all formal training. If all smoke trajectories still cap at2048, abort. No test-driven length choice.",
    "verifier": "Pinned HeteroFold numeric extractor and normalization. EOS-completed unparsable answers are incorrect; cap without EOS is censored irrespective of parser output. EOS tokens participate in completed-trajectory loss.",
    "evaluation": "Pinned original4096-token greedy generate, Qwen thinking=False, same Full-Sync interface and numeric parser. First128 official test rows, including historical32 and new96; new96 are the primary new-evidence subset. Initial Reader vs final Reader only.",
    "continuous_monitoring": false,
    "automatic_upload": false,
    "reference": "https://arxiv.org/html/2608.24696v4#S4"
  },
  "checkpoint_selection": "Fixed last step256 primary; no test/validation checkpoint selection or hyperparameter search.",
  "results": {
    "experiment": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234",
    "status": "completed",
    "config": {
      "name": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234",
      "seed": 1234,
      "revision_reason": "Repair trajectory/diagnostic tokens field collision before any successful optimizer update. Hyperparameters and dataset unchanged. Preserve original failure and reuse its existing Sender caches with their original identity.",
      "source_cache_reuse_manifest_sha256": "57950947e95d9e4e94b9d2730a5df7ecc112c015a5da0a07ecfdfef198caeefd",
      "reader_parent": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234",
      "reader_parent_manifest_sha256": "e3cdafb48dca164e6193671c5271cf1772772aecb5e8300c7ddbbac98d04ee41",
      "reader_initial_sha256": "a757ce6e4559168ee227ec53e380f16df9f6f6cd2c82b02c556802fdcf6f9599",
      "writer_stage_a_sha256": "db1f295c502f27305a9d2ac410c47a1bdef4db7e0255f817483867b19d78cf0d",
      "writer_stage_b_sha256": "25942a14d77a2a36173cccf6653d0e86cef01bdfdbd082836395faa895ba995e",
      "updates": 256,
      "questions_per_update": 8,
      "rollouts_per_question": 2,
      "train_questions": 512,
      "evaluation_questions": 128,
      "historical_questions": 32,
      "learning_rate": 1e-05,
      "weight_decay": 0,
      "gradient_clip": 1.0,
      "temperature": 1.0,
      "top_p": 1.0,
      "initial_train_max_new_tokens": 1024,
      "extended_train_max_new_tokens": 2048,
      "smoke_questions": 2,
      "tbptt_tokens": 64,
      "formal_max_new_tokens": 4096,
      "checkpoint_every_steps": 16,
      "queue_min_free_gpu_gib": 20,
      "queue_retry_seconds": 60,
      "disk_reserve_gib": 20,
      "selection": "Fixed last step256 primary; no test/validation checkpoint selection or hyperparameter search.",
      "objective": "Only OPDVR gated sampled-token log-probability surrogate. Correct: max(0,logP_native-logP_student); incorrect: min(0,logP_native-logP_student). Gate fully detached. No additive KL, CE or format reward.",
      "normalization": "Mean over tokens within each completed trajectory, then divide by16 attempted trajectories/update. Truncated trajectories have zero loss, not negative reward. No group centering, reward normalization or gate clipping.",
      "gradient_approximation": "Complete trajectories determine correctness. Teacher/student replay identical sampled IDs; current64-token blocks retain their full gradient, previous continuation KV is detached at each block. This is truncated BPTT, not full-sequence policy gradient.",
      "training_length_rule": "Smoke4 sampled training trajectories at1024; any cap triggers2048 for all formal training. If all smoke trajectories still cap at2048, abort. No test-driven length choice.",
      "verifier": "Pinned HeteroFold numeric extractor and normalization. EOS-completed unparsable answers are incorrect; cap without EOS is censored irrespective of parser output. EOS tokens participate in completed-trajectory loss.",
      "evaluation": "Pinned original4096-token greedy generate, Qwen thinking=False, same Full-Sync interface and numeric parser. First128 official test rows, including historical32 and new96; new96 are the primary new-evidence subset. Initial Reader vs final Reader only.",
      "continuous_monitoring": false,
      "automatic_upload": false,
      "reference": "https://arxiv.org/html/2608.24696v4#S4"
    },
    "subsets": {
      "all128": {
        "metrics": {
          "initial_reader": {
            "n": 128,
            "correct": 55,
            "accuracy": 0.4296875,
            "mean_generated_tokens": 391.84375,
            "cap_count": 2
          },
          "reward_reader": {
            "n": 128,
            "correct": 70,
            "accuracy": 0.546875,
            "mean_generated_tokens": 474.328125,
            "cap_count": 5
          }
        },
        "paired": {
          "both_correct": 51,
          "initial_only": 4,
          "reward_only": 19,
          "both_wrong": 54
        }
      },
      "historical32": {
        "metrics": {
          "initial_reader": {
            "n": 32,
            "correct": 15,
            "accuracy": 0.46875,
            "mean_generated_tokens": 353.46875,
            "cap_count": 0
          },
          "reward_reader": {
            "n": 32,
            "correct": 17,
            "accuracy": 0.53125,
            "mean_generated_tokens": 489.65625,
            "cap_count": 1
          }
        },
        "paired": {
          "both_correct": 13,
          "initial_only": 2,
          "reward_only": 4,
          "both_wrong": 13
        }
      },
      "new96": {
        "metrics": {
          "initial_reader": {
            "n": 96,
            "correct": 40,
            "accuracy": 0.4166666666666667,
            "mean_generated_tokens": 404.6354166666667,
            "cap_count": 2
          },
          "reward_reader": {
            "n": 96,
            "correct": 53,
            "accuracy": 0.5520833333333334,
            "mean_generated_tokens": 469.21875,
            "cap_count": 4
          }
        },
        "paired": {
          "both_correct": 38,
          "initial_only": 2,
          "reward_only": 15,
          "both_wrong": 41
        }
      }
    },
    "training": {
      "updates": 256,
      "attempted": 4096,
      "completed": 3967,
      "censored": 129,
      "generated_tokens": 1364844
    },
    "smoke": {
      "status": "completed",
      "checks": {
        "correct_token_pushed_up": true,
        "incorrect_token_pushed_down": true,
        "conflicting_updates_zero": true,
        "gate_detached": true
      },
      "train_max_new_tokens": 1024,
      "completed_trajectories": 4,
      "attempted_trajectories": 4,
      "native_teacher_unchanged": true,
      "reader_only_gradients": true,
      "transmitted_prefix_unchanged": true,
      "peak_allocated_gib": 9.624284267425537,
      "gradient_approximation": "Complete trajectories determine correctness. Teacher/student replay identical sampled IDs; current64-token blocks retain their full gradient, previous continuation KV is detached at each block. This is truncated BPTT, not full-sequence policy gradient.",
      "manifest_sha256": "2a573a3d8713e474ae5c7c053772c7842aedf095ae229dbce1ae8f3bdfeebee0"
    },
    "manifest_sha256": "2a573a3d8713e474ae5c7c053772c7842aedf095ae229dbce1ae8f3bdfeebee0",
    "caveats": [
      "Only one trained arm; gain cannot be attributed solely to reward gating.",
      "64-token detached continuation boundaries approximate full policy gradients.",
      "Test128 includes historical32; report new96 separately.",
      "No MCQ retention claim or equal-budget HeteroFold comparison."
    ]
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234/logs/pipeline.log"
}
```

