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
