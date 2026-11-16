# Experiment Results


<!-- BEGIN heterogeneous_multidoc_frozen_writers_v0_seed1234 -->
## heterogeneous_multidoc_frozen_writers_v0_seed1234

- Status: `completed`
- Timestamp: 2026-10-14T11:25:27.334655+00:00
- Server: B; seed: 1234
- Protocol: `hotpot_bridge_gold2_dense_frozen_legacy_writers_v0`
- Config signature: `a17156a1797c7206dac6ce542f4809f9a1c0bc3672df278bd32a6128cceca654`
- Split: official HotpotQA DEV bridge/gold2; no training; 64 seeded eligible examples.
- Checkpoints: fixed prior OpenBookQA validation selections, no new checkpoint selection.
- Raw results: `/hy-tmp/yezhe/异构模型/heterogeneous_multidoc_frozen_writers_v0_seed1234/runs/pilot/summary.json`
- Manifest/checkpoint hashes: `/hy-tmp/yezhe/异构模型/heterogeneous_multidoc_frozen_writers_v0_seed1234/manifest.json`
- Per-sample outputs: `/hy-tmp/yezhe/异构模型/heterogeneous_multidoc_frozen_writers_v0_seed1234/runs/pilot/per_sample`
- Log: `/hy-tmp/yezhe/异构模型/heterogeneous_multidoc_frozen_writers_v0_seed1234/logs/pipeline.log`

| Arm | EM | F1 | First-token KL to Full |
|---|---:|---:|---:|
| query_only | 0.000000 | 0.056058 | 4.484333 |
| native_full | 0.234375 | 0.463391 | 0.000000 |
| native_independent | 0.281250 | 0.470084 | 0.606300 |
| llama_joint | 0.000000 | 0.000000 | 6.686125 |
| gemma_joint | 0.000000 | 0.000000 | 9.693691 |
| llama_independent | 0.000000 | 0.001359 | 6.677595 |
| gemma_independent | 0.000000 | 0.000000 | 9.630795 |
| llama_a_gemma_b | 0.000000 | 0.000000 | 8.757836 |
| gemma_a_llama_b | 0.000000 | 0.000000 | 8.642403 |
| llama_a_native_b | 0.093750 | 0.207717 | 3.851638 |
| native_a_llama_b | 0.093750 | 0.118837 | 4.829875 |
| gemma_a_native_b | 0.031250 | 0.063994 | 7.699021 |
| native_a_gemma_b | 0.000000 | 0.011513 | 9.217577 |

Paired F1 differences and confidence intervals are stored in the raw summary.
Custom mechanism pilot; old writer interface/domain extrapolation is recorded; no impossibility claim.

<!-- END heterogeneous_multidoc_frozen_writers_v0_seed1234 -->

<!-- BEGIN heterogeneous_multidoc_fullsync_writers_v1_seed1234 -->
## heterogeneous_multidoc_fullsync_writers_v1_seed1234

- Status: `incomplete`
- Timestamp: 2026-10-14T13:23:01.005293+00:00
- Server: B; seed: 1234
- Protocol: `hotpot_bridge_gold2_dense_frozen_retrained_fullsync_writers_v1`
- Config signature: `b74d334bde7fa86775507b1d5e24fc2166ac89d567ce501e3d79e80c452c483a`
- Split: official HotpotQA DEV bridge/gold2; no training; 64 seeded eligible examples.
- Checkpoints: fixed prior full-token Full-Sync retraining validation selections, no new selection.
- Raw results: `/hy-tmp/yezhe/异构模型/heterogeneous_multidoc_fullsync_writers_v1_seed1234/runs/pilot/failure.json`
- Manifest/checkpoint hashes: `/hy-tmp/yezhe/异构模型/heterogeneous_multidoc_fullsync_writers_v1_seed1234/manifest.json`
- Per-sample outputs: `/hy-tmp/yezhe/异构模型/heterogeneous_multidoc_fullsync_writers_v1_seed1234/runs/pilot/per_sample`
- Log: `/hy-tmp/yezhe/异构模型/heterogeneous_multidoc_fullsync_writers_v1_seed1234/logs/pipeline.log`

Failure: `User requested direct execution on server A; B stopped before GPU evaluation; partial copies preserved`

<!-- END heterogeneous_multidoc_fullsync_writers_v1_seed1234 -->

<!-- BEGIN lm_eval_fullsync_kv_pca_native_vs_translated_v1_seed1234 -->
## lm_eval_fullsync_kv_pca_native_vs_translated_v1_seed1234

- Status: `completed`
- Timestamp: 2026-10-14T13:57:00.801312+00:00
- Server: B; seed:1234
- Config: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/config.json`; manifest/checkpoint hashes: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/manifest.json`
- Official evaluation prompt subset: OBQA/ARC/GSM each8 calibration +24 analysis documents; no training.
- Comparison: native, lm-eval Stage-A, lm-eval Stage-A+Stage-B; original validation-loss-only selections.
- Raw summary: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/summary.json`; report: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/REPORT.md`
- Per-sample outputs: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/per_sample`; logs: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/logs/pipeline.log`

| Group | KV | Arm | NMSE | Cosine | Variance ratio | CKA |
|---|---|---|---:|---:|---:|---:|
| mcq | K | stage_a | 0.127007 | 0.934273 | 0.610315 | 0.845243 |
| mcq | K | stage_b | 0.134041 | 0.930961 | 0.611835 | 0.845005 |
| mcq | V | stage_a | 0.764188 | 0.730444 | 1.089286 | 0.740876 |
| mcq | V | stage_b | 0.769466 | 0.725872 | 1.089091 | 0.740375 |
| gsm8k | K | stage_a | 0.130794 | 0.931329 | 0.633201 | 0.833500 |
| gsm8k | K | stage_b | 0.140131 | 0.926620 | 0.637240 | 0.833112 |
| gsm8k | V | stage_a | 0.474643 | 0.780511 | 0.720605 | 0.750926 |
| gsm8k | V | stage_b | 0.483049 | 0.774668 | 0.720710 | 0.750007 |

PCA is native-calibrated, per layer/head; controlled reading, output KL and reused official task scores are in the raw summary. No composition/impossibility claim.

<!-- END lm_eval_fullsync_kv_pca_native_vs_translated_v1_seed1234 -->


<!-- BEGIN lm_eval_fullsync_llama_native_kv_pca_seed1234 -->
## lm_eval_fullsync_llama_native_kv_pca_seed1234

- Status: `completed`
- Timestamp: 2026-10-15T03:09:27.998531+00:00
- Server B; seed1234; exact same original PCA96 prompts and calibration/analysis roles; no training.
- Native model: fixed parent Llama3.2-3B-Instruct, FP16/eager; no checkpoint selection.
- Config: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/llama_native/config.json`; manifest: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/llama_native/manifest.json`
- Raw summary: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/llama_native/summary.json`; samples: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/llama_native/per_sample`
- Log: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/llama_native/logs/pipeline.log`; report: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/llama_native/REPORT.md`

| Group | KV | View | PC90 median ceil | PC95 median ceil | Effective rank mean |
|---|---|---|---:|---:|---:|
| mcq | K | llama_native | 53 | 71 | 44.6786 |
| mcq | K | llama_aligned | 53 | 71 | 44.8654 |
| mcq | K | qwen_native | 47 | 65 | 38.1850 |
| mcq | K | stage_b | 20 | 28 | 17.6986 |
| mcq | V | llama_native | 81 | 98 | 77.4345 |
| mcq | V | llama_aligned | 81 | 98 | 77.6523 |
| mcq | V | qwen_native | 81 | 99 | 72.3364 |
| mcq | V | stage_b | 40 | 55 | 35.9234 |
| gsm8k | K | llama_native | 51 | 70 | 42.2595 |
| gsm8k | K | llama_aligned | 50 | 68 | 40.6198 |
| gsm8k | K | qwen_native | 43 | 61 | 35.8183 |
| gsm8k | K | stage_b | 21 | 30 | 18.8926 |
| gsm8k | V | llama_native | 80 | 98 | 75.5491 |
| gsm8k | V | llama_aligned | 78 | 97 | 72.3887 |
| gsm8k | V | qwen_native | 79 | 98 | 73.1334 |
| gsm8k | V | stage_b | 50 | 67 | 45.0216 |

<!-- END lm_eval_fullsync_llama_native_kv_pca_seed1234 -->

<!-- BEGIN qwen_native_kv_pca_compression_v1_seed1234 -->
## qwen_native_kv_pca_compression_v1_seed1234

- Server B; status: completed
- Timestamp: 2026-10-15T04:25:30.709428+00:00
- Report: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/native_compression/REPORT.md`
- Config: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/native_compression/config.json`; manifest: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/native_compression/manifest.json`
- Log: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/native_compression/logs/pipeline.log`; samples: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/native_compression/per_sample`
- Fresh scores; exact original PCA72 analysis prompts; 12 PCA interventions + native; no training.
- Raw summary: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/native_compression/summary.json`
- arc_challenge: {"native": {"none": {"acc": 0.5416666666666666, "acc_norm": 0.5833333333333334}}, "K_16": {"none": {"acc": 0.5, "acc_norm": 0.5416666666666666}}, "V_16": {"none": {"acc": 0.4166666666666667, "acc_norm": 0.5}}, "KV_16": {"none": {"acc": 0.4166666666666667, "acc_norm": 0.5}}, "K_32": {"none": {"acc": 0.5833333333333334, "acc_norm": 0.5416666666666666}}, "V_32": {"none": {"acc": 0.5416666666666666, "acc_norm": 0.5}}, "KV_32": {"none": {"acc": 0.4583333333333333, "acc_norm": 0.5416666666666666}}, "K_64": {"none": {"acc": 0.5416666666666666, "acc_norm": 0.5833333333333334}}, "V_64": {"none": {"acc": 0.5, "acc_norm": 0.5416666666666666}}, "KV_64": {"none": {"acc": 0.5, "acc_norm": 0.5416666666666666}}, "K_96": {"none": {"acc": 0.5416666666666666, "acc_norm": 0.5833333333333334}}, "V_96": {"none": {"acc": 0.5416666666666666, "acc_norm": 0.5416666666666666}}, "KV_96": {"none": {"acc": 0.5416666666666666, "acc_norm": 0.5416666666666666}}}
- gsm8k: {"native": {"flexible-extract": {"exact_match": 0.875}, "strict-match": {"exact_match": 0.9166666666666666}}, "K_16": {"flexible-extract": {"exact_match": 0.0}, "strict-match": {"exact_match": 0.0}}, "V_16": {"flexible-extract": {"exact_match": 0.125}, "strict-match": {"exact_match": 0.0}}, "KV_16": {"flexible-extract": {"exact_match": 0.041666666666666664}, "strict-match": {"exact_match": 0.0}}, "K_32": {"flexible-extract": {"exact_match": 0.75}, "strict-match": {"exact_match": 0.6666666666666666}}, "V_32": {"flexible-extract": {"exact_match": 0.4166666666666667}, "strict-match": {"exact_match": 0.0}}, "KV_32": {"flexible-extract": {"exact_match": 0.20833333333333334}, "strict-match": {"exact_match": 0.0}}, "K_64": {"flexible-extract": {"exact_match": 0.8333333333333334}, "strict-match": {"exact_match": 0.8333333333333334}}, "V_64": {"flexible-extract": {"exact_match": 0.7083333333333334}, "strict-match": {"exact_match": 0.625}}, "KV_64": {"flexible-extract": {"exact_match": 0.7916666666666666}, "strict-match": {"exact_match": 0.75}}, "K_96": {"flexible-extract": {"exact_match": 0.8333333333333334}, "strict-match": {"exact_match": 0.9166666666666666}}, "V_96": {"flexible-extract": {"exact_match": 0.75}, "strict-match": {"exact_match": 0.75}}, "KV_96": {"flexible-extract": {"exact_match": 0.75}, "strict-match": {"exact_match": 0.7916666666666666}}}
- openbookqa: {"native": {"none": {"acc": 0.2916666666666667, "acc_norm": 0.4166666666666667}}, "K_16": {"none": {"acc": 0.2916666666666667, "acc_norm": 0.375}}, "V_16": {"none": {"acc": 0.25, "acc_norm": 0.3333333333333333}}, "KV_16": {"none": {"acc": 0.25, "acc_norm": 0.2916666666666667}}, "K_32": {"none": {"acc": 0.4166666666666667, "acc_norm": 0.375}}, "V_32": {"none": {"acc": 0.375, "acc_norm": 0.2916666666666667}}, "KV_32": {"none": {"acc": 0.2916666666666667, "acc_norm": 0.3333333333333333}}, "K_64": {"none": {"acc": 0.4166666666666667, "acc_norm": 0.4583333333333333}}, "V_64": {"none": {"acc": 0.3333333333333333, "acc_norm": 0.4166666666666667}}, "KV_64": {"none": {"acc": 0.375, "acc_norm": 0.375}}, "K_96": {"none": {"acc": 0.3333333333333333, "acc_norm": 0.4166666666666667}}, "V_96": {"none": {"acc": 0.2916666666666667, "acc_norm": 0.4166666666666667}}, "KV_96": {"none": {"acc": 0.2916666666666667, "acc_norm": 0.4166666666666667}}}

<!-- END qwen_native_kv_pca_compression_v1_seed1234 -->


<!-- BEGIN qwen_native_kv_pca_compression_ood_transfer_v1_seed1234 -->
## qwen_native_kv_pca_compression_ood_transfer_v1_seed1234

- Server B; status: completed within user-requested scope.
- Timestamp: 2026-10-15T06:29:34.326426+00:00
- Effective scope: HellaSwag24 + MMLU-Pro13 =37 completed samples,13 conditions each.
- User omitted final psychology sample; its2 partial conditions are excluded and preserved.
- Original config/manifest unchanged; scope override: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/native_compression/ood_transfer/user_stop_request.json`
- Summary: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/native_compression/ood_transfer/summary.json`; report: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234/native_compression/ood_transfer/REPORT.md`
- Frozen ID PCA; original official requests, filters and budgets; no OOD refit.
- hellaswag: {"native": {"score": 0.5416666666666666, "drop_from_native": 0.0}, "K_16": {"score": 0.4583333333333333, "drop_from_native": 0.08333333333333333}, "V_16": {"score": 0.4583333333333333, "drop_from_native": 0.08333333333333333}, "KV_16": {"score": 0.4583333333333333, "drop_from_native": 0.08333333333333333}, "K_32": {"score": 0.4166666666666667, "drop_from_native": 0.125}, "V_32": {"score": 0.4583333333333333, "drop_from_native": 0.08333333333333333}, "KV_32": {"score": 0.3333333333333333, "drop_from_native": 0.20833333333333334}, "K_64": {"score": 0.5, "drop_from_native": 0.041666666666666664}, "V_64": {"score": 0.4166666666666667, "drop_from_native": 0.125}, "KV_64": {"score": 0.375, "drop_from_native": 0.16666666666666666}, "K_96": {"score": 0.5, "drop_from_native": 0.041666666666666664}, "V_96": {"score": 0.5, "drop_from_native": 0.041666666666666664}, "KV_96": {"score": 0.5, "drop_from_native": 0.041666666666666664}}
- mmlu_pro: {"native": {"score": 0.6923076923076923, "drop_from_native": 0.0}, "K_16": {"score": 0.0, "drop_from_native": 0.6923076923076923}, "V_16": {"score": 0.0, "drop_from_native": 0.6923076923076923}, "KV_16": {"score": 0.0, "drop_from_native": 0.6923076923076923}, "K_32": {"score": 0.5384615384615384, "drop_from_native": 0.15384615384615385}, "V_32": {"score": 0.0, "drop_from_native": 0.6923076923076923}, "KV_32": {"score": 0.0, "drop_from_native": 0.6923076923076923}, "K_64": {"score": 0.6153846153846154, "drop_from_native": 0.07692307692307693}, "V_64": {"score": 0.46153846153846156, "drop_from_native": 0.23076923076923078}, "KV_64": {"score": 0.15384615384615385, "drop_from_native": 0.5384615384615384}, "K_96": {"score": 0.6153846153846154, "drop_from_native": 0.07692307692307693}, "V_96": {"score": 0.5384615384615384, "drop_from_native": 0.15384615384615385}, "KV_96": {"score": 0.5384615384615384, "drop_from_native": 0.15384615384615385}}

<!-- END qwen_native_kv_pca_compression_ood_transfer_v1_seed1234 -->





## receiver_gated_reader_v1_seed1234

2026-10-17T08:24:26Z

Completed. Server B; seed 1234. Frozen latest translator; reader only. Same predecessor training/validation splits: 256 references, 24 held-out references, 32 held-out fixed prefixes; 128/128/128/128 benchmark pilots. Checkpoint selected on held-out mixed loss with epoch zero eligible.

```json
{
  "experiment": "receiver_gated_reader_v1_seed1234",
  "completed": true,
  "server": "B",
  "configuration": {
    "protocol": "lm_eval_receiver_gated_external_reader_v1",
    "seed": 1234,
    "previous_directory": "analysis/mcq_stage_b_semantic_followup_v1_seed1234",
    "parent_stage_a": "runs/mcq/stage_a/best.pt",
    "mmlu_baseline_directory": "analysis/mmlu_mcq_transfer_v1_seed1234",
    "adapter_rank": 64,
    "epochs": 2,
    "effective_batch": 4,
    "learning_rate": 2e-05,
    "weight_decay": 0.0,
    "gradient_clip": 30.0,
    "token_kl_weight": 1.0,
    "gold_ce_weight": 0.1,
    "terminal_ce_weight": 0.1,
    "mcq_replay_per_epoch": 86,
    "projection_chunk_tokens": 32,
    "checkpoint_interval_updates": 16,
    "min_free_gpu_gib": 23,
    "student_rollout_tokens_by_epoch": [
      64,
      128
    ],
    "validation_rollout_max_tokens": 128,
    "validation_generation_max_new_tokens": 384,
    "evaluation_families": [
      "mcq",
      "mmlu"
    ],
    "schedule": "128 teacher-reference + 128 live student-prefix + 86 MCQ draws per epoch; swap reference/on-policy halves in epoch 2. Same 342 draws and 86 updates per epoch as trajectory v1.",
    "loss": "Reference: previous complete-trajectory forward KL + 0.1 answer-phrase CE + 0.1 observed-terminal CE. Student-prefix: full-vocabulary forward KL only, no hard student-token targets and no forced EOS. MCQ: unchanged scheme-3 objective.",
    "selection": "Lowest equal-weight held-out reference loss and KL on the same 32 fixed validation prefixes as predecessor. Epoch zero eligible. No benchmark-based selection.",
    "receiver_lora": false,
    "on_policy_training": true,
    "reader_rank": 8,
    "receiver_reader": true,
    "translator_trainable": false,
    "initialization": "Freeze latest student-prefix Stage A/B; zero-init external reader. Retain predecessor training data, losses, 342 draws/epoch, 172 total updates, two epochs.",
    "diagnostic_questions_per_task": 2
  },
  "pilot_subset_not_full_benchmark": true,
  "selection": {
    "selection": "Lowest equal-weight held-out reference loss and KL on the same 32 fixed validation prefixes as predecessor. Epoch zero eligible. No benchmark-based selection.",
    "best": {
      "epoch": 2,
      "step": 172,
      "validation": {
        "total": 0.6962837246246636,
        "reference": {
          "total": 0.7605292797088623,
          "token_kl": 0.6563656566043695,
          "answer_ce": 0.9787250931064287,
          "terminal_ce": 0.06291120623548825,
          "tokens": 130.45833333333334,
          "context_tokens": 1061.0416666666667,
          "n": 24
        },
        "fixed_student_prefix_kl": 0.6320381695404649,
        "fixed_prefix_n": 32
      }
    },
    "epochs": [
      {
        "epoch": 0,
        "step": 0,
        "validation": {
          "total": 0.747362957180788,
          "reference": {
            "total": 0.8222668046752611,
            "token_kl": 0.7001174973944823,
            "answer_ce": 1.1401850307981174,
            "terminal_ce": 0.08130801562219858,
            "tokens": 130.45833333333334,
            "context_tokens": 1061.0416666666667,
            "n": 24
          },
          "fixed_student_prefix_kl": 0.6724591096863151,
          "fixed_prefix_n": 32
        }
      },
      {
        "epoch": 1,
        "step": 86,
        "validation": {
          "total": 0.7222785310198863,
          "reference": {
            "total": 0.7918247307340304,
            "token_kl": 0.6785766892135143,
            "answer_ce": 1.0637830942869186,
            "terminal_ce": 0.06869729096069932,
            "tokens": 130.45833333333334,
            "context_tokens": 1061.0416666666667,
            "n": 24
          },
          "fixed_student_prefix_kl": 0.6527323313057423,
          "fixed_prefix_n": 32
        }
      },
      {
        "epoch": 2,
        "step": 172,
        "validation": {
          "total": 0.6962837246246636,
          "reference": {
            "total": 0.7605292797088623,
            "token_kl": 0.6563656566043695,
            "answer_ce": 0.9787250931064287,
            "terminal_ce": 0.06291120623548825,
            "tokens": 130.45833333333334,
            "context_tokens": 1061.0416666666667,
            "n": 24
          },
          "fixed_student_prefix_kl": 0.6320381695404649,
          "fixed_prefix_n": 32
        }
      }
    ]
  },
  "checkpoint_sha256": "2043ff82da83511f25a800c077e2ebfe95646911adc4ffe15feaee0f58e228ea",
  "families": {
    "mcq": {
      "reader": {
        "openbookqa": {
          "name": "openbookqa",
          "alias": "openbookqa",
          "sample_len": 128,
          "acc,none": 0.2265625,
          "acc_stderr,none": 0.037145376625835905,
          "acc_norm,none": 0.4453125,
          "acc_norm_stderr,none": 0.04410164327680508
        },
        "arc_challenge": {
          "name": "arc_challenge",
          "alias": "arc_challenge",
          "sample_len": 128,
          "acc,none": 0.4453125,
          "acc_stderr,none": 0.04410164327680508,
          "acc_norm,none": 0.46875,
          "acc_norm_stderr,none": 0.044281084771084105
        }
      },
      "frozen_translator_baseline": {
        "openbookqa": {
          "name": "openbookqa",
          "alias": "openbookqa",
          "sample_len": 128,
          "acc,none": 0.2265625,
          "acc_stderr,none": 0.037145376625835905,
          "acc_norm,none": 0.4453125,
          "acc_norm_stderr,none": 0.04410164327680508
        },
        "arc_challenge": {
          "name": "arc_challenge",
          "alias": "arc_challenge",
          "sample_len": 128,
          "acc,none": 0.453125,
          "acc_stderr,none": 0.04417241936858356,
          "acc_norm,none": 0.46875,
          "acc_norm_stderr,none": 0.044281084771084105
        }
      }
    },
    "mmlu": {
      "reader": {
        "mmlu_abstract_algebra": {
          "name": "mmlu_abstract_algebra",
          "alias": "abstract_algebra",
          "sample_len": 3,
          "acc,none": 0.6666666666666666,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_anatomy": {
          "name": "mmlu_anatomy",
          "alias": "anatomy",
          "sample_len": 3,
          "acc,none": 0.3333333333333333,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_astronomy": {
          "name": "mmlu_astronomy",
          "alias": "astronomy",
          "sample_len": 3,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_college_biology": {
          "name": "mmlu_college_biology",
          "alias": "college_biology",
          "sample_len": 3,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_college_chemistry": {
          "name": "mmlu_college_chemistry",
          "alias": "college_chemistry",
          "sample_len": 3,
          "acc,none": 0.3333333333333333,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_college_computer_science": {
          "name": "mmlu_college_computer_science",
          "alias": "college_computer_science",
          "sample_len": 3,
          "acc,none": 0.3333333333333333,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_college_mathematics": {
          "name": "mmlu_college_mathematics",
          "alias": "college_mathematics",
          "sample_len": 3,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_college_physics": {
          "name": "mmlu_college_physics",
          "alias": "college_physics",
          "sample_len": 3,
          "acc,none": 0.3333333333333333,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_computer_security": {
          "name": "mmlu_computer_security",
          "alias": "computer_security",
          "sample_len": 3,
          "acc,none": 0.6666666666666666,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_conceptual_physics": {
          "name": "mmlu_conceptual_physics",
          "alias": "conceptual_physics",
          "sample_len": 3,
          "acc,none": 0.6666666666666666,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_electrical_engineering": {
          "name": "mmlu_electrical_engineering",
          "alias": "electrical_engineering",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_elementary_mathematics": {
          "name": "mmlu_elementary_mathematics",
          "alias": "elementary_mathematics",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_biology": {
          "name": "mmlu_high_school_biology",
          "alias": "high_school_biology",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_chemistry": {
          "name": "mmlu_high_school_chemistry",
          "alias": "high_school_chemistry",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_computer_science": {
          "name": "mmlu_high_school_computer_science",
          "alias": "high_school_computer_science",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_mathematics": {
          "name": "mmlu_high_school_mathematics",
          "alias": "high_school_mathematics",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_physics": {
          "name": "mmlu_high_school_physics",
          "alias": "high_school_physics",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_statistics": {
          "name": "mmlu_high_school_statistics",
          "alias": "high_school_statistics",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_machine_learning": {
          "name": "mmlu_machine_learning",
          "alias": "machine_learning",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_business_ethics": {
          "name": "mmlu_business_ethics",
          "alias": "business_ethics",
          "sample_len": 3,
          "acc,none": 0.3333333333333333,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_clinical_knowledge": {
          "name": "mmlu_clinical_knowledge",
          "alias": "clinical_knowledge",
          "sample_len": 3,
          "acc,none": 0.3333333333333333,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_college_medicine": {
          "name": "mmlu_college_medicine",
          "alias": "college_medicine",
          "sample_len": 3,
          "acc,none": 0.3333333333333333,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_global_facts": {
          "name": "mmlu_global_facts",
          "alias": "global_facts",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_human_aging": {
          "name": "mmlu_human_aging",
          "alias": "human_aging",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_management": {
          "name": "mmlu_management",
          "alias": "management",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_marketing": {
          "name": "mmlu_marketing",
          "alias": "marketing",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_medical_genetics": {
          "name": "mmlu_medical_genetics",
          "alias": "medical_genetics",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_miscellaneous": {
          "name": "mmlu_miscellaneous",
          "alias": "miscellaneous",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_nutrition": {
          "name": "mmlu_nutrition",
          "alias": "nutrition",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_professional_accounting": {
          "name": "mmlu_professional_accounting",
          "alias": "professional_accounting",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_professional_medicine": {
          "name": "mmlu_professional_medicine",
          "alias": "professional_medicine",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_virology": {
          "name": "mmlu_virology",
          "alias": "virology",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_econometrics": {
          "name": "mmlu_econometrics",
          "alias": "econometrics",
          "sample_len": 3,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_geography": {
          "name": "mmlu_high_school_geography",
          "alias": "high_school_geography",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_high_school_government_and_politics": {
          "name": "mmlu_high_school_government_and_politics",
          "alias": "high_school_government_and_politics",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_macroeconomics": {
          "name": "mmlu_high_school_macroeconomics",
          "alias": "high_school_macroeconomics",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_microeconomics": {
          "name": "mmlu_high_school_microeconomics",
          "alias": "high_school_microeconomics",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_psychology": {
          "name": "mmlu_high_school_psychology",
          "alias": "high_school_psychology",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_human_sexuality": {
          "name": "mmlu_human_sexuality",
          "alias": "human_sexuality",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_professional_psychology": {
          "name": "mmlu_professional_psychology",
          "alias": "professional_psychology",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_public_relations": {
          "name": "mmlu_public_relations",
          "alias": "public_relations",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_security_studies": {
          "name": "mmlu_security_studies",
          "alias": "security_studies",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_sociology": {
          "name": "mmlu_sociology",
          "alias": "sociology",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_us_foreign_policy": {
          "name": "mmlu_us_foreign_policy",
          "alias": "us_foreign_policy",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_formal_logic": {
          "name": "mmlu_formal_logic",
          "alias": "formal_logic",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_high_school_european_history": {
          "name": "mmlu_high_school_european_history",
          "alias": "high_school_european_history",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_high_school_us_history": {
          "name": "mmlu_high_school_us_history",
          "alias": "high_school_us_history",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_high_school_world_history": {
          "name": "mmlu_high_school_world_history",
          "alias": "high_school_world_history",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_international_law": {
          "name": "mmlu_international_law",
          "alias": "international_law",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_jurisprudence": {
          "name": "mmlu_jurisprudence",
          "alias": "jurisprudence",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_logical_fallacies": {
          "name": "mmlu_logical_fallacies",
          "alias": "logical_fallacies",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_moral_disputes": {
          "name": "mmlu_moral_disputes",
          "alias": "moral_disputes",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_moral_scenarios": {
          "name": "mmlu_moral_scenarios",
          "alias": "moral_scenarios",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_philosophy": {
          "name": "mmlu_philosophy",
          "alias": "philosophy",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_prehistory": {
          "name": "mmlu_prehistory",
          "alias": "prehistory",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_professional_law": {
          "name": "mmlu_professional_law",
          "alias": "professional_law",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_world_religions": {
          "name": "mmlu_world_religions",
          "alias": "world_religions",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_stem": {
          "alias": "stem",
          "name": "mmlu_stem",
          "sample_len": 48,
          "acc,none": 0.5,
          "acc_stderr,none": 0.06380345779073138,
          "sample_count": {
            "acc,none": 48
          }
        },
        "mmlu_other": {
          "alias": "other",
          "name": "mmlu_other",
          "sample_len": 29,
          "acc,none": 0.5172413793103449,
          "acc_stderr,none": 0.09284766908852593,
          "sample_count": {
            "acc,none": 29
          }
        },
        "mmlu_social_sciences": {
          "alias": "social sciences",
          "name": "mmlu_social_sciences",
          "sample_len": 25,
          "acc,none": 0.56,
          "acc_stderr,none": 0.07844645405527362,
          "sample_count": {
            "acc,none": 25
          }
        },
        "mmlu_humanities": {
          "alias": "humanities",
          "name": "mmlu_humanities",
          "sample_len": 26,
          "acc,none": 0.3076923076923077,
          "acc_stderr,none": 0.09421114395319916,
          "sample_count": {
            "acc,none": 26
          }
        },
        "mmlu": {
          "alias": "mmlu",
          "name": "mmlu",
          "sample_len": 128,
          "acc,none": 0.4765625,
          "acc_stderr,none": 0.040172749037851715,
          "sample_count": {
            "acc,none": 128
          }
        }
      },
      "frozen_translator_baseline": {
        "mmlu_abstract_algebra": {
          "name": "mmlu_abstract_algebra",
          "alias": "abstract_algebra",
          "sample_len": 3,
          "acc,none": 0.6666666666666666,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_anatomy": {
          "name": "mmlu_anatomy",
          "alias": "anatomy",
          "sample_len": 3,
          "acc,none": 0.6666666666666666,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_astronomy": {
          "name": "mmlu_astronomy",
          "alias": "astronomy",
          "sample_len": 3,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_college_biology": {
          "name": "mmlu_college_biology",
          "alias": "college_biology",
          "sample_len": 3,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_college_chemistry": {
          "name": "mmlu_college_chemistry",
          "alias": "college_chemistry",
          "sample_len": 3,
          "acc,none": 0.3333333333333333,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_college_computer_science": {
          "name": "mmlu_college_computer_science",
          "alias": "college_computer_science",
          "sample_len": 3,
          "acc,none": 0.3333333333333333,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_college_mathematics": {
          "name": "mmlu_college_mathematics",
          "alias": "college_mathematics",
          "sample_len": 3,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_college_physics": {
          "name": "mmlu_college_physics",
          "alias": "college_physics",
          "sample_len": 3,
          "acc,none": 0.3333333333333333,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_computer_security": {
          "name": "mmlu_computer_security",
          "alias": "computer_security",
          "sample_len": 3,
          "acc,none": 0.6666666666666666,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_conceptual_physics": {
          "name": "mmlu_conceptual_physics",
          "alias": "conceptual_physics",
          "sample_len": 3,
          "acc,none": 0.6666666666666666,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_electrical_engineering": {
          "name": "mmlu_electrical_engineering",
          "alias": "electrical_engineering",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_elementary_mathematics": {
          "name": "mmlu_elementary_mathematics",
          "alias": "elementary_mathematics",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_biology": {
          "name": "mmlu_high_school_biology",
          "alias": "high_school_biology",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_chemistry": {
          "name": "mmlu_high_school_chemistry",
          "alias": "high_school_chemistry",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_computer_science": {
          "name": "mmlu_high_school_computer_science",
          "alias": "high_school_computer_science",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_mathematics": {
          "name": "mmlu_high_school_mathematics",
          "alias": "high_school_mathematics",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_physics": {
          "name": "mmlu_high_school_physics",
          "alias": "high_school_physics",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_statistics": {
          "name": "mmlu_high_school_statistics",
          "alias": "high_school_statistics",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_machine_learning": {
          "name": "mmlu_machine_learning",
          "alias": "machine_learning",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_business_ethics": {
          "name": "mmlu_business_ethics",
          "alias": "business_ethics",
          "sample_len": 3,
          "acc,none": 0.3333333333333333,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_clinical_knowledge": {
          "name": "mmlu_clinical_knowledge",
          "alias": "clinical_knowledge",
          "sample_len": 3,
          "acc,none": 0.3333333333333333,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_college_medicine": {
          "name": "mmlu_college_medicine",
          "alias": "college_medicine",
          "sample_len": 3,
          "acc,none": 0.6666666666666666,
          "acc_stderr,none": 0.33333333333333337
        },
        "mmlu_global_facts": {
          "name": "mmlu_global_facts",
          "alias": "global_facts",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_human_aging": {
          "name": "mmlu_human_aging",
          "alias": "human_aging",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_management": {
          "name": "mmlu_management",
          "alias": "management",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_marketing": {
          "name": "mmlu_marketing",
          "alias": "marketing",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_medical_genetics": {
          "name": "mmlu_medical_genetics",
          "alias": "medical_genetics",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_miscellaneous": {
          "name": "mmlu_miscellaneous",
          "alias": "miscellaneous",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_nutrition": {
          "name": "mmlu_nutrition",
          "alias": "nutrition",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_professional_accounting": {
          "name": "mmlu_professional_accounting",
          "alias": "professional_accounting",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_professional_medicine": {
          "name": "mmlu_professional_medicine",
          "alias": "professional_medicine",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_virology": {
          "name": "mmlu_virology",
          "alias": "virology",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_econometrics": {
          "name": "mmlu_econometrics",
          "alias": "econometrics",
          "sample_len": 3,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_geography": {
          "name": "mmlu_high_school_geography",
          "alias": "high_school_geography",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_high_school_government_and_politics": {
          "name": "mmlu_high_school_government_and_politics",
          "alias": "high_school_government_and_politics",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_macroeconomics": {
          "name": "mmlu_high_school_macroeconomics",
          "alias": "high_school_macroeconomics",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_microeconomics": {
          "name": "mmlu_high_school_microeconomics",
          "alias": "high_school_microeconomics",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_high_school_psychology": {
          "name": "mmlu_high_school_psychology",
          "alias": "high_school_psychology",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_human_sexuality": {
          "name": "mmlu_human_sexuality",
          "alias": "human_sexuality",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_professional_psychology": {
          "name": "mmlu_professional_psychology",
          "alias": "professional_psychology",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_public_relations": {
          "name": "mmlu_public_relations",
          "alias": "public_relations",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_security_studies": {
          "name": "mmlu_security_studies",
          "alias": "security_studies",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_sociology": {
          "name": "mmlu_sociology",
          "alias": "sociology",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_us_foreign_policy": {
          "name": "mmlu_us_foreign_policy",
          "alias": "us_foreign_policy",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_formal_logic": {
          "name": "mmlu_formal_logic",
          "alias": "formal_logic",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_high_school_european_history": {
          "name": "mmlu_high_school_european_history",
          "alias": "high_school_european_history",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_high_school_us_history": {
          "name": "mmlu_high_school_us_history",
          "alias": "high_school_us_history",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_high_school_world_history": {
          "name": "mmlu_high_school_world_history",
          "alias": "high_school_world_history",
          "sample_len": 2,
          "acc,none": 1.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_international_law": {
          "name": "mmlu_international_law",
          "alias": "international_law",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_jurisprudence": {
          "name": "mmlu_jurisprudence",
          "alias": "jurisprudence",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_logical_fallacies": {
          "name": "mmlu_logical_fallacies",
          "alias": "logical_fallacies",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_moral_disputes": {
          "name": "mmlu_moral_disputes",
          "alias": "moral_disputes",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_moral_scenarios": {
          "name": "mmlu_moral_scenarios",
          "alias": "moral_scenarios",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_philosophy": {
          "name": "mmlu_philosophy",
          "alias": "philosophy",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_prehistory": {
          "name": "mmlu_prehistory",
          "alias": "prehistory",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_professional_law": {
          "name": "mmlu_professional_law",
          "alias": "professional_law",
          "sample_len": 2,
          "acc,none": 0.5,
          "acc_stderr,none": 0.5
        },
        "mmlu_world_religions": {
          "name": "mmlu_world_religions",
          "alias": "world_religions",
          "sample_len": 2,
          "acc,none": 0.0,
          "acc_stderr,none": 0.0
        },
        "mmlu_stem": {
          "alias": "stem",
          "name": "mmlu_stem",
          "sample_len": 48,
          "acc,none": 0.5208333333333334,
          "acc_stderr,none": 0.06380345779073138,
          "sample_count": {
            "acc,none": 48
          }
        },
        "mmlu_other": {
          "alias": "other",
          "name": "mmlu_other",
          "sample_len": 29,
          "acc,none": 0.5172413793103449,
          "acc_stderr,none": 0.09847982464479192,
          "sample_count": {
            "acc,none": 29
          }
        },
        "mmlu_social_sciences": {
          "alias": "social sciences",
          "name": "mmlu_social_sciences",
          "sample_len": 25,
          "acc,none": 0.56,
          "acc_stderr,none": 0.07844645405527362,
          "sample_count": {
            "acc,none": 25
          }
        },
        "mmlu_humanities": {
          "alias": "humanities",
          "name": "mmlu_humanities",
          "sample_len": 26,
          "acc,none": 0.34615384615384615,
          "acc_stderr,none": 0.10175966581017655,
          "sample_count": {
            "acc,none": 26
          }
        },
        "mmlu": {
          "alias": "mmlu",
          "name": "mmlu",
          "sample_len": 128,
          "acc,none": 0.4921875,
          "acc_stderr,none": 0.04151969383950009,
          "sample_count": {
            "acc,none": 128
          }
        }
      }
    },
    "mmlu_pro": {
      "reader": {
        "mmlu_pro_biology": {
          "name": "mmlu_pro_biology",
          "alias": "biology",
          "sample_len": 10,
          "exact_match,custom-extract": 0.1,
          "exact_match_stderr,custom-extract": 0.09999999999999999
        },
        "mmlu_pro_business": {
          "name": "mmlu_pro_business",
          "alias": "business",
          "sample_len": 10,
          "exact_match,custom-extract": 0.0,
          "exact_match_stderr,custom-extract": 0.0
        },
        "mmlu_pro_chemistry": {
          "name": "mmlu_pro_chemistry",
          "alias": "chemistry",
          "sample_len": 9,
          "exact_match,custom-extract": 0.2222222222222222,
          "exact_match_stderr,custom-extract": 0.1469861839480328
        },
        "mmlu_pro_computer_science": {
          "name": "mmlu_pro_computer_science",
          "alias": "computer_science",
          "sample_len": 9,
          "exact_match,custom-extract": 0.1111111111111111,
          "exact_match_stderr,custom-extract": 0.11111111111111115
        },
        "mmlu_pro_economics": {
          "name": "mmlu_pro_economics",
          "alias": "economics",
          "sample_len": 9,
          "exact_match,custom-extract": 0.0,
          "exact_match_stderr,custom-extract": 0.0
        },
        "mmlu_pro_engineering": {
          "name": "mmlu_pro_engineering",
          "alias": "engineering",
          "sample_len": 9,
          "exact_match,custom-extract": 0.1111111111111111,
          "exact_match_stderr,custom-extract": 0.11111111111111112
        },
        "mmlu_pro_health": {
          "name": "mmlu_pro_health",
          "alias": "health",
          "sample_len": 9,
          "exact_match,custom-extract": 0.1111111111111111,
          "exact_match_stderr,custom-extract": 0.1111111111111111
        },
        "mmlu_pro_history": {
          "name": "mmlu_pro_history",
          "alias": "history",
          "sample_len": 9,
          "exact_match,custom-extract": 0.1111111111111111,
          "exact_match_stderr,custom-extract": 0.11111111111111115
        },
        "mmlu_pro_law": {
          "name": "mmlu_pro_law",
          "alias": "law",
          "sample_len": 9,
          "exact_match,custom-extract": 0.0,
          "exact_match_stderr,custom-extract": 0.0
        },
        "mmlu_pro_math": {
          "name": "mmlu_pro_math",
          "alias": "math",
          "sample_len": 9,
          "exact_match,custom-extract": 0.1111111111111111,
          "exact_match_stderr,custom-extract": 0.11111111111111112
        },
        "mmlu_pro_other": {
          "name": "mmlu_pro_other",
          "alias": "other",
          "sample_len": 9,
          "exact_match,custom-extract": 0.0,
          "exact_match_stderr,custom-extract": 0.0
        },
        "mmlu_pro_philosophy": {
          "name": "mmlu_pro_philosophy",
          "alias": "philosophy",
          "sample_len": 9,
          "exact_match,custom-extract": 0.1111111111111111,
          "exact_match_stderr,custom-extract": 0.11111111111111112
        },
        "mmlu_pro_physics": {
          "name": "mmlu_pro_physics",
          "alias": "physics",
          "sample_len": 9,
          "exact_match,custom-extract": 0.1111111111111111,
          "exact_match_stderr,custom-extract": 0.11111111111111112
        },
        "mmlu_pro_psychology": {
          "name": "mmlu_pro_psychology",
          "alias": "psychology",
          "sample_len": 9,
          "exact_match,custom-extract": 0.0,
          "exact_match_stderr,custom-extract": 0.0
        },
        "mmlu_pro": {
          "alias": "mmlu_pro",
          "name": "mmlu_pro",
          "sample_len": 128,
          "exact_match,custom-extract": 0.078125,
          "exact_match_stderr,custom-extract": 0.02438635275681313,
          "sample_count": {
            "exact_match,custom-extract": 128
          }
        }
      },
      "frozen_translator_baseline": {
        "mmlu_pro_biology": {
          "name": "mmlu_pro_biology",
          "alias": "biology",
          "sample_len": 10,
          "exact_match,custom-extract": 0.0,
          "exact_match_stderr,custom-extract": 0.0
        },
        "mmlu_pro_business": {
          "name": "mmlu_pro_business",
          "alias": "business",
          "sample_len": 10,
          "exact_match,custom-extract": 0.1,
          "exact_match_stderr,custom-extract": 0.09999999999999999
        },
        "mmlu_pro_chemistry": {
          "name": "mmlu_pro_chemistry",
          "alias": "chemistry",
          "sample_len": 9,
          "exact_match,custom-extract": 0.1111111111111111,
          "exact_match_stderr,custom-extract": 0.11111111111111112
        },
        "mmlu_pro_computer_science": {
          "name": "mmlu_pro_computer_science",
          "alias": "computer_science",
          "sample_len": 9,
          "exact_match,custom-extract": 0.1111111111111111,
          "exact_match_stderr,custom-extract": 0.11111111111111115
        },
        "mmlu_pro_economics": {
          "name": "mmlu_pro_economics",
          "alias": "economics",
          "sample_len": 9,
          "exact_match,custom-extract": 0.1111111111111111,
          "exact_match_stderr,custom-extract": 0.11111111111111112
        },
        "mmlu_pro_engineering": {
          "name": "mmlu_pro_engineering",
          "alias": "engineering",
          "sample_len": 9,
          "exact_match,custom-extract": 0.0,
          "exact_match_stderr,custom-extract": 0.0
        },
        "mmlu_pro_health": {
          "name": "mmlu_pro_health",
          "alias": "health",
          "sample_len": 9,
          "exact_match,custom-extract": 0.1111111111111111,
          "exact_match_stderr,custom-extract": 0.1111111111111111
        },
        "mmlu_pro_history": {
          "name": "mmlu_pro_history",
          "alias": "history",
          "sample_len": 9,
          "exact_match,custom-extract": 0.1111111111111111,
          "exact_match_stderr,custom-extract": 0.11111111111111115
        },
        "mmlu_pro_law": {
          "name": "mmlu_pro_law",
          "alias": "law",
          "sample_len": 9,
          "exact_match,custom-extract": 0.1111111111111111,
          "exact_match_stderr,custom-extract": 0.11111111111111115
        },
        "mmlu_pro_math": {
          "name": "mmlu_pro_math",
          "alias": "math",
          "sample_len": 9,
          "exact_match,custom-extract": 0.0,
          "exact_match_stderr,custom-extract": 0.0
        },
        "mmlu_pro_other": {
          "name": "mmlu_pro_other",
          "alias": "other",
          "sample_len": 9,
          "exact_match,custom-extract": 0.1111111111111111,
          "exact_match_stderr,custom-extract": 0.11111111111111112
        },
        "mmlu_pro_philosophy": {
          "name": "mmlu_pro_philosophy",
          "alias": "philosophy",
          "sample_len": 9,
          "exact_match,custom-extract": 0.0,
          "exact_match_stderr,custom-extract": 0.0
        },
        "mmlu_pro_physics": {
          "name": "mmlu_pro_physics",
          "alias": "physics",
          "sample_len": 9,
          "exact_match,custom-extract": 0.0,
          "exact_match_stderr,custom-extract": 0.0
        },
        "mmlu_pro_psychology": {
          "name": "mmlu_pro_psychology",
          "alias": "psychology",
          "sample_len": 9,
          "exact_match,custom-extract": 0.0,
          "exact_match_stderr,custom-extract": 0.0
        },
        "mmlu_pro": {
          "alias": "mmlu_pro",
          "name": "mmlu_pro",
          "sample_len": 128,
          "exact_match,custom-extract": 0.0625,
          "exact_match_stderr,custom-extract": 0.022092779067821488,
          "sample_count": {
            "exact_match,custom-extract": 128
          }
        }
      }
    }
  },
  "diagnostics": {
    "validation_only_not_official_benchmark": true,
    "inference_ablations_not_separately_trained": true,
    "wrong_memory_replaces_question_and_evidence": true,
    "selected_reader_sha256": "2043ff82da83511f25a800c077e2ebfe95646911adc4ffe15feaee0f58e228ea",
    "conditions": {
      "disabled": {
        "n": 4,
        "correct": 1,
        "valid": 2,
        "terminal_counts": {
          "question_stop": 3,
          "length_limit": 1
        },
        "predicted_letters": {
          "B": 1,
          "[invalid]": 2,
          "A": 1
        }
      },
      "full": {
        "n": 4,
        "correct": 2,
        "valid": 3,
        "terminal_counts": {
          "question_stop": 2,
          "length_limit": 2
        },
        "predicted_letters": {
          "A": 2,
          "[invalid]": 1,
          "B": 1
        }
      },
      "gate_only": {
        "n": 4,
        "correct": 1,
        "valid": 2,
        "terminal_counts": {
          "question_stop": 3,
          "length_limit": 1
        },
        "predicted_letters": {
          "B": 1,
          "[invalid]": 2,
          "A": 1
        }
      },
      "masked": {
        "n": 4,
        "correct": 0,
        "valid": 0,
        "terminal_counts": {
          "length_limit": 4
        },
        "predicted_letters": {
          "[invalid]": 4
        }
      },
      "read_only": {
        "n": 4,
        "correct": 2,
        "valid": 3,
        "terminal_counts": {
          "question_stop": 2,
          "length_limit": 2
        },
        "predicted_letters": {
          "A": 2,
          "[invalid]": 1,
          "B": 1
        }
      },
      "rotated": {
        "n": 4,
        "correct": 0,
        "valid": 2,
        "terminal_counts": {
          "question_stop": 3,
          "length_limit": 1
        },
        "predicted_letters": {
          "B": 2,
          "[invalid]": 2
        }
      },
      "wrong_memory": {
        "n": 4,
        "correct": 0,
        "valid": 2,
        "terminal_counts": {
          "question_stop": 3,
          "length_limit": 1
        },
        "predicted_letters": {
          "[invalid]": 2,
          "B": 2
        }
      }
    },
    "option_semantic_consistency": {
      "consistent": 0,
      "pairs": 4
    }
  }
}
```

Raw results: /hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/receiver_gated_reader_v1_seed1234/evaluation; log: /hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/receiver_gated_reader_v1_seed1234/logs/pipeline.log


## nativechat_evidence_kv_baselines_v1_seed1234 — user-requested pause

```json
{
  "experiment": "nativechat_evidence_kv_baselines_v1_seed1234",
  "timestamp_utc": "2026-10-19T02:04:41.455249+00:00",
  "server": "B",
  "status": "paused_by_user_after_llama_stage_a",
  "full_experiment_complete": false,
  "configuration": {
    "name": "nativechat_evidence_kv_baselines_v1_seed1234",
    "protocol": "nativechat_evidence_only_qwen_grid_v1",
    "seed": 1234,
    "server": "B",
    "model_dtype": "float32",
    "map_dtype": "float32",
    "moment_dtype": "float64",
    "attention_implementation": "eager",
    "cpu_threads": 8,
    "models": {
      "llama": "/hy-tmp/yezhe/all_models/models/LLM-Research/Llama-3.2-3B-Instruct",
      "gemma": "/hy-tmp/yezhe/all_models/models/LLM-Research/gemma-3-4b-it",
      "qwen": "/hy-tmp/yezhe/all_models/models/Qwen/Qwen3-4B"
    },
    "train_data": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/data/calibration/sources/hotpot-distractor-train-00000-of-00002.parquet",
    "dev_data": "/hy-tmp/yezhe/异构模型/heterogeneous_multidoc_frozen_writers_v0_seed1234/data/hotpot_dev_distractor_v1.json",
    "old_pilot_manifests": [
      "/hy-tmp/yezhe/异构模型/heterogeneous_multidoc_fullsync_writers_v1_seed1234/reference_sample_ids.json",
      "/hy-tmp/yezhe/异构模型/heterogeneous_multidoc_frozen_writers_v0_seed1234/manifest.json"
    ],
    "train_count": 512,
    "validation_count": 64,
    "pilot_count": 128,
    "max_evidence_qwen_tokens": 1024,
    "max_sender_tokens": 1536,
    "max_new_tokens": 128,
    "generation": "greedy_native_eos_no_custom_answer_parser",
    "thinking": false,
    "capture_batch_questions": 4,
    "cpu_kv_cache_gib": 4,
    "min_initial_free_gpu_gib": 23,
    "model_activation_reserve_gib": 3,
    "our_stage_a": {
      "epochs": 4,
      "learning_rate": 0.001,
      "effective_batch": 8,
      "chunk_tokens": 64,
      "gradient_clip": 30.0,
      "loss": "post_k_norm_pre_rope_K_and_V_NMSE_plus_cosine",
      "hidden_dim": 1024
    },
    "our_stage_b": {
      "epochs": 2,
      "learning_rate": 0.0001,
      "effective_batch": 4,
      "gradient_clip": 30.0,
      "rank": 64,
      "loss": "native_Qwen_full_vocabulary_KL_only_on_teacher_greedy_prefix",
      "projection_chunk_tokens": 8,
      "teacher_max_new_tokens": 128
    },
    "heterofold": {
      "rank": 16,
      "epochs": 4,
      "learning_rate": 0.0003,
      "weight_decay": 0.0001,
      "effective_batch": 8,
      "gradient_clip": 1.0,
      "lambda_k_output": 1.0,
      "layer_offsets": [
        -4,
        0,
        4
      ],
      "moment_mass": "natural_exact_end_aligned_token_mass",
      "loss": "native_attention_KL_plus_per_head_key_post_WO_MSE_plus_value_post_WO_NMSE_detached_routing"
    },
    "checkpoint_selection": "heldout_loss_only_including_epoch_zero",
    "validation_gate": {
      "min_f1_gain_over_question_only": 0.05,
      "paired_bootstrap_lower_bound_positive": true,
      "correct_memory_better_than_wrong": true,
      "bootstrap_replicates": 2000
    },
    "calibration_units": [
      "0",
      "1",
      "joint"
    ],
    "single_source": "one_sender_jointly_encodes_both_complete_gold_paragraphs",
    "fusion_assignments": [
      "llama_llama",
      "gemma_gemma",
      "llama_gemma",
      "gemma_llama"
    ],
    "receiver_trainable": false,
    "old_checkpoints_loaded": false,
    "automatic_github_upload": false,
    "persistent_kv_tensors": false,
    "native_template_date": "18 Oct 2026",
    "scratch_directory": "/dev/shm/nativechat_evidence_kv_baselines_v1_seed1234",
    "scratch_max_gib": 60,
    "durable_stage_a_directory": "/root/nativechat_evidence_kv_baselines_v1_seed1234_selected",
    "native_logit_atol": 0.001,
    "checkpoint_interval_batches": 16
  },
  "checkpoint_selection": "held-out Stage A loss only, including epoch zero",
  "pause_request": {
    "requested_utc": "2026-10-19T01:28:25.783748+00:00",
    "instruction": "Finish current Llama Stage A, preserve results, then pause all later stages.",
    "dispatcher": {
      "pid": 675276,
      "state": "S",
      "ppid": 1,
      "start_ticks": "3561847542",
      "cmd": "/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u /hy-tmp/yezhe/异构模型/nativechat_evidence_kv_baselines_v1_seed1234/run_pipeline.py"
    },
    "trainer": {
      "pid": 680806,
      "state": "R",
      "ppid": 675276,
      "start_ticks": "3561995680",
      "cmd": "/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u /hy-tmp/yezhe/异构模型/nativechat_evidence_kv_baselines_v1_seed1234/train.py llama our_a"
    },
    "controller_pid": 724789,
    "frozen_sources_changed": false,
    "phase": "paused_after_stage_a",
    "dispatcher_stopped": true,
    "stage_a_training_continues": false,
    "completed_utc": "2026-10-19T02:04:41.453739+00:00",
    "selected_checkpoint_verified": true,
    "last_update_checkpoint": {
      "path": "/root/nativechat_evidence_kv_baselines_v1_seed1234_selected/our_llama_stage_a_last.pt",
      "sha256": "5bd07b09be9e97dba2ed2898209493df377fda41729179d77b798d7f106119db",
      "persistent": true,
      "selected_for_stage_b": false,
      "purpose": "Archive final Stage A update; validation-selected checkpoint remains authoritative"
    }
  },
  "llama_stage_a": {
    "completed": true,
    "checkpoint": {
      "path": "/root/nativechat_evidence_kv_baselines_v1_seed1234_selected/our_llama_stage_a.pt",
      "sha256": "91b86f9c1391de16008214b101dc5c885e0878f9fd249a6734e67060d7cd00cb",
      "persistent": true
    },
    "selection": {
      "best": {
        "epoch": 4,
        "validation_loss": 0.8411352104699859,
        "step": 768
      },
      "epochs": [
        {
          "epoch": 0,
          "validation_loss": 4.00296828975103,
          "step": 0
        },
        {
          "epoch": 1,
          "validation_loss": 1.0495939680796689,
          "step": 192
        },
        {
          "epoch": 2,
          "validation_loss": 0.9271993857997698,
          "step": 384
        },
        {
          "epoch": 3,
          "validation_loss": 0.873126953459638,
          "step": 576
        },
        {
          "epoch": 4,
          "validation_loss": 0.8411352104699859,
          "step": 768
        }
      ]
    },
    "updates": 768
  },
  "last_epoch_archive": {
    "path": "/root/nativechat_evidence_kv_baselines_v1_seed1234_selected/our_llama_stage_a_last.pt",
    "sha256": "5bd07b09be9e97dba2ed2898209493df377fda41729179d77b798d7f106119db",
    "persistent": true,
    "selected_for_stage_b": false,
    "purpose": "Archive final Stage A update; validation-selected checkpoint remains authoritative"
  },
  "error": null,
  "evaluation": {
    "evaluation/validation/native/summary.json": {
      "completed": true,
      "split": "validation",
      "pilot_not_official_full_benchmark": true,
      "arms": {
        "native_joint": {
          "n": 64,
          "em": 0.6875,
          "f1": 0.7840844051781554,
          "eos_rate": 1.0,
          "mean_generated_tokens": 5.6875
        },
        "question_only": {
          "n": 64,
          "em": 0.125,
          "f1": 0.20846025910364144,
          "eos_rate": 1.0,
          "mean_generated_tokens": 7.640625
        }
      }
    }
  },
  "paths": {
    "experiment": "/hy-tmp/yezhe/异构模型/nativechat_evidence_kv_baselines_v1_seed1234",
    "logs": "/hy-tmp/yezhe/异构模型/nativechat_evidence_kv_baselines_v1_seed1234/logs/pipeline.log",
    "pause_controller_log": "/hy-tmp/yezhe/异构模型/experiment_controls/nativechat_evidence_kv_baselines_v1_seed1234/pause.log"
  }
}
```


## heterofold_official_l3_to_q4_seed20260825_evaluation_128

```json
{
  "experiment": "heterofold_official_l3_to_q4_seed20260825_evaluation_128",
  "timestamp_utc": "2026-10-19T03:30:59.145186+00:00",
  "status": "incomplete_during_user_requested_gsm8k_pause",
  "error": "BlockingIOError(11, 'Resource temporarily unavailable')",
  "configuration": {
    "name": "heterofold_official_l3_to_q4_seed20260825_evaluation_128",
    "server": "B",
    "seed": 20260825,
    "samples_per_task": 128,
    "sample_indices": "official dataset order [0,128), no filtering",
    "tasks": [
      "arc",
      "gsm8k",
      "qasper"
    ],
    "conditions": [
      "native",
      "recolor",
      "heterofold"
    ],
    "pair": "llama3_to_qwen4",
    "reuse_previous_indices": [
      0,
      32
    ],
    "shard_size": {
      "arc": 32,
      "gsm8k": 4,
      "qasper": 1
    },
    "model_dtype": "float16",
    "capture_device": "cpu",
    "explicit_gqa_tasks": [
      "qasper"
    ],
    "gpu_min_free_gib": 25,
    "cpu_threads": 8,
    "bootstrap_replicates": 2000,
    "wait_for_stage_a_pause": true,
    "training_or_calibration": false,
    "automatic_github_upload": false,
    "protocol": "Existing official evaluators, prompts, parser, context and generation limits, per-item generation seeds retained"
  },
  "completed": false,
  "results": [
    {
      "condition": "native",
      "benchmark": "arc",
      "metric": "accuracy",
      "n": 128,
      "expected": 128,
      "complete": true,
      "mean_percent": 86.71875,
      "ci95_percent": [
        80.46875,
        92.1875
      ]
    },
    {
      "condition": "recolor",
      "benchmark": "arc",
      "metric": "accuracy",
      "n": 128,
      "expected": 128,
      "complete": true,
      "mean_percent": 63.28125,
      "ci95_percent": [
        54.6875,
        71.875
      ]
    },
    {
      "condition": "heterofold",
      "benchmark": "arc",
      "metric": "accuracy",
      "n": 128,
      "expected": 128,
      "complete": true,
      "mean_percent": 57.03125,
      "ci95_percent": [
        47.65625,
        65.625
      ]
    },
    {
      "condition": "native",
      "benchmark": "gsm8k",
      "metric": "accuracy",
      "n": 128,
      "expected": 128,
      "complete": true,
      "mean_percent": 94.53125,
      "ci95_percent": [
        89.84375,
        98.4375
      ]
    },
    {
      "condition": "recolor",
      "benchmark": "gsm8k",
      "metric": "accuracy",
      "n": 128,
      "expected": 128,
      "complete": true,
      "mean_percent": 66.40625,
      "ci95_percent": [
        57.8125,
        75.0
      ]
    },
    {
      "condition": "heterofold",
      "benchmark": "gsm8k",
      "metric": "accuracy",
      "n": 76,
      "expected": 128,
      "complete": false,
      "mean_percent": 50.0,
      "ci95_percent": [
        39.473684210526315,
        60.526315789473685
      ]
    },
    {
      "condition": "native",
      "benchmark": "qasper",
      "metric": "token_f1",
      "n": 32,
      "expected": 128,
      "complete": false,
      "mean_percent": 46.253392750330526,
      "ci95_percent": [
        33.227962651745216,
        58.86406263506073
      ]
    },
    {
      "condition": "recolor",
      "benchmark": "qasper",
      "metric": "token_f1",
      "n": 32,
      "expected": 128,
      "complete": false,
      "mean_percent": 39.840396582527596,
      "ci95_percent": [
        28.194577208763956,
        51.808730092228096
      ]
    },
    {
      "condition": "heterofold",
      "benchmark": "qasper",
      "metric": "token_f1",
      "n": 32,
      "expected": 128,
      "complete": false,
      "mean_percent": 12.089616103720912,
      "ci95_percent": [
        7.119995706389492,
        17.256644952022693
      ]
    }
  ],
  "paired_differences": {
    "arc/heterofold minus recolor": {
      "n": 128,
      "mean_percent": -6.25,
      "ci95_percent": [
        -13.28125,
        0.0
      ]
    },
    "arc/recolor minus native": {
      "n": 128,
      "mean_percent": -23.4375,
      "ci95_percent": [
        -32.03125,
        -15.625
      ]
    },
    "arc/heterofold minus native": {
      "n": 128,
      "mean_percent": -29.6875,
      "ci95_percent": [
        -38.28125,
        -21.09375
      ]
    },
    "gsm8k/heterofold minus recolor": {
      "n": 76,
      "mean_percent": -13.157894736842104,
      "ci95_percent": [
        -25.0,
        -1.3157894736842104
      ]
    },
    "gsm8k/recolor minus native": {
      "n": 128,
      "mean_percent": -28.125,
      "ci95_percent": [
        -36.71875,
        -19.53125
      ]
    },
    "gsm8k/heterofold minus native": {
      "n": 76,
      "mean_percent": -47.368421052631575,
      "ci95_percent": [
        -59.210526315789465,
        -35.526315789473685
      ]
    },
    "qasper/heterofold minus recolor": {
      "n": 32,
      "mean_percent": -27.750780478806693,
      "ci95_percent": [
        -39.17203260042938,
        -17.72363509330059
      ]
    },
    "qasper/recolor minus native": {
      "n": 32,
      "mean_percent": -6.412996167802926,
      "ci95_percent": [
        -18.569730813378573,
        6.12959978253834
      ]
    },
    "qasper/heterofold minus native": {
      "n": 32,
      "mean_percent": -34.16377664660961,
      "ci95_percent": [
        -47.29225982961779,
        -21.730443799883016
      ]
    }
  },
  "original_32_item_results_preserved": true,
  "weights_not_retrained": true,
  "checkpoint_selection": "Fixed existing maps; no selection based on expanded test scores",
  "paths": {
    "raw_results": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_128_seed20260825/results",
    "log": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_128_seed20260825/logs/pipeline.log",
    "manifest": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_128_seed20260825/manifest.json"
  }
}
```


## heterofold_official_l3_to_q4_seed20260825_evaluation_128

```json
{
  "experiment": "heterofold_official_l3_to_q4_seed20260825_evaluation_128",
  "timestamp_utc": "2026-10-19T04:02:40.511465+00:00",
  "status": "paused_by_user_after_gsm8k",
  "error": null,
  "configuration": {
    "name": "heterofold_official_l3_to_q4_seed20260825_evaluation_128",
    "server": "B",
    "seed": 20260825,
    "samples_per_task": 128,
    "sample_indices": "official dataset order [0,128), no filtering",
    "tasks": [
      "arc",
      "gsm8k",
      "qasper"
    ],
    "conditions": [
      "native",
      "recolor",
      "heterofold"
    ],
    "pair": "llama3_to_qwen4",
    "reuse_previous_indices": [
      0,
      32
    ],
    "shard_size": {
      "arc": 32,
      "gsm8k": 4,
      "qasper": 1
    },
    "model_dtype": "float16",
    "capture_device": "cpu",
    "explicit_gqa_tasks": [
      "qasper"
    ],
    "gpu_min_free_gib": 25,
    "cpu_threads": 8,
    "bootstrap_replicates": 2000,
    "wait_for_stage_a_pause": true,
    "training_or_calibration": false,
    "automatic_github_upload": false,
    "protocol": "Existing official evaluators, prompts, parser, context and generation limits, per-item generation seeds retained"
  },
  "completed": false,
  "results": [
    {
      "condition": "native",
      "benchmark": "arc",
      "metric": "accuracy",
      "n": 128,
      "expected": 128,
      "complete": true,
      "mean_percent": 86.71875,
      "ci95_percent": [
        80.46875,
        92.1875
      ]
    },
    {
      "condition": "recolor",
      "benchmark": "arc",
      "metric": "accuracy",
      "n": 128,
      "expected": 128,
      "complete": true,
      "mean_percent": 63.28125,
      "ci95_percent": [
        54.6875,
        71.875
      ]
    },
    {
      "condition": "heterofold",
      "benchmark": "arc",
      "metric": "accuracy",
      "n": 128,
      "expected": 128,
      "complete": true,
      "mean_percent": 57.03125,
      "ci95_percent": [
        47.65625,
        65.625
      ]
    },
    {
      "condition": "native",
      "benchmark": "gsm8k",
      "metric": "accuracy",
      "n": 128,
      "expected": 128,
      "complete": true,
      "mean_percent": 94.53125,
      "ci95_percent": [
        89.84375,
        98.4375
      ]
    },
    {
      "condition": "recolor",
      "benchmark": "gsm8k",
      "metric": "accuracy",
      "n": 128,
      "expected": 128,
      "complete": true,
      "mean_percent": 66.40625,
      "ci95_percent": [
        57.8125,
        75.0
      ]
    },
    {
      "condition": "heterofold",
      "benchmark": "gsm8k",
      "metric": "accuracy",
      "n": 128,
      "expected": 128,
      "complete": true,
      "mean_percent": 54.6875,
      "ci95_percent": [
        46.09375,
        63.28125
      ]
    },
    {
      "condition": "native",
      "benchmark": "qasper",
      "metric": "token_f1",
      "n": 32,
      "expected": 128,
      "complete": false,
      "mean_percent": 46.253392750330526,
      "ci95_percent": [
        33.227962651745216,
        58.86406263506073
      ]
    },
    {
      "condition": "recolor",
      "benchmark": "qasper",
      "metric": "token_f1",
      "n": 32,
      "expected": 128,
      "complete": false,
      "mean_percent": 39.840396582527596,
      "ci95_percent": [
        28.194577208763956,
        51.808730092228096
      ]
    },
    {
      "condition": "heterofold",
      "benchmark": "qasper",
      "metric": "token_f1",
      "n": 32,
      "expected": 128,
      "complete": false,
      "mean_percent": 12.089616103720912,
      "ci95_percent": [
        7.119995706389492,
        17.256644952022693
      ]
    }
  ],
  "paired_differences": {
    "arc/heterofold minus recolor": {
      "n": 128,
      "mean_percent": -6.25,
      "ci95_percent": [
        -13.28125,
        0.0
      ]
    },
    "arc/recolor minus native": {
      "n": 128,
      "mean_percent": -23.4375,
      "ci95_percent": [
        -32.03125,
        -15.625
      ]
    },
    "arc/heterofold minus native": {
      "n": 128,
      "mean_percent": -29.6875,
      "ci95_percent": [
        -38.28125,
        -21.09375
      ]
    },
    "gsm8k/heterofold minus recolor": {
      "n": 128,
      "mean_percent": -11.71875,
      "ci95_percent": [
        -20.3125,
        -3.125
      ]
    },
    "gsm8k/recolor minus native": {
      "n": 128,
      "mean_percent": -28.125,
      "ci95_percent": [
        -36.71875,
        -19.53125
      ]
    },
    "gsm8k/heterofold minus native": {
      "n": 128,
      "mean_percent": -39.84375,
      "ci95_percent": [
        -49.23828125,
        -30.46875
      ]
    },
    "qasper/heterofold minus recolor": {
      "n": 32,
      "mean_percent": -27.750780478806693,
      "ci95_percent": [
        -39.17203260042938,
        -17.72363509330059
      ]
    },
    "qasper/recolor minus native": {
      "n": 32,
      "mean_percent": -6.412996167802926,
      "ci95_percent": [
        -18.569730813378573,
        6.12959978253834
      ]
    },
    "qasper/heterofold minus native": {
      "n": 32,
      "mean_percent": -34.16377664660961,
      "ci95_percent": [
        -47.29225982961779,
        -21.730443799883016
      ]
    }
  },
  "original_32_item_results_preserved": true,
  "weights_not_retrained": true,
  "checkpoint_selection": "Fixed existing maps; no selection based on expanded test scores",
  "paths": {
    "raw_results": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_128_seed20260825/results",
    "log": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_128_seed20260825/logs/pipeline.log",
    "manifest": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_128_seed20260825/manifest.json"
  }
}
```


## frozen_gsm8k_writer_vs_heterofold_arc128_hellaswag64_seed1234 / heterofold

```json
{
  "experiment": "frozen_gsm8k_writer_vs_heterofold_arc128_hellaswag64_seed1234",
  "time": "2026-10-19 17:03:10",
  "status": "completed_side_only",
  "role": "heterofold",
  "config": {
    "name": "frozen_gsm8k_writer_vs_heterofold_arc128_hellaswag64_seed1234",
    "seed": 1234,
    "samples": {
      "arc": 128,
      "hellaswag": 64
    },
    "smoke_per_task": 2,
    "cpu_threads": 1,
    "dtype": "float16",
    "server_assignment": "Our existing frozen GSM8K Writer on A; HeteroFold existing map on B. No weight migration and no training.",
    "server_b_hf_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825",
    "server_b_output_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_crossdataset_arc128_hellaswag64_seed1234",
    "hf_map": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/maps/llama3_to_qwen4/heterofold.pt",
    "hf_map_sha256": "dc2ea778b2b6ea4d6c74c678548946f59fb1e5970f935be3ecc5e4fb58ac718e",
    "models_b": {
      "llama": "/hy-tmp/yezhe/all_models/models/LLM-Research/Llama-3.2-3B-Instruct",
      "qwen": "/hy-tmp/yezhe/all_models/models/Qwen/Qwen3-4B"
    },
    "reference_hashes": {
      "evaluation/short_context.py": "5528df3b3a8f54697ab2fb4c8d5748e3e6050eb0155a76cfaa0ef4bb3719d383",
      "evaluation/runtime.py": "baed2e7fa47c1acc00c43d0c7baaea225beaa3e5f4b6b4e0601294103b84d8a1",
      "modeling.py": "c36066d40ae9836cf31f324603af9fd1a1af3c70af2d61c0e492456a367749e8"
    },
    "scoring": "Exact HeteroFold format_item, ANSWER_CUE, native_candidate_scores and transfer_candidate_scores. ARC label likelihood; HellaSwag full-continuation mean token log-likelihood. No generation or numeric parser.",
    "reuse_arc": "Reuse immutable Native/Recolor/HeteroFold test128 on B, after row fingerprints and gold validation. Verify A Native predictions on two smoke rows.",
    "our_protocol": "Fixed GSM8K Stage-A epoch2 + Hybrid Stage-B 2048x4 epoch4, both models Chat; same native anchor, punctuation-only native boundary and Full-Sync mapping as the completed GSM8K chat-only evaluation.",
    "heterofold_protocol": "Original raw Sender capture, fixed folded map, native Receiver chat prefix and original KV alignment/finalization. Sequential model residency with FP16 CPU-staged source KV, not simultaneous Sender/Receiver GPU loading.",
    "minimum_free_gpu_gib": {
      "llama": 8.0,
      "qwen": 9.5,
      "writer": 3.0
    },
    "native_cache_choice_kl_max": 0.001,
    "training": false,
    "automatic_upload": false,
    "continuous_monitoring": false
  },
  "checkpoint_selection": "Fixed existing checkpoints/maps; no test tuning",
  "results": {
    "experiment": "frozen_gsm8k_writer_vs_heterofold_arc128_hellaswag64_seed1234",
    "role": "heterofold",
    "tasks": {
      "hellaswag": {
        "metrics": {
          "heterofold": {
            "n": 64,
            "correct": 26,
            "accuracy": 0.40625
          },
          "native": {
            "n": 64,
            "correct": 23,
            "accuracy": 0.359375
          }
        },
        "paired": {
          "heterofold__vs__native": {
            "n": 64,
            "both_correct": 19,
            "only_a": 7,
            "only_b": 4,
            "both_wrong": 34
          }
        }
      }
    },
    "scope": "side-specific results; final merged comparison has both servers",
    "primary_metrics": {
      "arc": "label likelihood accuracy",
      "hellaswag": "length-normalized continuation likelihood accuracy"
    }
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_crossdataset_arc128_hellaswag64_seed1234/outputs/heterofold/formal/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_crossdataset_arc128_hellaswag64_seed1234/logs/heterofold.log"
}
```


## frozen_mcq_writer_arc128_hellaswag64_heterofold_protocol_seed1234 / our

```json
{
  "experiment": "frozen_mcq_writer_arc128_hellaswag64_heterofold_protocol_seed1234",
  "time": "2026-10-19 17:47:47",
  "status": "completed_side_only",
  "role": "our",
  "config": {
    "name": "frozen_mcq_writer_arc128_hellaswag64_heterofold_protocol_seed1234",
    "seed": 1234,
    "samples": {
      "arc": 128,
      "hellaswag": 64
    },
    "smoke_per_task": 2,
    "cpu_threads": 1,
    "dtype": "float16",
    "server_assignment": "Only MCQ Writer evaluated on B with original resident weights; completed GSM8K/Native/HeteroFold scores reused. No weights migrated.",
    "server_b_hf_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825",
    "server_b_output_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_crossdataset_arc128_hellaswag64_seed1234/mcq_writer",
    "hf_map": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/maps/llama3_to_qwen4/heterofold.pt",
    "hf_map_sha256": "dc2ea778b2b6ea4d6c74c678548946f59fb1e5970f935be3ecc5e4fb58ac718e",
    "models_b": {
      "llama": "/hy-tmp/yezhe/all_models/models/LLM-Research/Llama-3.2-3B-Instruct",
      "qwen": "/hy-tmp/yezhe/all_models/models/Qwen/Qwen3-4B"
    },
    "reference_hashes": {
      "evaluation/short_context.py": "5528df3b3a8f54697ab2fb4c8d5748e3e6050eb0155a76cfaa0ef4bb3719d383",
      "evaluation/runtime.py": "baed2e7fa47c1acc00c43d0c7baaea225beaa3e5f4b6b4e0601294103b84d8a1",
      "modeling.py": "c36066d40ae9836cf31f324603af9fd1a1af3c70af2d61c0e492456a367749e8"
    },
    "scoring": "Exact HeteroFold format_item, ANSWER_CUE, native_candidate_scores and transfer_candidate_scores. ARC label likelihood; HellaSwag full-continuation mean token log-likelihood. No generation or numeric parser.",
    "reuse_arc": "Reuse immutable Native/Recolor/HeteroFold test128 on B, after row fingerprints and gold validation. Verify A Native predictions on two smoke rows.",
    "our_protocol": "Existing MCQ Stage-A best epoch2 step512 + Stage-B best epoch4 step1024 rank64; exact prior chat/native anchor/Full-Sync/punctuation-tail interface. Only weights changed.",
    "heterofold_protocol": "Original raw Sender capture, fixed folded map, native Receiver chat prefix and original KV alignment/finalization. Sequential model residency with FP16 CPU-staged source KV, not simultaneous Sender/Receiver GPU loading.",
    "minimum_free_gpu_gib": {
      "llama": 8,
      "qwen": 9.5,
      "writer": 3
    },
    "native_cache_choice_kl_max": 0.001,
    "training": false,
    "automatic_upload": false,
    "continuous_monitoring": false,
    "checkpoint_sha256": {
      "stage_a": "62beafbd377374ed8b6a3f15752150a2c23ef81e114c72147d93d3edd527671e",
      "adapter": "663c212151816abf1a44e0db441741389555743b6bcda84cae58c117307f1fc9"
    }
  },
  "checkpoint_selection": "Fixed existing checkpoints/maps; no test tuning",
  "results": {
    "experiment": "frozen_mcq_writer_arc128_hellaswag64_heterofold_protocol_seed1234",
    "role": "our",
    "tasks": {
      "arc": {
        "metrics": {
          "heterofold": {
            "n": 128,
            "correct": 73,
            "accuracy": 0.5703125
          },
          "native": {
            "n": 128,
            "correct": 111,
            "accuracy": 0.8671875
          },
          "ours": {
            "n": 128,
            "correct": 26,
            "accuracy": 0.203125
          },
          "recolor": {
            "n": 128,
            "correct": 81,
            "accuracy": 0.6328125
          }
        },
        "paired": {
          "heterofold__vs__native": {
            "n": 128,
            "both_correct": 70,
            "only_a": 3,
            "only_b": 41,
            "both_wrong": 14
          },
          "heterofold__vs__ours": {
            "n": 128,
            "both_correct": 23,
            "only_a": 50,
            "only_b": 3,
            "both_wrong": 52
          },
          "heterofold__vs__recolor": {
            "n": 128,
            "both_correct": 66,
            "only_a": 7,
            "only_b": 15,
            "both_wrong": 40
          },
          "native__vs__ours": {
            "n": 128,
            "both_correct": 23,
            "only_a": 88,
            "only_b": 3,
            "both_wrong": 14
          },
          "native__vs__recolor": {
            "n": 128,
            "both_correct": 77,
            "only_a": 34,
            "only_b": 4,
            "both_wrong": 13
          },
          "ours__vs__recolor": {
            "n": 128,
            "both_correct": 21,
            "only_a": 5,
            "only_b": 60,
            "both_wrong": 42
          }
        }
      },
      "hellaswag": {
        "metrics": {
          "native": {
            "n": 64,
            "correct": 23,
            "accuracy": 0.359375
          },
          "ours": {
            "n": 64,
            "correct": 24,
            "accuracy": 0.375
          }
        },
        "paired": {
          "native__vs__ours": {
            "n": 64,
            "both_correct": 13,
            "only_a": 10,
            "only_b": 11,
            "both_wrong": 30
          }
        }
      }
    },
    "scope": "side-specific results; final merged comparison has both servers",
    "primary_metrics": {
      "arc": "label likelihood accuracy",
      "hellaswag": "length-normalized continuation likelihood accuracy"
    }
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_crossdataset_arc128_hellaswag64_seed1234/mcq_writer/outputs/our/formal/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_crossdataset_arc128_hellaswag64_seed1234/mcq_writer/logs/our.log"
}
```


## frozen_mcq_writer_arc128_hellaswag64_heterofold_protocol_seed1234 / our

```json
{
  "experiment": "frozen_mcq_writer_arc128_hellaswag64_heterofold_protocol_seed1234",
  "time": "2026-10-19 17:47:52",
  "status": "completed_mcq_followup_comparison",
  "role": "our",
  "config": {
    "name": "frozen_mcq_writer_arc128_hellaswag64_heterofold_protocol_seed1234",
    "seed": 1234,
    "samples": {
      "arc": 128,
      "hellaswag": 64
    },
    "smoke_per_task": 2,
    "cpu_threads": 1,
    "dtype": "float16",
    "server_assignment": "Only MCQ Writer evaluated on B with original resident weights; completed GSM8K/Native/HeteroFold scores reused. No weights migrated.",
    "server_b_hf_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825",
    "server_b_output_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_crossdataset_arc128_hellaswag64_seed1234/mcq_writer",
    "hf_map": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/maps/llama3_to_qwen4/heterofold.pt",
    "hf_map_sha256": "dc2ea778b2b6ea4d6c74c678548946f59fb1e5970f935be3ecc5e4fb58ac718e",
    "models_b": {
      "llama": "/hy-tmp/yezhe/all_models/models/LLM-Research/Llama-3.2-3B-Instruct",
      "qwen": "/hy-tmp/yezhe/all_models/models/Qwen/Qwen3-4B"
    },
    "reference_hashes": {
      "evaluation/short_context.py": "5528df3b3a8f54697ab2fb4c8d5748e3e6050eb0155a76cfaa0ef4bb3719d383",
      "evaluation/runtime.py": "baed2e7fa47c1acc00c43d0c7baaea225beaa3e5f4b6b4e0601294103b84d8a1",
      "modeling.py": "c36066d40ae9836cf31f324603af9fd1a1af3c70af2d61c0e492456a367749e8"
    },
    "scoring": "Exact HeteroFold format_item, ANSWER_CUE, native_candidate_scores and transfer_candidate_scores. ARC label likelihood; HellaSwag full-continuation mean token log-likelihood. No generation or numeric parser.",
    "reuse_arc": "Reuse immutable Native/Recolor/HeteroFold test128 on B, after row fingerprints and gold validation. Verify A Native predictions on two smoke rows.",
    "our_protocol": "Existing MCQ Stage-A best epoch2 step512 + Stage-B best epoch4 step1024 rank64; exact prior chat/native anchor/Full-Sync/punctuation-tail interface. Only weights changed.",
    "heterofold_protocol": "Original raw Sender capture, fixed folded map, native Receiver chat prefix and original KV alignment/finalization. Sequential model residency with FP16 CPU-staged source KV, not simultaneous Sender/Receiver GPU loading.",
    "minimum_free_gpu_gib": {
      "llama": 8,
      "qwen": 9.5,
      "writer": 3
    },
    "native_cache_choice_kl_max": 0.001,
    "training": false,
    "automatic_upload": false,
    "continuous_monitoring": false,
    "checkpoint_sha256": {
      "stage_a": "62beafbd377374ed8b6a3f15752150a2c23ef81e114c72147d93d3edd527671e",
      "adapter": "663c212151816abf1a44e0db441741389555743b6bcda84cae58c117307f1fc9"
    }
  },
  "checkpoint_selection": "Fixed existing checkpoints/maps; no test tuning",
  "results": {
    "experiment": "frozen_mcq_writer_arc128_hellaswag64_heterofold_protocol_seed1234",
    "role": "our",
    "tasks": {
      "arc": {
        "metrics": {
          "gsm8k_writer": {
            "n": 128,
            "correct": 30,
            "accuracy": 0.234375
          },
          "heterofold": {
            "n": 128,
            "correct": 73,
            "accuracy": 0.5703125
          },
          "mcq_writer": {
            "n": 128,
            "correct": 26,
            "accuracy": 0.203125
          },
          "native": {
            "n": 128,
            "correct": 111,
            "accuracy": 0.8671875
          },
          "recolor": {
            "n": 128,
            "correct": 81,
            "accuracy": 0.6328125
          }
        },
        "paired": {
          "gsm8k_writer__vs__heterofold": {
            "n": 128,
            "both_correct": 23,
            "only_a": 7,
            "only_b": 50,
            "both_wrong": 48
          },
          "gsm8k_writer__vs__mcq_writer": {
            "n": 128,
            "both_correct": 20,
            "only_a": 10,
            "only_b": 6,
            "both_wrong": 92
          },
          "gsm8k_writer__vs__native": {
            "n": 128,
            "both_correct": 27,
            "only_a": 3,
            "only_b": 84,
            "both_wrong": 14
          },
          "gsm8k_writer__vs__recolor": {
            "n": 128,
            "both_correct": 24,
            "only_a": 6,
            "only_b": 57,
            "both_wrong": 41
          },
          "heterofold__vs__mcq_writer": {
            "n": 128,
            "both_correct": 23,
            "only_a": 50,
            "only_b": 3,
            "both_wrong": 52
          },
          "heterofold__vs__native": {
            "n": 128,
            "both_correct": 70,
            "only_a": 3,
            "only_b": 41,
            "both_wrong": 14
          },
          "heterofold__vs__recolor": {
            "n": 128,
            "both_correct": 66,
            "only_a": 7,
            "only_b": 15,
            "both_wrong": 40
          },
          "mcq_writer__vs__native": {
            "n": 128,
            "both_correct": 23,
            "only_a": 3,
            "only_b": 88,
            "both_wrong": 14
          },
          "mcq_writer__vs__recolor": {
            "n": 128,
            "both_correct": 21,
            "only_a": 5,
            "only_b": 60,
            "both_wrong": 42
          },
          "native__vs__recolor": {
            "n": 128,
            "both_correct": 77,
            "only_a": 34,
            "only_b": 4,
            "both_wrong": 13
          }
        }
      },
      "hellaswag": {
        "metrics": {
          "gsm8k_writer": {
            "n": 64,
            "correct": 22,
            "accuracy": 0.34375
          },
          "heterofold": {
            "n": 64,
            "correct": 26,
            "accuracy": 0.40625
          },
          "mcq_writer": {
            "n": 64,
            "correct": 24,
            "accuracy": 0.375
          },
          "native": {
            "n": 64,
            "correct": 23,
            "accuracy": 0.359375
          }
        },
        "paired": {
          "gsm8k_writer__vs__heterofold": {
            "n": 64,
            "both_correct": 17,
            "only_a": 5,
            "only_b": 9,
            "both_wrong": 33
          },
          "gsm8k_writer__vs__mcq_writer": {
            "n": 64,
            "both_correct": 18,
            "only_a": 4,
            "only_b": 6,
            "both_wrong": 36
          },
          "gsm8k_writer__vs__native": {
            "n": 64,
            "both_correct": 15,
            "only_a": 7,
            "only_b": 8,
            "both_wrong": 34
          },
          "heterofold__vs__mcq_writer": {
            "n": 64,
            "both_correct": 16,
            "only_a": 10,
            "only_b": 8,
            "both_wrong": 30
          },
          "heterofold__vs__native": {
            "n": 64,
            "both_correct": 19,
            "only_a": 7,
            "only_b": 4,
            "both_wrong": 34
          },
          "mcq_writer__vs__native": {
            "n": 64,
            "both_correct": 13,
            "only_a": 11,
            "only_b": 10,
            "both_wrong": 30
          }
        }
      }
    },
    "scope": "Only MCQ Writer new; Native/Recolor/HeteroFold/GSM8K Writer reused",
    "primary_metrics": {
      "arc": "label likelihood accuracy",
      "hellaswag": "length-normalized continuation likelihood accuracy"
    },
    "status": "completed",
    "manifest_sha256": "ec67a8fcd8203406d9e14f832275c34d471448d7119c315fa583d263108fed46",
    "checkpoint_selection": "Existing MCQ validation-loss-selected best checkpoints; no test tuning",
    "caveats": [
      "MCQ training used no chat template; this evaluates frozen weights under the same chat interface as the preceding GSM8K Writer",
      "HeteroFold has different training data and Sender interface; not a same-training-budget architecture comparison"
    ]
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_crossdataset_arc128_hellaswag64_seed1234/mcq_writer/outputs/our/formal/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_crossdataset_arc128_hellaswag64_seed1234/mcq_writer/logs/our.log"
}
```


## mcq_heterofold_chat_fullsync_l3_to_q4_seed1234

```json
{
  "experiment": "mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
  "time": "2026-10-19 20:55:47",
  "status": "failed_or_incomplete",
  "config": {
    "name": "mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
    "seed": 1234,
    "legacy_root": "/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234",
    "hf_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825",
    "previous_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_crossdataset_arc128_hellaswag64_seed1234/mcq_writer",
    "models": {
      "llama": "/hy-tmp/yezhe/all_models/models/LLM-Research/Llama-3.2-3B-Instruct",
      "qwen": "/hy-tmp/yezhe/all_models/models/Qwen/Qwen3-4B"
    },
    "train_counts": {
      "openbookqa": 1024,
      "arc_challenge": 1024
    },
    "validation_counts": {
      "openbookqa": 128,
      "arc_challenge": 128
    },
    "stage_a_epochs": 2,
    "stage_b_epochs": 4,
    "effective_batch": 8,
    "stage_a_learning_rate": 0.001,
    "stage_b_learning_rate": 0.0001,
    "gradient_clip": 30,
    "weight_decay": 0,
    "adapter_rank": 64,
    "mlp_hidden_dim": 1024,
    "chunk_tokens": 64,
    "checkpoint_every_steps": 128,
    "llama_template_date": "19 Oct 2026",
    "dtype": "float16",
    "attention": "sdpa",
    "native_cache_choice_kl_max": 0.001,
    "min_free_gpu_gib": {
      "llama": 8,
      "qwen": 9.5,
      "stage_a": 5,
      "stage_b": 10.5
    },
    "disk_reserve_gib": 20,
    "reference_hashes": {
      "evaluation/short_context.py": "5528df3b3a8f54697ab2fb4c8d5748e3e6050eb0155a76cfaa0ef4bb3719d383",
      "evaluation/runtime.py": "baed2e7fa47c1acc00c43d0c7baaea225beaa3e5f4b6b4e0601294103b84d8a1",
      "modeling.py": "c36066d40ae9836cf31f324603af9fd1a1af3c70af2d61c0e492456a367749e8"
    },
    "losses": {
      "stage_a": "K NMSE + V NMSE + (1-K cosine) + (1-V cosine), per example; translated slots only",
      "stage_b": "KL(native Qwen candidate distribution || transferred candidate distribution), temperature1; no gold CE"
    },
    "checkpoint_selection": "lowest held-out validation reconstruction loss for Stage-A; lowest held-out validation choice KL for Stage-B",
    "evaluation": {
      "arc": 128,
      "hellaswag": 64
    },
    "training_gsm8k": false,
    "continuous_monitoring": false,
    "automatic_upload": false
  },
  "checkpoint_selection": "lowest held-out validation reconstruction loss for Stage-A; lowest held-out validation choice KL for Stage-B",
  "results": {
    "type": "AssertionError",
    "message": "('Official document changed', 'openbookqa:db68d96ac4ce6f947420fa7b4d711d285672b3e430d3d99ba9b89f3216972a90')"
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/logs/pipeline.log"
}
```


## mcq_heterofold_chat_fullsync_l3_to_q4_seed1234

```json
{
  "experiment": "mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
  "time": "2026-10-19 21:43:04",
  "status": "stage_a_completed_evaluation_pending",
  "config": {
    "name": "mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
    "seed": 1234,
    "legacy_root": "/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234",
    "hf_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825",
    "previous_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_crossdataset_arc128_hellaswag64_seed1234/mcq_writer",
    "models": {
      "llama": "/hy-tmp/yezhe/all_models/models/LLM-Research/Llama-3.2-3B-Instruct",
      "qwen": "/hy-tmp/yezhe/all_models/models/Qwen/Qwen3-4B"
    },
    "train_counts": {
      "openbookqa": 1024,
      "arc_challenge": 1024
    },
    "validation_counts": {
      "openbookqa": 128,
      "arc_challenge": 128
    },
    "stage_a_epochs": 2,
    "stage_b_epochs": 4,
    "effective_batch": 8,
    "stage_a_learning_rate": 0.001,
    "stage_b_learning_rate": 0.0001,
    "gradient_clip": 30,
    "weight_decay": 0,
    "adapter_rank": 64,
    "mlp_hidden_dim": 1024,
    "chunk_tokens": 64,
    "checkpoint_every_steps": 128,
    "llama_template_date": "19 Oct 2026",
    "dtype": "float16",
    "attention": "sdpa",
    "native_cache_choice_kl_max": 0.001,
    "min_free_gpu_gib": {
      "llama": 8,
      "qwen": 9.5,
      "stage_a": 5,
      "stage_b": 10.5
    },
    "disk_reserve_gib": 20,
    "reference_hashes": {
      "evaluation/short_context.py": "5528df3b3a8f54697ab2fb4c8d5748e3e6050eb0155a76cfaa0ef4bb3719d383",
      "evaluation/runtime.py": "baed2e7fa47c1acc00c43d0c7baaea225beaa3e5f4b6b4e0601294103b84d8a1",
      "modeling.py": "c36066d40ae9836cf31f324603af9fd1a1af3c70af2d61c0e492456a367749e8"
    },
    "losses": {
      "stage_a": "K NMSE + V NMSE + (1-K cosine) + (1-V cosine), per example; translated slots only",
      "stage_b": "KL(native Qwen candidate distribution || transferred candidate distribution), temperature1; no gold CE"
    },
    "checkpoint_selection": "lowest held-out validation reconstruction loss for Stage-A; lowest held-out validation choice KL for Stage-B",
    "evaluation": {
      "arc": 128,
      "hellaswag": 64
    },
    "training_gsm8k": false,
    "continuous_monitoring": false,
    "automatic_upload": false
  },
  "checkpoint_selection": "lowest held-out validation reconstruction loss for Stage-A; lowest held-out validation choice KL for Stage-B",
  "results": {
    "status": "completed",
    "epochs": 2,
    "steps": 512,
    "best": {
      "epoch": 2,
      "step": 512,
      "validation": {
        "total": 0.48218073847215237,
        "k_nmse": 0.03598784277289724,
        "v_nmse": 0.25432120699861116,
        "k_cosine": 0.9604574938966834,
        "v_cosine": 0.8476708216003317
      }
    },
    "best_sha256": "db1f295c502f27305a9d2ac410c47a1bdef4db7e0255f817483867b19d78cf0d",
    "manifest_sha256": "b5c8ddb4500933320e8f2262fa51215bcd6d60837e1cbccc59910154b7e1c2d6"
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/logs/pipeline.log"
}
```


## mcq_heterofold_chat_fullsync_l3_to_q4_seed1234

```json
{
  "experiment": "mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
  "time": "2026-10-20 02:46:53",
  "status": "stage_b_completed_evaluation_pending",
  "config": {
    "name": "mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
    "seed": 1234,
    "legacy_root": "/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234",
    "hf_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825",
    "previous_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_crossdataset_arc128_hellaswag64_seed1234/mcq_writer",
    "models": {
      "llama": "/hy-tmp/yezhe/all_models/models/LLM-Research/Llama-3.2-3B-Instruct",
      "qwen": "/hy-tmp/yezhe/all_models/models/Qwen/Qwen3-4B"
    },
    "train_counts": {
      "openbookqa": 1024,
      "arc_challenge": 1024
    },
    "validation_counts": {
      "openbookqa": 128,
      "arc_challenge": 128
    },
    "stage_a_epochs": 2,
    "stage_b_epochs": 4,
    "effective_batch": 8,
    "stage_a_learning_rate": 0.001,
    "stage_b_learning_rate": 0.0001,
    "gradient_clip": 30,
    "weight_decay": 0,
    "adapter_rank": 64,
    "mlp_hidden_dim": 1024,
    "chunk_tokens": 64,
    "checkpoint_every_steps": 128,
    "llama_template_date": "19 Oct 2026",
    "dtype": "float16",
    "attention": "sdpa",
    "native_cache_choice_kl_max": 0.001,
    "min_free_gpu_gib": {
      "llama": 8,
      "qwen": 9.5,
      "stage_a": 5,
      "stage_b": 10.5
    },
    "disk_reserve_gib": 20,
    "reference_hashes": {
      "evaluation/short_context.py": "5528df3b3a8f54697ab2fb4c8d5748e3e6050eb0155a76cfaa0ef4bb3719d383",
      "evaluation/runtime.py": "baed2e7fa47c1acc00c43d0c7baaea225beaa3e5f4b6b4e0601294103b84d8a1",
      "modeling.py": "c36066d40ae9836cf31f324603af9fd1a1af3c70af2d61c0e492456a367749e8"
    },
    "losses": {
      "stage_a": "K NMSE + V NMSE + (1-K cosine) + (1-V cosine), per example; translated slots only",
      "stage_b": "KL(native Qwen candidate distribution || transferred candidate distribution), temperature1; no gold CE"
    },
    "checkpoint_selection": "lowest held-out validation reconstruction loss for Stage-A; lowest held-out validation choice KL for Stage-B",
    "evaluation": {
      "arc": 128,
      "hellaswag": 64
    },
    "training_gsm8k": false,
    "continuous_monitoring": false,
    "automatic_upload": false
  },
  "checkpoint_selection": "lowest held-out validation reconstruction loss for Stage-A; lowest held-out validation choice KL for Stage-B",
  "results": {
    "status": "completed",
    "epochs": 4,
    "steps": 1024,
    "best": {
      "epoch": 1,
      "step": 256,
      "validation": {
        "total": 0.5738657102564275,
        "choice_kl": 0.5738657102564275,
        "native_correct": 0.8359375,
        "writer_correct": 0.7109375
      }
    },
    "best_sha256": "06190cfcda6ce65e7b81447d2fa77b926bd43b384e2bfb5866083cc1588e278f",
    "manifest_sha256": "b5c8ddb4500933320e8f2262fa51215bcd6d60837e1cbccc59910154b7e1c2d6"
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/logs/pipeline.log"
}
```


## mcq_heterofold_chat_fullsync_l3_to_q4_seed1234

```json
{
  "experiment": "mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
  "time": "2026-10-20 02:49:30",
  "status": "completed_retraining_and_evaluation",
  "config": {
    "name": "mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
    "seed": 1234,
    "legacy_root": "/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234",
    "hf_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825",
    "previous_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_crossdataset_arc128_hellaswag64_seed1234/mcq_writer",
    "models": {
      "llama": "/hy-tmp/yezhe/all_models/models/LLM-Research/Llama-3.2-3B-Instruct",
      "qwen": "/hy-tmp/yezhe/all_models/models/Qwen/Qwen3-4B"
    },
    "train_counts": {
      "openbookqa": 1024,
      "arc_challenge": 1024
    },
    "validation_counts": {
      "openbookqa": 128,
      "arc_challenge": 128
    },
    "stage_a_epochs": 2,
    "stage_b_epochs": 4,
    "effective_batch": 8,
    "stage_a_learning_rate": 0.001,
    "stage_b_learning_rate": 0.0001,
    "gradient_clip": 30,
    "weight_decay": 0,
    "adapter_rank": 64,
    "mlp_hidden_dim": 1024,
    "chunk_tokens": 64,
    "checkpoint_every_steps": 128,
    "llama_template_date": "19 Oct 2026",
    "dtype": "float16",
    "attention": "sdpa",
    "native_cache_choice_kl_max": 0.001,
    "min_free_gpu_gib": {
      "llama": 8,
      "qwen": 9.5,
      "stage_a": 5,
      "stage_b": 10.5
    },
    "disk_reserve_gib": 20,
    "reference_hashes": {
      "evaluation/short_context.py": "5528df3b3a8f54697ab2fb4c8d5748e3e6050eb0155a76cfaa0ef4bb3719d383",
      "evaluation/runtime.py": "baed2e7fa47c1acc00c43d0c7baaea225beaa3e5f4b6b4e0601294103b84d8a1",
      "modeling.py": "c36066d40ae9836cf31f324603af9fd1a1af3c70af2d61c0e492456a367749e8"
    },
    "losses": {
      "stage_a": "K NMSE + V NMSE + (1-K cosine) + (1-V cosine), per example; translated slots only",
      "stage_b": "KL(native Qwen candidate distribution || transferred candidate distribution), temperature1; no gold CE"
    },
    "checkpoint_selection": "lowest held-out validation reconstruction loss for Stage-A; lowest held-out validation choice KL for Stage-B",
    "evaluation": {
      "arc": 128,
      "hellaswag": 64
    },
    "training_gsm8k": false,
    "continuous_monitoring": false,
    "automatic_upload": false
  },
  "checkpoint_selection": "lowest held-out validation reconstruction loss for Stage-A; lowest held-out validation choice KL for Stage-B",
  "results": {
    "experiment": "mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
    "tasks": {
      "arc": {
        "metrics": {
          "gsm8k_writer": {
            "n": 128,
            "correct": 30,
            "accuracy": 0.234375
          },
          "heterofold": {
            "n": 128,
            "correct": 73,
            "accuracy": 0.5703125
          },
          "mcq_retrained": {
            "n": 128,
            "correct": 94,
            "accuracy": 0.734375
          },
          "mcq_writer": {
            "n": 128,
            "correct": 26,
            "accuracy": 0.203125
          },
          "native": {
            "n": 128,
            "correct": 111,
            "accuracy": 0.8671875
          },
          "recolor": {
            "n": 128,
            "correct": 81,
            "accuracy": 0.6328125
          }
        },
        "paired": {
          "gsm8k_writer__vs__heterofold": {
            "both_correct": 23,
            "only_a": 7,
            "only_b": 50,
            "both_wrong": 48
          },
          "gsm8k_writer__vs__mcq_retrained": {
            "both_correct": 23,
            "only_a": 7,
            "only_b": 71,
            "both_wrong": 27
          },
          "gsm8k_writer__vs__mcq_writer": {
            "both_correct": 20,
            "only_a": 10,
            "only_b": 6,
            "both_wrong": 92
          },
          "gsm8k_writer__vs__native": {
            "both_correct": 27,
            "only_a": 3,
            "only_b": 84,
            "both_wrong": 14
          },
          "gsm8k_writer__vs__recolor": {
            "both_correct": 24,
            "only_a": 6,
            "only_b": 57,
            "both_wrong": 41
          },
          "heterofold__vs__mcq_retrained": {
            "both_correct": 69,
            "only_a": 4,
            "only_b": 25,
            "both_wrong": 30
          },
          "heterofold__vs__mcq_writer": {
            "both_correct": 23,
            "only_a": 50,
            "only_b": 3,
            "both_wrong": 52
          },
          "heterofold__vs__native": {
            "both_correct": 70,
            "only_a": 3,
            "only_b": 41,
            "both_wrong": 14
          },
          "heterofold__vs__recolor": {
            "both_correct": 66,
            "only_a": 7,
            "only_b": 15,
            "both_wrong": 40
          },
          "mcq_retrained__vs__mcq_writer": {
            "both_correct": 22,
            "only_a": 72,
            "only_b": 4,
            "both_wrong": 30
          },
          "mcq_retrained__vs__native": {
            "both_correct": 86,
            "only_a": 8,
            "only_b": 25,
            "both_wrong": 9
          },
          "mcq_retrained__vs__recolor": {
            "both_correct": 77,
            "only_a": 17,
            "only_b": 4,
            "both_wrong": 30
          },
          "mcq_writer__vs__native": {
            "both_correct": 23,
            "only_a": 3,
            "only_b": 88,
            "both_wrong": 14
          },
          "mcq_writer__vs__recolor": {
            "both_correct": 21,
            "only_a": 5,
            "only_b": 60,
            "both_wrong": 42
          },
          "native__vs__recolor": {
            "both_correct": 77,
            "only_a": 34,
            "only_b": 4,
            "both_wrong": 13
          }
        }
      },
      "hellaswag": {
        "metrics": {
          "gsm8k_writer": {
            "n": 64,
            "correct": 22,
            "accuracy": 0.34375
          },
          "heterofold": {
            "n": 64,
            "correct": 26,
            "accuracy": 0.40625
          },
          "mcq_retrained": {
            "n": 64,
            "correct": 21,
            "accuracy": 0.328125
          },
          "mcq_writer": {
            "n": 64,
            "correct": 24,
            "accuracy": 0.375
          },
          "native": {
            "n": 64,
            "correct": 23,
            "accuracy": 0.359375
          }
        },
        "paired": {
          "gsm8k_writer__vs__heterofold": {
            "both_correct": 17,
            "only_a": 5,
            "only_b": 9,
            "both_wrong": 33
          },
          "gsm8k_writer__vs__mcq_retrained": {
            "both_correct": 14,
            "only_a": 8,
            "only_b": 7,
            "both_wrong": 35
          },
          "gsm8k_writer__vs__mcq_writer": {
            "both_correct": 18,
            "only_a": 4,
            "only_b": 6,
            "both_wrong": 36
          },
          "gsm8k_writer__vs__native": {
            "both_correct": 15,
            "only_a": 7,
            "only_b": 8,
            "both_wrong": 34
          },
          "heterofold__vs__mcq_retrained": {
            "both_correct": 13,
            "only_a": 13,
            "only_b": 8,
            "both_wrong": 30
          },
          "heterofold__vs__mcq_writer": {
            "both_correct": 16,
            "only_a": 10,
            "only_b": 8,
            "both_wrong": 30
          },
          "heterofold__vs__native": {
            "both_correct": 19,
            "only_a": 7,
            "only_b": 4,
            "both_wrong": 34
          },
          "mcq_retrained__vs__mcq_writer": {
            "both_correct": 17,
            "only_a": 4,
            "only_b": 7,
            "both_wrong": 36
          },
          "mcq_retrained__vs__native": {
            "both_correct": 12,
            "only_a": 9,
            "only_b": 11,
            "both_wrong": 32
          },
          "mcq_writer__vs__native": {
            "both_correct": 13,
            "only_a": 11,
            "only_b": 10,
            "both_wrong": 30
          }
        }
      }
    },
    "status": "completed",
    "training_manifest_sha256": "b5c8ddb4500933320e8f2262fa51215bcd6d60837e1cbccc59910154b7e1c2d6",
    "selections": {
      "a": {
        "selection": "lowest held-out validation reconstruction loss for Stage-A; lowest held-out validation choice KL for Stage-B",
        "best": {
          "epoch": 2,
          "step": 512,
          "validation": {
            "total": 0.48218073847215237,
            "k_nmse": 0.03598784277289724,
            "v_nmse": 0.25432120699861116,
            "k_cosine": 0.9604574938966834,
            "v_cosine": 0.8476708216003317
          }
        },
        "epochs": [
          {
            "epoch": 1,
            "step": 256,
            "validation": {
              "total": 0.5599581444737396,
              "k_nmse": 0.0432025756004748,
              "v_nmse": 0.2968428789123436,
              "k_cosine": 0.9536438117793296,
              "v_cosine": 0.8264435011988098
            }
          },
          {
            "epoch": 2,
            "step": 512,
            "validation": {
              "total": 0.48218073847215237,
              "k_nmse": 0.03598784277289724,
              "v_nmse": 0.25432120699861116,
              "k_cosine": 0.9604574938966834,
              "v_cosine": 0.8476708216003317
            }
          }
        ]
      },
      "b": {
        "selection": "lowest held-out validation reconstruction loss for Stage-A; lowest held-out validation choice KL for Stage-B",
        "best": {
          "epoch": 1,
          "step": 256,
          "validation": {
            "total": 0.5738657102564275,
            "choice_kl": 0.5738657102564275,
            "native_correct": 0.8359375,
            "writer_correct": 0.7109375
          }
        },
        "epochs": [
          {
            "epoch": 1,
            "step": 256,
            "validation": {
              "total": 0.5738657102564275,
              "choice_kl": 0.5738657102564275,
              "native_correct": 0.8359375,
              "writer_correct": 0.7109375
            }
          },
          {
            "epoch": 2,
            "step": 512,
            "validation": {
              "total": 0.592348562197003,
              "choice_kl": 0.592348562197003,
              "native_correct": 0.8359375,
              "writer_correct": 0.7109375
            }
          },
          {
            "epoch": 3,
            "step": 768,
            "validation": {
              "total": 0.5947345329179825,
              "choice_kl": 0.5947345329179825,
              "native_correct": 0.8359375,
              "writer_correct": 0.71875
            }
          },
          {
            "epoch": 4,
            "step": 1024,
            "validation": {
              "total": 0.5798809761916459,
              "choice_kl": 0.5798809761916459,
              "native_correct": 0.8359375,
              "writer_correct": 0.74609375
            }
          }
        ]
      }
    },
    "scope": "New MCQ Writer trained and evaluated under fixed HeteroFold chat/task protocol; no GSM8K training",
    "caveats": [
      "Previous methods reused on same rows; only mcq_retrained is new",
      "HeteroFold training data and Sender interface differ; not a same-training-budget architecture comparison"
    ]
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/logs/pipeline.log"
}
```


## mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/stage_a_only_hellaswag64

```json
{
  "time": "2026-10-20 13:04:56",
  "config": {
    "name": "mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
    "seed": 1234,
    "legacy_root": "/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234",
    "hf_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825",
    "previous_root": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/evaluation_crossdataset_arc128_hellaswag64_seed1234/mcq_writer",
    "models": {
      "llama": "/hy-tmp/yezhe/all_models/models/LLM-Research/Llama-3.2-3B-Instruct",
      "qwen": "/hy-tmp/yezhe/all_models/models/Qwen/Qwen3-4B"
    },
    "train_counts": {
      "openbookqa": 1024,
      "arc_challenge": 1024
    },
    "validation_counts": {
      "openbookqa": 128,
      "arc_challenge": 128
    },
    "stage_a_epochs": 2,
    "stage_b_epochs": 4,
    "effective_batch": 8,
    "stage_a_learning_rate": 0.001,
    "stage_b_learning_rate": 0.0001,
    "gradient_clip": 30,
    "weight_decay": 0,
    "adapter_rank": 64,
    "mlp_hidden_dim": 1024,
    "chunk_tokens": 64,
    "checkpoint_every_steps": 128,
    "llama_template_date": "19 Oct 2026",
    "dtype": "float16",
    "attention": "sdpa",
    "native_cache_choice_kl_max": 0.001,
    "min_free_gpu_gib": {
      "llama": 8,
      "qwen": 9.5,
      "stage_a": 5,
      "stage_b": 10.5
    },
    "disk_reserve_gib": 20,
    "reference_hashes": {
      "evaluation/short_context.py": "5528df3b3a8f54697ab2fb4c8d5748e3e6050eb0155a76cfaa0ef4bb3719d383",
      "evaluation/runtime.py": "baed2e7fa47c1acc00c43d0c7baaea225beaa3e5f4b6b4e0601294103b84d8a1",
      "modeling.py": "c36066d40ae9836cf31f324603af9fd1a1af3c70af2d61c0e492456a367749e8"
    },
    "losses": {
      "stage_a": "K NMSE + V NMSE + (1-K cosine) + (1-V cosine), per example; translated slots only",
      "stage_b": "KL(native Qwen candidate distribution || transferred candidate distribution), temperature1; no gold CE"
    },
    "checkpoint_selection": "lowest held-out validation reconstruction loss for Stage-A; lowest held-out validation choice KL for Stage-B",
    "evaluation": {
      "arc": 128,
      "hellaswag": 64
    },
    "training_gsm8k": false,
    "continuous_monitoring": false,
    "automatic_upload": false
  },
  "experiment": "mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/stage_a_only_hellaswag64",
  "tasks": {
    "hellaswag": {
      "metrics": {
        "gsm8k_writer": {
          "n": 64,
          "correct": 22,
          "accuracy": 0.34375
        },
        "heterofold": {
          "n": 64,
          "correct": 26,
          "accuracy": 0.40625
        },
        "mcq_retrained": {
          "n": 64,
          "correct": 21,
          "accuracy": 0.328125
        },
        "mcq_retrained_stage_a_only": {
          "n": 64,
          "correct": 19,
          "accuracy": 0.296875
        },
        "mcq_writer": {
          "n": 64,
          "correct": 24,
          "accuracy": 0.375
        },
        "native": {
          "n": 64,
          "correct": 23,
          "accuracy": 0.359375
        }
      },
      "paired": {
        "gsm8k_writer__vs__heterofold": {
          "both_correct": 17,
          "only_a": 5,
          "only_b": 9,
          "both_wrong": 33
        },
        "gsm8k_writer__vs__mcq_retrained": {
          "both_correct": 14,
          "only_a": 8,
          "only_b": 7,
          "both_wrong": 35
        },
        "gsm8k_writer__vs__mcq_retrained_stage_a_only": {
          "both_correct": 15,
          "only_a": 7,
          "only_b": 4,
          "both_wrong": 38
        },
        "gsm8k_writer__vs__mcq_writer": {
          "both_correct": 18,
          "only_a": 4,
          "only_b": 6,
          "both_wrong": 36
        },
        "gsm8k_writer__vs__native": {
          "both_correct": 15,
          "only_a": 7,
          "only_b": 8,
          "both_wrong": 34
        },
        "heterofold__vs__mcq_retrained": {
          "both_correct": 13,
          "only_a": 13,
          "only_b": 8,
          "both_wrong": 30
        },
        "heterofold__vs__mcq_retrained_stage_a_only": {
          "both_correct": 12,
          "only_a": 14,
          "only_b": 7,
          "both_wrong": 31
        },
        "heterofold__vs__mcq_writer": {
          "both_correct": 16,
          "only_a": 10,
          "only_b": 8,
          "both_wrong": 30
        },
        "heterofold__vs__native": {
          "both_correct": 19,
          "only_a": 7,
          "only_b": 4,
          "both_wrong": 34
        },
        "mcq_retrained__vs__mcq_retrained_stage_a_only": {
          "both_correct": 15,
          "only_a": 6,
          "only_b": 4,
          "both_wrong": 39
        },
        "mcq_retrained__vs__mcq_writer": {
          "both_correct": 17,
          "only_a": 4,
          "only_b": 7,
          "both_wrong": 36
        },
        "mcq_retrained__vs__native": {
          "both_correct": 12,
          "only_a": 9,
          "only_b": 11,
          "both_wrong": 32
        },
        "mcq_retrained_stage_a_only__vs__mcq_writer": {
          "both_correct": 17,
          "only_a": 2,
          "only_b": 7,
          "both_wrong": 38
        },
        "mcq_retrained_stage_a_only__vs__native": {
          "both_correct": 11,
          "only_a": 8,
          "only_b": 12,
          "both_wrong": 33
        },
        "mcq_writer__vs__native": {
          "both_correct": 13,
          "only_a": 11,
          "only_b": 10,
          "both_wrong": 30
        }
      }
    }
  },
  "status": "completed",
  "manifest": {
    "experiment": "mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/stage_a_only_hellaswag64",
    "parent_manifest_sha256": "b5c8ddb4500933320e8f2262fa51215bcd6d60837e1cbccc59910154b7e1c2d6",
    "script_sha256": "2f12303a6e1a189f585d6cb0d1af14b640b5b91ee633c44c8b2002d43611885f",
    "stage_a_sha256": "db1f295c502f27305a9d2ac410c47a1bdef4db7e0255f817483867b19d78cf0d",
    "stage_a_selection": {
      "epoch": 2,
      "step": 512,
      "validation": {
        "total": 0.48218073847215237,
        "k_nmse": 0.03598784277289724,
        "v_nmse": 0.25432120699861116,
        "k_cosine": 0.9604574938966834,
        "v_cosine": 0.8476708216003317
      }
    },
    "previous_results_sha256": "6aca0634e6ee65c66e871827e2b08f816ef48b7d3fa167e2500026fe2bae312f",
    "prepared_sha256": "9c51c0b4a0d8e9b878d35009f6e33d09955e72d2f8d284efb97e869ff0cdc123",
    "conditions": "Only new MCQ Stage-A-only scored; existing Native, Stage-A+B, HeteroFold and historical writers reused",
    "protocol": "Unchanged pinned HeteroFold HellaSwag chat prompt, Full-Sync mapping, native anchor/tail, token-mean continuation likelihood",
    "stage_b_loaded": false,
    "training": false,
    "indices": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50,
      51,
      52,
      53,
      54,
      55,
      56,
      57,
      58,
      59,
      60,
      61,
      62,
      63
    ],
    "seed": 1234
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/stage_a_only_hellaswag64/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/stage_a_only_hellaswag64/logs/pipeline.log"
}
```


## mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/attention_output_arc128_hellaswag64

```json
{
  "experiment": "mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/attention_output_arc128_hellaswag64",
  "time": "2026-10-20 13:15:43",
  "status": "completed",
  "config": {
    "experiment": "mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/attention_output_arc128_hellaswag64",
    "seed": 1234,
    "parent_manifest_sha256": "b5c8ddb4500933320e8f2262fa51215bcd6d60837e1cbccc59910154b7e1c2d6",
    "script_sha256": "6820e955ffbf8b5a0bbf72c818b141be27937472bee301f07a6ea24a0efd29f2",
    "parent_evaluation": {
      "manifest_sha256": "b5c8ddb4500933320e8f2262fa51215bcd6d60837e1cbccc59910154b7e1c2d6",
      "stage_a_sha256": "db1f295c502f27305a9d2ac410c47a1bdef4db7e0255f817483867b19d78cf0d",
      "stage_b_sha256": "06190cfcda6ce65e7b81447d2fa77b926bd43b384e2bfb5866083cc1588e278f",
      "prepared_sha256": "109ecef804b72a9431d6075ef4fe34579dd6336c98e97434497553720abfbd06"
    },
    "previous_results_sha256": "6aca0634e6ee65c66e871827e2b08f816ef48b7d3fa167e2500026fe2bae312f",
    "samples": {
      "arc": 128,
      "hellaswag": 64
    },
    "indices": {
      "arc": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127
      ],
      "hellaswag": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63
      ]
    },
    "dtype": "float16; diagnostic attention arithmetic float32",
    "backend": "unchanged sdpa",
    "conditions": [
      "native_kv",
      "stage_a_kv",
      "stage_a_b_kv"
    ],
    "query_heads": 32,
    "kv_heads": 8,
    "head_dim": 128,
    "layers": 36,
    "teacher_forcing": "All four (or official count) candidates, same known candidate tokens in every condition; never free generation",
    "aggregation": "Average candidates within each example, then examples within dataset; do not pool tokens across examples",
    "controlled": "Fix Native query and Native suffix K/V at last native readout token; replace only prefix memory K/V",
    "closed_loop": "Actual Receiver forward with each cache; compare pre-o_proj per query head and post-o_proj output at scored positions",
    "caveat": "Closed-loop output differences include propagated hidden/query changes; fixed-query intervention is a local counterfactual, not an actual replacement trajectory",
    "training": false
  },
  "datasets": {
    "arc": {
      "n": 128,
      "accuracy": {
        "native": {
          "correct": 111,
          "n": 128,
          "accuracy": 0.8671875
        },
        "stage_a": {
          "correct": 86,
          "n": 128,
          "accuracy": 0.671875
        },
        "stage_ab": {
          "correct": 94,
          "n": 128,
          "accuracy": 0.734375
        }
      },
      "stage_a_vs_stage_ab": {
        "both_correct": 82,
        "stage_a_only": 4,
        "stage_ab_only": 12,
        "both_wrong": 30
      },
      "global_metrics": {
        "closed_loop_readout": {
          "nmse_stage_a": 2.7691770577553982,
          "nmse_stage_ab": 2.9813191629090716,
          "nmse_reduction": -0.2121421031827121,
          "cosine_stage_a": 0.7447176478677029,
          "cosine_stage_ab": 0.7079045656370035,
          "correction_direction_cosine": 0.1203528326937527,
          "correction_to_stage_a_norm": 0.42747848245151093
        },
        "closed_loop_scored_positions": {
          "nmse_stage_a": 2.7691770577553982,
          "nmse_stage_ab": 2.9813191629090716,
          "nmse_reduction": -0.2121421031827121,
          "cosine_stage_a": 0.7447176478677029,
          "cosine_stage_ab": 0.7079045656370035,
          "correction_direction_cosine": 0.1203528326937527,
          "correction_to_stage_a_norm": 0.42747848245151093
        },
        "closed_loop_projected_output": {
          "nmse_stage_a": 0.776096534296875,
          "nmse_stage_ab": 0.8624330419726562,
          "nmse_reduction": -0.0863365079904514,
          "cosine_stage_a": 0.6777006768381075,
          "cosine_stage_ab": 0.6368555996050347,
          "correction_direction_cosine": 0.16822149026909722,
          "correction_to_stage_a_norm": 0.4738195263606771
        },
        "fixed_native_query_readout": {
          "nmse_stage_a": 2.6145650772904463,
          "nmse_stage_ab": 2.6358435363944497,
          "nmse_reduction": -0.02127845676927355,
          "cosine_stage_a": 0.9468369868427193,
          "cosine_stage_ab": 0.9440155012738715,
          "correction_direction_cosine": -0.01781029918965658,
          "correction_to_stage_a_norm": 0.06315648983378092
        },
        "memory_key": {
          "nmse_stage_a": 0.09015669540581596,
          "nmse_stage_ab": 0.09088370674207899,
          "nmse_reduction": -0.0007270113650173612,
          "cosine_stage_a": 0.9536200786857096,
          "cosine_stage_ab": 0.9532555703062608,
          "correction_direction_cosine": -0.0018732629003906248,
          "correction_to_stage_a_norm": 0.025053247750651042
        },
        "memory_value": {
          "nmse_stage_a": 0.40558556512342664,
          "nmse_stage_ab": 0.4059822608230252,
          "nmse_reduction": -0.00039669567762586804,
          "cosine_stage_a": 0.8229100092944336,
          "cosine_stage_ab": 0.8226788440874565,
          "correction_direction_cosine": 0.0003569949842664931,
          "correction_to_stage_a_norm": 0.022042101148817276
        }
      },
      "mean_choice_kl": {
        "stage_a": 1.819393790157653,
        "stage_ab": 0.6764343478365618
      },
      "max_native_attention_reconstruction_nmse": 5.559813942568326e-08,
      "max_stage_ab_score_error": 0.0,
      "top_changed_heads": {
        "fixed_native_query_readout": {
          "best": [
            {
              "layer": 4,
              "query_head": 1,
              "kv_group": 0,
              "nmse_reduction": 8.12714806421875
            },
            {
              "layer": 0,
              "query_head": 20,
              "kv_group": 5,
              "nmse_reduction": 6.721032333359375
            },
            {
              "layer": 0,
              "query_head": 21,
              "kv_group": 5,
              "nmse_reduction": 4.26824544
            },
            {
              "layer": 2,
              "query_head": 8,
              "kv_group": 2,
              "nmse_reduction": 3.2829614532031255
            },
            {
              "layer": 0,
              "query_head": 23,
              "kv_group": 5,
              "nmse_reduction": 2.9080012189062496
            },
            {
              "layer": 0,
              "query_head": 22,
              "kv_group": 5,
              "nmse_reduction": 2.752026117265625
            },
            {
              "layer": 4,
              "query_head": 12,
              "kv_group": 3,
              "nmse_reduction": 1.9690410757031251
            },
            {
              "layer": 4,
              "query_head": 2,
              "kv_group": 0,
              "nmse_reduction": 1.902892831171875
            },
            {
              "layer": 4,
              "query_head": 15,
              "kv_group": 3,
              "nmse_reduction": 1.865210095234375
            },
            {
              "layer": 4,
              "query_head": 13,
              "kv_group": 3,
              "nmse_reduction": 1.812336818203125
            },
            {
              "layer": 1,
              "query_head": 28,
              "kv_group": 7,
              "nmse_reduction": 1.619851884375
            },
            {
              "layer": 0,
              "query_head": 6,
              "kv_group": 1,
              "nmse_reduction": 1.597664275078125
            }
          ],
          "worst": [
            {
              "layer": 6,
              "query_head": 6,
              "kv_group": 1,
              "nmse_reduction": -1.4883619229687501
            },
            {
              "layer": 0,
              "query_head": 28,
              "kv_group": 7,
              "nmse_reduction": -1.5578945036718748
            },
            {
              "layer": 3,
              "query_head": 28,
              "kv_group": 7,
              "nmse_reduction": -1.6743120489843748
            },
            {
              "layer": 2,
              "query_head": 27,
              "kv_group": 6,
              "nmse_reduction": -1.7065163665625
            },
            {
              "layer": 1,
              "query_head": 2,
              "kv_group": 0,
              "nmse_reduction": -2.0067607417187503
            },
            {
              "layer": 1,
              "query_head": 9,
              "kv_group": 2,
              "nmse_reduction": -2.2708339722656254
            },
            {
              "layer": 3,
              "query_head": 14,
              "kv_group": 3,
              "nmse_reduction": -2.4174320780468754
            },
            {
              "layer": 2,
              "query_head": 23,
              "kv_group": 5,
              "nmse_reduction": -3.6213975824218747
            },
            {
              "layer": 2,
              "query_head": 17,
              "kv_group": 4,
              "nmse_reduction": -4.98832053875
            },
            {
              "layer": 2,
              "query_head": 21,
              "kv_group": 5,
              "nmse_reduction": -6.873658557968749
            },
            {
              "layer": 3,
              "query_head": 26,
              "kv_group": 6,
              "nmse_reduction": -11.60840630171875
            },
            {
              "layer": 0,
              "query_head": 26,
              "kv_group": 6,
              "nmse_reduction": -14.96633401203125
            }
          ]
        },
        "closed_loop_scored_positions": {
          "best": [
            {
              "layer": 5,
              "query_head": 27,
              "kv_group": 6,
              "nmse_reduction": 19.8808179865625
            },
            {
              "layer": 30,
              "query_head": 27,
              "kv_group": 6,
              "nmse_reduction": 11.245499808828123
            },
            {
              "layer": 5,
              "query_head": 21,
              "kv_group": 5,
              "nmse_reduction": 7.0911893308593745
            },
            {
              "layer": 2,
              "query_head": 1,
              "kv_group": 0,
              "nmse_reduction": 7.032117926484375
            },
            {
              "layer": 0,
              "query_head": 20,
              "kv_group": 5,
              "nmse_reduction": 6.720784024921874
            },
            {
              "layer": 9,
              "query_head": 9,
              "kv_group": 2,
              "nmse_reduction": 6.287641902578125
            },
            {
              "layer": 4,
              "query_head": 18,
              "kv_group": 4,
              "nmse_reduction": 5.912078359140624
            },
            {
              "layer": 31,
              "query_head": 2,
              "kv_group": 0,
              "nmse_reduction": 5.8146536860937506
            },
            {
              "layer": 5,
              "query_head": 24,
              "kv_group": 6,
              "nmse_reduction": 5.731026066484374
            },
            {
              "layer": 26,
              "query_head": 5,
              "kv_group": 1,
              "nmse_reduction": 5.634534536406249
            },
            {
              "layer": 4,
              "query_head": 13,
              "kv_group": 3,
              "nmse_reduction": 5.51344408921875
            },
            {
              "layer": 19,
              "query_head": 26,
              "kv_group": 6,
              "nmse_reduction": 5.223931444218749
            }
          ],
          "worst": [
            {
              "layer": 12,
              "query_head": 29,
              "kv_group": 7,
              "nmse_reduction": -5.360386159687499
            },
            {
              "layer": 2,
              "query_head": 21,
              "kv_group": 5,
              "nmse_reduction": -5.96382412234375
            },
            {
              "layer": 17,
              "query_head": 15,
              "kv_group": 3,
              "nmse_reduction": -6.0533986749999995
            },
            {
              "layer": 2,
              "query_head": 23,
              "kv_group": 5,
              "nmse_reduction": -6.596310280625
            },
            {
              "layer": 34,
              "query_head": 0,
              "kv_group": 0,
              "nmse_reduction": -6.886723087031251
            },
            {
              "layer": 19,
              "query_head": 17,
              "kv_group": 4,
              "nmse_reduction": -8.57341274859375
            },
            {
              "layer": 2,
              "query_head": 2,
              "kv_group": 0,
              "nmse_reduction": -8.675111347890626
            },
            {
              "layer": 0,
              "query_head": 26,
              "kv_group": 6,
              "nmse_reduction": -14.966381542343749
            },
            {
              "layer": 34,
              "query_head": 1,
              "kv_group": 0,
              "nmse_reduction": -17.711855034218747
            },
            {
              "layer": 35,
              "query_head": 6,
              "kv_group": 1,
              "nmse_reduction": -30.3807821703125
            },
            {
              "layer": 16,
              "query_head": 23,
              "kv_group": 5,
              "nmse_reduction": -47.536160260624996
            },
            {
              "layer": 4,
              "query_head": 1,
              "kv_group": 0,
              "nmse_reduction": -95.95717868195311
            }
          ]
        }
      }
    },
    "hellaswag": {
      "n": 64,
      "accuracy": {
        "native": {
          "correct": 23,
          "n": 64,
          "accuracy": 0.359375
        },
        "stage_a": {
          "correct": 19,
          "n": 64,
          "accuracy": 0.296875
        },
        "stage_ab": {
          "correct": 21,
          "n": 64,
          "accuracy": 0.328125
        }
      },
      "stage_a_vs_stage_ab": {
        "both_correct": 15,
        "stage_a_only": 4,
        "stage_ab_only": 6,
        "both_wrong": 39
      },
      "global_metrics": {
        "closed_loop_readout": {
          "nmse_stage_a": 8.544958780754937,
          "nmse_stage_ab": 7.727475548100042,
          "nmse_reduction": 0.8174832289196777,
          "cosine_stage_a": 0.5282610334510633,
          "cosine_stage_ab": 0.5193311775876194,
          "correction_direction_cosine": 0.06234386660766601,
          "correction_to_stage_a_norm": 0.2466932677362739
        },
        "closed_loop_scored_positions": {
          "nmse_stage_a": 2.704274012642008,
          "nmse_stage_ab": 2.212879037926839,
          "nmse_reduction": 0.4913949732847765,
          "cosine_stage_a": 0.5217715115291341,
          "cosine_stage_ab": 0.565699215802002,
          "correction_direction_cosine": 0.2552953845350478,
          "correction_to_stage_a_norm": 0.3659636718181695
        },
        "closed_loop_projected_output": {
          "nmse_stage_a": 1.3091112317881943,
          "nmse_stage_ab": 1.1446712923480904,
          "nmse_reduction": 0.16443994011718752,
          "cosine_stage_a": 0.44006210453993055,
          "cosine_stage_ab": 0.4858696905598958,
          "correction_direction_cosine": 0.2718028274001736,
          "correction_to_stage_a_norm": 0.38509794922309026
        },
        "fixed_native_query_readout": {
          "nmse_stage_a": 3.092507868471409,
          "nmse_stage_ab": 3.120520339731581,
          "nmse_reduction": -0.028012470001898868,
          "cosine_stage_a": 0.902142127639567,
          "cosine_stage_ab": 0.9011787139520943,
          "correction_direction_cosine": -0.010802972430826823,
          "correction_to_stage_a_norm": 0.036305872081705735
        },
        "memory_key": {
          "nmse_stage_a": 0.14148582210666233,
          "nmse_stage_ab": 0.14218224793891063,
          "nmse_reduction": -0.0006964259168836805,
          "cosine_stage_a": 0.9266138227110461,
          "cosine_stage_ab": 0.9262528936724175,
          "correction_direction_cosine": -0.0018088228814019099,
          "correction_to_stage_a_norm": 0.024835863296440972
        },
        "memory_value": {
          "nmse_stage_a": 0.6430489834499783,
          "nmse_stage_ab": 0.6432142476215277,
          "nmse_reduction": -0.00016526429524739588,
          "cosine_stage_a": 0.6739292130072699,
          "cosine_stage_ab": 0.6737762286349827,
          "correction_direction_cosine": 0.0009574099815538198,
          "correction_to_stage_a_norm": 0.018027125466579863
        }
      },
      "mean_choice_kl": {
        "stage_a": 0.4918299695418682,
        "stage_ab": 0.4228156559402123
      },
      "max_native_attention_reconstruction_nmse": 6.050687062497673e-08,
      "max_stage_ab_score_error": 0.0,
      "top_changed_heads": {
        "fixed_native_query_readout": {
          "best": [
            {
              "layer": 4,
              "query_head": 1,
              "kv_group": 0,
              "nmse_reduction": 17.993341892500002
            },
            {
              "layer": 1,
              "query_head": 28,
              "kv_group": 7,
              "nmse_reduction": 4.0579521175
            },
            {
              "layer": 4,
              "query_head": 2,
              "kv_group": 0,
              "nmse_reduction": 3.45786775640625
            },
            {
              "layer": 4,
              "query_head": 0,
              "kv_group": 0,
              "nmse_reduction": 2.433347495625
            },
            {
              "layer": 4,
              "query_head": 11,
              "kv_group": 2,
              "nmse_reduction": 2.3949784801562495
            },
            {
              "layer": 6,
              "query_head": 22,
              "kv_group": 5,
              "nmse_reduction": 2.037110019375
            },
            {
              "layer": 4,
              "query_head": 15,
              "kv_group": 3,
              "nmse_reduction": 1.99482912703125
            },
            {
              "layer": 4,
              "query_head": 17,
              "kv_group": 4,
              "nmse_reduction": 1.82260294359375
            },
            {
              "layer": 4,
              "query_head": 14,
              "kv_group": 3,
              "nmse_reduction": 1.80398582640625
            },
            {
              "layer": 4,
              "query_head": 10,
              "kv_group": 2,
              "nmse_reduction": 1.6694226229687499
            },
            {
              "layer": 2,
              "query_head": 10,
              "kv_group": 2,
              "nmse_reduction": 1.5688975096875002
            },
            {
              "layer": 4,
              "query_head": 9,
              "kv_group": 2,
              "nmse_reduction": 1.5222620109375
            }
          ],
          "worst": [
            {
              "layer": 0,
              "query_head": 30,
              "kv_group": 7,
              "nmse_reduction": -2.8608902
            },
            {
              "layer": 0,
              "query_head": 26,
              "kv_group": 6,
              "nmse_reduction": -3.27507746046875
            },
            {
              "layer": 3,
              "query_head": 26,
              "kv_group": 6,
              "nmse_reduction": -3.487903863125
            },
            {
              "layer": 3,
              "query_head": 13,
              "kv_group": 3,
              "nmse_reduction": -3.5340647398437497
            },
            {
              "layer": 5,
              "query_head": 11,
              "kv_group": 2,
              "nmse_reduction": -3.9092908967187503
            },
            {
              "layer": 2,
              "query_head": 20,
              "kv_group": 5,
              "nmse_reduction": -4.8802475746875
            },
            {
              "layer": 3,
              "query_head": 24,
              "kv_group": 6,
              "nmse_reduction": -4.9344835365624995
            },
            {
              "layer": 3,
              "query_head": 25,
              "kv_group": 6,
              "nmse_reduction": -7.0014730545312505
            },
            {
              "layer": 2,
              "query_head": 21,
              "kv_group": 5,
              "nmse_reduction": -7.95329742953125
            },
            {
              "layer": 2,
              "query_head": 23,
              "kv_group": 5,
              "nmse_reduction": -8.04431669421875
            },
            {
              "layer": 2,
              "query_head": 22,
              "kv_group": 5,
              "nmse_reduction": -8.176828190625
            },
            {
              "layer": 5,
              "query_head": 10,
              "kv_group": 2,
              "nmse_reduction": -9.162673822968749
            }
          ]
        },
        "closed_loop_scored_positions": {
          "best": [
            {
              "layer": 29,
              "query_head": 8,
              "kv_group": 2,
              "nmse_reduction": 39.81512524375
            },
            {
              "layer": 9,
              "query_head": 28,
              "kv_group": 7,
              "nmse_reduction": 27.82407575109375
            },
            {
              "layer": 9,
              "query_head": 30,
              "kv_group": 7,
              "nmse_reduction": 22.060571451093754
            },
            {
              "layer": 30,
              "query_head": 8,
              "kv_group": 2,
              "nmse_reduction": 20.2154910859375
            },
            {
              "layer": 29,
              "query_head": 10,
              "kv_group": 2,
              "nmse_reduction": 15.2634012509375
            },
            {
              "layer": 30,
              "query_head": 11,
              "kv_group": 2,
              "nmse_reduction": 13.37376078484375
            },
            {
              "layer": 30,
              "query_head": 27,
              "kv_group": 6,
              "nmse_reduction": 12.531074184062499
            },
            {
              "layer": 9,
              "query_head": 31,
              "kv_group": 7,
              "nmse_reduction": 11.381568175624999
            },
            {
              "layer": 9,
              "query_head": 8,
              "kv_group": 2,
              "nmse_reduction": 11.19374361421875
            },
            {
              "layer": 16,
              "query_head": 20,
              "kv_group": 5,
              "nmse_reduction": 9.30441041375
            },
            {
              "layer": 7,
              "query_head": 7,
              "kv_group": 1,
              "nmse_reduction": 8.32259932859375
            },
            {
              "layer": 7,
              "query_head": 6,
              "kv_group": 1,
              "nmse_reduction": 8.24520076296875
            }
          ],
          "worst": [
            {
              "layer": 2,
              "query_head": 21,
              "kv_group": 5,
              "nmse_reduction": -1.04926215265625
            },
            {
              "layer": 26,
              "query_head": 4,
              "kv_group": 1,
              "nmse_reduction": -1.216363753125
            },
            {
              "layer": 1,
              "query_head": 14,
              "kv_group": 3,
              "nmse_reduction": -1.29382248078125
            },
            {
              "layer": 34,
              "query_head": 1,
              "kv_group": 0,
              "nmse_reduction": -1.4348949340624997
            },
            {
              "layer": 1,
              "query_head": 13,
              "kv_group": 3,
              "nmse_reduction": -1.6924058670312498
            },
            {
              "layer": 34,
              "query_head": 4,
              "kv_group": 1,
              "nmse_reduction": -1.92337489140625
            },
            {
              "layer": 33,
              "query_head": 1,
              "kv_group": 0,
              "nmse_reduction": -2.0086748296875
            },
            {
              "layer": 32,
              "query_head": 25,
              "kv_group": 6,
              "nmse_reduction": -2.14340010484375
            },
            {
              "layer": 4,
              "query_head": 2,
              "kv_group": 0,
              "nmse_reduction": -2.51700797671875
            },
            {
              "layer": 34,
              "query_head": 25,
              "kv_group": 6,
              "nmse_reduction": -4.08792619609375
            },
            {
              "layer": 31,
              "query_head": 2,
              "kv_group": 0,
              "nmse_reduction": -5.58788557328125
            },
            {
              "layer": 4,
              "query_head": 1,
              "kv_group": 0,
              "nmse_reduction": -11.972898617343748
            }
          ]
        }
      }
    }
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/attention_output_arc128_hellaswag64/formal/per_sample.jsonl",
  "full_summary": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/attention_output_arc128_hellaswag64/formal/summary.json",
  "log": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/attention_output_arc128_hellaswag64/logs/pipeline.log"
}
```


## heterofold_official_l3_to_q4_seed20260825/attention_output_arc128_hellaswag64_seed1234

```json
{
  "experiment": "heterofold_official_l3_to_q4_seed20260825/attention_output_arc128_hellaswag64_seed1234",
  "time": "2026-10-20 13:34:05",
  "status": "completed",
  "configuration": {
    "experiment": "heterofold_official_l3_to_q4_seed20260825/attention_output_arc128_hellaswag64_seed1234",
    "seed": 1234,
    "training": false,
    "map_path": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/maps/llama3_to_qwen4/heterofold.pt",
    "map_sha256": "dc2ea778b2b6ea4d6c74c678548946f59fb1e5970f935be3ecc5e4fb58ac718e",
    "checkpoint_selection": "Existing frozen folded HeteroFold map; no selection on these tests",
    "models": {
      "llama": "/hy-tmp/yezhe/all_models/models/LLM-Research/Llama-3.2-3B-Instruct",
      "qwen": "/hy-tmp/yezhe/all_models/models/Qwen/Qwen3-4B"
    },
    "dependencies": {
      "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/core.py": "27417f5480e067af53c9e1871a0e7dc22f11f27ddc55dfedfaf9747380edcbfc",
      "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/evaluate.py": "bf82d696ecf01e181074f8cee84e12b5819184c647b7aad916da20f565c34e48",
      "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234/config.json": "2fc399bd6d94b805aafc4ece8f9d39e65421b0fecb9497f3adec11c3eb25acd1",
      "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/attention_output_arc128_hellaswag64_seed1234/run.py": "637b9c438f0153bf04ca508efb6437854ed9c64ebcfe83398be1899769608d7c",
      "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/heterofold/transfer.py": "863bc71c2c06d9209b36f35208285de88aabd989703eedaa55f20ac2d3db5971",
      "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/heterofold/capture.py": "9e7dd3165c9d6b74d8e7a8d64fe5a4a844db44f87dc75962bae004916fae6b45",
      "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/heterofold/alignment.py": "d2009d9f7b95117190d3f1dad7abae42f58005d99101abe93127f7f65fb90d98",
      "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/heterofold/attention.py": "fd752812c10c085941266beaac34a7364e0dcb455b6aef0993cab0a1ab93fae7",
      "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/heterofold/maps.py": "bda861367721565827b4336a129ad5352ab1018f6181d3b2db50eb96e3914329"
    },
    "reference_hashes": {
      "evaluation/short_context.py": "5528df3b3a8f54697ab2fb4c8d5748e3e6050eb0155a76cfaa0ef4bb3719d383",
      "evaluation/runtime.py": "baed2e7fa47c1acc00c43d0c7baaea225beaa3e5f4b6b4e0601294103b84d8a1",
      "modeling.py": "c36066d40ae9836cf31f324603af9fd1a1af3c70af2d61c0e492456a367749e8"
    },
    "previous_results_sha256": "6aca0634e6ee65c66e871827e2b08f816ef48b7d3fa167e2500026fe2bae312f",
    "prior_hf_manifest_sha256": "348746e52f0f79553725adefb5f27986af57e5774bb5a6fb9e5bff1149e14e04",
    "samples": {
      "ARC-Challenge official test first indices": 128,
      "HellaSwag validation first indices": 64
    },
    "conditions": [
      "qwen_native",
      "heterofold"
    ],
    "dtype": "float16; diagnostic arithmetic float32",
    "backend": "sdpa",
    "sender": "Original capture_sender on raw task prompt, no Sender chat template",
    "receiver": "Original build_receiver_cache: Native chat header + translated FULL task prompt; Native assistant suffix + cue",
    "cache_domain": "Final post-k_norm, post-RoPE K and V consumed directly; never inverse/reapply RoPE to transferred cache",
    "metrics_domain": "Memory K comparison post-RoPE; attention output pre-o_proj at 32 query heads and post-o_proj full vector",
    "fixed_query": "Native query and Native suffix KV fixed; swap only final prefix memory (local counterfactual)",
    "closed_loop": "Actual Reader forwards; differences include downstream hidden-state and query changes",
    "scoring": "Reference candidate log-likelihood, ARC label sum; HellaSwag continuation token mean. All candidates teacher-forced.",
    "aggregation": "Mean positions within head; equal candidates within example, equal examples within task; no token pooling",
    "gates": {
      "manual_native_attention_nmse": 0.0001,
      "manual_native_attention_cosine": 0.999,
      "hf_reference_score_abs": 0.02,
      "native_reference_choice_kl": 0.001
    }
  },
  "results": {
    "arc": {
      "n": 128,
      "accuracy": {
        "native": {
          "correct": 111,
          "n": 128,
          "accuracy": 0.8671875
        },
        "heterofold": {
          "correct": 73,
          "n": 128,
          "accuracy": 0.5703125
        }
      },
      "paired": {
        "both_correct": 70,
        "native_only": 41,
        "heterofold_only": 3,
        "both_wrong": 14
      },
      "mean_choice_kl": 1.612569834733295,
      "global_metrics": {
        "memory_key_post_rope": {
          "nmse": 0.42461299591959634,
          "cosine": 0.7838024965296767,
          "translated_to_native_norm": 0.9734790190993924
        },
        "memory_value": {
          "nmse": 0.743600386127116,
          "cosine": 0.6043782318793404,
          "translated_to_native_norm": 0.9591009831952584
        },
        "closed_loop_readout": {
          "nmse": 1.6449492600439453,
          "cosine": 0.7172436152613662,
          "translated_to_native_norm": 1.3299929468373617
        },
        "closed_loop_scored_positions": {
          "nmse": 1.6449492600439453,
          "cosine": 0.7172436152613662,
          "translated_to_native_norm": 1.3299929468373617
        },
        "closed_loop_projected_output": {
          "nmse": 0.614915490234375,
          "cosine": 0.7003777228472222,
          "translated_to_native_norm": 0.9930192753342014
        },
        "fixed_native_query_readout": {
          "nmse": 0.9097973067171223,
          "cosine": 0.8066380811094835,
          "translated_to_native_norm": 1.1865797298768446
        }
      },
      "max_native_reconstruction_nmse": 5.572779926410476e-08,
      "max_hf_reference_score_error": 0.0,
      "max_native_reference_choice_kl": 0.00045318376858315276
    },
    "hellaswag": {
      "n": 64,
      "accuracy": {
        "native": {
          "correct": 23,
          "n": 64,
          "accuracy": 0.359375
        },
        "heterofold": {
          "correct": 26,
          "n": 64,
          "accuracy": 0.40625
        }
      },
      "paired": {
        "both_correct": 19,
        "native_only": 4,
        "heterofold_only": 7,
        "both_wrong": 34
      },
      "mean_choice_kl": 0.2845581119254348,
      "global_metrics": {
        "memory_key_post_rope": {
          "nmse": 0.41265857905002173,
          "cosine": 0.7890425260286459,
          "translated_to_native_norm": 0.978483391688368
        },
        "memory_value": {
          "nmse": 0.6987753144949003,
          "cosine": 0.6293091923616536,
          "translated_to_native_norm": 0.9695372022960069
        },
        "closed_loop_readout": {
          "nmse": 1.3520567597066244,
          "cosine": 0.7506522961950683,
          "translated_to_native_norm": 1.182023263673367
        },
        "closed_loop_scored_positions": {
          "nmse": 0.8879062009493002,
          "cosine": 0.7364873190855579,
          "translated_to_native_norm": 1.1202799150363498
        },
        "closed_loop_projected_output": {
          "nmse": 0.5301209009765624,
          "cosine": 0.7375302999522569,
          "translated_to_native_norm": 0.9891914494314236
        },
        "fixed_native_query_readout": {
          "nmse": 0.9634658875428603,
          "cosine": 0.822792344409044,
          "translated_to_native_norm": 1.18396444914917
        }
      },
      "max_native_reconstruction_nmse": 6.051611478596897e-08,
      "max_hf_reference_score_error": 0.0,
      "max_native_reference_choice_kl": 2.2230356556907083e-05
    }
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/attention_output_arc128_hellaswag64_seed1234/formal/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/attention_output_arc128_hellaswag64_seed1234/logs/pipeline.log"
}
```


## decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234

```json
{
  "experiment": "decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234",
  "time": "2026-10-20 14:34:55",
  "status": "completed",
  "config": {
    "name": "decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234",
    "seed": 1234,
    "parent": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
    "archive": "/hy-tmp/yezhe/异构模型/openbookqa_factkv_frozen64_seed1234/data/OpenBookQA-V1-Sep2018.zip",
    "prerequisite": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/attention_output_arc128_hellaswag64_seed1234/formal/completed.json",
    "old_reader": "/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/receiver_gated_reader_v1_seed1234/reader.py",
    "stage_a_sha256": "db1f295c502f27305a9d2ac410c47a1bdef4db7e0255f817483867b19d78cf0d",
    "stage_b_sha256": "06190cfcda6ce65e7b81447d2fa77b926bd43b384e2bfb5866083cc1588e278f",
    "counts": {
      "train": 1024,
      "validation": 128,
      "test": 128
    },
    "layers_zero_based": [
      7,
      15,
      23,
      31
    ],
    "rank": 32,
    "old_reader_rank": 32,
    "arms": [
      "native_query",
      "independent_query",
      "old_coupled_reader"
    ],
    "epochs": 4,
    "effective_batch": 8,
    "learning_rate": 0.0003,
    "gradient_clip": 5.0,
    "weight_decay": 0.0,
    "checkpoint_every_steps": 64,
    "loss": "Full-vocabulary cross-entropy of the single gold option-label token; no KL, attention/output loss or gate penalty",
    "selection": "Lowest official-development Gold CE, independently per arm; no selection on test, wrong-memory or attention diagnostics",
    "memory": "Frozen latest MCQ Stage-A+Stage-B; all fact-content tokens, post-k_norm pre-RoPE K; no compression",
    "receiver": "Question and options only, pinned HeteroFold ARC MCQ formatter and Qwen chat, thinking disabled, same label-scoring cue",
    "sender": "Own Llama chat containing fact1 only; no question, options, answerKey or answer cue",
    "scope": "Reader-only evidence diagnostic. Not a claim of shared multi-Sender compatibility, retrieval performance or closed-book benchmark parity.",
    "min_free_gpu_gib": 15,
    "disk_reserve_gib": 20,
    "continuous_monitoring": false,
    "automatic_upload": false
  },
  "results": {
    "status": "completed",
    "experiment": "decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234",
    "metrics": {
      "baselines/question_only": {
        "n": 128,
        "correct": 103,
        "accuracy": 0.8046875,
        "gold_ce": 19.163342878222466
      },
      "baselines/native_full_evidence_text": {
        "n": 128,
        "correct": 122,
        "accuracy": 0.953125,
        "gold_ce": 18.30207148939371
      },
      "baselines/frozen_stage_a_physical_prefix": {
        "n": 128,
        "correct": 105,
        "accuracy": 0.8203125,
        "gold_ce": 19.153694450855255
      },
      "baselines/frozen_stage_b_physical_prefix": {
        "n": 128,
        "correct": 107,
        "accuracy": 0.8359375,
        "gold_ce": 19.02811983972788
      },
      "native_query/correct_memory": {
        "n": 128,
        "correct": 106,
        "accuracy": 0.828125,
        "gold_ce": 0.41914943133499794
      },
      "native_query/no_memory": {
        "n": 128,
        "correct": 103,
        "accuracy": 0.8046875,
        "gold_ce": 19.163342878222466
      },
      "native_query/wrong_memory": {
        "n": 128,
        "correct": 106,
        "accuracy": 0.828125,
        "gold_ce": 0.42875786239953584
      },
      "independent_query/correct_memory": {
        "n": 128,
        "correct": 105,
        "accuracy": 0.8203125,
        "gold_ce": 0.436045697515822
      },
      "independent_query/no_memory": {
        "n": 128,
        "correct": 103,
        "accuracy": 0.8046875,
        "gold_ce": 19.163342878222466
      },
      "independent_query/wrong_memory": {
        "n": 128,
        "correct": 105,
        "accuracy": 0.8203125,
        "gold_ce": 0.43290166169767375
      },
      "old_coupled_reader/correct_memory": {
        "n": 128,
        "correct": 107,
        "accuracy": 0.8359375,
        "gold_ce": 0.5118580277825799
      },
      "old_coupled_reader/no_memory": {
        "n": 128,
        "correct": 103,
        "accuracy": 0.8046875,
        "gold_ce": 19.163342878222466
      },
      "old_coupled_reader/wrong_memory": {
        "n": 128,
        "correct": 89,
        "accuracy": 0.6953125,
        "gold_ce": 0.8109324796217123
      }
    },
    "paired": {
      "native_query": {
        "no_memory": {
          "both_correct": 98,
          "first_only": 8,
          "second_only": 5,
          "both_wrong": 17
        },
        "wrong_memory": {
          "both_correct": 106,
          "first_only": 0,
          "second_only": 0,
          "both_wrong": 22
        },
        "vs_frozen_writer": {
          "both_correct": 100,
          "first_only": 6,
          "second_only": 7,
          "both_wrong": 15
        }
      },
      "independent_query": {
        "no_memory": {
          "both_correct": 98,
          "first_only": 7,
          "second_only": 5,
          "both_wrong": 18
        },
        "wrong_memory": {
          "both_correct": 105,
          "first_only": 0,
          "second_only": 0,
          "both_wrong": 23
        },
        "vs_frozen_writer": {
          "both_correct": 100,
          "first_only": 5,
          "second_only": 7,
          "both_wrong": 16
        }
      },
      "old_coupled_reader": {
        "no_memory": {
          "both_correct": 91,
          "first_only": 16,
          "second_only": 12,
          "both_wrong": 9
        },
        "wrong_memory": {
          "both_correct": 84,
          "first_only": 23,
          "second_only": 5,
          "both_wrong": 16
        },
        "vs_frozen_writer": {
          "both_correct": 100,
          "first_only": 7,
          "second_only": 7,
          "both_wrong": 14
        }
      }
    },
    "validation_selection": {
      "native_query": {
        "epoch": 2,
        "step": 256,
        "validation": {
          "gold_ce": 0.5591081037395522,
          "correct": 104,
          "n": 128,
          "accuracy": 0.8125
        }
      },
      "independent_query": {
        "epoch": 2,
        "step": 256,
        "validation": {
          "gold_ce": 0.5503373281071298,
          "correct": 105,
          "n": 128,
          "accuracy": 0.8203125
        }
      },
      "old_coupled_reader": {
        "epoch": 2,
        "step": 256,
        "validation": {
          "gold_ce": 0.5369653313422305,
          "correct": 109,
          "n": 128,
          "accuracy": 0.8515625
        }
      }
    },
    "same_evidence_multiple_question_groups": {
      "the ocean contains large amounts of salt water": [
        "test_9-163",
        "test_7-157"
      ],
      "if the population of an organism increases then the ecosystem may become overpopulated with that organism": [
        "test_8-395",
        "test_7-50"
      ],
      "seeing is used for sensing visual things": [
        "test_266",
        "test_7-140"
      ],
      "if an animal hatches from an egg then that animal is born": [
        "test_1180",
        "test_9-149"
      ],
      "as heat increases , a flexible container containing gas will expand": [
        "test_132",
        "test_8-79"
      ],
      "a seed is used for storing food for a new plant": [
        "test_1184",
        "test_8-491"
      ]
    },
    "counts": {
      "train": 1024,
      "validation": 128,
      "test": 128
    },
    "steps_per_arm": 512,
    "caveats": [
      "Gold fact1 is supplied, not retrieved. OpenBookQA can often be solved from prior knowledge; memory ablations do not establish causal dependence by themselves.",
      "Independent Reader has question-only autoregressive positions; coupled baseline shifts question positions after external memory. This is an intentional interface difference.",
      "Frozen Writer is evaluated out-of-distribution on fact-only input; no new Stage-A, Writer training, shared new Sender, compression or joint optimization in this pilot.",
      "Layer pattern preregistered, not claimed optimal; old coupled baseline has different trainable parameter count, recorded in smoke and completed files."
    ]
  },
  "manifest": "/hy-tmp/yezhe/异构模型/decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234/manifest.json",
  "raw_results": "/hy-tmp/yezhe/异构模型/decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234/logs/pipeline.log"
}
```


## decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234/sciq_frozen64

```json
{
  "experiment": "decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234/sciq_frozen64",
  "status": "completed",
  "training": false,
  "metrics": {
    "baselines/question_only": {
      "n": 64,
      "correct": 57,
      "accuracy": 0.890625,
      "gold_ce": 18.903542697429657
    },
    "baselines/native_full_evidence_text": {
      "n": 64,
      "correct": 64,
      "accuracy": 1.0,
      "gold_ce": 17.52895388007164
    },
    "baselines/frozen_stage_b_physical_prefix": {
      "n": 64,
      "correct": 41,
      "accuracy": 0.640625,
      "gold_ce": 19.096840485930443
    },
    "native_query/correct_memory": {
      "n": 64,
      "correct": 57,
      "accuracy": 0.890625,
      "gold_ce": 0.672315117477412
    },
    "native_query/no_memory": {
      "n": 64,
      "correct": 57,
      "accuracy": 0.890625,
      "gold_ce": 18.903542697429657
    },
    "native_query/wrong_memory": {
      "n": 64,
      "correct": 57,
      "accuracy": 0.890625,
      "gold_ce": 0.7095959978671615
    },
    "independent_query/correct_memory": {
      "n": 64,
      "correct": 57,
      "accuracy": 0.890625,
      "gold_ce": 0.6645319885105749
    },
    "independent_query/no_memory": {
      "n": 64,
      "correct": 57,
      "accuracy": 0.890625,
      "gold_ce": 18.903542697429657
    },
    "independent_query/wrong_memory": {
      "n": 64,
      "correct": 57,
      "accuracy": 0.890625,
      "gold_ce": 0.6921894851375168
    },
    "old_coupled_reader/correct_memory": {
      "n": 64,
      "correct": 43,
      "accuracy": 0.671875,
      "gold_ce": 5.54573088396171
    },
    "old_coupled_reader/no_memory": {
      "n": 64,
      "correct": 57,
      "accuracy": 0.890625,
      "gold_ce": 18.903542697429657
    },
    "old_coupled_reader/wrong_memory": {
      "n": 64,
      "correct": 41,
      "accuracy": 0.640625,
      "gold_ce": 5.00245579811164
    }
  },
  "paired": {
    "native_query": {
      "no_memory": {
        "both_correct": 57,
        "correct_memory_only": 0,
        "control_only": 0,
        "both_wrong": 7,
        "prediction_identical": 64
      },
      "wrong_memory": {
        "both_correct": 57,
        "correct_memory_only": 0,
        "control_only": 0,
        "both_wrong": 7,
        "prediction_identical": 64
      }
    },
    "independent_query": {
      "no_memory": {
        "both_correct": 57,
        "correct_memory_only": 0,
        "control_only": 0,
        "both_wrong": 7,
        "prediction_identical": 64
      },
      "wrong_memory": {
        "both_correct": 57,
        "correct_memory_only": 0,
        "control_only": 0,
        "both_wrong": 7,
        "prediction_identical": 64
      }
    },
    "old_coupled_reader": {
      "no_memory": {
        "both_correct": 40,
        "correct_memory_only": 3,
        "control_only": 17,
        "both_wrong": 4,
        "prediction_identical": 41
      },
      "wrong_memory": {
        "both_correct": 34,
        "correct_memory_only": 9,
        "control_only": 7,
        "both_wrong": 14,
        "prediction_identical": 42
      }
    }
  },
  "config": {
    "name": "decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234/sciq_frozen64",
    "parent": "/hy-tmp/yezhe/异构模型/decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234",
    "dataset_repo": "allenai/sciq",
    "dataset_card": "https://huggingface.co/datasets/allenai/sciq",
    "dataset_split": "test",
    "count": 64,
    "seed": 1234,
    "selection": "First64 official test rows with nonempty support, four distinct nonempty answers and no exact normalized question/support overlap with known Reader/Writer training data; no performance filtering",
    "choice_order": "Deterministically shuffled distractors and balanced shuffled gold positions (16 each for A/B/C/D)",
    "arms": [
      "native_query",
      "independent_query",
      "old_coupled_reader"
    ],
    "reader_sha256": {
      "native_query": "5cacfcc54922587ee25264987a0f5fc2455095c61b87aa9799ec5ed4b5d78a23",
      "independent_query": "420522d3cb9b68ef7ee93959a3c9b5a8c5685ae7b602fba72fa1037099033536",
      "old_coupled_reader": "d54d7c44f4dc9b8d4801d161910b1642b6ced71acfa715b2674dde67a5c7a38a"
    },
    "training": false,
    "protocol": "Frozen OBQA Reader transfer; support-only Llama chat -> same frozen MCQ Writer -> full external memory; question+options Qwen chat, same ARC label formatter/cue/scorer; not native lm-eval SciQ leaderboard configuration",
    "memory_ablation": "Correct support, absent bank, deterministic support from another selected question with a distinct normalized support text",
    "full_evidence": "Untruncated support text, no 32/64-token cap; gold support supplied, no retrieval",
    "selection_of_checkpoints": "Keep previous OBQA validation-CE selected epoch2/step256 checkpoints; never select on SciQ test",
    "continuous_monitoring": false,
    "automatic_upload": false
  },
  "caveats": [
    "Gold support supplied; not retrieval. No-memory bypasses Reader; wrong supports may differ in length and are not guaranteed contradictory.",
    "Reader and Writer are frozen OBQA/MCQ models. This is a cross-dataset evidence protocol diagnostic, not unmodified lm-eval SciQ scoring."
  ],
  "time": "2026-10-20 15:16:57",
  "manifest": "/hy-tmp/yezhe/异构模型/decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234/sciq_frozen64/manifest.json",
  "raw_results": "/hy-tmp/yezhe/异构模型/decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234/sciq_frozen64/formal/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234/sciq_frozen64/logs/pipeline.log"
}
```


## ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234

```json
{
  "experiment": "ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
  "time": "2026-10-20 22:02:58",
  "status": "stage_b_training_completed_evaluation_pending",
  "config": {
    "name": "ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
    "seed": 1234,
    "parent": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
    "stage_a_sha256": "db1f295c502f27305a9d2ac410c47a1bdef4db7e0255f817483867b19d78cf0d",
    "old_stage_b_sha256": "06190cfcda6ce65e7b81447d2fa77b926bd43b384e2bfb5866083cc1588e278f",
    "train_counts": {
      "openbookqa": 512,
      "arc_challenge": 512,
      "gsm8k": 512,
      "mmlu_pro": 512
    },
    "validation_counts": {
      "openbookqa": 64,
      "arc_challenge": 64,
      "gsm8k": 64,
      "mmlu_pro": 64
    },
    "epochs": 4,
    "effective_batch": 8,
    "expected_updates": 1024,
    "adapter_rank": 64,
    "learning_rate": 0.0001,
    "weight_decay": 0,
    "gradient_clip": 30,
    "checkpoint_every_steps": 64,
    "warmup_native_epochs": 1,
    "prefix_max_new_tokens": 64,
    "supervised_positions_per_example_max": 8,
    "student_fraction_after_warmup": 0.5,
    "loss": "Exact full-vocabulary forward KL(Native || Translated), temperature1, FP32 log-softmax; mean valid positions per example, mean8 examples per update. No Gold CE/candidate KL/EOS CE/attention/KV loss.",
    "initialization": "New zero-up rank64 receiver-space residual on frozen existing Stage-A; do not initialize from old Stage-B",
    "sampling": "Native greedy rollout cached as token IDs; current Student greedy rollout online under no_grad on half of examples in every balanced batch after epoch1. Teacher scores the EXACT same selected prefix. No prefixes after first EOS.",
    "selection": "Primary: fixed epoch4/step1024 last.pt; best.pt by fixed Native-prefix held-out macro KL is diagnostic only, never test-selected",
    "native_parity_kl_max": 0.001,
    "native_parity_score_max": 0.02,
    "disk_reserve_gib": 20,
    "monitor_validation_per_task": 16,
    "formal_counts": {
      "openbookqa": 128,
      "arc_challenge": 128,
      "hellaswag": 64,
      "gsm8k": 32,
      "mmlu_pro": 128,
      "sciq": 64
    },
    "gsm8k_formal_max_new_tokens": 4096,
    "mmlu_pro_protocol": "Explicit extension: pinned ARC MCQ format generalized to actual A-J option count, label likelihood, Qwen chat thinking disabled; NOT official MMLU-Pro5shotCoT or lm-eval",
    "mmlu_pro_split": "Custom question-grouped split of local official test parquet; protect first128 historical test question groups, stratify512 unique training and64 validation questions by category; all remaining question groups held out. Never claim full official test is unseen.",
    "cache": "Reuse identity-checked parent MCQ aligned Sender FP16 caches; cache only additional aligned Sender KV, not every Native cache or full-vocab logits. Frozen Stage-A mapping computed on demand in train/eval. No truncation, deletion or quantization.",
    "continuous_monitoring": false,
    "automatic_upload": false
  },
  "results": {
    "status": "completed",
    "epochs": 4,
    "updates": 1024,
    "primary_checkpoint": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234/runs/stage_b/last.pt",
    "primary_sha256": "25942a14d77a2a36173cccf6653d0e86cef01bdfdbd082836395faa895ba995e",
    "selection": "Primary: fixed epoch4/step1024 last.pt; best.pt by fixed Native-prefix held-out macro KL is diagnostic only, never test-selected",
    "validation": [
      {
        "epoch": 1,
        "step": 256,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.12047904456267133,
              "supervised_positions": 98,
              "reached_cap": 0
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.13186672539450228,
              "supervised_positions": 120,
              "reached_cap": 0
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.3762377807870507,
              "supervised_positions": 128,
              "reached_cap": 16
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 1.0919599439948797,
              "supervised_positions": 116,
              "reached_cap": 4
            }
          },
          "macro_kl": 0.430135873684776
        }
      },
      {
        "epoch": 2,
        "step": 512,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.13234974723309278,
              "supervised_positions": 98,
              "reached_cap": 0
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.11422365659382194,
              "supervised_positions": 120,
              "reached_cap": 0
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.2689780853688717,
              "supervised_positions": 128,
              "reached_cap": 16
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 0.9263343522325158,
              "supervised_positions": 116,
              "reached_cap": 4
            }
          },
          "macro_kl": 0.36047146035707556
        }
      },
      {
        "epoch": 3,
        "step": 768,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.12413511835620739,
              "supervised_positions": 98,
              "reached_cap": 0
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.08736835546733346,
              "supervised_positions": 120,
              "reached_cap": 0
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.23387668933719397,
              "supervised_positions": 128,
              "reached_cap": 16
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 0.8788232505321503,
              "supervised_positions": 116,
              "reached_cap": 4
            }
          },
          "macro_kl": 0.33105085342322127
        }
      },
      {
        "epoch": 4,
        "step": 1024,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.10133792072883807,
              "supervised_positions": 98,
              "reached_cap": 0
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.06549449812882813,
              "supervised_positions": 120,
              "reached_cap": 0
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.2236092472448945,
              "supervised_positions": 128,
              "reached_cap": 16
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 0.8563042180612683,
              "supervised_positions": 116,
              "reached_cap": 4
            }
          },
          "macro_kl": 0.31168647104095726
        }
      }
    ],
    "manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739"
  },
  "manifest": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234/manifest.json",
  "raw_results": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234/logs/pipeline.log"
}
```


## ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234

```json
{
  "experiment": "ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
  "time": "2026-10-21 00:33:25",
  "status": "completed",
  "config": {
    "name": "ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
    "seed": 1234,
    "parent": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
    "stage_a_sha256": "db1f295c502f27305a9d2ac410c47a1bdef4db7e0255f817483867b19d78cf0d",
    "old_stage_b_sha256": "06190cfcda6ce65e7b81447d2fa77b926bd43b384e2bfb5866083cc1588e278f",
    "train_counts": {
      "openbookqa": 512,
      "arc_challenge": 512,
      "gsm8k": 512,
      "mmlu_pro": 512
    },
    "validation_counts": {
      "openbookqa": 64,
      "arc_challenge": 64,
      "gsm8k": 64,
      "mmlu_pro": 64
    },
    "epochs": 4,
    "effective_batch": 8,
    "expected_updates": 1024,
    "adapter_rank": 64,
    "learning_rate": 0.0001,
    "weight_decay": 0,
    "gradient_clip": 30,
    "checkpoint_every_steps": 64,
    "warmup_native_epochs": 1,
    "prefix_max_new_tokens": 64,
    "supervised_positions_per_example_max": 8,
    "student_fraction_after_warmup": 0.5,
    "loss": "Exact full-vocabulary forward KL(Native || Translated), temperature1, FP32 log-softmax; mean valid positions per example, mean8 examples per update. No Gold CE/candidate KL/EOS CE/attention/KV loss.",
    "initialization": "New zero-up rank64 receiver-space residual on frozen existing Stage-A; do not initialize from old Stage-B",
    "sampling": "Native greedy rollout cached as token IDs; current Student greedy rollout online under no_grad on half of examples in every balanced batch after epoch1. Teacher scores the EXACT same selected prefix. No prefixes after first EOS.",
    "selection": "Primary: fixed epoch4/step1024 last.pt; best.pt by fixed Native-prefix held-out macro KL is diagnostic only, never test-selected",
    "native_parity_kl_max": 0.001,
    "native_parity_score_max": 0.02,
    "disk_reserve_gib": 20,
    "monitor_validation_per_task": 16,
    "formal_counts": {
      "openbookqa": 128,
      "arc_challenge": 128,
      "hellaswag": 64,
      "gsm8k": 32,
      "mmlu_pro": 128,
      "sciq": 64
    },
    "gsm8k_formal_max_new_tokens": 4096,
    "mmlu_pro_protocol": "Explicit extension: pinned ARC MCQ format generalized to actual A-J option count, label likelihood, Qwen chat thinking disabled; NOT official MMLU-Pro5shotCoT or lm-eval",
    "mmlu_pro_split": "Custom question-grouped split of local official test parquet; protect first128 historical test question groups, stratify512 unique training and64 validation questions by category; all remaining question groups held out. Never claim full official test is unseen.",
    "cache": "Reuse identity-checked parent MCQ aligned Sender FP16 caches; cache only additional aligned Sender KV, not every Native cache or full-vocab logits. Frozen Stage-A mapping computed on demand in train/eval. No truncation, deletion or quantization.",
    "continuous_monitoring": false,
    "automatic_upload": false
  },
  "results": {
    "experiment": "ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
    "status": "completed",
    "config": {
      "name": "ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
      "seed": 1234,
      "parent": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
      "stage_a_sha256": "db1f295c502f27305a9d2ac410c47a1bdef4db7e0255f817483867b19d78cf0d",
      "old_stage_b_sha256": "06190cfcda6ce65e7b81447d2fa77b926bd43b384e2bfb5866083cc1588e278f",
      "train_counts": {
        "openbookqa": 512,
        "arc_challenge": 512,
        "gsm8k": 512,
        "mmlu_pro": 512
      },
      "validation_counts": {
        "openbookqa": 64,
        "arc_challenge": 64,
        "gsm8k": 64,
        "mmlu_pro": 64
      },
      "epochs": 4,
      "effective_batch": 8,
      "expected_updates": 1024,
      "adapter_rank": 64,
      "learning_rate": 0.0001,
      "weight_decay": 0,
      "gradient_clip": 30,
      "checkpoint_every_steps": 64,
      "warmup_native_epochs": 1,
      "prefix_max_new_tokens": 64,
      "supervised_positions_per_example_max": 8,
      "student_fraction_after_warmup": 0.5,
      "loss": "Exact full-vocabulary forward KL(Native || Translated), temperature1, FP32 log-softmax; mean valid positions per example, mean8 examples per update. No Gold CE/candidate KL/EOS CE/attention/KV loss.",
      "initialization": "New zero-up rank64 receiver-space residual on frozen existing Stage-A; do not initialize from old Stage-B",
      "sampling": "Native greedy rollout cached as token IDs; current Student greedy rollout online under no_grad on half of examples in every balanced batch after epoch1. Teacher scores the EXACT same selected prefix. No prefixes after first EOS.",
      "selection": "Primary: fixed epoch4/step1024 last.pt; best.pt by fixed Native-prefix held-out macro KL is diagnostic only, never test-selected",
      "native_parity_kl_max": 0.001,
      "native_parity_score_max": 0.02,
      "disk_reserve_gib": 20,
      "monitor_validation_per_task": 16,
      "formal_counts": {
        "openbookqa": 128,
        "arc_challenge": 128,
        "hellaswag": 64,
        "gsm8k": 32,
        "mmlu_pro": 128,
        "sciq": 64
      },
      "gsm8k_formal_max_new_tokens": 4096,
      "mmlu_pro_protocol": "Explicit extension: pinned ARC MCQ format generalized to actual A-J option count, label likelihood, Qwen chat thinking disabled; NOT official MMLU-Pro5shotCoT or lm-eval",
      "mmlu_pro_split": "Custom question-grouped split of local official test parquet; protect first128 historical test question groups, stratify512 unique training and64 validation questions by category; all remaining question groups held out. Never claim full official test is unseen.",
      "cache": "Reuse identity-checked parent MCQ aligned Sender FP16 caches; cache only additional aligned Sender KV, not every Native cache or full-vocab logits. Frozen Stage-A mapping computed on demand in train/eval. No truncation, deletion or quantization.",
      "continuous_monitoring": false,
      "automatic_upload": false
    },
    "tasks": {
      "openbookqa": {
        "metrics": {
          "qwen_native": {
            "n": 128,
            "correct": 103,
            "accuracy": 0.8046875
          },
          "old_mcq_stage_b": {
            "n": 128,
            "correct": 91,
            "accuracy": 0.7109375
          },
          "new_ufd_stage_b": {
            "n": 128,
            "correct": 84,
            "accuracy": 0.65625
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 76,
            "new_only": 8,
            "other_only": 27,
            "both_wrong": 17
          },
          "old_mcq_stage_b": {
            "both_correct": 78,
            "new_only": 6,
            "other_only": 13,
            "both_wrong": 31
          }
        }
      },
      "arc_challenge": {
        "metrics": {
          "qwen_native": {
            "n": 128,
            "correct": 111,
            "accuracy": 0.8671875
          },
          "old_mcq_stage_b": {
            "n": 128,
            "correct": 94,
            "accuracy": 0.734375
          },
          "new_ufd_stage_b": {
            "n": 128,
            "correct": 83,
            "accuracy": 0.6484375
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 78,
            "new_only": 5,
            "other_only": 33,
            "both_wrong": 12
          },
          "old_mcq_stage_b": {
            "both_correct": 79,
            "new_only": 4,
            "other_only": 15,
            "both_wrong": 30
          }
        }
      },
      "hellaswag": {
        "metrics": {
          "qwen_native": {
            "n": 64,
            "correct": 23,
            "accuracy": 0.359375
          },
          "old_mcq_stage_b": {
            "n": 64,
            "correct": 21,
            "accuracy": 0.328125
          },
          "new_ufd_stage_b": {
            "n": 64,
            "correct": 21,
            "accuracy": 0.328125
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 12,
            "new_only": 9,
            "other_only": 11,
            "both_wrong": 32
          },
          "old_mcq_stage_b": {
            "both_correct": 13,
            "new_only": 8,
            "other_only": 8,
            "both_wrong": 35
          }
        }
      },
      "gsm8k": {
        "metrics": {
          "qwen_native": {
            "n": 32,
            "correct": 31,
            "accuracy": 0.96875,
            "mean_generated_tokens": 296.375,
            "cap_count": 0
          },
          "old_mcq_stage_b": {
            "n": 32,
            "correct": 0,
            "accuracy": 0.0,
            "mean_generated_tokens": 3114.1875,
            "cap_count": 23
          },
          "new_ufd_stage_b": {
            "n": 32,
            "correct": 9,
            "accuracy": 0.28125,
            "mean_generated_tokens": 444.78125,
            "cap_count": 1
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 9,
            "new_only": 0,
            "other_only": 22,
            "both_wrong": 1
          },
          "old_mcq_stage_b": {
            "both_correct": 0,
            "new_only": 9,
            "other_only": 0,
            "both_wrong": 23
          }
        }
      },
      "mmlu_pro": {
        "metrics": {
          "qwen_native": {
            "n": 128,
            "correct": 52,
            "accuracy": 0.40625
          },
          "old_mcq_stage_b": {
            "n": 128,
            "correct": 17,
            "accuracy": 0.1328125
          },
          "new_ufd_stage_b": {
            "n": 128,
            "correct": 24,
            "accuracy": 0.1875
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 16,
            "new_only": 8,
            "other_only": 36,
            "both_wrong": 68
          },
          "old_mcq_stage_b": {
            "both_correct": 10,
            "new_only": 14,
            "other_only": 7,
            "both_wrong": 97
          }
        }
      },
      "sciq": {
        "metrics": {
          "qwen_native": {
            "n": 64,
            "correct": 57,
            "accuracy": 0.890625
          },
          "old_mcq_stage_b": {
            "n": 64,
            "correct": 59,
            "accuracy": 0.921875
          },
          "new_ufd_stage_b": {
            "n": 64,
            "correct": 58,
            "accuracy": 0.90625
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 53,
            "new_only": 5,
            "other_only": 4,
            "both_wrong": 2
          },
          "old_mcq_stage_b": {
            "both_correct": 55,
            "new_only": 3,
            "other_only": 4,
            "both_wrong": 2
          }
        }
      }
    },
    "training": {
      "status": "completed",
      "epochs": 4,
      "updates": 1024,
      "primary_checkpoint": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234/runs/stage_b/last.pt",
      "primary_sha256": "25942a14d77a2a36173cccf6653d0e86cef01bdfdbd082836395faa895ba995e",
      "selection": "Primary: fixed epoch4/step1024 last.pt; best.pt by fixed Native-prefix held-out macro KL is diagnostic only, never test-selected",
      "validation": [
        {
          "epoch": 1,
          "step": 256,
          "validation": {
            "tasks": {
              "openbookqa": {
                "n": 16,
                "mean_kl": 0.12047904456267133,
                "supervised_positions": 98,
                "reached_cap": 0
              },
              "arc_challenge": {
                "n": 16,
                "mean_kl": 0.13186672539450228,
                "supervised_positions": 120,
                "reached_cap": 0
              },
              "gsm8k": {
                "n": 16,
                "mean_kl": 0.3762377807870507,
                "supervised_positions": 128,
                "reached_cap": 16
              },
              "mmlu_pro": {
                "n": 16,
                "mean_kl": 1.0919599439948797,
                "supervised_positions": 116,
                "reached_cap": 4
              }
            },
            "macro_kl": 0.430135873684776
          }
        },
        {
          "epoch": 2,
          "step": 512,
          "validation": {
            "tasks": {
              "openbookqa": {
                "n": 16,
                "mean_kl": 0.13234974723309278,
                "supervised_positions": 98,
                "reached_cap": 0
              },
              "arc_challenge": {
                "n": 16,
                "mean_kl": 0.11422365659382194,
                "supervised_positions": 120,
                "reached_cap": 0
              },
              "gsm8k": {
                "n": 16,
                "mean_kl": 0.2689780853688717,
                "supervised_positions": 128,
                "reached_cap": 16
              },
              "mmlu_pro": {
                "n": 16,
                "mean_kl": 0.9263343522325158,
                "supervised_positions": 116,
                "reached_cap": 4
              }
            },
            "macro_kl": 0.36047146035707556
          }
        },
        {
          "epoch": 3,
          "step": 768,
          "validation": {
            "tasks": {
              "openbookqa": {
                "n": 16,
                "mean_kl": 0.12413511835620739,
                "supervised_positions": 98,
                "reached_cap": 0
              },
              "arc_challenge": {
                "n": 16,
                "mean_kl": 0.08736835546733346,
                "supervised_positions": 120,
                "reached_cap": 0
              },
              "gsm8k": {
                "n": 16,
                "mean_kl": 0.23387668933719397,
                "supervised_positions": 128,
                "reached_cap": 16
              },
              "mmlu_pro": {
                "n": 16,
                "mean_kl": 0.8788232505321503,
                "supervised_positions": 116,
                "reached_cap": 4
              }
            },
            "macro_kl": 0.33105085342322127
          }
        },
        {
          "epoch": 4,
          "step": 1024,
          "validation": {
            "tasks": {
              "openbookqa": {
                "n": 16,
                "mean_kl": 0.10133792072883807,
                "supervised_positions": 98,
                "reached_cap": 0
              },
              "arc_challenge": {
                "n": 16,
                "mean_kl": 0.06549449812882813,
                "supervised_positions": 120,
                "reached_cap": 0
              },
              "gsm8k": {
                "n": 16,
                "mean_kl": 0.2236092472448945,
                "supervised_positions": 128,
                "reached_cap": 16
              },
              "mmlu_pro": {
                "n": 16,
                "mean_kl": 0.8563042180612683,
                "supervised_positions": 116,
                "reached_cap": 4
              }
            },
            "macro_kl": 0.31168647104095726
          }
        }
      ],
      "manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739"
    },
    "caveats": [
      "MMLU-Pro is custom heldout split with explicit ARC-style A-J extension, NOT untouched official test/CoT benchmark.",
      "New loss and training mixture both changed; comparisons are frozen checkpoints, not same-data loss ablations.",
      "Old MCQ best checkpoint was validation-choice-KL selected; new primary is fixed epoch4. Report checkpoint identity, not a hidden common selection claim."
    ]
  },
  "manifest": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234/manifest.json",
  "raw_results": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234/logs/pipeline.log"
}
```


## ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234

```json
{
  "experiment": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234",
  "timestamp": "2026-10-21 03:54:38",
  "status": "reader_training_completed_evaluation_pending",
  "config": {
    "name": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234",
    "seed": 1234,
    "parent": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
    "parent_manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739",
    "writer_stage_b_sha256": "25942a14d77a2a36173cccf6653d0e86cef01bdfdbd082836395faa895ba995e",
    "rank": 16,
    "alpha": 16,
    "dropout": 0,
    "layers": 36,
    "targets": [
      "q_proj",
      "o_proj"
    ],
    "expected_trainable_parameters": 7667712,
    "epochs": 4,
    "expected_updates": 1024,
    "effective_batch": 8,
    "learning_rate": 0.0001,
    "weight_decay": 0,
    "gradient_clip": 30,
    "checkpoint_every_steps": 64,
    "validation_every_steps": 256,
    "monitor_validation_per_task": 16,
    "prefix_max_new_tokens": 64,
    "supervised_positions_per_example_max": 8,
    "train_counts": {
      "openbookqa": 512,
      "arc_challenge": 512,
      "gsm8k": 512,
      "mmlu_pro": 512
    },
    "formal_counts": {
      "openbookqa": 128,
      "arc_challenge": 128,
      "hellaswag": 64,
      "gsm8k": 32,
      "mmlu_pro": 128,
      "sciq": 64
    },
    "loss": "Unchanged UFD full-vocabulary forward KL(Native||Writer+Reader), temperature1, FP32 loss; up to8 unique prediction positions, mean examples. No gold CE, choice KL, attention or KV loss.",
    "prefix_sampling": "Exact parent epoch order and positions. Epoch1 Native greedy; epochs2-4 exactly4 current Writer+Reader Student examples and4 Native examples per balanced batch. Student prefix sampling detached. Native Teacher always disables Reader.",
    "interface": "Frozen Stage-A and fixed UFD Stage-B step1024; unchanged Full-Sync, native anchor, native tail, chat, suffix, physical cache and absolute positions. Reader disabled during Native body/cache/Teacher computation; enabled only for translated-cache continuation including suffix and generation.",
    "selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
    "validation_task_metrics": "On fixed16 validation rows/task: KL plus same-prefix next-token agreement for all4 tasks; label accuracy on3 MCQ tasks. Full GSM8K free-generation accuracy only in final test, using unchanged4096 cap.",
    "baseline_reuse": "Reuse identity- and prompt-checked parent Qwen Native / old MCQ Writer / UFD Writer per-sample outputs. Compute only frozen UFD Writer+Reader. Require complete parent pipeline before GPU launch.",
    "queue_min_free_gpu_gib": 20,
    "queue_retry_seconds": 300,
    "disk_reserve_gib": 20,
    "continuous_monitoring": false,
    "automatic_upload": false
  },
  "checkpoint_selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
  "results": {
    "status": "completed",
    "epochs": 4,
    "updates": 1024,
    "selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
    "checkpoint": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/runs/reader/epoch4_step1024.pt",
    "checkpoint_sha256": "a757ce6e4559168ee227ec53e380f16df9f6f6cd2c82b02c556802fdcf6f9599",
    "validation": [
      {
        "epoch": 1,
        "step": 256,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.09355103113921359,
              "same_prefix_next_token_agreement": 0.9591836734693877,
              "supervised_positions": 98,
              "label_accuracy": {
                "qwen_native": 0.8125,
                "writer_only": 0.625,
                "writer_reader": 0.625
              }
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.06087099425349152,
              "same_prefix_next_token_agreement": 0.9833333333333333,
              "supervised_positions": 120,
              "label_accuracy": {
                "qwen_native": 1.0,
                "writer_only": 0.8125,
                "writer_reader": 0.8125
              }
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.20126126985996962,
              "same_prefix_next_token_agreement": 0.9140625,
              "supervised_positions": 128
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 0.7360535431653261,
              "same_prefix_next_token_agreement": 0.7586206896551724,
              "supervised_positions": 116,
              "label_accuracy": {
                "qwen_native": 0.3125,
                "writer_only": 0.3125,
                "writer_reader": 0.3125
              }
            }
          },
          "macro_kl": 0.2729342096045002
        }
      },
      {
        "epoch": 2,
        "step": 512,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.10811749477579724,
              "same_prefix_next_token_agreement": 0.9591836734693877,
              "supervised_positions": 98,
              "label_accuracy": {
                "qwen_native": 0.8125,
                "writer_only": 0.625,
                "writer_reader": 0.625
              }
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.06442411229363643,
              "same_prefix_next_token_agreement": 0.9833333333333333,
              "supervised_positions": 120,
              "label_accuracy": {
                "qwen_native": 1.0,
                "writer_only": 0.8125,
                "writer_reader": 0.8125
              }
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.1949733137153089,
              "same_prefix_next_token_agreement": 0.8828125,
              "supervised_positions": 128
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 0.6555280368775129,
              "same_prefix_next_token_agreement": 0.75,
              "supervised_positions": 116,
              "label_accuracy": {
                "qwen_native": 0.3125,
                "writer_only": 0.3125,
                "writer_reader": 0.3125
              }
            }
          },
          "macro_kl": 0.2557607394155639
        }
      },
      {
        "epoch": 3,
        "step": 768,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.10256033984478563,
              "same_prefix_next_token_agreement": 0.9591836734693877,
              "supervised_positions": 98,
              "label_accuracy": {
                "qwen_native": 0.8125,
                "writer_only": 0.625,
                "writer_reader": 0.6875
              }
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.06867390857223654,
              "same_prefix_next_token_agreement": 0.975,
              "supervised_positions": 120,
              "label_accuracy": {
                "qwen_native": 1.0,
                "writer_only": 0.8125,
                "writer_reader": 0.8125
              }
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.20785359921865165,
              "same_prefix_next_token_agreement": 0.8828125,
              "supervised_positions": 128
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 0.6279409043490887,
              "same_prefix_next_token_agreement": 0.8017241379310345,
              "supervised_positions": 116,
              "label_accuracy": {
                "qwen_native": 0.3125,
                "writer_only": 0.3125,
                "writer_reader": 0.25
              }
            }
          },
          "macro_kl": 0.2517571879961906
        }
      },
      {
        "epoch": 4,
        "step": 1024,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.08937330276239663,
              "same_prefix_next_token_agreement": 0.9795918367346939,
              "supervised_positions": 98,
              "label_accuracy": {
                "qwen_native": 0.8125,
                "writer_only": 0.625,
                "writer_reader": 0.625
              }
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.05219782930362271,
              "same_prefix_next_token_agreement": 0.9916666666666667,
              "supervised_positions": 120,
              "label_accuracy": {
                "qwen_native": 1.0,
                "writer_only": 0.8125,
                "writer_reader": 0.8125
              }
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.18463176197838038,
              "same_prefix_next_token_agreement": 0.90625,
              "supervised_positions": 128
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 0.6679957266896963,
              "same_prefix_next_token_agreement": 0.7586206896551724,
              "supervised_positions": 116,
              "label_accuracy": {
                "qwen_native": 0.3125,
                "writer_only": 0.3125,
                "writer_reader": 0.25
              }
            }
          },
          "macro_kl": 0.248549655183524
        }
      }
    ],
    "memory": {
      "peak_pytorch_allocated_gib": 9.928777694702148,
      "peak_pytorch_reserved_gib": 10.841796875,
      "scope": "This process CUDA allocator since load_all; not whole-GPU nvidia-smi peak"
    },
    "manifest_sha256": "e3cdafb48dca164e6193671c5271cf1772772aecb5e8300c7ddbbac98d04ee41"
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/logs/pipeline.log"
}
```


## ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234

```json
{
  "experiment": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234",
  "timestamp": "2026-10-21 04:14:28",
  "status": "completed",
  "config": {
    "name": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234",
    "seed": 1234,
    "parent": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
    "parent_manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739",
    "writer_stage_b_sha256": "25942a14d77a2a36173cccf6653d0e86cef01bdfdbd082836395faa895ba995e",
    "rank": 16,
    "alpha": 16,
    "dropout": 0,
    "layers": 36,
    "targets": [
      "q_proj",
      "o_proj"
    ],
    "expected_trainable_parameters": 7667712,
    "epochs": 4,
    "expected_updates": 1024,
    "effective_batch": 8,
    "learning_rate": 0.0001,
    "weight_decay": 0,
    "gradient_clip": 30,
    "checkpoint_every_steps": 64,
    "validation_every_steps": 256,
    "monitor_validation_per_task": 16,
    "prefix_max_new_tokens": 64,
    "supervised_positions_per_example_max": 8,
    "train_counts": {
      "openbookqa": 512,
      "arc_challenge": 512,
      "gsm8k": 512,
      "mmlu_pro": 512
    },
    "formal_counts": {
      "openbookqa": 128,
      "arc_challenge": 128,
      "hellaswag": 64,
      "gsm8k": 32,
      "mmlu_pro": 128,
      "sciq": 64
    },
    "loss": "Unchanged UFD full-vocabulary forward KL(Native||Writer+Reader), temperature1, FP32 loss; up to8 unique prediction positions, mean examples. No gold CE, choice KL, attention or KV loss.",
    "prefix_sampling": "Exact parent epoch order and positions. Epoch1 Native greedy; epochs2-4 exactly4 current Writer+Reader Student examples and4 Native examples per balanced batch. Student prefix sampling detached. Native Teacher always disables Reader.",
    "interface": "Frozen Stage-A and fixed UFD Stage-B step1024; unchanged Full-Sync, native anchor, native tail, chat, suffix, physical cache and absolute positions. Reader disabled during Native body/cache/Teacher computation; enabled only for translated-cache continuation including suffix and generation.",
    "selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
    "validation_task_metrics": "On fixed16 validation rows/task: KL plus same-prefix next-token agreement for all4 tasks; label accuracy on3 MCQ tasks. Full GSM8K free-generation accuracy only in final test, using unchanged4096 cap.",
    "baseline_reuse": "Reuse identity- and prompt-checked parent Qwen Native / old MCQ Writer / UFD Writer per-sample outputs. Compute only frozen UFD Writer+Reader. Require complete parent pipeline before GPU launch.",
    "queue_min_free_gpu_gib": 20,
    "queue_retry_seconds": 300,
    "disk_reserve_gib": 20,
    "continuous_monitoring": false,
    "automatic_upload": false
  },
  "checkpoint_selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
  "results": {
    "experiment": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234",
    "status": "completed",
    "config": {
      "name": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234",
      "seed": 1234,
      "parent": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
      "parent_manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739",
      "writer_stage_b_sha256": "25942a14d77a2a36173cccf6653d0e86cef01bdfdbd082836395faa895ba995e",
      "rank": 16,
      "alpha": 16,
      "dropout": 0,
      "layers": 36,
      "targets": [
        "q_proj",
        "o_proj"
      ],
      "expected_trainable_parameters": 7667712,
      "epochs": 4,
      "expected_updates": 1024,
      "effective_batch": 8,
      "learning_rate": 0.0001,
      "weight_decay": 0,
      "gradient_clip": 30,
      "checkpoint_every_steps": 64,
      "validation_every_steps": 256,
      "monitor_validation_per_task": 16,
      "prefix_max_new_tokens": 64,
      "supervised_positions_per_example_max": 8,
      "train_counts": {
        "openbookqa": 512,
        "arc_challenge": 512,
        "gsm8k": 512,
        "mmlu_pro": 512
      },
      "formal_counts": {
        "openbookqa": 128,
        "arc_challenge": 128,
        "hellaswag": 64,
        "gsm8k": 32,
        "mmlu_pro": 128,
        "sciq": 64
      },
      "loss": "Unchanged UFD full-vocabulary forward KL(Native||Writer+Reader), temperature1, FP32 loss; up to8 unique prediction positions, mean examples. No gold CE, choice KL, attention or KV loss.",
      "prefix_sampling": "Exact parent epoch order and positions. Epoch1 Native greedy; epochs2-4 exactly4 current Writer+Reader Student examples and4 Native examples per balanced batch. Student prefix sampling detached. Native Teacher always disables Reader.",
      "interface": "Frozen Stage-A and fixed UFD Stage-B step1024; unchanged Full-Sync, native anchor, native tail, chat, suffix, physical cache and absolute positions. Reader disabled during Native body/cache/Teacher computation; enabled only for translated-cache continuation including suffix and generation.",
      "selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
      "validation_task_metrics": "On fixed16 validation rows/task: KL plus same-prefix next-token agreement for all4 tasks; label accuracy on3 MCQ tasks. Full GSM8K free-generation accuracy only in final test, using unchanged4096 cap.",
      "baseline_reuse": "Reuse identity- and prompt-checked parent Qwen Native / old MCQ Writer / UFD Writer per-sample outputs. Compute only frozen UFD Writer+Reader. Require complete parent pipeline before GPU launch.",
      "queue_min_free_gpu_gib": 20,
      "queue_retry_seconds": 300,
      "disk_reserve_gib": 20,
      "continuous_monitoring": false,
      "automatic_upload": false
    },
    "tasks": {
      "openbookqa": {
        "metrics": {
          "qwen_native": {
            "n": 128,
            "correct": 103,
            "accuracy": 0.8046875
          },
          "old_mcq_stage_b": {
            "n": 128,
            "correct": 91,
            "accuracy": 0.7109375
          },
          "new_ufd_stage_b": {
            "n": 128,
            "correct": 84,
            "accuracy": 0.65625
          },
          "ufd_writer_qo_reader": {
            "n": 128,
            "correct": 86,
            "accuracy": 0.671875
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 78,
            "reader_only": 8,
            "other_only": 25,
            "both_wrong": 17
          },
          "old_mcq_stage_b": {
            "both_correct": 80,
            "reader_only": 6,
            "other_only": 11,
            "both_wrong": 31
          },
          "new_ufd_stage_b": {
            "both_correct": 84,
            "reader_only": 2,
            "other_only": 0,
            "both_wrong": 42
          }
        }
      },
      "arc_challenge": {
        "metrics": {
          "qwen_native": {
            "n": 128,
            "correct": 111,
            "accuracy": 0.8671875
          },
          "old_mcq_stage_b": {
            "n": 128,
            "correct": 94,
            "accuracy": 0.734375
          },
          "new_ufd_stage_b": {
            "n": 128,
            "correct": 83,
            "accuracy": 0.6484375
          },
          "ufd_writer_qo_reader": {
            "n": 128,
            "correct": 83,
            "accuracy": 0.6484375
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 78,
            "reader_only": 5,
            "other_only": 33,
            "both_wrong": 12
          },
          "old_mcq_stage_b": {
            "both_correct": 81,
            "reader_only": 2,
            "other_only": 13,
            "both_wrong": 32
          },
          "new_ufd_stage_b": {
            "both_correct": 81,
            "reader_only": 2,
            "other_only": 2,
            "both_wrong": 43
          }
        }
      },
      "hellaswag": {
        "metrics": {
          "qwen_native": {
            "n": 64,
            "correct": 23,
            "accuracy": 0.359375
          },
          "old_mcq_stage_b": {
            "n": 64,
            "correct": 21,
            "accuracy": 0.328125
          },
          "new_ufd_stage_b": {
            "n": 64,
            "correct": 21,
            "accuracy": 0.328125
          },
          "ufd_writer_qo_reader": {
            "n": 64,
            "correct": 21,
            "accuracy": 0.328125
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 14,
            "reader_only": 7,
            "other_only": 9,
            "both_wrong": 34
          },
          "old_mcq_stage_b": {
            "both_correct": 14,
            "reader_only": 7,
            "other_only": 7,
            "both_wrong": 36
          },
          "new_ufd_stage_b": {
            "both_correct": 19,
            "reader_only": 2,
            "other_only": 2,
            "both_wrong": 41
          }
        }
      },
      "gsm8k": {
        "metrics": {
          "qwen_native": {
            "n": 32,
            "correct": 31,
            "accuracy": 0.96875,
            "mean_generated_tokens": 296.375,
            "cap_count": 0
          },
          "old_mcq_stage_b": {
            "n": 32,
            "correct": 0,
            "accuracy": 0.0,
            "mean_generated_tokens": 3114.1875,
            "cap_count": 23
          },
          "new_ufd_stage_b": {
            "n": 32,
            "correct": 9,
            "accuracy": 0.28125,
            "mean_generated_tokens": 444.78125,
            "cap_count": 1
          },
          "ufd_writer_qo_reader": {
            "n": 32,
            "correct": 15,
            "accuracy": 0.46875,
            "mean_generated_tokens": 353.46875,
            "cap_count": 0
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 15,
            "reader_only": 0,
            "other_only": 16,
            "both_wrong": 1
          },
          "old_mcq_stage_b": {
            "both_correct": 0,
            "reader_only": 15,
            "other_only": 0,
            "both_wrong": 17
          },
          "new_ufd_stage_b": {
            "both_correct": 9,
            "reader_only": 6,
            "other_only": 0,
            "both_wrong": 17
          }
        }
      },
      "mmlu_pro": {
        "metrics": {
          "qwen_native": {
            "n": 128,
            "correct": 52,
            "accuracy": 0.40625
          },
          "old_mcq_stage_b": {
            "n": 128,
            "correct": 17,
            "accuracy": 0.1328125
          },
          "new_ufd_stage_b": {
            "n": 128,
            "correct": 24,
            "accuracy": 0.1875
          },
          "ufd_writer_qo_reader": {
            "n": 128,
            "correct": 26,
            "accuracy": 0.203125
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 18,
            "reader_only": 8,
            "other_only": 34,
            "both_wrong": 68
          },
          "old_mcq_stage_b": {
            "both_correct": 9,
            "reader_only": 17,
            "other_only": 8,
            "both_wrong": 94
          },
          "new_ufd_stage_b": {
            "both_correct": 23,
            "reader_only": 3,
            "other_only": 1,
            "both_wrong": 101
          }
        }
      },
      "sciq": {
        "metrics": {
          "qwen_native": {
            "n": 64,
            "correct": 57,
            "accuracy": 0.890625
          },
          "old_mcq_stage_b": {
            "n": 64,
            "correct": 59,
            "accuracy": 0.921875
          },
          "new_ufd_stage_b": {
            "n": 64,
            "correct": 58,
            "accuracy": 0.90625
          },
          "ufd_writer_qo_reader": {
            "n": 64,
            "correct": 59,
            "accuracy": 0.921875
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 55,
            "reader_only": 4,
            "other_only": 2,
            "both_wrong": 3
          },
          "old_mcq_stage_b": {
            "both_correct": 57,
            "reader_only": 2,
            "other_only": 2,
            "both_wrong": 3
          },
          "new_ufd_stage_b": {
            "both_correct": 57,
            "reader_only": 2,
            "other_only": 1,
            "both_wrong": 4
          }
        }
      }
    },
    "training": {
      "status": "completed",
      "epochs": 4,
      "updates": 1024,
      "selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
      "checkpoint": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/runs/reader/epoch4_step1024.pt",
      "checkpoint_sha256": "a757ce6e4559168ee227ec53e380f16df9f6f6cd2c82b02c556802fdcf6f9599",
      "validation": [
        {
          "epoch": 1,
          "step": 256,
          "validation": {
            "tasks": {
              "openbookqa": {
                "n": 16,
                "mean_kl": 0.09355103113921359,
                "same_prefix_next_token_agreement": 0.9591836734693877,
                "supervised_positions": 98,
                "label_accuracy": {
                  "qwen_native": 0.8125,
                  "writer_only": 0.625,
                  "writer_reader": 0.625
                }
              },
              "arc_challenge": {
                "n": 16,
                "mean_kl": 0.06087099425349152,
                "same_prefix_next_token_agreement": 0.9833333333333333,
                "supervised_positions": 120,
                "label_accuracy": {
                  "qwen_native": 1.0,
                  "writer_only": 0.8125,
                  "writer_reader": 0.8125
                }
              },
              "gsm8k": {
                "n": 16,
                "mean_kl": 0.20126126985996962,
                "same_prefix_next_token_agreement": 0.9140625,
                "supervised_positions": 128
              },
              "mmlu_pro": {
                "n": 16,
                "mean_kl": 0.7360535431653261,
                "same_prefix_next_token_agreement": 0.7586206896551724,
                "supervised_positions": 116,
                "label_accuracy": {
                  "qwen_native": 0.3125,
                  "writer_only": 0.3125,
                  "writer_reader": 0.3125
                }
              }
            },
            "macro_kl": 0.2729342096045002
          }
        },
        {
          "epoch": 2,
          "step": 512,
          "validation": {
            "tasks": {
              "openbookqa": {
                "n": 16,
                "mean_kl": 0.10811749477579724,
                "same_prefix_next_token_agreement": 0.9591836734693877,
                "supervised_positions": 98,
                "label_accuracy": {
                  "qwen_native": 0.8125,
                  "writer_only": 0.625,
                  "writer_reader": 0.625
                }
              },
              "arc_challenge": {
                "n": 16,
                "mean_kl": 0.06442411229363643,
                "same_prefix_next_token_agreement": 0.9833333333333333,
                "supervised_positions": 120,
                "label_accuracy": {
                  "qwen_native": 1.0,
                  "writer_only": 0.8125,
                  "writer_reader": 0.8125
                }
              },
              "gsm8k": {
                "n": 16,
                "mean_kl": 0.1949733137153089,
                "same_prefix_next_token_agreement": 0.8828125,
                "supervised_positions": 128
              },
              "mmlu_pro": {
                "n": 16,
                "mean_kl": 0.6555280368775129,
                "same_prefix_next_token_agreement": 0.75,
                "supervised_positions": 116,
                "label_accuracy": {
                  "qwen_native": 0.3125,
                  "writer_only": 0.3125,
                  "writer_reader": 0.3125
                }
              }
            },
            "macro_kl": 0.2557607394155639
          }
        },
        {
          "epoch": 3,
          "step": 768,
          "validation": {
            "tasks": {
              "openbookqa": {
                "n": 16,
                "mean_kl": 0.10256033984478563,
                "same_prefix_next_token_agreement": 0.9591836734693877,
                "supervised_positions": 98,
                "label_accuracy": {
                  "qwen_native": 0.8125,
                  "writer_only": 0.625,
                  "writer_reader": 0.6875
                }
              },
              "arc_challenge": {
                "n": 16,
                "mean_kl": 0.06867390857223654,
                "same_prefix_next_token_agreement": 0.975,
                "supervised_positions": 120,
                "label_accuracy": {
                  "qwen_native": 1.0,
                  "writer_only": 0.8125,
                  "writer_reader": 0.8125
                }
              },
              "gsm8k": {
                "n": 16,
                "mean_kl": 0.20785359921865165,
                "same_prefix_next_token_agreement": 0.8828125,
                "supervised_positions": 128
              },
              "mmlu_pro": {
                "n": 16,
                "mean_kl": 0.6279409043490887,
                "same_prefix_next_token_agreement": 0.8017241379310345,
                "supervised_positions": 116,
                "label_accuracy": {
                  "qwen_native": 0.3125,
                  "writer_only": 0.3125,
                  "writer_reader": 0.25
                }
              }
            },
            "macro_kl": 0.2517571879961906
          }
        },
        {
          "epoch": 4,
          "step": 1024,
          "validation": {
            "tasks": {
              "openbookqa": {
                "n": 16,
                "mean_kl": 0.08937330276239663,
                "same_prefix_next_token_agreement": 0.9795918367346939,
                "supervised_positions": 98,
                "label_accuracy": {
                  "qwen_native": 0.8125,
                  "writer_only": 0.625,
                  "writer_reader": 0.625
                }
              },
              "arc_challenge": {
                "n": 16,
                "mean_kl": 0.05219782930362271,
                "same_prefix_next_token_agreement": 0.9916666666666667,
                "supervised_positions": 120,
                "label_accuracy": {
                  "qwen_native": 1.0,
                  "writer_only": 0.8125,
                  "writer_reader": 0.8125
                }
              },
              "gsm8k": {
                "n": 16,
                "mean_kl": 0.18463176197838038,
                "same_prefix_next_token_agreement": 0.90625,
                "supervised_positions": 128
              },
              "mmlu_pro": {
                "n": 16,
                "mean_kl": 0.6679957266896963,
                "same_prefix_next_token_agreement": 0.7586206896551724,
                "supervised_positions": 116,
                "label_accuracy": {
                  "qwen_native": 0.3125,
                  "writer_only": 0.3125,
                  "writer_reader": 0.25
                }
              }
            },
            "macro_kl": 0.248549655183524
          }
        }
      ],
      "memory": {
        "peak_pytorch_allocated_gib": 9.928777694702148,
        "peak_pytorch_reserved_gib": 10.841796875,
        "scope": "This process CUDA allocator since load_all; not whole-GPU nvidia-smi peak"
      },
      "manifest_sha256": "e3cdafb48dca164e6193671c5271cf1772772aecb5e8300c7ddbbac98d04ee41"
    },
    "reader_manifest_sha256": "e3cdafb48dca164e6193671c5271cf1772772aecb5e8300c7ddbbac98d04ee41",
    "parent_manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739",
    "caveats": [
      "MMLU-Pro remains the approved CUSTOM split and label-scoring extension, not untouched official test.",
      "Reader changes only continuation computation, never transmitted prefix KV. Future decoded KV can change indirectly.",
      "Native greedy prefixes and64-token training horizon are unchanged from parent UFD. Formal GSM8K cap remains4096."
    ]
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/logs/pipeline.log"
}
```


## ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/llama_native

```json
{
  "experiment": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/llama_native",
  "timestamp": "2026-10-21 09:35:18",
  "status": "completed",
  "config": {
    "name": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/llama_native",
    "seed": 1234,
    "ufd_root": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
    "ufd_manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739",
    "reader_manifest_sha256": "e3cdafb48dca164e6193671c5271cf1772772aecb5e8300c7ddbbac98d04ee41",
    "counts": {
      "openbookqa": 128,
      "arc_challenge": 128,
      "hellaswag": 64,
      "gsm8k": 32,
      "mmlu_pro": 128,
      "sciq": 64
    },
    "model": "Llama-3.2-3B-Instruct, frozen FP16 on V100, no Writer or Reader",
    "template": "Exact existing Sender rendered_full chat template from the audited row, fixed original template date, then same task-specific answer cue. No Qwen template or duplicate special tokens.",
    "mcq": "Pinned HeteroFold native_candidate_scores: label log-likelihood, no length normalization, same candidates and labels as existing experiment",
    "hellaswag": "Pinned HeteroFold native_candidate_scores: length-normalized continuation log-likelihood, not A/B/C/D generation",
    "gsm8k": "Pinned HeteroFold greedy_decode on complete Llama-native chat, same boxed math instruction, max4096 new tokens, reference EOS stopping and extract_numeric_answer/normalize_number. No legacy stop/parser.",
    "mmlu_pro": "Same approved CUSTOM heldout128 and ARC-style actual A-J label extension as UFD/Reader, NOT untouched full official test/5shotCoT",
    "comparison": "Reuse exact Reader per-sample four-arm baselines; add only llama_native. Preserve original summaries; write combined summary_with_llama_native.json at Reader root.",
    "checkpoint_selection": "No training or checkpoint selection: frozen native Llama only",
    "continuous_monitoring": false,
    "automatic_upload": false
  },
  "split_counts": {
    "openbookqa": 128,
    "arc_challenge": 128,
    "hellaswag": 64,
    "gsm8k": 32,
    "mmlu_pro": 128,
    "sciq": 64
  },
  "checkpoint_selection": "No training or checkpoint selection: frozen native Llama only",
  "results": {
    "experiment": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/llama_native",
    "status": "completed",
    "config": {
      "name": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/llama_native",
      "seed": 1234,
      "ufd_root": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
      "ufd_manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739",
      "reader_manifest_sha256": "e3cdafb48dca164e6193671c5271cf1772772aecb5e8300c7ddbbac98d04ee41",
      "counts": {
        "openbookqa": 128,
        "arc_challenge": 128,
        "hellaswag": 64,
        "gsm8k": 32,
        "mmlu_pro": 128,
        "sciq": 64
      },
      "model": "Llama-3.2-3B-Instruct, frozen FP16 on V100, no Writer or Reader",
      "template": "Exact existing Sender rendered_full chat template from the audited row, fixed original template date, then same task-specific answer cue. No Qwen template or duplicate special tokens.",
      "mcq": "Pinned HeteroFold native_candidate_scores: label log-likelihood, no length normalization, same candidates and labels as existing experiment",
      "hellaswag": "Pinned HeteroFold native_candidate_scores: length-normalized continuation log-likelihood, not A/B/C/D generation",
      "gsm8k": "Pinned HeteroFold greedy_decode on complete Llama-native chat, same boxed math instruction, max4096 new tokens, reference EOS stopping and extract_numeric_answer/normalize_number. No legacy stop/parser.",
      "mmlu_pro": "Same approved CUSTOM heldout128 and ARC-style actual A-J label extension as UFD/Reader, NOT untouched full official test/5shotCoT",
      "comparison": "Reuse exact Reader per-sample four-arm baselines; add only llama_native. Preserve original summaries; write combined summary_with_llama_native.json at Reader root.",
      "checkpoint_selection": "No training or checkpoint selection: frozen native Llama only",
      "continuous_monitoring": false,
      "automatic_upload": false
    },
    "tasks": {
      "openbookqa": {
        "metrics": {
          "qwen_native": {
            "n": 128,
            "correct": 103,
            "accuracy": 0.8046875
          },
          "old_mcq_stage_b": {
            "n": 128,
            "correct": 91,
            "accuracy": 0.7109375
          },
          "new_ufd_stage_b": {
            "n": 128,
            "correct": 84,
            "accuracy": 0.65625
          },
          "ufd_writer_qo_reader": {
            "n": 128,
            "correct": 86,
            "accuracy": 0.671875
          },
          "llama_native": {
            "n": 128,
            "correct": 97,
            "accuracy": 0.7578125
          }
        },
        "paired": {
          "qwen_native__vs__old_mcq_stage_b": {
            "both_correct": 82,
            "only_a": 21,
            "only_b": 9,
            "both_wrong": 16
          },
          "qwen_native__vs__new_ufd_stage_b": {
            "both_correct": 76,
            "only_a": 27,
            "only_b": 8,
            "both_wrong": 17
          },
          "qwen_native__vs__ufd_writer_qo_reader": {
            "both_correct": 78,
            "only_a": 25,
            "only_b": 8,
            "both_wrong": 17
          },
          "qwen_native__vs__llama_native": {
            "both_correct": 86,
            "only_a": 17,
            "only_b": 11,
            "both_wrong": 14
          },
          "old_mcq_stage_b__vs__new_ufd_stage_b": {
            "both_correct": 78,
            "only_a": 13,
            "only_b": 6,
            "both_wrong": 31
          },
          "old_mcq_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 80,
            "only_a": 11,
            "only_b": 6,
            "both_wrong": 31
          },
          "old_mcq_stage_b__vs__llama_native": {
            "both_correct": 86,
            "only_a": 5,
            "only_b": 11,
            "both_wrong": 26
          },
          "new_ufd_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 84,
            "only_a": 0,
            "only_b": 2,
            "both_wrong": 42
          },
          "new_ufd_stage_b__vs__llama_native": {
            "both_correct": 81,
            "only_a": 3,
            "only_b": 16,
            "both_wrong": 28
          },
          "ufd_writer_qo_reader__vs__llama_native": {
            "both_correct": 83,
            "only_a": 3,
            "only_b": 14,
            "both_wrong": 28
          }
        }
      },
      "arc_challenge": {
        "metrics": {
          "qwen_native": {
            "n": 128,
            "correct": 111,
            "accuracy": 0.8671875
          },
          "old_mcq_stage_b": {
            "n": 128,
            "correct": 94,
            "accuracy": 0.734375
          },
          "new_ufd_stage_b": {
            "n": 128,
            "correct": 83,
            "accuracy": 0.6484375
          },
          "ufd_writer_qo_reader": {
            "n": 128,
            "correct": 83,
            "accuracy": 0.6484375
          },
          "llama_native": {
            "n": 128,
            "correct": 99,
            "accuracy": 0.7734375
          }
        },
        "paired": {
          "qwen_native__vs__old_mcq_stage_b": {
            "both_correct": 86,
            "only_a": 25,
            "only_b": 8,
            "both_wrong": 9
          },
          "qwen_native__vs__new_ufd_stage_b": {
            "both_correct": 78,
            "only_a": 33,
            "only_b": 5,
            "both_wrong": 12
          },
          "qwen_native__vs__ufd_writer_qo_reader": {
            "both_correct": 78,
            "only_a": 33,
            "only_b": 5,
            "both_wrong": 12
          },
          "qwen_native__vs__llama_native": {
            "both_correct": 91,
            "only_a": 20,
            "only_b": 8,
            "both_wrong": 9
          },
          "old_mcq_stage_b__vs__new_ufd_stage_b": {
            "both_correct": 79,
            "only_a": 15,
            "only_b": 4,
            "both_wrong": 30
          },
          "old_mcq_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 81,
            "only_a": 13,
            "only_b": 2,
            "both_wrong": 32
          },
          "old_mcq_stage_b__vs__llama_native": {
            "both_correct": 85,
            "only_a": 9,
            "only_b": 14,
            "both_wrong": 20
          },
          "new_ufd_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 81,
            "only_a": 2,
            "only_b": 2,
            "both_wrong": 43
          },
          "new_ufd_stage_b__vs__llama_native": {
            "both_correct": 77,
            "only_a": 6,
            "only_b": 22,
            "both_wrong": 23
          },
          "ufd_writer_qo_reader__vs__llama_native": {
            "both_correct": 78,
            "only_a": 5,
            "only_b": 21,
            "both_wrong": 24
          }
        }
      },
      "hellaswag": {
        "metrics": {
          "qwen_native": {
            "n": 64,
            "correct": 23,
            "accuracy": 0.359375
          },
          "old_mcq_stage_b": {
            "n": 64,
            "correct": 21,
            "accuracy": 0.328125
          },
          "new_ufd_stage_b": {
            "n": 64,
            "correct": 21,
            "accuracy": 0.328125
          },
          "ufd_writer_qo_reader": {
            "n": 64,
            "correct": 21,
            "accuracy": 0.328125
          },
          "llama_native": {
            "n": 64,
            "correct": 29,
            "accuracy": 0.453125
          }
        },
        "paired": {
          "qwen_native__vs__old_mcq_stage_b": {
            "both_correct": 12,
            "only_a": 11,
            "only_b": 9,
            "both_wrong": 32
          },
          "qwen_native__vs__new_ufd_stage_b": {
            "both_correct": 12,
            "only_a": 11,
            "only_b": 9,
            "both_wrong": 32
          },
          "qwen_native__vs__ufd_writer_qo_reader": {
            "both_correct": 14,
            "only_a": 9,
            "only_b": 7,
            "both_wrong": 34
          },
          "qwen_native__vs__llama_native": {
            "both_correct": 18,
            "only_a": 5,
            "only_b": 11,
            "both_wrong": 30
          },
          "old_mcq_stage_b__vs__new_ufd_stage_b": {
            "both_correct": 13,
            "only_a": 8,
            "only_b": 8,
            "both_wrong": 35
          },
          "old_mcq_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 14,
            "only_a": 7,
            "only_b": 7,
            "both_wrong": 36
          },
          "old_mcq_stage_b__vs__llama_native": {
            "both_correct": 15,
            "only_a": 6,
            "only_b": 14,
            "both_wrong": 29
          },
          "new_ufd_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 19,
            "only_a": 2,
            "only_b": 2,
            "both_wrong": 41
          },
          "new_ufd_stage_b__vs__llama_native": {
            "both_correct": 15,
            "only_a": 6,
            "only_b": 14,
            "both_wrong": 29
          },
          "ufd_writer_qo_reader__vs__llama_native": {
            "both_correct": 15,
            "only_a": 6,
            "only_b": 14,
            "both_wrong": 29
          }
        }
      },
      "gsm8k": {
        "metrics": {
          "qwen_native": {
            "n": 32,
            "correct": 31,
            "accuracy": 0.96875,
            "mean_generated_tokens": 296.375,
            "cap_count": 0
          },
          "old_mcq_stage_b": {
            "n": 32,
            "correct": 0,
            "accuracy": 0.0,
            "mean_generated_tokens": 3114.1875,
            "cap_count": 23
          },
          "new_ufd_stage_b": {
            "n": 32,
            "correct": 9,
            "accuracy": 0.28125,
            "mean_generated_tokens": 444.78125,
            "cap_count": 1
          },
          "ufd_writer_qo_reader": {
            "n": 32,
            "correct": 15,
            "accuracy": 0.46875,
            "mean_generated_tokens": 353.46875,
            "cap_count": 0
          },
          "llama_native": {
            "n": 32,
            "correct": 23,
            "accuracy": 0.71875,
            "mean_generated_tokens": 217.8125,
            "cap_count": 0
          }
        },
        "paired": {
          "qwen_native__vs__old_mcq_stage_b": {
            "both_correct": 0,
            "only_a": 31,
            "only_b": 0,
            "both_wrong": 1
          },
          "qwen_native__vs__new_ufd_stage_b": {
            "both_correct": 9,
            "only_a": 22,
            "only_b": 0,
            "both_wrong": 1
          },
          "qwen_native__vs__ufd_writer_qo_reader": {
            "both_correct": 15,
            "only_a": 16,
            "only_b": 0,
            "both_wrong": 1
          },
          "qwen_native__vs__llama_native": {
            "both_correct": 22,
            "only_a": 9,
            "only_b": 1,
            "both_wrong": 0
          },
          "old_mcq_stage_b__vs__new_ufd_stage_b": {
            "both_correct": 0,
            "only_a": 0,
            "only_b": 9,
            "both_wrong": 23
          },
          "old_mcq_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 0,
            "only_a": 0,
            "only_b": 15,
            "both_wrong": 17
          },
          "old_mcq_stage_b__vs__llama_native": {
            "both_correct": 0,
            "only_a": 0,
            "only_b": 23,
            "both_wrong": 9
          },
          "new_ufd_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 9,
            "only_a": 0,
            "only_b": 6,
            "both_wrong": 17
          },
          "new_ufd_stage_b__vs__llama_native": {
            "both_correct": 8,
            "only_a": 1,
            "only_b": 15,
            "both_wrong": 8
          },
          "ufd_writer_qo_reader__vs__llama_native": {
            "both_correct": 11,
            "only_a": 4,
            "only_b": 12,
            "both_wrong": 5
          }
        }
      },
      "mmlu_pro": {
        "metrics": {
          "qwen_native": {
            "n": 128,
            "correct": 52,
            "accuracy": 0.40625
          },
          "old_mcq_stage_b": {
            "n": 128,
            "correct": 17,
            "accuracy": 0.1328125
          },
          "new_ufd_stage_b": {
            "n": 128,
            "correct": 24,
            "accuracy": 0.1875
          },
          "ufd_writer_qo_reader": {
            "n": 128,
            "correct": 26,
            "accuracy": 0.203125
          },
          "llama_native": {
            "n": 128,
            "correct": 40,
            "accuracy": 0.3125
          }
        },
        "paired": {
          "qwen_native__vs__old_mcq_stage_b": {
            "both_correct": 9,
            "only_a": 43,
            "only_b": 8,
            "both_wrong": 68
          },
          "qwen_native__vs__new_ufd_stage_b": {
            "both_correct": 16,
            "only_a": 36,
            "only_b": 8,
            "both_wrong": 68
          },
          "qwen_native__vs__ufd_writer_qo_reader": {
            "both_correct": 18,
            "only_a": 34,
            "only_b": 8,
            "both_wrong": 68
          },
          "qwen_native__vs__llama_native": {
            "both_correct": 24,
            "only_a": 28,
            "only_b": 16,
            "both_wrong": 60
          },
          "old_mcq_stage_b__vs__new_ufd_stage_b": {
            "both_correct": 10,
            "only_a": 7,
            "only_b": 14,
            "both_wrong": 97
          },
          "old_mcq_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 9,
            "only_a": 8,
            "only_b": 17,
            "both_wrong": 94
          },
          "old_mcq_stage_b__vs__llama_native": {
            "both_correct": 5,
            "only_a": 12,
            "only_b": 35,
            "both_wrong": 76
          },
          "new_ufd_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 23,
            "only_a": 1,
            "only_b": 3,
            "both_wrong": 101
          },
          "new_ufd_stage_b__vs__llama_native": {
            "both_correct": 14,
            "only_a": 10,
            "only_b": 26,
            "both_wrong": 78
          },
          "ufd_writer_qo_reader__vs__llama_native": {
            "both_correct": 14,
            "only_a": 12,
            "only_b": 26,
            "both_wrong": 76
          }
        }
      },
      "sciq": {
        "metrics": {
          "qwen_native": {
            "n": 64,
            "correct": 57,
            "accuracy": 0.890625
          },
          "old_mcq_stage_b": {
            "n": 64,
            "correct": 59,
            "accuracy": 0.921875
          },
          "new_ufd_stage_b": {
            "n": 64,
            "correct": 58,
            "accuracy": 0.90625
          },
          "ufd_writer_qo_reader": {
            "n": 64,
            "correct": 59,
            "accuracy": 0.921875
          },
          "llama_native": {
            "n": 64,
            "correct": 58,
            "accuracy": 0.90625
          }
        },
        "paired": {
          "qwen_native__vs__old_mcq_stage_b": {
            "both_correct": 57,
            "only_a": 0,
            "only_b": 2,
            "both_wrong": 5
          },
          "qwen_native__vs__new_ufd_stage_b": {
            "both_correct": 53,
            "only_a": 4,
            "only_b": 5,
            "both_wrong": 2
          },
          "qwen_native__vs__ufd_writer_qo_reader": {
            "both_correct": 55,
            "only_a": 2,
            "only_b": 4,
            "both_wrong": 3
          },
          "qwen_native__vs__llama_native": {
            "both_correct": 56,
            "only_a": 1,
            "only_b": 2,
            "both_wrong": 5
          },
          "old_mcq_stage_b__vs__new_ufd_stage_b": {
            "both_correct": 55,
            "only_a": 4,
            "only_b": 3,
            "both_wrong": 2
          },
          "old_mcq_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 57,
            "only_a": 2,
            "only_b": 2,
            "both_wrong": 3
          },
          "old_mcq_stage_b__vs__llama_native": {
            "both_correct": 57,
            "only_a": 2,
            "only_b": 1,
            "both_wrong": 4
          },
          "new_ufd_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 57,
            "only_a": 1,
            "only_b": 2,
            "both_wrong": 4
          },
          "new_ufd_stage_b__vs__llama_native": {
            "both_correct": 55,
            "only_a": 3,
            "only_b": 3,
            "both_wrong": 3
          },
          "ufd_writer_qo_reader__vs__llama_native": {
            "both_correct": 57,
            "only_a": 2,
            "only_b": 1,
            "both_wrong": 4
          }
        }
      }
    },
    "manifest_sha256": "256d7c03a2dc4bdca9b52ac68affa811933b8bf193bdf3c665f147f999259875",
    "raw_results": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/llama_native/evaluation/per_sample.jsonl",
    "caveats": [
      "MMLU-Pro uses same approved custom heldout split and label protocol.",
      "Native Llama uses its OWN recorded Sender chat template, not Qwen template.",
      "Other four arms reused unchanged on the exact same rows; only llama_native newly computed."
    ]
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/llama_native/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/llama_native/logs/pipeline.log"
}
```


## ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/heterofold_baseline

```json
{
  "experiment": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/heterofold_baseline",
  "timestamp": "2026-10-21 10:00:29",
  "status": "completed",
  "config": {
    "name": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/heterofold_baseline",
    "seed": 1234,
    "ufd_root": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
    "ufd_manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739",
    "reader_manifest_sha256": "e3cdafb48dca164e6193671c5271cf1772772aecb5e8300c7ddbbac98d04ee41",
    "llama_native_manifest_sha256": "256d7c03a2dc4bdca9b52ac68affa811933b8bf193bdf3c665f147f999259875",
    "map_path": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/maps/llama3_to_qwen4/heterofold.pt",
    "map_sha256": "dc2ea778b2b6ea4d6c74c678548946f59fb1e5970f935be3ecc5e4fb58ac718e",
    "counts": {
      "openbookqa": 128,
      "arc_challenge": 128,
      "hellaswag": 64,
      "gsm8k": 32,
      "mmlu_pro": 128,
      "sciq": 64
    },
    "newly_evaluate": [
      "openbookqa",
      "mmlu_pro",
      "sciq"
    ],
    "reuse_existing": [
      "arc_challenge",
      "hellaswag",
      "gsm8k"
    ],
    "protocol": "Original HeteroFold RAW Sender prompt (no Sender chat/BOS), capture_sender, unchanged folded map, build_receiver_cache with Qwen native chat prefix, assistant+task cue and pinned candidate scoring. Do NOT use our Full-Sync/native token0 anchor or post-k_norm Writer convention.",
    "execution": "Both frozen FP16 Llama/Qwen resident on the available V100; map FP16; per-row raw Sender KV staged on CPU RAM and discarded. No persistent KV dataset, no training and no generation for the3 new label-scored tasks.",
    "mmlu_pro": "Existing approved CUSTOM heldout128, actual A-J labels using the same ARC-format extension; NOT untouched official full test/CoT.",
    "scoring": "Exactly same prompts/candidates/gold as UFD/Reader/Llama Native. ARC/OBQA/SciQ/MMLU label likelihood, HellaSwag length-normalized continuation. Existing GSM8K32 retains reference4096 greedy and boxed numeric parser.",
    "checkpoint_selection": "One fixed existing HeteroFold map, no calibration/retraining/test-based selection",
    "continuous_monitoring": false,
    "automatic_upload": false
  },
  "split_counts": {
    "openbookqa": 128,
    "arc_challenge": 128,
    "hellaswag": 64,
    "gsm8k": 32,
    "mmlu_pro": 128,
    "sciq": 64
  },
  "checkpoint_selection": "One fixed existing HeteroFold map, no calibration/retraining/test-based selection",
  "results": {
    "experiment": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/heterofold_baseline",
    "status": "completed",
    "config": {
      "name": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/heterofold_baseline",
      "seed": 1234,
      "ufd_root": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
      "ufd_manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739",
      "reader_manifest_sha256": "e3cdafb48dca164e6193671c5271cf1772772aecb5e8300c7ddbbac98d04ee41",
      "llama_native_manifest_sha256": "256d7c03a2dc4bdca9b52ac68affa811933b8bf193bdf3c665f147f999259875",
      "map_path": "/hy-tmp/yezhe/异构模型/heterofold_official_l3_to_q4_seed20260825/maps/llama3_to_qwen4/heterofold.pt",
      "map_sha256": "dc2ea778b2b6ea4d6c74c678548946f59fb1e5970f935be3ecc5e4fb58ac718e",
      "counts": {
        "openbookqa": 128,
        "arc_challenge": 128,
        "hellaswag": 64,
        "gsm8k": 32,
        "mmlu_pro": 128,
        "sciq": 64
      },
      "newly_evaluate": [
        "openbookqa",
        "mmlu_pro",
        "sciq"
      ],
      "reuse_existing": [
        "arc_challenge",
        "hellaswag",
        "gsm8k"
      ],
      "protocol": "Original HeteroFold RAW Sender prompt (no Sender chat/BOS), capture_sender, unchanged folded map, build_receiver_cache with Qwen native chat prefix, assistant+task cue and pinned candidate scoring. Do NOT use our Full-Sync/native token0 anchor or post-k_norm Writer convention.",
      "execution": "Both frozen FP16 Llama/Qwen resident on the available V100; map FP16; per-row raw Sender KV staged on CPU RAM and discarded. No persistent KV dataset, no training and no generation for the3 new label-scored tasks.",
      "mmlu_pro": "Existing approved CUSTOM heldout128, actual A-J labels using the same ARC-format extension; NOT untouched official full test/CoT.",
      "scoring": "Exactly same prompts/candidates/gold as UFD/Reader/Llama Native. ARC/OBQA/SciQ/MMLU label likelihood, HellaSwag length-normalized continuation. Existing GSM8K32 retains reference4096 greedy and boxed numeric parser.",
      "checkpoint_selection": "One fixed existing HeteroFold map, no calibration/retraining/test-based selection",
      "continuous_monitoring": false,
      "automatic_upload": false
    },
    "tasks": {
      "openbookqa": {
        "metrics": {
          "qwen_native": {
            "n": 128,
            "correct": 103,
            "accuracy": 0.8046875
          },
          "old_mcq_stage_b": {
            "n": 128,
            "correct": 91,
            "accuracy": 0.7109375
          },
          "new_ufd_stage_b": {
            "n": 128,
            "correct": 84,
            "accuracy": 0.65625
          },
          "ufd_writer_qo_reader": {
            "n": 128,
            "correct": 86,
            "accuracy": 0.671875
          },
          "llama_native": {
            "n": 128,
            "correct": 97,
            "accuracy": 0.7578125
          },
          "heterofold": {
            "n": 128,
            "correct": 71,
            "accuracy": 0.5546875
          }
        },
        "paired": {
          "qwen_native__vs__old_mcq_stage_b": {
            "both_correct": 82,
            "only_a": 21,
            "only_b": 9,
            "both_wrong": 16
          },
          "qwen_native__vs__new_ufd_stage_b": {
            "both_correct": 76,
            "only_a": 27,
            "only_b": 8,
            "both_wrong": 17
          },
          "qwen_native__vs__ufd_writer_qo_reader": {
            "both_correct": 78,
            "only_a": 25,
            "only_b": 8,
            "both_wrong": 17
          },
          "qwen_native__vs__llama_native": {
            "both_correct": 86,
            "only_a": 17,
            "only_b": 11,
            "both_wrong": 14
          },
          "qwen_native__vs__heterofold": {
            "both_correct": 66,
            "only_a": 37,
            "only_b": 5,
            "both_wrong": 20
          },
          "old_mcq_stage_b__vs__new_ufd_stage_b": {
            "both_correct": 78,
            "only_a": 13,
            "only_b": 6,
            "both_wrong": 31
          },
          "old_mcq_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 80,
            "only_a": 11,
            "only_b": 6,
            "both_wrong": 31
          },
          "old_mcq_stage_b__vs__llama_native": {
            "both_correct": 86,
            "only_a": 5,
            "only_b": 11,
            "both_wrong": 26
          },
          "old_mcq_stage_b__vs__heterofold": {
            "both_correct": 66,
            "only_a": 25,
            "only_b": 5,
            "both_wrong": 32
          },
          "new_ufd_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 84,
            "only_a": 0,
            "only_b": 2,
            "both_wrong": 42
          },
          "new_ufd_stage_b__vs__llama_native": {
            "both_correct": 81,
            "only_a": 3,
            "only_b": 16,
            "both_wrong": 28
          },
          "new_ufd_stage_b__vs__heterofold": {
            "both_correct": 65,
            "only_a": 19,
            "only_b": 6,
            "both_wrong": 38
          },
          "ufd_writer_qo_reader__vs__llama_native": {
            "both_correct": 83,
            "only_a": 3,
            "only_b": 14,
            "both_wrong": 28
          },
          "ufd_writer_qo_reader__vs__heterofold": {
            "both_correct": 66,
            "only_a": 20,
            "only_b": 5,
            "both_wrong": 37
          },
          "llama_native__vs__heterofold": {
            "both_correct": 67,
            "only_a": 30,
            "only_b": 4,
            "both_wrong": 27
          }
        }
      },
      "arc_challenge": {
        "metrics": {
          "qwen_native": {
            "n": 128,
            "correct": 111,
            "accuracy": 0.8671875
          },
          "old_mcq_stage_b": {
            "n": 128,
            "correct": 94,
            "accuracy": 0.734375
          },
          "new_ufd_stage_b": {
            "n": 128,
            "correct": 83,
            "accuracy": 0.6484375
          },
          "ufd_writer_qo_reader": {
            "n": 128,
            "correct": 83,
            "accuracy": 0.6484375
          },
          "llama_native": {
            "n": 128,
            "correct": 99,
            "accuracy": 0.7734375
          },
          "heterofold": {
            "n": 128,
            "correct": 73,
            "accuracy": 0.5703125
          }
        },
        "paired": {
          "qwen_native__vs__old_mcq_stage_b": {
            "both_correct": 86,
            "only_a": 25,
            "only_b": 8,
            "both_wrong": 9
          },
          "qwen_native__vs__new_ufd_stage_b": {
            "both_correct": 78,
            "only_a": 33,
            "only_b": 5,
            "both_wrong": 12
          },
          "qwen_native__vs__ufd_writer_qo_reader": {
            "both_correct": 78,
            "only_a": 33,
            "only_b": 5,
            "both_wrong": 12
          },
          "qwen_native__vs__llama_native": {
            "both_correct": 91,
            "only_a": 20,
            "only_b": 8,
            "both_wrong": 9
          },
          "qwen_native__vs__heterofold": {
            "both_correct": 70,
            "only_a": 41,
            "only_b": 3,
            "both_wrong": 14
          },
          "old_mcq_stage_b__vs__new_ufd_stage_b": {
            "both_correct": 79,
            "only_a": 15,
            "only_b": 4,
            "both_wrong": 30
          },
          "old_mcq_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 81,
            "only_a": 13,
            "only_b": 2,
            "both_wrong": 32
          },
          "old_mcq_stage_b__vs__llama_native": {
            "both_correct": 85,
            "only_a": 9,
            "only_b": 14,
            "both_wrong": 20
          },
          "old_mcq_stage_b__vs__heterofold": {
            "both_correct": 69,
            "only_a": 25,
            "only_b": 4,
            "both_wrong": 30
          },
          "new_ufd_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 81,
            "only_a": 2,
            "only_b": 2,
            "both_wrong": 43
          },
          "new_ufd_stage_b__vs__llama_native": {
            "both_correct": 77,
            "only_a": 6,
            "only_b": 22,
            "both_wrong": 23
          },
          "new_ufd_stage_b__vs__heterofold": {
            "both_correct": 64,
            "only_a": 19,
            "only_b": 9,
            "both_wrong": 36
          },
          "ufd_writer_qo_reader__vs__llama_native": {
            "both_correct": 78,
            "only_a": 5,
            "only_b": 21,
            "both_wrong": 24
          },
          "ufd_writer_qo_reader__vs__heterofold": {
            "both_correct": 65,
            "only_a": 18,
            "only_b": 8,
            "both_wrong": 37
          },
          "llama_native__vs__heterofold": {
            "both_correct": 68,
            "only_a": 31,
            "only_b": 5,
            "both_wrong": 24
          }
        }
      },
      "hellaswag": {
        "metrics": {
          "qwen_native": {
            "n": 64,
            "correct": 23,
            "accuracy": 0.359375
          },
          "old_mcq_stage_b": {
            "n": 64,
            "correct": 21,
            "accuracy": 0.328125
          },
          "new_ufd_stage_b": {
            "n": 64,
            "correct": 21,
            "accuracy": 0.328125
          },
          "ufd_writer_qo_reader": {
            "n": 64,
            "correct": 21,
            "accuracy": 0.328125
          },
          "llama_native": {
            "n": 64,
            "correct": 29,
            "accuracy": 0.453125
          },
          "heterofold": {
            "n": 64,
            "correct": 26,
            "accuracy": 0.40625
          }
        },
        "paired": {
          "qwen_native__vs__old_mcq_stage_b": {
            "both_correct": 12,
            "only_a": 11,
            "only_b": 9,
            "both_wrong": 32
          },
          "qwen_native__vs__new_ufd_stage_b": {
            "both_correct": 12,
            "only_a": 11,
            "only_b": 9,
            "both_wrong": 32
          },
          "qwen_native__vs__ufd_writer_qo_reader": {
            "both_correct": 14,
            "only_a": 9,
            "only_b": 7,
            "both_wrong": 34
          },
          "qwen_native__vs__llama_native": {
            "both_correct": 18,
            "only_a": 5,
            "only_b": 11,
            "both_wrong": 30
          },
          "qwen_native__vs__heterofold": {
            "both_correct": 19,
            "only_a": 4,
            "only_b": 7,
            "both_wrong": 34
          },
          "old_mcq_stage_b__vs__new_ufd_stage_b": {
            "both_correct": 13,
            "only_a": 8,
            "only_b": 8,
            "both_wrong": 35
          },
          "old_mcq_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 14,
            "only_a": 7,
            "only_b": 7,
            "both_wrong": 36
          },
          "old_mcq_stage_b__vs__llama_native": {
            "both_correct": 15,
            "only_a": 6,
            "only_b": 14,
            "both_wrong": 29
          },
          "old_mcq_stage_b__vs__heterofold": {
            "both_correct": 13,
            "only_a": 8,
            "only_b": 13,
            "both_wrong": 30
          },
          "new_ufd_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 19,
            "only_a": 2,
            "only_b": 2,
            "both_wrong": 41
          },
          "new_ufd_stage_b__vs__llama_native": {
            "both_correct": 15,
            "only_a": 6,
            "only_b": 14,
            "both_wrong": 29
          },
          "new_ufd_stage_b__vs__heterofold": {
            "both_correct": 15,
            "only_a": 6,
            "only_b": 11,
            "both_wrong": 32
          },
          "ufd_writer_qo_reader__vs__llama_native": {
            "both_correct": 15,
            "only_a": 6,
            "only_b": 14,
            "both_wrong": 29
          },
          "ufd_writer_qo_reader__vs__heterofold": {
            "both_correct": 15,
            "only_a": 6,
            "only_b": 11,
            "both_wrong": 32
          },
          "llama_native__vs__heterofold": {
            "both_correct": 20,
            "only_a": 9,
            "only_b": 6,
            "both_wrong": 29
          }
        }
      },
      "gsm8k": {
        "metrics": {
          "qwen_native": {
            "n": 32,
            "correct": 31,
            "accuracy": 0.96875,
            "mean_generated_tokens": 296.375,
            "cap_count": 0
          },
          "old_mcq_stage_b": {
            "n": 32,
            "correct": 0,
            "accuracy": 0.0,
            "mean_generated_tokens": 3114.1875,
            "cap_count": 23
          },
          "new_ufd_stage_b": {
            "n": 32,
            "correct": 9,
            "accuracy": 0.28125,
            "mean_generated_tokens": 444.78125,
            "cap_count": 1
          },
          "ufd_writer_qo_reader": {
            "n": 32,
            "correct": 15,
            "accuracy": 0.46875,
            "mean_generated_tokens": 353.46875,
            "cap_count": 0
          },
          "llama_native": {
            "n": 32,
            "correct": 23,
            "accuracy": 0.71875,
            "mean_generated_tokens": 217.8125,
            "cap_count": 0
          },
          "heterofold": {
            "n": 32,
            "correct": 17,
            "accuracy": 0.53125,
            "mean_generated_tokens": 568.4375,
            "cap_count": 2
          }
        },
        "paired": {
          "qwen_native__vs__old_mcq_stage_b": {
            "both_correct": 0,
            "only_a": 31,
            "only_b": 0,
            "both_wrong": 1
          },
          "qwen_native__vs__new_ufd_stage_b": {
            "both_correct": 9,
            "only_a": 22,
            "only_b": 0,
            "both_wrong": 1
          },
          "qwen_native__vs__ufd_writer_qo_reader": {
            "both_correct": 15,
            "only_a": 16,
            "only_b": 0,
            "both_wrong": 1
          },
          "qwen_native__vs__llama_native": {
            "both_correct": 22,
            "only_a": 9,
            "only_b": 1,
            "both_wrong": 0
          },
          "qwen_native__vs__heterofold": {
            "both_correct": 16,
            "only_a": 15,
            "only_b": 1,
            "both_wrong": 0
          },
          "old_mcq_stage_b__vs__new_ufd_stage_b": {
            "both_correct": 0,
            "only_a": 0,
            "only_b": 9,
            "both_wrong": 23
          },
          "old_mcq_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 0,
            "only_a": 0,
            "only_b": 15,
            "both_wrong": 17
          },
          "old_mcq_stage_b__vs__llama_native": {
            "both_correct": 0,
            "only_a": 0,
            "only_b": 23,
            "both_wrong": 9
          },
          "old_mcq_stage_b__vs__heterofold": {
            "both_correct": 0,
            "only_a": 0,
            "only_b": 17,
            "both_wrong": 15
          },
          "new_ufd_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 9,
            "only_a": 0,
            "only_b": 6,
            "both_wrong": 17
          },
          "new_ufd_stage_b__vs__llama_native": {
            "both_correct": 8,
            "only_a": 1,
            "only_b": 15,
            "both_wrong": 8
          },
          "new_ufd_stage_b__vs__heterofold": {
            "both_correct": 7,
            "only_a": 2,
            "only_b": 10,
            "both_wrong": 13
          },
          "ufd_writer_qo_reader__vs__llama_native": {
            "both_correct": 11,
            "only_a": 4,
            "only_b": 12,
            "both_wrong": 5
          },
          "ufd_writer_qo_reader__vs__heterofold": {
            "both_correct": 9,
            "only_a": 6,
            "only_b": 8,
            "both_wrong": 9
          },
          "llama_native__vs__heterofold": {
            "both_correct": 14,
            "only_a": 9,
            "only_b": 3,
            "both_wrong": 6
          }
        }
      },
      "mmlu_pro": {
        "metrics": {
          "qwen_native": {
            "n": 128,
            "correct": 52,
            "accuracy": 0.40625
          },
          "old_mcq_stage_b": {
            "n": 128,
            "correct": 17,
            "accuracy": 0.1328125
          },
          "new_ufd_stage_b": {
            "n": 128,
            "correct": 24,
            "accuracy": 0.1875
          },
          "ufd_writer_qo_reader": {
            "n": 128,
            "correct": 26,
            "accuracy": 0.203125
          },
          "llama_native": {
            "n": 128,
            "correct": 40,
            "accuracy": 0.3125
          },
          "heterofold": {
            "n": 128,
            "correct": 33,
            "accuracy": 0.2578125
          }
        },
        "paired": {
          "qwen_native__vs__old_mcq_stage_b": {
            "both_correct": 9,
            "only_a": 43,
            "only_b": 8,
            "both_wrong": 68
          },
          "qwen_native__vs__new_ufd_stage_b": {
            "both_correct": 16,
            "only_a": 36,
            "only_b": 8,
            "both_wrong": 68
          },
          "qwen_native__vs__ufd_writer_qo_reader": {
            "both_correct": 18,
            "only_a": 34,
            "only_b": 8,
            "both_wrong": 68
          },
          "qwen_native__vs__llama_native": {
            "both_correct": 24,
            "only_a": 28,
            "only_b": 16,
            "both_wrong": 60
          },
          "qwen_native__vs__heterofold": {
            "both_correct": 25,
            "only_a": 27,
            "only_b": 8,
            "both_wrong": 68
          },
          "old_mcq_stage_b__vs__new_ufd_stage_b": {
            "both_correct": 10,
            "only_a": 7,
            "only_b": 14,
            "both_wrong": 97
          },
          "old_mcq_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 9,
            "only_a": 8,
            "only_b": 17,
            "both_wrong": 94
          },
          "old_mcq_stage_b__vs__llama_native": {
            "both_correct": 5,
            "only_a": 12,
            "only_b": 35,
            "both_wrong": 76
          },
          "old_mcq_stage_b__vs__heterofold": {
            "both_correct": 7,
            "only_a": 10,
            "only_b": 26,
            "both_wrong": 85
          },
          "new_ufd_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 23,
            "only_a": 1,
            "only_b": 3,
            "both_wrong": 101
          },
          "new_ufd_stage_b__vs__llama_native": {
            "both_correct": 14,
            "only_a": 10,
            "only_b": 26,
            "both_wrong": 78
          },
          "new_ufd_stage_b__vs__heterofold": {
            "both_correct": 16,
            "only_a": 8,
            "only_b": 17,
            "both_wrong": 87
          },
          "ufd_writer_qo_reader__vs__llama_native": {
            "both_correct": 14,
            "only_a": 12,
            "only_b": 26,
            "both_wrong": 76
          },
          "ufd_writer_qo_reader__vs__heterofold": {
            "both_correct": 16,
            "only_a": 10,
            "only_b": 17,
            "both_wrong": 85
          },
          "llama_native__vs__heterofold": {
            "both_correct": 19,
            "only_a": 21,
            "only_b": 14,
            "both_wrong": 74
          }
        }
      },
      "sciq": {
        "metrics": {
          "qwen_native": {
            "n": 64,
            "correct": 57,
            "accuracy": 0.890625
          },
          "old_mcq_stage_b": {
            "n": 64,
            "correct": 59,
            "accuracy": 0.921875
          },
          "new_ufd_stage_b": {
            "n": 64,
            "correct": 58,
            "accuracy": 0.90625
          },
          "ufd_writer_qo_reader": {
            "n": 64,
            "correct": 59,
            "accuracy": 0.921875
          },
          "llama_native": {
            "n": 64,
            "correct": 58,
            "accuracy": 0.90625
          },
          "heterofold": {
            "n": 64,
            "correct": 58,
            "accuracy": 0.90625
          }
        },
        "paired": {
          "qwen_native__vs__old_mcq_stage_b": {
            "both_correct": 57,
            "only_a": 0,
            "only_b": 2,
            "both_wrong": 5
          },
          "qwen_native__vs__new_ufd_stage_b": {
            "both_correct": 53,
            "only_a": 4,
            "only_b": 5,
            "both_wrong": 2
          },
          "qwen_native__vs__ufd_writer_qo_reader": {
            "both_correct": 55,
            "only_a": 2,
            "only_b": 4,
            "both_wrong": 3
          },
          "qwen_native__vs__llama_native": {
            "both_correct": 56,
            "only_a": 1,
            "only_b": 2,
            "both_wrong": 5
          },
          "qwen_native__vs__heterofold": {
            "both_correct": 54,
            "only_a": 3,
            "only_b": 4,
            "both_wrong": 3
          },
          "old_mcq_stage_b__vs__new_ufd_stage_b": {
            "both_correct": 55,
            "only_a": 4,
            "only_b": 3,
            "both_wrong": 2
          },
          "old_mcq_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 57,
            "only_a": 2,
            "only_b": 2,
            "both_wrong": 3
          },
          "old_mcq_stage_b__vs__llama_native": {
            "both_correct": 57,
            "only_a": 2,
            "only_b": 1,
            "both_wrong": 4
          },
          "old_mcq_stage_b__vs__heterofold": {
            "both_correct": 56,
            "only_a": 3,
            "only_b": 2,
            "both_wrong": 3
          },
          "new_ufd_stage_b__vs__ufd_writer_qo_reader": {
            "both_correct": 57,
            "only_a": 1,
            "only_b": 2,
            "both_wrong": 4
          },
          "new_ufd_stage_b__vs__llama_native": {
            "both_correct": 55,
            "only_a": 3,
            "only_b": 3,
            "both_wrong": 3
          },
          "new_ufd_stage_b__vs__heterofold": {
            "both_correct": 56,
            "only_a": 2,
            "only_b": 2,
            "both_wrong": 4
          },
          "ufd_writer_qo_reader__vs__llama_native": {
            "both_correct": 57,
            "only_a": 2,
            "only_b": 1,
            "both_wrong": 4
          },
          "ufd_writer_qo_reader__vs__heterofold": {
            "both_correct": 57,
            "only_a": 2,
            "only_b": 1,
            "both_wrong": 4
          },
          "llama_native__vs__heterofold": {
            "both_correct": 55,
            "only_a": 3,
            "only_b": 3,
            "both_wrong": 3
          }
        }
      }
    },
    "manifest_sha256": "1667de46d31d5cc58cf2823c8584d0b7220f51e353ae2f2c555566031adc4817",
    "cuda_allocator_peak_gib": 14.096534729003906,
    "caveats": [
      "Same evaluation rows and Receiver scoring, but HeteroFold RAW Sender/cache interface and calibration differ from our methods: NOT same-budget architecture ablation.",
      "MMLU-Pro remains approved custom heldout and A-J label extension, not official test/CoT.",
      "Only OBQA/MMLU-Pro/SciQ HeteroFold outputs are new; ARC128/Hella64/GSM32 reused after input/identity verification."
    ]
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/heterofold_baseline/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/heterofold_baseline/logs/pipeline.log"
}
```


## ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234

```json
{
  "experiment": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234",
  "timestamp": "2026-10-21 19:59:36",
  "status": "failed_or_incomplete",
  "seed": 1234,
  "config": {
    "name": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234",
    "seed": 1234,
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
    "status": "failed_or_incomplete",
    "type": "TypeError",
    "message": "dict() got multiple values for keyword argument 'tokens'"
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234/logs/pipeline.log"
}
```


## ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234

```json
{
  "experiment": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234",
  "timestamp": "2026-10-21 20:50:52",
  "status": "failed_or_incomplete",
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
    "status": "failed_or_incomplete",
    "type": "FileNotFoundError",
    "message": "[Errno 2] No such file or directory: '/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234/manifest.json'"
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234/logs/pipeline.log"
}
```


## ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234

```json
{
  "experiment": "ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234",
  "timestamp": "2026-10-23 00:08:23",
  "status": "reader_training_completed_evaluation_pending",
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
    "status": "completed",
    "updates": 256,
    "checkpoint": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234/runs/reader/step256.pt",
    "checkpoint_sha256": "e3c89a31de1f2f5d0fc21426a46f865b93d65ae8a45a60ade5fe4a92f6b31154",
    "selection": "Fixed last step256 primary; no test/validation checkpoint selection or hyperparameter search.",
    "manifest_sha256": "2a573a3d8713e474ae5c7c053772c7842aedf095ae229dbce1ae8f3bdfeebee0"
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/ufd_qo_lora_reader_r16_full36_2048x4_l3_to_q4_seed1234/reward_aligned_reader_gsm8k_pilot_seed1234/logs/pipeline.log"
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


## grouped_headmix4_l3_to_q4_seed1234

```json
{
  "experiment": "grouped_headmix4_l3_to_q4_seed1234",
  "time": "2026-10-23 01:57:47",
  "status": "stage_a_completed",
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
    "status": "completed",
    "stage": "a",
    "steps": 512,
    "history": [
      {
        "epoch": 1,
        "step": 256,
        "validation": {
          "total": 0.5073760609113833,
          "k_nmse": 0.05093518327055335,
          "v_nmse": 0.24438724361777986,
          "k_cosine": 0.945762161478342,
          "v_cosine": 0.8421842141578963
        }
      },
      {
        "epoch": 2,
        "step": 512,
        "validation": {
          "total": 0.44577715137256746,
          "k_nmse": 0.04240127331567578,
          "v_nmse": 0.21497247569720912,
          "k_cosine": 0.9536803563569265,
          "v_cosine": 0.8579162468922732
        }
      }
    ],
    "selected": "best.pt"
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/grouped_headmix4_l3_to_q4_seed1234/evaluation/per_sample.jsonl"
}
```


## grouped_headmix4_l3_to_q4_seed1234

```json
{
  "experiment": "grouped_headmix4_l3_to_q4_seed1234",
  "time": "2026-10-23 04:08:46",
  "status": "stage_b_completed",
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
    "status": "completed",
    "stage": "b",
    "steps": 1024,
    "history": [
      {
        "epoch": 1,
        "step": 256,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "kl": 0.12346929294290021
            },
            "arc_challenge": {
              "n": 16,
              "kl": 0.20394760952331126
            },
            "gsm8k": {
              "n": 16,
              "kl": 0.2544833542779088
            },
            "mmlu_pro": {
              "n": 16,
              "kl": 1.6780101768672466
            }
          },
          "macro_kl": 0.5649776084028417
        }
      },
      {
        "epoch": 2,
        "step": 512,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "kl": 0.15944780597055797
            },
            "arc_challenge": {
              "n": 16,
              "kl": 0.15045209118397906
            },
            "gsm8k": {
              "n": 16,
              "kl": 0.28642644407227635
            },
            "mmlu_pro": {
              "n": 16,
              "kl": 0.9299563504755497
            }
          },
          "macro_kl": 0.38157067292559077
        }
      },
      {
        "epoch": 3,
        "step": 768,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "kl": 0.14819869661005214
            },
            "arc_challenge": {
              "n": 16,
              "kl": 0.18328311713412404
            },
            "gsm8k": {
              "n": 16,
              "kl": 0.8324835933744907
            },
            "mmlu_pro": {
              "n": 16,
              "kl": 1.150621353648603
            }
          },
          "macro_kl": 0.5786466901918175
        }
      },
      {
        "epoch": 4,
        "step": 1024,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "kl": 0.18599062762223184
            },
            "arc_challenge": {
              "n": 16,
              "kl": 0.12720273654849734
            },
            "gsm8k": {
              "n": 16,
              "kl": 0.23620908265002072
            },
            "mmlu_pro": {
              "n": 16,
              "kl": 0.9601299688220024
            }
          },
          "macro_kl": 0.3773831039106881
        }
      }
    ],
    "selected": "last.pt"
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/grouped_headmix4_l3_to_q4_seed1234/evaluation/per_sample.jsonl"
}
```


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


## grouped_headmix4_residual64_l3_to_q4_seed1234

```json
{
  "experiment": "grouped_headmix4_residual64_l3_to_q4_seed1234",
  "time": "2026-10-23 13:04:31",
  "status": "stage_b_completed",
  "manifest": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_l3_to_q4_seed1234/manifest.json",
  "config": {
    "name": "grouped_headmix4_residual64_l3_to_q4_seed1234",
    "seed": 1234,
    "ufd_parent": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
    "mcq_parent": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
    "wait_for_experiment": "/hy-tmp/yezhe/异构模型/grouped_headmix4_l3_to_q4_seed1234",
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
    "stage_a_selection": "Reuse frozen grouped Stage-A best.pt; do not use previous fully tuned Stage-B",
    "stage_b_selection": "Fixed epoch4 step1024 last checkpoint, never test-selected",
    "stage_b_objective": "Same UFD full-vocabulary forward KL and mixed prefixes. Freeze Stage-A translator and both LLMs. Train only zero-up per-target-layer CROSS-HEAD rank64 K/V residuals; no Reader.",
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
      "Same grouped Stage-A and Stage-B data/objective/budget as full-translator run; trainable scope and residual parameterization change together.",
      "Stage-B residuals mix heads within each target layer; source-layer grouping remains in the frozen translator.",
      "Reuse historical evaluation subsets and Full-Sync accuracy interface; no zero-prefill latency claim.",
      "MMLU-Pro remains a custom split."
    ],
    "base_parent": "/hy-tmp/yezhe/异构模型/grouped_headmix4_l3_to_q4_seed1234",
    "base_parent_manifest_sha256": "3f487169ddc521bb6ac5e803344786521a9901c4b95659fc5864083175bd12c9",
    "base_stage_a_sha256": "a9318f4b2f1d9e7981e0e50e2dc88654f0baaf89a080a0746017555c48ba5c84",
    "residual_rank": 64,
    "expected_trainable_parameters": 9437184,
    "stage_a_retrained": false
  },
  "results": {
    "status": "completed",
    "stage": "b",
    "steps": 1024,
    "history": [
      {
        "epoch": 1,
        "step": 256,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "kl": 0.11173932859674096
            },
            "arc_challenge": {
              "n": 16,
              "kl": 0.14676392765250057
            },
            "gsm8k": {
              "n": 16,
              "kl": 0.24674388207495213
            },
            "mmlu_pro": {
              "n": 16,
              "kl": 1.030614223331213
            }
          },
          "macro_kl": 0.38396534041385166
        }
      },
      {
        "epoch": 2,
        "step": 512,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "kl": 0.1348447557829786
            },
            "arc_challenge": {
              "n": 16,
              "kl": 0.11653592679067515
            },
            "gsm8k": {
              "n": 16,
              "kl": 0.26886977115646005
            },
            "mmlu_pro": {
              "n": 16,
              "kl": 1.0135492095723748
            }
          },
          "macro_kl": 0.38344991582562216
        }
      },
      {
        "epoch": 3,
        "step": 768,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "kl": 0.12636066728737205
            },
            "arc_challenge": {
              "n": 16,
              "kl": 0.10534414689755067
            },
            "gsm8k": {
              "n": 16,
              "kl": 0.2189960773102939
            },
            "mmlu_pro": {
              "n": 16,
              "kl": 0.7957085408270359
            }
          },
          "macro_kl": 0.31160235808056314
        }
      },
      {
        "epoch": 4,
        "step": 1024,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "kl": 0.10594114020932466
            },
            "arc_challenge": {
              "n": 16,
              "kl": 0.07262166512373369
            },
            "gsm8k": {
              "n": 16,
              "kl": 0.20145857660099864
            },
            "mmlu_pro": {
              "n": 16,
              "kl": 0.8005681857466698
            }
          },
          "macro_kl": 0.2951473919201817
        }
      }
    ],
    "base_bitwise_unchanged": true,
    "trainable_parameters": 9437184,
    "frozen_stage_a_sha256": "a9318f4b2f1d9e7981e0e50e2dc88654f0baaf89a080a0746017555c48ba5c84",
    "selected": "last.pt"
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_l3_to_q4_seed1234/evaluation/per_sample.jsonl"
}
```


## grouped_headmix4_residual64_l3_to_q4_seed1234

```json
{
  "experiment": "grouped_headmix4_residual64_l3_to_q4_seed1234",
  "time": "2026-10-23 13:26:02",
  "status": "completed",
  "manifest": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_l3_to_q4_seed1234/manifest.json",
  "config": {
    "name": "grouped_headmix4_residual64_l3_to_q4_seed1234",
    "seed": 1234,
    "ufd_parent": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
    "mcq_parent": "/hy-tmp/yezhe/异构模型/mcq_heterofold_chat_fullsync_l3_to_q4_seed1234",
    "wait_for_experiment": "/hy-tmp/yezhe/异构模型/grouped_headmix4_l3_to_q4_seed1234",
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
    "stage_a_selection": "Reuse frozen grouped Stage-A best.pt; do not use previous fully tuned Stage-B",
    "stage_b_selection": "Fixed epoch4 step1024 last checkpoint, never test-selected",
    "stage_b_objective": "Same UFD full-vocabulary forward KL and mixed prefixes. Freeze Stage-A translator and both LLMs. Train only zero-up per-target-layer CROSS-HEAD rank64 K/V residuals; no Reader.",
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
      "Same grouped Stage-A and Stage-B data/objective/budget as full-translator run; trainable scope and residual parameterization change together.",
      "Stage-B residuals mix heads within each target layer; source-layer grouping remains in the frozen translator.",
      "Reuse historical evaluation subsets and Full-Sync accuracy interface; no zero-prefill latency claim.",
      "MMLU-Pro remains a custom split."
    ],
    "base_parent": "/hy-tmp/yezhe/异构模型/grouped_headmix4_l3_to_q4_seed1234",
    "base_parent_manifest_sha256": "3f487169ddc521bb6ac5e803344786521a9901c4b95659fc5864083175bd12c9",
    "base_stage_a_sha256": "a9318f4b2f1d9e7981e0e50e2dc88654f0baaf89a080a0746017555c48ba5c84",
    "residual_rank": 64,
    "expected_trainable_parameters": 9437184,
    "stage_a_retrained": false
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
      "grouped_full_stage_b": {
        "n": 128,
        "correct": 84,
        "accuracy": 0.65625
      },
      "grouped_residual": {
        "n": 128,
        "correct": 88,
        "accuracy": 0.6875
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
      "grouped_full_stage_b": {
        "n": 128,
        "correct": 81,
        "accuracy": 0.6328125
      },
      "grouped_residual": {
        "n": 128,
        "correct": 82,
        "accuracy": 0.640625
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
      "grouped_full_stage_b": {
        "n": 64,
        "correct": 20,
        "accuracy": 0.3125
      },
      "grouped_residual": {
        "n": 64,
        "correct": 22,
        "accuracy": 0.34375
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
      "grouped_full_stage_b": {
        "n": 32,
        "correct": 8,
        "accuracy": 0.25
      },
      "grouped_residual": {
        "n": 32,
        "correct": 10,
        "accuracy": 0.3125
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
      "grouped_full_stage_b": {
        "n": 128,
        "correct": 44,
        "accuracy": 0.34375
      },
      "grouped_residual": {
        "n": 128,
        "correct": 33,
        "accuracy": 0.2578125
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
      "grouped_full_stage_b": {
        "n": 64,
        "correct": 57,
        "accuracy": 0.890625
      },
      "grouped_residual": {
        "n": 64,
        "correct": 56,
        "accuracy": 0.875
      }
    }
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_l3_to_q4_seed1234/evaluation/per_sample.jsonl"
}
```


## grouped_headmix4_reader_r16_answerkl_l3_to_q4_seed1234

```json
{
  "experiment": "grouped_headmix4_reader_r16_answerkl_l3_to_q4_seed1234",
  "timestamp": "2026-10-23 17:03:55",
  "status": "reader_training_completed_evaluation_pending",
  "config": {
    "name": "grouped_headmix4_reader_r16_answerkl_l3_to_q4_seed1234",
    "seed": 1234,
    "parent": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
    "parent_manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739",
    "rank": 16,
    "alpha": 16,
    "dropout": 0,
    "layers": 36,
    "targets": [
      "q_proj",
      "o_proj"
    ],
    "expected_trainable_parameters": 7667712,
    "epochs": 4,
    "expected_updates": 1024,
    "effective_batch": 8,
    "learning_rate": 0.0001,
    "weight_decay": 0,
    "gradient_clip": 30,
    "checkpoint_every_steps": 64,
    "validation_every_steps": 256,
    "monitor_validation_per_task": 16,
    "prefix_max_new_tokens": 64,
    "supervised_positions_per_example_max": 8,
    "train_counts": {
      "openbookqa": 512,
      "arc_challenge": 512,
      "gsm8k": 512,
      "mmlu_pro": 512
    },
    "formal_counts": {
      "openbookqa": 128,
      "arc_challenge": 128,
      "hellaswag": 64,
      "gsm8k": 32,
      "mmlu_pro": 128,
      "sciq": 64
    },
    "loss": "Full-vocabulary forward KL(Native||StageA+Reader), temperature1, FP32 loss. MCQ: ONLY answer-label prediction boundary, all labels audited single-token. GSM8K: unchanged up-to8 positions in max64-token same-prefix continuation. Equal example weighting. No gold CE or reward.",
    "prefix_sampling": "Same parent balanced question order. MCQ has no generated answer prefix or rollout. GSM8K epoch1 native prefixes; epochs2-4 one native and one online Student example per batch. Teacher always scores identical histories with Reader disabled.",
    "interface": "Frozen grouped Stage-A best.pt ONLY, no residual and no previous Reader weights. Full-Sync native anchor, native tail, chat and absolute positions unchanged. Reader disabled for native prefill/teacher; enabled for translated-cache suffix and continuation.",
    "selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
    "validation_task_metrics": "On fixed16 validation rows/task: KL plus same-prefix next-token agreement for all4 tasks; label accuracy on3 MCQ tasks. Full GSM8K free-generation accuracy only in final test, using unchanged4096 cap.",
    "baseline_reuse": "Reuse checked residual experiment per-sample Native/oldUFD/groupedFull/groupedResidual results. Compute Stage-A-only and Stage-A+Reader on identical held-out rows.",
    "queue_min_free_gpu_gib": 20,
    "queue_retry_seconds": 300,
    "disk_reserve_gib": 20,
    "continuous_monitoring": false,
    "automatic_upload": false,
    "base_parent": "/hy-tmp/yezhe/异构模型/grouped_headmix4_l3_to_q4_seed1234",
    "base_parent_manifest_sha256": "3f487169ddc521bb6ac5e803344786521a9901c4b95659fc5864083175bd12c9",
    "base_stage_a_sha256": "a9318f4b2f1d9e7981e0e50e2dc88654f0baaf89a080a0746017555c48ba5c84",
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
    "expected_translator_parameters": 134217728,
    "chunk_tokens": 64,
    "residual_parent": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_l3_to_q4_seed1234",
    "residual_parent_manifest_sha256": "318e7ab9a56381345602cb82c4f1b78087c16e094901ea897f93555c864932f3",
    "wait_for_experiment": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_l3_to_q4_seed1234",
    "evaluation_tasks": [
      "openbookqa",
      "arc_challenge",
      "hellaswag",
      "gsm8k",
      "mmlu_pro",
      "sciq"
    ],
    "caveats": [
      "Training scope and MCQ supervised positions differ from previous residual Stage B; not a pure scope-only ablation.",
      "MMLU-Pro uses existing custom train/validation/test split from official test parquet, not untouched official test.",
      "GSM8K training horizon64, up to8 supervised positions; formal generation cap4096.",
      "Historical test subsets have been inspected before; do not use them for checkpoint selection.",
      "Native full-body prefill is used to obtain teacher and anchor; accuracy experiment, no prefill-free latency claim."
    ]
  },
  "checkpoint_selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
  "results": {
    "status": "completed",
    "epochs": 4,
    "updates": 1024,
    "selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
    "checkpoint": "/hy-tmp/yezhe/异构模型/grouped_headmix4_reader_r16_answerkl_l3_to_q4_seed1234/runs/reader/epoch4_step1024.pt",
    "checkpoint_sha256": "3a352ba425d3d5964e26381593099256e19300e01bc433f6a37fe99444bb1947",
    "validation": [
      {
        "epoch": 1,
        "step": 256,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.312242526182672,
              "same_prefix_next_token_agreement": 0.8125,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.8125,
                "stage_a_only": 0.625,
                "stage_a_reader": 0.6875
              }
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.43983244319679216,
              "same_prefix_next_token_agreement": 0.9375,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 1.0,
                "stage_a_only": 0.6875,
                "stage_a_reader": 0.75
              }
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.3658923301845789,
              "same_prefix_next_token_agreement": 0.8125,
              "supervised_positions": 128
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 1.5323241442092694,
              "same_prefix_next_token_agreement": 0.5,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.3125,
                "stage_a_only": 0.0625,
                "stage_a_reader": 0.375
              }
            }
          },
          "macro_kl": 0.6625728609433281
        }
      },
      {
        "epoch": 2,
        "step": 512,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.3823943884926848,
              "same_prefix_next_token_agreement": 0.75,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.8125,
                "stage_a_only": 0.625,
                "stage_a_reader": 0.75
              }
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.3380934148590313,
              "same_prefix_next_token_agreement": 0.875,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 1.0,
                "stage_a_only": 0.6875,
                "stage_a_reader": 0.75
              }
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.3010891806334257,
              "same_prefix_next_token_agreement": 0.8515625,
              "supervised_positions": 128
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 1.2106766487995628,
              "same_prefix_next_token_agreement": 0.5,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.3125,
                "stage_a_only": 0.0625,
                "stage_a_reader": 0.3125
              }
            }
          },
          "macro_kl": 0.5580634081961762
        }
      },
      {
        "epoch": 3,
        "step": 768,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.35512947326060385,
              "same_prefix_next_token_agreement": 0.9375,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.8125,
                "stage_a_only": 0.625,
                "stage_a_reader": 0.75
              }
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.44158744654851034,
              "same_prefix_next_token_agreement": 0.875,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 1.0,
                "stage_a_only": 0.6875,
                "stage_a_reader": 0.8125
              }
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.26402737968601286,
              "same_prefix_next_token_agreement": 0.890625,
              "supervised_positions": 128
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 1.20833721710369,
              "same_prefix_next_token_agreement": 0.5625,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.3125,
                "stage_a_only": 0.0625,
                "stage_a_reader": 0.3125
              }
            }
          },
          "macro_kl": 0.5672703791497042
        }
      },
      {
        "epoch": 4,
        "step": 1024,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.21726004368247231,
              "same_prefix_next_token_agreement": 0.9375,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.8125,
                "stage_a_only": 0.625,
                "stage_a_reader": 0.75
              }
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.24407396434253315,
              "same_prefix_next_token_agreement": 0.9375,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 1.0,
                "stage_a_only": 0.6875,
                "stage_a_reader": 0.875
              }
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.2717722888337448,
              "same_prefix_next_token_agreement": 0.8671875,
              "supervised_positions": 128
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 1.2064578047138639,
              "same_prefix_next_token_agreement": 0.625,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.3125,
                "stage_a_only": 0.0625,
                "stage_a_reader": 0.375
              }
            }
          },
          "macro_kl": 0.48489102539315354
        }
      }
    ],
    "memory": {
      "peak_pytorch_allocated_gib": 9.100572109222412,
      "peak_pytorch_reserved_gib": 9.568359375,
      "scope": "This process CUDA allocator since load_all; not whole-GPU nvidia-smi peak"
    },
    "manifest_sha256": "6d381f294ffe6d7c24b955dd564b6d1938296c8723d277dcc451a2c13057d96f",
    "stage_a_bitwise_unchanged": true
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/grouped_headmix4_reader_r16_answerkl_l3_to_q4_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/grouped_headmix4_reader_r16_answerkl_l3_to_q4_seed1234/logs/pipeline.log"
}
```


## grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234

```json
{
  "experiment": "grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234",
  "timestamp": "2026-10-23 22:36:53",
  "status": "reader_training_completed_evaluation_pending",
  "config": {
    "name": "grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234",
    "seed": 1234,
    "parent": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
    "parent_manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739",
    "rank": 16,
    "alpha": 16,
    "dropout": 0,
    "layers": 36,
    "targets": [
      "q_proj",
      "o_proj"
    ],
    "expected_trainable_parameters": 7667712,
    "epochs": 4,
    "expected_updates": 1024,
    "effective_batch": 8,
    "learning_rate": 0.0001,
    "weight_decay": 0,
    "gradient_clip": 30,
    "checkpoint_every_steps": 64,
    "validation_every_steps": 256,
    "monitor_validation_per_task": 16,
    "prefix_max_new_tokens": 64,
    "supervised_positions_per_example_max": 8,
    "train_counts": {
      "openbookqa": 512,
      "arc_challenge": 512,
      "gsm8k": 512,
      "mmlu_pro": 512
    },
    "formal_counts": {
      "openbookqa": 128,
      "arc_challenge": 128,
      "hellaswag": 64,
      "gsm8k": 32,
      "mmlu_pro": 128,
      "sciq": 64
    },
    "loss": "Full-vocabulary forward KL(Native||StageA+Residual+Reader), temperature1, FP32 loss. MCQ: ONLY answer-label prediction boundary, all labels audited single-token. GSM8K: unchanged up-to8 positions in max64-token same-prefix continuation. Equal example weighting. No gold CE or reward.",
    "prefix_sampling": "Same parent balanced question order. MCQ has no generated answer prefix or rollout. GSM8K epoch1 native prefixes; epochs2-4 one native and one online Student example per batch. Teacher always scores identical histories with Reader disabled.",
    "interface": "Frozen grouped Stage-A best.pt + frozen cross-head residual Stage-B last.pt, both pinned by SHA256. Fresh all36 Q/O LoRA; no old Reader weights. Native teacher/prefill Reader disabled; only translated-cache suffix/continuation enables Reader.",
    "selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
    "validation_task_metrics": "On fixed16 validation rows/task: KL plus same-prefix next-token agreement for all4 tasks; label accuracy on3 MCQ tasks. Full GSM8K free-generation accuracy only in final test, using unchanged4096 cap.",
    "baseline_reuse": "Reuse fixed residual per-sample Native/oldUFD/groupedFull/groupedResidual outputs. Compute ONLY grouped_residual_reader. Compare to paused StageA+Reader on identity-checked, available matched samples; never resume the previous evaluation automatically.",
    "queue_min_free_gpu_gib": 20,
    "queue_retry_seconds": 300,
    "disk_reserve_gib": 20,
    "continuous_monitoring": false,
    "automatic_upload": false,
    "base_parent": "/hy-tmp/yezhe/异构模型/grouped_headmix4_l3_to_q4_seed1234",
    "base_parent_manifest_sha256": "3f487169ddc521bb6ac5e803344786521a9901c4b95659fc5864083175bd12c9",
    "base_stage_a_sha256": "a9318f4b2f1d9e7981e0e50e2dc88654f0baaf89a080a0746017555c48ba5c84",
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
    "expected_translator_parameters": 134217728,
    "chunk_tokens": 64,
    "residual_parent": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_l3_to_q4_seed1234",
    "residual_parent_manifest_sha256": "318e7ab9a56381345602cb82c4f1b78087c16e094901ea897f93555c864932f3",
    "wait_for_experiment": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_l3_to_q4_seed1234",
    "evaluation_tasks": [
      "openbookqa",
      "arc_challenge",
      "hellaswag",
      "gsm8k",
      "mmlu_pro",
      "sciq"
    ],
    "caveats": [
      "The extra Reader adds training budget beyond Stage A+B; this is an incremental-gain experiment, not an equal-budget comparison.",
      "MCQ answer-boundary objective matches recent A+Reader experiment; differs from original residual training positions.",
      "Previous A+Reader evaluation is paused: comparison uses only available matched samples and explicitly reports n/expected.",
      "MMLU-Pro remains the custom split from official test parquet, not untouched full official test.",
      "GSM8K training cap64 and up to8 positions; formal greedy evaluation cap4096.",
      "Historical test subsets have been inspected; fixed final checkpoint only, no test selection.",
      "Existing Full-Sync interface uses Native body prefill for teacher/anchor; no prefill-free latency claim."
    ],
    "residual_rank": 64,
    "expected_residual_parameters": 9437184,
    "frozen_stage_b_sha256": "7e9f602d08409574c8ebe56242a08ba2c6861032b5ed70570a807c5777d775f2",
    "previous_reader_parent": "/hy-tmp/yezhe/异构模型/grouped_headmix4_reader_r16_answerkl_l3_to_q4_seed1234",
    "previous_reader_manifest_sha256": "6d381f294ffe6d7c24b955dd564b6d1938296c8723d277dcc451a2c13057d96f"
  },
  "checkpoint_selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
  "results": {
    "status": "completed",
    "epochs": 4,
    "updates": 1024,
    "selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
    "checkpoint": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234/runs/reader/epoch4_step1024.pt",
    "checkpoint_sha256": "96cde5b41f1a15434d24ef57f8eaf34e3af89af109c0b5d6cf378e60ab577314",
    "validation": [
      {
        "epoch": 1,
        "step": 256,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.29942402364395093,
              "same_prefix_next_token_agreement": 0.9375,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.8125,
                "stage_ab_only": 0.625,
                "stage_ab_reader": 0.625
              }
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.3376882626544102,
              "same_prefix_next_token_agreement": 0.9375,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 1.0,
                "stage_ab_only": 0.75,
                "stage_ab_reader": 0.6875
              }
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.1873050353024155,
              "same_prefix_next_token_agreement": 0.8984375,
              "supervised_positions": 128
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 1.0935497938553453,
              "same_prefix_next_token_agreement": 0.5,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.3125,
                "stage_ab_only": 0.3125,
                "stage_ab_reader": 0.3125
              }
            }
          },
          "macro_kl": 0.4794917788640305
        }
      },
      {
        "epoch": 2,
        "step": 512,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.36091980057244655,
              "same_prefix_next_token_agreement": 0.8125,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.8125,
                "stage_ab_only": 0.625,
                "stage_ab_reader": 0.6875
              }
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.3823155289537681,
              "same_prefix_next_token_agreement": 0.875,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 1.0,
                "stage_ab_only": 0.75,
                "stage_ab_reader": 0.75
              }
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.18423763429746032,
              "same_prefix_next_token_agreement": 0.890625,
              "supervised_positions": 128
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 1.032063302482129,
              "same_prefix_next_token_agreement": 0.625,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.3125,
                "stage_ab_only": 0.3125,
                "stage_ab_reader": 0.375
              }
            }
          },
          "macro_kl": 0.489884066576451
        }
      },
      {
        "epoch": 3,
        "step": 768,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.36019871625467204,
              "same_prefix_next_token_agreement": 0.875,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.8125,
                "stage_ab_only": 0.625,
                "stage_ab_reader": 0.625
              }
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.7243947678070981,
              "same_prefix_next_token_agreement": 0.8125,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 1.0,
                "stage_ab_only": 0.75,
                "stage_ab_reader": 0.8125
              }
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.1738615392241627,
              "same_prefix_next_token_agreement": 0.8984375,
              "supervised_positions": 128
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 1.2310237512429012,
              "same_prefix_next_token_agreement": 0.625,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.3125,
                "stage_ab_only": 0.3125,
                "stage_ab_reader": 0.25
              }
            }
          },
          "macro_kl": 0.6223696936322085
        }
      },
      {
        "epoch": 4,
        "step": 1024,
        "validation": {
          "tasks": {
            "openbookqa": {
              "n": 16,
              "mean_kl": 0.40170132838829886,
              "same_prefix_next_token_agreement": 0.875,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.8125,
                "stage_ab_only": 0.625,
                "stage_ab_reader": 0.625
              }
            },
            "arc_challenge": {
              "n": 16,
              "mean_kl": 0.26070858562889043,
              "same_prefix_next_token_agreement": 0.875,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 1.0,
                "stage_ab_only": 0.75,
                "stage_ab_reader": 0.875
              }
            },
            "gsm8k": {
              "n": 16,
              "mean_kl": 0.18696373759303242,
              "same_prefix_next_token_agreement": 0.890625,
              "supervised_positions": 128
            },
            "mmlu_pro": {
              "n": 16,
              "mean_kl": 1.22758460240766,
              "same_prefix_next_token_agreement": 0.5625,
              "supervised_positions": 16,
              "label_accuracy": {
                "qwen_native": 0.3125,
                "stage_ab_only": 0.3125,
                "stage_ab_reader": 0.25
              }
            }
          },
          "macro_kl": 0.5192395635044704
        }
      }
    ],
    "memory": {
      "peak_pytorch_allocated_gib": 9.131556987762451,
      "peak_pytorch_reserved_gib": 9.6015625,
      "scope": "This process CUDA allocator since load_all; not whole-GPU nvidia-smi peak"
    },
    "manifest_sha256": "9472f20d101851631929e5fc60de19dd6f9e59f8f51b5eb10694c0d4ddc47e10",
    "stage_a_and_b_bitwise_unchanged": true
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234/logs/pipeline.log"
}
```


## grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234

```json
{
  "experiment": "grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234",
  "timestamp": "2026-10-23 22:58:19",
  "status": "completed",
  "config": {
    "name": "grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234",
    "seed": 1234,
    "parent": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
    "parent_manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739",
    "rank": 16,
    "alpha": 16,
    "dropout": 0,
    "layers": 36,
    "targets": [
      "q_proj",
      "o_proj"
    ],
    "expected_trainable_parameters": 7667712,
    "epochs": 4,
    "expected_updates": 1024,
    "effective_batch": 8,
    "learning_rate": 0.0001,
    "weight_decay": 0,
    "gradient_clip": 30,
    "checkpoint_every_steps": 64,
    "validation_every_steps": 256,
    "monitor_validation_per_task": 16,
    "prefix_max_new_tokens": 64,
    "supervised_positions_per_example_max": 8,
    "train_counts": {
      "openbookqa": 512,
      "arc_challenge": 512,
      "gsm8k": 512,
      "mmlu_pro": 512
    },
    "formal_counts": {
      "openbookqa": 128,
      "arc_challenge": 128,
      "hellaswag": 64,
      "gsm8k": 32,
      "mmlu_pro": 128,
      "sciq": 64
    },
    "loss": "Full-vocabulary forward KL(Native||StageA+Residual+Reader), temperature1, FP32 loss. MCQ: ONLY answer-label prediction boundary, all labels audited single-token. GSM8K: unchanged up-to8 positions in max64-token same-prefix continuation. Equal example weighting. No gold CE or reward.",
    "prefix_sampling": "Same parent balanced question order. MCQ has no generated answer prefix or rollout. GSM8K epoch1 native prefixes; epochs2-4 one native and one online Student example per batch. Teacher always scores identical histories with Reader disabled.",
    "interface": "Frozen grouped Stage-A best.pt + frozen cross-head residual Stage-B last.pt, both pinned by SHA256. Fresh all36 Q/O LoRA; no old Reader weights. Native teacher/prefill Reader disabled; only translated-cache suffix/continuation enables Reader.",
    "selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
    "validation_task_metrics": "On fixed16 validation rows/task: KL plus same-prefix next-token agreement for all4 tasks; label accuracy on3 MCQ tasks. Full GSM8K free-generation accuracy only in final test, using unchanged4096 cap.",
    "baseline_reuse": "Reuse fixed residual per-sample Native/oldUFD/groupedFull/groupedResidual outputs. Compute ONLY grouped_residual_reader. Compare to paused StageA+Reader on identity-checked, available matched samples; never resume the previous evaluation automatically.",
    "queue_min_free_gpu_gib": 20,
    "queue_retry_seconds": 300,
    "disk_reserve_gib": 20,
    "continuous_monitoring": false,
    "automatic_upload": false,
    "base_parent": "/hy-tmp/yezhe/异构模型/grouped_headmix4_l3_to_q4_seed1234",
    "base_parent_manifest_sha256": "3f487169ddc521bb6ac5e803344786521a9901c4b95659fc5864083175bd12c9",
    "base_stage_a_sha256": "a9318f4b2f1d9e7981e0e50e2dc88654f0baaf89a080a0746017555c48ba5c84",
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
    "expected_translator_parameters": 134217728,
    "chunk_tokens": 64,
    "residual_parent": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_l3_to_q4_seed1234",
    "residual_parent_manifest_sha256": "318e7ab9a56381345602cb82c4f1b78087c16e094901ea897f93555c864932f3",
    "wait_for_experiment": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_l3_to_q4_seed1234",
    "evaluation_tasks": [
      "openbookqa",
      "arc_challenge",
      "hellaswag",
      "gsm8k",
      "mmlu_pro",
      "sciq"
    ],
    "caveats": [
      "The extra Reader adds training budget beyond Stage A+B; this is an incremental-gain experiment, not an equal-budget comparison.",
      "MCQ answer-boundary objective matches recent A+Reader experiment; differs from original residual training positions.",
      "Previous A+Reader evaluation is paused: comparison uses only available matched samples and explicitly reports n/expected.",
      "MMLU-Pro remains the custom split from official test parquet, not untouched full official test.",
      "GSM8K training cap64 and up to8 positions; formal greedy evaluation cap4096.",
      "Historical test subsets have been inspected; fixed final checkpoint only, no test selection.",
      "Existing Full-Sync interface uses Native body prefill for teacher/anchor; no prefill-free latency claim."
    ],
    "residual_rank": 64,
    "expected_residual_parameters": 9437184,
    "frozen_stage_b_sha256": "7e9f602d08409574c8ebe56242a08ba2c6861032b5ed70570a807c5777d775f2",
    "previous_reader_parent": "/hy-tmp/yezhe/异构模型/grouped_headmix4_reader_r16_answerkl_l3_to_q4_seed1234",
    "previous_reader_manifest_sha256": "6d381f294ffe6d7c24b955dd564b6d1938296c8723d277dcc451a2c13057d96f"
  },
  "checkpoint_selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
  "results": {
    "experiment": "grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234",
    "status": "completed",
    "config": {
      "name": "grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234",
      "seed": 1234,
      "parent": "/hy-tmp/yezhe/异构模型/ufd_mixedprefix_stageb_4data_2048x4_l3_to_q4_seed1234",
      "parent_manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739",
      "rank": 16,
      "alpha": 16,
      "dropout": 0,
      "layers": 36,
      "targets": [
        "q_proj",
        "o_proj"
      ],
      "expected_trainable_parameters": 7667712,
      "epochs": 4,
      "expected_updates": 1024,
      "effective_batch": 8,
      "learning_rate": 0.0001,
      "weight_decay": 0,
      "gradient_clip": 30,
      "checkpoint_every_steps": 64,
      "validation_every_steps": 256,
      "monitor_validation_per_task": 16,
      "prefix_max_new_tokens": 64,
      "supervised_positions_per_example_max": 8,
      "train_counts": {
        "openbookqa": 512,
        "arc_challenge": 512,
        "gsm8k": 512,
        "mmlu_pro": 512
      },
      "formal_counts": {
        "openbookqa": 128,
        "arc_challenge": 128,
        "hellaswag": 64,
        "gsm8k": 32,
        "mmlu_pro": 128,
        "sciq": 64
      },
      "loss": "Full-vocabulary forward KL(Native||StageA+Residual+Reader), temperature1, FP32 loss. MCQ: ONLY answer-label prediction boundary, all labels audited single-token. GSM8K: unchanged up-to8 positions in max64-token same-prefix continuation. Equal example weighting. No gold CE or reward.",
      "prefix_sampling": "Same parent balanced question order. MCQ has no generated answer prefix or rollout. GSM8K epoch1 native prefixes; epochs2-4 one native and one online Student example per batch. Teacher always scores identical histories with Reader disabled.",
      "interface": "Frozen grouped Stage-A best.pt + frozen cross-head residual Stage-B last.pt, both pinned by SHA256. Fresh all36 Q/O LoRA; no old Reader weights. Native teacher/prefill Reader disabled; only translated-cache suffix/continuation enables Reader.",
      "selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
      "validation_task_metrics": "On fixed16 validation rows/task: KL plus same-prefix next-token agreement for all4 tasks; label accuracy on3 MCQ tasks. Full GSM8K free-generation accuracy only in final test, using unchanged4096 cap.",
      "baseline_reuse": "Reuse fixed residual per-sample Native/oldUFD/groupedFull/groupedResidual outputs. Compute ONLY grouped_residual_reader. Compare to paused StageA+Reader on identity-checked, available matched samples; never resume the previous evaluation automatically.",
      "queue_min_free_gpu_gib": 20,
      "queue_retry_seconds": 300,
      "disk_reserve_gib": 20,
      "continuous_monitoring": false,
      "automatic_upload": false,
      "base_parent": "/hy-tmp/yezhe/异构模型/grouped_headmix4_l3_to_q4_seed1234",
      "base_parent_manifest_sha256": "3f487169ddc521bb6ac5e803344786521a9901c4b95659fc5864083175bd12c9",
      "base_stage_a_sha256": "a9318f4b2f1d9e7981e0e50e2dc88654f0baaf89a080a0746017555c48ba5c84",
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
      "expected_translator_parameters": 134217728,
      "chunk_tokens": 64,
      "residual_parent": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_l3_to_q4_seed1234",
      "residual_parent_manifest_sha256": "318e7ab9a56381345602cb82c4f1b78087c16e094901ea897f93555c864932f3",
      "wait_for_experiment": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_l3_to_q4_seed1234",
      "evaluation_tasks": [
        "openbookqa",
        "arc_challenge",
        "hellaswag",
        "gsm8k",
        "mmlu_pro",
        "sciq"
      ],
      "caveats": [
        "The extra Reader adds training budget beyond Stage A+B; this is an incremental-gain experiment, not an equal-budget comparison.",
        "MCQ answer-boundary objective matches recent A+Reader experiment; differs from original residual training positions.",
        "Previous A+Reader evaluation is paused: comparison uses only available matched samples and explicitly reports n/expected.",
        "MMLU-Pro remains the custom split from official test parquet, not untouched full official test.",
        "GSM8K training cap64 and up to8 positions; formal greedy evaluation cap4096.",
        "Historical test subsets have been inspected; fixed final checkpoint only, no test selection.",
        "Existing Full-Sync interface uses Native body prefill for teacher/anchor; no prefill-free latency claim."
      ],
      "residual_rank": 64,
      "expected_residual_parameters": 9437184,
      "frozen_stage_b_sha256": "7e9f602d08409574c8ebe56242a08ba2c6861032b5ed70570a807c5777d775f2",
      "previous_reader_parent": "/hy-tmp/yezhe/异构模型/grouped_headmix4_reader_r16_answerkl_l3_to_q4_seed1234",
      "previous_reader_manifest_sha256": "6d381f294ffe6d7c24b955dd564b6d1938296c8723d277dcc451a2c13057d96f"
    },
    "tasks": {
      "openbookqa": {
        "metrics": {
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
          "grouped_full_stage_b": {
            "n": 128,
            "correct": 84,
            "accuracy": 0.65625
          },
          "grouped_residual": {
            "n": 128,
            "correct": 88,
            "accuracy": 0.6875
          },
          "grouped_residual_reader": {
            "n": 128,
            "correct": 92,
            "accuracy": 0.71875
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 82,
            "reader_only": 10,
            "other_only": 21,
            "both_wrong": 15
          },
          "new_ufd_stage_b": {
            "both_correct": 82,
            "reader_only": 10,
            "other_only": 2,
            "both_wrong": 34
          },
          "grouped_full_stage_b": {
            "both_correct": 80,
            "reader_only": 12,
            "other_only": 4,
            "both_wrong": 32
          },
          "grouped_residual": {
            "both_correct": 83,
            "reader_only": 9,
            "other_only": 5,
            "both_wrong": 31
          }
        },
        "previous_reader_matched_subset": {
          "n": 128,
          "expected": 128,
          "complete": true,
          "note": "Only fixed matched rows saved before previous evaluation was paused; no extrapolation.",
          "correct": {
            "stage_a_only": 62,
            "stage_a_reader": 85,
            "stage_ab_reader": 92
          }
        }
      },
      "arc_challenge": {
        "metrics": {
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
          "grouped_full_stage_b": {
            "n": 128,
            "correct": 81,
            "accuracy": 0.6328125
          },
          "grouped_residual": {
            "n": 128,
            "correct": 82,
            "accuracy": 0.640625
          },
          "grouped_residual_reader": {
            "n": 128,
            "correct": 86,
            "accuracy": 0.671875
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 82,
            "reader_only": 4,
            "other_only": 29,
            "both_wrong": 13
          },
          "new_ufd_stage_b": {
            "both_correct": 77,
            "reader_only": 9,
            "other_only": 6,
            "both_wrong": 36
          },
          "grouped_full_stage_b": {
            "both_correct": 76,
            "reader_only": 10,
            "other_only": 5,
            "both_wrong": 37
          },
          "grouped_residual": {
            "both_correct": 80,
            "reader_only": 6,
            "other_only": 2,
            "both_wrong": 40
          }
        },
        "previous_reader_matched_subset": {
          "n": 128,
          "expected": 128,
          "complete": true,
          "note": "Only fixed matched rows saved before previous evaluation was paused; no extrapolation.",
          "correct": {
            "stage_a_only": 45,
            "stage_a_reader": 86,
            "stage_ab_reader": 86
          }
        }
      },
      "hellaswag": {
        "metrics": {
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
          "grouped_full_stage_b": {
            "n": 64,
            "correct": 20,
            "accuracy": 0.3125
          },
          "grouped_residual": {
            "n": 64,
            "correct": 22,
            "accuracy": 0.34375
          },
          "grouped_residual_reader": {
            "n": 64,
            "correct": 22,
            "accuracy": 0.34375
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 13,
            "reader_only": 9,
            "other_only": 10,
            "both_wrong": 32
          },
          "new_ufd_stage_b": {
            "both_correct": 19,
            "reader_only": 3,
            "other_only": 2,
            "both_wrong": 40
          },
          "grouped_full_stage_b": {
            "both_correct": 15,
            "reader_only": 7,
            "other_only": 5,
            "both_wrong": 37
          },
          "grouped_residual": {
            "both_correct": 21,
            "reader_only": 1,
            "other_only": 1,
            "both_wrong": 41
          }
        },
        "previous_reader_matched_subset": {
          "n": 64,
          "expected": 64,
          "complete": true,
          "note": "Only fixed matched rows saved before previous evaluation was paused; no extrapolation.",
          "correct": {
            "stage_a_only": 17,
            "stage_a_reader": 20,
            "stage_ab_reader": 22
          }
        }
      },
      "gsm8k": {
        "metrics": {
          "qwen_native": {
            "n": 32,
            "correct": 31,
            "accuracy": 0.96875,
            "mean_generated_tokens": 296.375,
            "cap_count": 0
          },
          "new_ufd_stage_b": {
            "n": 32,
            "correct": 9,
            "accuracy": 0.28125,
            "mean_generated_tokens": 444.78125,
            "cap_count": 1
          },
          "grouped_full_stage_b": {
            "n": 32,
            "correct": 8,
            "accuracy": 0.25,
            "mean_generated_tokens": 815.53125,
            "cap_count": 4
          },
          "grouped_residual": {
            "n": 32,
            "correct": 10,
            "accuracy": 0.3125,
            "mean_generated_tokens": 562.78125,
            "cap_count": 1
          },
          "grouped_residual_reader": {
            "n": 32,
            "correct": 14,
            "accuracy": 0.4375,
            "mean_generated_tokens": 467.84375,
            "cap_count": 1
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 14,
            "reader_only": 0,
            "other_only": 17,
            "both_wrong": 1
          },
          "new_ufd_stage_b": {
            "both_correct": 8,
            "reader_only": 6,
            "other_only": 1,
            "both_wrong": 17
          },
          "grouped_full_stage_b": {
            "both_correct": 3,
            "reader_only": 11,
            "other_only": 5,
            "both_wrong": 13
          },
          "grouped_residual": {
            "both_correct": 8,
            "reader_only": 6,
            "other_only": 2,
            "both_wrong": 16
          }
        },
        "previous_reader_matched_subset": {
          "n": 22,
          "expected": 32,
          "complete": false,
          "note": "Only fixed matched rows saved before previous evaluation was paused; no extrapolation.",
          "correct": {
            "stage_a_only": 0,
            "stage_a_reader": 4,
            "stage_ab_reader": 9
          }
        }
      },
      "mmlu_pro": {
        "metrics": {
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
          "grouped_full_stage_b": {
            "n": 128,
            "correct": 44,
            "accuracy": 0.34375
          },
          "grouped_residual": {
            "n": 128,
            "correct": 33,
            "accuracy": 0.2578125
          },
          "grouped_residual_reader": {
            "n": 128,
            "correct": 35,
            "accuracy": 0.2734375
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 23,
            "reader_only": 12,
            "other_only": 29,
            "both_wrong": 64
          },
          "new_ufd_stage_b": {
            "both_correct": 15,
            "reader_only": 20,
            "other_only": 9,
            "both_wrong": 84
          },
          "grouped_full_stage_b": {
            "both_correct": 25,
            "reader_only": 10,
            "other_only": 19,
            "both_wrong": 74
          },
          "grouped_residual": {
            "both_correct": 24,
            "reader_only": 11,
            "other_only": 9,
            "both_wrong": 84
          }
        },
        "previous_reader_matched_subset": {
          "n": 0,
          "expected": 128,
          "complete": false,
          "note": "Only fixed matched rows saved before previous evaluation was paused; no extrapolation."
        }
      },
      "sciq": {
        "metrics": {
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
          "grouped_full_stage_b": {
            "n": 64,
            "correct": 57,
            "accuracy": 0.890625
          },
          "grouped_residual": {
            "n": 64,
            "correct": 56,
            "accuracy": 0.875
          },
          "grouped_residual_reader": {
            "n": 64,
            "correct": 57,
            "accuracy": 0.890625
          }
        },
        "paired": {
          "qwen_native": {
            "both_correct": 54,
            "reader_only": 3,
            "other_only": 3,
            "both_wrong": 4
          },
          "new_ufd_stage_b": {
            "both_correct": 55,
            "reader_only": 2,
            "other_only": 3,
            "both_wrong": 4
          },
          "grouped_full_stage_b": {
            "both_correct": 55,
            "reader_only": 2,
            "other_only": 2,
            "both_wrong": 5
          },
          "grouped_residual": {
            "both_correct": 54,
            "reader_only": 3,
            "other_only": 2,
            "both_wrong": 5
          }
        },
        "previous_reader_matched_subset": {
          "n": 0,
          "expected": 64,
          "complete": false,
          "note": "Only fixed matched rows saved before previous evaluation was paused; no extrapolation."
        }
      }
    },
    "training": {
      "status": "completed",
      "epochs": 4,
      "updates": 1024,
      "selection": "Primary fixed epoch4/step1024. Keep epoch1/2/3/4 Reader checkpoints. Best macro validation KL checkpoint is diagnostic only, never test-selected.",
      "checkpoint": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234/runs/reader/epoch4_step1024.pt",
      "checkpoint_sha256": "96cde5b41f1a15434d24ef57f8eaf34e3af89af109c0b5d6cf378e60ab577314",
      "validation": [
        {
          "epoch": 1,
          "step": 256,
          "validation": {
            "tasks": {
              "openbookqa": {
                "n": 16,
                "mean_kl": 0.29942402364395093,
                "same_prefix_next_token_agreement": 0.9375,
                "supervised_positions": 16,
                "label_accuracy": {
                  "qwen_native": 0.8125,
                  "stage_ab_only": 0.625,
                  "stage_ab_reader": 0.625
                }
              },
              "arc_challenge": {
                "n": 16,
                "mean_kl": 0.3376882626544102,
                "same_prefix_next_token_agreement": 0.9375,
                "supervised_positions": 16,
                "label_accuracy": {
                  "qwen_native": 1.0,
                  "stage_ab_only": 0.75,
                  "stage_ab_reader": 0.6875
                }
              },
              "gsm8k": {
                "n": 16,
                "mean_kl": 0.1873050353024155,
                "same_prefix_next_token_agreement": 0.8984375,
                "supervised_positions": 128
              },
              "mmlu_pro": {
                "n": 16,
                "mean_kl": 1.0935497938553453,
                "same_prefix_next_token_agreement": 0.5,
                "supervised_positions": 16,
                "label_accuracy": {
                  "qwen_native": 0.3125,
                  "stage_ab_only": 0.3125,
                  "stage_ab_reader": 0.3125
                }
              }
            },
            "macro_kl": 0.4794917788640305
          }
        },
        {
          "epoch": 2,
          "step": 512,
          "validation": {
            "tasks": {
              "openbookqa": {
                "n": 16,
                "mean_kl": 0.36091980057244655,
                "same_prefix_next_token_agreement": 0.8125,
                "supervised_positions": 16,
                "label_accuracy": {
                  "qwen_native": 0.8125,
                  "stage_ab_only": 0.625,
                  "stage_ab_reader": 0.6875
                }
              },
              "arc_challenge": {
                "n": 16,
                "mean_kl": 0.3823155289537681,
                "same_prefix_next_token_agreement": 0.875,
                "supervised_positions": 16,
                "label_accuracy": {
                  "qwen_native": 1.0,
                  "stage_ab_only": 0.75,
                  "stage_ab_reader": 0.75
                }
              },
              "gsm8k": {
                "n": 16,
                "mean_kl": 0.18423763429746032,
                "same_prefix_next_token_agreement": 0.890625,
                "supervised_positions": 128
              },
              "mmlu_pro": {
                "n": 16,
                "mean_kl": 1.032063302482129,
                "same_prefix_next_token_agreement": 0.625,
                "supervised_positions": 16,
                "label_accuracy": {
                  "qwen_native": 0.3125,
                  "stage_ab_only": 0.3125,
                  "stage_ab_reader": 0.375
                }
              }
            },
            "macro_kl": 0.489884066576451
          }
        },
        {
          "epoch": 3,
          "step": 768,
          "validation": {
            "tasks": {
              "openbookqa": {
                "n": 16,
                "mean_kl": 0.36019871625467204,
                "same_prefix_next_token_agreement": 0.875,
                "supervised_positions": 16,
                "label_accuracy": {
                  "qwen_native": 0.8125,
                  "stage_ab_only": 0.625,
                  "stage_ab_reader": 0.625
                }
              },
              "arc_challenge": {
                "n": 16,
                "mean_kl": 0.7243947678070981,
                "same_prefix_next_token_agreement": 0.8125,
                "supervised_positions": 16,
                "label_accuracy": {
                  "qwen_native": 1.0,
                  "stage_ab_only": 0.75,
                  "stage_ab_reader": 0.8125
                }
              },
              "gsm8k": {
                "n": 16,
                "mean_kl": 0.1738615392241627,
                "same_prefix_next_token_agreement": 0.8984375,
                "supervised_positions": 128
              },
              "mmlu_pro": {
                "n": 16,
                "mean_kl": 1.2310237512429012,
                "same_prefix_next_token_agreement": 0.625,
                "supervised_positions": 16,
                "label_accuracy": {
                  "qwen_native": 0.3125,
                  "stage_ab_only": 0.3125,
                  "stage_ab_reader": 0.25
                }
              }
            },
            "macro_kl": 0.6223696936322085
          }
        },
        {
          "epoch": 4,
          "step": 1024,
          "validation": {
            "tasks": {
              "openbookqa": {
                "n": 16,
                "mean_kl": 0.40170132838829886,
                "same_prefix_next_token_agreement": 0.875,
                "supervised_positions": 16,
                "label_accuracy": {
                  "qwen_native": 0.8125,
                  "stage_ab_only": 0.625,
                  "stage_ab_reader": 0.625
                }
              },
              "arc_challenge": {
                "n": 16,
                "mean_kl": 0.26070858562889043,
                "same_prefix_next_token_agreement": 0.875,
                "supervised_positions": 16,
                "label_accuracy": {
                  "qwen_native": 1.0,
                  "stage_ab_only": 0.75,
                  "stage_ab_reader": 0.875
                }
              },
              "gsm8k": {
                "n": 16,
                "mean_kl": 0.18696373759303242,
                "same_prefix_next_token_agreement": 0.890625,
                "supervised_positions": 128
              },
              "mmlu_pro": {
                "n": 16,
                "mean_kl": 1.22758460240766,
                "same_prefix_next_token_agreement": 0.5625,
                "supervised_positions": 16,
                "label_accuracy": {
                  "qwen_native": 0.3125,
                  "stage_ab_only": 0.3125,
                  "stage_ab_reader": 0.25
                }
              }
            },
            "macro_kl": 0.5192395635044704
          }
        }
      ],
      "memory": {
        "peak_pytorch_allocated_gib": 9.131556987762451,
        "peak_pytorch_reserved_gib": 9.6015625,
        "scope": "This process CUDA allocator since load_all; not whole-GPU nvidia-smi peak"
      },
      "manifest_sha256": "9472f20d101851631929e5fc60de19dd6f9e59f8f51b5eb10694c0d4ddc47e10",
      "stage_a_and_b_bitwise_unchanged": true
    },
    "reader_manifest_sha256": "9472f20d101851631929e5fc60de19dd6f9e59f8f51b5eb10694c0d4ddc47e10",
    "parent_manifest_sha256": "eb0fe67b8e7665f9df880ee86551b1524db6ef3cb67ddede7cafe6ada3a6a739",
    "caveats": [
      "The extra Reader adds training budget beyond Stage A+B; this is an incremental-gain experiment, not an equal-budget comparison.",
      "MCQ answer-boundary objective matches recent A+Reader experiment; differs from original residual training positions.",
      "Previous A+Reader evaluation is paused: comparison uses only available matched samples and explicitly reports n/expected.",
      "MMLU-Pro remains the custom split from official test parquet, not untouched full official test.",
      "GSM8K training cap64 and up to8 positions; formal greedy evaluation cap4096.",
      "Historical test subsets have been inspected; fixed final checkpoint only, no test selection.",
      "Existing Full-Sync interface uses Native body prefill for teacher/anchor; no prefill-free latency claim."
    ]
  },
  "raw_results": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234/evaluation/per_sample.jsonl",
  "log": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234/logs/pipeline.log"
}
```


## attention_mixed_precision_senderkv_2wiki128_l3_to_q4_frozen_klreader_seed1234

```json
{
  "experiment": "attention_mixed_precision_senderkv_2wiki128_l3_to_q4_frozen_klreader_seed1234",
  "status": "completed",
  "timestamp": "2026-10-25 22:44:57",
  "manifest_sha256": "5c938cb6d4974f4085817ce607057b3347abfddb3aaa8c2b3d51f39e62307b21",
  "config": {
    "name": "attention_mixed_precision_senderkv_2wiki128_l3_to_q4_frozen_klreader_seed1234",
    "seed": 1234,
    "n": 128,
    "dataset": "/hy-tmp/yezhe/all_models/datasets/voidful/2WikiMultihopQA/dev.json",
    "reader_experiment": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234",
    "reference_reader": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234",
    "reader_checkpoint": "runs/reader/epoch4_step1024.pt",
    "models": {
      "llama": "/hy-tmp/yezhe/all_models/models/LLM-Research/Llama-3.2-3B-Instruct",
      "qwen": "/hy-tmp/yezhe/all_models/models/Qwen/Qwen3-4B"
    },
    "excluded_ids": [
      "66eae2780bdd11eba7f7acde48001122",
      "82ef6eec0bda11eba7f7acde48001122",
      "38056cd40bde11eba7f7acde48001122",
      "528f7ff9093c11ebbdaeac1f6bf848b6",
      "654e76b808e111ebbda2ac1f6bf848b6",
      "c218f9d20bdd11eba7f7acde48001122",
      "0bbded440bda11eba7f7acde48001122",
      "96e142e40bdb11eba7f7acde48001122",
      "ffee28d80bae11ebab90acde48001122",
      "341347220bde11eba7f7acde48001122",
      "6e31b2920bdc11eba7f7acde48001122",
      "75af4f140bde11eba7f7acde48001122",
      "179f51c60bde11eba7f7acde48001122",
      "6b6ef0280bb011ebab90acde48001122",
      "3a13d0620baf11ebab90acde48001122",
      "a8cbc4c40bdc11eba7f7acde48001122",
      "161e9e380bde11eba7f7acde48001122",
      "615178b20bde11eba7f7acde48001122",
      "71530df00bdb11eba7f7acde48001122",
      "fd56936708a811ebbd7fac1f6bf848b6",
      "067dd55a0bdb11eba7f7acde48001122",
      "90c88e000bd911eba7f7acde48001122",
      "97f5e2360bd911eba7f7acde48001122",
      "374ff12e0bde11eba7f7acde48001122"
    ],
    "excluded_question_keys": [
      "when is victor levasseur (cartographer)'s father's birthday?",
      "where did the director of film the tracker (2002 film) graduate from?",
      "when did thomas knox, 3rd earl of ranfurly's father die?",
      "which album was released more recently, epistemological despondency or tonight the stars revolt!?",
      "who lived longer, rolf billberg or yeo woon-kay?",
      "where was the place of death of the composer of song i've got a crush on you?",
      "which country the director of film daadagiri is from?",
      "where was the director of film bangarada mane born?",
      "who is the mother-in-law of catherine i, latin empress?",
      "where was the performer of song the name of love born?",
      "where did the director of film a.w.o.l. (2006 film) die?",
      "what nationality is henry clifford, 2nd earl of cumberland's wife?",
      "what is the date of death of andrew, duke of slavonia's father?",
      "who is the uncle of bermudo i of asturias?",
      "who is the paternal grandmother of jacques d'armagnac?",
      "where was the director of film the half-way girl born?",
      "where did the composer of song that certain feeling die?",
      "what nationality is adolf, duke of holstein-gottorp's father?",
      "who is the spouse of the director of film aladdin aur jadui chirag?",
      "who is older, marius mitu or bea palya?",
      "where was the director of film caresses born?",
      "where was the performer of song la ballade des gens heureux born?",
      "what nationality is the director of film that summer of white roses?",
      "when is philipp, prince of hohenlohe-langenburg's father's birthday?"
    ],
    "max_prompt_tokens": 768,
    "max_gold_tokens": 16,
    "max_answer_characters": 48,
    "max_new_tokens": 32,
    "distractor_sentences": 8,
    "answer_prefix": "The answer is:",
    "group_size": 64,
    "protected_fraction": 0.1,
    "random_seeds": [
      1234,
      1235,
      1236
    ],
    "free_gpu_gib": 12,
    "disk_reserve_gib": 12,
    "attention_backend": "eager",
    "automatic_upload": false,
    "training": false,
    "wait_for_previous_experiment": false,
    "reader_selection": "Original completed KL-only Reader, fixed epoch4/step1024; taskCEKL Reader explicitly excluded.",
    "quantizer": "Fixed shared asymmetric FP16 min/range; K grouped over64 transmitted source positions per channel, V over64 channels per token. Identical quantization ranges for INT4 and INT8; per-token bit allocation. KIVI-inspired axes, NOT an unmodified KIVI/KVTuner reproduction.",
    "wire_policy": "Unique original Sender positions used by the existing Full-Sync map. Quantize once before alignment expansion. Same decoded tensors replace those positions in native Llama or feed the fixed Translator after original map. Other native cache positions / Qwen anchor unchanged.",
    "scope": "128 previously unseen 2Wiki controlled evidence questions, oracle supporting sentences plus8 distractors; exploratory diagnostic, not official full-context benchmark.",
    "ranking": "FP16 Llama last forced-prefix query, query-head/layer mean projected to evidence text units then to original transmitted Llama token positions. No gold answer in ranking; top/bottom/random protect identical ceil(0.1*Nwire) evidence-overlapping positions across all layers/KV heads and K,V.",
    "bytes": "Actual packed packet length including FP16 quantization min/range, bitmask, source positions, receiver mapping, dimensions and section lengths. C/D/E exact byte matched; INT4 is lower-budget reference."
  },
  "paths": {
    "llama": {
      "arms": {
        "fp16": {
          "n": 128,
          "accuracy": 0.390625,
          "cap_rate": 0.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 7.339857437116404e-10,
            "ci95": [
              -4.660658521377266e-09,
              6.145720886244297e-09
            ]
          },
          "mean_wire_bytes": 46032791.015625,
          "effective_bits_per_element": 16.001210695375196
        },
        "int4": {
          "n": 128,
          "accuracy": 0.3984375,
          "cap_rate": 0.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": 0.01258504723568104,
            "ci95": [
              0.0021286686541122026,
              0.024560133089622342
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": 0.0033429184451023116,
            "ci95": [
              0.0003062673400652484,
              0.00673078290823597
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 0.008490744874052325,
            "ci95": [
              0.0070097563430636,
              0.010138051868773345
            ]
          },
          "mean_wire_bytes": 13004174.1484375,
          "effective_bits_per_element": 4.520852989495632
        },
        "attention_top": {
          "n": 128,
          "accuracy": 0.375,
          "cap_rate": 0.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": 0.0050990517553556524,
            "ci95": [
              -0.0037026754677126714,
              0.013930472195123337
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": 0.0017429276602334725,
            "ci95": [
              -0.0005041626461344945,
              0.004341978045109435
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 0.005240486535285527,
            "ci95": [
              0.004178229350179663,
              0.006246348900685916
            ]
          },
          "mean_wire_bytes": 14167423.6796875,
          "effective_bits_per_element": 4.9253652364013725
        },
        "random_0": {
          "n": 128,
          "accuracy": 0.40625,
          "cap_rate": 0.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": 0.007536755072151971,
            "ci95": [
              -0.0033080231666689467,
              0.01917518574256304
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": 0.001955261841721912,
            "ci95": [
              -0.0009497091706684079,
              0.0049433396624098775
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 0.00777409989973421,
            "ci95": [
              0.0065063897034463025,
              0.00913379376620469
            ]
          },
          "mean_wire_bytes": 14167423.6796875,
          "effective_bits_per_element": 4.9253652364013725
        },
        "random_1": {
          "n": 128,
          "accuracy": 0.390625,
          "cap_rate": 0.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": 0.015002196766098971,
            "ci95": [
              0.004957848398392166,
              0.026625478506867264
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": 0.004373664906131468,
            "ci95": [
              0.001675051001514329,
              0.007335838758215496
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 0.008099196628108984,
            "ci95": [
              0.006684839589860303,
              0.009654035314798647
            ]
          },
          "mean_wire_bytes": 14167423.6796875,
          "effective_bits_per_element": 4.9253652364013725
        },
        "random_2": {
          "n": 128,
          "accuracy": 0.390625,
          "cap_rate": 0.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": 0.015088234327189753,
            "ci95": [
              0.004719827608861167,
              0.026629364494375068
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": 0.004104222969796083,
            "ci95": [
              0.0014614541467837634,
              0.007231721868676487
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 0.008021186845326156,
            "ci95": [
              0.00661637362733245,
              0.009654750441222859
            ]
          },
          "mean_wire_bytes": 14167423.6796875,
          "effective_bits_per_element": 4.9253652364013725
        },
        "attention_bottom": {
          "n": 128,
          "accuracy": 0.3984375,
          "cap_rate": 0.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": 0.012443192463757993,
            "ci95": [
              0.0018888967970371723,
              0.02434717760347804
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": 0.0033024509676324357,
            "ci95": [
              0.00024888794591624165,
              0.00666385341149636
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 0.008449383055904036,
            "ci95": [
              0.00697825550264497,
              0.01006133405539913
            ]
          },
          "mean_wire_bytes": 14167423.6796875,
          "effective_bits_per_element": 4.9253652364013725
        },
        "random_mean": {
          "n": 128,
          "accuracy": 0.39583333333333337,
          "cap_rate": 0.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": 0.012542395388480228,
            "ci95": [
              0.002842719176641047,
              0.02400668531851353
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": 0.003477716572549821,
            "ci95": [
              0.0009365660153744721,
              0.00642617791904035
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 0.00796482779105645,
            "ci95": [
              0.006660139120490764,
              0.009445552215472237
            ]
          },
          "mean_wire_bytes": 14167423.6796875,
          "effective_bits_per_element": 4.9253652364013725
        }
      },
      "paired_contrasts": {
        "attention_top_minus_random_mean": {
          "n": 128,
          "mean": -0.0074433436331245795,
          "ci95": [
            -0.015404000053131873,
            0.0010176915420214382
          ]
        },
        "attention_top_minus_attention_bottom": {
          "n": 128,
          "mean": -0.007344140708402338,
          "ci95": [
            -0.015495720724422495,
            0.0013021445661606178
          ]
        }
      }
    },
    "qwen": {
      "arms": {
        "fp16": {
          "n": 128,
          "accuracy": 0.015625,
          "cap_rate": 1.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 2.012911087811986e-09,
            "ci95": [
              -7.451558388556212e-10,
              4.896655849632308e-09
            ]
          },
          "mean_wire_bytes": 46032791.015625,
          "effective_bits_per_element": 16.001210695375196
        },
        "int4": {
          "n": 128,
          "accuracy": 0.015625,
          "cap_rate": 1.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": -0.014984251100655578,
            "ci95": [
              -0.02952787636323054,
              0.00020565644446841322
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": -0.003945383483162408,
            "ci95": [
              -0.00721982363832245,
              -0.0004658067106211339
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 6.434298835578112e-05,
            "ci95": [
              4.2469077371382057e-05,
              9.213105101253393e-05
            ]
          },
          "mean_wire_bytes": 13004174.1484375,
          "effective_bits_per_element": 4.520852989495632
        },
        "attention_top": {
          "n": 128,
          "accuracy": 0.015625,
          "cap_rate": 1.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": -0.014473775583241338,
            "ci95": [
              -0.02780863208689116,
              -0.0006524198016627893
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": -0.002878812753781912,
            "ci95": [
              -0.006007251671104028,
              0.00046386581100323103
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 5.663188779436723e-05,
            "ci95": [
              3.8996182682260463e-05,
              7.775622656194247e-05
            ]
          },
          "mean_wire_bytes": 14167423.6796875,
          "effective_bits_per_element": 4.9253652364013725
        },
        "random_0": {
          "n": 128,
          "accuracy": 0.015625,
          "cap_rate": 1.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": -0.014680291647735575,
            "ci95": [
              -0.028073714896148533,
              -0.00047838770573035484
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": -0.003808860077142885,
            "ci95": [
              -0.006880073201276155,
              -0.0006848783770982555
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 5.875946846289892e-05,
            "ci95": [
              4.02958358722616e-05,
              8.162900366871506e-05
            ]
          },
          "mean_wire_bytes": 14167423.6796875,
          "effective_bits_per_element": 4.9253652364013725
        },
        "random_1": {
          "n": 128,
          "accuracy": 0.015625,
          "cap_rate": 1.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": -0.01490715497182366,
            "ci95": [
              -0.02952287591118932,
              5.962957037163063e-05
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": -0.0036571105985700038,
            "ci95": [
              -0.006983003851872574,
              -0.00025330883478459913
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 5.9815304842114614e-05,
            "ci95": [
              4.132210035419081e-05,
              8.318594069975926e-05
            ]
          },
          "mean_wire_bytes": 14167423.6796875,
          "effective_bits_per_element": 4.9253652364013725
        },
        "random_2": {
          "n": 128,
          "accuracy": 0.015625,
          "cap_rate": 1.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": -0.013857651062045968,
            "ci95": [
              -0.027224522342550614,
              0.0008966296236228964
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": -0.0034012483656017576,
            "ci95": [
              -0.006554287392797216,
              9.231088770146122e-05
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 5.8444790616896114e-05,
            "ci95": [
              3.920314705452288e-05,
              8.238436074361477e-05
            ]
          },
          "mean_wire_bytes": 14167423.6796875,
          "effective_bits_per_element": 4.9253652364013725
        },
        "attention_bottom": {
          "n": 128,
          "accuracy": 0.015625,
          "cap_rate": 1.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": -0.015223224675151564,
            "ci95": [
              -0.030035948712492114,
              -0.0002378660076579229
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": -0.003938225443777298,
            "ci95": [
              -0.007252993611005582,
              -0.000420596494821332
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 5.9623185135304624e-05,
            "ci95": [
              4.0592782229220695e-05,
              8.416274439653203e-05
            ]
          },
          "mean_wire_bytes": 14167423.6796875,
          "effective_bits_per_element": 4.9253652364013725
        },
        "random_mean": {
          "n": 128,
          "accuracy": 0.015625,
          "cap_rate": 1.0,
          "delta_nll_per_token": {
            "n": 128,
            "mean": -0.014481699227201734,
            "ci95": [
              -0.028153133143812868,
              -0.00021612989773021622
            ]
          },
          "delta_nll_per_character": {
            "n": 128,
            "mean": -0.003622406347104883,
            "ci95": [
              -0.0067196038953311445,
              -0.0002743880099427037
            ]
          },
          "next_token_kl": {
            "n": 128,
            "mean": 5.900652130730322e-05,
            "ci95": [
              4.0300246546914886e-05,
              8.207603291342328e-05
            ]
          },
          "mean_wire_bytes": 14167423.6796875,
          "effective_bits_per_element": 4.9253652364013725
        }
      },
      "paired_contrasts": {
        "attention_top_minus_random_mean": {
          "n": 128,
          "mean": 7.923643960395816e-06,
          "ci95": [
            -0.00790361543310993,
            0.006593376597648093
          ]
        },
        "attention_top_minus_attention_bottom": {
          "n": 128,
          "mean": 0.0007494490919102252,
          "ci95": [
            -0.00867911117383873,
            0.008838376704788873
          ]
        }
      }
    }
  },
  "uncertainty": "Paired question bootstrap, random repeats averaged within question; exploratory uncorrected95% intervals."
}
```


## attention_mixed_precision_senderkv_2wiki128_l3_to_q4_frozen_klreader_seed1234 / unquantized-chain diagnostic

```json
{
  "status": "completed",
  "timestamp": "2026-10-26 00:04:51",
  "n": 16,
  "max_new_tokens": 128,
  "original_experiment_manifest_sha256": "5c938cb6d4974f4085817ce607057b3347abfddb3aaa8c2b3d51f39e62307b21",
  "sample_selection": "First16 predetermined prepared questions; diagnostic subset, not a new benchmark.",
  "code_sha256": "c7c07fd8fc82428446db3952203f2f5daeac0f01021e277add1c91bb6f99eabc",
  "arms": {
    "native_qwen": {
      "correct": 6,
      "capped": 0,
      "mean_nll": 1.8351083106004236,
      "mean_tokens": 6.625
    },
    "stage_ab": {
      "correct": 0,
      "capped": 10,
      "mean_nll": 6.915514600702695,
      "mean_tokens": 120.1875
    },
    "stage_ab_reader": {
      "correct": 0,
      "capped": 10,
      "mean_nll": 6.964273583179429,
      "mean_tokens": 120.5625
    }
  },
  "implementation_audit": {
    "sender_key_capture_max_error": 0.0,
    "sender_value_capture_max_error": 0.0,
    "canonical_mapped_key_max_error": 0.0,
    "canonical_mapped_value_max_error": 0.0,
    "canonical_vs_new_gold_nll_error": 0.0,
    "formal_nll_replay_error": 0.0,
    "formal_generation_is_prefix": true
  },
  "mapped_key_nmse_mean": 0.1465194346383214,
  "mapped_value_nmse_mean": 0.6850650794804096
}
```


## attention_mixed_precision_senderkv_2wiki128_l3_to_q4_frozen_klreader_seed1234 / known_tasks_obqa_arc128_seed1234

```json
{
  "experiment": "known_tasks_obqa_arc128_seed1234",
  "timestamp": "2026-10-26 09:38:44",
  "status": "completed",
  "manifest_sha256": "051c1233659dd011688671ead87eefd75f745b87779d8be2715682f3a457324f",
  "config": {
    "name": "known_tasks_obqa_arc128_seed1234",
    "seed": 1234,
    "tasks": {
      "openbookqa": 128,
      "arc_challenge": 128
    },
    "training": false,
    "automatic_upload": false,
    "protected_fraction": 0.1,
    "group_size": 64,
    "random_seeds": [
      1234,
      1235,
      1236
    ],
    "free_gpu_gib": 12,
    "disk_reserve_gib": 12,
    "scoring": "Original fixed single-token candidate labels; one causal suffix forward gives all option log probabilities. Primary: candidate-normalized gold NLL and accuracy; full-vocabulary gold-label NLL diagnostic. No free generation.",
    "ranking": "Unquantized Llama last existing answer-cue query, query-head/layer mean on unique transmitted Sender positions; top/bottom/random10% of all transmitted prompt tokens, shared across layers, heads and K/V.",
    "scope": "Existing held-out128 OpenBookQA and128 ARC-Challenge from old KL-only Reader evaluation; within previously adapted tasks, not proof of cross-task generalization.",
    "baseline": "First replay every unquantized translated+KL-only Reader sample using original cached Sender states; compare historical candidate scores/predictions. Stop before quantization if either task accuracy below50%.",
    "optimization": "Reuse fixed per-question min/range and both INT4/INT8 codebooks across allocation variants; actual wire encoding/decoding retained. Evaluate all single-token labels in one forward rather than4 repeated forwards."
  },
  "checkpoints": {
    "reader": {
      "path": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234/runs/reader/epoch4_step1024.pt",
      "sha256": "96cde5b41f1a15434d24ef57f8eaf34e3af89af109c0b5d6cf378e60ab577314"
    },
    "stage_a": {
      "path": "/hy-tmp/yezhe/异构模型/grouped_headmix4_l3_to_q4_seed1234/runs/stage_a/best.pt",
      "sha256": "a9318f4b2f1d9e7981e0e50e2dc88654f0baaf89a080a0746017555c48ba5c84"
    },
    "stage_b": {
      "path": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_l3_to_q4_seed1234/runs/stage_b/last.pt",
      "sha256": "7e9f602d08409574c8ebe56242a08ba2c6861032b5ed70570a807c5777d775f2"
    }
  },
  "baseline": {
    "status": "completed",
    "manifest_sha256": "051c1233659dd011688671ead87eefd75f745b87779d8be2715682f3a457324f",
    "tasks": {
      "openbookqa": {
        "n": 128,
        "correct": 92,
        "accuracy": 0.71875,
        "historical_prediction_matches": 127,
        "max_historical_score_error": 0.09713935852050781
      },
      "arc_challenge": {
        "n": 128,
        "correct": 86,
        "accuracy": 0.671875,
        "historical_prediction_matches": 128,
        "max_historical_score_error": 0.09763908386230469
      }
    }
  },
  "tasks": {
    "openbookqa": {
      "llama": {
        "arms": {
          "fp16": {
            "n": 128,
            "correct": 97.0,
            "accuracy": 0.7578125,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.0,
              "ci95": [
                0.0,
                0.0
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": 0.0,
              "ci95": [
                0.0,
                0.0
              ]
            },
            "mean_wire_bytes": 6922212.3046875,
            "mean_effective_bits": 16.001425870171822
          },
          "int4": {
            "n": 128,
            "correct": 93.0,
            "accuracy": 0.7265625,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.0036286283041704337,
              "ci95": [
                -0.01359510784196516,
                0.019058954994122246
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.022637411952018738,
              "ci95": [
                -0.04699337892234325,
                0.003653606772422786
              ]
            },
            "mean_wire_bytes": 1993289.21875,
            "mean_effective_bits": 4.602559916566274
          },
          "attention_top": {
            "n": 128,
            "correct": 94.0,
            "accuracy": 0.734375,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.006919456156197157,
              "ci95": [
                -0.002078156719633739,
                0.016486235631595035
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.01647966355085373,
              "ci95": [
                -0.030611450783908367,
                -0.0004268841817975102
              ]
            },
            "mean_wire_bytes": 2179897.015625,
            "mean_effective_bits": 5.034620899836887
          },
          "random_0": {
            "n": 128,
            "correct": 94.0,
            "accuracy": 0.734375,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.0033427189727603945,
              "ci95": [
                -0.012301731857380104,
                0.017327512534453148
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.021116182208061218,
              "ci95": [
                -0.04271480720490217,
                0.00146548617631196
              ]
            },
            "mean_wire_bytes": 2179897.015625,
            "mean_effective_bits": 5.034620899836887
          },
          "random_1": {
            "n": 128,
            "correct": 93.0,
            "accuracy": 0.7265625,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.0032478841458214447,
              "ci95": [
                -0.01393834496857096,
                0.019528156578024886
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.023868680000305176,
              "ci95": [
                -0.04675509911030531,
                0.001914618164300911
              ]
            },
            "mean_wire_bytes": 2179897.015625,
            "mean_effective_bits": 5.034620899836887
          },
          "random_2": {
            "n": 128,
            "correct": 94.0,
            "accuracy": 0.734375,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.004673248010249154,
              "ci95": [
                -0.011681860441876779,
                0.020321494025685633
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.017326287925243378,
              "ci95": [
                -0.03968162313103676,
                0.008192687667906281
              ]
            },
            "mean_wire_bytes": 2179897.015625,
            "mean_effective_bits": 5.034620899836887
          },
          "attention_bottom": {
            "n": 128,
            "correct": 93.0,
            "accuracy": 0.7265625,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.004338092536102067,
              "ci95": [
                -0.012884641947793795,
                0.01989269699959095
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.02248246967792511,
              "ci95": [
                -0.04653852432966232,
                0.004116121865808963
              ]
            },
            "mean_wire_bytes": 2179897.015625,
            "mean_effective_bits": 5.034620899836887
          },
          "random_mean": {
            "n": 128,
            "correct": 93.66666666666667,
            "accuracy": 0.7317708333333334,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.0037546170429436643,
              "ci95": [
                -0.011858805659060563,
                0.01781652620176856
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.02077038337786992,
              "ci95": [
                -0.04308765046298504,
                0.003005666472017761
              ]
            },
            "mean_wire_bytes": 2179897.015625,
            "mean_effective_bits": 5.034620899836887
          }
        },
        "paired_choice_nll_contrasts": {
          "top_minus_random_mean": {
            "n": 128,
            "mean": 0.0031648391132534917,
            "ci95": [
              -0.00889026822423773,
              0.016498873855770545
            ]
          },
          "top_minus_attention_bottom": {
            "n": 128,
            "mean": 0.0025813636200950896,
            "ci95": [
              -0.010723245226176913,
              0.016808704239929282
            ]
          }
        }
      },
      "qwen": {
        "arms": {
          "fp16": {
            "n": 128,
            "correct": 92.0,
            "accuracy": 0.71875,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.0,
              "ci95": [
                0.0,
                0.0
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": 0.0,
              "ci95": [
                0.0,
                0.0
              ]
            },
            "mean_wire_bytes": 6922212.3046875,
            "mean_effective_bits": 16.001425870171822
          },
          "int4": {
            "n": 128,
            "correct": 90.0,
            "accuracy": 0.703125,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.010550060802813732,
              "ci95": [
                -0.023182070417635713,
                0.0009549380704029877
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": 0.001951649785041809,
              "ci95": [
                -0.02053266130387783,
                0.02630281150341034
              ]
            },
            "mean_wire_bytes": 1993289.21875,
            "mean_effective_bits": 4.602559916566274
          },
          "attention_top": {
            "n": 128,
            "correct": 92.0,
            "accuracy": 0.71875,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.005954808477810047,
              "ci95": [
                -0.014875358284926676,
                0.002547606592928271
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": 0.010270670056343079,
              "ci95": [
                -0.007335486635565758,
                0.02761272843927145
              ]
            },
            "mean_wire_bytes": 2179897.015625,
            "mean_effective_bits": 5.034620899836887
          },
          "random_0": {
            "n": 128,
            "correct": 91.0,
            "accuracy": 0.7109375,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.010819724338887227,
              "ci95": [
                -0.02205640191669289,
                -7.11216605544992e-05
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.0006339326500892639,
              "ci95": [
                -0.021096020936965942,
                0.021188536286354052
              ]
            },
            "mean_wire_bytes": 2179897.015625,
            "mean_effective_bits": 5.034620899836887
          },
          "random_1": {
            "n": 128,
            "correct": 90.0,
            "accuracy": 0.703125,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.014183301194123032,
              "ci95": [
                -0.026912583108641284,
                -0.002275984850594175
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.001020766794681549,
              "ci95": [
                -0.022043761610984803,
                0.020418697036802765
              ]
            },
            "mean_wire_bytes": 2179897.015625,
            "mean_effective_bits": 5.034620899836887
          },
          "random_2": {
            "n": 128,
            "correct": 90.0,
            "accuracy": 0.703125,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.014567760548175102,
              "ci95": [
                -0.02987306161814516,
                -0.0007029452585058517
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.003783084452152252,
              "ci95": [
                -0.028542053326964377,
                0.020747092366218557
              ]
            },
            "mean_wire_bytes": 2179897.015625,
            "mean_effective_bits": 5.034620899836887
          },
          "attention_bottom": {
            "n": 128,
            "correct": 90.0,
            "accuracy": 0.703125,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.012007538184462874,
              "ci95": [
                -0.02417075484022675,
                -0.0003080265199002992
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.008489027619361877,
              "ci95": [
                -0.03020941633731127,
                0.015790183655917642
              ]
            },
            "mean_wire_bytes": 2179897.015625,
            "mean_effective_bits": 5.034620899836887
          },
          "random_mean": {
            "n": 128,
            "correct": 90.33333333333334,
            "accuracy": 0.7057291666666666,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.013190262027061786,
              "ci95": [
                -0.025553084163553307,
                -0.0019655214130494887
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.0018125946323076878,
              "ci95": [
                -0.022959416235486665,
                0.01977528582016627
              ]
            },
            "mean_wire_bytes": 2179897.015625,
            "mean_effective_bits": 5.034620899836887
          }
        },
        "paired_choice_nll_contrasts": {
          "top_minus_random_mean": {
            "n": 128,
            "mean": 0.007235453549251739,
            "ci95": [
              -0.0011781706798695767,
              0.016738361862703256
            ]
          },
          "top_minus_attention_bottom": {
            "n": 128,
            "mean": 0.006052729706652826,
            "ci95": [
              -0.003901956147203833,
              0.016831457481085006
            ]
          }
        }
      }
    },
    "arc_challenge": {
      "llama": {
        "arms": {
          "fp16": {
            "n": 128,
            "correct": 99.0,
            "accuracy": 0.7734375,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.0,
              "ci95": [
                0.0,
                0.0
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": 0.0,
              "ci95": [
                0.0,
                0.0
              ]
            },
            "mean_wire_bytes": 9588899.421875,
            "mean_effective_bits": 16.001354843594797
          },
          "int4": {
            "n": 128,
            "correct": 99.0,
            "accuracy": 0.7734375,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.0002573067644107141,
              "ci95": [
                -0.014296489562815396,
                0.014710029959894655
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.02435095456894487,
              "ci95": [
                -0.05095792078936938,
                0.0012062316614901607
              ]
            },
            "mean_wire_bytes": 2757321.6875,
            "mean_effective_bits": 4.604225321582973
          },
          "attention_top": {
            "n": 128,
            "correct": 98.0,
            "accuracy": 0.765625,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.0013368119654160182,
              "ci95": [
                -0.0089138040095321,
                0.006436493496289583
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.011885703424923122,
              "ci95": [
                -0.024435229619848543,
                0.0019908131827833095
              ]
            },
            "mean_wire_bytes": 3010681.4375,
            "mean_effective_bits": 5.029042128333167
          },
          "random_0": {
            "n": 128,
            "correct": 99.0,
            "accuracy": 0.7734375,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.00026661859442356217,
              "ci95": [
                -0.01493739293802605,
                0.015054927907596034
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.018142331624403596,
              "ci95": [
                -0.043390674190595745,
                0.0063134386320598404
              ]
            },
            "mean_wire_bytes": 3010681.4375,
            "mean_effective_bits": 5.029042128333167
          },
          "random_1": {
            "n": 128,
            "correct": 99.0,
            "accuracy": 0.7734375,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.000843557813368534,
              "ci95": [
                -0.015545618635837853,
                0.012999523129178663
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.017622720275539905,
              "ci95": [
                -0.04178869184106588,
                0.0067147271271096485
              ]
            },
            "mean_wire_bytes": 3010681.4375,
            "mean_effective_bits": 5.029042128333167
          },
          "random_2": {
            "n": 128,
            "correct": 99.0,
            "accuracy": 0.7734375,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.005225871910397473,
              "ci95": [
                -0.008669653754901674,
                0.019604475649396137
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.009488474577665329,
              "ci95": [
                -0.03337244698777795,
                0.015493340883404011
              ]
            },
            "mean_wire_bytes": 3010681.4375,
            "mean_effective_bits": 5.029042128333167
          },
          "attention_bottom": {
            "n": 128,
            "correct": 99.0,
            "accuracy": 0.7734375,
            "delta_choice_nll": {
              "n": 128,
              "mean": 6.967100466681586e-05,
              "ci95": [
                -0.014634811090883205,
                0.014639729933543316
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.023804662807378918,
              "ci95": [
                -0.050155441065726336,
                0.0022971295067691242
              ]
            },
            "mean_wire_bytes": 3010681.4375,
            "mean_effective_bits": 5.029042128333167
          },
          "random_mean": {
            "n": 128,
            "correct": 99.0,
            "accuracy": 0.7734375,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.0015496442304841684,
              "ci95": [
                -0.012968119077485575,
                0.015872755996516233
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.01508450882586961,
              "ci95": [
                -0.03895825793345769,
                0.008902510876456896
              ]
            },
            "mean_wire_bytes": 3010681.4375,
            "mean_effective_bits": 5.029042128333167
          }
        },
        "paired_choice_nll_contrasts": {
          "top_minus_random_mean": {
            "n": 128,
            "mean": -0.0028864561959001875,
            "ci95": [
              -0.016538835099444787,
              0.010885656165848443
            ]
          },
          "top_minus_attention_bottom": {
            "n": 128,
            "mean": -0.001406482970082834,
            "ci95": [
              -0.015435287253930596,
              0.012494694908303922
            ]
          }
        }
      },
      "qwen": {
        "arms": {
          "fp16": {
            "n": 128,
            "correct": 86.0,
            "accuracy": 0.671875,
            "delta_choice_nll": {
              "n": 128,
              "mean": 0.0,
              "ci95": [
                0.0,
                0.0
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": 0.0,
              "ci95": [
                0.0,
                0.0
              ]
            },
            "mean_wire_bytes": 9588899.421875,
            "mean_effective_bits": 16.001354843594797
          },
          "int4": {
            "n": 128,
            "correct": 86.0,
            "accuracy": 0.671875,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.01372914716057494,
              "ci95": [
                -0.029175320850856678,
                0.0005784690455683962
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.025672728617792018,
              "ci95": [
                -0.04773418350268912,
                -0.002782078509335407
              ]
            },
            "mean_wire_bytes": 2757321.6875,
            "mean_effective_bits": 4.604225321582973
          },
          "attention_top": {
            "n": 128,
            "correct": 85.0,
            "accuracy": 0.6640625,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.0038347185835334585,
              "ci95": [
                -0.009815471478826509,
                0.002500117520671096
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": 0.0013249630137579516,
              "ci95": [
                -0.009871558099985123,
                0.013150493800640106
              ]
            },
            "mean_wire_bytes": 3010681.4375,
            "mean_effective_bits": 5.029042128333167
          },
          "random_0": {
            "n": 128,
            "correct": 86.0,
            "accuracy": 0.671875,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.010446806562583788,
              "ci95": [
                -0.02385575933808326,
                0.002306689517033076
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.024450807613902725,
              "ci95": [
                -0.044735337013480606,
                -0.0030189886369044014
              ]
            },
            "mean_wire_bytes": 3010681.4375,
            "mean_effective_bits": 5.029042128333167
          },
          "random_1": {
            "n": 128,
            "correct": 86.0,
            "accuracy": 0.671875,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.01592284496331331,
              "ci95": [
                -0.03136903930426911,
                -0.0016210844433089764
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.02992591577640269,
              "ci95": [
                -0.05175731956733216,
                -0.007615670826999122
              ]
            },
            "mean_wire_bytes": 3010681.4375,
            "mean_effective_bits": 5.029042128333167
          },
          "random_2": {
            "n": 128,
            "correct": 86.0,
            "accuracy": 0.671875,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.011167973605353154,
              "ci95": [
                -0.02648931150363012,
                0.0037298012672550823
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.024957003683084622,
              "ci95": [
                -0.04641522037782124,
                -0.004003857076168077
              ]
            },
            "mean_wire_bytes": 3010681.4375,
            "mean_effective_bits": 5.029042128333167
          },
          "attention_bottom": {
            "n": 128,
            "correct": 86.0,
            "accuracy": 0.671875,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.01290229587880276,
              "ci95": [
                -0.027502593268062016,
                0.001557109830162061
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.03440221195342019,
              "ci95": [
                -0.055218125073588455,
                -0.012366520189971204
              ]
            },
            "mean_wire_bytes": 3010681.4375,
            "mean_effective_bits": 5.029042128333167
          },
          "random_mean": {
            "n": 128,
            "correct": 86.0,
            "accuracy": 0.671875,
            "delta_choice_nll": {
              "n": 128,
              "mean": -0.012512541710416748,
              "ci95": [
                -0.026862759634345444,
                0.0011196676563874535
              ]
            },
            "delta_full_vocab_nll": {
              "n": 128,
              "mean": -0.026444575691130016,
              "ci95": [
                -0.04667934015402958,
                -0.00609696556639392
              ]
            },
            "mean_wire_bytes": 3010681.4375,
            "mean_effective_bits": 5.029042128333167
          }
        },
        "paired_choice_nll_contrasts": {
          "top_minus_random_mean": {
            "n": 128,
            "mean": 0.008677823126883292,
            "ci95": [
              -0.002910854669570521,
              0.02199686272092264
            ]
          },
          "top_minus_attention_bottom": {
            "n": 128,
            "mean": 0.009067577295269302,
            "ci95": [
              -0.0031336049262343122,
              0.022347788126796785
            ]
          }
        }
      }
    }
  },
  "uncertainty": "Paired question bootstrap; average random repetitions within each question; exploratory uncorrected95% intervals."
}
```


## kv_content_dependence_seed1234

```json
{
  "experiment": "kv_content_dependence_seed1234",
  "status": "completed",
  "timestamp": "2026-10-26 10:58:37",
  "manifest_sha256": "853655e532ecfff53f84ef6152218d595d8bc346a6a48ded775fa734e3d7dff1",
  "config": {
    "name": "kv_content_dependence_seed1234",
    "seed": 1234,
    "tasks": {
      "openbookqa": 128,
      "arc_challenge": 128
    },
    "training": false,
    "automatic_upload": false,
    "reader": "Original fixed KL-only epoch4/step1024; A+B and all model parameters frozen",
    "attention_backend": "sdpa",
    "repeats": 3,
    "free_gpu_gib": 12,
    "disk_reserve_gib": 12,
    "arms": [
      "true",
      "masked",
      "wrong_0",
      "wrong_1",
      "wrong_2",
      "random_target_stats_0",
      "random_target_stats_1",
      "random_target_stats_2",
      "random_donor_stats_0",
      "random_donor_stats_1",
      "random_donor_stats_2"
    ],
    "permutation_arms": [
      "reordered_true",
      "reordered_masked",
      "reordered_old_kv"
    ],
    "donors": "Seeded no-self assignments within task and option count minimizing length distance; singleton option-count bucket falls back to same-task nearest-length donors and is reported separately. No gold labels used.",
    "resize": "Same monotone nearest-neighbor map for K/V before receiver RoPE. Report exact-length/same-option-count donor subsets.",
    "noise": "Independent Gaussian K/V with per-layer/head/channel moments over token axis, three seeded repeats; compare target-derived vs donor-derived moments to expose retained aggregate target information.",
    "input": "Qwen prefills ONLY the common native anchor. Question/options enter solely via intervened KV. Same suffix, absolute positions and token count within each question; blocked condition masks all transferred columns.",
    "permutation": "Seeded full derangement of option content positions, independent of gold; remap gold label and regenerate Llama Sender states. Compare fresh reordered KV, masked KV and original-order KV resized to reordered length.",
    "scoring": "Original single-token labels, one forward with a fixed dummy candidate appended and score at preceding position. Audit agreement with legacy per-candidate scorer and dummy-token independence. Primary accuracy, choice NLL, paired prediction changes; uniform/majority-label baselines reported.",
    "scope": "Existing previously inspected held-out OBQA128 and ARC128. Test KV-content dependence and option-label binding, not whether world knowledge resides only in KV."
  },
  "checkpoint_identity": {
    "reader": {
      "path": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_reader_r16_answerkl_l3_to_q4_seed1234/runs/reader/epoch4_step1024.pt",
      "sha256": "96cde5b41f1a15434d24ef57f8eaf34e3af89af109c0b5d6cf378e60ab577314"
    },
    "stage_a": {
      "path": "/hy-tmp/yezhe/异构模型/grouped_headmix4_l3_to_q4_seed1234/runs/stage_a/best.pt",
      "sha256": "a9318f4b2f1d9e7981e0e50e2dc88654f0baaf89a080a0746017555c48ba5c84"
    },
    "stage_b": {
      "path": "/hy-tmp/yezhe/异构模型/grouped_headmix4_residual64_l3_to_q4_seed1234/runs/stage_b/last.pt",
      "sha256": "7e9f602d08409574c8ebe56242a08ba2c6861032b5ed70570a807c5777d775f2"
    }
  },
  "direct_inputs": {
    "anchor_ids": [
      151644,
      872,
      198,
      16141
    ],
    "suffix_ids": [
      13,
      151645,
      198,
      151644,
      77091,
      198,
      151667,
      271,
      151668,
      271,
      785,
      4396,
      2999,
      374,
      220
    ],
    "anchor_text": "<|im_start|>user\nAnswer",
    "suffix_text": ".<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe correct option is ",
    "question_option_text_directly_prefilled": false,
    "dummy_candidate": 32
  },
  "tasks": {
    "openbookqa": {
      "n": 128,
      "uniform_random_accuracy": 0.25,
      "majority_label_accuracy": 0.3125,
      "gold_label_counts": {
        "1": 28,
        "0": 40,
        "2": 34,
        "3": 26
      },
      "arms": {
        "true": {
          "n": 128,
          "correct": 92.0,
          "accuracy": 0.71875,
          "mean_choice_nll": 1.1677082279989577,
          "delta_choice_nll": {
            "n": 128,
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ]
          },
          "prediction_changed_vs_true": 0.0,
          "predicted_label_counts": {
            "1": 25,
            "0": 36,
            "2": 27,
            "3": 40
          }
        },
        "masked": {
          "n": 128,
          "correct": 26.0,
          "accuracy": 0.203125,
          "mean_choice_nll": 2.0542682979721576,
          "delta_choice_nll": {
            "n": 128,
            "mean": 0.8865600699731999,
            "ci95": [
              0.4904378237924647,
              1.250516724416647
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.515625,
            "ci95": [
              0.4140625,
              0.6171875
            ]
          },
          "prediction_changed_vs_true": 0.6875,
          "predicted_label_counts": {
            "3": 128
          }
        },
        "wrong": {
          "n": 128,
          "correct": 35.00000000000001,
          "accuracy": 0.2734375,
          "mean_choice_nll": 4.433161296212404,
          "delta_choice_nll": {
            "n": 128,
            "mean": 3.2654530682134464,
            "ci95": [
              2.778766388798389,
              3.7486016406254623
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.4453125,
            "ci95": [
              0.3541666666666667,
              0.5364583333333333
            ]
          },
          "prediction_changed_vs_true": 0.7135416666666665,
          "predicted_label_counts": {
            "3": 120,
            "2": 81,
            "0": 108,
            "1": 75
          }
        },
        "random_target_stats": {
          "n": 128,
          "correct": 31.999999999999996,
          "accuracy": 0.25,
          "mean_choice_nll": 1.7623620374167028,
          "delta_choice_nll": {
            "n": 128,
            "mean": 0.594653809417745,
            "ci95": [
              0.20302191536866018,
              0.9325773217463913
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.46875,
            "ci95": [
              0.3775390625,
              0.5521484375
            ]
          },
          "prediction_changed_vs_true": 0.7421875,
          "predicted_label_counts": {
            "3": 152,
            "1": 94,
            "2": 96,
            "0": 42
          }
        },
        "random_donor_stats": {
          "n": 128,
          "correct": 30.666666666666654,
          "accuracy": 0.23958333333333331,
          "mean_choice_nll": 1.7927458666381426,
          "delta_choice_nll": {
            "n": 128,
            "mean": 0.6250376386391849,
            "ci95": [
              0.23474691139504905,
              0.9772042587696143
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.47916666666666663,
            "ci95": [
              0.39583333333333337,
              0.5598958333333334
            ]
          },
          "prediction_changed_vs_true": 0.7526041666666665,
          "predicted_label_counts": {
            "3": 132,
            "1": 97,
            "2": 106,
            "0": 49
          }
        },
        "reordered_true": {
          "n": 128,
          "correct": 86.0,
          "accuracy": 0.671875,
          "mean_choice_nll": 1.2369179111750697,
          "delta_choice_nll": {
            "n": 128,
            "mean": 0.06920968317611198,
            "ci95": [
              -0.28450580926974567,
              0.4534373493105675
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.046875,
            "ci95": [
              -0.0390625,
              0.1328125
            ]
          },
          "prediction_changed_vs_true": 0.859375,
          "predicted_label_counts": {
            "3": 42,
            "0": 19,
            "1": 30,
            "2": 37
          }
        },
        "reordered_masked": {
          "n": 128,
          "correct": 25.0,
          "accuracy": 0.1953125,
          "mean_choice_nll": 2.005897948052734,
          "delta_choice_nll": {
            "n": 128,
            "mean": 0.8381897200537765,
            "ci95": [
              0.42776314714805325,
              1.2228276526500026
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.5234375,
            "ci95": [
              0.4140625,
              0.625
            ]
          },
          "prediction_changed_vs_true": 0.6875,
          "predicted_label_counts": {
            "3": 128
          }
        },
        "reordered_old_kv": {
          "n": 128,
          "correct": 15.0,
          "accuracy": 0.1171875,
          "mean_choice_nll": 5.574689677575407,
          "delta_choice_nll": {
            "n": 128,
            "mean": 4.406981449576449,
            "ci95": [
              3.6318560020522823,
              5.136236726593758
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.6015625,
            "ci95": [
              0.484375,
              0.7111328124999998
            ]
          },
          "prediction_changed_vs_true": 0.0,
          "predicted_label_counts": {
            "1": 25,
            "0": 36,
            "2": 27,
            "3": 40
          }
        }
      },
      "option_reordering": {
        "gold_label_changed": 128,
        "content_prediction_tracking": {
          "n": 128,
          "mean": 0.671875,
          "ci95": [
            0.59375,
            0.75
          ]
        },
        "same_label_rate": 0.140625,
        "original_correct_n": 92,
        "original_correct_retained": 72,
        "old_kv_keeps_original_prediction": 1.0
      },
      "wrong_exact_length_subset": {
        "question_count": 116,
        "donor_pair_count": 286,
        "true_accuracy": 0.7241379310344828,
        "wrong_accuracy": 0.24425287356321837
      },
      "wrong_same_option_count_subset": {
        "question_count": 128,
        "donor_pair_count": 384,
        "true_accuracy": 0.71875,
        "wrong_accuracy": 0.2734375
      },
      "true_baseline_usable": true
    },
    "arc_challenge": {
      "n": 128,
      "uniform_random_accuracy": 0.25065104166666663,
      "majority_label_accuracy": 0.2890625,
      "gold_label_counts": {
        "2": 37,
        "1": 34,
        "3": 24,
        "0": 33
      },
      "arms": {
        "true": {
          "n": 128,
          "correct": 86.0,
          "accuracy": 0.671875,
          "mean_choice_nll": 1.0236651278380577,
          "delta_choice_nll": {
            "n": 128,
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ]
          },
          "prediction_changed_vs_true": 0.0,
          "predicted_label_counts": {
            "2": 42,
            "1": 25,
            "0": 26,
            "3": 35
          }
        },
        "masked": {
          "n": 128,
          "correct": 25.0,
          "accuracy": 0.1953125,
          "mean_choice_nll": 2.0054005237761885,
          "delta_choice_nll": {
            "n": 128,
            "mean": 0.9817353959381308,
            "ci95": [
              0.6647459606509287,
              1.2600391815590612
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.4765625,
            "ci95": [
              0.375,
              0.5703125
            ]
          },
          "prediction_changed_vs_true": 0.71875,
          "predicted_label_counts": {
            "3": 126,
            "1": 1,
            "2": 1
          }
        },
        "wrong": {
          "n": 128,
          "correct": 29.33333333333332,
          "accuracy": 0.22916666666666666,
          "mean_choice_nll": 4.4757791804547455,
          "delta_choice_nll": {
            "n": 128,
            "mean": 3.4521140526166882,
            "ci95": [
              3.022503400142962,
              3.8617167726977755
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.44270833333333337,
            "ci95": [
              0.3488932291666667,
              0.5365234375
            ]
          },
          "prediction_changed_vs_true": 0.7994791666666666,
          "predicted_label_counts": {
            "2": 126,
            "1": 78,
            "3": 103,
            "0": 77
          }
        },
        "random_target_stats": {
          "n": 128,
          "correct": 30.999999999999993,
          "accuracy": 0.2421875,
          "mean_choice_nll": 1.770614922589933,
          "delta_choice_nll": {
            "n": 128,
            "mean": 0.7469497947518751,
            "ci95": [
              0.45771783741512967,
              1.0045517616870232
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.4296875,
            "ci95": [
              0.34375,
              0.5156901041666666
            ]
          },
          "prediction_changed_vs_true": 0.7083333333333333,
          "predicted_label_counts": {
            "3": 169,
            "0": 28,
            "2": 117,
            "1": 70
          }
        },
        "random_donor_stats": {
          "n": 128,
          "correct": 31.666666666666657,
          "accuracy": 0.24739583333333331,
          "mean_choice_nll": 1.7981842238611232,
          "delta_choice_nll": {
            "n": 128,
            "mean": 0.7745190960230655,
            "ci95": [
              0.4641844692855618,
              1.0436069257503837
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.4244791666666667,
            "ci95": [
              0.33333333333333337,
              0.5104817708333333
            ]
          },
          "prediction_changed_vs_true": 0.7369791666666666,
          "predicted_label_counts": {
            "3": 149,
            "0": 24,
            "1": 95,
            "2": 116
          }
        },
        "reordered_true": {
          "n": 128,
          "correct": 86.0,
          "accuracy": 0.671875,
          "mean_choice_nll": 0.9766372547577191,
          "delta_choice_nll": {
            "n": 128,
            "mean": -0.04702787308033862,
            "ci95": [
              -0.3602581996167579,
              0.2920883509659944
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.0,
            "ci95": [
              -0.0859375,
              0.09375
            ]
          },
          "prediction_changed_vs_true": 0.828125,
          "predicted_label_counts": {
            "0": 37,
            "1": 29,
            "3": 40,
            "2": 22
          }
        },
        "reordered_masked": {
          "n": 128,
          "correct": 31.0,
          "accuracy": 0.2421875,
          "mean_choice_nll": 1.9466236690059304,
          "delta_choice_nll": {
            "n": 128,
            "mean": 0.9229585411678727,
            "ci95": [
              0.5493076286498507,
              1.286344468736523
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.4296875,
            "ci95": [
              0.3046875,
              0.546875
            ]
          },
          "prediction_changed_vs_true": 0.71875,
          "predicted_label_counts": {
            "3": 126,
            "1": 1,
            "2": 1
          }
        },
        "reordered_old_kv": {
          "n": 128,
          "correct": 15.0,
          "accuracy": 0.1171875,
          "mean_choice_nll": 5.434759765641502,
          "delta_choice_nll": {
            "n": 128,
            "mean": 4.411094637803444,
            "ci95": [
              3.6903200311773348,
              5.09949439172415
            ]
          },
          "accuracy_drop_vs_true": {
            "n": 128,
            "mean": 0.5546875,
            "ci95": [
              0.4296875,
              0.671875
            ]
          },
          "prediction_changed_vs_true": 0.0,
          "predicted_label_counts": {
            "2": 42,
            "1": 25,
            "0": 26,
            "3": 35
          }
        }
      },
      "option_reordering": {
        "gold_label_changed": 128,
        "content_prediction_tracking": {
          "n": 128,
          "mean": 0.65625,
          "ci95": [
            0.5703125,
            0.7421875
          ]
        },
        "same_label_rate": 0.171875,
        "original_correct_n": 86,
        "original_correct_retained": 69,
        "old_kv_keeps_original_prediction": 1.0
      },
      "wrong_exact_length_subset": {
        "question_count": 88,
        "donor_pair_count": 165,
        "true_accuracy": 0.6931818181818182,
        "wrong_accuracy": 0.2234848484848485
      },
      "wrong_same_option_count_subset": {
        "question_count": 127,
        "donor_pair_count": 381,
        "true_accuracy": 0.6692913385826772,
        "wrong_accuracy": 0.22572178477690286
      },
      "true_baseline_usable": true
    }
  },
  "interpretation": "Target-moment Gaussian noise retains aggregate target information; compare donor-moment noise and mismatched natural KV before attributing effects to model priors. MCQ questions/options themselves enter through KV.",
  "uncertainty": "Question-level paired bootstrap, average repeats within question; uncorrected exploratory95% intervals."
}
```


## receiver_selective_evidencekv_sciq_seed1234

```json
{
  "experiment": "receiver_selective_evidencekv_sciq_seed1234",
  "status": "completed",
  "timestamp": "2026-10-26 17:06:31",
  "manifest_sha256": "e22f06b59ad3da36897fca2899e98af5659572866c68332fce56a4797e531de1",
  "config": {
    "name": "receiver_selective_evidencekv_sciq_seed1234",
    "seed": 1234,
    "reference": "/hy-tmp/yezhe/异构模型/attention_mixed_precision_senderkv_2wiki128_l3_to_q4_frozen_klreader_seed1234",
    "source_parquet": "/hy-tmp/yezhe/异构模型/decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234/sciq_frozen64/data/test.parquet",
    "previous_sciq": "/hy-tmp/yezhe/异构模型/decoupled_memory_reader_obqa_fact_l3_to_q4_seed1234/sciq_frozen64/prepared/test.jsonl",
    "calibration_count": 32,
    "test_count": 128,
    "max_evidence_tokens": 384,
    "max_total_tokens": 768,
    "conditions": [
      "clean",
      "wrong",
      "partial_wrong",
      "gaussian"
    ],
    "partial_corruption_fraction": 0.5,
    "reader": "No Q/O adapters installed or loaded. Original Qwen and existing grouped Stage-A + rank64 residual Stage-B frozen.",
    "attention_backend": "sdpa",
    "free_gpu_gib": 12,
    "disk_reserve_gib": 10,
    "selector": "Whole packet accept/reject. Features: change in mean native question token log probability, and decrease in candidate entropy. Global thresholds selected only on32 calibration questions across equal corruption conditions, then locked before test inference. Include fixed zero-threshold and always-accept/reject controls.",
    "question_inputs": "Question and all options are read natively by Qwen after the evidence cache. Sender sees support only. No answer labels or target answers in Sender prompt or gate features.",
    "scope": "Pilot using cached official SciQ test parquet split into disjoint calibration32 and evaluation128, excluding previous SciQ64 and exact known training/validation questions. Not an official SciQ leaderboard evaluation; oracle support supplied.",
    "training": false,
    "automatic_upload": false
  },
  "selector": {
    "rules": {
      "always_accept": {
        "kind": "always_accept"
      },
      "always_reject": {
        "kind": "always_reject"
      },
      "question_gain_calibrated": {
        "kind": "question_gain",
        "threshold": 1e+30
      },
      "question_gain_zero": {
        "kind": "question_gain",
        "threshold": 0.0
      },
      "confidence_gain_calibrated": {
        "kind": "confidence_gain",
        "threshold": 0.0
      },
      "confidence_gain_zero": {
        "kind": "confidence_gain",
        "threshold": 0.0
      }
    },
    "selected_rule": "confidence_gain_calibrated",
    "calibration_scores": {
      "always_reject": {
        "correct": 124,
        "accept_count": 0
      },
      "always_accept": {
        "correct": 115,
        "accept_count": 128
      },
      "question_gain_calibrated": {
        "correct": 124,
        "accept_count": 0
      },
      "confidence_gain_calibrated": {
        "correct": 126,
        "accept_count": 12
      }
    },
    "n": 128,
    "manifest_sha256": "e22f06b59ad3da36897fca2899e98af5659572866c68332fce56a4797e531de1",
    "locked_at": "2026-10-26 17:04:19",
    "calibration_only": true,
    "thresholds_before_test": true,
    "question_count": 32
  },
  "n_questions": 128,
  "condition_count": 4,
  "conditions": {
    "clean": {
      "always_accept": {
        "n": 128,
        "correct": 119,
        "accuracy": 0.9296875,
        "mean_choice_nll": 0.2233953326825614,
        "accept_rate": 1.0,
        "accuracy_gain_vs_accept": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": -0.03125,
          "ci95": [
            -0.078125,
            0.0078125
          ],
          "n": 128
        }
      },
      "always_reject": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.2864568213494774,
        "accept_rate": 0.0,
        "accuracy_gain_vs_accept": {
          "mean": 0.03125,
          "ci95": [
            -0.0078125,
            0.078125
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "question_gain_calibrated": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.2864568213494774,
        "accept_rate": 0.0,
        "accuracy_gain_vs_accept": {
          "mean": 0.03125,
          "ci95": [
            -0.0078125,
            0.078125
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "question_gain_zero": {
        "n": 128,
        "correct": 119,
        "accuracy": 0.9296875,
        "mean_choice_nll": 0.22329203477589743,
        "accept_rate": 0.9765625,
        "accuracy_gain_vs_accept": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": -0.03125,
          "ci95": [
            -0.078125,
            0.0078125
          ],
          "n": 128
        }
      },
      "confidence_gain_calibrated": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.29040904880386026,
        "accept_rate": 0.109375,
        "accuracy_gain_vs_accept": {
          "mean": 0.03125,
          "ci95": [
            -0.0078125,
            0.078125
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "confidence_gain_zero": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.29040904880386026,
        "accept_rate": 0.109375,
        "accuracy_gain_vs_accept": {
          "mean": 0.03125,
          "ci95": [
            -0.0078125,
            0.078125
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "oracle_best_of_two": {
        "correct": 125,
        "n": 128
      },
      "useful_packet_count": 2,
      "harmful_packet_count": 6
    },
    "wrong": {
      "always_accept": {
        "n": 128,
        "correct": 112,
        "accuracy": 0.875,
        "mean_choice_nll": 0.732188953010737,
        "accept_rate": 1.0,
        "accuracy_gain_vs_accept": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": -0.0859375,
          "ci95": [
            -0.140625,
            -0.03125
          ],
          "n": 128
        }
      },
      "always_reject": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.2864568213494774,
        "accept_rate": 0.0,
        "accuracy_gain_vs_accept": {
          "mean": 0.0859375,
          "ci95": [
            0.03125,
            0.140625
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "question_gain_calibrated": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.2864568213494774,
        "accept_rate": 0.0,
        "accuracy_gain_vs_accept": {
          "mean": 0.0859375,
          "ci95": [
            0.03125,
            0.140625
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "question_gain_zero": {
        "n": 128,
        "correct": 122,
        "accuracy": 0.953125,
        "mean_choice_nll": 0.3697769692606312,
        "accept_rate": 0.3671875,
        "accuracy_gain_vs_accept": {
          "mean": 0.078125,
          "ci95": [
            0.03125,
            0.1328125
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": -0.0078125,
          "ci95": [
            -0.0234375,
            0.0
          ],
          "n": 128
        }
      },
      "confidence_gain_calibrated": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.3194557235769338,
        "accept_rate": 0.09375,
        "accuracy_gain_vs_accept": {
          "mean": 0.0859375,
          "ci95": [
            0.03125,
            0.140625
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "confidence_gain_zero": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.3194557235769338,
        "accept_rate": 0.09375,
        "accuracy_gain_vs_accept": {
          "mean": 0.0859375,
          "ci95": [
            0.03125,
            0.140625
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "oracle_best_of_two": {
        "correct": 124,
        "n": 128
      },
      "useful_packet_count": 1,
      "harmful_packet_count": 12
    },
    "partial_wrong": {
      "always_accept": {
        "n": 128,
        "correct": 116,
        "accuracy": 0.90625,
        "mean_choice_nll": 0.46589067014383495,
        "accept_rate": 1.0,
        "accuracy_gain_vs_accept": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": -0.0546875,
          "ci95": [
            -0.1015625,
            -0.0078125
          ],
          "n": 128
        }
      },
      "always_reject": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.2864568213494774,
        "accept_rate": 0.0,
        "accuracy_gain_vs_accept": {
          "mean": 0.0546875,
          "ci95": [
            0.0078125,
            0.1015625
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "question_gain_calibrated": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.2864568213494774,
        "accept_rate": 0.0,
        "accuracy_gain_vs_accept": {
          "mean": 0.0546875,
          "ci95": [
            0.0078125,
            0.1015625
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "question_gain_zero": {
        "n": 128,
        "correct": 118,
        "accuracy": 0.921875,
        "mean_choice_nll": 0.5025440606918113,
        "accept_rate": 0.9375,
        "accuracy_gain_vs_accept": {
          "mean": 0.015625,
          "ci95": [
            0.0,
            0.0390625
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": -0.0390625,
          "ci95": [
            -0.0859375,
            0.0078125
          ],
          "n": 128
        }
      },
      "confidence_gain_calibrated": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.283739045872105,
        "accept_rate": 0.0859375,
        "accuracy_gain_vs_accept": {
          "mean": 0.0546875,
          "ci95": [
            0.0078125,
            0.1015625
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "confidence_gain_zero": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.283739045872105,
        "accept_rate": 0.0859375,
        "accuracy_gain_vs_accept": {
          "mean": 0.0546875,
          "ci95": [
            0.0078125,
            0.1015625
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "oracle_best_of_two": {
        "correct": 125,
        "n": 128
      },
      "useful_packet_count": 2,
      "harmful_packet_count": 9
    },
    "gaussian": {
      "always_accept": {
        "n": 128,
        "correct": 121,
        "accuracy": 0.9453125,
        "mean_choice_nll": 0.26223241028992345,
        "accept_rate": 1.0,
        "accuracy_gain_vs_accept": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": -0.015625,
          "ci95": [
            -0.046875,
            0.015625
          ],
          "n": 128
        }
      },
      "always_reject": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.2864568213494774,
        "accept_rate": 0.0,
        "accuracy_gain_vs_accept": {
          "mean": 0.015625,
          "ci95": [
            -0.015625,
            0.046875
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "question_gain_calibrated": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.2864568213494774,
        "accept_rate": 0.0,
        "accuracy_gain_vs_accept": {
          "mean": 0.015625,
          "ci95": [
            -0.015625,
            0.046875
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "question_gain_zero": {
        "n": 128,
        "correct": 121,
        "accuracy": 0.9453125,
        "mean_choice_nll": 0.26219142715666394,
        "accept_rate": 0.96875,
        "accuracy_gain_vs_accept": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": -0.015625,
          "ci95": [
            -0.046875,
            0.015625
          ],
          "n": 128
        }
      },
      "confidence_gain_calibrated": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.32539310372545033,
        "accept_rate": 0.046875,
        "accuracy_gain_vs_accept": {
          "mean": 0.015625,
          "ci95": [
            -0.015625,
            0.046875
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "confidence_gain_zero": {
        "n": 128,
        "correct": 123,
        "accuracy": 0.9609375,
        "mean_choice_nll": 0.32539310372545033,
        "accept_rate": 0.046875,
        "accuracy_gain_vs_accept": {
          "mean": 0.015625,
          "ci95": [
            -0.015625,
            0.046875
          ],
          "n": 128
        },
        "accuracy_gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        }
      },
      "oracle_best_of_two": {
        "correct": 124,
        "n": 128
      },
      "useful_packet_count": 1,
      "harmful_packet_count": 3
    }
  },
  "balanced_mixture": {
    "always_accept": {
      "accuracy": 0.9140625,
      "accept_rate": 1.0,
      "question_bootstrap_accuracy": {
        "mean": 0.9140625,
        "ci95": [
          0.873046875,
          0.947265625
        ],
        "n": 128
      },
      "accuracy_gain_vs_accept": {
        "mean": 0.0,
        "ci95": [
          0.0,
          0.0
        ],
        "n": 128
      },
      "accuracy_gain_vs_reject": {
        "mean": -0.046875,
        "ci95": [
          -0.080078125,
          -0.013671875
        ],
        "n": 128
      },
      "rate_matched_random_expected_accuracy": 0.9140625
    },
    "always_reject": {
      "accuracy": 0.9609375,
      "accept_rate": 0.0,
      "question_bootstrap_accuracy": {
        "mean": 0.9609375,
        "ci95": [
          0.921875,
          0.9921875
        ],
        "n": 128
      },
      "accuracy_gain_vs_accept": {
        "mean": 0.046875,
        "ci95": [
          0.013671875,
          0.080078125
        ],
        "n": 128
      },
      "accuracy_gain_vs_reject": {
        "mean": 0.0,
        "ci95": [
          0.0,
          0.0
        ],
        "n": 128
      },
      "rate_matched_random_expected_accuracy": 0.9609375
    },
    "question_gain_calibrated": {
      "accuracy": 0.9609375,
      "accept_rate": 0.0,
      "question_bootstrap_accuracy": {
        "mean": 0.9609375,
        "ci95": [
          0.921875,
          0.9921875
        ],
        "n": 128
      },
      "accuracy_gain_vs_accept": {
        "mean": 0.046875,
        "ci95": [
          0.013671875,
          0.080078125
        ],
        "n": 128
      },
      "accuracy_gain_vs_reject": {
        "mean": 0.0,
        "ci95": [
          0.0,
          0.0
        ],
        "n": 128
      },
      "rate_matched_random_expected_accuracy": 0.9609375
    },
    "question_gain_zero": {
      "accuracy": 0.9375,
      "accept_rate": 0.8125,
      "question_bootstrap_accuracy": {
        "mean": 0.9375,
        "ci95": [
          0.90234375,
          0.966796875
        ],
        "n": 128
      },
      "accuracy_gain_vs_accept": {
        "mean": 0.0234375,
        "ci95": [
          0.0078125,
          0.041015625
        ],
        "n": 128
      },
      "accuracy_gain_vs_reject": {
        "mean": -0.0234375,
        "ci95": [
          -0.046875,
          0.0
        ],
        "n": 128
      },
      "rate_matched_random_expected_accuracy": 0.9228515625
    },
    "confidence_gain_calibrated": {
      "accuracy": 0.9609375,
      "accept_rate": 0.083984375,
      "question_bootstrap_accuracy": {
        "mean": 0.9609375,
        "ci95": [
          0.921875,
          0.9921875
        ],
        "n": 128
      },
      "accuracy_gain_vs_accept": {
        "mean": 0.046875,
        "ci95": [
          0.013671875,
          0.080078125
        ],
        "n": 128
      },
      "accuracy_gain_vs_reject": {
        "mean": 0.0,
        "ci95": [
          0.0,
          0.0
        ],
        "n": 128
      },
      "rate_matched_random_expected_accuracy": 0.957000732421875
    },
    "confidence_gain_zero": {
      "accuracy": 0.9609375,
      "accept_rate": 0.083984375,
      "question_bootstrap_accuracy": {
        "mean": 0.9609375,
        "ci95": [
          0.921875,
          0.9921875
        ],
        "n": 128
      },
      "accuracy_gain_vs_accept": {
        "mean": 0.046875,
        "ci95": [
          0.013671875,
          0.080078125
        ],
        "n": 128
      },
      "accuracy_gain_vs_reject": {
        "mean": 0.0,
        "ci95": [
          0.0,
          0.0
        ],
        "n": 128
      },
      "rate_matched_random_expected_accuracy": 0.957000732421875
    }
  },
  "caveats": [
    "No Reader installed; Translator frozen. Evidence-only interface differs from Translator training interface.",
    "Candidate entropy and question likelihood are uncalibrated proxies; neither guarantees truth.",
    "Gaussian retains target aggregate moments; partial replacement is synthetic corruption.",
    "Calibration32 and evaluation128 come from disjoint subsets of official SciQ test, excluding prior64; not leaderboard scoring.",
    "Selectors require accept and reject evaluations; this pilot does not claim inference compute savings.",
    "Oracle is diagnostic and reads gold, unavailable at inference. Bootstrap by question; exploratory intervals."
  ],
  "native_clean_evidence": {
    "n": 128,
    "correct": 128
  }
}
```


## receiver_selective_evidencekv_hotpotqa_seed1234

```json
{
  "experiment": "receiver_selective_evidencekv_hotpotqa_seed1234",
  "status": "completed_pilot",
  "timestamp": "2026-10-26 21:55:02",
  "n": 32,
  "manifest_sha256": "c991bbe48c46c7e18d23cf3c32a95203513d4ccbec65724af2c96b775669dcc4",
  "conditions": {
    "question_only_compact": {
      "n": 32,
      "correct": 2.0,
      "em": 0.0625,
      "f1": 0.12754779942279942,
      "answer_token_nll": 5.906183751532808,
      "eos": 1.0,
      "generated_tokens": 11.875,
      "seconds": 0.6746410131454468
    },
    "reject_all": {
      "n": 32,
      "correct": 1.0,
      "em": 0.03125,
      "f1": 0.095509984296749,
      "answer_token_nll": 6.22376719606109,
      "eos": 1.0,
      "generated_tokens": 15.46875,
      "seconds": 0.8207120597362518
    },
    "native_evidence": {
      "n": 32,
      "correct": 17.0,
      "em": 0.53125,
      "f1": 0.6200181159420289,
      "answer_token_nll": 2.639604116634213,
      "eos": 1.0,
      "generated_tokens": 7.875,
      "seconds": 0.47421494871377945
    },
    "clean": {
      "n": 32,
      "correct": 5.0,
      "em": 0.15625,
      "f1": 0.25580357142857146,
      "answer_token_nll": 3.6282992833293974,
      "eos": 1.0,
      "generated_tokens": 6.96875,
      "seconds": 0.41982241719961166
    },
    "wrong": {
      "n": 32,
      "correct": 1.0,
      "em": 0.03125,
      "f1": 0.06096813725490196,
      "answer_token_nll": 4.868006609380245,
      "eos": 0.96875,
      "generated_tokens": 10.03125,
      "seconds": 0.575363427400589
    },
    "partial_wrong": {
      "n": 32,
      "correct": 3.0,
      "em": 0.09375,
      "f1": 0.14097222222222222,
      "answer_token_nll": 3.796427723020315,
      "eos": 1.0,
      "generated_tokens": 7.15625,
      "seconds": 0.42653344571590424
    }
  },
  "comparisons": {
    "native_evidence_minus_question_only_compact": {
      "em": {
        "mean": 0.46875,
        "ci95": [
          0.3125,
          0.65625
        ],
        "n": 32
      },
      "f1": {
        "mean": 0.49247031651922957,
        "ci95": [
          0.3325564211603614,
          0.6482271031353911
        ],
        "n": 32
      }
    },
    "native_evidence_minus_reject_all": {
      "em": {
        "mean": 0.5,
        "ci95": [
          0.3125,
          0.6875
        ],
        "n": 32
      },
      "f1": {
        "mean": 0.52450813164528,
        "ci95": [
          0.36651512383609575,
          0.6804488744196621
        ],
        "n": 32
      }
    },
    "clean_minus_reject_all": {
      "em": {
        "mean": 0.125,
        "ci95": [
          0.03125,
          0.25
        ],
        "n": 32
      },
      "f1": {
        "mean": 0.16029358713182243,
        "ci95": [
          0.06119087407690348,
          0.27686206205967223
        ],
        "n": 32
      }
    },
    "clean_minus_question_only_compact": {
      "em": {
        "mean": 0.09375,
        "ci95": [
          -0.03125,
          0.21875
        ],
        "n": 32
      },
      "f1": {
        "mean": 0.12825577200577198,
        "ci95": [
          0.003500405844155842,
          0.25946958423520905
        ],
        "n": 32
      }
    },
    "clean_minus_wrong": {
      "em": {
        "mean": 0.125,
        "ci95": [
          0.03125,
          0.25
        ],
        "n": 32
      },
      "f1": {
        "mean": 0.19483543417366947,
        "ci95": [
          0.090625,
          0.31254376750700275
        ],
        "n": 32
      }
    },
    "clean_minus_partial_wrong": {
      "em": {
        "mean": 0.0625,
        "ci95": [
          0.0,
          0.15625
        ],
        "n": 32
      },
      "f1": {
        "mean": 0.1148313492063492,
        "ci95": [
          0.027777777777777776,
          0.21899801587301587
        ],
        "n": 32
      }
    }
  },
  "reserved_test_evaluated": false,
  "pilot_gate": {
    "checks": {
      "native_evidence_useful": true,
      "translated_gain_large_enough": true,
      "translated_paired_ci_positive": true,
      "clean_better_than_wrong": true
    },
    "passed": true,
    "next_step": "Ready to lock selector on pilot and evaluate reserved128"
  },
  "helpful_clean_em": 4,
  "harmful_clean_em": 0,
  "caveats": [
    "Controlled bridge questions with two gold paragraphs, not full HotpotQA distractor benchmark.",
    "Only calibration32 evaluated; no selector fitted and reserved128 untouched.",
    "Wrong/partial KV corruption is synthetic; length matching and resizing do not model all real transmission errors.",
    "Masks hide evidence keys from all question and answer tokens, retaining common anchor and position slots.",
    "No Reader, training or quantization; original grouped Stage-A and residual Stage-B frozen.",
    "Short-answer generation scored with normalized whole-output EM/F1; no answer-dependent parsing."
  ]
}
```


## receiver_selective_evidencekv_hotpotqa_seed1234/selector_eval128_seed1234

```json
{
  "experiment": "receiver_selective_evidencekv_hotpotqa_seed1234/selector_eval128_seed1234",
  "status": "completed",
  "timestamp": "2026-10-26 22:19:55",
  "n": 128,
  "manifest_sha256": "68d169274cf92893a6c727cd06408b5b099c650aab20a9cb4ef2ffd078766b9c",
  "selector": {
    "rules": {
      "always_reject": {
        "kind": "always_reject"
      },
      "always_accept": {
        "kind": "always_accept"
      },
      "question_gain_calibrated": {
        "kind": "question_gain",
        "threshold": 0.0
      },
      "question_gain_zero": {
        "kind": "question_gain",
        "threshold": 0.0
      },
      "first_token_entropy_gain_calibrated": {
        "kind": "first_token_entropy_gain",
        "threshold": -1e+30
      },
      "first_token_entropy_gain_zero": {
        "kind": "first_token_entropy_gain",
        "threshold": 0.0
      }
    },
    "selected_rule": "question_gain_calibrated",
    "calibration_scores": {
      "always_reject": {
        "f1": 0.09550998429674899,
        "em": 0.03125,
        "accept_count": 0
      },
      "always_accept": {
        "f1": 0.15258131030189853,
        "em": 0.09375,
        "accept_count": 96
      },
      "question_gain_calibrated": {
        "f1": 0.16481676003734827,
        "em": 0.10416666666666667,
        "accept_count": 80
      },
      "question_gain_zero": {
        "f1": 0.16481676003734827,
        "em": 0.10416666666666667,
        "accept_count": 80
      },
      "first_token_entropy_gain_calibrated": {
        "f1": 0.15258131030189853,
        "em": 0.09375,
        "accept_count": 96
      },
      "first_token_entropy_gain_zero": {
        "f1": 0.13546909925586395,
        "em": 0.07291666666666667,
        "accept_count": 20
      }
    },
    "question_count": 32,
    "case_count": 96,
    "conditions": [
      "clean",
      "wrong",
      "partial_wrong"
    ],
    "primary_metric": "equal-condition mean F1",
    "label_free_features": true,
    "manifest_sha256": "68d169274cf92893a6c727cd06408b5b099c650aab20a9cb4ef2ffd078766b9c",
    "locked_at": "2026-10-26 22:08:57",
    "calibration_only": true
  },
  "baselines": {
    "question_only_compact": {
      "n": 128,
      "correct": 11.0,
      "em": 0.0859375,
      "f1": 0.14444848962312196,
      "answer_token_nll": 5.4207722371506275,
      "eos": 1.0,
      "generated_tokens": 12.71875
    },
    "reject_all": {
      "n": 128,
      "correct": 10.0,
      "em": 0.078125,
      "f1": 0.14843733931509667,
      "answer_token_nll": 5.689652765387562,
      "eos": 0.9921875,
      "generated_tokens": 13.640625
    },
    "native_evidence": {
      "n": 128,
      "correct": 68.0,
      "em": 0.53125,
      "f1": 0.716235426644434,
      "answer_token_nll": 2.620600922802626,
      "eos": 1.0,
      "generated_tokens": 7.8828125
    },
    "clean": {
      "n": 128,
      "correct": 17.0,
      "em": 0.1328125,
      "f1": 0.3027685759333378,
      "answer_token_nll": 3.174168522686614,
      "eos": 0.9921875,
      "generated_tokens": 7.4453125
    },
    "wrong": {
      "n": 128,
      "correct": 5.0,
      "em": 0.0390625,
      "f1": 0.08104901051329623,
      "answer_token_nll": 5.115792689903174,
      "eos": 1.0,
      "generated_tokens": 9.1328125
    },
    "partial_wrong": {
      "n": 128,
      "correct": 9.0,
      "em": 0.0703125,
      "f1": 0.15200563325563327,
      "answer_token_nll": 3.9565023292670958,
      "eos": 0.9921875,
      "generated_tokens": 8.265625
    }
  },
  "conditions": {
    "clean": {
      "n": 128,
      "selected_rule": "question_gain_calibrated",
      "accept_count": 126,
      "em": 0.1328125,
      "f1": 0.2996435759333378,
      "gain_vs_always_accept": {
        "em": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        },
        "f1": {
          "mean": -0.003125,
          "ci95": [
            -0.009375000000000001,
            0.0
          ],
          "n": 128
        }
      },
      "gain_vs_always_reject": {
        "em": {
          "mean": 0.0546875,
          "ci95": [
            -0.0078125,
            0.1171875
          ],
          "n": 128
        },
        "f1": {
          "mean": 0.1512062366182411,
          "ci95": [
            0.08296160683662432,
            0.21946713997414077
          ],
          "n": 128
        }
      },
      "clean_or_corrupt_useful_em": 12,
      "clean_or_corrupt_harmful_em": 5,
      "oracle_em": 0.171875,
      "prediction_changes_vs_reject": 119
    },
    "wrong": {
      "n": 128,
      "selected_rule": "question_gain_calibrated",
      "accept_count": 77,
      "em": 0.0703125,
      "f1": 0.12005671696126002,
      "gain_vs_always_accept": {
        "em": {
          "mean": 0.03125,
          "ci95": [
            0.0078125,
            0.0625
          ],
          "n": 128
        },
        "f1": {
          "mean": 0.03900770644796381,
          "ci95": [
            0.011604247818919234,
            0.0723919124941643
          ],
          "n": 128
        }
      },
      "gain_vs_always_reject": {
        "em": {
          "mean": -0.0078125,
          "ci95": [
            -0.0234375,
            0.0
          ],
          "n": 128
        },
        "f1": {
          "mean": -0.02838062235383664,
          "ci95": [
            -0.05750387788055199,
            -0.003815150468331296
          ],
          "n": 128
        }
      },
      "clean_or_corrupt_useful_em": 0,
      "clean_or_corrupt_harmful_em": 5,
      "oracle_em": 0.078125,
      "prediction_changes_vs_reject": 72
    },
    "partial_wrong": {
      "n": 128,
      "selected_rule": "question_gain_calibrated",
      "accept_count": 118,
      "em": 0.0859375,
      "f1": 0.16085979992229993,
      "gain_vs_always_accept": {
        "em": {
          "mean": 0.015625,
          "ci95": [
            0.0,
            0.0390625
          ],
          "n": 128
        },
        "f1": {
          "mean": 0.008854166666666666,
          "ci95": [
            -0.003125,
            0.028645833333333336
          ],
          "n": 128
        }
      },
      "gain_vs_always_reject": {
        "em": {
          "mean": 0.0078125,
          "ci95": [
            -0.046875,
            0.0625
          ],
          "n": 128
        },
        "f1": {
          "mean": 0.012422460607203254,
          "ci95": [
            -0.04424156272903975,
            0.06876993589539544
          ],
          "n": 128
        }
      },
      "clean_or_corrupt_useful_em": 6,
      "clean_or_corrupt_harmful_em": 7,
      "oracle_em": 0.125,
      "prediction_changes_vs_reject": 110
    }
  },
  "balanced_mixture": {
    "always_reject": {
      "accept_rate": 0.0,
      "em": {
        "score": {
          "mean": 0.078125,
          "ci95": [
            0.03125,
            0.125
          ],
          "n": 128
        },
        "gain_vs_accept": {
          "mean": -0.0026041666666666644,
          "ci95": [
            -0.04427083333333333,
            0.04427083333333333
          ],
          "n": 128
        },
        "gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        },
        "matched_accept_rate_random_expected": 0.078125
      },
      "f1": {
        "score": {
          "mean": 0.14843733931509667,
          "ci95": [
            0.09960447949849376,
            0.202474353446033
          ],
          "n": 128
        },
        "gain_vs_accept": {
          "mean": -0.030170400585659073,
          "ci95": [
            -0.07567593811187104,
            0.01753893232187462
          ],
          "n": 128
        },
        "gain_vs_reject": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        },
        "matched_accept_rate_random_expected": 0.14843733931509667
      }
    },
    "always_accept": {
      "accept_rate": 1.0,
      "em": {
        "score": {
          "mean": 0.08072916666666666,
          "ci95": [
            0.04427083333333333,
            0.12239583333333333
          ],
          "n": 128
        },
        "gain_vs_accept": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        },
        "gain_vs_reject": {
          "mean": 0.0026041666666666644,
          "ci95": [
            -0.04427083333333333,
            0.04427083333333333
          ],
          "n": 128
        },
        "matched_accept_rate_random_expected": 0.08072916666666666
      },
      "f1": {
        "score": {
          "mean": 0.17860773990075574,
          "ci95": [
            0.1372615518573756,
            0.22419966288502136
          ],
          "n": 128
        },
        "gain_vs_accept": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        },
        "gain_vs_reject": {
          "mean": 0.030170400585659073,
          "ci95": [
            -0.01753893232187463,
            0.07567593811187102
          ],
          "n": 128
        },
        "matched_accept_rate_random_expected": 0.17860773990075574
      }
    },
    "question_gain_calibrated": {
      "accept_rate": 0.8359375,
      "em": {
        "score": {
          "mean": 0.09635416666666666,
          "ci95": [
            0.05729166666666666,
            0.140625
          ],
          "n": 128
        },
        "gain_vs_accept": {
          "mean": 0.015625,
          "ci95": [
            0.0026041666666666665,
            0.03125
          ],
          "n": 128
        },
        "gain_vs_reject": {
          "mean": 0.018229166666666664,
          "ci95": [
            -0.020833333333333332,
            0.0546875
          ],
          "n": 128
        },
        "matched_accept_rate_random_expected": 0.08030192057291666
      },
      "f1": {
        "score": {
          "mean": 0.19352003093896591,
          "ci95": [
            0.1485307968734968,
            0.24318942051046372
          ],
          "n": 128
        },
        "gain_vs_accept": {
          "mean": 0.014912291038210155,
          "ci95": [
            0.004278610482654601,
            0.02758284368434873
          ],
          "n": 128
        },
        "gain_vs_reject": {
          "mean": 0.045082691623869235,
          "ci95": [
            0.0020753859330789708,
            0.08665010857623444
          ],
          "n": 128
        },
        "matched_accept_rate_random_expected": 0.17365790855467103
      }
    },
    "question_gain_zero": {
      "accept_rate": 0.8359375,
      "em": {
        "score": {
          "mean": 0.09635416666666666,
          "ci95": [
            0.05729166666666666,
            0.140625
          ],
          "n": 128
        },
        "gain_vs_accept": {
          "mean": 0.015625,
          "ci95": [
            0.0026041666666666665,
            0.03125
          ],
          "n": 128
        },
        "gain_vs_reject": {
          "mean": 0.018229166666666664,
          "ci95": [
            -0.020833333333333332,
            0.0546875
          ],
          "n": 128
        },
        "matched_accept_rate_random_expected": 0.08030192057291666
      },
      "f1": {
        "score": {
          "mean": 0.19352003093896591,
          "ci95": [
            0.1485307968734968,
            0.24318942051046372
          ],
          "n": 128
        },
        "gain_vs_accept": {
          "mean": 0.014912291038210155,
          "ci95": [
            0.004278610482654601,
            0.02758284368434873
          ],
          "n": 128
        },
        "gain_vs_reject": {
          "mean": 0.045082691623869235,
          "ci95": [
            0.0020753859330789708,
            0.08665010857623444
          ],
          "n": 128
        },
        "matched_accept_rate_random_expected": 0.17365790855467103
      }
    },
    "first_token_entropy_gain_calibrated": {
      "accept_rate": 1.0,
      "em": {
        "score": {
          "mean": 0.08072916666666666,
          "ci95": [
            0.04427083333333333,
            0.12239583333333333
          ],
          "n": 128
        },
        "gain_vs_accept": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        },
        "gain_vs_reject": {
          "mean": 0.0026041666666666644,
          "ci95": [
            -0.04427083333333333,
            0.04427083333333333
          ],
          "n": 128
        },
        "matched_accept_rate_random_expected": 0.08072916666666666
      },
      "f1": {
        "score": {
          "mean": 0.17860773990075574,
          "ci95": [
            0.1372615518573756,
            0.22419966288502136
          ],
          "n": 128
        },
        "gain_vs_accept": {
          "mean": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "n": 128
        },
        "gain_vs_reject": {
          "mean": 0.030170400585659073,
          "ci95": [
            -0.01753893232187463,
            0.07567593811187102
          ],
          "n": 128
        },
        "matched_accept_rate_random_expected": 0.17860773990075574
      }
    },
    "first_token_entropy_gain_zero": {
      "accept_rate": 0.2526041666666667,
      "em": {
        "score": {
          "mean": 0.08333333333333331,
          "ci95": [
            0.041666666666666664,
            0.13020833333333334
          ],
          "n": 128
        },
        "gain_vs_accept": {
          "mean": 0.002604166666666667,
          "ci95": [
            -0.028645833333333332,
            0.036458333333333336
          ],
          "n": 128
        },
        "gain_vs_reject": {
          "mean": 0.005208333333333332,
          "ci95": [
            -0.0234375,
            0.03125
          ],
          "n": 128
        },
        "matched_accept_rate_random_expected": 0.07878282335069445
      },
      "f1": {
        "score": {
          "mean": 0.15571846539217862,
          "ci95": [
            0.11185563620449879,
            0.2053549983832038
          ],
          "n": 128
        },
        "gain_vs_accept": {
          "mean": -0.02288927450857712,
          "ci95": [
            -0.056844153438806506,
            0.011501408800639023
          ],
          "n": 128
        },
        "gain_vs_reject": {
          "mean": 0.007281126077081958,
          "ci95": [
            -0.022683408894346398,
            0.03569849918769951
          ],
          "n": 128
        },
        "matched_accept_rate_random_expected": 0.15605850821303657
      }
    }
  },
  "caveats": [
    "128 heldout questions, fixed before calibration32 inference; not official full HotpotQA benchmark.",
    "Short-answer generation uses entire normalized output EM/F1. No gold-dependent answer extraction.",
    "Selected rule fixed on pilot F1. Other rules are diagnostics, not alternative test-selected winners.",
    "Balanced synthetic clean/wrong/partial corruption is a controlled mixture, not an estimate of deployment prevalence.",
    "Both accept and reject branches are computed: no compute savings claimed.",
    "First-token entropy and question likelihood are proxies, not calibrated truth probabilities.",
    "No Reader, training, or quantization. Bootstrap resamples questions with their corruption variants together."
  ]
}
```


## receiver_kv_utility_hotpotqa_overnight_seed1234

```json
{
  "experiment": "receiver_kv_utility_hotpotqa_overnight_seed1234",
  "counts": {
    "train": 1024,
    "calibration": 256,
    "test": 512
  },
  "primary_model": "utility_stability8_f1",
  "in_distribution": {
    "conditions": [
      "clean",
      "wrong",
      "partial50"
    ],
    "n_questions": 512,
    "n_cases": 1536,
    "policies": {
      "always_accept": {
        "n_cases": 1536,
        "accept_rate": 1.0,
        "em": 0.07291666666666667,
        "f1": 0.17413605862750456,
        "helpful_recall": 1.0,
        "harmful_rejection_rate": 0.0,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.07291666666666666,
          "ci95": [
            0.056640625,
            0.09114583333333331
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.022135416666666664,
            "ci95": [
              0.0012858072916666695,
              0.04361979166666666
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": -0.004557291666666667,
            "ci95": [
              -0.009114583333333334,
              -0.0006510416666666666
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.014322916666666668,
            "ci95": [
              -0.027994791666666664,
              -0.001953125000000001
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": -0.008463541666666666,
            "ci95": [
              -0.020182291666666664,
              0.002604166666666666
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.013020833333333332,
            "ci95": [
              -0.025390625,
              -0.0019531249999999996
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.07291666666666667,
        "f1_interval": {
          "mean": 0.1741360586275046,
          "ci95": [
            0.15344877539967933,
            0.19660042500989194
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.03689332229572635,
            "ci95": [
              0.014402774338622025,
              0.06066349626225164
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": -0.006171202005173191,
            "ci95": [
              -0.011349597633750427,
              -0.0013976764709002667
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.020664952651801217,
            "ci95": [
              -0.03417228851027953,
              -0.007669609060139981
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": -0.016321596736216998,
            "ci95": [
              -0.02800159941771358,
              -0.0046939431768653955
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.022455489498306974,
            "ci95": [
              -0.035024460237363095,
              -0.010592926249825227
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.17413605862750456
      },
      "always_reject": {
        "n_cases": 1536,
        "accept_rate": 0.0,
        "em": 0.05078125,
        "f1": 0.13724273633177822,
        "helpful_recall": 0.0,
        "harmful_rejection_rate": 1.0,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.05078125,
          "ci95": [
            0.033203125,
            0.0703125
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": -0.022135416666666664,
            "ci95": [
              -0.04361979166666666,
              -0.0012858072916669006
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": -0.02669270833333333,
            "ci95": [
              -0.04622395833333333,
              -0.007161458333333332
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.03645833333333333,
            "ci95": [
              -0.052083333333333336,
              -0.021484374999999997
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": -0.030598958333333332,
            "ci95": [
              -0.046875,
              -0.014973958333333332
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.03515625,
            "ci95": [
              -0.05143229166666666,
              -0.01953125
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.05078125,
        "f1_interval": {
          "mean": 0.13724273633177822,
          "ci95": [
            0.11453230456960681,
            0.16084752647230513
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": -0.03689332229572635,
            "ci95": [
              -0.06066349626225165,
              -0.014402774338622026
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": -0.043064524300899534,
            "ci95": [
              -0.06471824773584459,
              -0.0229913521279414
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.05755827494752756,
            "ci95": [
              -0.07488834943908174,
              -0.041427641943166955
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": -0.05321491903194334,
            "ci95": [
              -0.07077174079239047,
              -0.03688766853813799
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.059348811794033327,
            "ci95": [
              -0.07658422260417286,
              -0.04324099077621853
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.13724273633177822
      },
      "question_gain_zero": {
        "n_cases": 1536,
        "accept_rate": 0.8170572916666666,
        "em": 0.07747395833333333,
        "f1": 0.18030726063267777,
        "helpful_recall": 0.9764705882352941,
        "harmful_rejection_rate": 0.17647058823529413,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.07747395833333333,
          "ci95": [
            0.061197916666666664,
            0.09570312499999999
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": 0.004557291666666667,
            "ci95": [
              0.0006510416666666666,
              0.009114583333333334
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.02669270833333333,
            "ci95": [
              0.007161458333333332,
              0.04622395833333333
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.009765625,
            "ci95": [
              -0.020833333333333332,
              0.0006510416666666661
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": -0.00390625,
            "ci95": [
              -0.012369791666666666,
              0.005208333333333332
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.008463541666666666,
            "ci95": [
              -0.018229166666666664,
              0.000651041666666667
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.06886715359157985,
        "f1_interval": {
          "mean": 0.18030726063267777,
          "ci95": [
            0.15979859435167618,
            0.20285067119441585
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": 0.006171202005173191,
            "ci95": [
              0.0013976764709002609,
              0.011349597633750427
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.043064524300899534,
            "ci95": [
              0.022991352127941397,
              0.06471824773584456
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.014493750646628032,
            "ci95": [
              -0.026151986394378147,
              -0.0029576304060458204
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": -0.010150394731043805,
            "ci95": [
              -0.01992771113539731,
              -0.00016125495852849936
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.016284287493133785,
            "ci95": [
              -0.02723374018868497,
              -0.005820292880800096
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.16738669432730988
      },
      "question_gain_calibrated": {
        "n_cases": 1536,
        "accept_rate": 0.8170572916666666,
        "em": 0.07747395833333333,
        "f1": 0.18030726063267777,
        "helpful_recall": 0.9764705882352941,
        "harmful_rejection_rate": 0.17647058823529413,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.07747395833333333,
          "ci95": [
            0.061197916666666664,
            0.09570312499999999
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": 0.004557291666666667,
            "ci95": [
              0.0006510416666666666,
              0.009114583333333334
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.02669270833333333,
            "ci95": [
              0.007161458333333332,
              0.04622395833333333
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.009765625,
            "ci95": [
              -0.020833333333333332,
              0.0006510416666666661
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": -0.00390625,
            "ci95": [
              -0.012369791666666666,
              0.005208333333333332
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.008463541666666666,
            "ci95": [
              -0.018229166666666664,
              0.000651041666666667
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.06886715359157985,
        "f1_interval": {
          "mean": 0.18030726063267777,
          "ci95": [
            0.15979859435167618,
            0.20285067119441585
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": 0.006171202005173191,
            "ci95": [
              0.0013976764709002609,
              0.011349597633750427
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.043064524300899534,
            "ci95": [
              0.022991352127941397,
              0.06471824773584456
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.014493750646628032,
            "ci95": [
              -0.026151986394378147,
              -0.0029576304060458204
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": -0.010150394731043805,
            "ci95": [
              -0.01992771113539731,
              -0.00016125495852849936
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.016284287493133785,
            "ci95": [
              -0.02723374018868497,
              -0.005820292880800096
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.16738669432730988
      },
      "entropy_gain_zero": {
        "n_cases": 1536,
        "accept_rate": 0.28125,
        "em": 0.06770833333333333,
        "f1": 0.16025083489712627,
        "helpful_recall": 0.49411764705882355,
        "harmful_rejection_rate": 0.6862745098039216,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.06770833333333333,
          "ci95": [
            0.05013020833333333,
            0.08723958333333333
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": -0.0052083333333333304,
            "ci95": [
              -0.02083333333333333,
              0.009765625
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.016927083333333332,
            "ci95": [
              0.0039062499999999996,
              0.029947916666666664
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": -0.009765624999999998,
            "ci95": [
              -0.022786458333333332,
              0.003255208333333332
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.01953125,
            "ci95": [
              -0.029947916666666664,
              -0.009114583333333332
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": -0.013671874999999998,
            "ci95": [
              -0.0234375,
              -0.004557291666666666
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.018229166666666664,
            "ci95": [
              -0.02864583333333333,
              -0.008463541666666666
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.0570068359375,
        "f1_interval": {
          "mean": 0.16025083489712627,
          "ci95": [
            0.13812813995526538,
            0.18310832812795547
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": -0.013885223730378302,
            "ci95": [
              -0.031378419854334066,
              0.003335700805520929
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.023008098565348044,
            "ci95": [
              0.009869542464462002,
              0.036319382545233266
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": -0.02005642573555149,
            "ci95": [
              -0.03538475604772031,
              -0.0049853370490087565
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.034550176382179526,
            "ci95": [
              -0.04768094020934188,
              -0.022226407136894798
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": -0.030206820466595303,
            "ci95": [
              -0.042245687125938146,
              -0.018898067718616123
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.036340713228685276,
            "ci95": [
              -0.04868373108988041,
              -0.024878582786700527
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.14761898322745126
      },
      "entropy_gain_calibrated": {
        "n_cases": 1536,
        "accept_rate": 0.4505208333333333,
        "em": 0.06901041666666667,
        "f1": 0.16650240727046178,
        "helpful_recall": 0.6470588235294118,
        "harmful_rejection_rate": 0.47058823529411764,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.06901041666666666,
          "ci95": [
            0.05208333333333333,
            0.08789062499999999
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": -0.00390625,
            "ci95": [
              -0.016276041666666664,
              0.008463541666666668
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.018229166666666664,
            "ci95": [
              0.0026041666666666652,
              0.033203125
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": -0.008463541666666666,
            "ci95": [
              -0.01953125,
              0.001953125000000001
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.018229166666666664,
            "ci95": [
              -0.029296874999999997,
              -0.007812499999999999
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": -0.012369791666666666,
            "ci95": [
              -0.021484375,
              -0.0032552083333333335
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.016927083333333332,
            "ci95": [
              -0.026692708333333332,
              -0.007812499999999999
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.06075371636284723,
        "f1_interval": {
          "mean": 0.16650240727046178,
          "ci95": [
            0.14467711824644378,
            0.18905566599935833
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": -0.00763365135704278,
            "ci95": [
              -0.022121346103782625,
              0.00667625013673988
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.029259670938683566,
            "ci95": [
              0.014240476457109694,
              0.04499164968198006
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": -0.013804853362215971,
            "ci95": [
              -0.027281223861937442,
              -0.0007778221011093121
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.028298604008844,
            "ci95": [
              -0.04058557281112223,
              -0.016304374842256664
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": -0.023955248093259777,
            "ci95": [
              -0.034947623513259196,
              -0.013736234392576221
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.030089140855349757,
            "ci95": [
              -0.04154096769175523,
              -0.01945566676079809
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.1538639466368841
      },
      "calibrated_confidence": {
        "n_cases": 1536,
        "accept_rate": 0.6666666666666666,
        "em": 0.08723958333333333,
        "f1": 0.19480101127930582,
        "helpful_recall": 0.8470588235294118,
        "harmful_rejection_rate": 0.6862745098039216,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.08723958333333333,
          "ci95": [
            0.068359375,
            0.10872395833333333
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": 0.014322916666666668,
            "ci95": [
              0.001953125000000001,
              0.027994791666666664
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.03645833333333333,
            "ci95": [
              0.021484374999999997,
              0.052083333333333336
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.009765625,
            "ci95": [
              -0.0006510416666666661,
              0.020833333333333332
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.005859375,
            "ci95": [
              0.0,
              0.012369791666666664
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": 0.0013020833333333335,
            "ci95": [
              -0.005859375,
              0.008463541666666666
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.06553819444444443,
        "f1_interval": {
          "mean": 0.19480101127930582,
          "ci95": [
            0.17129844973264546,
            0.21968989524802346
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": 0.020664952651801217,
            "ci95": [
              0.007669609060139972,
              0.034172288510279515
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.05755827494752756,
            "ci95": [
              0.04142764194316695,
              0.07488834943908172
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.014493750646628032,
            "ci95": [
              0.00295763040604581,
              0.026151986394378147
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.004343355915584223,
            "ci95": [
              -0.0025584505303873264,
              0.011726999647422923
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.0017905368465057528,
            "ci95": [
              -0.009479267525770246,
              0.005966861390680875
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.16183828452892915
      },
      "utility_output_em": {
        "n_cases": 1536,
        "accept_rate": 0.4947916666666667,
        "em": 0.09244791666666667,
        "f1": 0.19821999615246103,
        "helpful_recall": 0.8352941176470589,
        "harmful_rejection_rate": 0.8627450980392157,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.09244791666666666,
          "ci95": [
            0.07161458333333333,
            0.11458333333333333
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": 0.01953125,
            "ci95": [
              0.005843098958333337,
              0.03515625
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.04166666666666667,
            "ci95": [
              0.027994791666666664,
              0.056640625
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.014973958333333332,
            "ci95": [
              0.0032552083333333335,
              0.027994791666666664
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": 0.005208333333333333,
            "ci95": [
              -0.0006510416666666665,
              0.01171875
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.011067708333333334,
            "ci95": [
              0.0026041666666666665,
              0.020182291666666664
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": 0.006510416666666666,
            "ci95": [
              -0.0013020833333333335,
              0.014973958333333332
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.061733669704861105,
        "f1_interval": {
          "mean": 0.19821999615246103,
          "ci95": [
            0.17430729600951433,
            0.2235183824608884
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": 0.024083937524956443,
            "ci95": [
              0.010312797968619692,
              0.03846489436045008
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.06097725982068279,
            "ci95": [
              0.04472494654332608,
              0.0785350308002453
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.017912735519783254,
            "ci95": [
              0.005612915614029495,
              0.030365894861964924
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": 0.0034189848731552225,
            "ci95": [
              -0.0010674323882105174,
              0.0085430891329438
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.007762340788739448,
            "ci95": [
              -0.00038277919725591465,
              0.016431012643134568
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": 0.0016284480266494702,
            "ci95": [
              -0.006842078095715934,
              0.01048444597155043
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.15549724475935114
      },
      "utility_output_f1": {
        "n_cases": 1536,
        "accept_rate": 0.5442708333333334,
        "em": 0.08138020833333333,
        "f1": 0.19045765536372158,
        "helpful_recall": 0.8470588235294118,
        "harmful_rejection_rate": 0.5098039215686274,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.08138020833333333,
          "ci95": [
            0.06380208333333333,
            0.10091145833333333
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": 0.008463541666666666,
            "ci95": [
              -0.002604166666666666,
              0.020182291666666664
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.030598958333333332,
            "ci95": [
              0.014973958333333332,
              0.046875
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.00390625,
            "ci95": [
              -0.005208333333333332,
              0.012369791666666666
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.005859375,
            "ci95": [
              -0.012369791666666664,
              0.0
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.004557291666666667,
            "ci95": [
              -0.01171875,
              0.0019531249999999998
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.06282891167534722,
        "f1_interval": {
          "mean": 0.19045765536372156,
          "ci95": [
            0.168162359313975,
            0.21352257187886262
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": 0.016321596736216998,
            "ci95": [
              0.004693943176865392,
              0.02800159941771358
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.05321491903194334,
            "ci95": [
              0.03688766853813799,
              0.07077174079239047
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.010150394731043805,
            "ci95": [
              0.0001612549585284953,
              0.01992771113539729
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.004343355915584223,
            "ci95": [
              -0.011726999647422926,
              0.0025584505303873225
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.006133892762089976,
            "ci95": [
              -0.013407462106078093,
              0.00035782777193223923
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.15732269560210846
      },
      "utility_cross_em": {
        "n_cases": 1536,
        "accept_rate": 0.6614583333333334,
        "em": 0.09505208333333333,
        "f1": 0.19941312663481803,
        "helpful_recall": 0.8941176470588236,
        "harmful_rejection_rate": 0.8431372549019608,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.09505208333333333,
          "ci95": [
            0.07486979166666666,
            0.11655273437499977
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": 0.022135416666666664,
            "ci95": [
              0.009114583333333332,
              0.037109375
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.04427083333333333,
            "ci95": [
              0.03059895833333333,
              0.059244791666666664
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.017578125,
            "ci95": [
              0.006510416666666666,
              0.029947916666666664
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": 0.0078125,
            "ci95": [
              0.0006510416666666667,
              0.015625
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.013671874999999998,
            "ci95": [
              0.005208333333333333,
              0.023437499999999997
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": 0.009114583333333332,
            "ci95": [
              0.002604166666666667,
              0.016276041666666664
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.06542290581597222,
        "f1_interval": {
          "mean": 0.19941312663481803,
          "ci95": [
            0.17534286711725658,
            0.22479834233663398
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": 0.02527706800731344,
            "ci95": [
              0.013424413238005445,
              0.037825022674897
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.06217039030303979,
            "ci95": [
              0.04524144896839663,
              0.08059506023417322
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.019105866002140252,
            "ci95": [
              0.008239523995318214,
              0.03072206774539095
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": 0.00461211535551222,
            "ci95": [
              -0.0011305743452185494,
              0.010935303600752332
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.008955471271096443,
            "ci95": [
              0.0005088483803880291,
              0.01795090654893538
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": 0.0028215785090064667,
            "ci95": [
              -0.003901019831846297,
              0.009646562981973706
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.16164613180863888
      },
      "utility_cross_f1": {
        "n_cases": 1536,
        "accept_rate": 0.53125,
        "em": 0.0859375,
        "f1": 0.1965915481258115,
        "helpful_recall": 0.8705882352941177,
        "harmful_rejection_rate": 0.6078431372549019,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.0859375,
          "ci95": [
            0.06770833333333333,
            0.10611979166666666
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": 0.013020833333333332,
            "ci95": [
              0.0019531249999999996,
              0.025390625
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.03515625,
            "ci95": [
              0.01953125,
              0.05143229166666666
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.008463541666666666,
            "ci95": [
              -0.000651041666666667,
              0.018229166666666664
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.0013020833333333335,
            "ci95": [
              -0.008463541666666666,
              0.005859375
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.004557291666666667,
            "ci95": [
              -0.0019531249999999998,
              0.01171875
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.06254069010416667,
        "f1_interval": {
          "mean": 0.19659154812581153,
          "ci95": [
            0.1734503339226231,
            0.22069013999645143
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": 0.022455489498306974,
            "ci95": [
              0.010592926249825215,
              0.035024460237363095
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.059348811794033327,
            "ci95": [
              0.04324099077621852,
              0.07658422260417284
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.016284287493133785,
            "ci95": [
              0.005820292880800096,
              0.027233740188684966
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": 0.0017905368465057528,
            "ci95": [
              -0.005966861390680875,
              0.009479267525770245
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.006133892762089976,
            "ci95": [
              -0.00035782777193224454,
              0.013407462106078082
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": 0.0,
            "ci95": [
              0.0,
              0.0
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.15684231380138283
      },
      "utility_stability2_em": {
        "n_cases": 1536,
        "accept_rate": 0.90625,
        "em": 0.09114583333333333,
        "f1": 0.1912066109355158,
        "helpful_recall": 0.9529411764705882,
        "harmful_rejection_rate": 0.6274509803921569,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.09114583333333333,
          "ci95": [
            0.07161458333333333,
            0.11263020833333333
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": 0.018229166666666664,
            "ci95": [
              0.0078125,
              0.029947916666666664
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.04036458333333333,
            "ci95": [
              0.024088541666666664,
              0.05794270833333332
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.013671875,
            "ci95": [
              0.005208333333333333,
              0.0234375
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": 0.00390625,
            "ci95": [
              -0.005208333333333334,
              0.013671875
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.009765625,
            "ci95": [
              0.0006510416666666667,
              0.01953125
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": 0.005208333333333333,
            "ci95": [
              -0.0026041666666666665,
              0.013020833333333332
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.07084147135416667,
        "f1_interval": {
          "mean": 0.19120661093551578,
          "ci95": [
            0.16816837866046508,
            0.215120894148602
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": 0.01707055230801119,
            "ci95": [
              0.008214774700126262,
              0.02701558614862994
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.05396387460373754,
            "ci95": [
              0.034875047614328816,
              0.07435855998624545
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.010899350302837998,
            "ci95": [
              0.002186682281874803,
              0.01991894773335159
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.0035944003437900314,
            "ci95": [
              -0.012801203198473323,
              0.005659722342029205
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.0007489555717941918,
            "ci95": [
              -0.008656332202456646,
              0.011061314275273303
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.005384937190295784,
            "ci95": [
              -0.01388144257351698,
              0.002849041864131333
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.17067730966228023
      },
      "utility_stability2_f1": {
        "n_cases": 1536,
        "accept_rate": 0.5143229166666666,
        "em": 0.08658854166666667,
        "f1": 0.19469733569822734,
        "helpful_recall": 0.8705882352941177,
        "harmful_rejection_rate": 0.6274509803921569,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.08658854166666666,
          "ci95": [
            0.06770833333333333,
            0.10678710937499977
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": 0.013671875,
            "ci95": [
              0.0026041666666666665,
              0.02540690104166643
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.035807291666666664,
            "ci95": [
              0.020182291666666664,
              0.051432291666666664
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.009114583333333332,
            "ci95": [
              -2.7105054312128366e-21,
              0.018880208333333332
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.0006510416666666667,
            "ci95": [
              -0.007812499999999999,
              0.006510416666666666
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.005208333333333333,
            "ci95": [
              -0.001953125,
              0.013020833333333332
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": 0.0006510416666666666,
            "ci95": [
              -0.0026041666666666665,
              0.00390625
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.06216600206163194,
        "f1_interval": {
          "mean": 0.19469733569822728,
          "ci95": [
            0.17172199915771988,
            0.21896088261232205
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": 0.02056127707072274,
            "ci95": [
              0.009023749784255651,
              0.032939021140037245
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.05745459936644908,
            "ci95": [
              0.0409250697474884,
              0.07495175124846173
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.014390075065549544,
            "ci95": [
              0.004340067503427698,
              0.02522154241844229
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.00010367558107848406,
            "ci95": [
              -0.008027693696390037,
              0.008016433542358528
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.004239680334505738,
            "ci95": [
              -0.0025874222723480194,
              0.011595274962360432
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.0018942124275842381,
            "ci95": [
              -0.005143630195430909,
              0.0012228669215903001
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.15621781746043958
      },
      "utility_stability4_em": {
        "n_cases": 1536,
        "accept_rate": 0.48046875,
        "em": 0.09375,
        "f1": 0.20006598254839134,
        "helpful_recall": 0.8941176470588236,
        "harmful_rejection_rate": 0.803921568627451,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.09375,
          "ci95": [
            0.07356770833333333,
            0.115234375
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": 0.020833333333333332,
            "ci95": [
              0.0078125,
              0.03515625
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.04296875,
            "ci95": [
              0.028645833333333332,
              0.05859374999999999
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.016276041666666664,
            "ci95": [
              0.005208333333333334,
              0.02864583333333333
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": 0.006510416666666666,
            "ci95": [
              -0.0006510416666666668,
              0.014973958333333332
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.012369791666666666,
            "ci95": [
              0.00390625,
              0.02150065104166643
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": 0.007812499999999999,
            "ci95": [
              0.001953125,
              0.01497395833333333
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.0614166259765625,
        "f1_interval": {
          "mean": 0.2000659825483913,
          "ci95": [
            0.17600794188391014,
            0.22561405764406484
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": 0.025929923920886735,
            "ci95": [
              0.013356610766169105,
              0.03889988245665873
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.06282324621661309,
            "ci95": [
              0.045922463037251664,
              0.08085147445088604
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.019758721915713547,
            "ci95": [
              0.008711952592461507,
              0.031569355321515044
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": 0.0052649712690855145,
            "ci95": [
              -0.0016161965529562304,
              0.012715042374560934
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.009608327184669738,
            "ci95": [
              0.0015554181248555795,
              0.01834575709879155
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": 0.003474434422579761,
            "ci95": [
              -0.002568609158412631,
              0.009623722160555136
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.15496882477855298
      },
      "utility_stability4_f1": {
        "n_cases": 1536,
        "accept_rate": 0.5078125,
        "em": 0.08723958333333333,
        "f1": 0.19532134808120108,
        "helpful_recall": 0.8588235294117647,
        "harmful_rejection_rate": 0.6666666666666666,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.08723958333333333,
          "ci95": [
            0.068359375,
            0.10807291666666666
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": 0.014322916666666664,
            "ci95": [
              0.0026041666666666665,
              0.026692708333333336
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.03645833333333333,
            "ci95": [
              0.021484375,
              0.05208333333333333
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.009765625,
            "ci95": [
              -2.168404344971009e-19,
              0.020182291666666664
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": 0.0,
            "ci95": [
              -0.007161458333333333,
              0.0078125
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.005859374999999999,
            "ci95": [
              -0.0013020833333333333,
              0.013671875
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": 0.0013020833333333333,
            "ci95": [
              -0.002604166666666667,
              0.005859375
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.062021891276041664,
        "f1_interval": {
          "mean": 0.1953213480812011,
          "ci95": [
            0.1722770817796742,
            0.21970764523604522
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": 0.021185289453696517,
            "ci95": [
              0.009361759257520539,
              0.03381339600461419
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.05807861174942286,
            "ci95": [
              0.041719237072766306,
              0.07529090691835053
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.015014087448523325,
            "ci95": [
              0.004873817058493302,
              0.02613253422008575
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": 0.000520336801895296,
            "ci95": [
              -0.00756274060621348,
              0.008702609145374206
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.004863692717479519,
            "ci95": [
              -0.0017422651698206215,
              0.012241055148755921
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.001270200044610458,
            "ci95": [
              -0.005949327225783864,
              0.003066765589756247
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.15597762656007674
      },
      "utility_stability8_em": {
        "n_cases": 1536,
        "accept_rate": 0.89453125,
        "em": 0.0859375,
        "f1": 0.18997742923758407,
        "helpful_recall": 0.9294117647058824,
        "harmful_rejection_rate": 0.5098039215686274,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.0859375,
          "ci95": [
            0.06769205729166666,
            0.10611979166666666
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": 0.013020833333333332,
            "ci95": [
              0.0039062499999999996,
              0.0234375
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.03515625,
            "ci95": [
              0.017578125,
              0.05273437499999999
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.008463541666666666,
            "ci95": [
              0.0006510416666666666,
              0.016927083333333332
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.0013020833333333335,
            "ci95": [
              -0.009114583333333332,
              0.005859375
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.004557291666666666,
            "ci95": [
              -0.002604166666666667,
              0.012369791666666666
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -2.168404344971009e-19,
            "ci95": [
              -0.006510416666666667,
              0.006510416666666666
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.07058207194010417,
        "f1_interval": {
          "mean": 0.18997742923758404,
          "ci95": [
            0.16734178617806406,
            0.21387072241417426
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": 0.01584137061007949,
            "ci95": [
              0.006751065353270329,
              0.02557304187364322
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.05273469290580585,
            "ci95": [
              0.03406976483229377,
              0.07309246212730487
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.009670168604906302,
            "ci95": [
              0.0010542486907841565,
              0.018544092812664774
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.004823582041721728,
            "ci95": [
              -0.013162239043870327,
              0.0033433339763080205
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": -0.0004802261261375059,
            "ci95": [
              -0.008676376224753713,
              0.008183057758539322
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.0066141188882274824,
            "ci95": [
              -0.013745910624779124,
              0.0004412085692046663
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.1702449660416272
      },
      "utility_stability8_f1": {
        "n_cases": 1536,
        "accept_rate": 0.513671875,
        "em": 0.0859375,
        "f1": 0.1953601216795818,
        "helpful_recall": 0.8352941176470589,
        "harmful_rejection_rate": 0.6666666666666666,
        "helpful_cases": 85,
        "harmful_cases": 51,
        "em_interval": {
          "mean": 0.0859375,
          "ci95": [
            0.06705729166666666,
            0.10677083333333333
          ],
          "n": 512
        },
        "em_gains": {
          "always_accept": {
            "mean": 0.013020833333333332,
            "ci95": [
              0.0013020833333333335,
              0.026041666666666664
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.03515624999999999,
            "ci95": [
              0.01953125,
              0.05078125
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.008463541666666668,
            "ci95": [
              -0.0013020833333333335,
              0.018880208333333332
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": -0.001302083333333333,
            "ci95": [
              -0.007812499999999999,
              0.005208333333333333
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.004557291666666667,
            "ci95": [
              -0.001953125,
              0.012369791666666664
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": 2.168404344971009e-19,
            "ci95": [
              -0.004557291666666666,
              0.004557291666666667
            ],
            "n": 512
          }
        },
        "em_rate_matched_random_expected": 0.062151590983072914,
        "f1_interval": {
          "mean": 0.19536012167958183,
          "ci95": [
            0.17244279049775907,
            0.21972253337143205
          ],
          "n": 512
        },
        "f1_gains": {
          "always_accept": {
            "mean": 0.021224063052077257,
            "ci95": [
              0.008837107384078093,
              0.034538257057362666
            ],
            "n": 512
          },
          "always_reject": {
            "mean": 0.0581173853478036,
            "ci95": [
              0.04206917899207174,
              0.07521297586703518
            ],
            "n": 512
          },
          "question_gain_zero": {
            "mean": 0.015052861046904065,
            "ci95": [
              0.004225359124855598,
              0.026220788281070512
            ],
            "n": 512
          },
          "calibrated_confidence": {
            "mean": 0.0005591104002760361,
            "ci95": [
              -0.0068157893721272865,
              0.00810627594552832
            ],
            "n": 512
          },
          "utility_output_f1": {
            "mean": 0.004902466315860259,
            "ci95": [
              -0.0014586713427131464,
              0.012060997434466373
            ],
            "n": 512
          },
          "utility_cross_f1": {
            "mean": -0.0012314264462297175,
            "ci95": [
              -0.0059696849094537435,
              0.0033066849876219337
            ],
            "n": 512
          }
        },
        "f1_rate_matched_random_expected": 0.1561937983704033
      }
    },
    "oracle": {
      "em": 0.10611979166666667,
      "f1": 0.23544424068553982
    },
    "both_wrong_cases": 1373
  },
  "result": "/hy-tmp/yezhe/异构模型/receiver_kv_utility_hotpotqa_overnight_seed1234/summary.json"
}
```


## DST depth-recurrence ablation (Llama→Qwen, seed 1234)

[Full results and intervals](experiments/dst_depth_recurrence_ablation_l3_to_q4_seed1234/RESULTS.md)

| Stage | Carry macro | No-Carry macro | Carry − No-Carry (95% CI, pp) |
|---|---:|---:|---:|
| stage1 | 43.66% | 33.49% | +10.18 [+8.12, +12.19] |
| stage2 | 56.84% | 53.58% | +3.26 [+1.33, +5.18] |
