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
