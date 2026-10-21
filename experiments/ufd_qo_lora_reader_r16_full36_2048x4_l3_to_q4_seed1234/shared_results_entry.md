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
