# Standard lm-eval legacy-method reproduction

Pilot subsets, not full-benchmark scores. All metrics below are returned by the official evaluator.

## mcq / evaluation / llama_native

```json
{
  "openbookqa": {
    "name": "openbookqa",
    "alias": "openbookqa",
    "sample_len": 128,
    "acc,none": 0.28125,
    "acc_stderr,none": 0.0398963655025738,
    "acc_norm,none": 0.40625,
    "acc_norm_stderr,none": 0.04358094446512651
  },
  "arc_challenge": {
    "name": "arc_challenge",
    "alias": "arc_challenge",
    "sample_len": 128,
    "acc,none": 0.453125,
    "acc_stderr,none": 0.04417241936858356,
    "acc_norm,none": 0.421875,
    "acc_norm_stderr,none": 0.04382287939944462
  }
}
```

## mcq / evaluation / native_oracle

```json
{
  "openbookqa": {
    "name": "openbookqa",
    "alias": "openbookqa",
    "sample_len": 128,
    "acc,none": 0.2421875,
    "acc_stderr,none": 0.038014990292760314,
    "acc_norm,none": 0.453125,
    "acc_norm_stderr,none": 0.04417241936858356
  },
  "arc_challenge": {
    "name": "arc_challenge",
    "alias": "arc_challenge",
    "sample_len": 128,
    "acc,none": 0.5625,
    "acc_stderr,none": 0.0440198371531759,
    "acc_norm,none": 0.5546875,
    "acc_norm_stderr,none": 0.04410164327680508
  }
}
```

## mcq / evaluation / qwen_native

```json
{
  "openbookqa": {
    "name": "openbookqa",
    "alias": "openbookqa",
    "sample_len": 128,
    "acc,none": 0.2421875,
    "acc_stderr,none": 0.038014990292760314,
    "acc_norm,none": 0.453125,
    "acc_norm_stderr,none": 0.04417241936858356
  },
  "arc_challenge": {
    "name": "arc_challenge",
    "alias": "arc_challenge",
    "sample_len": 128,
    "acc,none": 0.5625,
    "acc_stderr,none": 0.0440198371531759,
    "acc_norm,none": 0.5546875,
    "acc_norm_stderr,none": 0.04410164327680508
  }
}
```

## mcq / evaluation / stage_a

```json
{
  "openbookqa": {
    "name": "openbookqa",
    "alias": "openbookqa",
    "sample_len": 128,
    "acc,none": 0.1640625,
    "acc_stderr,none": 0.032861675748836264,
    "acc_norm,none": 0.3984375,
    "acc_norm_stderr,none": 0.04344288118823427
  },
  "arc_challenge": {
    "name": "arc_challenge",
    "alias": "arc_challenge",
    "sample_len": 128,
    "acc,none": 0.2265625,
    "acc_stderr,none": 0.037145376625835905,
    "acc_norm,none": 0.25,
    "acc_norm_stderr,none": 0.038423663968391995
  }
}
```

## mcq / evaluation / stage_b

```json
{
  "openbookqa": {
    "name": "openbookqa",
    "alias": "openbookqa",
    "sample_len": 128,
    "acc,none": 0.234375,
    "acc_stderr,none": 0.037589092002846054,
    "acc_norm,none": 0.4375,
    "acc_norm_stderr,none": 0.0440198371531759
  },
  "arc_challenge": {
    "name": "arc_challenge",
    "alias": "arc_challenge",
    "sample_len": 128,
    "acc,none": 0.4140625,
    "acc_stderr,none": 0.04370757750604408,
    "acc_norm,none": 0.4140625,
    "acc_norm_stderr,none": 0.04370757750604408
  }
}
```

## mcq / evaluation_ood / llama_native

```json
{
  "hellaswag": {
    "name": "hellaswag",
    "alias": "hellaswag",
    "sample_len": 128,
    "acc,none": 0.46875,
    "acc_stderr,none": 0.044281084771084105,
    "acc_norm,none": 0.625,
    "acc_norm_stderr,none": 0.042958962288966916
  },
  "mmlu_pro_biology": {
    "name": "mmlu_pro_biology",
    "alias": "biology",
    "sample_len": 10,
    "exact_match,custom-extract": 0.5,
    "exact_match_stderr,custom-extract": 0.16666666666666666
  },
  "mmlu_pro_business": {
    "name": "mmlu_pro_business",
    "alias": "business",
    "sample_len": 10,
    "exact_match,custom-extract": 0.3,
    "exact_match_stderr,custom-extract": 0.15275252316519464
  },
  "mmlu_pro_chemistry": {
    "name": "mmlu_pro_chemistry",
    "alias": "chemistry",
    "sample_len": 9,
    "exact_match,custom-extract": 0.3333333333333333,
    "exact_match_stderr,custom-extract": 0.16666666666666666
  },
  "mmlu_pro_computer_science": {
    "name": "mmlu_pro_computer_science",
    "alias": "computer_science",
    "sample_len": 9,
    "exact_match,custom-extract": 0.2222222222222222,
    "exact_match_stderr,custom-extract": 0.1469861839480328
  },
  "mmlu_pro_economics": {
    "name": "mmlu_pro_economics",
    "alias": "economics",
    "sample_len": 9,
    "exact_match,custom-extract": 0.2222222222222222,
    "exact_match_stderr,custom-extract": 0.1469861839480328
  },
  "mmlu_pro_engineering": {
    "name": "mmlu_pro_engineering",
    "alias": "engineering",
    "sample_len": 9,
    "exact_match,custom-extract": 0.5555555555555556,
    "exact_match_stderr,custom-extract": 0.17568209223157663
  },
  "mmlu_pro_health": {
    "name": "mmlu_pro_health",
    "alias": "health",
    "sample_len": 9,
    "exact_match,custom-extract": 0.4444444444444444,
    "exact_match_stderr,custom-extract": 0.17568209223157663
  },
  "mmlu_pro_history": {
    "name": "mmlu_pro_history",
    "alias": "history",
    "sample_len": 9,
    "exact_match,custom-extract": 0.3333333333333333,
    "exact_match_stderr,custom-extract": 0.16666666666666666
  },
  "mmlu_pro_law": {
    "name": "mmlu_pro_law",
    "alias": "law",
    "sample_len": 9,
    "exact_match,custom-extract": 0.2222222222222222,
    "exact_match_stderr,custom-extract": 0.1469861839480328
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
    "exact_match,custom-extract": 0.4444444444444444,
    "exact_match_stderr,custom-extract": 0.17568209223157663
  },
  "mmlu_pro_philosophy": {
    "name": "mmlu_pro_philosophy",
    "alias": "philosophy",
    "sample_len": 9,
    "exact_match,custom-extract": 0.2222222222222222,
    "exact_match_stderr,custom-extract": 0.1469861839480328
  },
  "mmlu_pro_physics": {
    "name": "mmlu_pro_physics",
    "alias": "physics",
    "sample_len": 9,
    "exact_match,custom-extract": 0.2222222222222222,
    "exact_match_stderr,custom-extract": 0.1469861839480328
  },
  "mmlu_pro_psychology": {
    "name": "mmlu_pro_psychology",
    "alias": "psychology",
    "sample_len": 9,
    "exact_match,custom-extract": 0.2222222222222222,
    "exact_match_stderr,custom-extract": 0.1469861839480328
  },
  "mmlu_pro": {
    "alias": "mmlu_pro",
    "name": "mmlu_pro",
    "sample_len": 128,
    "exact_match,custom-extract": 0.3046875,
    "exact_match_stderr,custom-extract": 0.04105916783323665,
    "sample_count": {
      "exact_match,custom-extract": 128
    }
  }
}
```

## mcq / evaluation_ood / qwen_native

```json
{
  "hellaswag": {
    "name": "hellaswag",
    "alias": "hellaswag",
    "sample_len": 128,
    "acc,none": 0.4765625,
    "acc_stderr,none": 0.04431905471661607,
    "acc_norm,none": 0.5703125,
    "acc_norm_stderr,none": 0.04392693937331274
  },
  "mmlu_pro_biology": {
    "name": "mmlu_pro_biology",
    "alias": "biology",
    "sample_len": 10,
    "exact_match,custom-extract": 0.9,
    "exact_match_stderr,custom-extract": 0.09999999999999999
  },
  "mmlu_pro_business": {
    "name": "mmlu_pro_business",
    "alias": "business",
    "sample_len": 10,
    "exact_match,custom-extract": 0.7,
    "exact_match_stderr,custom-extract": 0.15275252316519466
  },
  "mmlu_pro_chemistry": {
    "name": "mmlu_pro_chemistry",
    "alias": "chemistry",
    "sample_len": 9,
    "exact_match,custom-extract": 0.8888888888888888,
    "exact_match_stderr,custom-extract": 0.11111111111111112
  },
  "mmlu_pro_computer_science": {
    "name": "mmlu_pro_computer_science",
    "alias": "computer_science",
    "sample_len": 9,
    "exact_match,custom-extract": 0.5555555555555556,
    "exact_match_stderr,custom-extract": 0.17568209223157663
  },
  "mmlu_pro_economics": {
    "name": "mmlu_pro_economics",
    "alias": "economics",
    "sample_len": 9,
    "exact_match,custom-extract": 0.5555555555555556,
    "exact_match_stderr,custom-extract": 0.17568209223157663
  },
  "mmlu_pro_engineering": {
    "name": "mmlu_pro_engineering",
    "alias": "engineering",
    "sample_len": 9,
    "exact_match,custom-extract": 0.7777777777777778,
    "exact_match_stderr,custom-extract": 0.1469861839480328
  },
  "mmlu_pro_health": {
    "name": "mmlu_pro_health",
    "alias": "health",
    "sample_len": 9,
    "exact_match,custom-extract": 0.6666666666666666,
    "exact_match_stderr,custom-extract": 0.16666666666666666
  },
  "mmlu_pro_history": {
    "name": "mmlu_pro_history",
    "alias": "history",
    "sample_len": 9,
    "exact_match,custom-extract": 0.5555555555555556,
    "exact_match_stderr,custom-extract": 0.17568209223157663
  },
  "mmlu_pro_law": {
    "name": "mmlu_pro_law",
    "alias": "law",
    "sample_len": 9,
    "exact_match,custom-extract": 0.2222222222222222,
    "exact_match_stderr,custom-extract": 0.1469861839480328
  },
  "mmlu_pro_math": {
    "name": "mmlu_pro_math",
    "alias": "math",
    "sample_len": 9,
    "exact_match,custom-extract": 0.6666666666666666,
    "exact_match_stderr,custom-extract": 0.16666666666666666
  },
  "mmlu_pro_other": {
    "name": "mmlu_pro_other",
    "alias": "other",
    "sample_len": 9,
    "exact_match,custom-extract": 0.5555555555555556,
    "exact_match_stderr,custom-extract": 0.17568209223157663
  },
  "mmlu_pro_philosophy": {
    "name": "mmlu_pro_philosophy",
    "alias": "philosophy",
    "sample_len": 9,
    "exact_match,custom-extract": 0.6666666666666666,
    "exact_match_stderr,custom-extract": 0.16666666666666666
  },
  "mmlu_pro_physics": {
    "name": "mmlu_pro_physics",
    "alias": "physics",
    "sample_len": 9,
    "exact_match,custom-extract": 0.6666666666666666,
    "exact_match_stderr,custom-extract": 0.16666666666666666
  },
  "mmlu_pro_psychology": {
    "name": "mmlu_pro_psychology",
    "alias": "psychology",
    "sample_len": 9,
    "exact_match,custom-extract": 0.2222222222222222,
    "exact_match_stderr,custom-extract": 0.1469861839480328
  },
  "mmlu_pro": {
    "alias": "mmlu_pro",
    "name": "mmlu_pro",
    "sample_len": 128,
    "exact_match,custom-extract": 0.6171875,
    "exact_match_stderr,custom-extract": 0.041757940963704915,
    "sample_count": {
      "exact_match,custom-extract": 128
    }
  }
}
```

## mcq / evaluation_ood / stage_a

```json
{
  "hellaswag": {
    "name": "hellaswag",
    "alias": "hellaswag",
    "sample_len": 128,
    "acc,none": 0.2734375,
    "acc_stderr,none": 0.03955156442885268,
    "acc_norm,none": 0.375,
    "acc_norm_stderr,none": 0.042958962288966916
  },
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
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0
  },
  "mmlu_pro_chemistry": {
    "name": "mmlu_pro_chemistry",
    "alias": "chemistry",
    "sample_len": 9,
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0
  },
  "mmlu_pro_computer_science": {
    "name": "mmlu_pro_computer_science",
    "alias": "computer_science",
    "sample_len": 9,
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0
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
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0
  },
  "mmlu_pro_health": {
    "name": "mmlu_pro_health",
    "alias": "health",
    "sample_len": 9,
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0
  },
  "mmlu_pro_history": {
    "name": "mmlu_pro_history",
    "alias": "history",
    "sample_len": 9,
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0
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
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0
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
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0,
    "sample_count": {
      "exact_match,custom-extract": 128
    }
  }
}
```

## mcq / evaluation_ood / stage_b

```json
{
  "hellaswag": {
    "name": "hellaswag",
    "alias": "hellaswag",
    "sample_len": 128,
    "acc,none": 0.375,
    "acc_stderr,none": 0.042958962288966916,
    "acc_norm,none": 0.421875,
    "acc_norm_stderr,none": 0.04382287939944462
  },
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
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0
  },
  "mmlu_pro_chemistry": {
    "name": "mmlu_pro_chemistry",
    "alias": "chemistry",
    "sample_len": 9,
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0
  },
  "mmlu_pro_computer_science": {
    "name": "mmlu_pro_computer_science",
    "alias": "computer_science",
    "sample_len": 9,
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0
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
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0
  },
  "mmlu_pro_health": {
    "name": "mmlu_pro_health",
    "alias": "health",
    "sample_len": 9,
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0
  },
  "mmlu_pro_history": {
    "name": "mmlu_pro_history",
    "alias": "history",
    "sample_len": 9,
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0
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
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0
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
    "exact_match,custom-extract": 0.0,
    "exact_match_stderr,custom-extract": 0.0,
    "sample_count": {
      "exact_match,custom-extract": 128
    }
  }
}
```

## mcq official-sample correctness overlap

```json
{
  "evaluation/openbookqa/none/acc/llama_native_vs_qwen_native": {
    "both_correct": 21,
    "method_only_correct": 15,
    "reference_only_correct": 10,
    "both_wrong": 82
  },
  "evaluation/openbookqa/none/acc_norm/llama_native_vs_qwen_native": {
    "both_correct": 48,
    "method_only_correct": 4,
    "reference_only_correct": 10,
    "both_wrong": 66
  },
  "evaluation/arc_challenge/none/acc/llama_native_vs_qwen_native": {
    "both_correct": 50,
    "method_only_correct": 8,
    "reference_only_correct": 22,
    "both_wrong": 48
  },
  "evaluation/arc_challenge/none/acc_norm/llama_native_vs_qwen_native": {
    "both_correct": 48,
    "method_only_correct": 6,
    "reference_only_correct": 23,
    "both_wrong": 51
  },
  "evaluation/openbookqa/none/acc/llama_native_vs_native_oracle": {
    "both_correct": 21,
    "method_only_correct": 15,
    "reference_only_correct": 10,
    "both_wrong": 82
  },
  "evaluation/openbookqa/none/acc_norm/llama_native_vs_native_oracle": {
    "both_correct": 48,
    "method_only_correct": 4,
    "reference_only_correct": 10,
    "both_wrong": 66
  },
  "evaluation/arc_challenge/none/acc/llama_native_vs_native_oracle": {
    "both_correct": 50,
    "method_only_correct": 8,
    "reference_only_correct": 22,
    "both_wrong": 48
  },
  "evaluation/arc_challenge/none/acc_norm/llama_native_vs_native_oracle": {
    "both_correct": 48,
    "method_only_correct": 6,
    "reference_only_correct": 23,
    "both_wrong": 51
  },
  "evaluation/openbookqa/none/acc/llama_native_vs_stage_a": {
    "both_correct": 16,
    "method_only_correct": 20,
    "reference_only_correct": 5,
    "both_wrong": 87
  },
  "evaluation/openbookqa/none/acc_norm/llama_native_vs_stage_a": {
    "both_correct": 39,
    "method_only_correct": 13,
    "reference_only_correct": 12,
    "both_wrong": 64
  },
  "evaluation/arc_challenge/none/acc/llama_native_vs_stage_a": {
    "both_correct": 22,
    "method_only_correct": 36,
    "reference_only_correct": 7,
    "both_wrong": 63
  },
  "evaluation/arc_challenge/none/acc_norm/llama_native_vs_stage_a": {
    "both_correct": 23,
    "method_only_correct": 31,
    "reference_only_correct": 9,
    "both_wrong": 65
  },
  "evaluation/openbookqa/none/acc/native_oracle_vs_qwen_native": {
    "both_correct": 31,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 97
  },
  "evaluation/openbookqa/none/acc_norm/native_oracle_vs_qwen_native": {
    "both_correct": 58,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 70
  },
  "evaluation/arc_challenge/none/acc/native_oracle_vs_qwen_native": {
    "both_correct": 72,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 56
  },
  "evaluation/arc_challenge/none/acc_norm/native_oracle_vs_qwen_native": {
    "both_correct": 71,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 57
  },
  "evaluation/openbookqa/none/acc/native_oracle_vs_stage_a": {
    "both_correct": 16,
    "method_only_correct": 15,
    "reference_only_correct": 5,
    "both_wrong": 92
  },
  "evaluation/openbookqa/none/acc_norm/native_oracle_vs_stage_a": {
    "both_correct": 41,
    "method_only_correct": 17,
    "reference_only_correct": 10,
    "both_wrong": 60
  },
  "evaluation/arc_challenge/none/acc/native_oracle_vs_stage_a": {
    "both_correct": 21,
    "method_only_correct": 51,
    "reference_only_correct": 8,
    "both_wrong": 48
  },
  "evaluation/arc_challenge/none/acc_norm/native_oracle_vs_stage_a": {
    "both_correct": 26,
    "method_only_correct": 45,
    "reference_only_correct": 6,
    "both_wrong": 51
  },
  "evaluation/openbookqa/none/acc/qwen_native_vs_native_oracle": {
    "both_correct": 31,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 97
  },
  "evaluation/openbookqa/none/acc_norm/qwen_native_vs_native_oracle": {
    "both_correct": 58,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 70
  },
  "evaluation/arc_challenge/none/acc/qwen_native_vs_native_oracle": {
    "both_correct": 72,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 56
  },
  "evaluation/arc_challenge/none/acc_norm/qwen_native_vs_native_oracle": {
    "both_correct": 71,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 57
  },
  "evaluation/openbookqa/none/acc/qwen_native_vs_stage_a": {
    "both_correct": 16,
    "method_only_correct": 15,
    "reference_only_correct": 5,
    "both_wrong": 92
  },
  "evaluation/openbookqa/none/acc_norm/qwen_native_vs_stage_a": {
    "both_correct": 41,
    "method_only_correct": 17,
    "reference_only_correct": 10,
    "both_wrong": 60
  },
  "evaluation/arc_challenge/none/acc/qwen_native_vs_stage_a": {
    "both_correct": 21,
    "method_only_correct": 51,
    "reference_only_correct": 8,
    "both_wrong": 48
  },
  "evaluation/arc_challenge/none/acc_norm/qwen_native_vs_stage_a": {
    "both_correct": 26,
    "method_only_correct": 45,
    "reference_only_correct": 6,
    "both_wrong": 51
  },
  "evaluation/openbookqa/none/acc/stage_a_vs_qwen_native": {
    "both_correct": 16,
    "method_only_correct": 5,
    "reference_only_correct": 15,
    "both_wrong": 92
  },
  "evaluation/openbookqa/none/acc_norm/stage_a_vs_qwen_native": {
    "both_correct": 41,
    "method_only_correct": 10,
    "reference_only_correct": 17,
    "both_wrong": 60
  },
  "evaluation/arc_challenge/none/acc/stage_a_vs_qwen_native": {
    "both_correct": 21,
    "method_only_correct": 8,
    "reference_only_correct": 51,
    "both_wrong": 48
  },
  "evaluation/arc_challenge/none/acc_norm/stage_a_vs_qwen_native": {
    "both_correct": 26,
    "method_only_correct": 6,
    "reference_only_correct": 45,
    "both_wrong": 51
  },
  "evaluation/openbookqa/none/acc/stage_a_vs_native_oracle": {
    "both_correct": 16,
    "method_only_correct": 5,
    "reference_only_correct": 15,
    "both_wrong": 92
  },
  "evaluation/openbookqa/none/acc_norm/stage_a_vs_native_oracle": {
    "both_correct": 41,
    "method_only_correct": 10,
    "reference_only_correct": 17,
    "both_wrong": 60
  },
  "evaluation/arc_challenge/none/acc/stage_a_vs_native_oracle": {
    "both_correct": 21,
    "method_only_correct": 8,
    "reference_only_correct": 51,
    "both_wrong": 48
  },
  "evaluation/arc_challenge/none/acc_norm/stage_a_vs_native_oracle": {
    "both_correct": 26,
    "method_only_correct": 6,
    "reference_only_correct": 45,
    "both_wrong": 51
  },
  "evaluation/openbookqa/none/acc/stage_b_vs_qwen_native": {
    "both_correct": 20,
    "method_only_correct": 10,
    "reference_only_correct": 11,
    "both_wrong": 87
  },
  "evaluation/openbookqa/none/acc_norm/stage_b_vs_qwen_native": {
    "both_correct": 51,
    "method_only_correct": 5,
    "reference_only_correct": 7,
    "both_wrong": 65
  },
  "evaluation/arc_challenge/none/acc/stage_b_vs_qwen_native": {
    "both_correct": 43,
    "method_only_correct": 10,
    "reference_only_correct": 29,
    "both_wrong": 46
  },
  "evaluation/arc_challenge/none/acc_norm/stage_b_vs_qwen_native": {
    "both_correct": 49,
    "method_only_correct": 4,
    "reference_only_correct": 22,
    "both_wrong": 53
  },
  "evaluation/openbookqa/none/acc/stage_b_vs_native_oracle": {
    "both_correct": 20,
    "method_only_correct": 10,
    "reference_only_correct": 11,
    "both_wrong": 87
  },
  "evaluation/openbookqa/none/acc_norm/stage_b_vs_native_oracle": {
    "both_correct": 51,
    "method_only_correct": 5,
    "reference_only_correct": 7,
    "both_wrong": 65
  },
  "evaluation/arc_challenge/none/acc/stage_b_vs_native_oracle": {
    "both_correct": 43,
    "method_only_correct": 10,
    "reference_only_correct": 29,
    "both_wrong": 46
  },
  "evaluation/arc_challenge/none/acc_norm/stage_b_vs_native_oracle": {
    "both_correct": 49,
    "method_only_correct": 4,
    "reference_only_correct": 22,
    "both_wrong": 53
  },
  "evaluation/openbookqa/none/acc/stage_b_vs_stage_a": {
    "both_correct": 15,
    "method_only_correct": 15,
    "reference_only_correct": 6,
    "both_wrong": 92
  },
  "evaluation/openbookqa/none/acc_norm/stage_b_vs_stage_a": {
    "both_correct": 45,
    "method_only_correct": 11,
    "reference_only_correct": 6,
    "both_wrong": 66
  },
  "evaluation/arc_challenge/none/acc/stage_b_vs_stage_a": {
    "both_correct": 17,
    "method_only_correct": 36,
    "reference_only_correct": 12,
    "both_wrong": 63
  },
  "evaluation/arc_challenge/none/acc_norm/stage_b_vs_stage_a": {
    "both_correct": 25,
    "method_only_correct": 28,
    "reference_only_correct": 7,
    "both_wrong": 68
  },
  "evaluation_ood/hellaswag/none/acc/llama_native_vs_qwen_native": {
    "both_correct": 56,
    "method_only_correct": 4,
    "reference_only_correct": 5,
    "both_wrong": 63
  },
  "evaluation_ood/hellaswag/none/acc_norm/llama_native_vs_qwen_native": {
    "both_correct": 64,
    "method_only_correct": 16,
    "reference_only_correct": 9,
    "both_wrong": 39
  },
  "evaluation_ood/mmlu_pro_biology/custom-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 5,
    "method_only_correct": 0,
    "reference_only_correct": 4,
    "both_wrong": 1
  },
  "evaluation_ood/mmlu_pro_business/custom-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 3,
    "method_only_correct": 0,
    "reference_only_correct": 4,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_chemistry/custom-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 2,
    "method_only_correct": 1,
    "reference_only_correct": 6,
    "both_wrong": 0
  },
  "evaluation_ood/mmlu_pro_computer_science/custom-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 1,
    "method_only_correct": 1,
    "reference_only_correct": 4,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_economics/custom-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 2,
    "method_only_correct": 0,
    "reference_only_correct": 3,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_engineering/custom-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 4,
    "method_only_correct": 1,
    "reference_only_correct": 3,
    "both_wrong": 1
  },
  "evaluation_ood/mmlu_pro_health/custom-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 3,
    "method_only_correct": 1,
    "reference_only_correct": 3,
    "both_wrong": 2
  },
  "evaluation_ood/mmlu_pro_history/custom-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 3,
    "method_only_correct": 0,
    "reference_only_correct": 2,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_law/custom-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 1,
    "method_only_correct": 1,
    "reference_only_correct": 1,
    "both_wrong": 6
  },
  "evaluation_ood/mmlu_pro_math/custom-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 6,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_other/custom-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 4,
    "method_only_correct": 0,
    "reference_only_correct": 1,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_philosophy/custom-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 2,
    "method_only_correct": 0,
    "reference_only_correct": 4,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_physics/custom-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 2,
    "method_only_correct": 0,
    "reference_only_correct": 4,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_psychology/custom-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 1,
    "method_only_correct": 1,
    "reference_only_correct": 1,
    "both_wrong": 6
  },
  "evaluation_ood/hellaswag/none/acc/llama_native_vs_stage_a": {
    "both_correct": 32,
    "method_only_correct": 28,
    "reference_only_correct": 3,
    "both_wrong": 65
  },
  "evaluation_ood/hellaswag/none/acc_norm/llama_native_vs_stage_a": {
    "both_correct": 35,
    "method_only_correct": 45,
    "reference_only_correct": 13,
    "both_wrong": 35
  },
  "evaluation_ood/mmlu_pro_biology/custom-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 5,
    "reference_only_correct": 0,
    "both_wrong": 5
  },
  "evaluation_ood/mmlu_pro_business/custom-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 3,
    "reference_only_correct": 0,
    "both_wrong": 7
  },
  "evaluation_ood/mmlu_pro_chemistry/custom-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 3,
    "reference_only_correct": 0,
    "both_wrong": 6
  },
  "evaluation_ood/mmlu_pro_computer_science/custom-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 2,
    "reference_only_correct": 0,
    "both_wrong": 7
  },
  "evaluation_ood/mmlu_pro_economics/custom-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 2,
    "reference_only_correct": 0,
    "both_wrong": 7
  },
  "evaluation_ood/mmlu_pro_engineering/custom-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 5,
    "reference_only_correct": 0,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_health/custom-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 4,
    "reference_only_correct": 0,
    "both_wrong": 5
  },
  "evaluation_ood/mmlu_pro_history/custom-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 3,
    "reference_only_correct": 0,
    "both_wrong": 6
  },
  "evaluation_ood/mmlu_pro_law/custom-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 2,
    "reference_only_correct": 0,
    "both_wrong": 7
  },
  "evaluation_ood/mmlu_pro_math/custom-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 9
  },
  "evaluation_ood/mmlu_pro_other/custom-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 4,
    "reference_only_correct": 0,
    "both_wrong": 5
  },
  "evaluation_ood/mmlu_pro_philosophy/custom-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 2,
    "reference_only_correct": 0,
    "both_wrong": 7
  },
  "evaluation_ood/mmlu_pro_physics/custom-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 2,
    "reference_only_correct": 0,
    "both_wrong": 7
  },
  "evaluation_ood/mmlu_pro_psychology/custom-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 2,
    "reference_only_correct": 0,
    "both_wrong": 7
  },
  "evaluation_ood/hellaswag/none/acc/qwen_native_vs_stage_a": {
    "both_correct": 32,
    "method_only_correct": 29,
    "reference_only_correct": 3,
    "both_wrong": 64
  },
  "evaluation_ood/hellaswag/none/acc_norm/qwen_native_vs_stage_a": {
    "both_correct": 36,
    "method_only_correct": 37,
    "reference_only_correct": 12,
    "both_wrong": 43
  },
  "evaluation_ood/mmlu_pro_biology/custom-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 9,
    "reference_only_correct": 0,
    "both_wrong": 1
  },
  "evaluation_ood/mmlu_pro_business/custom-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 7,
    "reference_only_correct": 0,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_chemistry/custom-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 8,
    "reference_only_correct": 0,
    "both_wrong": 1
  },
  "evaluation_ood/mmlu_pro_computer_science/custom-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 5,
    "reference_only_correct": 0,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_economics/custom-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 5,
    "reference_only_correct": 0,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_engineering/custom-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 7,
    "reference_only_correct": 0,
    "both_wrong": 2
  },
  "evaluation_ood/mmlu_pro_health/custom-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 6,
    "reference_only_correct": 0,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_history/custom-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 5,
    "reference_only_correct": 0,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_law/custom-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 2,
    "reference_only_correct": 0,
    "both_wrong": 7
  },
  "evaluation_ood/mmlu_pro_math/custom-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 6,
    "reference_only_correct": 0,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_other/custom-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 5,
    "reference_only_correct": 0,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_philosophy/custom-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 6,
    "reference_only_correct": 0,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_physics/custom-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 6,
    "reference_only_correct": 0,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_psychology/custom-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 2,
    "reference_only_correct": 0,
    "both_wrong": 7
  },
  "evaluation_ood/hellaswag/none/acc/stage_a_vs_qwen_native": {
    "both_correct": 32,
    "method_only_correct": 3,
    "reference_only_correct": 29,
    "both_wrong": 64
  },
  "evaluation_ood/hellaswag/none/acc_norm/stage_a_vs_qwen_native": {
    "both_correct": 36,
    "method_only_correct": 12,
    "reference_only_correct": 37,
    "both_wrong": 43
  },
  "evaluation_ood/mmlu_pro_biology/custom-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 9,
    "both_wrong": 1
  },
  "evaluation_ood/mmlu_pro_business/custom-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 7,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_chemistry/custom-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 8,
    "both_wrong": 1
  },
  "evaluation_ood/mmlu_pro_computer_science/custom-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 5,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_economics/custom-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 5,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_engineering/custom-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 7,
    "both_wrong": 2
  },
  "evaluation_ood/mmlu_pro_health/custom-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 6,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_history/custom-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 5,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_law/custom-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 2,
    "both_wrong": 7
  },
  "evaluation_ood/mmlu_pro_math/custom-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 6,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_other/custom-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 5,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_philosophy/custom-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 6,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_physics/custom-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 6,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_psychology/custom-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 2,
    "both_wrong": 7
  },
  "evaluation_ood/hellaswag/none/acc/stage_b_vs_qwen_native": {
    "both_correct": 45,
    "method_only_correct": 3,
    "reference_only_correct": 16,
    "both_wrong": 64
  },
  "evaluation_ood/hellaswag/none/acc_norm/stage_b_vs_qwen_native": {
    "both_correct": 47,
    "method_only_correct": 7,
    "reference_only_correct": 26,
    "both_wrong": 48
  },
  "evaluation_ood/mmlu_pro_biology/custom-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 9,
    "both_wrong": 1
  },
  "evaluation_ood/mmlu_pro_business/custom-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 7,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_chemistry/custom-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 8,
    "both_wrong": 1
  },
  "evaluation_ood/mmlu_pro_computer_science/custom-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 5,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_economics/custom-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 5,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_engineering/custom-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 7,
    "both_wrong": 2
  },
  "evaluation_ood/mmlu_pro_health/custom-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 6,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_history/custom-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 5,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_law/custom-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 2,
    "both_wrong": 7
  },
  "evaluation_ood/mmlu_pro_math/custom-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 6,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_other/custom-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 5,
    "both_wrong": 4
  },
  "evaluation_ood/mmlu_pro_philosophy/custom-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 6,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_physics/custom-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 6,
    "both_wrong": 3
  },
  "evaluation_ood/mmlu_pro_psychology/custom-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 2,
    "both_wrong": 7
  },
  "evaluation_ood/hellaswag/none/acc/stage_b_vs_stage_a": {
    "both_correct": 31,
    "method_only_correct": 17,
    "reference_only_correct": 4,
    "both_wrong": 76
  },
  "evaluation_ood/hellaswag/none/acc_norm/stage_b_vs_stage_a": {
    "both_correct": 30,
    "method_only_correct": 24,
    "reference_only_correct": 18,
    "both_wrong": 56
  },
  "evaluation_ood/mmlu_pro_biology/custom-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 10
  },
  "evaluation_ood/mmlu_pro_business/custom-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 10
  },
  "evaluation_ood/mmlu_pro_chemistry/custom-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 9
  },
  "evaluation_ood/mmlu_pro_computer_science/custom-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 9
  },
  "evaluation_ood/mmlu_pro_economics/custom-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 9
  },
  "evaluation_ood/mmlu_pro_engineering/custom-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 9
  },
  "evaluation_ood/mmlu_pro_health/custom-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 9
  },
  "evaluation_ood/mmlu_pro_history/custom-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 9
  },
  "evaluation_ood/mmlu_pro_law/custom-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 9
  },
  "evaluation_ood/mmlu_pro_math/custom-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 9
  },
  "evaluation_ood/mmlu_pro_other/custom-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 9
  },
  "evaluation_ood/mmlu_pro_philosophy/custom-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 9
  },
  "evaluation_ood/mmlu_pro_physics/custom-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 9
  },
  "evaluation_ood/mmlu_pro_psychology/custom-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 0,
    "method_only_correct": 0,
    "reference_only_correct": 0,
    "both_wrong": 9
  }
}
```

## mcq gradient diagnostics

```json
{
  "stage_a": {
    "optimizer_steps": 512,
    "clip_rate": 0.0,
    "pre_clip_grad_norm": {
      "min": 0.035165537148714066,
      "p10": 0.042609299346804616,
      "p25": 0.045788779854774475,
      "median": 0.050436533987522125,
      "p75": 0.05602484382688999,
      "p90": 0.06878965198993685,
      "p95": 0.11540909148752686,
      "max": 0.4731084704399109
    }
  },
  "stage_b": {
    "optimizer_steps": 1024,
    "clip_rate": 0.08984375,
    "pre_clip_grad_norm": {
      "min": 0.19783668220043182,
      "p10": 0.7716872096061707,
      "p25": 1.2590268552303314,
      "median": 2.5455933809280396,
      "p75": 7.881008744239807,
      "p90": 26.311702919006354,
      "p95": 51.59217262268066,
      "max": 589.1163330078125
    }
  }
}
```

## gsm8k / evaluation / llama_native

```json
{
  "gsm8k": {
    "name": "gsm8k",
    "alias": "gsm8k",
    "sample_len": 128,
    "exact_match,strict-match": 0.640625,
    "exact_match_stderr,strict-match": 0.04257689748916834,
    "exact_match,flexible-extract": 0.6875,
    "exact_match_stderr,flexible-extract": 0.041130075016539196
  }
}
```

## gsm8k / evaluation / native_oracle

```json
{
  "gsm8k": {
    "name": "gsm8k",
    "alias": "gsm8k",
    "sample_len": 128,
    "exact_match,strict-match": 0.8671875,
    "exact_match_stderr,strict-match": 0.030114393430435732,
    "exact_match,flexible-extract": 0.8671875,
    "exact_match_stderr,flexible-extract": 0.030114393430435732
  }
}
```

## gsm8k / evaluation / qwen_native

```json
{
  "gsm8k": {
    "name": "gsm8k",
    "alias": "gsm8k",
    "sample_len": 128,
    "exact_match,strict-match": 0.859375,
    "exact_match_stderr,strict-match": 0.030847556262404395,
    "exact_match,flexible-extract": 0.859375,
    "exact_match_stderr,flexible-extract": 0.030847556262404395
  }
}
```

## gsm8k / evaluation / stage_a

```json
{
  "gsm8k": {
    "name": "gsm8k",
    "alias": "gsm8k",
    "sample_len": 128,
    "exact_match,strict-match": 0.015625,
    "exact_match_stderr,strict-match": 0.011004959288293975,
    "exact_match,flexible-extract": 0.0390625,
    "exact_match_stderr,flexible-extract": 0.017191973290465488
  }
}
```

## gsm8k / evaluation / stage_b

```json
{
  "gsm8k": {
    "name": "gsm8k",
    "alias": "gsm8k",
    "sample_len": 128,
    "exact_match,strict-match": 0.6875,
    "exact_match_stderr,strict-match": 0.041130075016539196,
    "exact_match,flexible-extract": 0.6875,
    "exact_match_stderr,flexible-extract": 0.041130075016539196
  }
}
```

## gsm8k official-sample correctness overlap

```json
{
  "evaluation/gsm8k/strict-match/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 77,
    "method_only_correct": 5,
    "reference_only_correct": 33,
    "both_wrong": 13
  },
  "evaluation/gsm8k/flexible-extract/exact_match/llama_native_vs_qwen_native": {
    "both_correct": 82,
    "method_only_correct": 6,
    "reference_only_correct": 28,
    "both_wrong": 12
  },
  "evaluation/gsm8k/strict-match/exact_match/llama_native_vs_native_oracle": {
    "both_correct": 78,
    "method_only_correct": 4,
    "reference_only_correct": 33,
    "both_wrong": 13
  },
  "evaluation/gsm8k/flexible-extract/exact_match/llama_native_vs_native_oracle": {
    "both_correct": 83,
    "method_only_correct": 5,
    "reference_only_correct": 28,
    "both_wrong": 12
  },
  "evaluation/gsm8k/strict-match/exact_match/llama_native_vs_stage_a": {
    "both_correct": 2,
    "method_only_correct": 80,
    "reference_only_correct": 0,
    "both_wrong": 46
  },
  "evaluation/gsm8k/flexible-extract/exact_match/llama_native_vs_stage_a": {
    "both_correct": 4,
    "method_only_correct": 84,
    "reference_only_correct": 1,
    "both_wrong": 39
  },
  "evaluation/gsm8k/strict-match/exact_match/native_oracle_vs_qwen_native": {
    "both_correct": 110,
    "method_only_correct": 1,
    "reference_only_correct": 0,
    "both_wrong": 17
  },
  "evaluation/gsm8k/flexible-extract/exact_match/native_oracle_vs_qwen_native": {
    "both_correct": 110,
    "method_only_correct": 1,
    "reference_only_correct": 0,
    "both_wrong": 17
  },
  "evaluation/gsm8k/strict-match/exact_match/native_oracle_vs_stage_a": {
    "both_correct": 2,
    "method_only_correct": 109,
    "reference_only_correct": 0,
    "both_wrong": 17
  },
  "evaluation/gsm8k/flexible-extract/exact_match/native_oracle_vs_stage_a": {
    "both_correct": 5,
    "method_only_correct": 106,
    "reference_only_correct": 0,
    "both_wrong": 17
  },
  "evaluation/gsm8k/strict-match/exact_match/qwen_native_vs_native_oracle": {
    "both_correct": 110,
    "method_only_correct": 0,
    "reference_only_correct": 1,
    "both_wrong": 17
  },
  "evaluation/gsm8k/flexible-extract/exact_match/qwen_native_vs_native_oracle": {
    "both_correct": 110,
    "method_only_correct": 0,
    "reference_only_correct": 1,
    "both_wrong": 17
  },
  "evaluation/gsm8k/strict-match/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 2,
    "method_only_correct": 108,
    "reference_only_correct": 0,
    "both_wrong": 18
  },
  "evaluation/gsm8k/flexible-extract/exact_match/qwen_native_vs_stage_a": {
    "both_correct": 5,
    "method_only_correct": 105,
    "reference_only_correct": 0,
    "both_wrong": 18
  },
  "evaluation/gsm8k/strict-match/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 2,
    "method_only_correct": 0,
    "reference_only_correct": 108,
    "both_wrong": 18
  },
  "evaluation/gsm8k/flexible-extract/exact_match/stage_a_vs_qwen_native": {
    "both_correct": 5,
    "method_only_correct": 0,
    "reference_only_correct": 105,
    "both_wrong": 18
  },
  "evaluation/gsm8k/strict-match/exact_match/stage_a_vs_native_oracle": {
    "both_correct": 2,
    "method_only_correct": 0,
    "reference_only_correct": 109,
    "both_wrong": 17
  },
  "evaluation/gsm8k/flexible-extract/exact_match/stage_a_vs_native_oracle": {
    "both_correct": 5,
    "method_only_correct": 0,
    "reference_only_correct": 106,
    "both_wrong": 17
  },
  "evaluation/gsm8k/strict-match/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 83,
    "method_only_correct": 5,
    "reference_only_correct": 27,
    "both_wrong": 13
  },
  "evaluation/gsm8k/flexible-extract/exact_match/stage_b_vs_qwen_native": {
    "both_correct": 82,
    "method_only_correct": 6,
    "reference_only_correct": 28,
    "both_wrong": 12
  },
  "evaluation/gsm8k/strict-match/exact_match/stage_b_vs_native_oracle": {
    "both_correct": 84,
    "method_only_correct": 4,
    "reference_only_correct": 27,
    "both_wrong": 13
  },
  "evaluation/gsm8k/flexible-extract/exact_match/stage_b_vs_native_oracle": {
    "both_correct": 83,
    "method_only_correct": 5,
    "reference_only_correct": 28,
    "both_wrong": 12
  },
  "evaluation/gsm8k/strict-match/exact_match/stage_b_vs_stage_a": {
    "both_correct": 2,
    "method_only_correct": 86,
    "reference_only_correct": 0,
    "both_wrong": 40
  },
  "evaluation/gsm8k/flexible-extract/exact_match/stage_b_vs_stage_a": {
    "both_correct": 4,
    "method_only_correct": 84,
    "reference_only_correct": 1,
    "both_wrong": 39
  }
}
```

## gsm8k gradient diagnostics

```json
{
  "stage_a": {
    "optimizer_steps": 512,
    "clip_rate": 0.0,
    "pre_clip_grad_norm": {
      "min": 0.024977440014481544,
      "p10": 0.030105697736144067,
      "p25": 0.037641274742782116,
      "median": 0.04164319299161434,
      "p75": 0.04508815985172987,
      "p90": 0.05816899798810482,
      "p95": 0.09158315248787398,
      "max": 0.4643726348876953
    }
  },
  "stage_b": {
    "optimizer_steps": 1024,
    "clip_rate": 0.0,
    "pre_clip_grad_norm": {
      "min": 0.06583500653505325,
      "p10": 0.09852892085909844,
      "p25": 0.11467713862657547,
      "median": 0.14380714297294617,
      "p75": 0.2008536420762539,
      "p90": 0.5349262893199923,
      "p95": 1.5878032088279723,
      "max": 17.76038360595703
    }
  }
}
```
