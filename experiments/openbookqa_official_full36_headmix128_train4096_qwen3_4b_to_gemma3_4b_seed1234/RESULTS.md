# Qwen3-4B -> Gemma3-4B Full36 HeadMix128 Results

Protocol: `openbookqa_official_qwen3_4b_to_gemma3_4b_full36_headmix128_options32_v1`; split: `4096/128/128`; seed `1234`.

## Native baselines and controls

| Condition | Correct | Accuracy |
|---|---:|---:|
| qwen_full_native | 96 | 75.00% |
| qwen_native_selected32 | 94 | 73.44% |
| gemma_full_native | 97 | 75.78% |
| gemma_native_oracle32 | 91 | 71.09% |
| gemma_self_selected32 | 90 | 70.31% |
| gemma_zero32 | 32 | 25.00% |
| gemma_no_memory | 33 | 25.78% |
| gemma_shuffled_oracle32 | 34 | 26.56% |

## Translator results

| Method | Accuracy | Oracle agreement | Choice KL | K cosine | V cosine | Oracle gap |
|---|---:|---:|---:|---:|---:|---:|
| stage_a | 67.19% | 71.88% | 0.783121 | 0.949462 | 0.812088 | 3.91 pp |
| residual | 70.31% | 74.22% | 0.513237 | 0.949173 | 0.811704 | 0.78 pp |
| shared | 70.31% | 73.44% | 0.610492 | 0.725969 | 0.747135 | 0.78 pp |

## Paired correctness

```json
{
  "paired_stage_a_vs_residual": {
    "both_correct": 86,
    "only_first_correct": 0,
    "only_second_correct": 4,
    "both_wrong": 38,
    "second_minus_first": 0.03125,
    "mcnemar_p": 0.125
  },
  "paired_stage_a_vs_shared": {
    "both_correct": 75,
    "only_first_correct": 11,
    "only_second_correct": 15,
    "both_wrong": 27,
    "second_minus_first": 0.03125,
    "mcnemar_p": 0.557197093963623
  },
  "paired_residual_vs_shared": {
    "both_correct": 78,
    "only_first_correct": 12,
    "only_second_correct": 12,
    "both_wrong": 26,
    "second_minus_first": 0.0,
    "mcnemar_p": 1.0
  }
}
```

The machine-readable per-sample predictions, training traces, native controls, and learned head/layer correspondence matrices are stored under `runs/study/`.
