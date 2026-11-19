# HotpotQA-trained Translator: 128-question evidence KV diagnostic

Same 128 previously evaluated official-dev questions, same prompt and donor mapping. No overlap with the HotpotQA checkpoint train/validation questions. No Reader or parameter training. This is an exploratory checkpoint comparison, not a new independent selector test.

| Version / arm | EM | F1 |
|---|---:|---:|
| mixed_task_reference / compact | 8.59% | 15.72% |
| mixed_task_reference / reject | 6.25% | 13.99% |
| mixed_task_reference / native | 43.75% | 63.67% |
| mixed_task_reference / clean | 14.06% | 25.01% |
| mixed_task_reference / wrong | 5.47% | 11.21% |
| mixed_task_reference / partial50 | 7.03% | 20.07% |
| hotpot_trained / compact | 8.59% | 15.72% |
| hotpot_trained / reject | 6.25% | 13.99% |
| hotpot_trained / native | 43.75% | 63.67% |
| hotpot_trained / clean | 14.06% | 27.08% |
| hotpot_trained / wrong | 0.78% | 4.96% |
| hotpot_trained / partial50 | 5.47% | 15.49% |

## Potential for selection

| New checkpoint condition | EM helpful / harmful cases | EM oracle | F1 oracle | F1 oracle minus best fixed |
|---|---:|---:|---:|---:|
| clean | 15 / 5 | 17.97% | 33.14% | 6.07 pp [3.44, 9.22] |
| wrong | 0 / 7 | 6.25% | 14.95% | 0.96 pp [0.32, 1.78] |
| partial50 | 4 / 5 | 9.38% | 21.45% | 5.96 pp [3.33, 9.19] |
| equal_mixture | 19 / 17 | 11.20% | 23.18% | 7.34 pp [4.66, 10.58] |

## Paired checkpoint changes

- clean F1: +2.06 pp; paired 95% CI [-4.80, +9.13].
- wrong F1: -6.24 pp; paired 95% CI [-10.67, -2.26].
- partial50 F1: -4.58 pp; paired 95% CI [-10.21, +0.89].

The oracle uses gold only after generation, and is an upper bound, not an operational selector. All mixture bootstrap intervals resample the 128 original questions with their conditions together (5000 replicates). The equal mixture has 384 cases, not 384 independent questions.

Question-gain > 0 is included as a fixed historical diagnostic in summary.json; no threshold is fitted. Stability features and learned selectors are deliberately not run until task utility is established.

The new checkpoint was trained on full distractor prompts. This evidence-only interface remains a distribution shift even though the dataset name matches. Previously reported distractor512 accuracy is not directly comparable.

Baseline generation mismatches versus historical run: {"compact": 0, "reject": 0, "native": 0}

Evaluation elapsed seconds: 859.1.
