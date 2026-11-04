# New3-task checkpoint: restrict memory access to19-30

Same new3-task checkpoint; no retraining; eight identical64-question cohorts.

| Task | Metric | C | Old2-task Keep-both | New3-task all36 | New3-task restricted19-30 | Native |
|---|---|---:|---:|---:|---:|---:|
| arc | accuracy | 67.19% | 67.19% | 70.31% | 64.06% | 87.50% |
| hotpotqa | em | 29.69% | 40.62% | 46.88% | 48.44% | 56.25% |
| hotpotqa | f1 | 46.53% | 56.50% | 61.09% | 60.36% | 68.52% |
| 2wiki | em | 9.38% | 29.69% | 32.81% | 21.88% | 32.81% |
| 2wiki | f1 | 22.05% | 42.91% | 43.95% | 35.60% | 43.45% |
| musique | em | 1.56% | 23.44% | 21.88% | 21.88% | 25.00% |
| musique | f1 | 7.64% | 31.88% | 27.79% | 27.01% | 33.07% |
| sciq | accuracy | 98.44% | 98.44% | 98.44% | 96.88% | 100.00% |
| mmlu_pro | accuracy | 25.00% | 28.12% | 21.88% | 26.56% | 53.12% |
| hellaswag | accuracy | 57.81% | 60.94% | 60.94% | 59.38% | 45.31% |
| gsm8k | strict final numeric | 70.31% | 43.75% | 75.00% | 65.62% | 89.06% |
| gsm8k | legacy numeric | 70.31% | 70.31% | 75.00% | 67.19% | 89.06% |

- Use the NEW ARC+HotpotQA+GSM8K-trained all36 Keep-both epoch4/step1536 checkpoint. Never substitute old2-task partial-layer weights. Only inference-time memory access changes to19-30; translated body remains visible at every layer.
- Memory construction,32 learned tokens,4 trained adapters after6/12/18/24, native K/V projection and RoPE positions remain unchanged. Preserve causal suffix/generated-token access. No native evidence prefill, body overwrite, gating or output fusion.
- The new checkpoint was trained with memory readable at all36 layers, so19-30 restriction introduces train/test routing mismatch. This intervention is not a newly trained partial-layer method.
- Reuse all eight fixed64-question cohorts, scorers, QA64/GSM512 generation caps and both GSM numeric extraction metrics. Reproduce the parent all36 output on the first example of every task; reuse immutable baseline predictions with source hashes.
- Other paused experiments remain paused. No retraining, automatic upload or continuous monitoring.
