# All-layer Keep-both inference mask intervention

Same existing checkpoint; no training; seven identical64-question cohorts.

| Task | Metric | C | RN-KV | Keep-both19-30 | Keep-both1-36 | Native |
|---|---|---:|---:|---:|---:|---:|
| arc | correct | 67.19% | 64.06% | 67.19% | 67.19% | 87.50% |
| hotpotqa | em | 29.69% | 35.94% | 40.62% | 40.62% | 56.25% |
| hotpotqa | f1 | 46.53% | 49.80% | 56.50% | 55.06% | 68.52% |
| 2wiki | em | 9.38% | 21.88% | 29.69% | 28.12% | 32.81% |
| 2wiki | f1 | 22.05% | 35.32% | 42.91% | 41.31% | 43.45% |
| musique | em | 1.56% | 12.50% | 23.44% | 18.75% | 25.00% |
| musique | f1 | 7.64% | 17.85% | 31.88% | 29.76% | 33.07% |
| sciq | correct | 98.44% | 100.00% | 98.44% | 100.00% | 100.00% |
| mmlu_pro | correct | 25.00% | 26.56% | 28.12% | 29.69% | 53.12% |
| hellaswag | correct | 57.81% | 50.00% | 60.94% | 59.38% | 45.31% |

- Use the existing separately trained Keep-both epoch4/step1024 checkpoint, not original replacement RN-KV weights. No training or weight changes.
- Only answer-time memory visibility changes: all36 layers may read translated body plus32 Memory KV. Construction, four memory adapters, native K/V projections, positions and causal suffix/generated-token visibility remain unchanged. No native evidence prefill, blending or learned gate.
- The checkpoint was trained with memory readable at answer layers19-30 only. This is an inference mask intervention with a train/test routing mismatch, not an all-layer-trained Keep-both method.
- Reuse exact seven64-question cohorts and scorers. ARC/SciQ/MMLU-Pro label likelihood, HellaSwag normalized ending likelihood, and Hotpot/2Wiki/MuSiQue64-token greedy generation. Baseline predictions are reused with source hashes; partial-layer baseline is reproduced on the first example of every task before the new arm.
- GSM8K and earlier paused jobs remain paused. No automatic upload or continuous monitoring.
