# Different-text KV composition, fixed32 GSM8K questions

Frozen GSM8K Stage-A+Hybrid Writers. Old greedy256/stops/Flexible. Text order A→B fixed.
full_* references reused, nine fragment conditions newly generated.

| Condition | Correct | Flexible accuracy | Mean generated tokens | Max-token rate |
|---|---:|---:|---:|---:|
| full_gemma | 17 | 53.12% | 152.75 | 21.88% |
| full_mixed50 | 21 | 65.62% | 159.44 | 18.75% |
| full_llama | 15 | 46.88% | 181.31 | 34.38% |
| llamaA_llamaB | 5 | 15.62% | 148.50 | 28.12% |
| gemmaA_gemmaB | 10 | 31.25% | 164.97 | 34.38% |
| llamaA_gemmaB | 7 | 21.88% | 139.38 | 12.50% |
| gemmaA_llamaB | 8 | 25.00% | 154.81 | 25.00% |
| llamaA_gemmaB_maskA | 2 | 6.25% | 123.28 | 21.88% |
| llamaA_gemmaB_maskB | 0 | 0.00% | 150.16 | 31.25% |
| gemmaA_llamaB_maskA | 3 | 9.38% | 109.06 | 21.88% |
| gemmaA_llamaB_maskB | 3 | 9.38% | 188.19 | 56.25% |
| nativeA_nativeB | 7 | 21.88% | 191.41 | 65.62% |

Pairwise counts/intervals and endpoint groups are in formal/summary.json.
Inspect splits.md before semantic conclusions; no new parameters were trained.
