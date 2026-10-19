# Same-question heterogeneous KV concatenation

Completed 128/128 examples. No training. Legacy Flexible accuracy.
Single-source/Mean50 baselines reused from the preceding run, not rerun formal arms.

| Arm | Correct | Accuracy | Mean generated tokens | Max-token rate |
|---|---:|---:|---:|---:|
| G (reused) | 86 | 67.19% | 148.17 | 16.41% |
| Mean50 (reused) | 87 | 67.97% | 155.73 | 20.31% |
| L (reused) | 70 | 54.69% | 163.04 | 23.44% |
| L+L | 74 | 57.81% | 159.12 | 17.97% |
| G+G | 84 | 65.62% | 150.54 | 15.62% |
| L+G | 90 | 70.31% | 146.71 | 12.50% |
| G+L | 89 | 69.53% | 154.41 | 16.41% |

## Endpoint-group retention

| Group | n | L+L | G+G | L+G | G+L |
|---|---:|---:|---:|---:|---:|
| both | 59 | 55 | 54 | 58 | 59 |
| neither | 31 | 4 | 6 | 5 | 4 |
| gemma_only | 27 | 6 | 19 | 19 | 19 |
| llama_only | 11 | 9 | 5 | 8 | 7 |

Paired counts and paired bootstrap intervals for all arm pairs are in summary.json.
Interpret heterogeneous arms against SAME-LENGTH G+G/L+L before claiming multi-source benefit.
No distinct evidence fusion claim; each source saw the same question. Order effects are reported, not optimized on test.
