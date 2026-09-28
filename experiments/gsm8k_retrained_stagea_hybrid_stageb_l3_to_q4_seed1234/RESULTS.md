# Results

## Protocol

- Direction: Llama3.2-3B -> Qwen3-4B.
- Dataset: GSM8K, exact standard `Question: ...\nAnswer:` prefix.
- Split: 1024 training examples, 128 validation examples, and 128 untouched official-test examples.
- Stage-A: fresh Full28 MLP, pure K/V reconstruction, two epochs.
- Stage-B: frozen Stage-A plus a fresh zero-initialized Residual64 adapter; trajectory KL + 0.1 answer CE + 0.1 EOS CE, two epochs.
- Checkpoint selection uses validation only.

## Selected checkpoints

Stage-A selected epoch 2 / step 256:

| Metric | Value |
|---|---:|
| K NMSE | 0.069967 |
| V NMSE | 0.550893 |
| K cosine | 0.920535 |
| V cosine | 0.718011 |
| Representation loss | 0.982313 |

Hybrid Stage-B selected epoch 2 / step 256:

| Metric | Value |
|---|---:|
| Validation flexible accuracy | 36.72% (47/128) |
| EOS/stop rate | 88.28% |
| Max-token rate | 11.72% |
| Trajectory KL | 0.223282 |
| Answer CE | 0.819371 |
| EOS CE | 0.004912 |

## Untouched test128

| Condition | Flexible accuracy | EOS/stop rate | Max-token rate | Mean generated tokens |
|---|---:|---:|---:|---:|
| New GSM8K Stage-A only | 0.00% (0/128) | 0.00% | 100.00% | 256.00 |
| New GSM8K Stage-A + Hybrid Stage-B | **39.84% (51/128)** | **85.94%** | 14.06% | 139.97 |
| Previous out-of-domain Stage-A only | 1.56% (2/128) | 0.00% | 100.00% | 256.00 |
| Previous Stage-A + Hybrid Stage-B | **39.84% (51/128)** | 76.56% | 23.44% | 155.61 |

## Conclusion

Retraining Stage-A on GSM8K did not improve final flexible accuracy: the new
and previous Hybrid systems both scored 51/128. It did improve generation
stability: EOS/stop rate rose from 76.56% to 85.94%, max-length failures fell
from 23.44% to 14.06%, and mean output length fell by about 15.6 tokens.
Stage-A-only generation remained unusable, so the result does not support
Stage-A domain mismatch as the main explanation for the remaining accuracy
gap.

Raw evidence is preserved in `runs/test128_per_sample.jsonl`, validation JSONL
files, training-step JSONL files, and `logs/pipeline.log`. Model checkpoints and
runtime K/V caches are intentionally excluded from GitHub.
