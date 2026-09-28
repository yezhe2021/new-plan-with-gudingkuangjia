# Results

Generation Stage-B v2 completed on 2026-09-22. It kept Full-Sync alignment,
the frozen Stage-A Full28 MLP, and the zero-initialized Residual64 architecture
fixed while varying only the Stage-B objective. The seed-1234 split exactly
reused the old 1,024 training indices and used the next 128 shuffled GSM8K
train rows for validation. Official test data was untouched by training and
checkpoint selection.

## Protocol

- Prompt: `Question: {question}\nAnswer:` (lm-eval GSM8K v3 parity)
- Train / validation / test: 1024 / 128 / first 128 official test rows
- Two epochs, effective batch 8, 256 optimizer steps per objective
- AdamW, LR `1e-4`, weight decay 0, gradient clip 30
- Greedy generation, maximum 256 new tokens
- Checkpoint selection: first-`####` accuracy, then strict, flexible, lower
  max-token rate, and lower trajectory KL

All required protocol audits passed: prompt parity, native full-prefill versus
prefix-cache trajectory parity, exact zero-residual identity, gradient
ownership, cache-length parity, EOS membership, and native-cache KL self-check.

## Selected validation checkpoints

| Objective | Epoch | First-`####` | Strict | Flexible | Trajectory KL | EOS rate | Max256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE + EOS | 2 | 32/128 (25.00%) | 25.00% | 25.00% | 0.4653 | 92.97% | 7.03% |
| Trajectory KL + EOS | 2 | 0/128 (0%) | 0% | 61/128 (47.66%) | 0.2170 | 77.34% | 22.66% |
| Hybrid | 2 | 0/128 (0%) | 0% | 57/128 (44.53%) | 0.2187 | 78.91% | 21.09% |

## Untouched official test128

| Condition | First-`####` | Strict | Flexible | Mean tokens | EOS rate | Max256 |
|---|---:|---:|---:|---:|---:|---:|
| Llama native | 0/128 (0%) | 0% | 67/128 (52.34%) | 154.13 | 69.53% | 30.47% |
| Qwen native | 0/128 (0%) | 0% | 72/128 (56.25%) | 211.73 | 4.69% | 68.75% |
| Frozen Stage-A | 0/128 (0%) | 0% | 2/128 (1.56%) | 256.00 | 0% | 100% |
| Old Generation Stage-B | 30/128 (23.44%) | 23.44% | 29/128 (22.66%) | 256.00 | 0% | 100% |
| **B1 CE + EOS** | **29/128 (22.66%)** | **22.66%** | 33/128 (25.78%) | **102.28** | **94.53%** | **5.47%** |
| B2 Trajectory KL + EOS | 0/128 (0%) | 0% | 50/128 (39.06%) | 166.65 | 68.75% | 31.25% |
| B3 Hybrid | 0/128 (0%) | 0% | 51/128 (39.84%) | 155.61 | 76.56% | 23.44% |

The objectives produced distinct behavior. CE+EOS preserved explicit GSM8K
answer formatting and almost eliminated max-length degeneration. KL and Hybrid
substantially reduced trajectory KL and improved the official flexible
extraction score, but did not emit parseable `####` answers, so their strict
and first-hash scores were zero. This metric conflict is retained rather than
collapsed into a single headline accuracy.

The repository includes the complete scripts, split manifest, audit output,
all optimizer-step logs, both validation epochs for every objective, checkpoint
selection files, the complete pipeline log, the test128 summary, and all 128
per-sample outputs. Model weights, checkpoints, KV caches, and Python caches are
intentionally excluded.
