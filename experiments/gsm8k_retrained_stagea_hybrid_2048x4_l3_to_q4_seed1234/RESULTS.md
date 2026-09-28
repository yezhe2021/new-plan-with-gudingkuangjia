# GSM8K Hybrid Stage-B: 2048 samples × 4 epochs

## Protocol

- Llama3.2-3B Sender to Qwen3-4B Receiver, Full-Sync K/V translation.
- The GSM8K-trained Full28 MLP Stage-A epoch-2 checkpoint was reused and frozen.
- Stage-B used a newly initialized Residual64 adapter and the same trajectory KL + 0.1 answer CE + 0.1 EOS CE objective, learning rate 1e-4 and effective batch 8.
- Training used 2048 official GSM8K train questions, including the previous 1024. The previous 128 validation questions stayed excluded from training. Four epochs produced 1024 optimizer steps and 8192 sample exposures.
- Validation ran only after epoch 4. The epoch-4 checkpoint was fixed by protocol and then evaluated on the same 128 official GSM8K test questions.

## Results

| Condition | Flexible accuracy | EOS/stop rate | Max-token rate | Mean generated tokens |
|---|---:|---:|---:|---:|
| Frozen Stage-A only | 0/128 (0.00%) | 0.00% | 100.00% | 256.00 |
| **2048 train × 4 epochs Hybrid** | **73/128 (57.03%)** | 78.91% | 21.09% | 162.36 |
| Prior 1024 train × 4 epochs Hybrid | 67/128 (52.34%) | 88.28% | 11.72% | 143.80 |
| Prior 1024 train × 2 epochs Hybrid | 51/128 (39.84%) | 85.94% | 14.06% | 139.97 |

The final validation flexible accuracy was 69/128 (53.91%), trajectory KL was 0.15357, and answer CE was 0.69630. The 2048-sample run gained 6 correct test answers over the 1024-sample four-epoch run, while generation stopped normally less often. Because both the number of distinct questions and optimizer steps doubled, this experiment does not isolate which change caused the accuracy gain. The test128 set has also been examined across prior iterations, so the comparison is exploratory.

Strict exact match and first-`####` accuracy were 0 for all conditions; the table uses the experiment's flexible numeric extraction metric.

Scripts, configuration, split manifest, full log, all 1024 training-step records, final validation records and all 128 test generations are included. The selected Stage-A and Stage-B checkpoints remain on the server; generated K/V caches were not saved.
