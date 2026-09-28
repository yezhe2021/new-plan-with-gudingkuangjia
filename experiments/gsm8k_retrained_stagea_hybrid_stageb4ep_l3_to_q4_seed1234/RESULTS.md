# GSM8K four-epoch Hybrid Stage-B results

## Protocol

- Direction: Llama3.2-3B to Qwen3-4B, Full-Sync K/V translation.
- Data: the same 1024 training and 128 validation samples as the preceding two-epoch experiment; final evaluation uses 128 official GSM8K test samples.
- Stage-A: the GSM8K-trained epoch-2 Full28 MLP checkpoint from the preceding experiment, frozen throughout this run.
- Stage-B: fresh zero-initialized Residual64 Hybrid adapter; trajectory KL + 0.1 answer CE + 0.1 EOS CE. Four complete epochs, effective batch 8, 512 optimizer steps, learning rate 1e-4.
- Checkpoint selection: validation metrics only. The selected epoch-4 checkpoint was evaluated on test128 once.

## Validation progression

| Stage-B epoch | Steps | Flexible accuracy | Trajectory KL | Answer CE | EOS rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 35/128 (27.34%) | 0.31074 | 1.02125 | 79.69% |
| 2 | 256 | 47/128 (36.72%) | 0.22328 | 0.81937 | 88.28% |
| 3 | 384 | 53/128 (41.41%) | 0.19882 | 0.74102 | 84.38% |
| 4 | 512 | 65/128 (50.78%) | 0.18386 | 0.72619 | 85.94% |

Epoch 4 was selected. The Stage-A selection remained epoch 2 / step 256, with K cosine 0.92054 and V cosine 0.71801 on its original validation set.

## Official test128

| Condition | Flexible accuracy | EOS/stop rate | Maximum-token rate | Mean generated tokens |
|---|---:|---:|---:|---:|
| Frozen Stage-A only | 0/128 (0.00%) | 0.00% | 100.00% | 256.00 |
| Four-epoch Hybrid | **67/128 (52.34%)** | 88.28% | 11.72% | 143.80 |
| Previous two-epoch Hybrid | 51/128 (39.84%) | 85.94% | 14.06% | 139.97 |

The four-epoch run gained 16 correct answers out of 128 over the preceding two-epoch run. Its validation accuracy was still rising at epoch 4, so this experiment does not establish that Stage-B has converged. Under the current generation and scoring protocol, strict exact match and first-`####` accuracy were both 0 for all reported conditions; the accuracy comparison above uses the flexible extraction metric.

The implementation, split manifest, full logs, training-step records, all four validation summaries and per-sample files, test summary, and test per-sample generations are included here. Checkpoints and runtime K/V state remain on the server and are not included in GitHub.
