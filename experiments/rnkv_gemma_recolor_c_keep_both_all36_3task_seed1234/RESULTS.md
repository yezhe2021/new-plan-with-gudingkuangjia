# Gemma Recolor+C: fresh all36 vs fresh19-30 Keep-both, identical three-task budget

| Task | Metric | Recolor+C | Memory19-30 | Memory1-36 | Qwen Native |
|---|---|---:|---:|---:|---:|
| arc | correct | 32.81% | 43.75% | 46.88% | 87.50% |
| hotpotqa | em | 25.00% | 35.94% | 32.81% | 56.25% |
| hotpotqa | f1 | 35.52% | 50.80% | 49.41% | 68.52% |
| 2wiki | em | 6.25% | 21.88% | 25.00% | 32.81% |
| 2wiki | f1 | 21.79% | 33.70% | 38.21% | 43.45% |
| musique | em | 3.12% | 6.25% | 4.69% | 25.00% |
| musique | f1 | 7.24% | 14.87% | 10.79% | 33.07% |
| sciq | correct | 78.12% | 93.75% | 93.75% | 100.00% |
| mmlu_pro | correct | 15.62% | 17.19% | 15.62% | 53.12% |
| hellaswag | correct | 56.25% | 53.12% | 54.69% | 45.31% |
| gsm8k | correct | 70.31% | 68.75% | 70.31% | 89.06% |
| gsm8k | legacy_correct | 70.31% | 68.75% | 70.31% | 89.06% |

- Only answer-time Memory access scope changes relative to the completed Gemma Recolor+C19-30 training experiment. Fresh all36 training with same32 tokens,4rank16 adapters after6/12/18/24,409600 parameters. Do not fine-tune old Memory weights.
- Freeze EXACT same Gemma Recolor6fb66bc1 and C epoch4/step800 checkpoint99061a28; do not rebuild Recolor or calibrate C. RAW/token/layer/cache interface and pre-k_norm/pre-RoPE coordinates unchanged; exactly one Qwen k_norm and RoPE. Native public header only.
- Gemma sender FP32,Qwen receiver FP16 as reference. Reuse existing frozen-parent base cache read-only; misses run the identical parent compute_base function without new cache writes. No deletion or duplicated large caches. Train-time tensor precision, checkpointing and no CPU gradient offload match the reference.
- Byte-identical ARC/Hotpot/GSM3072train/192val/512test. Same four epochs/1536 accepted updates/batch8/lr1e-4/CE+.5full-vocab KL, same prefix and batch schedule. Frozen native prefixes may be reused only with reference provenance.
- Reference M32 Memory construction must match at same weights, and real Native/C/partial Memory predictions must reproduce pinned reference examples before reuse. CPU checks cover all36 masks/initial states/long-target gradients; disposable GPU checks use full longest samples and consecutive batches before fresh formal initialization.
- All36 model trains with all36 visible and evaluates with all36 visible; comparison to separately trained19-30 reference avoids inference-only routing mismatch. Native/C/partial predictions reuse immutable same-cohort records after parity checks. No checkpoint selection on test outcomes.
- Eight tasks64each;QA64/GSM512 caps and both numeric scores. Preserve original labels including disputed GSM952; no score-driven filtering. Single seed and previously inspected cohorts provide exploratory evidence.
- Detached single-GPU queue, wait through NVML/CUDA access failures; accepted-update checkpoints and same-batch AMP overflow retries. No changes to A jobs, earlier held Grouped Gemma all36 jobs, automatic upload or continual monitoring.
