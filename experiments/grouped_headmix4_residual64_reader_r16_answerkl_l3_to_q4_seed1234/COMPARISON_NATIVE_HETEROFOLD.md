# Native / HeteroFold / grouped A+B+Reader comparison

Accuracy counts on aligned existing evaluation subsets. Local model pair: Llama-3.2-3B-Instruct → Qwen3-4B, FP16.

| Dataset | n | Native | HeteroFold reproduction | A+B | A+B+Reader |
|---|---:|---:|---:|---:|---:|
| OpenBookQA | 128 | 103 | unmeasured | 88 | 92 |
| ARC | 128 | 111 | 73 | 82 | 86 |
| HellaSwag | 64 | 23 | 26 | 22 | 22 |
| GSM8K | 32 | 31 | 17 | 10 | 14 |
| MMLU-Pro | 128 | 52 | unmeasured | 33 | 35 |
| SciQ | 64 | 57 | unmeasured | 56 | 57 |

ARC128 / GSM32 raw dataset hashes match. ARC128 / HellaSwag64 prompts, choices, gold, scoring and native context IDs match. Transfer cache construction differs and training data/budgets differ; not an equal-budget method comparison.

Recolor initialization: ARC81/128; GSM23/32. Calibrated HeteroFold: ARC73/128; GSM17/32. Current combined Reader: ARC86/128; GSM14/32.

Published paper, Llama-3.1-8B → Qwen3-4B, BF16, full benchmark:

| Dataset | Native/TextMas (%) | HeteroFold (%) |
|---|---:|---:|
| ARC-C | 87.80 | 77.05 |
| HellaSwag | 41.54 | 41.74 |
| GSM8K | 90.14 | 70.74 |

[Paper Table 1](https://arxiv.org/html/2609.32259v1#S4.T1). Different sender size and evaluation coverage prevent direct comparison. MMLU is not MMLU-Pro.

Conclusion: incremental gains over A+B are present; current combination does not outperform the paper-method reproduction across all overlapping tasks. Remaining Native gap is large on GSM8K. Existing calibration regression warrants retaining Recolor as a baseline.
