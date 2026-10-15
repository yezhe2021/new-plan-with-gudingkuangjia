# HeteroFold KV PCA analysis

Equal-prompt weighted, per-layer/head centered PCA. K is post-k_norm/pre-RoPE; K_raw is supplementary.
Descriptive 32-example/task analysis, not a benchmark or causal intervention. No new training/generation.

| Task | Role | Condition | Mean n90 | Mean n95 | Mean n99 | Effective rank | Native variance coverage@32/@64 |
|---|---|---|---:|---:|---:|---:|---|
| arc | K | native | 42.34 | 59.19 | 89.49 | 32.36 | 0.8526/0.9546 |
| arc | K | recolor | 42.23 | 59.03 | 89.19 | 32.38 | 0.7940/0.9212 |
| arc | K | heterofold | 40.51 | 57.32 | 87.84 | 30.50 | 0.7846/0.9168 |
| arc | V | native | 76.72 | 96.14 | 119.41 | 66.70 | 0.6786/0.8522 |
| arc | V | recolor | 79.26 | 98.07 | 120.17 | 71.11 | 0.5421/0.7395 |
| arc | V | heterofold | 80.53 | 99.06 | 120.48 | 73.62 | 0.5243/0.7247 |
| arc | K_raw | native | 44.42 | 64.62 | 100.08 | 31.54 | 0.8481/0.9452 |
| arc | K_raw | recolor | 45.59 | 65.86 | 100.99 | 32.68 | 0.7919/0.9061 |
| arc | K_raw | heterofold | 44.33 | 64.66 | 100.40 | 31.57 | 0.7834/0.9015 |
| gsm8k | K | native | 40.97 | 57.76 | 88.51 | 31.47 | 0.8590/0.9570 |
| gsm8k | K | recolor | 39.89 | 56.59 | 87.50 | 31.31 | 0.8011/0.9239 |
| gsm8k | K | heterofold | 38.01 | 54.57 | 85.94 | 29.41 | 0.7916/0.9195 |
| gsm8k | V | native | 73.89 | 93.69 | 118.36 | 65.46 | 0.6915/0.8625 |
| gsm8k | V | recolor | 74.36 | 94.14 | 118.68 | 66.87 | 0.5615/0.7555 |
| gsm8k | V | heterofold | 75.36 | 94.94 | 118.99 | 68.85 | 0.5419/0.7401 |
| gsm8k | K_raw | native | 43.01 | 63.03 | 98.97 | 31.51 | 0.8533/0.9480 |
| gsm8k | K_raw | recolor | 43.09 | 62.93 | 98.99 | 32.29 | 0.7961/0.9088 |
| gsm8k | K_raw | heterofold | 41.89 | 61.74 | 98.36 | 31.15 | 0.7876/0.9041 |
| qasper | K | native | 47.28 | 63.68 | 92.48 | 40.56 | 0.8263/0.9411 |
| qasper | K | recolor | 48.09 | 64.39 | 92.59 | 41.16 | 0.8080/0.9299 |
| qasper | K | heterofold | 47.02 | 63.43 | 91.83 | 39.24 | 0.8004/0.9263 |
| qasper | V | native | 91.32 | 106.93 | 122.91 | 89.21 | 0.5692/0.7799 |
| qasper | V | recolor | 91.49 | 106.98 | 122.93 | 89.39 | 0.5192/0.7321 |
| qasper | V | heterofold | 92.05 | 107.39 | 123.06 | 91.25 | 0.5150/0.7284 |
| qasper | K_raw | native | 52.79 | 73.18 | 105.00 | 41.23 | 0.8148/0.9272 |
| qasper | K_raw | recolor | 53.40 | 73.62 | 105.30 | 41.46 | 0.7969/0.9137 |
| qasper | K_raw | heterofold | 52.39 | 72.79 | 104.95 | 40.50 | 0.7887/0.9093 |
