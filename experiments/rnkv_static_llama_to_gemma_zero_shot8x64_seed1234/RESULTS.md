# Llama-trained Static Memory transferred to Gemma: zero updates

| Task | Metric | Gemma C | Gemma Dynamic Memory | Transferred Static Memory | Native |
|---|---|---:|---:|---:|---:|
| arc | correct | 32.81% | 46.88% | 42.19% | 87.50% |
| hotpotqa | em | 25.00% | 32.81% | 31.25% | 56.25% |
| hotpotqa | f1 | 35.52% | 49.41% | 44.16% | 68.52% |
| 2wiki | em | 6.25% | 25.00% | 18.75% | 32.81% |
| 2wiki | f1 | 21.79% | 38.21% | 30.81% | 43.45% |
| musique | em | 3.12% | 4.69% | 4.69% | 25.00% |
| musique | f1 | 7.24% | 10.79% | 10.78% | 33.07% |
| sciq | correct | 78.12% | 93.75% | 81.25% | 100.00% |
| mmlu_pro | correct | 15.62% | 15.62% | 17.19% | 53.12% |
| hellaswag | correct | 56.25% | 54.69% | 53.12% | 45.31% |
| gsm8k | correct | 70.31% | 70.31% | 70.31% | 89.06% |
| gsm8k | legacy_correct | 70.31% | 70.31% | 70.31% | 89.06% |

- Sender-transfer test with identical Qwen receiver, tokenizer, fixed header,32 memory slots and scoring protocol. Canonical Memory KV must be bitwise identical under the two receiver-loading paths. All static checkpoint parameters are frozen and checked exactly.
- Gemma Recolor+C, Gemma-trained dynamic all36 Memory and Native baselines reuse completed same512-question records only after eight-task endpoint parity. The Gemma dynamic model was trained on the same task data; this does not substitute for a separately Gemma-trained static control.
- Training examples/tasks and test cohorts are shared with earlier runs. This tests transfer across Sender/translator, not unseen-task or unseen-data generalization.64 examples/task and one checkpoint are exploratory; Native-state/fact recovery are not implied.
- Correct receiver-space K coordinates are retained. Rephase static post-k_norm/post-RoPE K at recipient positions, do not normalize again; V untouched. Translated body always remains readable in all36 layers.
- QA64/GSM512 caps, unchanged labels, two explicit GSM scoring rules. No cache deletion, translator retraining, new large KV cache, other queued jobs or automatic upload.
