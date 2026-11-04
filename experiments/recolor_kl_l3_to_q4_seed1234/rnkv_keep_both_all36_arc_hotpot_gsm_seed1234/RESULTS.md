# All-layer Keep-both trained on ARC + HotpotQA + GSM8K

| Task | Metric | C | Original RN-KV | Old Keep-both | New all-layer3-task | Native |
|---|---|---:|---:|---:|---:|---:|
| arc | accuracy | 67.19% | 64.06% | 67.19% | 70.31% | 87.50% |
| hotpotqa | em | 29.69% | 35.94% | 40.62% | 46.88% | 56.25% |
| hotpotqa | f1 | 46.53% | 49.80% | 56.50% | 61.09% | 68.52% |
| 2wiki | em | 9.38% | 21.88% | 29.69% | 32.81% | 32.81% |
| 2wiki | f1 | 22.05% | 35.32% | 42.91% | 43.95% | 43.45% |
| musique | em | 1.56% | 12.50% | 23.44% | 21.88% | 25.00% |
| musique | f1 | 7.64% | 17.85% | 31.88% | 27.79% | 33.07% |
| sciq | accuracy | 98.44% | 100.00% | 98.44% | 98.44% | 100.00% |
| mmlu_pro | accuracy | 25.00% | 26.56% | 28.12% | 21.88% | 53.12% |
| hellaswag | accuracy | 57.81% | 50.00% | 60.94% | 60.94% | 45.31% |
| gsm8k | strict final numeric | 70.31% | 6.25% | 43.75% | 75.00% | 89.06% |
| gsm8k | legacy numeric | 70.31% | 6.25% | 70.31% | 75.00% | 89.06% |

- Combines adding GSM8K training with all-layer memory access. Does not isolate data vs routing effects; adds512 updates to retain4 epochs as training count grows from2048 to3072.
- ARC/Hotpot train and validation rows are byte-for-byte reused individually. Add1024 GSM8K official-train samples and64 disjoint held-out official-train samples, excluding normalized question overlap with C calibration, RN data and all eight test cohorts. GSM8K test uses the already frozen64 official-test questions.
- Full gold task CE covers math reasoning and final answer, mean over tokens then equally over8 examples. KL retains max64 Native/Student rollout tokens and8 full-vocabulary positions; gold prefixes use full targets as in original RN. Both sides see identical text prefixes.
- Answer layers1-36 retain translated body and native-projected Memory KV. Native evidence body never enters student test. Memory is constructed once; generated answer tokens use normal Qwen KV. Four adapters only act on memory construction after blocks6/12/18/24.
- C Sender/Recolor/Translator and Qwen weights freeze; only409600 memory/adapter parameters train. Frozen Qwen must remain differentiable to inputs. AMP rejects and retries overflowing identical batches without skipping samples.
- Reuse seven64-question fixed evaluation cohorts and original scorers; additionally GSM64 at512. Reuse baseline predictions with provenance hashes. Previous datasets have been inspected; not an untouched confirmation set. Other paused jobs remain paused. No automatic upload or continuous monitoring.
