# Frozen DST Stage2 + existing Static Memory, zero training

| Task | Metric | Recolor+C | Recolor+C+Static | DST Stage2 | DST+Static | Native |
|---|---|---:|---:|---:|---:|---:|
| arc | correct | 67.19% | 68.75% | 60.94% | 60.94% | 87.50% |
| hotpotqa | em | 29.69% | 43.75% | 35.94% | 40.62% | 56.25% |
| hotpotqa | f1 | 46.53% | 58.16% | 53.47% | 54.00% | 68.52% |
| 2wiki | em | 9.38% | 28.12% | 25.00% | 25.00% | 32.81% |
| 2wiki | f1 | 22.05% | 42.07% | 35.22% | 30.55% | 43.45% |
| musique | em | 1.56% | 20.31% | 17.19% | 14.06% | 25.00% |
| musique | f1 | 7.64% | 28.50% | 23.12% | 22.22% | 33.07% |
| sciq | correct | 98.44% | 100.00% | 96.88% | 96.88% | 100.00% |
| mmlu_pro | correct | 25.00% | 35.94% | 23.44% | 23.44% | 53.12% |
| hellaswag | correct | 57.81% | 60.94% | 50.00% | 50.00% | 45.31% |
| gsm8k | correct | 70.31% | 70.31% | 68.75% | 20.31% | 89.06% |
| gsm8k | legacy_correct | 70.31% | 70.31% | 68.75% | 20.31% | 89.06% |

- No DST/Memory/Adapter/Sender/Receiver optimization, no validation/test checkpoint selection. DST is the predeclared final Stage2 checkpoint, not the epoch2 validation minimum.
- Reuses the same fixed512 test examples, eight64-question tasks, original prompts/scorers, QA64/GSM512 caps and both numeric metrics. Recolor+C+Static and Native results are immutable references after endpoint checks.
- Adding32 Memory slots shifts suffix positions by32. This first experiment estimates the complete Memory interface effect; it does not separately attribute gains to Memory content versus position changes. A position-matched masked-Memory control may follow if useful, rather than expanding this pilot.
- Canonical Memory accepts no sample input. Its K/V must remain bitwise identical to the prior independent Qwen/header-only export before relocation. Translated body remains untouched. Native body prefill is forbidden for the student; Native is used only in labeled parity checks.
- A successful frozen transfer supports usefulness of the existing shared calibration on a new translator. Failure can reflect translator-specific mismatch and does not establish that Memory trained on DST could not help. No new large cache, checkpoint duplication, resumption of paused jobs or automatic upload.
