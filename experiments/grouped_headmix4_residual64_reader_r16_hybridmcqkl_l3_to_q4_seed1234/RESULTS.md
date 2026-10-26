# Hybrid Reader results

Completed2026-10-26 01:56:11 Asia/Shanghai; detached pipeline launched2026-10-25 22:48:06. Diagnostic completed before training;1024 fresh Reader updates completed with frozen A+B unchanged. Primary epoch4 checkpoint SHA256: `b158f8f8f32dacac043347046341143058efe914a531b66650e3e2374d5804b2`.

## Formal test: same544 examples

Counts are correct answers. All544 records, each arm's counts and the80 diagnostic generation records were checked against machine summaries.

| Task | n | Qwen native | A+B | Old KL Reader | Task-supervised Reader | Hybrid Reader |
|---|---:|---:|---:|---:|---:|---:|
| OBQA |128|103|88|92|96|96 (75.00%)|
| ARC-Challenge |128|111|82|86|92|91 (71.09%)|
| HellaSwag |64|23|22|22|23|21 (32.81%)|
| GSM8K |32|31|10|14|1|10 (31.25%)|
| MMLU-Pro custom split |128|52|33|35|40|40 (31.25%)|
| SciQ |64|57|56|57|52|54 (84.38%)|

Versus the old KL Reader: +4,+5,-1,-4,+5,-3 questions respectively. Versus the task-supervised Reader, GSM recovers1→10 and SciQ52→54, while OBQA/MMLU tie, ARC drops1 and Hella drops2. The goal of retaining MCQ gains while restoring old GSM performance is only partly met. Hybrid GSM ties A+B and remains below the old Reader14/32 and native31/32.

GSM paired versus old Reader:9 both correct,1 hybrid-only,5 old-only,17 both wrong. Versus task-supervised Reader:1 both correct,9 hybrid-only,0 task-only,22 both wrong. Hybrid mean generated tokens592.41 versus467.84 for old Reader; cap4096 hits2 versus1. All GSM arms use uniform chat-control-token cleanup before the historical extractor; original reused predictions/correctness remain stored alongside cleaned values.

## Frozen existing Reader diagnostic:16 validation examples

| Cache / Reader | Correct |
|---|---:|
| Native KV, Reader disabled |14/16|
| Native KV, old KL Reader |15/16|
| Translated A+B KV, old KL Reader |5/16|
| Native KV, task-supervised Reader |16/16|
| Translated A+B KV, task-supervised Reader |1/16|

The previous task-supervised Reader performs well on native KV on this subset while failing on translated KV. This argues against a blanket loss of native mathematical ability as the explanation for its collapse and points toward failure under translated-cache conditioning. It does not identify a unique mechanism such as weak KL, reference-prefix dependence, or shared-task interference. Native body is prefetched with Reader disabled; Reader acts only on suffix/continuation. This is a small fixed validation interface diagnostic, separate from the32 formal test items and not a full-prompt LoRA benchmark.

## Hybrid free-generation validation

Same fixed16 GSM validation questions at epoch ends:3/16,4/16,3/16,3/16. The old Reader scores5/16 and the task-supervised Reader1/16 on these same translated-cache questions. Hybrid sampled-position validation KL is0.1870,0.1882,0.1790,0.1852; this does not reliably track free-generation correctness. No validation/test result changed the preselected epoch4 checkpoint.

## Training scope

Only a fresh rank16 Q/O Reader is trained. A+B checkpoints, questions, balanced order,4epochs,batch8,1024updates,lr1e-4,seed1234 are unchanged. MCQ retains gold-choice CE+0.1 candidate KL; GSM restores old native/student mixed-prefix full-vocabulary KL,64-token cap and up to8 sampled positions, no gold-solution CE. Restoring GSM loss/prefixes/coverage together does not isolate their effects; shared MCQ updates may still interfere. No follow-up training was launched after inspecting these results. MMLU-Pro remains a custom split and Full-Sync retains native prefill.
