# Frozen DST Stage2 + existing Static Memory: zero-training pilot

Append the existing Llama→Qwen Static Memory (M32, four Rank16 construction adapters after blocks6/12/18/24,409600 frozen parameters) to the final epoch4/step1536 DST Stage2 body KV. Sender, Receiver, DST, Memory and all adapters are frozen. There is no optimizer, fitting, early-checkpoint substitution or new Memory training. This is a separate evaluation folder; original DST and Memory experiments stay unchanged.

Pinned DST checkpoint: `1d46e308fe1ddbf8297a67210b7f05df0f753842512c6779882477648b7d7cfc`.

Pinned Static Memory checkpoint: `e4ab49f860c6a67261cd9c4c71d060937287238953c9b7d8f2801058775ffeb4`.

The DST body uses its original complete RAW Full-Sync mapping, pre-k_norm/pre-RoPE prediction and exactly-once Qwen cache finalization. Memory is constructed only from the same fixed native public header plus the learned32 prompt vectors and four adapters. Adapters operate only during Memory construction. The append step rephases the already-normalized native Memory K from canonical positions to L..L+31; V and the complete DST body are unchanged. All36 answer layers read body+Memory, with suffix positions starting at L+32.

Checks require bitwise agreement of the public header and all72 canonical Memory tensors with the prior independent Qwen/header-only export, zero k_norm calls during cached-Memory attachment, unchanged body K/V, all36 Keep-both masks, frozen checkpoint parameter equality, and identical DST/Native predictions on one example per task before reusing baselines. Native body prefill is forbidden for the student and allowed only for explicitly labeled Native parity. Source features and body KV are computed on demand, with no new large disk cache or duplicated model checkpoint.

Evaluate the original512 questions: ARC, HotpotQA,2Wiki, MuSiQue, SciQ, MMLU-Pro, HellaSwag and GSM8K,64each. Preserve every original semantic row field, scorer, label and generation cap (QA64, GSM512), and report strict/legacy GSM metrics separately. Compare DST+Static to DST alone, historical Recolor+C+the same Static Memory, Recolor+C and Native. Write per-question outputs, paired bootstrap differences and rescue/harm IDs.

`python run_pipeline.py launch` starts a detached locked GPU queue: prepare → interface/endpoint checks → evaluation → summary. Saved question records support resumption; pause requests are honored per example. No training, automatic upload, resumption of unrelated jobs or continuing monitor is scheduled.

This pilot measures the complete added-Memory interface, including the32-position shift. It does not isolate content versus position effects; a position-matched masked-Memory control is deferred until useful. Successful transfer supports reusable calibration on a new translator; failure does not establish that Memory trained specifically on DST would fail. The cohorts have been inspected before and remain exploratory, not an untouched confirmation set.

Read `status.json`, `runner_status.json` and `logs/pipeline.log`. Final results are in `RESULTS.md`, `summary.json` and `evaluation/<task>/per_sample/`.
