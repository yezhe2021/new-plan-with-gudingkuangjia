# Expanded QASPER data experiment

Authorized: server B only, requested256 official Train complete papers up to4096
tokens, allow single-question papers, fixed previous8-paper Dev. Available-budget
fallback is explicit and reported, not silent. Fresh Stage-A2 epochs and
zero-initialized Residual64 Stage-B4 epochs. Same seed/models/learning rates/loss
and interface as24-paper pilot. Keep original experiment untouched.

Data audit records actual selected papers, question counts and planned optimizer
updates, length buckets, Train paper holdout and identical fixed Dev IDs. No
gold cropping, no paper truncation, no Test gradients/evaluation. New input length
and dataset coverage are changed variables and must be acknowledged in results.

Startup gates: CPU tests, native FP16/FP32 numerical audit, cache reconstruction,
attention replay, native first-token agreement and gradient isolation. Longest
selected paper included in smoke. Tests stop the pipeline on actual failure.
No indefinite monitoring or heartbeat: confirm launch then give log command.

Remote: /hy-tmp/yezhe/异构模型/qasper_evidence_kv_train256_seed1234
Python: /hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python
Logs: logs/prepare.log and logs/pipeline.log.
