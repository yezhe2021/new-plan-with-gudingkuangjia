# QASPER expanded-data training

Llama3.2-3B to Qwen3-4B, same full-render evidence-only interface and Writer/loss
as the24-paper pilot. This is a data-coverage and length-range expansion, not a
single-variable update-count ablation. Seed1234, fresh Stage-A initialization,
zero-residual Stage-B after freezing selected Stage-A. No old checkpoint resume.

Target256 official Train papers, whole evidence512-4096 tokens, at least1 query.
If fewer qualify, explicitly report/use the available count (user-authorized),
never truncate papers or borrow Dev/Test. Preserve10% Train paper holdout and
cross-split ID/text duplicate checks. Papers with>=3 queries hold out1 query;
one/two-query papers train on their available queries. All annotations retained.
Dev is copied exactly from the24-paper pilot, not resampled:8 papers/25 questions.
Audit records selected lengths, actual queries, expected updates and fixed IDs.

Stage-A2 epochs, paper batch8, lr1e-3, chunk64, clip30, NMSE+cosine unchanged.
Stage-B4 epochs, query batch8, lr1e-4, rank64, clip1, native-query attention KL
and K/V projected output NMSE unchanged. Models frozen, FP16 eager. Native probe
loss is not actual answer supervision. Greedy128/native EOS, official raw F1.
CPU LRU2 papers, no unrestricted KV cache dumping. Select only Dev losses.

Smoke includes7 previous short papers and the longest newly selected paper,
with FP32 native full/split precision audit, exact FP16 cache rebuild logits,
native first-token parity,36-layer attention replay and gradient isolation.
Stop on a real code/protocol/numerical error. No relaxation of those gates.

Run `python -u run_pipeline.py`; launch logs/pipeline.log. Automatically runs
preparation, tests, fresh Stage-A, zero-init Stage-B, fixed Dev and seen-evidence
heldout-query evaluations. All epoch predictions/metrics retained. Final Native,
QueryOnly, old GSM8K A/B and new A/B conditions are compared on identical inputs.
No warm-start retraining control is added in this expanded-data run.
