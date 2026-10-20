# MCQ Writer retraining under the pinned HeteroFold chat evaluation

Server B only. Fresh Full28-MLP Stage-A (2 epochs, 512 updates), then
frozen Stage-A + fresh rank64 residual Stage-B (4 epochs, 1024 updates).
Same existing 1024 OpenBookQA + 1024 ARC-Challenge training documents and
128 official-development documents per task. No GSM8K training.

Use the pinned HeteroFold ARC formatter for both MCQ corpora (OpenBookQA
is explicitly adapted to the ARC document schema); each model uses its own
Chat Template; Qwen thinking disabled. Preserve the preceding 192-example
evaluation's Full-Sync content mapping, native template/first-token anchor,
punctuation-only native tail and exact candidate-scoring cue.

Stage-A uses sum of K/V per-example NMSE and cosine distance, Stage-B
uses native-Qwen candidate-distribution KL only. Both main models frozen.
The differentiable candidate scorer is audited against the pinned reference,
including KV gradient flow. Do not use inference_mode reference wrappers
inside training backward passes.

Fixed FP16 raw source/target KV and teacher scores are captured once in
separate model-residency phases and saved losslessly at native tensor dtype.
Cache roundtrip and native-cache parity gates precede training. Cache size
is estimated from actual prepared token counts and checked against free disk.
No weight or cache deletion, no process preemption, no automatic upload.

Validation-loss checkpoint selection only; official held-out test documents
are never used for training, PCA, early stopping or hyperparameter selection.
Final evaluation is the same ARC128 + HellaSwag64 snapshot as the prior run.
Automatic machine summaries and paired correctness accompany raw outputs.
