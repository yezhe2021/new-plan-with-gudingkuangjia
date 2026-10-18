# Frozen legacy writers: heterogeneous multi-document pilot

Server B only. Evaluation only; no writer/reader training or checkpoint selection.
Reuse the previous non-lm-eval OpenBookQA Llama->Qwen and Gemma->Qwen
Stage-A plus Residual64 writers, pinned to their old validation selections.
Qwen3-4B is the common receiver and supplies a native-source control.

First pilot: 64 seeded official HotpotQA distractor DEV bridge questions, each
with its two COMPLETE gold supporting paragraphs, in original relative order.
Gold support annotations select documents only. No gold answer, supporting
sentence indices, rationale, or query is given to a sender or writer. Each
document is encoded independently; joint-source controls encode both together.
All methods use the same eligible examples. Overlength or token-boundary
mismatch examples are excluded explicitly before any model answer is observed.
This is a custom gold-evidence mechanism diagnostic, not an official benchmark
score or a proof that arbitrary heterogeneous memories cannot compose.

All evidence tokens are retained. The original token-wise writer classes
require exactly 32 slots per call; dense application uses consecutive 32-slot
chunks and repeated-last-slot padding that is discarded. No parameters or
per-token computation are changed. This extrapolates the OLD 32-anchor MCQ
training protocol to document-only dense memory; domain/interface shift must
be distinguished from composition failure using joint-source controls.

Capture K after native key normalization and before RoPE; never normalize
mapped keys a second time. Reapply Qwen RoPE at contiguous receiver positions.
Retain one native public prefix. Check independent-document token IDs against
the full receiver sequence. No native evidence is inserted into translated
branches. Full-native and independent-native arms are separate controls.

Source models, translators, and receiver load sequentially. No KV tensors are
written to disk; per-group CPU memory is released after evaluation. Existing
experiments, jobs, checkpoints, logs and caches remain intact.

The pipeline runs CPU protocol checks and a real FP32 Qwen cache-reconstruction
gate, then two-example smoke evaluation, then the 64-example pilot. Smoke and
pilot results remain separate. Resume skips already completed per-example JSON
files after checking the immutable config/manifest signature.

Outputs include exact manifests, input/checkpoint/module hashes, generated IDs,
raw and parsed answers, EM/F1, paired differences with bootstrap intervals,
KV reconstruction diagnostics, runtime/memory measurements and failure status.
Code automatically updates the shared EXPERIMENT_RESULTS.md on pilot completion
or failure. Interpretive conclusions are not hard-coded.

Run: `/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u experiment.py pipeline`
Log: `tail -f logs/pipeline.log`
