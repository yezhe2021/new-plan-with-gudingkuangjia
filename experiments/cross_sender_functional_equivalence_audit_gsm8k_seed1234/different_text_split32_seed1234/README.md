# Complementary-text KV composition: frozen GSM8K Writers, 32-question pilot

Server A only. The first 32 questions of the original 128-test-row set, no
selection by correctness. Question text is split at a deterministic boundary
without looking at its answer. Raw A+B must equal the complete original text;
source fragments are whitespace-trimmed and separately wrapped in the SAME
old `Question: {fragment}\n` template. Sender sees only its fragment, never the
other fragment or the gold answer. Reader receives only native token0 + two
memory blocks + Answer:, not the full original question as text.

Text order is always A then B. Four main assignments are L(A)+L(B), G(A)+G(B),
L(A)+G(B), G(A)+L(B). Both source processes encode each fragment independently.
Each fragment is Full-Sync aligned to its Qwen-token grid; all four main arms
have exactly the same body positions and budget, even when A/B lengths differ.
K is post-key-normalization/pre-RoPE; one native Qwen token0; contiguous RoPE
positions are applied once at cache construction; no separators/source IDs.

Four ablations mask A or B in each heterogeneous arm WITHOUT changing cache
length or Answer/generated positions. Masked KV physically stays in cache but
is inaccessible to all receiver suffix/generated tokens. The other block was
independently encoded, so it cannot carry information from the masked block.
Native A/B independently-prefilled Qwen memories provide a split-interface
control; this is not a complete-prompt native baseline. Archived complete-question
L/G/mean50 results on the same32 questions are reported as reused references.

New assertions: lossless boundaries; no numerical string/word split; first32 IDs;
checkpoint/parser hashes; native fragment cache vs full-prefill first-logit
parity; unequal-length block assembly and single anchor; one-time contiguous
RoPE; all-ones masked decoder reproduces the old decoder exactly. Two-question
smoke precedes formal32 evaluation. All raw texts/splits/generations are saved.

The automatic splitter is a mechanical rule, NOT proof that every question
requires both fragments. Pronouns, irrelevant sentences and partial questions
remain possible. `splits.md` exposes all32 for human review. Do not select a
test-optimal split or claim arbitrary multi-document evidence fusion.
If all translated split arms fail, use the native split and full-question
controls to separate input-distribution shift from heterogeneous composition.

Original GSM8K Stage-A+Hybrid2048x4 checkpoints are frozen. No training, loss,
prompt or legacy parser changes. Gemma Sender remains FP32; Llama/Qwen FP16;
unchanged frozen Writer autocast. CPU math threads=1. Resource guards do not
preempt others. Only this experiment's temporary tensors are cleaned after
successful final saving. No monitoring automation.

Run `bash run.sh`; log `tail -f logs/pipeline.log`.
