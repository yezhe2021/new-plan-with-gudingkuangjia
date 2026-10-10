# QASPER Evidence-KV Strong Baseline — first small-scale pilot

Llama3.2-3B-Instruct → Qwen3-4B, frozen FP16 eager base models, seed 1234.
This implements the supplied design's final recommended scope: protocol tests
and small-scale Stage-A / Attention Stage-B only. No official Test score is
produced, no new Multi-Query loss, no full HeteroFold reproduction is claimed.

## Protocol and data

Render the complete native chat template with system, evidence user message,
and a separate question user message. Tokenize that FULL rendered text once.
Select the evidence prefix boundary from offsets; prove it is identical across
all questions of the paper. Do NOT concatenate separately tokenized fragments.
System instruction is the prior concise-answer repair; thinking is disabled.
Sender uses its own evidence-only chat wrapper. Full-Sync copy/drop covers
EVERY receiver evidence slot. The public Qwen wrapper is separately prefilled
without evidence. End-of-evidence message/Query/assistant prefix are all processed
natively after the injected memory. Reader receives memory + Query suffix IDs
only. This is interface isolation, not a formal privacy claim.

Official Train only for gradients: reserve 10% of original Train paper IDs,
exclude Train evidence exact duplicates of Dev/Test, then use 24 eligible
complete papers with ≥2 questions and 512–2048 body tokens. The eight shortest
Train papers ≤2048 tokens are used for protocol smoke; eight Dev papers for selection.
The user approved24 after preparation found only24 eligible short Train papers.
There are zero eligible Train papers ≤1024; smoke therefore uses the same2048 cap.
Each training paper with ≥3 questions reserves one Query from Stage-B for a
Seen-Evidence/Held-out-Query diagnostic. No Gold-based evidence cropping.
If budgets cannot be met, preparation fails with counts rather than silently
reducing samples or increasing length limits. All annotations are preserved.

## Training and comparison

Stage-A Full28 MLP, from scratch and old GSM8K initialization as controlled
comparisons: same 24 papers, 2 epochs, AdamW 1e-3, effective paper batch 8,
chunk64, FP16 AMP with FP32 NMSE+cosine losses, clip30. Select lowest Dev
representation loss; report Dev answers each epoch (not used for selection).

Stage-B freezes the scratch Stage-A best checkpoint and trains zero-up-projection
Residual64 for 4 epochs, AdamW 1e-4, accumulation8 Question probes, clip1. Every
training Query except the explicitly held-out Query is used. Question weights
are inverse questions-per-paper, scaled by the corpus mean Question count and
averaged over each Query batch. Thus the full-corpus weighted Query objective
equals mean-of-paper-mean loss. This is not gold supervision.

Teacher Q is native, RMS-normalized then RoPE-rotated. K is already k_norm'ed
pre-RoPE; only RoPE is applied after translation, never a second normalization.
All 36 layers and all actual Question tokens are used. Attention includes the
public prefix, evidence, visible query/suffix tokens with causal masks and GQA.
W_O is applied to concatenated query-head outputs in native head order.
Key loss = route KL + normalized K-induced projected output error with native V.
Value loss = projected output NMSE under STOP-GRADIENT translated attention.
Gradient tests enforce K-only/V-only parameter updates and frozen base models.
Select lowest paper-normalized Dev functional loss, not answer F1/Test.

Reference probes and paper KV live in a bounded two-paper CPU LRU, not unlimited
disk files. Evicted papers are recomputed and counts recorded. Per paper-session,
evidence is captured once and teacher suffix probes cached per Question; final
inference builds one memory per Writer condition then clones it for every Query.
This is a deliberate memory budget compromise, not a claim of once-only prefill
over all training epochs. Full teacher trajectories are not stored.

Final Dev comparisons: Query Only, Native Full/Cache, old GSM8K Stage-A/B,
new Stage-A, new Attention Stage-B, warm-start Stage-A. Also evaluate held-out
Train queries, with all references kept. Official raw Answer F1, answer types,
answerable-only F1, EOS, refusal/abnormal outputs, first-token KL, per-paper
mean/worst F1 and joint correctness at PREDECLARED F1≥0.5. Teacher-probe losses
are reported separately from real student generation: good probe loss does not
guarantee native trajectories. Generation max128, greedy, native EOS.

Run `python -u run_pipeline.py`. Failed protocol/gradient tests stop training;
epoch-complete checkpoints resume without starting from scratch. Raw predictions,
generated IDs and summaries are retained. Timing separates warm cache read/build
costs and does not claim end-to-end speedup when source capture costs are absent.

### Native precision audit (2026-10-10)

The original absolute FP16 full-vs-split logit MAE gate0.02 rejected a native
sample at0.02904 despite identical generation/argmax. An isolated audit across
8 papers/16 questions in each precision confirmed FP32 full-vs-split MAE<1e-4
for every question; reconstructed-vs-native split logits were exactly equal.
The smoke gate now requires that FP32 audit, exact FP16 reconstructed/native
split logits, and matching full/split first-token argmax. The old0.02 threshold
remains a diagnostic warning, not an interface correctness criterion. Full
generation differences are still recorded; no bitwise FP16 full/split generation
equivalence claim is made. This changes the numerical verification criterion,
not training dtype, data, losses, Prompt or inference positions.
