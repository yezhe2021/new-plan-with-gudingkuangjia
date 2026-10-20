# Evidence-only Decoupled Memory Reader: Reader-only pilot

Server B; Llama3.2-3B-Instruct → frozen MCQ Writer → Qwen3-4B. Starts only
after the preceding HeteroFold Attention-Output diagnostic completes.
This implements the first controlled question in the supplied design, not
the later joint Writer–Reader, new-Sender or compressed-memory experiments.

## Inputs and splits

Sender chat user content is **only `fact1`**, without question, choices or
answer labels. Its frozen MCQ Full28-MLP Stage-A plus rank64 residual Stage-B
maps all evidence content tokens using Full-Sync copy/drop. The memory bank
excludes native template tokens and has no native first-content anchor or
native punctuation tail. That is an intentional new external-bank interface,
not a claim of exactly reproducing historical self-cache Writer results.

Receiver user content is the pinned HeteroFold ARC-style MCQ formatting of
OpenBookQA's question and options, without the fact. Qwen thinking is disabled
and the reference assistant option-label cue is retained. Native full-evidence
text is an explicitly labeled upper/control condition, not a closed-book score.

Official OpenBookQA Additional train/dev/test joined by exact ID, question,
choices and answer key to Main. Train 1024, dev 128, test 128. Filter training
questions **and normalized facts** against all official dev/test, not just the
selected evaluation subset. Test is never used for model/layer selection.
This is gold-supporting-fact evaluation, not retrieval; model pretraining data
cannot be exhaustively audited by these split checks.

## Trained arms (4 epochs / 512 optimizer updates each)

1. `native_query`: frozen native Q projection plus a zero-up rank32 residual.
2. `independent_query`: rank32 factorized Q independently initialized, without
   the native Q base. Same trainable parameter count as arm 1.
3. `old_coupled_reader`: retain the original Gated Reader algorithm with joint
   softmax over native/external cache; train its selected-layer branches only.
   Its parameter count and shifted autoregressive positions differ and are
   recorded, not hidden as a parameter-perfect isolated ablation.

Independent readers have a learned RMS normalization and a frozen native
output projection plus zero-up rank32 output residual. Scalar gates start at
zero. The output path is nonzero, so the gate has a nonzero first-step
gradient; projection gradients are checked after the gate opens. New trainable
parameters/optimizer states are FP32; the frozen Llama/Qwen execution is FP16.

Reader sites: zero-based layers 7, 15, 23, 31, after frozen self-attention
residual and before frozen MLP. This sparse pattern is preregistered, **not**
selected from test or the just-completed Attention-Output diagnostics, and
not claimed optimal. Layer-location search is deferred.

External cross-attention has its own softmax; it never appends external KV to
the independent reader's self-attention cache. Its Q and canonical pre-RoPE K
have no autoregressive RoPE. The native Self-Attention/MLP remain frozen.
The old coupled baseline uses the original physical native-token0 + fact-KV
prefix, rotated once at its actual positions, followed by native question IDs.

Gold single-option **full-vocabulary CE only**; no KL, attention matching,
attention-output loss or gate regularization. Effective batch8, AdamW3e-4,
gradient clip5. Lowest dev CE selects each arm's checkpoint after each epoch.
Model initialization and optimizer are fresh after the three-step smoke.

## Evaluation and limits

Controls: Question-only Qwen, Native full-evidence-text Qwen, frozen Stage-A
physical-prefix Writer, frozen Stage-A+B physical-prefix Writer. Each trained
reader is tested with correct, absent and deterministically wrong fact memory.
Wrong memory always comes from a different normalized fact; it is a diagnostic,
not asserted to be contradictory or guaranteed irrelevant. Save all scores,
predictions, Gold CE, paired correctness and same-evidence/multiple-question
groups that naturally exist in this test subset.

Question-only performance can be high from prior knowledge. A CE reduction
alone is not proof of evidence use, shared Sender compatibility or causation.
No new Stage-A is trained; fact-only input may be out of distribution for the
frozen MCQ Writer. Stage-A-only Reader, native-memory Reader training, joint
Writer–Reader optimization and multiple Sender banks remain future controls.

`run_pipeline.py` runs preparation, frozen source capture/mapping, smoke,
controls, three independent Reader-only training/evaluation arms and automatic
summary in sequence. Duplicate launch is locked; checkpoints resume safely.
All train/dev/test outputs stay under this directory. The final summary is
automatically appended to the shared `EXPERIMENT_RESULTS.md`. No automatic
upload, cache deletion or recurring monitoring is installed.
