# Gemma3-4B → Qwen3-4B: two matched three-task Keep-both trainings

User requested sequential training: first Memory access at19-30, then all36 layers. Each arm independently starts from the exact same seed1234 fresh32 Memory Tokens and four zero-up Rank16 adapters; the second never resumes the first.409600 trainable parameters per arm. The translated body remains readable in all36 layers; only answer-time Memory access differs. Construction, positionsL..L+31, suffix offsetL+32 and memory-only adapters after6/12/18/24 remain identical.

## Frozen upstream translator and coordinate convention

Use existing Gemma Grouped Stage-A best and cross-head residual Stage-B step1024, both pinned by SHA256. No old Reader. This is not the Llama Recolor C pipeline and does not support a sender-only controlled comparison against the Llama experiments. Preserve the Gemma chat Full-Sync byte-boundary token alignment and34→36 grouped mapping. Preparation reproduces historical Gemma token grids and mappings on one example of each old task.

Gemma FP16 compute previously produced nonfinite values. Load Sender inFP32, capture native post-k_norm/pre-RoPE keys and unrotated values, align and store transient source tensors inFP16 as the old translator expects. Apply the frozen Stage-A+B inFP16 autocast chunks. Its Qwen keys are already post-k_norm: apply only one Qwen RoPE, never an extra normalization. Build a canonical native Qwen header plus first prompt-token anchor independently; do not prefill the remaining body for student input. Sender/full-prompt transfer includes questions/options, not an evidence-only or prefill-free claim.

GPU smoke compares fresh FP32 captured states with an archived source example, checks longest training prompt+gold per task, four contiguous batches, all-layer visibility/causality, memory gradient flow, preserved body prefixes, native scoring parity and frozen model versions. Qwen and Gemma co-reside if real smoke memory fits. No corpus KV cache is created because disk space is limited; source states are computed on demand and discarded after mapping. Full targets are never truncated or resampled by length.

## Matched data and optimization

Copy the new Llama three-task3072 training rows: ARC1024,HotpotQA1024,GSM8K1024. Some upstream Gemma translator training questions overlap this memory-training split, which is permitted. Held-out validation and test must be disjoint from upstream calibration: redraw ARC validation64 and replace three contaminated MMLU-Pro test questions in the same categories, before either arm trains. Both arms share exactly the resulting train/validation/test rows and batch schedules. No choice is based on correctness or results.

Four epochs, effective batch8,384 updates/epoch,1536 updates/arm,AdamW1e-4,wd1e-4,clip1. Rotate task counts2/3/3 per batch,1024/task/epoch. Full gold TaskCE (math rationale plus final boxed number) +0.5 full-vocabulary KL; same Gold/Native/Student quotas and64 generated prefix/8 KL positions as the preceding three-task experiment. Frozen Qwen remains differentiable to memory inputs; overflow rejects/replays the same batch/RNG, never skips examples. Validation is diagnostic; fixed final epoch4/step1536 is primary.

## Queue and evaluation

`run_pipeline.py launch` runs detached: prepare → partial-layer CPU/GPU checks → partial-layer training → eight-task evaluation → fresh all-layer CPU/GPU checks → all-layer training → eight-task evaluation → paired summary. If any phase fails or a pause flag appears, stop the queue. Only advance after successful completion; no monitoring or client connection is required. Never resume unrelated paused jobs or upload automatically.

Eight tasks64 each: ARC,HotpotQA,2Wiki,MuSiQue,SciQ,MMLU-Pro,HellaSwag,GSM8K. Preserve label/ending-likelihood scoring, QA64 and GSM512 generation caps, both strict-final and legacy numeric extraction scores. Compute shared frozen Gemma Stage-A+B baselines. Reuse canonical Native outputs only for identical inputs after per-task parity checks; recompute the three changed MMLU items. Save per-question predictions and paired differences between independently trained arms. These are previously inspected diagnostic cohorts with explicit replacements, not a pristine first-use benchmark.
