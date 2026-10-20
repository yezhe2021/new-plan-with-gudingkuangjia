# Mixed-Prefix Unified Functional Distillation Stage-B

One training run on server B, seed1234. No Reader; no Stage-A retraining.

## Fixed model/interface

Frozen Llama3.2-3B-Instruct -> existing frozen MCQ Full28 MLP Stage-A -> new
zero-up rank64 per-layer/per-head K/V residual -> frozen Qwen3-4B. Only the
9,437,184 Stage-B parameters are optimized. Do not initialize from old Stage-B.
Both models retain their own chat templates; Qwen thinking is disabled. Keep
the existing Full-Sync byte-boundary copy/drop alignment, native chat header
and first-content-token anchor, punctuation-only native tail and native
assistant suffix. Keys are post-k_norm/pre-RoPE before cache assembly; RoPE is
applied exactly once at receiver positions.

## Data and supervision

512 examples each from OpenBookQA, ARC-Challenge, GSM8K, MMLU-Pro =2048.
Four epochs, effective batch8, 256 updates/epoch, exactly1024 updates.
Each batch contains two examples per task, with one fresh order per epoch.

The loss is ONLY full-vocabulary forward KL(P_Native || P_Translated), computed
with FP32 log-softmax at temperature1. Both receivers condition on the EXACT
same Qwen-tokenized prefix and positions. Average valid positions per example,
then average eight examples. No Gold CE, candidate KL, EOS CE, attention loss or
KV reconstruction term. Gold labels are not passed to this loss.

Epoch1 uses Native greedy prefixes. Epochs2-4 use one Native and one online
current-Student greedy prefix per task per batch (exactly50% Student). Sampling
is detached; backpropagation is rerun through the Student cache on the selected
prefix. Native trajectories are cached as token IDs, never full-vocabulary
logits. Training rollouts are capped at64 new tokens, with up to8 UNIQUE valid
prediction positions including first and last. Short trajectories contribute
their actual count; no fabricated repeated states or prefixes beyond EOS.
This short-prefix TRAINING coverage is not a formal generation-limit change.

## Dataset audit / important MMLU-Pro override

MCQ train/dev use audited official splits from the existing Stage-A experiment.
GSM8K uses official train for new train/dev; official test is excluded. Exact
normalized question keys are checked across all selected splits and against
the upstream Stage-A training questions. This is not a semantic/pretraining
contamination guarantee.

MMLU-Pro has no official train split here. This approved experiment explicitly
uses a CUSTOM question-grouped partition of its local official test parquet.
Reserve the historical first128 question groups; draw512 train and64 dev from
other eligible groups using category-balanced round-robin sampling. Every
duplicate normalized question group stays outside the opposite split. All
remaining eligible groups are held out; first128 form the fixed evaluation
subset. Persist exact IDs, source SHA, categories and exclusions. Never report
the original complete official test as unseen after this experiment.

MMLU-Pro is an explicit protocol extension: pinned ARC MCQ wording with actual
A-J option labels and label likelihood. This is NOT official MMLU-Pro5-shot CoT,
NOT lm-eval, and NOT an unchanged HeteroFold-supported task. No gold CoT is used.

## Verification, selection and evaluation

Smoke on one training example per task: zero-up identity; Native full-prefill
versus reused-cache full-vocab KL/top1 parity at multiple prefix positions;
new cache assembler versus the existing Full-Sync assembler; finite nonzero
Stage-B gradients on Native AND online-Student prefixes; no Qwen/Stage-A grads.

AdamW lr1e-4, weight_decay0, gradient_clip30, FP16 V100 execution with FP32 loss
and GradScaler. Save resumable optimizer/scaler checkpoints every64 updates.
The PRIMARY checkpoint is fixed epoch4/step1024 last.pt, not test-selected.
Monitor a fixed16/dev-task Native-prefix set each epoch. best.pt chosen by macro
validation KL is diagnostic only. The separate old MCQ Stage-B control was
selected by its original choice-KL validation criterion; not an identical
checkpoint-selection ablation.

After training, evaluate Qwen Native, frozen old MCQ Stage-B and new Stage-B on
fixed OpenBookQA128, ARC128, HellaSwag64, GSM8K32, custom MMLU-Pro128 and SciQ64.
GSM8K keeps pinned HeteroFold boxed math prompt, thinking-disabled chat, greedy,
4096 new tokens, reference stopping and numeric parser. ARC/OBQA/SciQ score
labels; HellaSwag scores length-normalized continuations, never letter answers.
SciQ supplies question+choices only (no support), unlike the prior fact-Reader
diagnostic. This is a mixed-data AND loss change, not an isolated loss ablation.
Save every prediction/score/full generation plus paired correctness counts.

## Storage and execution

Reuse identity-checked parent aligned Sender caches. Capture only additional
aligned Sender FP16 states, without truncation/quantization; map frozen Stage-A
on demand. No large Native KV/full-logit disk cache. Preserve20GiB disk reserve.
Do not stop other jobs, delete prior caches, lower Gemma precision or migrate
this run to server A. No continuous monitor after confirmed training updates.

Run with the server B attnkv Python:

```bash
python -u run_pipeline.py
```

Pipeline order: prepare/audit -> Sender smoke capture -> multi-prefix smoke ->
remaining Sender capture -> train -> six task evaluations -> automatic summary.
The lock prevents duplicate pipelines. Final machine-generated summary.json and
per-sample outputs are retained; training completion, failures and final results
are automatically appended to the shared ../EXPERIMENT_RESULTS.md.
