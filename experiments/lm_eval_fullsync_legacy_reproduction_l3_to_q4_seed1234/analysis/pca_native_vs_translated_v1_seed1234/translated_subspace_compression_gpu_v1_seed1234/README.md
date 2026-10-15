# Native KV projected through Native / Stage-A / Stage-A+B PCA directions

Server B only. No retraining. Uses the existing independent MCQ and GSM8K Full-Sync
Writers, **not** the HeteroFold maps. The only intervention is the direction basis
used to project/reconstruct native Qwen KV. All arms use the same native calibration
mean. Translated means are used only to center their own PCA fitting data.

Three frozen direction bases x K/V/KV x ranks16/32/64/96 plus native control =37
conditions per sample. Native KV heads remain128-dimensional after reconstruction;
no token is removed. Native token0, final native readout and generated KV are untouched.
K is post-key-normalization/pre-RoPE, matching the existing Writer interface.

Fit Stage-A/B axes only on the original disjoint8 calibration prompts/task and the
same32 positions/prompt. Reuse the exact existing Native bases. First evaluate the
same24 held-out requests/task for OpenBookQA, ARC-C and GSM8K (72 total), then frozen
MCQ directions on HellaSwag24 and frozen GSM8K directions on MMLU-Pro14 (110 total).
The new MMLU-Pro run includes all14 original manifest subjects; the earlier OOD run's
one-sample omission is not silently inherited or used as a comparison score.

PCA covariance accumulation/eigh (FP64), projection/reconstruction (FP32), model
inference (original FP16), geometry and bootstrap resampling run on CUDA. CPU BLAS,
PyTorch intra/inter-op, Arrow and tokenizer threading are limited to1. CPU remains
necessary for file I/O, tokenization, official score parsing and writing summaries.
No complete KV tensors are saved. Estimated new basis footprint is a few hundred MB.

Official installed lm-eval0.4.12 tasks and original request arguments own prompts,
few-shot, stopping, generation limits and filters. All37 conditions are scored anew.
ID/OOD budgets stay as originally logged (GSM8K256 versus MMLU-Pro2048); OOD drops
cannot be attributed solely to domain shift. The PCA calibration subset is disjoint
from held-out requests, but this is a small mechanism pilot, not a full benchmark.

Geometry reports both true per-prompt centered native variance retained by U and
energy about the frozen native calibration mean. Function reports original task
scores, paired drops, score ratios (null if native zero), paired correctness overlap,
and GPU bootstrap95% intervals. Neither variance retained nor score ratio is a
direct measure of semantic information retained.

Entry: `bash run.sh`; `--preflight` performs CPU-limited manifest/request replay;
`--gate-only` fits frozen bases and runs full-rank/native-path gates without formal
evaluation. The normal run fits missing bases, passes gates, evaluates ID then OOD,
and saves each sample/condition incrementally. Resume preserves completed outputs.
No auto-retry or preemption on OOM. Logs: `logs/pipeline.log`; status: `status.json`.
