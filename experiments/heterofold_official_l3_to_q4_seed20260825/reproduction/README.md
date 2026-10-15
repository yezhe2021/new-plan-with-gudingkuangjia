# Single-direction HeteroFold model adaptation

Upstream: https://github.com/daniel-eai/Prefill-Free-Multi-Agent-LLMs
Commit: d8018c9a103094511fd58bb6572d1e4b038e071e

Direction: local Llama-3.2-3B-Instruct -> local Qwen3-4B, server B only.
This is a model/hardware adaptation, not an exact reproduction of paper scores.
No prior project Writer weights or project losses are used.

Preserved: official character-end alignment, depth stencil span=1/stride=4,
equal corpus token-mass Recolor, pre-key-normalization target domain, output-aware
rank-16 calibration, folding, official calibration split/digest and evaluators.
Formal calibration: Open-R1/HotpotQA, 1600 train/400 validation, 4 epochs,
AdamW lr=3e-4, wd=1e-4, grad accumulation=8, clip=1, seed=20260825.

Adaptations: genuine llama3 alias with mandatory local path; runtime dtype selected
via HETEROFOLD_DTYPE=float16 for V100. Recolor statistics/fitting and trainable
correction parameters remain FP32. Non-finite gradients cause failure, not silent
continuation. An isolated system-site-packages venv reuses Torch 2.5.1+cu121,
with Transformers 5.9.0 and Accelerate 1.14.0; installed versions are recorded.
Upstream Transformers 5.10.2 imports a Torch FP8 dtype unavailable in Torch 2.5.1;
5.9.0 loads both model implementations without fabricating an FP8 dtype.
This differs from upstream Python 3.12/Torch 2.9.1 environment.
HF_ENDPOINT may select a transport mirror; pinned revisions, benchmark file
hashes and the official calibration prompt digest remain mandatory.

Pipeline: official CPU tests -> official data preparation/digest check -> GPU
native-cache parity and tiny synthetic-interface calibration (diagnostic only) -> full official calibration ->
Qwen Native, Recolor-only, folded HeteroFold ARC/GSM8K/LongBench-Qasper (32 each).
The 32-item results are diagnostics, not full benchmark scores. Official prompt,
stopping, generation limits and metrics are not shortened or replaced.
Completed stages are reused. No automatic retry on OOM or protocol failure.

Entrypoint: bash reproduction/run_pipeline.sh. Logs: logs/pipeline.log.
Environment setup: bash reproduction/setup_environment.sh (requires existing
attnkv packages; does not upgrade that environment).
The pipeline takes an exclusive flock and never kills other GPU jobs.

## Evaluation-only continuation (2026-10-15)

`bash reproduction/run_pipeline.sh evaluate` reuses the saved maps, skips completed
native ARC/GSM8K, completes both translated short-task conditions first, and then
evaluates LongBench Qasper as 32 one-item official CLI shards per condition.
Outputs are summarized with sample-index deduplication. Original logs/results
are retained; calibration is never called by this continuation.

V100 memory adaptation: Qasper opts into the existing Transformers explicit
`repeat_kv` SDPA branch instead of `enable_gqa=True`, which forces a high-memory
math fallback on Torch 2.5/V100. This changes kernel dispatch, not the attention
formula. Sender pre-RoPE KV is staged on CPU at the same FP16 precision before
each translated layer is materialized on GPU. No prompt, token budget, sample,
checkpoint, seed, parser or generation-setting changes are made. This execution
change is recorded in every new result and numerically smoke-tested. ARC and
GSM8K keep the original SDPA dispatch to match the existing native baselines.
The efficient Qasper kernel is mathematically equivalent but not bitwise equal
in FP16. The audit records raw score differences and candidate-distribution KL,
and requires unchanged diagnostic choice predictions; it is not a proof of
identical sampled long-context generations.
