# Task-supervised Reader, frozen Llama→Qwen Stage A+B

This experiment reuses the exact grouped/head-mixing Stage-A checkpoint and rank64 residual Stage-B checkpoint from the completed forward experiment. It trains only a fresh Q/O LoRA Reader (rank16, alpha16, all36 receiver layers, 7,667,712 parameters). Neither translator stage nor either base LLM is retrained. The previous KL-only Reader is a comparison arm, not the initialization.

## Objective

- MCQ: cross-entropy of the correct option among the original candidate-label logits, plus 0.1 forward KL from native Qwen's candidate-normalized distribution. Labels must be unique single tokens; unsupported encodings fail instead of being silently truncated.
- GSM8K: teacher-forced full reference solution, final numeric answer and EOS. Strip calculator annotations `<<...>>` only; preserve the written solution and `####` answer. Token-mean cross-entropy plus 0.1 token-mean full-vocabulary forward KL from frozen native Qwen, scoring identical reference histories. All tokens are supervised; targets exceeding4096 tokens fail the preparation audit rather than being cut.
- KL temperature1. FP32 losses/master LoRA weights, FP16 frozen Qwen execution. KL weight0.1 is fixed before training, not selected using validation or test results.
- GSM sequences are processed in32-token target blocks. Forward cache context persists; gradients are detached at block boundaries (TBPTT). This is complete token supervision, not full-sequence backpropagation. Each block is weighted by its target count divided by total answer length, then each example receives equal weight.

## Data and training

Reuse exactly the previous four-task train512each, validation64each and fixed six-task544 test examples. Only training/validation GSM answers are read from the original training source, matched by source-document hash, question and ID. No test-answer solution is used for training. Existing prompt, native anchor, token alignment, candidate scoring and evaluation generation cap4096 are preserved.

Four epochs, effective batch8,1024 updates, AdamW lr1e-4, weight_decay0, clip30, seed1234. Use the original balanced order. No online rollouts. Fresh Reader starts from the same initialization as the earlier Reader. Final epoch4 checkpoint is the preselected primary; epoch checkpoints are diagnostic. Validation reports16 fixed examples per task and separates CE/KL/total loss and MCQ accuracy.

## Execution and validation

`python -m unittest -v test_objectives` tests candidate gradients, teacher detachment, reference alignment across block boundaries, and normalization. `python experiment.py prepare` verifies frozen dependencies/splits and builds the lightweight target snapshot. `python run_pipeline.py --launch` starts a detached locked process that runs GPU smoke, training, all six evaluations, then summary. It stops on error and does not automatically retry, upload, or restart old experiments.

GPU smoke verifies zero-initialized Reader output parity, chunked/full causal-forward parity before and after Reader updates, the longest training solution, all72 Q/O branches receiving finite gradients, unchanged native teacher outputs and unchanged A+B tensors. Smoke weights are discarded before fresh training.

The new result is `task_supervised_reader`, compared on identical rows to existing `grouped_residual_reader` (KL-only), `grouped_residual` (A+B) and `qwen_native`. Existing results/checkpoints are read-only. New checkpoints are stored only under this experiment's `runs/reader`.

This changes task supervision, GSM histories and coverage together. It is a first gain experiment, not a single-factor attribution. MMLU-Pro retains the custom split and Full-Sync retains native body prefill; no prefill-free latency claim.
