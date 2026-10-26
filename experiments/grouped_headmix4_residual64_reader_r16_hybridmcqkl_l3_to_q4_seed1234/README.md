# Hybrid Reader and frozen-cache diagnostic

Runs on server B after `attention_mixed_precision_senderkv_2wiki128_l3_to_q4_frozen_klreader_seed1234` completes. The detached pipeline waits for dependency completion and24GiB free GPU memory, then performs GPU smoke, the diagnostic, fresh Reader training and six-task evaluation. A dependency failure stops the queue for review. No other process is stopped, no automatic upload is performed, and the client may disconnect.

## Experiment

Reuse the exact completed Llama→Qwen grouped/head-mixing Stage-A and rank64 residual Stage-B weights, both frozen. Train a fresh zero-up rank16/alpha16 Q/O Reader on all36 Qwen layers:7,667,712 trainable parameters. No A/B retraining and no initialization from either old Reader.

MCQ: correct-choice cross-entropy plus0.1 native-to-student candidate-distribution KL at the answer-label boundary. GSM8K: restore the old Reader's exact full-vocabulary native-to-student KL with coefficient1, max64 generated prefix tokens and up to8 sampled positions; epoch1 native prefixes, epochs2–4 balanced native and current-student prefixes. Teacher scores exactly the same history. No gold solution CE, no reference-answer teacher forcing and no TBPTT for GSM. These settings intentionally restore the known baseline before testing longer trajectories.

Reuse four-task512each train/64each validation splits, balanced order,4epochs, batch8,1024updates, AdamW lr1e-4,weight_decay0,clip30,seed1234. Final epoch4 checkpoint is fixed before running. Validation loss and16-question GSM free generation are recorded every epoch; neither diagnostic nor validation automatically selects a test checkpoint. No new experiment runs on server A.

## Diagnostic

On the same fixed first16 GSM validation examples, evaluate old KL Reader and the completed task-supervised Reader with native Qwen KV and frozen A+B translated KV. Also evaluate native KV with Reader disabled as the reference, for five arms total. Existing Reader checkpoints are frozen. Native body/anchor prefill always has Reader disabled; Reader is enabled only for the suffix/continuation, matching the translated interface. This is an interface diagnostic, not a full-prompt LoRA benchmark. It can narrow the failure mode but cannot uniquely identify its cause.

## Scoring and checks

The historical prompt requesting a boxed final answer stays unchanged. GSM training follows native/student trajectories and does not introduce `####` reference-format targets. Strip literal chat-control tokens before the historical numeric extractor, uniformly for all diagnostic, validation and final-test arms. Save the original prediction/correctness alongside cleaned reused baseline values. Unit text and other parsing rules stay unchanged.

CPU tests cover candidate-loss gradients, candidate-order invariance and control-token cleanup. GPU smoke checks exact equality to the original GSM loss on native and student histories, independence from gold answer labels, fresh Reader parity, finite gradients across all72 branches, unchanged native teacher outputs and unchanged A+B weights. Smoke weights are discarded.

Outputs: `reader_cache_diagnostic/`, `runs/reader/`, `validation_generation/`, `evaluation/`, `summary.json`. Final evaluation retains the original six-task544 examples and greedy4096 generation cap, comparing native, frozen A+B, previous KL Reader, task-supervised Reader and the new `hybrid_reader`. Preserve all historical data and checkpoints.

This is an incremental-gain experiment. Restoring the GSM objective, prefix distribution and training coverage together does not isolate their individual effects; shared MCQ updates may still affect GSM. MMLU-Pro remains the custom split and Full-Sync retains native prefill.
