# All-layer Keep-both trained on ARC + HotpotQA + GSM8K

User requested adding GSM8K to training and enabling Keep-both in every layer. This is a new independent experiment. Original C, RN-KV and Keep-both experiments, weights and paused jobs remain untouched.

## Architecture and initialization

Fresh seed1234 initialization of32 learned Memory Tokens and the original4 memory-only Rank16 adapters after complete Qwen blocks6/12/18/24,409600 trainable parameters. Do not load old learned memory weights. Freeze Llama Sender, Recolor, C checkpoint and Qwen parameters. Memory construction reads the translated full-prompt cache, preserving gradients through the frozen Receiver. During answering, every layer1-36 reads both translated body and the32 native-projected Memory KV. No gating, output fusion or physical overwrites. Memory is generated once at positionsL..L+31; suffix beginsL+32; Qwen produces later generated-token caches normally.

## Data and objectives

Reuse exactly the old ARC1024/Hotpot1024 training rows and64/64 validation rows. Deterministically add1024 official-train GSM8K examples plus64 disjoint validation examples from official train. Exclude normalized question overlap with C calibration, original RN data and all eight fixed test cohorts. Preserve the full selected math solution: remove inline calculator annotations, render the official final answer as a boxed number, append EOS. Never truncate targets or select examples by length/results.

Train4 epochs, effective batch8,AdamW1e-4,weight decay1e-4,clip1:384 updates/epoch,1536 total. Rotate task counts2/3/3 across successive batches so each task contributes exactly1024 examples per epoch. Use original Gold/Native/Student prefix quotas4/4/0,3/3/2,2/2/4,2/2/4. Native/Student rollout prefix limit64 and at most8 full-vocabulary KL positions remain unchanged; gold prefixes use complete gold targets as before. Task CE supervises all gold tokens, including complete GSM8K reasoning and final answer; average per token then per example. Total loss=TaskCE+0.5KL(Native||Student). Wrong Student prefixes are never paired with gold TaskCE.

Keep original AMP transaction recovery: replay the same full batch and RNG after nonfinite gradients, without skipped examples/updates. Save initial, first4, every32, epoch and final checkpoints. Resume restores optimizer/scaler/RNG/batch. Diagnostic validation never selects checkpoints; primary fixed final epoch4/step1536.

## Verification and evaluation

Actual tiny36-layer Qwen checks all-layer causal masks, gradients through a full81-token synthetic math target, checkpoint gradient parity, frozen Qwen gradients and exact task-balanced batch schedules. GPU checks longest combined prompt+target for each task and4 consecutive actual training batches before fresh formal initialization. Verify no change to translated body cache prefixes and no native evidence leakage during test.

Evaluate original seven tasks plus the already frozen64 GSM8K questions,64 each (512 samples). Reuse source-hashed old C/RN/Keep-both/Native per-question predictions. Original seven scoring protocols remain unchanged. GSM uses512 greedy new tokens to match the completed shared baseline; report both unchanged strict-final and official fallback-inclusive numeric scores, with raw outputs/caps. No automatic1024 extension; old paused GSM job remains paused.

This combines a new training task, all-layer routing and512 extra updates required to retain4 epochs. Do not claim the result isolates any one of these factors or is equal-update-budget to the old2-task model. Eight test cohorts have been inspected previously and are diagnostic rather than untouched confirmation sets.

## Operation

`run_pipeline.py launch` launches detached prepare → CPU checks → GPU smoke → training → evaluation. Exclusive lock, failure markers and sample/batch pause boundaries are enabled. Write `controls/pause_requested.json` to pause safely. No continuous monitoring or automatic upload.
