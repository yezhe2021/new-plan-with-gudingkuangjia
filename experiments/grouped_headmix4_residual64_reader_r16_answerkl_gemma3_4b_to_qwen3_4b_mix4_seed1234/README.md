# Gemma3-4B-IT → Qwen3-4B: grouped translator + residual + Reader

Full-scale follow-up authorized on 2026-10-24. Two intentional research changes relative to the preceding Llama experiment: replace the sender with Gemma3-4B-IT, and broaden Stage-A reconstruction data to the existing four-task mixture. This is not a sender-only ablation.

## Fixed protocol

- Stage A: OpenBookQA, ARC-Challenge, GSM8K, MMLU-Pro, 512 train + 64 validation per task. Same question IDs and exact Qwen frames as the existing Stage-B/Reader split. Inputs contain question/options only: no appended gold answer or CoT. Labels are metadata and never enter reconstruction loss.
- Stage A objective: K/V NMSE + K/V cosine losses over translated slots, 2 epochs, batch 8, 512 updates, AdamW lr 1e-3, weight decay 0, clip 30, seed 1234. Select minimum epoch-end held-out reconstruction loss.
- Four proportional contiguous groups: Gemma boundaries [0,8,17,25,34], Qwen [0,9,18,27,36]. Separate K/V group encoders, hidden1024, independent target-layer outputs, full head mixing inside each group. No cross-group or cross-token mixing. 146,800,640 parameters; the increase from 134,217,728 follows the larger sender layer count.
- Stage B: freeze Stage A, train only rank64 cross-head residuals (9,437,184 parameters), same four-task2048×4, 1024 updates, lr1e-4. Existing full-vocabulary forward KL, max64-token mixed Native/Student prefixes, up to8 positions. Fixed final epoch4/step1024 checkpoint.
- Reader: freeze A+B; fresh Q/O LoRA rank16 alpha16 across36 Qwen layers (7,667,712 parameters), four-task2048×4, 1024 updates, lr1e-4. MCQ answer-boundary full-vocabulary KL; GSM64-token/8-position mixed prefixes. Fixed final epoch4/step1024.
- All six formal subsets: OBQA128, ARC128, HellaSwag64, GSM8K32, MMLU-Pro128, SciQ64. Same scoring and GSM greedy cap4096. Evaluate A+B and A+B+Reader; reuse identity-checked Qwen Native outputs only after exact receiver-input equality checks.
- MMLU-Pro remains the historical custom split from official test, not the untouched official benchmark. Full-Sync interface still uses Native body prefill for the teacher/anchor; no prefill-free latency claim.

## Execution and artifacts

`prepare_data.py` pins historical dependencies and rebuilds only Gemma frames/alignment. `stage_a.py`, `stage_b.py`, `stage_reader.py` preserve the established training functions; see `provenance.json` for origins. `common.py` binds those functions to the new experiment and resolves newly trained A/B checkpoints. No Llama translator/Reader weights are loaded.

`capture_source.py` checks physical Gemma cache parity across local/global layers. Source is captured post-k_norm/pre-RoPE. A separate FP16 audit is recorded in `gemma_fp16_audit.json`; configured computation precision is explicit in config. Stored aligned KV is FP16, with finiteness checks before/after conversion. Qwen targets are generated deterministically on demand during Stage A to fit disk capacity; same extraction and loss as cached targets.

`run_pipeline.py --launch` creates a detached server process, which executes all capture/train/evaluation stages. A failed check stops the pipeline and writes failure.json; it does not silently retry. Each stage has independent checks before training. The launcher survives local computer shutdown.

After A/B/Reader all complete, the runner removes only this experiment's regenerable train/validation source caches before test capture, recording filenames policy/count/bytes in source_cleanup.json. Weights, metrics, prepared rows and all historical artifacts remain. Nothing is uploaded automatically.

Startup validation: 10 CPU tests passed; all34 Gemma physical-cache layers passed parity; all4 training tasks passed Stage-A GPU backward. Gemma FP16 produced nonfinite K/V/logits on4 audited inputs; FP32 computation with finite FP16 storage passed. Detached pipeline PID922517 was verified running with PPID1 and SID922517. Stage-B/Reader real checkpoint smoke tests execute automatically after their prerequisites complete.
