# RN-KV v1 — Receiver native KV regeneration

Original frozen C(Attention-R64), epoch4/step800, Llama3.2-3B→Qwen3-4B.32 trainable2560-dimensional embeddings, plus independent Rank16 GELU bottleneck residuals after decoder blocks6/12/18/24.409600 trainable parameters; no Qwen, C, Sender or Recolor parameters train. No attention-output blending or learned mixture.

## Construction and answer are different computations

1. Obtain translated C cache of length L, including the public header. Freeze/detach this source cache.
2. Execute Qwen on32 learned `inputs_embeds`, at positions L..L+31. Memory queries can read the entire translated body and causal memory prefix. Construction is differentiable; only frozen source-capture uses no_grad.
3. Take the new K/V actually written by Qwen at every layer. K has exactly the native k_norm/RoPE convention. A post-block l adapter affects layer l+1 and later; it never manufactures same-layer K/V from post-block output.
4. Preserve original cache tensors and concatenate new K/V. Suffix starts L+32. At answer layers19–30 mask original body, retain Memory; at all other layers mask Memory, retain original body. All layers retain public header and causal suffix/answer tokens. No original positions are overwritten or relocated.

`memory.py` defines parameters, `cache.py` preserves differentiable state, `attention.py` installs construction-only adapter hooks and per-layer causal masks, and `model.py` implements both phases. Activation checkpoint functions rebuild independent caches and re-enter routing contexts during recomputation, avoiding mutable HF-cache and stale-hook-state errors.

## Training

ARC-Challenge official train1024 + HotpotQA distractor train1024. Validation64/task from ARC official dev and a disjoint Hotpot train subset. Exclude original C calibration train/validation and historical parent/dependency tests by normalized questions and available source IDs. No paragraph selection/truncation or outcome-based sampling. Test uses the exact dependency's fixed fresh ARC64+Hotpot64, never training rows. This is supervised adaptation on two tasks, not unseen-task generalization.

4 epochs ×2048 examples /batch8 =1024 optimizer steps. Each batch4 ARC+4 Hotpot. Fresh AdamW lr1e-4, weight decay1e-4, clip1. Gold task CE +0.5 KL(Native||Student), full vocabulary. ARC task CE uses the gold candidate label and identical official answer cue; Hotpot supervises every answer token plus one im_end EOS. Task CE always has the gold prefix. Only KL uses generated prefixes, avoiding gold targets after incompatible wrong continuations.

KL prefix quotas(gold/native/current student per8): epoch1=4/4/0; epoch2=3/3/2; epochs3–4=2/2/4. Inherit64-token rollout cap and max8 KL positions. Native teacher and Student see the same text prefix; Native retains original positions, RN's answer is shifted32 as required by the new memory. No native body KV/hidden states enter construction or student evaluation inputs.

Primary checkpoint is fixed epoch4/step1024.128-row validation each epoch is diagnostic only. Evaluate C/RN-KV/Native on shared prompts, candidate scores,64-token Hotpot decoding, EM/F1 and paired per-question changes. Baselines retain original positions and caches; RN alone inserts32 memory positions as an intrinsic architectural change. No GSM8K, new C training, layer/head mechanism sweep, automatic upload, or resumption of paused jobs.

## Deployment and queue

CPU-only `prepare.py` and `cpu_checks.py` can execute while the previous experiment is running. CPU checks use a tiny real36-layer Qwen3 to verify native projection capture, adapter timing, construction body access, answer exclusion, positions, frozen parameters and checkpoint gradients.

`run_pipeline.py launch` runs detached on server B. It waits for **all** training/validation/evaluation of `c_local_functional_ft_l19_30_seed1234` to succeed, its runner to exit and GPU to be free. Then GPU Smoke→1024-step training→ARC/Hotpot evaluation→summary. GPU Smoke checks only correctness and longest-input memory, with disposable weights. If dependency fails/pauses, the queue stops and records the reason; it does not start on partially finished or failed results. Current and paused experiment scripts are never modified.

Pause with `touch controls/pause_requested.json`. Training saves optimizer/scaler/RNG at a batch boundary before exiting. To resume, remove the request and run `run_pipeline.py run`; do not launch a second queue. Existing completed phases and validation/evaluation samples are reused with manifest checks.

Full Prompt transfer includes ARC questions/options. Successful repair here does not establish a strictly separated private-evidence interface, nor prove regenerated KV numerically closer to native. The endpoint is actual task performance.
