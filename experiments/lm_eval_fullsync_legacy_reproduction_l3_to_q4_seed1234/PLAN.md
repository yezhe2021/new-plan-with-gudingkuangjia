# lm-eval Full-Sync legacy reproduction (server B)

User-approved order: first reproduce the OLD method under native lm-evaluation-harness task definitions, then inspect results and implement the new separate-versus-shared adapter comparison. Do not silently start with mixed/shared training.

Server: `ssh server-b`. Remote root: `/hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234`. Python: `/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python`.

Models: Llama-3.2-3B-Instruct sender, Qwen3-4B receiver. Frozen original Full28 MLP/diagonal heads, K/V independent, bias=False; zero-initialized rank64 receiver-space residual for Stage B.

Phase 1: independent MCQ and GSM8K pipelines. Each Stage A = 2 true epochs, Stage B = 4 true epochs, effective batch8, clip30, seed1234. MCQ = OpenBookQA train1024 + ARC-Challenge train1024; loss = native-to-translated candidate likelihood distribution KL. GSM8K train2048; Stage B = original trajectory full-vocabulary KL + 0.1 gold CE + 0.1 EOS CE. Validation128 per dataset. Test128 per dataset (explicit pilot subsets, not full published benchmark scores). MMLU-Pro and HellaSwag are evaluation-only. No old checkpoint trained on MMLU official test may initialize this run.

Use installed lm_eval0.4.12 task YAMLs/functions directly. No chat templates, no custom prompts, no answer/parser modifications. GSM8K default5-shot train demonstrations; MCQ default0-shot. MMLU-Pro default5-shot CoT generation, NOT A-J scoring. Use the actual harness context and continuation; keep receiver token0/native last context readout token, translate the intervening full context KV via unchanged Full-Sync copy/drop. Preserve original target position IDs. Native KV reconstruction must reproduce HF likelihoods and greedy generation before training.

Native, native-cache oracle, Stage-A-only and Stage-B results should be saved with official harness metrics and sample outputs. Summarizer automatically records per-dataset accuracy and joint/unique correctness counts. Maintain immutable initial test reports; only validation may drive follow-up hyperparameter changes.

Monitoring explicitly requested: retry failed SSH every5minutes; check running pipeline and logs without duplicate launches. Repair genuine code/protocol faults, resume completed stages/epochs. Do not stop unrelated GPU tasks. After legacy results complete, continue the approved shared-adapter study with identical Stage-A initialization and per-task training exposures; record all decisions separately. Do not alter benchmark protocol to improve scores.

User requested Codex reset card on 2026-10-04 09:00 Asia/Shanghai. Available tools do not provide redemption; remind user to click it, never claim it was used.

Implementation status at creation: inherited modules downloaded into `base_code/`; no training launched yet.

Update 2026-10-04: deployed to server B and launched serialized `run_pipeline.py` (PID2895955, first start01:29Asia/Shanghai). Log: `logs/pipeline.log`. Both independent MCQ/GSM A+B training smokes passed; formal MCQ Stage A began01:33. No old checkpoint initialization. Training pool excludes conservative question-text overlaps (OBQA156 rows, ARC7 rows); official test unchanged. Native direct likelihoods exactly match HFLM; cached oracle has documented FP16 per-token tolerance and agrees on candidate choice. GSM native-cache16-token generation exactly matched HF in smoke. CPU dummy API/schema test succeeded and must NEVER be included as model results. Monitoring automation id `lm-eval`;9am reset reminder id `9-codex`.
