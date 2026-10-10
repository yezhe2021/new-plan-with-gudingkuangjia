# Handoff / current execution state

UPLOAD STATUS (2026-10-10): all small-scale stages and final evaluation finished.
The execution notes below describe the earlier launch/debugging state, not a
pending run. See RESULTS.md and evaluation/summary.json for final metrics.
New Stage-A and Attention Stage-B both score0 F1 on Dev and seen-heldout queries;
all new Writer outputs reach the128-token cap. Native Dev F1=43.05; old GSM8K
Stage-B Dev F1=33.20. Fresh Stage-A2 epochs=6 updates, Stage-B4 epochs=36 updates.
No further training is implied or scheduled. Checkpoints/KV tensors excluded
from this GitHub upload; raw outputs, epoch records and scripts are included.

Requested first-round scope is the supplied design's small Step0/1/2 pilot,
not immediate full-scale training. Server B only. User does not want persistent
supervision: verify launch and provide log command, no heartbeat automation.

Implementation is in this directory: full-render independent evidence/question
chat protocol, Full28 Stage-A (fresh and old-GSM8K warm start), frozen-base
Residual64 Attention Stage-B, bounded CPU paper/probe LRU, gradient/protocol
tests, epoch-resume state, Dev/seen-heldout evaluation and automatic summaries.
Copied original translator/protocol modules live in base_code. CPU syntax tests
and 100 randomized copy/drop mapping comparisons passed. Initial CUDA smoke
passed paper1 including gradient isolation, then stopped at FP16 full/split
native logit MAE0.02904>0.02 (identical generation/argmax). Native-only precision
audit now passed all8 papers/16 questions in both FP16 and FP32: FP32 full/split
MAE<1e-4; split/rebuild logits exactly equal. See README numerical-gate change.
Pipeline resumed with logs appended. All8 original smoke checks and Stage-A/B
training completion must still be verified from actual outputs, not assumed.

Data preparation on B found, under min512/max2048 body tokens and ≥2 queries,
after 10% Train paper-ID holdout: Train24 eligible / official888, Dev9 / official281.
User approved Train24/Dev8, max2048 complete evidence tokens. There are no
eligible Train papers <=1024; disclosed smoke uses the shortest8 papers <=2048,
without truncation. New prepared.json was successfully produced on server B.
Stage-A remains2 epochs (fresh plus old-GSM8K warm-start control), Stage-B4.

Remote directory: /hy-tmp/yezhe/异构模型/qasper_evidence_kv_strong_baseline_seed1234
Python: /hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python
Latest local archive: qasper_strong_pilot_code.tgz in parent workspace.
Latest archive successfully deployed and extracted on server B this turn.
Preparation log: logs/prepare_24.log. CUDA=True and free GPU memory32495MiB
verified immediately before launch. Pipeline launched with nohup using absolute
Python/script paths; logs/pipeline.log. Protocol/gradient tests gate training.
Do not assume smoke or training completed without inspecting actual logs/results.

After budget agreement: update config explicitly, deploy, prepare, check CUDA
and unrelated GPU processes, run tests_cpu and GPU tests through run_pipeline.py.
Full pipeline must stop if protocol/replay/gradient checks fail. Select only Dev
representation/functional losses. Test content was inspected for paper/text
overlap audit only, not evaluated or used for gradients.

HeteroFold-style only: post-k_norm/pre-RoPE keys, native Query probes, GQA and
full W_O projection; no second k_norm, no CE/gold trajectory, no B5 in this pilot.
All Question tokens/all36 layers. Zero K-loss gradient to V branch and zero
V-loss gradient to K branch are mandatory real-CUDA tests, not yet passed.
