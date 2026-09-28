# Full-token Sync Boundary copy/drop V0

This independent Llama3.2-3B → Qwen3-4B pilot uses exactly the balanced
OpenBookQA/ARC-Challenge/MMLU-Pro test manifest and standard prompt from
`multidataset_obqa_arc_mmlupro_train1024_full28_mlp_l3_to_q4_seed1234`.

For each UTF-8 Sync Boundary unit, it emits one Sender pre-RoPE KV per Qwen
Options token. Equal-length units map in order; many→few keeps the rightmost
Sender state in each ordered group; few→many repeats states. Qwen's Question
cache remains native. Receiver RoPE is applied at its compact cache positions.

This first pilot freezes the already-trained Full28 MLP Stage-A and Residual64
Stage-B checkpoints. It compares them to the prior RawAnchor32/64 predictions
on identical test IDs. It does **not** claim the result of full-token retraining.
No generated KV tensors are saved to disk; only JSON audit and evaluation
outputs are written.

## Completed frozen-checkpoint test (384 questions)

| Condition | Correct | Accuracy |
|---|---:|---:|
| RawAnchor32/64 Stage-A | 160 | 41.67% |
| Full-Sync Stage-A | 140 | 36.46% |
| RawAnchor32/64 Stage-B | 204 | 53.13% |
| Full-Sync Stage-B | 222 | 57.81% |
| Qwen full native | 274 | 71.35% |
| Full-Sync native oracle | 275 | 71.61% |

The Stage-B paired comparison has 28 Full-Sync-only correct and 10
RawAnchor-only correct answers (two-sided exact McNemar p=0.0051). The full
Stage-A/Stage-B retraining experiment remains separate and was **not** run in
this pilot. Results are in `runs/study/`.

## Fresh Full-token retraining

`retrain_fullsync.py` and `run_retrain.sh` run a separate, newly initialized
Full28 MLP Stage-A (KV reconstruction) and newly initialized Residual64 Stage-B
(choice KL). They use the same 3,072/384/384 train/validation/test rows, seed,
effective batch 8, Stage-A 4 epochs, Stage-B 2 epochs, learning rates and
gradient clip as the balanced RawAnchor experiment. Stage-A trains on **every**
Qwen Options token per exposure; 64-token chunks only bound activation memory.
Validation chooses checkpoints; test evaluation runs once at the end.

Native Llama/Qwen KVs are recomputed per sample and never stored. Checkpoints,
logs and JSON metrics are saved under `runs/retrain/`.

The fresh run completed 1,536 Stage-A optimizer steps and 768 Stage-B steps.
On the held-out 384 questions, Stage-A scored **170/384 (44.27%)** and
Stage-B scored **236/384 (61.46%)**. The selected checkpoints were Stage-A
epoch 4 and Stage-B epoch 2. Per-dataset accuracy and joint-correct counts are
in `RESULTS_RETRAIN.md`.

The published retraining artifacts are `retrain_fullsync.py`, `run_retrain.sh`,
`runs/retrain/training_config.json`, `runs/retrain/results/summary.json`, and
`runs/retrain/results/per_sample.jsonl`. The same 384 test records are also
split into `runs/retrain/results/by_dataset/{arc_challenge,mmlu_pro,openbookqa}.jsonl`
with 128 records per file. Each record has the example ID, dataset, gold choice
index, Stage-A and Stage-B predictions, native Oracle prediction, selected
Options-token count, and choice logits. This task scores choices rather than
generating free-form text; no answer prose exists to publish. Native KV caches,
model checkpoint `.pt` files, and runtime logs remain on the server.

```bash
cd /home/yezhe/异构模型/fulltoken_sync_copydrop_v0_l3_to_q4_seed1234
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u retrain_fullsync.py stage_a --smoke
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u retrain_fullsync.py stage_b --smoke
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u retrain_fullsync.py evaluate --smoke
bash run_retrain.sh
```

On the server:

```bash
cd /home/yezhe/异构模型/fulltoken_sync_copydrop_v0_l3_to_q4_seed1234
/home/yezhe/data/miniconda3/envs/attnkv/bin/python test_alignment.py
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u fullsync_v0.py audit
nvidia-smi
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u fullsync_v0.py smoke --limit 3
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u fullsync_v0.py evaluate
```
