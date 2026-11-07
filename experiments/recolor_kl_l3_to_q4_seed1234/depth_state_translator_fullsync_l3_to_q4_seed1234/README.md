# Full-Body Full-Sync Depth-State Translator v1

Independent Llama3.2-3B → Qwen3-4B translator. No Recolor checkpoint, C checkpoint, Memory Tokens or Adapter weights are loaded into DST. The original receiver readout/scorers and immutable three-task data are reused. Static Memory remains disabled for the first independent-translator comparison.

## Architecture

Each Receiver body token has its own256-dimensional latent state. For Receiver layer l, choose the nearest Sender depth `floor(l*(Ls-1)/(Lr-1)+0.5)`; X_l contains that layer's aligned raw K and V, each width1024. There is no learned token alignment, layer attention, token attention or token-wise autoregression.

```
z_0 = 0
h_l = GELU(Wx_l [RMS(Ks_l); RMS(Vs_l)] + Wprev_l LayerNorm(z_{l-1}))
z_l = z_{l-1} + Wupdate_l h_l / sqrt(36)
K_l = DirectK_l(Ks_l) + StateK_l(z_l)
V_l = DirectV_l(Vs_l) + StateV_l(z_l)
```

Each layer has its own update and readout parameters. `DirectK/DirectV` are full1024×1024 dense maps, mixing all KV heads and preserving a direct path around the latent. K and V jointly influence the recurrent state. Raw source amplitudes are retained by the direct paths. The recurrence is in the translator's own latent coordinates; it does not add KV from different Qwen layers or claim those coordinates are smooth.

This full-width design has118,075,392 trainable parameters, about3% of the Receiver. Width256 describes the latent rather than the total capacity: most parameters are in full-width readouts. V100 throughput and memory must be measured. Every token is independent inside DST, so128-token chunks exactly preserve its depth recurrence and gradients. Chunk size bounds activation memory rather than truncating context.

`DSTConfig(carry_state=False)` resets the latent before each Receiver layer, retaining the same allocated parameter structure and direct readouts. It is an optional independently trained control; removing connections also changes the effective use of recurrent parameters. Do not interpret inference-only resets as a same-budget trained ablation. No second arm is automatically scheduled.

## Full-Sync and coordinates

The original `build_synchronized_units` and Copy/Drop rank rules are preserved in `reference/`. `alignment.py` uses an indexed equivalent of the original unit builder, checked against the original implementation. A synthetic zero-width index0 sentinel adapts its old BOS convention to RAW bodies without adding an actual model token. All Receiver body tokens are mapped, including token0. Many→few takes the rightmost source in each ordered group; few→many repeats source states. Unicode byte boundaries and token offsets are checked; unmappable tokens fail explicitly rather than silently dropping/resampling samples.

The Sender sees only the original RAW body, with `add_special_tokens=False`. Qwen keeps only its fixed native public header. The whole question/evidence/options body is translated; none of its initial content KV is filled from Native Qwen. This is a full-body protocol, different from the old Options-only experiment with native Question KV.

Full-Sync is causal at the shared synchronization-unit boundary. A selected source state can end after a particular target token inside that unit. Prepared rows and the manifest explicitly count this intra-unit lookahead. Do not claim strict per-target-token causality from the unit construction.

Sender features are captured at `k_proj/v_proj` before RoPE. Native distillation targets are Qwen `k_proj/v_proj` outputs from the same public header+body, before k_norm/RoPE. DST predicts this pre-k_norm/pre-RoPE space. Cache construction applies Qwen k_norm and Receiver-position RoPE exactly once per predicted key; it never re-normalizes an already-normalized cache K. Qwen projections remain frozen but differentiable during functional training. Native body access is rejected in student evaluation; Native is used only for supervised targets and explicitly labeled baseline checks.

## Two training stages

Planned configurable data: reuse the existing ARC/HotpotQA/GSM8K splits exactly—1024 train and64 validation per task,3072/192 total. Test eight existing cohorts64 questions each. Existing cohorts were previously inspected. No resampling, task-specific loss tuning or label changes.

1. **Stage1:** four epochs,1536 updates, effective batch8, AdamW3e-4. Full-body normalized K/V MSE, equally averaged across layers/roles/examples. Each example's normalization uses its full target energy; chunk losses are weighted by chunk token count so splitting does not change the objective. All body tokens are supervised on every exposure.
2. **Stage2:** warm-start all DST weights from the fixed final Stage1 checkpoint; fresh optimizer, four epochs,1536 updates, batch8, AdamW1e-4. Train the entire DST with full gold answer CE +0.5 full-vocabulary KL(Native||Student). Full gold rationales are preserved. Native/Student rollout limit64 and max8 KL positions match the earlier protocol. Epoch prefix quotas per8 are4/4/0,3/3/2,2/2/4,2/2/4 gold/native/student. Frozen Native prefixes can be reused with provenance; Student prefixes are generated by the current DST.

Gradient clip1, weight decay1e-4. Rejected AMP batches retry with the same RNG and no skipped examples. Checkpoints contain optimizer/scaler/RNG and accepted-update counts. Resume archives uncommitted step records after the last durable checkpoint. Final epoch4 is predeclared at each stage; validation is diagnostic, not test-based selection. Source/teacher KV are recomputed without a persistent large disk cache. Atomic checkpoint saves respect disk reserve and never delete old experiments.

## Validation and execution

`checks.py cpu` checks exact original sync-unit parity, Unicode mapping, full token coverage, token independence, chunk equivalence, early-to-late latent gradients and the reset control. A real tiny36-layer Qwen verifies Native RAW→cache reconstruction and end-to-end frozen-Qwen gradient parity with/without checkpointing.

`checks.py gpu` runs a Native RAW cache/logit oracle and checks k_norm is applied exactly once. It performs disposable real Stage1 and Stage2 accepted updates on the longest training example of each task, checking frozen models and finite connected gradients. These updates never warm-start formal training.

```
python run_pipeline.py launch checks
python run_pipeline.py launch train
```

Use only the first command for implementation validation. It ends in `ready_for_training`. The second starts the full independent run after checks and the training budget are settled: Stage1 → Stage2 → evaluate both final checkpoints. Run modes have separate launch records and a shared pipeline lock. No other experiments are resumed. Pause requests are honored at batch/example boundaries.

The final evaluator uses the existing prompts, conditional-likelihood scorers, QA64/GSM512 generation limits and both GSM numeric scoring rules. It compares historical Recolor+C, DST Stage1, DST Stage2 and Native on the same questions, records per-question changes and paired bootstrap differences, and validates Native predictions on each task.

## Interpretation limits

Historical Recolor+C differs in alignment, available layer taps and training data/budget. DST-versus-historical-C results compare complete systems, not an isolated effect of depth recurrence. Establish the recurrence effect using separately trained carry/reset controls with identical data and budgets; add a Full-Sync independent mapping control if attributing alignment gains. No novelty claim relative to Dual-Cache or other layer-mixing methods is established by this implementation. Static Memory integration remains a later, separate experiment.
