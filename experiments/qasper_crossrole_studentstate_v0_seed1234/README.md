# Cross-Role Reconstruction / Student-State Calibration — V0

Server B only. Retain frozen QASPER Stage-A512 and complete evidence/messages/token
alignment/post-k_norm pre-RoPE protocol. No native evidence fallback or truncation.

Following the proposal's recommended first round, run A0 original, A1 same-role and
A2 cross-role rank32 residuals (9,437,184 parameters each),128 updates each, LR1e-4,
8paper accumulation, original NMSE+token-vector cosine. Frozen base, same RNG,
same initialization and paper order. Three nearest source depth indices, clipped
with duplicates at boundaries. Separate K/V/head/token branches and zero up weights.

Independently fix original Stage-A512, train fresh rank64 B_native_matched and
B_student for4epochs/316updates, same initialization, data/order, optimizer/loss.
B_student refreshes actual Student query/suffix state every training query and
detaches that state. Counterfactual replaces evidence only; suffix states shared.
Loss=attention KL+joint projected-output NMSE+0.01 Stage-A anchor regularization.
Original Stage-B512-base checkpoint is a report-only external reference: its old
split K/V output loss/no-anchor/Dev-based selection differs from the matched control.
Therefore Student-query causal comparison is against B_native_matched, NOT merely
against the old reference's43.05/30.70% scores.

Use16 complete unused official train papers as internal validation (ID/text disjoint
from baseline training/dev/smoke and official dev/test). Select checkpoint on internal
representation/functional loss only. Original8paper25question Dev is report-only.
No official test evaluation, no adaptive LR/budget decisions based on Dev answers.

Save full generations, QASPER official F1/answerable-only/type strata, retention,
first-token KL, think/EOS rates, reconstruction, matched attention/output diagnostics,
mapping/reuse timings, parameters/peak GPU memory and independent checkpoints.
No disk KV cache; bounded existing CPU LRU. Small residual checkpoints conserve disk.

GPU gates: exact zero-residual identity, self/cross input routing, nonzero finite
up/down gradients after updates, immutable shared memory, actual Student attention
replay, native/Student identity, suffix/query detach, gradient reaches K/V adapter
only, no base-model gradient, and a real miniature A/B optimization step.

This is the explicitly recommended V0, not a completed paper-scale sweep.
A3 joint-input, A4 full-base continuation, B2 mixed probes, B3 differentiable rollout,
combined C branches, external baselines, cross-corpus task evaluation, multiple seeds
and official held-out test remain subsequent gated experiments, not silently claimed
as implemented/running here. No loss or full model redesign is made simultaneously.
