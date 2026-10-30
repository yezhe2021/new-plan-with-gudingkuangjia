# Prefix Native Sweep: A/C/D,64 per task

Overall preregistered descriptive label: **localized**.

| Method | Native prefix layers | ARC64 | HellaSwag64 | HotpotQA64 EM | HotpotQA64 F1 | Recovery ARC |
|---|---:|---:|---:|---:|---:|---:|
| Qwen native |36|90.62%|35.94%|48.44%|62.57%| — |
|A|0|65.62%|42.19%|15.62%|31.64%|0.00%|
|A|6|62.50%|40.62%|15.62%|30.58%|-12.50%|
|A|12|67.19%|35.94%|23.44%|40.63%|6.25%|
|A|18|67.19%|35.94%|23.44%|38.36%|6.25%|
|A|24|78.12%|35.94%|37.50%|51.58%|50.00%|
|A|30|90.62%|35.94%|46.88%|61.42%|100.00%|
|A|36|90.62%|35.94%|48.44%|62.57%|100.00%|
|C|0|65.62%|40.62%|29.69%|43.86%|0.00%|
|C|6|65.62%|42.19%|26.56%|41.61%|0.00%|
|C|12|67.19%|42.19%|32.81%|46.12%|6.25%|
|C|18|64.06%|40.62%|28.12%|42.01%|-6.25%|
|C|24|76.56%|40.62%|37.50%|50.34%|43.75%|
|C|30|92.19%|35.94%|46.88%|61.42%|106.25%|
|C|36|90.62%|35.94%|48.44%|62.57%|100.00%|
|D|0|64.06%|43.75%|10.94%|29.42%|0.00%|
|D|6|64.06%|43.75%|6.25%|24.32%|0.00%|
|D|12|67.19%|43.75%|7.81%|28.78%|11.76%|
|D|18|67.19%|43.75%|17.19%|37.12%|11.76%|
|D|24|78.12%|43.75%|39.06%|53.97%|52.94%|
|D|30|92.19%|43.75%|46.88%|62.05%|105.88%|
|D|36|90.62%|35.94%|48.44%|62.57%|100.00%|

Conditional marginals, percentage points:

| Method | Newly native layers | ARC Δ | HellaSwag Δ | Hotpot EM Δ | Hotpot F1 Δ |
|---|---|---:|---:|---:|---:|
|A|01–06|-3.12|-1.56|+0.00|-1.07|
|A|07–12|+4.69|-4.69|+7.81|+10.06|
|A|13–18|+0.00|+0.00|+0.00|-2.28|
|A|19–24|+10.94|+0.00|+14.06|+13.22|
|A|25–30|+12.50|+0.00|+9.38|+9.84|
|A|31–36|+0.00|+0.00|+1.56|+1.15|
|C|01–06|+0.00|+1.56|-3.12|-2.25|
|C|07–12|+1.56|+0.00|+6.25|+4.51|
|C|13–18|-3.12|-1.56|-4.69|-4.10|
|C|19–24|+12.50|+0.00|+9.38|+8.33|
|C|25–30|+15.62|-4.69|+9.38|+11.08|
|C|31–36|-1.56|+0.00|+1.56|+1.15|
|D|01–06|+0.00|+0.00|-4.69|-5.10|
|D|07–12|+3.12|+0.00|+1.56|+4.46|
|D|13–18|+0.00|+0.00|+9.38|+8.34|
|D|19–24|+10.94|+0.00|+21.88|+16.85|
|D|25–30|+14.06|+0.00|+7.81|+8.07|
|D|31–36|-1.56|-7.81|+1.56|+0.52|

Preregistered primary labels (accuracy / Hotpot EM):

| Method | Task | Label | Linear R² | Monotone | Native better |
|---|---|---|---:|---|---|
|A|arc|localized|0.8368|False|True|
|A|hellaswag|ambiguous|0.6429|False|False|
|A|hotpotqa|localized|0.9267|True|True|
|C|arc|localized|0.7527|False|True|
|C|hellaswag|ambiguous|0.6532|False|False|
|C|hotpotqa|localized|0.7744|False|True|
|D|arc|localized|0.8464|False|True|
|D|hellaswag|ambiguous|0.3750|False|False|
|D|hotpotqa|localized|0.8484|False|True|

Paired ARC mismatch diagnostics (percentage points):

| Method | prefix24 − isolated G4 V | prefix24 − isolated G4 KV | Lower-prefix × G4 KV interaction |
|---|---:|---:|---:|
|A|+54.69|+3.12|+1.56|
|C|+14.06|+1.56|+3.12|
|D|+57.81|+4.69|+1.56|

- Same 192 questions, same immutable parent order as layergroup_oracle_repair_acd_64_no_gsm_seed1234; reused unmasked and native results are identity-checked, never recomputed.
- Intervention is kv-only: each condition replaces body K and V with native Qwen values for layers 1..k, after cache assembly, in post-k_norm/post-RoPE coordinates on the matching Qwen token/position grid. Public header, positions and suffix are untouched.
- prefix36 is not an independent measurement: it must equal the native baseline exactly and serves as a full-scale endpoint check.
- Native KV is offline oracle information. This measures the potential value of perfect knowledge at a given depth, not a deployable repair.
- Because the sweep is cumulative, group marginals are conditional on everything below already being native. They are NOT the same quantity as the isolated single-group effects in the parent experiment; do not compare the two as if they were.
- Non-monotonicity is possible and reportable. No per-dataset best cut is promoted as a method.
- No GSM8K in this sweep: generation cost and a different protocol. Add only after the curve is read.
- Greedy decoding: 1 question = 1.56pp at n=64. Curve shape is the primary readout; individual point CIs are wide by design.
- Pipeline fails on error, preserves partial results per configuration, no automatic upload or monitoring, does not resume other paused experiments.
- localized/distributed are descriptive labels for cumulative hybrid-cache response curves, not proofs of intrinsic translator-error location, impossibility of distributed repair, or any deployable90% ceiling. No automatic follow-up training/interface changes are launched.
- prefix24 versus isolated G4 V changes both K and lower1-18 layers, so a significant paired difference cannot by itself prove mismatch or show all previous substitutions underestimate recoverability. Report it as requested, alongside prefix24 versus G4 KV and the paired difference-in-differences (prefix24-prefix18)-(G4 KV-unmasked), all using existing results.
- Native higher-layer body caches were computed under a fully native prefill; replacing a lower prefix does not recompute the retained translated upper body caches. The resulting cache can still contain a native/translated boundary. Marginals are conditional interventions, not independent layer-error estimates.
- Native is already below translated on HellaSwag for these baselines. Always report that curve and its fixed-rule descriptive label, but leave recovery ratio undefined for nonpositive native-minus-translated gaps and exclude these curves from the overall recovery consensus.
- Baseline native source JSON schema has id/task/result but no method/condition. Assert the actual schema, source manifest and ID/task, add explicit native metadata only to new copied records; never fabricate an upstream field assertion.
- No native or unmasked answers are regenerated, even during smoke. prefix00 cache/logit identity plus reused baseline score equality anchors the zero endpoint; every newly evaluated prefix36 must equal the reused native result dictionary exactly. Native cached/direct-prefill parity is inherited from the identity-checked prior smoke, not recomputed.
- The previous Transformer experiment remains paused. Results stay local/server until an explicit GitHub upload request. summary.json provides a suggested commit message containing the descriptive decision label for that later upload.
