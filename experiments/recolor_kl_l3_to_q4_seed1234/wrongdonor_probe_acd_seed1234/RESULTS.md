# Wrong-donor KV probe: A/C/D (ARC128, HellaSwag64, HotpotQA64)

Three sources per 6-layer group: **translated** (no injection), **native** (same-question native KV, oracle repair), **donor** (different-question native KV, same shape and position grid but wrong content).

If donor is far below translated while native sits within noise of translated, the group matters and the translation there is already accurate. If donor is within noise of translated, that group is not content-critical.

## arc (n=128, primary=correct, native=86.72%)

| Method | Group | Layers | translated | native | donor | donor−translated (95% CI) | reading |
|---|---:|---|---:|---:|---:|---|---|
|A|1|1–6|63.28%|61.72%|62.50%|-0.78 [-4.7,+2.3]|layers not content-critical here|
|A|2|7–12|63.28%|65.62%|64.84%|+1.56 [-2.3,+5.5]|layers not content-critical here|
|A|3|13–18|63.28%|64.84%|64.06%|+0.78 [-3.9,+5.5]|layers not content-critical here|
|A|4|19–24|63.28%|73.44%|52.34%|-10.94 [-18.8,-3.1]|translation inaccurate here|
|A|5|25–30|63.28%|79.69%|29.69%|-33.59 [-44.5,-22.7]|translation inaccurate here|
|A|6|31–36|63.28%|63.28%|62.50%|-0.78 [-2.3,+0.0]|layers not content-critical here|
|C|1|1–6|65.62%|65.62%|66.41%|+0.78 [+0.0,+2.3]|layers not content-critical here|
|C|2|7–12|65.62%|64.84%|64.06%|-1.56 [-6.2,+3.1]|layers not content-critical here|
|C|3|13–18|65.62%|65.62%|63.28%|-2.34 [-7.1,+2.3]|layers not content-critical here|
|C|4|19–24|65.62%|75.78%|54.69%|-10.94 [-18.8,-3.1]|translation inaccurate here|
|C|5|25–30|65.62%|80.47%|28.91%|-36.72 [-47.7,-25.8]|translation inaccurate here|
|C|6|31–36|65.62%|65.62%|64.84%|-0.78 [-3.1,+1.6]|layers not content-critical here|
|D|1|1–6|63.28%|63.28%|63.28%|+0.00 [+0.0,+0.0]|layers not content-critical here|
|D|2|7–12|63.28%|62.50%|64.84%|+1.56 [-1.6,+4.7]|layers not content-critical here|
|D|3|13–18|63.28%|64.06%|64.06%|+0.78 [-3.9,+5.5]|layers not content-critical here|
|D|4|19–24|63.28%|71.88%|52.34%|-10.94 [-18.8,-3.9]|translation inaccurate here|
|D|5|25–30|63.28%|78.12%|26.56%|-36.72 [-47.7,-25.8]|translation inaccurate here|
|D|6|31–36|63.28%|67.97%|67.19%|+3.91 [+0.0,+8.6]|layers not content-critical here|

Per-group relative K/V distance to same-question native (arc):

| Method | Group | K rel.dist | V rel.dist |
|---|---:|---:|---:|
|A|1|0.2007|0.6752|
|A|2|0.3886|0.9025|
|A|3|0.5213|0.9598|
|A|4|0.4887|0.8961|
|A|5|0.4795|0.8137|
|A|6|0.4660|0.9088|
|C|1|0.5115|0.6764|
|C|2|0.4372|0.8913|
|C|3|0.5540|0.9120|
|C|4|0.5229|0.8760|
|C|5|0.4978|0.8091|
|C|6|0.4787|0.8959|
|D|1|0.2489|0.6876|
|D|2|0.4083|0.9108|
|D|3|0.5605|0.9656|
|D|4|0.5054|0.8999|
|D|5|0.5070|0.8189|
|D|6|0.5143|0.9721|

ARC first-64 subset (comparable to the 64-question experiments):

- A: 65.62%
- C: 65.62%
- D: 64.06%

## hellaswag (n=64, primary=correct, native=35.94%)

| Method | Group | Layers | translated | native | donor | donor−translated (95% CI) | reading |
|---|---:|---|---:|---:|---:|---|---|
|A|1|1–6|42.19%|40.62%|42.19%|+0.00 [-6.2,+6.2]|layers not content-critical here|
|A|2|7–12|42.19%|39.06%|32.81%|-9.38 [-17.2,-3.1]|content-critical, translation already accurate|
|A|3|13–18|42.19%|39.06%|40.62%|-1.56 [-10.9,+7.8]|layers not content-critical here|
|A|4|19–24|42.19%|43.75%|40.62%|-1.56 [-9.4,+6.2]|layers not content-critical here|
|A|5|25–30|42.19%|43.75%|42.19%|+0.00 [-7.8,+7.8]|layers not content-critical here|
|A|6|31–36|42.19%|45.31%|46.88%|+4.69 [-1.6,+12.5]|layers not content-critical here|
|C|1|1–6|40.62%|42.19%|40.62%|+0.00 [-6.2,+6.2]|layers not content-critical here|
|C|2|7–12|40.62%|40.62%|32.81%|-7.81 [-15.6,-1.6]|content-critical, translation already accurate|
|C|3|13–18|40.62%|42.19%|34.38%|-6.25 [-12.5,-1.6]|content-critical, translation already accurate|
|C|4|19–24|40.62%|39.06%|39.06%|-1.56 [-10.9,+7.8]|layers not content-critical here|
|C|5|25–30|40.62%|39.06%|39.06%|-1.56 [-9.4,+6.2]|layers not content-critical here|
|C|6|31–36|40.62%|40.62%|39.06%|-1.56 [-7.8,+3.1]|layers not content-critical here|
|D|1|1–6|43.75%|43.75%|40.62%|-3.12 [-9.4,+3.1]|layers not content-critical here|
|D|2|7–12|43.75%|43.75%|39.06%|-4.69 [-12.5,+1.6]|layers not content-critical here|
|D|3|13–18|43.75%|45.31%|43.75%|+0.00 [-9.4,+9.4]|layers not content-critical here|
|D|4|19–24|43.75%|46.88%|40.62%|-3.12 [-10.9,+4.7]|layers not content-critical here|
|D|5|25–30|43.75%|43.75%|39.06%|-4.69 [-12.5,+3.1]|layers not content-critical here|
|D|6|31–36|43.75%|45.31%|42.19%|-1.56 [-10.9,+7.8]|layers not content-critical here|

Per-group relative K/V distance to same-question native (hellaswag):

| Method | Group | K rel.dist | V rel.dist |
|---|---:|---:|---:|
|A|1|0.2070|0.6707|
|A|2|0.4006|0.8775|
|A|3|0.5180|0.9799|
|A|4|0.4800|0.8863|
|A|5|0.4568|0.7487|
|A|6|0.4313|0.8256|
|C|1|0.5283|0.6713|
|C|2|0.4430|0.8682|
|C|3|0.5380|0.9199|
|C|4|0.4956|0.8610|
|C|5|0.4629|0.7460|
|C|6|0.4373|0.8145|
|D|1|0.2465|0.6850|
|D|2|0.4198|0.8849|
|D|3|0.5614|0.9848|
|D|4|0.4973|0.8900|
|D|5|0.4885|0.7551|
|D|6|0.4932|0.8844|

## hotpotqa (n=64, primary=em, native=48.44%)

| Method | Group | Layers | translated | native | donor | donor−translated (95% CI) | reading |
|---|---:|---|---:|---:|---:|---|---|
|A|1|1–6|15.62%|15.62%|10.94%|-4.69 [-10.9,+0.0]|layers not content-critical here|
|A|2|7–12|15.62%|25.00%|23.44%|+7.81 [-1.6,+17.2]|layers not content-critical here|
|A|3|13–18|15.62%|14.06%|12.50%|-3.12 [-10.9,+4.7]|layers not content-critical here|
|A|4|19–24|15.62%|37.50%|4.69%|-10.94 [-20.3,-1.6]|translation inaccurate here|
|A|5|25–30|15.62%|17.19%|4.69%|-10.94 [-20.3,-3.1]|content-critical, translation already accurate|
|A|6|31–36|15.62%|14.06%|9.38%|-6.25 [-12.5,-1.6]|content-critical, translation already accurate|
|C|1|1–6|29.69%|26.56%|31.25%|+1.56 [+0.0,+4.7]|layers not content-critical here|
|C|2|7–12|29.69%|31.25%|28.12%|-1.56 [-6.2,+3.1]|layers not content-critical here|
|C|3|13–18|29.69%|28.12%|20.31%|-9.38 [-18.8,-1.6]|content-critical, translation already accurate|
|C|4|19–24|29.69%|32.81%|3.12%|-26.56 [-37.5,-15.6]|content-critical, translation already accurate|
|C|5|25–30|29.69%|26.56%|14.06%|-15.62 [-26.6,-6.2]|content-critical, translation already accurate|
|C|6|31–36|29.69%|29.69%|15.62%|-14.06 [-23.4,-6.2]|content-critical, translation already accurate|
|D|1|1–6|10.94%|6.25%|14.06%|+3.12 [-3.1,+9.4]|layers not content-critical here|
|D|2|7–12|10.94%|10.94%|7.81%|-3.12 [-9.4,+3.1]|layers not content-critical here|
|D|3|13–18|10.94%|21.88%|17.19%|+6.25 [-1.6,+15.6]|layers not content-critical here|
|D|4|19–24|10.94%|35.94%|1.56%|-9.38 [-17.2,-1.6]|translation inaccurate here|
|D|5|25–30|10.94%|20.31%|7.81%|-3.12 [-10.9,+4.7]|layers not content-critical here|
|D|6|31–36|10.94%|6.25%|4.69%|-6.25 [-14.1,+1.6]|layers not content-critical here|

Per-group relative K/V distance to same-question native (hotpotqa):

| Method | Group | K rel.dist | V rel.dist |
|---|---:|---:|---:|
|A|1|0.1836|0.6009|
|A|2|0.3128|0.7856|
|A|3|0.4474|0.8026|
|A|4|0.4623|0.7696|
|A|5|0.4419|0.7075|
|A|6|0.4099|0.7916|
|C|1|0.5228|0.6098|
|C|2|0.3976|0.7822|
|C|3|0.5776|0.7714|
|C|4|0.5896|0.7615|
|C|5|0.5447|0.7150|
|C|6|0.4804|0.7885|
|D|1|0.2289|0.6220|
|D|2|0.3366|0.7959|
|D|3|0.4836|0.8167|
|D|4|0.4784|0.7758|
|D|5|0.4713|0.7151|
|D|6|0.4761|0.8534|

- Donor KV is same-prompt-format native KV from a DIFFERENT question of the same task, chosen by closest body token length with a deterministic id tiebreak; HotpotQA additionally requires a different question string. It carries the wrong content on the same positional grid, resampled nearest-neighbour to the target body length.
- All injections replace body K and V only, after cache assembly, in post-k_norm/post-RoPE coordinates; the shared public header and all other layers stay untouched. No re-normalization or re-ROPE.
- Native KV is offline oracle information. Neither the same-prompt nor the donor arm is deployable; they are controls that separate layer relevance from translation accuracy.
- HellaSwag and HotpotQA rows are exactly the first 64 in immutable parent order, matching the reused 64-question anchors; ARC extends the same order to 128. The ARC first-64 subset is reported separately so it can be checked against the 64-question experiments.
- The reused unmasked and same-prompt-native (g*_kv) results are identity-checked against the oracle manifest and never recomputed for HellaSwag/HotpotQA; ARC is recomputed in full because it extends to 128 questions.
- Donor caches are recomputed per row and discarded; no KV is persisted to disk. Only per-config scores, per-sample records and aggregate distance statistics are kept.
- Greedy decoding and fixed prompts; no task is used for tuning. Non-monotonicity and group non-additivity are expected and reportable.
- Per-group K/V relative distances to the same-prompt native are recorded as a zero-cost side measurement; they are descriptive only and do not by themselves establish a mechanism.
- No GSM8K, no Reader, no training. Pipeline fails on error, preserves per-configuration results, runs detached, no automatic upload or monitoring.
