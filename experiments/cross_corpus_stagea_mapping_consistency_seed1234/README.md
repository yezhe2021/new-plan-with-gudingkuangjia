# Cross-Corpus Stage-A Mapping Consistency Analysis

Server A only. Frozen official lm-eval legacy MCQ/GSM8K Stage-A best checkpoints,
both epoch2/step512. No Stage-B, training, answer generation or accuracy comparison.

384 held-out samples: OpenBookQA64, ARC-Challenge64, GSM8K128, WikiText103-raw-test128.
Exclude normalized question hashes of BOTH Writers' train and validation sets.
This audits experiment fine-tuning leakage, not pretrained model contamination.
No gold answers or training demonstrations enter prompts. Same neutral wrapper
`Input:\n{body}\n\nAnswer:` for both Writers and every corpus, explicitly a mapping
diagnostic rather than official benchmark prompts. Wiki paragraph prefixes are length
matched to an equal MCQ/GSM mixture; original test paragraph hash retained. MCQ/GSM
questions are not truncated; complete prompts must have16–512 Qwen tokens.
Report length distributions plus32-token-bin stratified and equal-bin-standardized
comparisons because corpus lengths need not be identical.

Reuse exact frozen `kv_backend.pair`, `source_map`, `capture`, `map_base` and translator.
One pair capture/sample is shared across both Writers. Compare only actual translated
positions1..N-2, post-k_norm/pre-RoPE; anchor/readout excluded. No KV persisted.
Native model architecture is checked: Qwen36layers×8KVheads×128dim.
FP16 model inference/autocast; FP64 CPU metric reduction. Global Frobenius cosine,
NMSE and error cosine per sample, then sample macro-average. Layer/head metrics
reduce tokens/dim within each sample, then macro-average. Additional token-vector
cosine is separately named. Zero-norm errors get cosine0 and a diagnostic flag;
that value is not evidence of uncorrelated error. No optimizer or gradients.

CPU analytic metric tests,4-corpus GPU smoke, repeated-map determinism check then
formal analysis. Output summaries, all layer/head matrices, two heatmaps, manifest,
data/checkpoint provenance and per-sample stats. Resume skips completed sample IDs;
flock prevents duplicate pipelines. Do not change config or assets during resumption.
