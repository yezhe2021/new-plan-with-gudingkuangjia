# Native versus translated KV PCA

Independent, frozen analysis of the **lm-eval retrained** Llama-to-Qwen
translators, not the non-lm-eval writers used in the composition experiment.
Runs on server B beside the original lm-eval experiment and uses its exact
sample-log prompts, Full-Sync alignment, original model arithmetic and
validation-selected checkpoints. No training or new checkpoint selection.

For each of OpenBookQA, ARC-Challenge and GSM8K, deterministically select eight
PCA calibration prompts and 24 disjoint analysis prompts from the original
128 official evaluation documents. GSM8K filters do not duplicate documents.
MCQ and GSM8K weights and spaces remain separate. No answer-based filtering.
Contexts with at most two Qwen tokens have no translated positions and are
excluded before seeded sampling; their IDs are recorded in the manifest.
Up to32 uniformly sampled external positions per prompt feed the geometric
analysis. Long prompts are not truncated; all external positions are translated
and used in receiver probes. Position0 and the final native readout are excluded
from geometric comparison.

Per Qwen layer, KV head and K/V, fit PCA to native calibration vectors only.
Project native, Stage-A and Stage-A+Stage-B into those same axes and center.
Report PC1+PC2 calibration explained variance, paired NMSE/cosine, centered
variance ratio, normalized mean shift, covariance difference, entropy effective
rank, linear CKA and energy outside the native90%-variance subspace. Two-dimensional
overlap alone is not treated as evidence of compatibility or information loss.

Controlled readout diagnostics hold the final native Q and self K/V fixed, apply
original RoPE positions and GQA expansion, and measure QK scores, attention KL,
and weighted-V output errors over the complete prefix. These isolate memory
effects, not the complete changed receiver trajectory. Separately, actual cache
injection measures first-token output KL, with native reconstruction as a
numerical floor. Original official lm-eval task scores are joined by document ID
and prompt identity, and labeled as reused results, not newly scored outputs.

The code produces machine-generated summary.json, REPORT.md, per-sample results,
per-layer/head metrics, 2D projection coordinates, PNG/PDF figures, provenance,
status and logs. It automatically updates the shared EXPERIMENT_RESULTS.md.
KV tensors never persist. Resume requires recapture because no raw KV is saved.

Run on B:

```bash
cd /hy-tmp/yezhe/异构模型/lm_eval_fullsync_legacy_reproduction_l3_to_q4_seed1234/analysis/pca_native_vs_translated_v1_seed1234
/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u analyze.py run
```
