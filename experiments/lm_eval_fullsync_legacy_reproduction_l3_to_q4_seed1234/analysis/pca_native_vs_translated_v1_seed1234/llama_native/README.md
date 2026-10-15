# Llama native KV PCA

Follow-up to the completed lm-eval translator PCA, using exactly the same96
prompt IDs, text, calibration/analysis assignment and held-out Qwen samples.
Runs on server B; no training, checkpoint selection, or raw KV files.

Llama3.2-3B has28 layers,8 KV heads and128 coordinates per head. Each model,
layer and head uses its own covariance basis. Layer/head aggregates do not
assume correspondence between Llama's28 layers and Qwen's36 layers.

Two Llama views separate the source distribution from tokenizer alignment:

- `llama_native`: unique native positions, uniformly sampled without replacement
  from tokens overlapping the original Qwen external-text interval. The same
  sender BOS and full untruncated encoding are retained during capture; BOS is
  excluded from sampling. Boundary-overlap tokens are recorded.
- `llama_aligned`: native Llama vectors at the original saved target positions
  after the exact original Full-Sync copy/drop routing, before any translator.
  Duplicate source slots remain present and are counted.

Report complete held-out spectra and components needed for80/90/95/99% variance,
effective rank, per-layer/head details, native-calibrated2D projections, and
comparisons with the existing Qwen native and translated spectra. Raw native
token points need not be in one-to-one correspondence with Qwen points;
aligned source slots are exactly matched to the previous analysis.

Existing PCA files remain unchanged. All outputs, logs and automatic shared
summary updates for this follow-up live here. Run `python -u analyze_llama.py`.
