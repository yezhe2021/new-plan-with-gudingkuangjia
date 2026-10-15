# HeteroFold Native / Recolor / folded-map KV PCA

Server B, no training and no answer generation. Uses the same first 32 ARC,
GSM8K and LongBench Qasper examples as the completed official-code adaptation.
The experiment root is the parent of `analysis/kv_pca_gpu_seed20260825`.

Reuse the official prompts, Qwen chat prefix, receiver-to-sender alignment,
folded FP16 runtime maps and Qasper head/tail truncation at 30000 tokens.
Sender sees bare prompt, Native receiver sees prefix+prompt. Analyze only body
positions (no prefix, assistant suffix or generated KV). Sample at most 256
nonempty-offset receiver positions uniformly over the full prompt. Do not shorten
the actual forward context. Capture matching sender positions with the official
character-end alignment, including its carry-forward behavior.

K: main post-k_norm/pre-RoPE, supplementary raw pre-k_norm. V: v_proj output.
Chunked causal prefill (256 tokens) avoids long quadratic attention allocation.
A short full-vs-chunked capture gate records numerical drift before the analysis.
No changes to the existing models, maps, code, evaluation or running jobs.

PCA centers each condition by its own mean. Dataset covariances are a mixture
with equal prompt weights, so long papers cannot dominate. Per-prompt spectra
are not the headline: a short prompt has a token-count-dependent rank ceiling.
Per-layer/head eigenvalues, 90/95/99% component counts, entropy effective rank,
participation rank, centered variance, norm and mean offsets are saved.
Translated axes' coverage of Native variance and Native-subspace overlap at
16/32/64/96 dimensions are computed from the same descriptive analysis corpus.
These are not held-out PCA intervention results or causal importance estimates.

All covariance, eigendecomposition, projection coverage and paired tensor metrics
run on GPU. CPU math threads are one; CPU still tokenizes and writes results.
Only small aggregate statistics are saved, never full KV tensors or PCA bases.
Completed task directories can be reused only under the identical manifest.

Run: `bash run.sh`. Smoke-only: `bash run.sh --smoke`.
