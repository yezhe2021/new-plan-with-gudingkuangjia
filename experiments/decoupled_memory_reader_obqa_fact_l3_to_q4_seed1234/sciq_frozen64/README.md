# Frozen OBQA Reader → SciQ support-only transfer (64)

Evaluation only on server B. No Writer, Reader or backbone training. Keep the
three OBQA validation-selected epoch2/step256 Reader checkpoints unchanged.

SciQ is four-option science QA with a separate `support` field. Fetch the public
`allenai/sciq` official test parquet, pin the repository revision, and preserve
the downloaded file's SHA256. Dataset card/license:
https://huggingface.co/datasets/allenai/sciq (CC BY-NC 3.0).

Select the first 64 eligible official test rows. Exclude empty support,
non-distinct/empty answers, duplicate test questions and exact normalized
question/support overlap with known parent training samples. Do not filter by
Native or Writer performance. This is an exact-overlap audit, not proof against
near-duplicate or pretraining contamination.

Shuffle distractors deterministically, with 16 correct labels at each A/B/C/D
position. Otherwise a fixed correct-answer field position could create a
trivial label shortcut. Record original row index and final option ordering.

Protocol is the same as the parent Reader pilot: only support text enters the
Llama chat; frozen Stage-A+B maps all evidence content tokens with Full-Sync;
only question/options enter the Qwen chat. Use the same ARC-style MCQ formatter,
assistant cue and single-label likelihood scorer; no generation, length cap,
memory compression or loss/parameter changes. This is **not** a claim to run
the unmodified lm-eval SciQ task or to report its leaderboard metric.

Conditions: Question-only Qwen; Native full-support-text Qwen; frozen Stage-A+B
physical-prefix cache; each of the three trained Readers with correct/absent/
wrong support memory. Wrong memory is a deterministic different support, not
guaranteed contradictory or semantically irrelevant. As in the parent,
no-memory bypasses the Reader, so it does not isolate generic residual
calibration from content-dependent reading.

Run two distinct-support smoke examples per arm, checking disabled-Reader
Native parity (no accuracy gate), then formal 64. Sender capture, frozen mapping
and Receiver runs have sequential model residency. No weight transfer between
servers, no other GPU jobs stopped. Per-example scores and predictions, paired
counts, exact rendered prompts/token IDs, source/checkpoint identities and
automatic summary remain in this directory. Final completion appends the
shared server-side `EXPERIMENT_RESULTS.md`.
