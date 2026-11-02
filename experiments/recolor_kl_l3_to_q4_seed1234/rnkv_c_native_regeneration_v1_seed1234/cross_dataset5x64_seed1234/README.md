# Frozen RN-KV: five additional datasets,64 each

Original frozen C and RN-KV final epoch4/step1024; original Qwen Native baseline. No weights train, all32 learned memory vectors/four Rank16 adapters remain enabled; fixed construction masks, replacement layers19–30 and suffix offset32.960 model/question configurations, plus correctness checks reused when they are exact fixed test rows.

## Data and task protocols

- **2WikiMultihopQA**: original dev format; all context passages retained. Same Hotpot short-answer prompt, greedy max64. Canonical-answer EM/F1, not v1.1 alias/joint scoring.
- **MuSiQue-Ans**: answerable-v1.0 dev; all20 supplied paragraphs retained, no support/decomposition labels in input. Same short-answer prompt. Max EM/F1 over canonical answer and provided aliases, including the dataset's empty-answer conventions. `qa_metrics.py` retains the previously audited answer-only implementation.
- **SciQ**: official test parquet, provided support plus question/four answer choices. Per-question seeded shuffle places the correct choice without position bias. Label conditional likelihood with the existing ARC answer cue; no training.
- **MMLU-Pro**: test parquet; near-equal64-question allocation across all source categories. Every provided option is included, not forcibly limited to4 or10. Label likelihood with the same answer cue. No COT/answer field enters the prompt. This stratified pilot is not the official full-test aggregate.
- **HellaSwag**: validation jsonl; original short_context prompt and endings, each ending's sum token log likelihood divided by its own token count. No replacement by label scoring. All3 models use the same candidate-tokenization and likelihood code.

Seed1234. Exclude exact normalized questions and dataset-scoped available source IDs in C calibration, RN train/validation, parent and dependency tests; enforce no duplicates among these320 rows. No source-length, correctness, gold-support or score-based selection. Preserve chosen long examples; fail rather than truncate/resample if context is exceeded. Question-level overlap checks do not establish document/pretraining disjointness or independence from all unrelated historical experiments.

## Correctness and execution

`experiment.py prepare` fixes data and source/code hashes. `run_pipeline.py launch` starts detached CPU preparation→GPU checks→five task evaluations→summary. Before broad evaluation, first and longest rows per task verify actual finite inference and frozen parameters; choice-task first rows compare Native with the existing official cached scorer and direct prefill. Canonical native public header is the established common header. Candidate-probability differences and prediction agreements are logged; no task-specific retuning follows checks.

Only the labeled Native baseline can prefill native body; its state is never an RN/C input. Store per-question results, task summaries, paired bootstrap intervals and rescue/harm IDs. Report each task's own metrics, without averaging unlike metrics into one score. Exact smoke test-row predictions may be reused with provenance, without any outcome-based sample selection. No GSM8K or adapter-off sweep, upload, model retraining, or resumption of paused jobs.

Logs: `logs/pipeline.log`. Pause with `touch controls/pause_requested.json` at a question boundary. Existing per-question predictions resume only under the same manifest. Parent experiment files remain unchanged.
