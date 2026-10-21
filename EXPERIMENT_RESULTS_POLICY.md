# Experiment Results Policy

Effective for all subsequent experiments.

1. Every formal experiment pipeline must automatically produce a machine-generated result summary.
2. All experiment summaries must be accumulated into one shared document, rather than manually summarized in chat or scattered across experiment folders.
3. The shared document must be updated by experiment code after successful final evaluation.
4. Codex should not proactively author, rewrite, or interpret the shared summary by hand. Codex may only verify that the automatic summary step completed and report its location when asked.
5. Each automatically appended entry must contain at least:
   - experiment name and timestamp;
   - immutable configuration and seed;
   - dataset split and sample counts;
   - compared arms and checkpoint-selection rule;
   - primary and diagnostic metrics;
   - completion or failure status;
   - paths or links to raw result files and logs.
6. Failed or incomplete runs must be recorded as failed/incomplete and must not be presented as final results.
7. Checkpoints and caches remain excluded from uploads; the shared document and raw lightweight result files are retained.
8. Hugging Face organization is experiment-centric: one complete research experiment gets one top-level directory under `main`; its phases, controls, and arms are nested as clearly named subdirectories inside that directory. Do not create a new top-level directory for every arm of the same experiment.

Default shared artifact name for future pipelines:

```text
EXPERIMENT_RESULTS.md
```
