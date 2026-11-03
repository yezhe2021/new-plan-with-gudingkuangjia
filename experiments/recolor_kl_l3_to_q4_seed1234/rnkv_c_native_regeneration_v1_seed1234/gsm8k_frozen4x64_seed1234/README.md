# Frozen GSM8K evaluation, four methods / 64 questions

This adds a new task to the frozen C, original replacement RN-KV and independently trained Keep-both checkpoints, plus canonical-cache Native Qwen. It changes no existing experiment or weight. The RAW Sender, Recolor, token/layer alignment, cache normalization/RoPE and RN routing all reuse the established implementation.

Use the original GSM8K reasoning/boxed numeric-answer prompt and thinking-disabled chat affixes for all arms. Deterministically select 64 test questions with seed1234, excluding C calibration, original RN training/validation, existing RN/structural tests and the parent's historical32 GSM8K questions. Preserve all prompt tokens. This exclusion is not a claim about foundation-model contamination or every older evaluation in the project.

Primary numeric accuracy requires a parseable explicit final answer or a purely numeric response. A truncated reasoning trace's arbitrary last number is never accepted by the primary scorer. Retain the old fallback-inclusive official scorer as a separately labelled secondary metric. Save complete generated text/token IDs, answer source, EOS, unfinished-at-cap, length, duration, paired bootstrap differences and rescue/harm IDs.

Start with512 new tokens. Predeclared extension: if any arm has at least4 unfinished-at-cap outputs, rerun the identical64 questions for ALL four arms at1024. Report both budgets, without selecting a different cap per method. Stop after1024 and retain a length-limited warning if appropriate; do not automatically tune or retrain.

Before full evaluation, check exact C/Native cached continuation-logit parity, inactive-routing parity, all36 answer masks, unchanged translated cache prefixes, memory/answer positions, finite predictions and frozen parameter versions. The same engine is shared, but only one module's routing context can be active. Native body prefill is allowed for the explicit Native baseline only, with a student leakage counter assertion.

Memory is constructed once before suffix/generation. Subsequent answer tokens have normal Qwen KV. This tests whether initial memory conditioning supports mathematical decoding, not whether the full reasoning chain is stored in32 memory states. C calibration includes Open-R1 math prompts; only RN task supervision is restricted to ARC/HotpotQA.

Server B command:

```sh
/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python -u /hy-tmp/yezhe/异构模型/rnkv_c_native_regeneration_v1_seed1234/gsm8k_frozen4x64_seed1234/run_pipeline.py launch
```

The detached runner holds an exclusive lock, stops on failure, and saves arm-wise progress after every generation. A `controls/pause_requested.json` file pauses at arm boundaries. No upload or ongoing monitoring is scheduled.
