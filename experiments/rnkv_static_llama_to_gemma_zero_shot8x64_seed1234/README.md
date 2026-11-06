# Zero-update Static Memory transfer: Llama → Gemma Sender

Load all learned prompt and four Adapter tensors from the completed Llama Static Memory epoch4/step1536 checkpoint, without changing any weights or fitting on Gemma. Keep Qwen3-4B,32 Memory slots, four Rank16 construction adapters, canonical fixed-header construction, K rephasing and all36-layer Keep-both access unchanged. Switch only the translated answer-body KV to the completed Gemma Recolor+C pipeline.

The source Static Memory checkpoint SHA256 is `e4ab49f860c6a67261cd9c4c71d060937287238953c9b7d8f2801058775ffeb4`. Gemma Recolor and C are pinned to the completed full36 reference, not the old Grouped Stage-A+B translator. Sender remains FP32 and Qwen FP16, as the Gemma baseline.

No optimizer or training code is invoked. All409600 transferred parameters, Receiver, Sender, Recolor and C are frozen. First export canonical Memory KV using Qwen and the public header only, without loading either Sender. In the Gemma engine, assert bitwise equality of the header and all72 canonical K/V tensors, verify full36 masks and unchanged translated body, and reproduce C/Native/Gemma-trained dynamic predictions on one example from each task. Then evaluate512 existing questions (eight tasks64each), preserving prompts/scorers/QA64/GSM512 limits and both GSM numeric metrics. Reuse baseline predictions only after parity; do not select checkpoints on outcomes.

This tests transfer of a generic learned receiver prefix between Sender/translator pipelines. It does not test unseen-task or unseen-training-data generalization: both prior training groups use the same task data. A successful transfer supports a reusable calibration component; failure could reflect translator-specific calibration. Neither result alone establishes factual recovery or absence of information in dynamic Memory. The Gemma-trained dynamic baseline is not a separately Gemma-trained static control.

`run_pipeline.py launch` starts a detached locked queue: prepare → source fixed-KV export → interface/endpoint checks → evaluation → paired bootstrap summaries. No large cache is created; the small derived canonical KV file is excluded from uploads. Existing Gemma caches are read-only, with identical cache-miss computation. Pause requests are honored per example; results resume from saved questions.

Read `runner_status.json`, `status.json` and `logs/pipeline.log`. Final `RESULTS.md`, `summary.json` and `evaluation/<task>/per_sample/` contain full tables, paired differences and predictions.
