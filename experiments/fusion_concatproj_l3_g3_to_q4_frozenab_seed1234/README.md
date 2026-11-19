# Learned concatenation projection: frozen Llama + Gemma A+B → Qwen

Authorized 2026-10-24 after the fixed50% fusion evaluation. Each target Qwen layer and token concatenates the two1024-dimensional K feature vectors and independently the two V feature vectors. Each of72 bias-free linear branches maps2048→1024, allowing all-head mixing within its target layer. No cross-layer or cross-token mixing. Total trainable parameters150,994,944.

Every projection is initialized to `[0.5I,0.5I]`, so training starts from the previous fixed equal-mean KV fusion. Both existing Stage-A+residual-Stage-B frontends and Qwen are frozen. No Reader is attached. No previous Reader weights or whole-translator Stage-B weights are used.

Training exactly follows residual Stage-B's four-task512each mixture and64each validation, question order, full-vocabulary forward KL, native/student mixed-prefix policy, max64 continuation tokens and up to8 supervised positions. Epoch1 uses Native prefixes; epochs2–4 use one Native and one online Student sample per task in each balanced batch8. Teacher always scores the same token histories. Four epochs /1024 updates; AdamW lr1e-4, wd0, clip30, seed1234. Labels and gold CoT do not enter the loss. Primary checkpoint is fixed epoch4/step1024; no test-based selection.

The frozen frontends' paired FP16 `[36,T,8,128]` outputs are regenerated deterministically and stored in this task's dedicated `/dev/shm` folder (roughly69GiB). Server RAM was checked before allocation. Checkpoints, prepared rows, code and logs live on persistent disk. This cache can be regenerated if lost. No historical source files are overwritten or deleted.

CPU tests verify equal-mean initialization, parameter count, locality and gradients. GPU smoke checks compare initial projected-cache logits with the fixed average, require nonzero finite gradients in all72 branches, and check that Native teacher logits remain bitwise unchanged after optimizer updates. Frozen feature builders use inference mode; training loads detached FP16 features, making frontend updates impossible.

The detached runner performs full feature capture, smoke,1024 training updates, and the same six fixed evaluations (544 rows). Initial smoke may use just the first row of each task to validate code before full capture. Failures stop with failure.json; no blind retries or automatic uploads. Results compare both single A+B branches, fixed average, learned projection, Qwen Native and Gemma Native.

This adds a training budget relative to the fixed average. The two old frontend Stage-A mixtures differ. MMLU-Pro remains the existing custom split. Full-Sync accuracy interface still uses Native prefill for teacher/anchors; no prefill-free latency claim.
