# Layer-group Oracle KV repair: evidence-only Hotpot32

Fixed new Hotpot-only Stage A+B, no Reader or training. Select the first32 of the fixed512 dev rows without looking at predictions, and reserve the next128 IDs without evaluating them.

Transfer evidence only, preserve all original distractor paragraphs; Qwen processes the entire Question and answer suffix natively from scratch under each condition. This differs from the previous whole-prompt transfer interface, so recompute Translated, Native and Reject baselines. The translator was trained on whole prompts; record this interface distribution shift when interpreting results.

Qwen36 layers split into6 fixed contiguous groups of6. Each group gets K-only, V-only, K+V replacement and a single same-location attention-mask arm. Together with Translated/Native/Reject-all:27 arms x32 =864 generation evaluations, all with EM/F1 and answer-token-mean Gold NLL. No NLL-based selection in this round. K/V replacements act on assembled physical caches after RoPE, never rotate again. Preserve the identical native anchor and positions across arms. Reject and mask block transmitted evidence accesses starting at the first Question token, including all answer/NLL continuation steps; do not simulate rejection by zeroing KV.

GPU gates: empty/full replacement tensor and generation identity, parity against the existing generation API, Native cached versus full evidence+question prefill, masked KV content invariance, K/V role isolation, frozen model versions and no gradients. Run first and longest pilot questions before formal evaluation. Per-arm records saved atomically. Output summary.json/RESULTS.md, recovery ratios, paired differences vs translated and same-location masking, strict rescues and harmed IDs. No automatic layer/head search, confirmation evaluation, repair training, monitoring, retries or GitHub upload.

Single-group improvement need not identify erroneous states, and failure does not exclude multi-group interactions. Oracle improvement does not prove deployable repair predictability. Existing Stage B is already a residual adapter; query-aware repair must demonstrate additional functional value.

Launch on server B: `/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python run_pipeline.py launch`

Log: `tail -f logs/pipeline.log`
