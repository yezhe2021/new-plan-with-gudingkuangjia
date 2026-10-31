# KV functional sensitivity: Phase1, A/C/D, ARC64+Hotpot64

Exploratory signal: **C_no_stable_signal_in_this_cohort**.

| Task | Method | Context | K/V L2 ρ | Route KL ρ | Output NMSE ρ | Route−L2 Δρ95%CI |
|---|---|---|---:|---:|---:|---|
|arc|A|gold_answer|-0.051|-0.088|0.243|[-0.24137017251077633, 0.1560140651417132]|
|arc|A|student_prefix|-0.039|0.257|0.141|[0.1235181674039423, 0.4490275896722114]|
|arc|C|gold_answer|-0.132|-0.201|0.163|[-0.2245052133645714, 0.10500451057056152]|
|arc|C|student_prefix|0.150|0.115|0.074|[-0.20674249870509398, 0.1399770776456284]|
|arc|D|gold_answer|-0.048|-0.173|0.221|[-0.29976305256141256, 0.041342059703681926]|
|arc|D|student_prefix|-0.064|0.062|0.023|[-0.10363726435811317, 0.3479983826771804]|
|hotpotqa|A|gold_answer|-0.006|0.116|0.151|[-0.057693910309910314, 0.30465426533682594]|
|hotpotqa|A|student_prefix|-0.147|-0.018|-0.007|[-0.06563899618752628, 0.33497935629581005]|
|hotpotqa|C|gold_answer|0.040|0.092|0.076|[-0.10674531681218392, 0.2184108109448256]|
|hotpotqa|C|student_prefix|-0.062|-0.078|-0.157|[-0.1740736799630671, 0.12959223014233115]|
|hotpotqa|D|gold_answer|-0.007|-0.046|0.146|[-0.2512429276166953, 0.1467393913344897]|
|hotpotqa|D|student_prefix|-0.129|-0.002|0.024|[-0.061563859288848555, 0.3413902869711168]|

Primary table outcome: native-KV repair improvement in target-token NLL under group-native-Q clamp. Gold context uses all gold answer tokens; student context uses the same fixed selected positions in the unmodified student rollout.

Full per-question diagnostics, actual-Q controls, K-only/V-only effects, output interactions, route/head statistics and Oracle free-answer gains are in per_group JSON and analysis/ summaries.

- Stage1 only: frozen existing A/C/D and Llama3.2-3B/Qwen3-4B; same historical ARC64 and HotpotQA64 IDs/order as the oracle experiment. No new training, Reader, Translator, MMLU-Pro,128-question confirmation, HellaSwag or GSM8K.
- Question/options/evidence are already part of the transferred body cache. Probes are the Receiver actually processed answer/continuation positions, not native-prefill question queries mislabeled as translated inference queries.
- All four local cases share the same Q, causal token grid, header and continuation tail KV within each fixed-Q reference. Native-Q and unmodified translated-Q references are separate; global intervention forwards additionally run with natural Q and with the selected group probe q_norm outputs clamped to Native. Clamping is post-q_norm/pre-RoPE: no repeated norm or rotation.
- K/V replacements use only the same prompt and same body positions in finalized post-k_norm/post-RoPE coordinates. Header/positions retained, suffix and continuation recomputed. No wrong-donor intervention or length-resampled key enters this experiment.
- Wrong-Donor phasefixed experiment already completed under manifest ac564a2d. Verify that source and independently test its exact reposition_key math against direct target-position rotation. Old unrephased donor results are excluded. Phase correction alone does not remove all resampling/context confounds.
- Gold-answer context computes full answer-token mean NLL and full-vocabulary KL at every gold token. Student-prefix context uses actual unmodified translated greedy token IDs at up to8 fixed positions; target-token NLL there is NOT gold-answer NLL. Every compared method/condition shares that exact context.
- Actual-Q forward queries and continuation tail states can change after repair. Its attention metrics include those changes and are not described as K-only isolation. Offline fixed-Q metrics hold both Q and nonbody tail fixed; query-clamped functional forwards fix selected probe queries but allow other states to recompute.
- Fixed-Q NN route/output differences must be zero, NT/NN routes must coincide, and TT/TN routes must coincide. Reconstruct post-Wo outputs from post-RoPE queries and GQA8KV/32Q attention, and compare to real o_proj hook outputs. Self-Q clamping and tracing must preserve original logits bitwise.
- Relative L2 is computed per layer and as group energy-weighted body K/V distances in the same finalized coordinates. RouteKL is NativeK||currentK attention probability divergence; output NMSE is post-Wo. Include per-head/per-probe diagnostics but never treat them as independent observations.
- Task accuracy/Hotpot EM/F1 and ARC original candidate margins are reused from identity-checked historical four-condition oracle results. New matched-context functional NLL/KL and first-answer choice margins are separate outcomes, not substituted for free-generation task results.
- Primary prediction features use the unmodified translated state, never pool all repaired states to inflate correlation mechanically. Partial Spearman removes layer-group mean rank effects; bootstrap resamples whole64-question clusters with their three groups. Report within-group associations as well.
- K/V output interaction is O_NN-O_NT-O_TN+O_TT under fixed Q; signed synergy is damageNT+damageTN-damageTT-damageNN. Functional NLL/KL synergy uses the same sign. Association is exploratory, not a proven causal mechanism.
- Both cohorts were used to identify sensitive layer groups. No held-out128-question confirmation exists in this phase, and no new method/loss/architecture follows automatically from A/B/C signals. Undefined correlations and inconclusive intervals are retained, not forced to zero or counted as support.
- Only JSON metrics/token prefixes/logs are persisted, not KV/query/attention tensors or model weights. Full copies are discarded between conditions. The earlier Transformer and fusion experiments remain paused; detached fail-stop pipeline, no automatic upload or continuous notifications.
