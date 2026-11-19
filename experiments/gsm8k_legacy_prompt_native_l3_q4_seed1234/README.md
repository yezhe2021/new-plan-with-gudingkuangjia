# GSM8K legacy-prompt native audit

Native-only Qwen3-4B and Llama3.2-3B evaluation on the first 128 official GSM8K test rows.
It exactly restores the previous dedicated experiment prompt (`Question: {question}\nAnswer:`),
greedy decoding budget, stop markers, and strict/first-hash/flexible numeric scoring.

No Writer checkpoint is loaded and no training occurs.
