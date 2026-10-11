# Cross-Corpus Stage-A Mapping Consistency

No training, no Stage-B. Same Sender and Native state per sample. Macro-average over samples; K post-k_norm/pre-RoPE. Neutral common wrapper: analysis, not accuracy benchmark.

|Corpus|N|KV|M/G cosine|M/G symmetric NMSE|MCQ→Native NMSE|GSM→Native NMSE|Error cosine|
|---|---:|---|---:|---:|---:|---:|---:|
|openbookqa|64|K|0.98120|0.03783|0.09237|0.09700|0.81394|
|openbookqa|64|V|0.73346|0.53384|0.60944|0.65284|0.81485|
|arc_challenge|64|K|0.98103|0.03813|0.08728|0.09319|0.80435|
|arc_challenge|64|V|0.72872|0.54428|0.58978|0.65044|0.80467|
|gsm8k|128|K|0.98250|0.03522|0.08596|0.06774|0.79458|
|gsm8k|128|V|0.72215|0.58087|0.65623|0.46874|0.76888|
|wikitext|128|K|0.98242|0.03524|0.09727|0.09905|0.83390|
|wikitext|128|V|0.72517|0.55095|0.69166|0.67649|0.84833|
|mcq|128|K|0.98112|0.03798|0.08983|0.09510|0.80914|
|mcq|128|V|0.73109|0.53906|0.59961|0.65164|0.80976|
|all|384|K|0.98201|0.03615|0.09102|0.08729|0.81254|
|all|384|V|0.72613|0.55696|0.64917|0.59896|0.80899|
