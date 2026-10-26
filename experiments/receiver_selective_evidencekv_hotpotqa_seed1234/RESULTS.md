# HotpotQA frozen evidence-KV pilot

Completed 32 calibration questions. Reserved128 has not been evaluated.

|Condition|EM|F1|Gold answer token NLL|
|---|---:|---:|---:|
|question_only_compact|0.0625|0.1275|5.9062|
|reject_all|0.0312|0.0955|6.2238|
|native_evidence|0.5312|0.6200|2.6396|
|clean|0.1562|0.2558|3.6283|
|wrong|0.0312|0.0610|4.8680|
|partial_wrong|0.0938|0.1410|3.7964|

Pilot gate: {'checks': {'native_evidence_useful': True, 'translated_gain_large_enough': True, 'translated_paired_ci_positive': True, 'clean_better_than_wrong': True}, 'passed': True, 'next_step': 'Ready to lock selector on pilot and evaluate reserved128'}

Controlled bridge questions with two gold paragraphs, not full HotpotQA distractor benchmark.
Only calibration32 evaluated; no selector fitted and reserved128 untouched.
Wrong/partial KV corruption is synthetic; length matching and resizing do not model all real transmission errors.
Masks hide evidence keys from all question and answer tokens, retaining common anchor and position slots.
No Reader, training or quantization; original grouped Stage-A and residual Stage-B frozen.
Short-answer generation scored with normalized whole-output EM/F1; no answer-dependent parsing.
