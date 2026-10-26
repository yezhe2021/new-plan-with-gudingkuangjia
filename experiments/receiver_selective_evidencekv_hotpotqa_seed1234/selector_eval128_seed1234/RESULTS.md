# HotpotQA heldout128 selective receiver

Locked selector: `question_gain_calibrated`

|Condition|Accept-all EM|Reject-all EM|Selected EM|Selected F1|Acceptance|
|---|---:|---:|---:|---:|---:|
|clean|0.1328|0.0781|0.1328|0.2996|0.9844|
|wrong|0.0391|0.0781|0.0703|0.1201|0.6016|
|partial_wrong|0.0703|0.0781|0.0859|0.1609|0.9219|

Mixture results and paired intervals: see summary.json.

128 heldout questions, fixed before calibration32 inference; not official full HotpotQA benchmark.
Short-answer generation uses entire normalized output EM/F1. No gold-dependent answer extraction.
Selected rule fixed on pilot F1. Other rules are diagnostics, not alternative test-selected winners.
Balanced synthetic clean/wrong/partial corruption is a controlled mixture, not an estimate of deployment prevalence.
Both accept and reject branches are computed: no compute savings claimed.
First-token entropy and question likelihood are proxies, not calibrated truth probabilities.
No Reader, training, or quantization. Bootstrap resamples questions with their corruption variants together.
