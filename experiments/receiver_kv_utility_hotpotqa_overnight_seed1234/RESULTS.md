# Overnight HotpotQA KV utility experiment

Train 1024, calibration 256, independent test 512.

Primary model: `utility_stability8_f1`. F1 is the primary metric; EM is secondary.

|Policy|EM|F1|Accept rate|Helpful recall|Harmful rejection|
|---|---:|---:|---:|---:|---:|
|always_accept|0.0729|0.1741|1.0000|1.0|0.0|
|always_reject|0.0508|0.1372|0.0000|0.0|1.0|
|question_gain_zero|0.0775|0.1803|0.8171|0.9764705882352941|0.17647058823529413|
|question_gain_calibrated|0.0775|0.1803|0.8171|0.9764705882352941|0.17647058823529413|
|calibrated_confidence|0.0872|0.1948|0.6667|0.8470588235294118|0.6862745098039216|
|utility_output_f1|0.0814|0.1905|0.5443|0.8470588235294118|0.5098039215686274|
|utility_cross_f1|0.0859|0.1966|0.5312|0.8705882352941177|0.6078431372549019|
|utility_stability2_f1|0.0866|0.1947|0.5143|0.8705882352941177|0.6274509803921569|
|utility_stability4_f1|0.0872|0.1953|0.5078|0.8588235294117647|0.6666666666666666|
|utility_stability8_f1|0.0859|0.1954|0.5137|0.8352941176470589|0.6666666666666666|

Full paired intervals, per-condition results, unseen corruption stress tests and utility AUROC/calibration: summary.json.

Controlled two-gold-paragraph HotpotQA bridge subset; not official distractor benchmark.
All hypotheses, model grids and thresholds frozen before test inference; primary model is utility_stability8_f1.
Equal clean/wrong/partial50 mixture is a controlled distribution; stress corruption types excluded from fitting and calibration.
Question/document disjointness includes repeated corruption versions and donor sources.
Gaussian retains target aggregate moments; synthetic corruption does not establish reliability on natural retrieval errors.
Candidate pair scores are sequence likelihood features, not calibrated probabilities of correctness.
No inference compute reduction claimed. Stability2/4/8 use nested subsets of the same probes.
Reported intervals resample original questions, preserving all their corruption variants. Exploratory secondary comparisons are not multiplicity-corrected.
