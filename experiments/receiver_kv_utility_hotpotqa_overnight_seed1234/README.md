# Overnight receiver KV utility experiment

1,792 new document-disjoint HotpotQA bridge questions: train1024/calibration256 from the official training shard; test512 from official dev. Known prior Hotpot questions/documents excluded. Two complete gold paragraphs are supplied to the Sender; this is a controlled task, not the full distractor benchmark.

Frozen Llama3.2-3B, grouped Stage-A + rank64 Stage-B, and Qwen3-4B; no Q/O Reader. Only small CPU utility estimators are fitted. All models co-reside on GPU; mapped KV uses a bounded CPU LRU cache and is never persisted to disk.

Primary conditions: clean, wrong donor, partial50. Independent test additionally evaluates unseen partial25/partial75/Gaussian corruption. Donors are assigned inside each split without answer labels. Gaussian noise preserves target aggregate moments.

Each evidence packet receives eight deterministic probes: random or contiguous 5%/10% masks, two seeds. All masks act before processing the question and preserve position indices. Features use generated Accept/Reject answers only: output confidence, paired sequence likelihoods under both branches, and perturbation response. Gold NLL/native evidence/corruption identity are excluded from estimator input allowlists. A small calibration-only subset also regenerates answers after perturbation to audit score-based stability.

Compare fixed/calibrated Question Gain and entropy, separately estimated branch correctness, and regularized utility models using output / cross-score / 2,4,8-probe features. EM three-class logistic utility and continuous F1 ridge utility are separate tracks. Hyperparameters and thresholds use calibration only and are locked before test inference. Primary model: utility_stability8_f1; primary metric F1, secondary EM. Report question-clustered paired intervals, helpful recall, harmful rejection, utility AUROC/calibration, unseen corruption generalization, and latency. Do not select a test winner or retune on test outcomes.

Pipeline: GPU preflight+smoke → development extraction → CPU fitting/lock → independent test → report. All stages run without assistant/API calls. Per-case outputs save atomically; completed records can be resumed with identical manifest. A failure is recorded rather than silently changing protocol. Run `python run_pipeline.py --launch`, or reviewed continuation `python run_pipeline.py --launch --resume`. Results automatically append to the shared EXPERIMENT_RESULTS.md. No automatic GitHub upload.
