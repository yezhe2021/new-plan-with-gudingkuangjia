# MMLU transfer evaluation of the existing MCQ translator

Evaluate the frozen original MCQ-trained Llama-to-Qwen Stage A and Stage B checkpoints on MMLU, with Llama and Qwen native baselines. This follow-up does not train or modify any model.

Use installed lm-eval 0.4.12's official `mmlu` task: four letter candidates scored by continuation log-likelihood, five official dev demonstrations per subject, no chat template. Allocate 128 total test questions across all 57 alphabetically sorted subjects (the first 2 or 3 test questions each). This is a pilot subset, not a full MMLU benchmark. There is no generated-answer parser.

Official `cais/mmlu` test/dev parquet bytes are pinned to revision `c30699e8356da336a370243923dbaf21066bb9fe`. Local dataset routing filters the official combined split by subject and preserves row order. No test examples are used as demonstrations. Data, source, checkpoint hashes, checkpoint selection metadata, question IDs, prompts, candidate scores and per-question metrics are retained.

Run `prepare_data.py` on a machine with access to Hugging Face, then copy this directory to the same relative location on server B. With the server B attnkv Python, run `run_mmlu.py prepare`, `run_mmlu.py smoke`, then `run_mmlu.py pipeline`. Native/oracle likelihood parity and a Stage B longest-prompt smoke check must pass before the pipeline starts. Each condition runs in a fresh subprocess and existing completed results are retained. Scripts and results stay in this follow-up directory; model checkpoints remain in the parent experiment.
