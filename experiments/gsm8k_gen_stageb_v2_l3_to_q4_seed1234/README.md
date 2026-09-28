# GSM8K Generation Stage-B v2

This independent experiment keeps Full-Sync alignment, the frozen Stage-A
Full28 MLP, and the zero-initialized Residual64 architecture fixed. It changes
only the Stage-B objective while making training prompts identical to the
lm-eval GSM8K v3 prompt (`Question: {question}\nAnswer:`).

Three adapters start from the same seeded initialization:

- `ce`: answer-token CE + 0.1 EOS CE;
- `kl`: native-Qwen full-vocabulary trajectory KL + 0.1 EOS CE;
- `hybrid`: trajectory KL + 0.1 answer CE + 0.1 EOS CE.

The seed-1234 split exactly reuses the old 1,024 training indices and assigns
the next 128 shuffled official-train rows to validation. Official test data is
never used for training or checkpoint selection. Every epoch saves a
checkpoint and evaluates 128 validation examples; selection prioritizes the
first-`####` accuracy. The final run evaluates the first 128 untouched test
rows with greedy decoding and a 256-token budget.

Run:

```bash
bash run_all.sh
```

Monitor:

```bash
tail -f logs/pipeline.log
```
