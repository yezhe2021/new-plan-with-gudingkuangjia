# Frozen A+B + task-supervised Reader: results

Completed 2026-10-25 17:43:24 Asia/Shanghai. Pipeline started16:11:01; 1024 Reader updates completed, both Stage-A/B tensors remained unchanged. The primary checkpoint was fixed before training: epoch4_step1024, SHA256 `4637d5de199d16b194b31c8ef5159a997339547a95b60511ab8325844d34e26c`.

| Task | n | Qwen native | A+B | Previous KL Reader | New task-supervised Reader | Change vs previous |
|---|---:|---:|---:|---:|---:|---:|
| OBQA |128|103|88|92|96 (75.00%)|+4|
| ARC-Challenge |128|111|82|86|92 (71.88%)|+6|
| HellaSwag |64|23|22|22|23 (35.94%)|+1|
| GSM8K |32|31|10|14|1 (3.13%)*|-13|
| MMLU-Pro custom split |128|52|33|35|40 (31.25%)|+5|
| SciQ |64|57|56|57|52 (81.25%)|-5|

All544 original records/counts were verified against the machine summary. Counts use identical historical examples and inference settings. Four tasks improve and two decline versus the prior Reader. This run does not support replacing the previous Reader across tasks; GSM8K substantially regresses. No new training or checkpoint selection was performed after viewing these results.

## GSM8K scoring audit

The original `summary.json` reports0/32. The historical extractor takes the full line after `####`; generation decoding retains chat control tokens, so a saved output `#### 7<|im_end|>` becomes the literal prediction `7<|im_end|>` and is marked wrong against gold7.

`scoring_audit/audit_saved_results.py` removes literal `<|...|>` control tokens from **all six arms' saved text**, then applies the unchanged historical number extractor. This gives1/32 for the new Reader; all other scores remain31/9/8/10/14. Only `gsm8k_test_22` becomes correct. Unit text and all other extraction rules are left unchanged. Original machine summaries and raw generations remain intact; no generation rerun.

The corrected new Reader retains1 of the prior Reader's14 correct answers and fixes none of its18 wrong answers. Other saved outputs show incorrect quantities/relations, e.g. test0 gives348 instead of18; the scoring bug does not explain the overall regression. This is observation, not proof of a particular training failure mechanism.

GSM average output length211.5 tokens versus467.84 for the prior Reader; both have one cap4096 hit. Teacher-forced validation CE improves from0.8424 at epoch1 to0.7616 at epoch4 while free-generation accuracy falls sharply. Teacher-forced loss is insufficient to establish autoregressive reasoning quality.

## Scope

Fresh7,667,712-parameter Q/O Reader; frozen existing Llama→Qwen grouped Stage A and rank64 residual B. MCQ correct-choice CE +0.1 candidate KL; GSM full reference solution/final answer/EOS token-mean CE +0.1 full-vocabulary KL,32-token TBPTT. Four-task512each×4epochs, batch8,lr1e-4. The objective, histories and GSM supervision coverage change together; these results do not isolate their individual effects. MMLU-Pro remains a custom split; Full-Sync still includes native body prefill.
