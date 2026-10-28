# Same128 HotpotQA interface ablation

Same HotpotQA-trained A+B, no Reader. Uniform 128-question subset of the original512; no length/type/outcome exclusions. Original native and translated predictions reproduced exactly for all128 before proceeding.

| Condition | Native correct | Native EM / F1 | Translated correct | Translated EM / F1 |
|---|---:|---:|---:|---:|
| A_original | 60/128 | 46.88% / 60.43% | 28/128 | 21.88% / 35.03% |
| B_support_only | 80/128 | 62.50% / 77.88% | 41/128 | 32.03% / 46.38% |
| C_selector_prompt | 62/128 | 48.44% / 66.16% | 29/128 | 22.66% / 35.35% |
| D_evidence_only | 62/128 | 48.44% / 66.16% | 16/128 | 12.50% / 22.94% |
| E_diagnostic_decoder | 62/128 | 48.44% / 66.16% | 16/128 | 12.50% / 22.94% |

A: original full distractor prompt and complete prompt KV. B: remove distractors, preserve original supporting passage numbers. C: use the previous selector evidence formatting/instructions, still transfer the complete question-containing prompt. D: same Qwen input token sequence as C, but transfer evidence only and process the question natively. E: same evidence cache as D, use the prior diagnostic greedy loop; common answer cleaning throughout.

## Paired stepwise changes

| Change | Native F1 delta (95% CI), pp | Translated F1 delta (95% CI), pp |
|---|---:|---:|
| B_support_only minus A_original | +17.45 [+11.61, +23.73] | +11.35 [+5.38, +17.70] |
| C_selector_prompt minus B_support_only | -11.72 [-16.53, -7.38] | -11.04 [-17.40, -4.86] |
| D_evidence_only minus C_selector_prompt | +0.00 [+0.00, +0.00] | -12.41 [-20.03, -4.91] |
| E_diagnostic_decoder minus D_evidence_only | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

Bootstrap resamples the same128 original questions, 5000 replicates. These are exploratory ordered contrasts; effects are conditional on earlier changes, not order-independent attributions. No selector is fitted or selected.

C versus D keeps the complete Qwen input tokens identical; it changes which model computes the question states and the transfer/prefill boundary. Native C/D/E results measure sensitivity to the prefill/decoder split. E retains the original model loading/backend, so this check alone does not isolate every runtime difference from the earlier diagnostic.

The question subset differs from the earlier short bridge-only128. Compare same-question A-E for interface effects; do not treat cross-subset accuracy gaps as pure interface effects. Support-only variants use gold supporting titles and are controlled diagnostics, not the official distractor task.

Total evaluation time: 13.27 minutes.
