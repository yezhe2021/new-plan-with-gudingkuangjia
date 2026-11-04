# Llama to Qwen frozen Memory mechanism analysis

All three diagnostics use the same trained3-task all-layer checkpoint.192 existing QA questions; no training.

## Free generation: identical64-token protocols

| Task | Metric | Native | C | Keep | Same-position masked | Wrong donor | Header-only Memory |
|---|---|---:|---:|---:|---:|---:|---:|
| hotpotqa | em | 56.25% | 29.69% | 46.88% | 31.25% | 43.75% | 42.19% |
| hotpotqa | f1 | 68.52% | 46.53% | 61.09% | 46.27% | 59.39% | 56.98% |
| 2wiki | em | 32.81% | 9.38% | 32.81% | 10.94% | 31.25% | 14.06% |
| 2wiki | f1 | 43.45% | 22.05% | 43.95% | 22.13% | 39.07% | 25.57% |
| musique | em | 25.00% | 1.56% | 21.88% | 4.69% | 20.31% | 7.81% |
| musique | f1 | 33.07% | 7.64% | 27.79% | 10.93% | 25.97% | 12.68% |

## Gold-answer prediction: per-question means, EOS separate

| Task | Condition | Gold NLL | KL to natural Native | KL to shifted Native | Native top1 match |
|---|---|---:|---:|---:|---:|
| hotpotqa | native | 3.7059 | 0.0000 | 0.0229 | 100.00% |
| hotpotqa | C | 4.5607 | 1.6572 | 1.6908 | 80.77% |
| hotpotqa | translated_shifted_masked | 4.3684 | 1.5996 | 1.6108 | 81.42% |
| hotpotqa | keep | 0.8876 | 0.7153 | 0.6838 | 72.42% |
| hotpotqa | native_shifted_masked | 3.6575 | 0.0231 | 0.0000 | 96.91% |
| hotpotqa | wrong_memory | 0.9048 | 0.7263 | 0.6940 | 70.99% |
| hotpotqa | header_only_memory | 1.4406 | 0.7201 | 0.7059 | 80.43% |
| 2wiki | native | 5.3289 | 0.0000 | 0.0278 | 100.00% |
| 2wiki | C | 6.7005 | 3.2630 | 3.2641 | 63.68% |
| 2wiki | translated_shifted_masked | 6.3112 | 3.1881 | 3.1818 | 64.22% |
| 2wiki | keep | 1.6029 | 1.0831 | 1.0757 | 68.29% |
| 2wiki | native_shifted_masked | 5.1977 | 0.0445 | 0.0000 | 98.71% |
| 2wiki | wrong_memory | 1.6820 | 1.1194 | 1.1025 | 65.43% |
| 2wiki | header_only_memory | 2.6027 | 1.3089 | 1.3162 | 69.81% |
| musique | native | 3.6719 | 0.0000 | 0.0462 | 100.00% |
| musique | C | 5.3271 | 2.7686 | 2.8063 | 65.98% |
| musique | translated_shifted_masked | 5.1363 | 2.6297 | 2.6545 | 68.03% |
| musique | keep | 1.4840 | 1.0531 | 1.0388 | 71.39% |
| musique | native_shifted_masked | 3.6400 | 0.0427 | 0.0000 | 97.85% |
| musique | wrong_memory | 1.5419 | 1.0870 | 1.0742 | 72.17% |
| musique | header_only_memory | 2.4493 | 1.2718 | 1.2827 | 69.44% |

Synthetic fact pairs: 16/16 have both Native endpoints correct. All pairs remain reported; no resampling by correctness.

Positive paired NLL gain for keep_vs_same_position_masked means Memory improves answer probability beyond position shift. Positive group_access gain means blocking that group harms answer NLL, conditional on all other groups; gains are not additive.
Fixed-Q cosines compare local operators with a common background and suffix states. Keep-Q native-body target is a hybrid oracle. Actual trajectory similarity is observational. Fixed-Q background token ratios cannot change when only Memory keys are added.

- Use latest Llama3.2-3B→Qwen3-4B Recolor+C frozen translator and3-task Keep-both epoch4/step1536. No Gemma mapping, new training, checkpoint selection or Reader. Answer-time Memory is open at all36 layers, matching training; focus groups19-24/25-30 while reporting all36.
- All192 existing QA examples remain primary, with old C-vs-Keep rescue/harm/both-correct/both-wrong strata reported separately. Hotpot is a seen task with held-out questions;2Wiki/MuSiQue were not Memory training tasks. Historical cohorts have been inspected.
- Functional alignment uses full canonical answer tokens under identical semantic gold prefixes. EOS is scored separately. Compare Native, C, position-shifted C with Memory masked, Keep-both, position-shifted Native with masked slots. Report Native-original and Native-shifted KL, gold NLL/logprob, top1 match, per-question paired differences; closeness to Native alone is not success.
- Counterfactual Memory changes only the32 Memory entries. Correct, length-bucketed same-task wrong donor, all-Memory masked, and position-matched header-only construction are compared. Donor is a fixed derangement, not correctness-selected. Rephase post-RoPE donor K from donor memory positions to recipient positions; leave V unchanged; never k_norm again. Rephasing does not erase donor hidden-state history.
- Fixed-Q analysis holds Q, positions and ALL non-Memory K/V including gold-prefix states fixed within each local operator comparison. Use Native-shifted and real Keep queries separately. Native-body reference under Keep Q/tail is a local hybrid oracle, not actual Native inference. Analyze all36 layers and32Query/8KV GQA heads without storing full attention matrices.
- Decompose fixed-Q output into background dilution and Memory-value addition, check identity before output projection, and check native/Keep reconstruction against actual model outputs. Attention mass/cosine correlations are diagnostic. Independently mask each predeclared6-layer Memory group and rerun the complete gold suffix/prefix to measure functional effects; do not add group effects as independent contributions.
- Counterfactual evidence addendum uses16 synthetic two-hop pairs with an unchanged question and one same-token-length city fact swapped. Both Native endpoints must answer their respective known fact before semantic interpretation. Use2x2 original/counterfactual body and memory combinations. These are controlled synthetic probes, not new Hotpot evidence or benchmark scores.
- No claim Memory creates facts absent from translated KV, no Native-state recovery requirement, and no private evidence-only protocol claim. Native body KV is offline diagnostic reference only and never supplies student Memory inputs. Other jobs including Gemma all36 remain held. No automatic upload or continual monitoring.
