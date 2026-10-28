# Evidence-only layer-group Oracle repair pilot

Exploratory32 questions,27 configurations. Frozen Hotpot A+B, no Reader; all distractor paragraphs preserved. All baselines recomputed on evidence-only transfer.

| Arm | EM | F1 | Gold token-mean NLL |
|---|---:|---:|---:|
| translated | 3.12% | 3.65% | 6.5829 |
| native | 40.62% | 54.80% | 4.8101 |
| reject_all | 15.62% | 18.87% | 6.6022 |
| g1_K | 0.00% | 0.00% | 8.2068 |
| g1_V | 9.38% | 11.34% | 5.2467 |
| g1_KV | 15.62% | 19.28% | 4.4066 |
| g1_mask | 9.38% | 12.82% | 4.6081 |
| g2_K | 9.38% | 13.58% | 5.5008 |
| g2_V | 3.12% | 3.12% | 6.5410 |
| g2_KV | 9.38% | 10.81% | 5.7197 |
| g2_mask | 3.12% | 3.65% | 6.3593 |
| g3_K | 15.62% | 18.96% | 5.0141 |
| g3_V | 3.12% | 3.91% | 6.5723 |
| g3_KV | 15.62% | 17.78% | 5.1172 |
| g3_mask | 6.25% | 7.47% | 5.5818 |
| g4_K | 9.38% | 10.73% | 4.9294 |
| g4_V | 3.12% | 3.38% | 6.7730 |
| g4_KV | 9.38% | 12.08% | 4.8487 |
| g4_mask | 3.12% | 4.21% | 5.3911 |
| g5_K | 3.12% | 3.33% | 6.5546 |
| g5_V | 3.12% | 3.57% | 6.9122 |
| g5_KV | 3.12% | 4.30% | 6.6790 |
| g5_mask | 3.12% | 5.06% | 6.6298 |
| g6_K | 3.12% | 3.12% | 7.8966 |
| g6_V | 3.12% | 4.27% | 7.8944 |
| g6_KV | 3.12% | 4.61% | 8.2875 |
| g6_mask | 3.12% | 3.75% | 9.2306 |

- This is a new evidence-only transfer interface: Qwen natively reprocesses Question and answer continuation for every condition; previously reported full-prompt transfer scores cannot be reused as these baselines.
- Frozen new Hotpot-only A+B checkpoints; no Reader, repair training, parameter updates or oracle-driven selection of pilot questions. Every original distractor paragraph is retained.
- Native and translated evidence caches are assembled on the same Qwen token grid. Only post-RoPE physical cache states are replaced, with identical common native anchor (including first content token). Rejection blocks all transmitted evidence positions but retains this common anchor and absolute positions.
- Qwen has36 layers,8 KV heads and32 query heads. Six groups of6 layers: K-only/V-only repair8.33% of transmitted K+V entries; KV repair16.67%. No head search in this first round.
- Group masking is a per-layer causal attention mask over the transmitted evidence columns, from the first Question token and throughout continuation/gold scoring; K/V are not zeroed. One shared-position mask arm per group,27 total conditions including three baselines.
- Gold NLL is answer-token mean cross-entropy, excludes EOS and is offline only. Gold labels never enter evidence, Question, generated context or layer selection policy. All27 arms receive full generation, no NLL filtering.
- All32 questions are exploratory discovery data. Next128 dev IDs are reserved, not evaluated automatically. Candidate winners and bootstrap intervals do not establish confirmatory effects after multiple comparisons.
- Recovery ratio is defined only for positive aggregate Native-minus-Translated F1, left unclipped. A beneficial hybrid need not imply erroneous local KV, and lack of single-group gain does not exclude interactions.
- Oracle repairability does not establish that Native residuals can be predicted from translated KV and Question. Existing Stage B already supplies a residual; a new repair module needs evidence for query-conditioned/local functional value beyond it.
