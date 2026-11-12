# Gemma → Qwen: DST Stage1 + frozen-base Rank64 Stage2 (server A)

Fresh Gemma-3-4B-it RAW KV -> Full-Sync Copy/Drop -> 34-to-36 layer DST with a 256-wide recurrent state -> Qwen3-4B. Stage1 trains the complete DST against Native pre-k_norm/pre-RoPE KV NMSE. Stage2 loads and freezes exactly the final Stage1 checkpoint and trains only 9,437,184 bias-free cross-head K/V Rank64 GELU residual weights (zero up projections). Qwen k_norm and RoPE are applied exactly once. No Memory training or inference stage is scheduled.

ARC/HotpotQA/GSM8K each1024 training+64 validation; four epochs, batch8,1536 updates per stage. Stage1 lr3e-4; Stage2 lr1e-4, full gold CE+.5 Native||Student full-vocabulary KL; unchanged mixed-prefix schedule, max64 generated prefix and8 KL positions. Gold rationale is not truncated. Final epoch4 selection fixed in advance. Eight evaluation tasks each64, QA generation64, GSM512 with strict/legacy scores.

Gemma FP32, Qwen FP16. Public model artifacts on A must match existing B artifacts by SHA256. Reuse verified canonical task data and scorers; regenerate Sender token alignment for Gemma. Gemma Recolor+C historical per-question results are pinned and joined by question, ID, gold; no Llama C scores are substituted. Alignment and training budgets differ from Recolor+C, so this is an end-to-end comparison, not a recurrence-only ablation.

CPU tests cover synchronization units, token independence, recurrent gradients, exact cache reconstruction, checkpoint gradients, and Rank64 identity/frozen-base behavior. Before each training stage, disposable GPU tests check raw Native cache equivalence, longest samples, and consecutive effective batches. Stage2 checks additionally require zero-residual cache identity to trained Stage1. Smoke updates are discarded.

Launch: `python -u run_pipeline.py launch train`. The detached server-side pipeline prepares assets/data, performs checks, trains Stage1 then Stage2, and evaluates both. It stops on an error and honors `controls/pause_requested.json`. No mutation of existing experiments, automatic Memory job, or GitHub upload.
