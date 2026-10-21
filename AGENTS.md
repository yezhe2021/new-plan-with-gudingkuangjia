# KV research handoff

This repository is the portable project context. Do not depend on the previous
computer's conversation history, absolute Windows paths, or globally installed skills.

Before doing project work, read:

1. `NEW_COMPUTER_SETUP.md` for access and migration boundaries.
2. `PROJECT_HANDOFF.md` for the current experiments and authoritative artifacts.
3. `EXPERIMENT_PROTOCOL_DEFAULTS.md` before implementing or running evaluations.
4. `EXPERIMENT_RESULTS_POLICY.md` before changing result reporting.

Default GPU execution is server B (`ssh server-b`). Server A is secondary; do not
silently move work between servers. Inspect processes, logs and completion markers
before launching anything. Cloning this repository is NOT permission to rerun training.
Do not stop/preempt unrelated jobs or delete caches/checkpoints without authorization.

The user selected HeteroFold's pinned PER-TASK inference/evaluation configuration
on 2026-10-19. Do not silently revert to legacy Answer:/256-token/last-number rules.
Keep dataset splits, prompt IDs, cache/position protocol, model and checkpoint
identities fixed. Explicit ablations must record every override. Custom MMLU-Pro
splits must not be described as untouched official full-test/5-shot-CoT benchmarks.

Use the explicit server Python path in the setup document. Verify GPU/CUDA before
GPU work. For requested launches, confirm startup and provide the log command;
continuous monitoring requires an explicit request. SSH failure is not proof that
a remote process failed; do not blindly relaunch it.

Upload lightweight scripts/config/logs/machine summaries/per-sample outputs to
`experiments/<experiment>/`; follow-ups stay nested in that experiment. Exclude
weights, optimizer states, KV caches, temporary tensors and credentials. Use a
clean upload checkout, stage only intended paths, push once, and verify remote
HEAD. Retry the SAME commit for transient failures, not duplicate commits.

Never commit passwords, tokens, private SSH keys or machine-specific credentials.
Do not assume the old computer's local proxy exists on this computer.
