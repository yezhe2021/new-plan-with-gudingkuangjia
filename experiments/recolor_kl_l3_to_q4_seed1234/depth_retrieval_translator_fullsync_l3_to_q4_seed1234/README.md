# DRT Full-Sync Llama → Qwen, seed1234

User-authorized fresh two-stage training, no Memory. Architecture is specified in translator.py and config.json: 28 source layers jointly read by input-dependent four-head depth attention, independent 36 full-width readouts and direct full-head bypass; width256. Outputs pre-k_norm/pre-RoPE, Qwen finalizes exactly once. No recurrence or cross-token attention.

Reuses byte-identical completed DST aligned splits. ARC/HotpotQA/GSM8K 1024 training +64 validation each; two stages each4 epochs/1536 accepted updates, effective batch8. Stage1 normalized raw KV MSE, Stage2 full gold CE+.5 full-vocabulary KL with identical mixed prefixes. Final epoch4 fixed. Eight tasks64 test each; GSM512 output cap, strict/legacy labels unchanged. Historical C budgets/interfaces differ. No test checkpoint selection.

CPU: zero-readout identity, depth dependence, no token interaction, chunk equivalence, genuine36-layer Qwen cache and checkpoint-gradient parity. GPU: exactly-once norm, Native scoring/cache parity, longest examples both stages, consecutive batch8 gradients, frozen models. Smoke updates discarded; formal training starts in separate fresh process. Detached pipeline automatically trains both stages then evaluates, pauses on controls/pause_requested.json. No KV disk cache created.

References are reused only for data, alignment and scoring. Existing DST/Recolor weights are not loaded into DRT. Source and Receiver remain frozen. Parameter count and hashes recorded in manifest. Complete run logs, accepted updates, validation and per-sample outputs preserved.
