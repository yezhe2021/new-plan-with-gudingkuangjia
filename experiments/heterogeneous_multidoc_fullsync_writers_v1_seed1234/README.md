# Corrected Full-Sync writer composition pilot — V1

Server A, evaluation only, frozen models/writers/readers. This corrects
the V0 writer selection: V0 accidentally used the old OpenBookQA sparse
RawAnchor32 checkpoints. V0 results do not evaluate the user-requested
full-token-retrained Full-Sync writers and remain preserved for audit.

Correct original validation-only selections:

- Llama->Qwen: `fulltoken_sync_copydrop_v0_l3_to_q4_seed1234/runs/retrain`,
  Stage-A epoch4/step1536 and Stage-B epoch2/step768.
- Gemma->Qwen: `fulltoken_sync_copydrop_v0_g3_to_q4_seed1234/runs/retrain`,
  Stage-A epoch2/step768 and Stage-B epoch2/step768.

Both were freshly trained using EVERY aligned receiver Options token. The
Llama checkpoint's protocol string inherits an old options32 label; its actual
full-token training identity is established by retraining-directory provenance,
training_config, original selection files, stage/epoch/step and SHA256. It must
not be mistaken for the earlier sparse checkpoint merely because of that label.

Original variable-length translator classes and UTF-8 synchronized-unit
copy/drop implementation are vendored unchanged. The writer accepts arbitrary
token count. Consecutive 64-token microchunks bound activation memory, as in
the original Full-Sync evaluator; no slot padding, selection or compression is
introduced. FP16 autocast follows the original evaluator.

Use the SAME 64 official HotpotQA DEV bridge questions as V0, same complete
gold supporting paragraph pair/order, prompt, document-only source encoding,
Qwen-native public prefix, answer parser and greedy32 decoding. Exact ordered
ID equality is asserted. Only writer identity, its proper variable-length
module/interface, and original inference arithmetic are corrected. The
document-only/free-generation protocol remains an out-of-domain mechanism
diagnostic, not an official benchmark score or an impossibility claim.

Arms: query only; Qwen full native; Qwen independent native; same-source joint
and independent translated memories; Llama/Gemma mixed source in both orders;
translated/native mixed memories in both orders. No receiver adaptation yet.
Source model/writer/receiver loading is sequential; no KV tensors persist.

Input copying is exact/resumable and checked against source SHA256. All inputs
must be verified before model checks and smoke. CPU tests check arbitrary-token
interfaces and original synchronized-unit mapping; real GPU tests verify native
cache reconstruction, source capture and frozen writer routing. Two-example
smoke precedes the64-question pilot. Code writes per-sample outputs and updates
the shared EXPERIMENT_RESULTS.md automatically on completion/failure.

Run: `/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u experiment.py pipeline`
Log: `tail -f logs/pipeline.log`

Execution moved to Server A at the user request. The original weights and dataset are linked in place; no weight transfer is required. With limited free GPU memory, whole transformer-layer weights may be retained on CPU and loaded for CUDA execution by Accelerate. All model arithmetic retains the original dtype; no CPU arithmetic or quantization is introduced. Loading placement is recorded in model_load_events.json.
