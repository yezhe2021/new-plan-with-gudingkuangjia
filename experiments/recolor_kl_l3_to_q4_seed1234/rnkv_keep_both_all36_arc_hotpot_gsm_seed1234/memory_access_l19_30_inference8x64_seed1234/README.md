# Restrict the new three-task checkpoint to Memory access at layers19-30

Use the completed ARC+HotpotQA+GSM8K-trained all36 Keep-both final checkpoint (epoch4/step1536). No training and no substitution of the old two-task checkpoint.

Only answer-time Memory visibility changes: layers19-30 can read translated body plus Memory; layers1-18/31-36 retain translated body and block Memory. Public header and causal suffix/generated positions remain visible. Memory construction and all four trained adapters after6/12/18/24 remain enabled as originally trained. Memory positions and answer offset remain unchanged; no cache overwrite or native evidence prefill.

The trained all36 model is restored after each temporary scoring context. Before evaluation, reproduce its cached scores and complete generated output on the first example of each of eight tasks. Verify bitwise identical Memory construction under the restricted setting, every layer's body/memory access, causal answer positions, frozen parameter versions and absence of native student prefills. Reuse those diagnostic new-arm outputs in evaluation.

Copy the parent's512 test rows byte-for-byte:64 per task. Reuse original seven-task scorers and GSM512-token protocol, including both unchanged strict-final and legacy numeric extraction metrics. Reuse source-hashed original C, RN-KV, old Keep-both, newly trained all36 and Native predictions. Save per-task paired differences, rescued/harmed IDs, caps and complete outputs. Keep baseline provenance rather than rerunning all models.

This is a train/test routing intervention because this checkpoint was trained with all36 layers reading Memory. It does not establish the effect of training a three-task model with Memory access restricted to19-30 from the start.

Run detached using `run_pipeline.py launch`; no continual monitoring or automatic upload. Other paused experiments stay paused.
