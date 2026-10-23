# Grouped translator with frozen base and cross-head residual Stage B

This is the user's corrected training scope. Reuse the best Stage-A weights from
`grouped_headmix4_l3_to_q4_seed1234`; never initialize from its fully tuned Stage B.
Keep its 134,217,728 translator parameters and both LLMs frozen. No Stage-A retraining.

For every target layer and independently for K/V, flatten all 8×128 head features,
then apply `x + Linear(64,1024)(GELU(Linear(1024,64)(x)))` and reshape into heads.
The up projection starts at zero. Initial output exactly equals the frozen base.
The residual can mix heads within a target layer; the frozen grouped translator
already supplies the source-layer grouping and cross-head connections. No Reader.
Exactly 9,437,184 parameters are trainable. Only these parameters enter AdamW and
checkpoints; every residual checkpoint pins the immutable Stage-A file hash.

Keep previous Stage-B data, KL loss, mixed Native/student prefixes, 4 epochs,
1024 updates, effective batch 8, lr 1e-4 and fixed final-checkpoint selection.
The full-translator run and this residual run share Stage A; they differ in Stage-B
trainable scope and parameterization. Do not call this a grouping-only ablation.

CPU tests check zero-up identity, cross-head connectivity, independent K/V,
real optimizer updates that leave base weights bitwise identical, adapter-only
serialization and parameter count. GPU smoke additionally tests the real cache
interface and Native/student backward passes, finite residual gradients and no
Receiver/base gradients. It checks the entire base tensor digest before/after.
Formal training checks the same base digest at every epoch boundary.

Use `python -m unittest -v test_translator test_residual test_queue`, then
`python experiment.py prepare`. The pipeline runs only Stage-B smoke, Stage-B
training and six-task evaluation. It reuses checked Native, old UFD and previous
full-translator outputs for direct comparison; test protocols remain unchanged.

`queue/server_queue.py` is the server-only detached launcher. It waits for prior
completion and enough resources, invokes the idempotent launch entry, and exits
after confirming startup. It survives local shutdown/SSH disconnection but not
a server reboot. Existing experiments and their records remain untouched.
