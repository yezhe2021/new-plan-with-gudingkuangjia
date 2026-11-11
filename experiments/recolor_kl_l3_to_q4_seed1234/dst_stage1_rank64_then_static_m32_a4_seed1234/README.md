# B then D: frozen original DST Stage1 + Rank64, then Static Memory

B starts at original frozen DST Stage1 epoch4/1536. Only bias-free per-layer K/V1024→64→1024 GELU residuals train (9,437,184 parameters). Up projections zero, placed before exactly-once Qwen k_norm and RoPE. No original full-parameter Stage2 weights. Same Full-Sync splits and mixed-prefix CE+.5 KL training,4epochs1536 updates/batch8/lr1e-4. Fixed final checkpoint. Sender/Qwen/DST Stage1 frozen.

B pipeline: prepare, CPU/GPU zero-residual native/Stage1 parity and actual backward checks, B training, eight-task64 evaluation. Then static_memory/ pipeline automatically pins B final checkpoint, prepares identical data/schedules, checks CPU/GPU parity and consecutive gradients, trains fresh Static M32+four Rank16 adapters after6/12/18/24 with all36 Keep-both,1536updates, then evaluates same512 questions. No warm-start of Memory, no extra KV cache or base-weight duplication.

D uses fixed-header-only canonical native Qwen Memory, existing rephasing and32-slot positions. Body remains readable all36 layers. Freeze both DST and Rank64. Checks compare original Stage1 A, full Stage2 E, fullStage2+Static F; B and D per-sample outcomes provide direct Memory increments. Small single-seed evaluation; no interpretation as proven capability separation.

Both flags controls/pause_requested.json and static_memory/controls/pause_requested.json are honored. Background runner and checkpoints are independent of the local laptop. No automatic uploads.
