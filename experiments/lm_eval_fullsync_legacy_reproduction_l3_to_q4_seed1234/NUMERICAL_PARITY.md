# Native interface checks

The Native mode is inherited unchanged from HFLM. The separate explicit likelihood implementation now matches HFLM's input exactly (`context + continuation[:-1]`) and its inference softmax dtype. This produced bit-identical direct Native likelihoods in the MCQ smoke inputs.

Cached-oracle versus full-prefill paths change FP16 GEMM shapes. Absolute summed likelihood error scales with candidate length and includes half-precision sum quantization. Oracle checks therefore require both identical chosen candidate and maximum per-token likelihood error <=0.02; all errors are saved explicitly. This is a numerical tolerance, not a task-performance threshold. No official metric is modified or rounded by our code. Native and cache-oracle benchmark results are independently reported.

GSM generation smoke requires exact equality of the first16 generated tokens' decoded text under the inherited HFLM generator and stop handling.
