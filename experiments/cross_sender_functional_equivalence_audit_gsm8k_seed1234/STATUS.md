# Execution status

- Implementation: complete.
- Preflight: passed (`both=61`, `llama_only=12`, `gemma_only=25`, `neither=30`).
- Smoke capture: passed for both Llama and Gemma sender states.
- Smoke audit: the initial implementation failed because Hugging Face `DynamicCache` was mutated during probing.
- Fix: probes now rebuild a fresh cache from frozen pre-RoPE K/V tensors for every forward pass.
- Formal run: pending because server A currently reports `torch.cuda.is_available() == False`, zero CUDA devices, and an NVML initialization failure.

`wait_for_gpu.sh` remains active on server A and will start the corrected pipeline once CUDA is available with at least 25 GiB free memory. No formal metrics are claimed in this revision.
