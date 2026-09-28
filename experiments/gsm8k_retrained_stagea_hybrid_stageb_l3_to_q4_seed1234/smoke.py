"""Two-sample end-to-end gradient and protocol smoke test."""

import gc

import torch

import experiment as E
import generation_common as C


def main():
    train_rows, _, manifest = C.prepare_manifest()
    pairs, _, tok = E.capture_pairs(train_rows[:2], manifest["train_indices"][:2], "smoke")
    module = E.translator_module()
    torch.manual_seed(C.CFG["seed"])
    stage_a = module.NativeKVTranslator("full28_mlp", hidden_dim=1024).cuda().train()
    optimizer = torch.optim.AdamW(stage_a.parameters(), lr=C.CFG["stage_a_learning_rate"])
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=1000000)
    optimizer.zero_grad(set_to_none=True)
    loss = sum(E.stage_a_backward(stage_a, pair, scaler, 2) for pair in pairs) / 2
    norm_a = E.optimizer_step(stage_a, optimizer, scaler)
    stage_a.eval().requires_grad_(False)
    cache = E.convert_pairs(stage_a, pairs, "smoke")
    qwen = C.ref.load_model("qwen").requires_grad_(False)
    token0_k, token0_v, _ = C.native_token0(qwen, tok, cache)
    torch.manual_seed(C.CFG["seed"])
    adapter = C.load_adapter(module).train()
    key, value = C.adapted_cache(adapter, cache[0], token0_k, token0_v)
    teacher, _, targets = C.native_trajectory(qwen, tok, cache[0]["source_text"], cache[0]["answer"])
    student, eos_logits, _ = C.cached_trajectory(qwen, tok, key, value, cache[0]["answer"])
    kl, ce, eos_ce = C.loss_components(student, eos_logits, teacher, targets, tok.eos_token_id)
    hybrid = C.objective_loss("hybrid", kl, ce, eos_ce); hybrid.backward()
    adapter_grads = sum(p.grad is not None and p.grad.abs().sum().item() > 0 for p in adapter.parameters())
    if adapter_grads == 0 or any(p.grad is not None for p in qwen.parameters()):
        raise RuntimeError("Smoke gradient ownership failed")
    C.save_json(C.HERE / "runs/smoke_audit.json", {
        "status": "passed", "stage_a_loss": loss, "stage_a_grad_norm": norm_a,
        "hybrid_loss": hybrid.item(), "trajectory_kl": kl.item(), "answer_ce": ce.item(),
        "eos_ce": eos_ce.item(), "adapter_parameters_with_grad": adapter_grads,
        "qwen_parameters_with_grad": sum(p.grad is not None for p in qwen.parameters()),
    })
    del qwen, adapter, stage_a, cache, pairs; gc.collect(); torch.cuda.empty_cache()
    print("GSM8K retrained Stage-A + Hybrid Stage-B smoke passed", flush=True)


if __name__ == "__main__":
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    main()
