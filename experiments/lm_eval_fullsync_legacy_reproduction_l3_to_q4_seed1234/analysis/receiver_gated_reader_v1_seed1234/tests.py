"""Mechanism tests: exact identity, causal/cache parity, bank isolation, gradients."""
import copy
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from reader import ExternalReader
from experiment import HERE, save


def check(device, dtype):
    torch.manual_seed(1234)
    cfg = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      vocab_size=97, attention_dropout=0.)
    cfg._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype).eval().requires_grad_(False)
    ids = torch.tensor([[3, 7, 9, 11, 13, 15, 17, 19]], device=device)
    with torch.no_grad():
        cache = model(ids[:, :5], use_cache=True).past_key_values
        native = model(ids, use_cache=False).logits
        original = model(ids[:, 5:], past_key_values=copy.deepcopy(cache), use_cache=False).logits
    reader = ExternalReader(cfg, rank=4).to(device)
    reader.attach(model)
    with torch.no_grad(), reader.scope(5):
        modified = model(ids[:, 5:], past_key_values=copy.deepcopy(cache), use_cache=False).logits
        assert torch.equal(original, modified), (original - modified).abs().max()
        current = copy.deepcopy(cache)
        pieces = []
        for i in range(5, ids.shape[1]):
            output = model(ids[:, i:i+1], past_key_values=current, use_cache=True)
            current = output.past_key_values
            pieces.append(output.logits)
        sequential = torch.cat(pieces, 1)
        assert torch.allclose(sequential, modified, atol=.003 if dtype == torch.float16 else 1e-6, rtol=.003)
        changed_ids = ids[:, 5:].clone()
        changed_ids[:, -1] = 23
        changed = model(changed_ids, past_key_values=copy.deepcopy(cache), use_cache=False).logits
        assert torch.equal(changed[:, :1], modified[:, :1]), 'Future token leak'
    with reader.scope(5):
        loss = model(ids[:, 5:], past_key_values=copy.deepcopy(cache), use_cache=False).logits.float().square().mean()
        loss.backward()
    norms = {part: sum(float(p.grad.float().square().sum()) for n, p in reader.named_parameters()
                      if part in n and p.grad is not None) ** .5
             for part in ('.query.', '.output.', '.gate.')}
    assert all(v > 0 for v in norms.values()), norms
    torch.optim.SGD(reader.parameters(), lr=.1).step()
    assert all(p.grad is None for p in model.parameters())
    with torch.no_grad():
        untouched = model(ids, use_cache=False).logits
        assert torch.equal(native, untouched), 'Reader changed native inference'
        with reader.scope(5, 'masked'):
            masked = model(ids[:, 5:], past_key_values=copy.deepcopy(cache), use_cache=False).logits
            corrupted = copy.deepcopy(cache)
            for layer in corrupted.layers:
                layer.keys[:, :, 1:5] += 30
                layer.values[:, :, 1:5] -= 30
            masked_corrupt = model(ids[:, 5:], past_key_values=corrupted, use_cache=False).logits
            assert torch.equal(masked, masked_corrupt), 'Masked memory influences output'
    return {'device': device, 'dtype': str(dtype), 'zero_bitwise_identity': True,
            'causal_future_isolation': True, 'incremental_cache_parity': True,
            'native_unchanged_after_reader_update': True, 'masked_bank_isolated': True,
            'branch_grad_norms': norms}


if __name__ == '__main__':
    results = [check('cpu', torch.float32)]
    if torch.cuda.is_available():
        results.append(check('cuda', torch.float16))
    save(HERE / 'reports/tests.json', {'passed': True, 'tests': results})
    print('READER MECHANISM TESTS PASSED', results, flush=True)
