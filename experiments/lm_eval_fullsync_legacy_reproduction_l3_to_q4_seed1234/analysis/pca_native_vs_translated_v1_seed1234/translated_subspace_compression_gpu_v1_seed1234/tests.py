"""Small mathematical checks on CUDA; no model weights or CPU eigendecomposition."""
import torch
from run import project


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Tests require CUDA; no CPU numerical fallback')
    torch.manual_seed(1234)
    q, _ = torch.linalg.qr(torch.randn(36, 8, 128, 128, device='cuda'))
    mean = torch.randn(36, 8, 128, device='cuda') * .1
    bases = {name: {'K_basis': q, 'V_basis': q,
                   'K_mean': mean if name == 'native' else mean + 999,
                   'V_mean': mean if name == 'native' else mean - 999}
             for name in ('native', 'stage_a', 'stage_b')}
    memory = (torch.randn(36, 12, 8, 128, device='cuda').half(),
              torch.randn(36, 12, 8, 128, device='cuda').half())
    for source in bases:
        full, _ = project(memory, bases, source, 'KV', 128)
        assert all(torch.allclose(a, b, atol=.002, rtol=.002) for a, b in zip(memory, full))
    native_projection, _ = project(memory, bases, 'native', 'K', 16)
    translated_projection, _ = project(memory, bases, 'stage_a', 'K', 16)
    assert torch.equal(native_projection[0], translated_projection[0]), 'Translated mean leaked into intervention'
    assert torch.equal(translated_projection[1], memory[1]), 'K-only modified V'
    assert torch.equal(translated_projection[0][:, :1], memory[0][:, :1]), 'Anchor modified'
    first = translated_projection
    twice, _ = project(first, bases, 'stage_a', 'K', 16)
    assert torch.allclose(first[0], twice[0], atol=.002, rtol=.002), 'Projection is not approximately idempotent'
    assert all(a.shape == b.shape for a, b in zip(memory, first))
    print('CUDA MATH TESTS PASSED: full rank, Native mean only, K/V isolation, anchor, slot count, idempotence')


if __name__ == '__main__':
    main()
