import unittest
import torch
from grouped_translator import GroupedKVTranslator
from residual_translator import CrossHeadResidual, FrozenTranslatorWithResidual


class ResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        torch.set_num_threads(1)
        self.base = GroupedKVTranslator(4, 6, 2, 3, 3, 2, 2, 8)
        self.module = FrozenTranslatorWithResidual(self.base, rank=4)
        self.k = torch.randn(1, 4, 5, 2, 3)
        self.v = torch.randn_like(self.k)

    def test_zero_initialized_residual_exactly_preserves_base(self):
        self.module.train()
        a, b = self.base(self.k, self.v)
        c, d = self.module(self.k, self.v)
        self.assertTrue(torch.equal(a, c) and torch.equal(b, d))
        self.module.assert_frozen()

    def test_real_update_changes_only_residual_and_leaves_base_bitwise_equal(self):
        before = {n: p.detach().clone() for n, p in self.base.state_dict().items()}
        residual = {n: p.detach().clone() for n, p in self.module.residual.state_dict().items()}
        optimizer = torch.optim.AdamW(self.module.residual.parameters(), lr=0.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            a, b = self.module(self.k, self.v)
            ((a - 1).square().mean() + (b + 1).square().mean()).backward()
            self.module.assert_frozen()
            optimizer.step()
        for n, p in self.base.state_dict().items():
            self.assertTrue(torch.equal(before[n], p))
        self.assertTrue(any(not torch.equal(residual[n], p) for n, p in self.module.residual.state_dict().items()))
        for branch in list(self.module.residual.k) + list(self.module.residual.v):
            self.assertGreater(branch[0].weight.grad.abs().sum().item(), 0)
            self.assertGreater(branch[-1].weight.grad.abs().sum().item(), 0)

    def test_residual_can_mix_heads_but_not_tokens_layers_or_roles(self):
        r = self.module.residual
        # The zero-up initialization is identity; activate the up projection to test connectivity.
        for branch in list(r.k) + list(r.v):
            torch.nn.init.normal_(branch[-1].weight, std=0.1)
        key = torch.randn(1, 6, 5, 3, 2, requires_grad=True)
        value = torch.randn_like(key, requires_grad=True)
        output, _ = r(key, value)
        grad, vgrad = torch.autograd.grad(output[0, 0, 2, 1].sum(), (key, value), allow_unused=True)
        self.assertGreater(grad[0, 0, 2, 0].abs().sum().item(), 0)
        self.assertEqual(grad[:, 1:].abs().sum().item(), 0)
        self.assertEqual(grad[:, :, [0, 1, 3, 4]].abs().sum().item(), 0)
        self.assertIsNone(vgrad)

    def test_production_budget_and_adapter_only_serialization(self):
        with torch.device('meta'):
            adapter = CrossHeadResidual()
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 9437184)
        state = self.module.residual.state_dict()
        other = FrozenTranslatorWithResidual(GroupedKVTranslator(4, 6, 2, 3, 3, 2, 2, 8), 4)
        other.base.load_state_dict(self.base.state_dict())
        other.residual.load_state_dict(state, strict=True)
        for a, b in zip(self.module(self.k, self.v), other(self.k, self.v)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(all(not n.startswith('base.') for n in state))


if __name__ == '__main__':
    unittest.main()
