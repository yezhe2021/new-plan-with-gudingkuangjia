import copy
import unittest
import torch
from grouped_translator import GroupedKVTranslator


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1234)
        self.model = GroupedKVTranslator(4, 6, 2, 3, 3, 2, 2, 8).double()
        self.key = torch.randn(2, 4, 5, 2, 3, dtype=torch.float64)
        self.value = torch.randn_like(self.key)

    def test_cross_head_and_layer_connectivity(self):
        key = self.key.requires_grad_()
        output, _ = self.model(key, self.value)
        self.assertEqual(output.shape, (2, 6, 5, 3, 2))
        gradient, = torch.autograd.grad(output[0, 0, 2, 1].sum(), key)
        # Target head 1 sees source head 0 and both layers of group 0.
        self.assertGreater(gradient[0, 0, 2, 0].abs().sum().item(), 0)
        self.assertGreater(gradient[0, 1, 2, 0].abs().sum().item(), 0)
        self.assertEqual(gradient[:, 2:].abs().sum().item(), 0)
        self.assertEqual(gradient[:, :, [0, 1, 3, 4]].abs().sum().item(), 0)
        self.assertEqual(gradient[1].abs().sum().item(), 0)

    def test_roles_are_separate_and_targets_not_copied(self):
        k, v = self.model(self.key, self.value)
        k2, v2 = self.model(self.key + 1, self.value)
        self.assertTrue(torch.equal(v, v2))
        self.assertFalse(torch.equal(k, k2))
        self.assertFalse(torch.equal(k[:, 0], k[:, 1]))
        self.assertFalse(set(map(id, self.model.k_encoders.parameters())) &
                         set(map(id, self.model.v_encoders.parameters())))

    def test_chunking_forward_and_gradients(self):
        other = copy.deepcopy(self.model)
        a, b = self.model(self.key, self.value)
        (a.square().sum() + b.square().sum()).backward()
        pieces = [other(self.key[:, :, i:i+2], self.value[:, :, i:i+2]) for i in range(0, 5, 2)]
        c, d = (torch.cat([p[j] for p in pieces], dim=2) for j in range(2))
        (c.square().sum() + d.square().sum()).backward()
        torch.testing.assert_close(a, c)
        torch.testing.assert_close(b, d)
        for p, q in zip(self.model.parameters(), other.parameters()):
            torch.testing.assert_close(p.grad, q.grad)

    def test_state_roundtrip_and_production_parameter_count(self):
        other = copy.deepcopy(self.model)
        other.load_state_dict(self.model.state_dict(), strict=True)
        for a, b in zip(self.model(self.key, self.value), other(self.key, self.value)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with torch.device('meta'):
            full = GroupedKVTranslator()
        self.assertEqual(sum(p.numel() for p in full.parameters()), 134217728)

    def test_activation_checkpoint_preserves_parameter_gradients(self):
        from torch.utils.checkpoint import checkpoint
        other = copy.deepcopy(self.model)
        a, b = self.model(self.key, self.value)
        (a.square().mean() + b.square().mean()).backward()
        c, d = checkpoint(other, self.key, self.value, use_reentrant=False)
        (c.square().mean() + d.square().mean()).backward()
        for p, q in zip(self.model.parameters(), other.parameters()):
            torch.testing.assert_close(p.grad, q.grad)

    def test_invalid_layouts(self):
        with self.assertRaises(ValueError):
            GroupedKVTranslator(source_layers=27)
        with self.assertRaises(ValueError):
            self.model(self.key[:, :, :, :1], self.value[:, :, :, :1])


if __name__ == '__main__':
    unittest.main()
