import unittest
from types import SimpleNamespace

import torch
from torch import nn
from reader import QOReader, cpu_test
from objective import mcq_positions


class ReaderTests(unittest.TestCase):
    def test_answer_boundary_does_not_use_gold(self):
        row = dict(task='openbookqa', candidates=['A', 'B', 'C', 'D'])
        encode = lambda s: [ord(s)]
        self.assertEqual(mcq_positions(row, encode), [0])
        for gold in [0, 1, None, 'unused']:
            self.assertEqual(mcq_positions(dict(row, gold=gold), encode), [0])

    def test_rejects_multitoken_and_duplicate_labels(self):
        row = dict(task='mmlu_pro', candidates=['A', 'B'])
        with self.assertRaisesRegex(ValueError, 'Multi-token'):
            mcq_positions(row, lambda s: [1, 2])
        with self.assertRaisesRegex(ValueError, 'distinct'):
            mcq_positions(row, lambda s: [1])

    def test_no_accidental_math_boundary_objective(self):
        with self.assertRaises(ValueError):
            mcq_positions(dict(task='gsm8k', candidates=None), lambda s: [1])

    def test_projection_zero_init_disable_and_gradients(self):
        self.assertTrue(all(cpu_test().values()))

    def test_only_reader_updates_and_teacher_is_unchanged(self):
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(7, 7)
                self.o_proj = nn.Linear(7, 7)

            def forward(self, x):
                return self.o_proj(torch.tanh(self.q_proj(x)))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Module() for _ in range(2)])
                for layer in self.model.layers:
                    layer.self_attn = Attention()

            def forward(self, x):
                for layer in self.model.layers:
                    x = layer.self_attn(x)
                return x

        torch.manual_seed(1234)
        model = Model().eval().requires_grad_(False)
        base = nn.Linear(7, 7).requires_grad_(False)
        reader = QOReader(model, rank=2, alpha=2, layers=2)
        allowed = {id(p) for p in reader.parameters()}
        before = {n: p.detach().clone() for n, p in model.named_parameters() if id(p) not in allowed}
        x = torch.randn(3, 7)
        teacher = model(x).detach().clone()
        opt = torch.optim.AdamW(reader.parameters(), lr=.01, weight_decay=0)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with reader.mode(True):
                model(x).square().mean().backward()
            reader.assert_trainable(model, base)
            self.assertTrue(all(v > 0 for v in reader.gradient_norms().values()))
            opt.step()
        for n, p in model.named_parameters():
            if id(p) not in allowed:
                self.assertTrue(torch.equal(p, before[n]))
        self.assertTrue(torch.equal(model(x), teacher))
        with reader.mode(True):
            self.assertFalse(torch.equal(model(x), teacher))
            with reader.mode(False):
                self.assertTrue(torch.equal(model(x), teacher))
            self.assertTrue(all(m.enabled for m in reader.branches.values()))
        self.assertTrue(all(not m.enabled for m in reader.branches.values()))
        self.assertEqual(len(reader.state_dict()), 8)
        self.assertTrue(all(k.endswith(('.a', '.b')) for k in reader.state_dict()))


if __name__ == '__main__':
    unittest.main()
