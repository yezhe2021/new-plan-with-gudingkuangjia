import unittest
import numpy as np
import torch
from quantization import encode, decode, pack4, unpack4


class QuantizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.k = torch.randn(2, 19, 2, 16).half()
        self.v = torch.randn_like(self.k)
        self.pos = list(range(3, 22))
        self.mapping = [0, 0, 1, 4, 18]

    def test_nibbles_and_odd_tail(self):
        for n in [1, 15, 16, 17, 129]:
            v = np.arange(n, dtype=np.uint8) % 16
            np.testing.assert_array_equal(v, unpack4(pack4(v), n))

    def test_fp16_exact_and_mapping(self):
        p, s = encode(self.k, self.v, [], 16, 8, self.pos, self.mapping)
        k, v, meta = decode(p)
        self.assertTrue(torch.equal(k, self.k) and torch.equal(v, self.v))
        self.assertEqual(meta['target_map'], self.mapping)
        self.assertEqual(meta['source_positions'], self.pos)
        self.assertEqual(s['wire_bytes'], len(p))

    def test_equal_count_equal_actual_bytes(self):
        sizes = []
        for ix in [[0, 1], [3, 7], [17, 18]]:
            packet, stats = encode(self.k, self.v, ix, 'mixed', 8, self.pos, self.mapping)
            k, v, meta = decode(packet)
            self.assertEqual(meta['high_positions'], ix)
            self.assertTrue(torch.isfinite(k).all() and torch.isfinite(v).all())
            sizes.append(len(packet))
        self.assertEqual(len(set(sizes)), 1)

    def test_protected_values_reduce_error_without_altering_other_tokens(self):
        a, _ = encode(self.k, self.v, [], 4, 8, self.pos, self.mapping)
        b, _ = encode(self.k, self.v, [1, 8], 'mixed', 8, self.pos, self.mapping)
        k4, v4, _ = decode(a); km, vm, _ = decode(b)
        other = [i for i in range(19) if i not in [1, 8]]
        self.assertTrue(torch.equal(k4[:, other], km[:, other]))
        self.assertTrue(torch.equal(v4[:, other], vm[:, other]))
        for x, q4, qm in [(self.k, k4, km), (self.v, v4, vm)]:
            self.assertLess((x[:, [1, 8]].float()-qm[:, [1, 8]].float()).square().mean(),
                            (x[:, [1, 8]].float()-q4[:, [1, 8]].float()).square().mean())

    def test_constant_groups(self):
        x = torch.full_like(self.k, 1.25)
        p, _ = encode(x, x, [1, 8], 'mixed', 8, self.pos, self.mapping)
        k, v, _ = decode(p)
        self.assertTrue(torch.equal(k, x) and torch.equal(v, x))

    def test_truncation_rejected(self):
        p, _ = encode(self.k, self.v, [], 4, 8, self.pos, self.mapping)
        with self.assertRaises((AssertionError, ValueError)):
            decode(p[:-3])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
