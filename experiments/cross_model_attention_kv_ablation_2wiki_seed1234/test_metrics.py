import unittest
import numpy as np
from metrics import projection, text_units, compare, jsd, to_token_scores, ranked_positions


class AlignmentTests(unittest.TestCase):
    def test_unicode_boundary_and_mass(self):
        spans = text_units("猫 café\nblue")
        matrix = projection([(0, 3), (3, 7), (7, 9), (9, 14), (0, 0)], 2, spans)
        self.assertTrue(np.all(matrix.sum(1) <= 1))
        self.assertAlmostEqual(matrix[0].sum(), 1 / 3)
        self.assertEqual(matrix[-1].sum(), 0)
        self.assertTrue(np.all(matrix.sum(0) > 0))

    def test_boundary_token_spreads_mass(self):
        matrix = projection([(0, 4), (4, 6)], 0, text_units("ab cd "))
        np.testing.assert_allclose(matrix[0], [.75, .25])
        np.testing.assert_allclose(matrix.sum(1), 1)

    def test_metrics_extremes_and_constant_ranks(self):
        self.assertAlmostEqual(jsd([1, 0], [0, 1]), 1)
        self.assertAlmostEqual(jsd([1, 2], [1, 2]), 0)
        result = compare(np.arange(10.), np.arange(10.), [.2])
        self.assertAlmostEqual(result["spearman"], 1)
        self.assertEqual(result["0.2"]["overlap"], 1)
        uniform = compare(np.ones(10), np.ones(10), [.2])
        self.assertIsNone(uniform["spearman"])
        self.assertAlmostEqual(uniform["0.2"]["expected_overlap_under_ties"], .2)

    def test_masks_cannot_include_non_evidence(self):
        matrix = projection([(0, 2), (2, 5), (5, 8), (8, 10)], 2, text_units("ab cd "))
        scores, eligible = to_token_scores(np.array([.8, .2]), matrix)
        positions = ranked_positions(scores, eligible, 1)
        self.assertEqual(positions, [1])
        self.assertNotIn(0, eligible)
        self.assertNotIn(3, eligible)


if __name__ == "__main__":
    unittest.main()
