import unittest
import torch
import torch.nn.functional as F
from objectives import answer_blocks, candidate_loss, kl, clean_solution


class ObjectivesTest(unittest.TestCase):
    def test_alignment_and_complete_final_answer(self):
        for n in [1, 31, 32, 33, 65, 137]:
            suffix, targets = [500, 501, 502], list(range(n))
            blocks = list(answer_blocks(suffix, targets, 32))
            self.assertEqual(sum([b['inputs'] for b in blocks], []), suffix + targets[:-1])
            self.assertEqual(sum([b['labels'] for b in blocks], []), targets)
            self.assertAlmostEqual(sum(b['weight'] for b in blocks), 1.)
            full = suffix + targets
            consumed = 0
            for b in blocks:
                for j, target in enumerate(b['labels']):
                    self.assertEqual(full[consumed + b['offset'] + j + 1], target)
                consumed += len(b['inputs'])

    def test_candidate_gold_gradient_and_teacher_frozen(self):
        teacher = torch.tensor([[9., 7., 1., -2., 4.]], requires_grad=True)
        student = teacher.detach().clone().requires_grad_(True)
        loss, ce, aux, pred = candidate_loss(teacher, student, [1, 2, 4], 1, .1)
        loss.backward()
        self.assertEqual(pred, 0)
        self.assertGreater(float(ce), 5)
        self.assertAlmostEqual(float(aux), 0., places=6)
        self.assertLess(float(student.grad[0, 2]), 0)
        self.assertEqual(float(student.grad[0, 0]), 0)
        self.assertIsNone(teacher.grad)

    def test_token_normalization(self):
        torch.manual_seed(4)
        t, s, labels = torch.randn(73, 19), torch.randn(73, 19), torch.randint(19, (73,))
        exact = F.cross_entropy(s, labels) + .1 * kl(t, s)
        parts = sum((min(i+32,73)-i)/73 * (F.cross_entropy(s[i:i+32], labels[i:i+32]) + .1*kl(t[i:i+32],s[i:i+32])) for i in range(0,73,32))
        torch.testing.assert_close(exact, parts)

    def test_solution_keeps_answer(self):
        self.assertEqual(clean_solution('2*3 = <<2*3=6>>6.\n#### 6'), '2*3 = 6.\n#### 6')


if __name__ == '__main__':
    unittest.main()
