"""CPU checks of supervision, option remapping and held-out data separation."""
import unittest
import torch
from data_views import audit_prepared, build, gold_index, permuted_doc, question, renderer, rng_for, text_view
from train_stage_b import token_terms


def fixture(index, split="train"):
    choices = [f"option {i}" for i in range(4)]
    context = f"{split} question {index}?"
    return {"id": f"task:{split}:{index}", "task": "openbookqa", "context": context, "fewshot": 0,
            "doc": {"question_stem": context, "choices": {"text": choices, "label": list("ABCD")},
                    "answerKey": "ABCD"[index % 4]},
            "requests": [[context, " " + answer] for answer in choices]}


class SupervisionTests(unittest.TestCase):
    def test_full_vocabulary_detects_mass_lost_outside_candidates(self):
        teacher = torch.tensor([[2., 4., 1., 0., -3.]])
        student = teacher.clone()
        student[:, :4] -= 8
        self.assertTrue(torch.allclose(teacher[:, :4].softmax(-1), student[:, :4].softmax(-1)))
        student.requires_grad_(True)
        kd, ce, _, _ = token_terms(student, teacher, [1])
        self.assertGreater(float(kd), 1.)
        (kd + 0.1 * ce).backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(float(student.grad.abs().sum()), 0.)

    def test_exact_teacher_has_zero_kl_but_nonzero_gold_ce(self):
        logits = torch.tensor([[1., 2., -1.]])
        kd, ce, _, _ = token_terms(logits, logits, [1])
        self.assertAlmostEqual(float(kd), 0., places=6)
        self.assertGreater(float(ce), 0.)

    def test_permutations_preserve_gold_content(self):
        row = fixture(2)
        for index in range(20):
            doc, permutation = permuted_doc(row, rng_for("test", index))
            self.assertEqual(doc["choices"][doc["answer"]], row["doc"]["choices"]["text"][gold_index(row)])
            self.assertEqual(sorted(permutation), list(range(4)))

    def test_view_schedule_and_train_only_demos(self):
        train, val = [fixture(i) for i in range(8)], [fixture(i, "validation") for i in range(2)]
        prepared = build(train, val)
        audit = audit_prepared(train, val, prepared)
        self.assertEqual(audit["validation"]["count"], 4)
        self.assertEqual(prepared["train_epoch_1"], build(train, val)["train_epoch_1"])
        for row in prepared["validation"]:
            self.assertTrue(all(":train:" in identifier for identifier in row["demo_ids"]))

    def test_original_text_view_and_variable_choice_layout(self):
        row = fixture(0)
        view = text_view(row)
        self.assertEqual(view["context"], row["context"])
        self.assertEqual(view["continuations"], [r[1] for r in row["requests"]])
        rendered = renderer()({"question": "q", "choices": ["a", "b", "c", "d", "e"], "answer": 4})
        self.assertIn("E. e\nAnswer:", rendered)


if __name__ == "__main__":
    unittest.main()
