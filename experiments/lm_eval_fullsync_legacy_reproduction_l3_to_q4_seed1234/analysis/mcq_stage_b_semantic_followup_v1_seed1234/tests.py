"""Check soft targets, teacher detachment, schedule and token-budget invariants."""
import unittest
import torch
from student_kd import full_vocab_kl,reference_partition
from followup_common import CONFIG


class Invariants(unittest.TestCase):
    def test_teacher_is_detached_and_full_vocab_mass_matters(self):
        student=torch.tensor([[2.,0.,-1.,1.]],requires_grad=True)
        teacher=student.detach().clone().requires_grad_(True)
        loss=full_vocab_kl(student,teacher);self.assertAlmostEqual(float(loss),0.,places=6)
        changed=teacher.detach().clone();changed[0,3]+=2
        loss=full_vocab_kl(student,changed.requires_grad_(True));loss.backward()
        self.assertIsNone(teacher.grad);self.assertIsNone(changed.grad)
        self.assertGreater(float(student.grad.abs().sum()),0.)

    def test_each_question_receives_both_views_once(self):
        rows=[{'id':i} for i in range(256)]
        first=reference_partition(rows,1);second=reference_partition(rows,2)
        self.assertEqual(sum(k=='reference' for k,r in first),128)
        self.assertEqual(sum(k=='student' for k,r in first),128)
        for a,b in zip(first,second):self.assertNotEqual(a[0],b[0])
        self.assertEqual(len(first)+CONFIG['mcq_replay_per_epoch'],342)


if __name__=='__main__':unittest.main()
