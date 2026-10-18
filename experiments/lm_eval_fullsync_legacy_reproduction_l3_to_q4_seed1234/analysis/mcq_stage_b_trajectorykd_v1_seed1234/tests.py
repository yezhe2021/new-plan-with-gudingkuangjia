"""Small semantic tests for terminal filtering, gold remapping and KD gradients."""
import unittest
import torch
from trajectory_data import acceptance,make_question,prompt
from train import projection_terms


class Invariants(unittest.TestCase):
    def test_truncated_correct_answer_is_not_a_reference(self):
        x=acceptance('The answer is (B).',[5,6],[99],'B')
        self.assertFalse(x['accepted']); self.assertEqual(x['terminal'],'length_limit')

    def test_observed_terminal_and_consistent_answer_required(self):
        self.assertTrue(acceptance('The answer is (B).',[5,99],[99],'B')['accepted'])
        self.assertTrue(acceptance('The answer is B.\nQuestion:',[5,6],[99],'B')['accepted'])
        self.assertFalse(acceptance('The answer is B. The answer is D.',[5,99],[99],'D')['accepted'])
        self.assertFalse(acceptance('The answer is B. '*8,[5,99],[99],'B')['accepted'])

    def test_permutation_and_demo_isolation(self):
        row={'id':'a','task':'openbookqa','doc':{'question_stem':'Which reflects light?','choices':{'label':['A','B','C','D'],'text':['water','coal','stone','soil']},'answerKey':'A'}}
        for split in ('train','validation','demo'):
            q=make_question(row,split)
            self.assertEqual(q['choices']['ABCDEFGHIJ'.index(q['gold'])],'water')
            with self.assertRaises(AssertionError): prompt(q,[dict(q,response=' The answer is A.')])

    def test_full_vocabulary_and_separate_terminal_gradient(self):
        torch.manual_seed(1234)
        head=torch.nn.Linear(5,11,bias=False).requires_grad_(False)
        student=torch.randn(4,5,requires_grad=True); teacher=student.detach().clone()
        ids=torch.tensor([1,2,3,10]); am=torch.tensor([0.,0.,1.,0.]); em=torch.tensor([0.,0.,0.,1.])
        terms=projection_terms(student,teacher,head,ids,am,em)
        self.assertAlmostEqual(float(terms[0]),0.,places=5)
        (terms[1]+terms[2]).backward()
        self.assertGreater(float(student.grad[2].abs().sum()),0.)
        self.assertGreater(float(student.grad[3].abs().sum()),0.)
        altered=teacher.clone(); altered[0,0]+=3
        self.assertGreater(float(projection_terms(student,altered,head,ids,am,em)[0]),0.)


if __name__=='__main__': unittest.main()
