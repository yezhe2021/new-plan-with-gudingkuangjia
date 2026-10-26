import unittest
import torch
from objectives import candidate_loss,clean_controls


class ObjectivesTest(unittest.TestCase):
    def test_candidate_ce_teacher_detached(self):
        teacher=torch.tensor([[9.,7.,1.,-2.,4.]],requires_grad=True)
        student=teacher.detach().clone().requires_grad_(True)
        loss,ce,aux,pred=candidate_loss(teacher,student,[1,2,4],1,.1)
        loss.backward()
        self.assertEqual(pred,0)
        self.assertLess(float(student.grad[0,2]),0)
        self.assertEqual(float(student.grad[0,0]),0)
        self.assertIsNone(teacher.grad)
        self.assertAlmostEqual(float(aux.detach()),0.,places=6)

    def test_answer_controls(self):
        self.assertEqual(clean_controls('#### 7<|im_end|>'),'#### 7')
        self.assertEqual(clean_controls('x=8\n\\boxed{8}<|im_end|>'),'x=8\n\\boxed{8}')
        self.assertEqual(clean_controls('3 < 4 and 9 > 7'),'3 < 4 and 9 > 7')
        self.assertEqual(clean_controls('#### 1,250.50<|endoftext|>'),'#### 1,250.50')

    def test_candidate_order(self):
        teacher=torch.tensor([[.2,1.,-1.,3.]])
        student=torch.tensor([[.7,-2.,1.,.1]],requires_grad=True)
        a=candidate_loss(teacher,student,[0,1,3],1,.1)[0]
        b=candidate_loss(teacher,student,[3,0,1],2,.1)[0]
        torch.testing.assert_close(a,b)


if __name__=='__main__':unittest.main()
