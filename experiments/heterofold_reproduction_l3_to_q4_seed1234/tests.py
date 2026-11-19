import unittest
import torch
from token_alignment import align_tokens
from layer_alignment import select_sender_layers
from recolor import Moments,fit_recolor
from fold_mapper import Correction,fold_correction,apply_map
from calibration import per_head_key_error
from make_manifests import adapt


class CoreTests(unittest.TestCase):
    def test_input_only_adapters(self):
        row=adapt({'problem':'How much is 2+2?','solution':'secret gold'},'open_r1',0)
        self.assertNotIn('secret gold',row['text']);self.assertNotIn('solution',row)
        row=adapt({'question':'Who?','context':[['Title',['A sentence.']]],'answer':'secret gold'},'hotpotqa',0)
        self.assertEqual(row['text'][row['question_start']:row['question_end']],'Who?')
        self.assertNotIn('secret gold',row['text'])

    def test_token_alignment(self):
        self.assertEqual(align_tokens([(0,3),(3,16),(16,23),(23,24)],[(0,3),(3,9),(9,12),(12,16),(16,23),(23,24)]),[0,0,0,1,2,3])
        self.assertEqual(align_tokens([(0,2),(2,4)],[(0,1),(1,3),(3,4)]),[0,0,1])
        with self.assertRaises(ValueError): align_tokens([(0,1)],[(0,0)])

    def test_layers(self):
        self.assertEqual(select_sender_layers(0),(0,0,4))
        self.assertEqual(select_sender_layers(35),(23,27,27))
        self.assertTrue(all(len(select_sender_layers(l))==3 for l in range(36)))

    def test_recolor(self):
        torch.manual_seed(1);x=torch.randn(1000,12); y=x[:,:5]@torch.randn(5,5)+2
        s=Moments(12,5);s.add(x,y,.5/len(x))
        a,_,report=fit_recolor(s,1e-10,1e-12)
        self.assertLess(report['mean_max_error'],1e-8)
        self.assertLess(report['covariance_relative_error'],1e-7)
        self.assertTrue(torch.isfinite(apply_map(a,x)).all())
        singular=Moments(12,5);singular.add(x[:2],y[:2],1.)
        m,_,r=fit_recolor(singular)
        self.assertTrue(torch.isfinite(m['A']).all()); self.assertGreater(r['source']['clipped_dimensions'],0)

    def test_fold_and_gradient(self):
        torch.manual_seed(2)
        m={'A':torch.randn(12,5),'b':torch.randn(5),'rms':torch.tensor(2.)}
        c=Correction(5,2);c.G.data.normal_();x=torch.randn(30,12)
        torch.testing.assert_close(c(apply_map(m,x)),apply_map(fold_correction(m,c),x),rtol=2e-5,atol=1e-5)
        c=Correction(5,2); y=torch.randn(30,5)
        torch.testing.assert_close(c(y),y,rtol=0,atol=0)
        c(y).sum().backward();self.assertGreater(float(c.G.grad.abs().sum()),0)

    def test_head_error_not_sum(self):
        d=torch.tensor([[[1.],[-1.]]]);w=torch.ones(3,2)
        self.assertEqual(float(per_head_key_error(d,w)),1.)
        self.assertEqual(float(torch.nn.functional.linear(d.flatten(1),w).square().mean()),0.)


if __name__=='__main__':unittest.main()
