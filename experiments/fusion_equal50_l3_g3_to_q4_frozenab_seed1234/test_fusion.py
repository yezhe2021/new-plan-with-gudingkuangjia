import unittest,torch
from fusion import fuse
class FusionTests(unittest.TestCase):
    def test_equal_mean_and_symmetric(self):
        k=torch.randn(3,5,2,4,dtype=torch.float16);v=torch.randn_like(k);g=torch.randn_like(k);w=torch.randn_like(k)
        a,b=fuse(k,v,g,w);c,d=fuse(g,w,k,v)
        self.assertTrue(torch.equal(a,c) and torch.equal(b,d))
        self.assertTrue(torch.equal(a,((k.float()+g.float())/2).half()))
        self.assertTrue(torch.equal(b,((v.float()+w.float())/2).half()))
        self.assertEqual(a.shape,k.shape)
    def test_endpoints_and_identical_inputs(self):
        k=torch.randn(2,3,2,4).half();v=torch.randn_like(k);g=torch.randn_like(k);w=torch.randn_like(k)
        for t,expected in [(0,(g,w)),(1,(k,v))]:
            for a,b in zip(fuse(k,v,g,w,t),expected):self.assertTrue(torch.equal(a,b))
        for a,b in zip(fuse(k,v,k,v),(k,v)):self.assertTrue(torch.equal(a,b))
    def test_no_layer_or_token_mixing(self):
        k=torch.zeros(2,3,2,4);v=k.clone();g=k.clone();w=k.clone();g[1,2,0,3]=8
        out,_=fuse(k,v,g,w);self.assertEqual(out[1,2,0,3],4);self.assertEqual(torch.count_nonzero(out),1)
    def test_invalid_layout(self):
        k=torch.zeros(2,3,2,4)
        with self.assertRaises(ValueError):fuse(k,k,k[:,:1],k[:,:1])
        with self.assertRaises(ValueError):fuse(k,k,k,k,1.1)
if __name__=='__main__':unittest.main()
