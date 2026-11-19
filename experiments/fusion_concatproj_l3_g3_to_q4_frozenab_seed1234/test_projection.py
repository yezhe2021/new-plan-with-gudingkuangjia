import torch,unittest
from projection import ConcatProjection
class ProjectionTests(unittest.TestCase):
    def test_initialization_is_equal_mean(self):
        m=ConcatProjection(3,2,4).double();lk=torch.randn(3,5,2,4,dtype=torch.double);lv=torch.randn_like(lk);gk=torch.randn_like(lk);gv=torch.randn_like(lk)
        a,b=m(lk,lv,gk,gv)
        torch.testing.assert_close(a,(lk+gk)/2,rtol=0,atol=0);torch.testing.assert_close(b,(lv+gv)/2,rtol=0,atol=0)
    def test_only_projection_updates_with_all_branches(self):
        m=ConcatProjection(3,2,4);xs=[torch.randn(3,5,2,4) for _ in range(4)];before=[x.clone() for x in xs]
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
        for _ in range(2):
            opt.zero_grad();a,b=m(*xs);(a.square().mean()+b.square().mean()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters()));opt.step()
        self.assertTrue(all(x.grad is None and torch.equal(x,y) for x,y in zip(xs,before)))
    def test_layer_and_token_locality_with_cross_head_mixing(self):
        m=ConcatProjection(3,2,4)
        with torch.no_grad():m.k[0].weight.fill_(.1)
        lk=torch.randn(3,5,2,4,requires_grad=True);lv=torch.randn_like(lk);gk=torch.randn_like(lk);gv=torch.randn_like(lk)
        out,_=m(lk,lv,gk,gv);g,=torch.autograd.grad(out[0,2,1].sum(),lk)
        self.assertGreater(float(g[0,2,0].abs().sum()),0);self.assertEqual(float(g[1:].abs().sum()),0);self.assertEqual(float(g[:,[0,1,3,4]].abs().sum()),0)
    def test_production_count(self):
        with torch.device('meta'):m=ConcatProjection()
        self.assertEqual(sum(p.numel() for p in m.parameters()),150994944)
if __name__=='__main__':unittest.main()
