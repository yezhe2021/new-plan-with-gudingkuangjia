import unittest,torch,importlib.util,json
from pathlib import Path
from grouped_translator import GroupedKVTranslator
from residual_translator import FrozenTranslatorWithResidual
class GroupingTests(unittest.TestCase):
    def test_unequal_groups_cover_every_layer_and_do_not_mix_tokens(self):
        torch.manual_seed(1234)
        m=GroupedKVTranslator(34,36,2,2,3,3,4,7).double()
        self.assertEqual(m.source_bounds,[0,8,17,25,34]);self.assertEqual(m.target_bounds,[0,9,18,27,36])
        k=torch.randn(1,34,3,2,3,dtype=torch.double,requires_grad=True);v=torch.randn_like(k)
        out,_=m(k,v)
        for g,(start,stop) in enumerate(zip(m.source_bounds,m.source_bounds[1:])):
            grad,=torch.autograd.grad(out[0,g*9,1,1].sum(),k,retain_graph=True)
            for layer in range(34):self.assertEqual(bool(grad[0,layer,1,0].abs().sum()>0),start<=layer<stop)
            self.assertEqual(float(grad[:,:,[0,2]].abs().sum()),0)
    def test_original_divisible_layout_is_bitwise_compatible(self):
        cfg=json.loads((Path(__file__).parent/'config.json').read_text())
        p=Path(cfg['reference_reader'])/'grouped_translator.py'
        spec=importlib.util.spec_from_file_location('old_grouped',p);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        args=(4,6,2,3,3,2,2,8)
        torch.manual_seed(7);a=old.GroupedKVTranslator(*args).double()
        torch.manual_seed(7);b=GroupedKVTranslator(*args).double()
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        for n,x in a.state_dict().items():self.assertTrue(torch.equal(x,b.state_dict()[n]))
        k=torch.randn(1,4,2,2,3,dtype=torch.double);v=torch.randn_like(k)
        for x,y in zip(a(k,v),b(k,v)):self.assertTrue(torch.equal(x,y))
    def test_production_geometry_and_residual_scope(self):
        with torch.device('meta'):
            a=GroupedKVTranslator(34,36,4,8,256,128,4,1024)
            b=FrozenTranslatorWithResidual(a,64)
        self.assertEqual(sum(p.numel() for p in a.parameters()),146800640)
        self.assertEqual(sum(p.numel() for p in b.parameters() if p.requires_grad),9437184)
    def test_invalid_empty_group(self):
        with self.assertRaises(ValueError):GroupedKVTranslator(source_layers=3,groups=4)
if __name__=='__main__':unittest.main()
