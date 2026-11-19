import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from codec import FixedCodec
from quantization import encode,decode


class CodecTests(unittest.TestCase):
    def test_optimized_packets_identical_to_original(self):
        torch.manual_seed(1234);torch.set_num_threads(1)
        k=torch.randn(2,19,2,16).half();v=torch.randn_like(k)
        positions=list(range(19));mapping=[0,0,4,18]
        c=FixedCodec(k,v,8,positions,mapping)
        for bits,ix in [(16,[]),(4,[]),('mixed',[0,8]),('mixed',[17,18])]:
            expected,_=encode(k,v,ix,bits,8,positions,mapping)
            actual=c.encode(ix,bits)
            self.assertEqual(actual,expected)
            a,b,m=decode(actual)
            self.assertEqual(m['high_positions'],ix)
            self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())


if __name__=='__main__':unittest.main()
