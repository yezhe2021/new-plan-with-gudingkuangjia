"""Mapping regression tests without models or torch."""
import ast
import math
from pathlib import Path
import random
import unittest


def reference(sb,tb):
    result={}; bounds=sorted({0}|({b for a,b in sb}&{b for a,b in tb}))
    for l,r in zip(bounds,bounds[1:]):
        si=[i for i,(a,b) in enumerate(sb) if a<b and l<b<=r]
        ti=[i for i,(a,b) in enumerate(tb) if a<b and l<b<=r]
        for rank,i in enumerate(ti):
            j=math.ceil((rank+1)*len(si)/len(ti))-1 if len(si)>len(ti) else rank*len(si)//len(ti)
            result[i]=2+si[j]
    return [result[i] for i in range(len(tb))]


class MappingTests(unittest.TestCase):
    def test_linear_mapping_equals_original(self):
        tree=ast.parse(Path(__file__).with_name('prompt_protocol.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='paper_protocol')
        rng=random.Random(1234)
        for _ in range(100):
            ends=lambda:sorted(set(rng.sample(range(1,40),rng.randrange(1,25)))|{40})
            se,te=ends(),ends(); sb=list(zip([0]+se[:-1],se)); tb=list(zip([0]+te[:-1],te))
            def mock(tok,paper,question=None):
                spans=sb if tok=='source' else tb
                return {'ids':list(range(len(spans)+3)),'head':2,'stop':2+len(spans),
                        'spans':spans,'query_positions':[2+len(spans)]}
            scope={'math':math,'render':mock}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'mapping','exec'),scope)
            result=scope['paper_protocol']('source','target','paper',[
                {'question_id':'q1','question':'one'},{'question_id':'q2','question':'two'}])
            self.assertEqual(result['source_positions'],reference(sb,tb))
            self.assertEqual(result['body_count'],len(tb))
            self.assertEqual(len(result['queries']),2)
    def test_reader_is_query_only(self):
        tree=ast.parse(Path(__file__).with_name('runtime.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='answer_with_memory')
        self.assertEqual([a.arg for a in fn.args.args],['receiver','tok','memory','query_ids'])


if __name__=='__main__': unittest.main()
