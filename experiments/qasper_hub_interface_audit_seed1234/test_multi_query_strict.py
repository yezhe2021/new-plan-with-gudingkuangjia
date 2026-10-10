"""CPU-only protocol tests; no torch/model/dataset imports required."""
import ast
import collections
import math
from pathlib import Path
import random
import unittest

source = ast.parse(Path(__file__).with_name('multi_query_strict.py').read_text(encoding='utf-8'))
names = {'sync_indices','template_parts','select_papers','encode','anomalies','read_memory'}
scope = {'math':math,'collections':collections,'random':random,'SYSTEM':'fixed',
         'paper_text':lambda p:p['body']}
exec(compile(ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names],
                        type_ignores=[]),'protocol_subset','exec'),scope)


class FakeTokenizer:
    def __init__(self,offsets): self.offsets = offsets
    def __call__(self,text,**kwargs):
        return {'input_ids':list(range(len(self.offsets))), 'offset_mapping':self.offsets}
    def encode(self,text,**kwargs): return list(range(600))
    def apply_chat_template(self,messages,**kwargs): return 'HEADER'+messages[-1]['content']+'TAIL'


class ProtocolTests(unittest.TestCase):
    def test_copy_all_body_slots(self):
        _,_,indices = scope['sync_indices'](FakeTokenizer([(0,2)]),FakeTokenizer([(0,1),(1,2)]),'ab')
        self.assertEqual(indices,[0,0])
    def test_drop_right_boundaries(self):
        _,_,indices = scope['sync_indices'](FakeTokenizer([(0,1),(1,2)]),FakeTokenizer([(0,2)]),'ab')
        self.assertEqual(indices,[1])
    def test_missing_slot_never_native_fallback(self):
        with self.assertRaises(RuntimeError):
            scope['sync_indices'](FakeTokenizer([(0,1)]),FakeTokenizer([(0,2)]),'ab')
    def test_multiquery_paper_sampling(self):
        data = {str(i):{'body':'paper','qas':[{'question_id':f'{i}_{j}'} for j in range(4)]} for i in range(20)}
        papers = scope['select_papers'](data,FakeTokenizer([]))
        self.assertEqual(len(papers),16)
        self.assertTrue(all(len(p['questions'])==3 for p in papers))
        self.assertEqual(papers,scope['select_papers'](data,FakeTokenizer([])))
    def test_reader_signature_has_no_evidence(self):
        fn = next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='read_memory')
        self.assertEqual([a.arg for a in fn.args.args],['receiver','tokenizer','state','query_ids'])
    def test_template_split(self):
        head,tail = scope['template_parts'](FakeTokenizer([]))
        self.assertEqual(head,'HEADERPaper:\n'); self.assertEqual(tail,'TAIL')


if __name__ == '__main__': unittest.main()
