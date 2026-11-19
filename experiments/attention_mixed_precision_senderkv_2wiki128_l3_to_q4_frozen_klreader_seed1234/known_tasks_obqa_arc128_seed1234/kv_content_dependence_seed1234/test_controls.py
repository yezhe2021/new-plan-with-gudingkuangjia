import unittest
import torch
from controls import permute_options,donor_maps,resize_pair,matched_noise,visibility


class ControlTests(unittest.TestCase):
    def test_options_content_and_gold_binding(self):
        prompt='Question: q\n\nChoices:\nA. first\nB. second\nC. third\nD. fourth\n\nAnswer with only the option label.'
        labels=list('ABCD');changed,perm,texts=permute_options(prompt,labels,1234)
        self.assertEqual(sorted(perm),list(range(4)));self.assertTrue(all(i!=p for i,p in enumerate(perm)))
        for gold in range(4):self.assertEqual(perm[perm.index(gold)],gold)
        self.assertEqual(permute_options(prompt,labels,1234)[0],changed)
        for text in texts:self.assertIn(text,changed)
        numeric=prompt
        for old,new in zip(labels,list('1234')):numeric=numeric.replace('\n'+old+'. ','\n'+new+'. ')
        numeric_changed,numeric_perm,numeric_texts=permute_options(numeric,list('1234'),1234)
        self.assertEqual(numeric_perm,perm);self.assertEqual(numeric_texts,texts)
        self.assertIn('\n1. ',numeric_changed)

    def test_donors_not_self_and_singleton_fallback(self):
        rows=[dict(id=str(i),task='arc',question_key=str(i),candidates=list('ABC') if i==0 else list('ABCD'),source_indices=list(range(20+i))) for i in range(9)]
        donors=donor_maps(rows,3,1234)
        for row in rows:
            self.assertEqual(len(donors[row['id']]),3)
            self.assertTrue(all(d['id']!=row['id'] for d in donors[row['id']]))
        self.assertFalse(donors['0'][0]['same_option_count'])

    def test_resize_preserves_pairs_order_and_identity(self):
        k=torch.arange(15).reshape(1,5,1,3).float();v=k+100
        for n in [3,5,8]:
            a,b,indices=resize_pair(k,v,n)
            self.assertTrue(torch.equal(b-a,torch.full_like(a,100)))
            self.assertEqual(indices,sorted(indices))
            if n==5:self.assertTrue(torch.equal(a,k))

    def test_noise_reproducible_and_matching_moments(self):
        torch.manual_seed(2);k=torch.randn(2,31,2,8).half();v=torch.randn_like(k)
        a,b=matched_noise(k,v,1234);c,d=matched_noise(k,v,1234)
        self.assertTrue(torch.equal(a,c) and torch.equal(b,d))
        self.assertFalse(torch.equal(a,k))
        self.assertLess(float((a.float().mean(1)-k.float().mean(1)).abs().max()),.001)
        self.assertLess(float((b.float().std(1,unbiased=False)-v.float().std(1,unbiased=False)).abs().max()),.001)

    def test_mask_only_transferred_positions(self):
        m=visibility(4,20,16,True)
        self.assertEqual(int((m==0).sum()),16)
        self.assertTrue(torch.all(m[0,:4]==1) and torch.all(m[0,20:]==1))


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
