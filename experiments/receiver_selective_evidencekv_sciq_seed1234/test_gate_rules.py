import unittest
from gate_rules import feature,decide,fit


class GateTests(unittest.TestCase):
    def test_inference_never_reads_gold_or_correctness(self):
        accept=dict(question_logp=-2.,choice_entropy=.4,correct=0,gold=1)
        reject=dict(question_logp=-3.,choice_entropy=.8,correct=1,gold=1)
        before=feature(accept,reject)
        accept.update(correct=1,gold=3);reject.update(correct=0,gold=3)
        self.assertEqual(before,feature(accept,reject))
        self.assertTrue(decide(before,dict(kind='question_gain',threshold=0.)))
        self.assertTrue(decide(before,dict(kind='confidence_gain',threshold=0.)))

    def test_threshold_direction_and_tie_rejection(self):
        self.assertFalse(decide(dict(question_gain=0.),dict(kind='question_gain',threshold=0.)))
        self.assertFalse(decide(dict(confidence_gain=-.1),dict(kind='confidence_gain',threshold=0.)))
        self.assertTrue(decide({},dict(kind='always_accept')))
        self.assertFalse(decide({},dict(kind='always_reject')))

    def test_calibration_can_choose_useful_packets_and_reject_all(self):
        rows=[dict(features=dict(question_gain=x,confidence_gain=x),accept=dict(correct=int(x>0)),reject=dict(correct=int(x<0))) for x in [-2.,-1.,1.,2.]]
        selected=fit(rows)
        self.assertEqual(selected['calibration_scores'][selected['selected_rule']]['correct'],4)
        rows=[dict(features=dict(question_gain=x,confidence_gain=x),accept=dict(correct=0),reject=dict(correct=1)) for x in [-2.,-1.,1.,2.]]
        selected=fit(rows);self.assertEqual(selected['selected_rule'],'always_reject')


if __name__=='__main__':unittest.main()
