"""Exact original official benchmark requests; compare all completed predecessors."""
import argparse
import json
from followup_common import (HERE,PARENT,TRAJ,CONFIG,parent,digest,gpu_check,load_new_adapter,log,plain,save,verify_manifest,write_rows)


def evaluate(family):
    evaluator=parent.import_previous('evaluate_stage_b')
    for name in ('HERE','PARENT','CONFIG','digest','gpu_check','load_new_adapter','log','plain','save','verify_manifest','write_rows'):
        setattr(evaluator,name,globals()[name])
    evaluator.evaluate(family)


def summarize():
    report={'protocol':CONFIG['protocol'],'selection':json.loads((HERE/'stage_b/selection.json').read_text()),'families':{},'pilot_subset':True}
    for family in ('mcq','mmlu','mmlu_pro'):
        result=json.loads((HERE/f'evaluation/{family}/new_stage_b.json').read_text())
        previous=json.loads((TRAJ/f'evaluation/{family}/new_stage_b.json').read_text())
        scheme3=json.loads((parent.PREVIOUS/f'evaluation/{family}/new_stage_b.json').read_text())
        report['families'][family]={'student_prefix':result['results'],'trajectory_v1':previous['results'],'scheme3':scheme3['results']}
    report['completed']=True;save(HERE/'summary.json',report);log('FINAL SUMMARY SAVED')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['mcq','mmlu','summarize']);args=ap.parse_args()
    if args.action=='summarize':summarize()
    else:evaluate(args.action)
