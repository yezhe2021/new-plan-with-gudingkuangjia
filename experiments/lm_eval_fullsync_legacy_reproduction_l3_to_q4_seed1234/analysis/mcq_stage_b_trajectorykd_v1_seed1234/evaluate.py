"""Reuse unchanged official MCQ/MMLU evaluation with a separate checkpoint loader."""
import argparse
import json
from trajectory_common import (CONFIG,HERE,PARENT,PREVIOUS,digest,gpu_check,load_new_adapter,log,
                               plain,save,verify_manifest,write_rows,import_previous)


def evaluate(family):
    old=import_previous('evaluate_stage_b')
    # Explicit dependency injection; no original source or output is modified.
    for name in ('CONFIG','HERE','PARENT','digest','gpu_check','load_new_adapter','log','plain','save','verify_manifest','write_rows'):
        setattr(old,name,globals()[name])
    old.evaluate(family)


def summarize():
    summary={'protocol':CONFIG['protocol'],'selection':json.loads((HERE/'stage_b/selection.json').read_text()),
             'families':{},'pilot_subset':True}
    for family in ('mcq','mmlu','mmlu_pro'):
        path=HERE/f'evaluation/{family}/new_stage_b.json'
        if path.exists():
            result=json.loads(path.read_text())
            previous=json.loads((PREVIOUS/f'evaluation/{family}/new_stage_b.json').read_text())
            summary['families'][family]={'trajectory_stage_b':result['results'],'scheme3_stage_b':previous['results']}
    summary['completed']=len(summary['families'])==3
    save(HERE/'summary.json',summary)
    log('FINAL SUMMARY '+json.dumps(summary))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['mcq','mmlu','summarize']); args=ap.parse_args()
    if args.action=='summarize': summarize()
    else: evaluate(args.action)
