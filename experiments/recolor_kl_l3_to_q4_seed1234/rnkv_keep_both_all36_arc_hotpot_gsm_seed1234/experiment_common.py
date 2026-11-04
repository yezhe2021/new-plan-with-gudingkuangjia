import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text());ORIGINAL=Path(CFG['original_rn'])
sys.path.insert(0,str(ORIGINAL))
import common as R
import model as M
import cache as K
import optimization as O
STRUCT=ORIGINAL/'structural_ablations_seed1234';GSM=ORIGINAL/'gsm8k_frozen4x64_seed1234'
sys.path.insert(0,str(STRUCT))
import ablation_common as A
import ablation_runtime as AR
C=R.C;T=R.T;torch=R.torch
ARM='all36_arc_hotpot_gsm'

def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**v):C.save(ROOT/name,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows(split):return C.read_jsonl(ROOT/'prepared'/(split+'.jsonl'))
def paused():return (ROOT/'controls/pause_requested.json').exists()
def done(name):
    f=ROOT/name
    if not f.exists():return False
    v=C.read(f);assert v['manifest_sha256']==identity() and v['status']=='completed'
    return True
def verify():
    R.verify();A.verify();v=C.read(ROOT/'manifest.json');assert C.sha(ROOT/'config.json')==v['config_sha256']
    for n,d in v['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in v['sources'].items():assert C.sha(n)==d,n

def baseline_path(row):
    if row['task']=='gsm8k':return GSM/f'evaluation/cap512/per_sample/{row["id"]}.json'
    return STRUCT/f'arms/keep_both/evaluation/{row["task"]}/per_sample/{row["id"]}.json'
def baseline(row):
    v=C.read(baseline_path(row));expected=CFG['gsm_experiment_manifest_sha256'] if row['task']=='gsm8k' else CFG['structural_manifest_sha256']
    assert v['manifest_sha256']==expected and v['id']==row['id'] and v['gold']==row['gold'] and v['question']==row['question']
    return v['conditions']
