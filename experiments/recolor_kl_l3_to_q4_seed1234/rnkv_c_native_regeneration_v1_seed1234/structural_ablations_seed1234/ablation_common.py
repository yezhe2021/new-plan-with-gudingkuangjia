import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text());PARENT=Path(CFG['parent'])
sys.path.insert(0,str(PARENT))
import common as R
import model as M
import cache as K
import optimization as O
C=R.C;T=R.T;torch=R.torch

def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**v):C.save(ROOT/name,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows(split):return C.read_jsonl(ROOT/'prepared'/(split+'.jsonl'))
def paused():return (ROOT/'controls/pause_requested.json').exists()
def verify():
    R.verify();m=C.read(ROOT/'manifest.json');assert C.sha(ROOT/'config.json')==m['config_sha256']
    for n,d in m['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in m['sources'].items():assert C.sha(n)==d,n
def done(name):
    f=ROOT/name
    if not f.exists():return False
    v=C.read(f);assert v['manifest_sha256']==identity() and v['status']=='completed'
    return True
def baseline(row):
    old=PARENT if row['task'] in ['arc','hotpotqa'] else PARENT/CFG['cross_experiment']
    value=C.read(old/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json')
    expected=CFG['parent_manifest_sha256'] if old==PARENT else CFG['cross_manifest_sha256']
    assert value['manifest_sha256']==expected and value['id']==row['id'] and value['gold']==row['gold']
    return value['conditions']
