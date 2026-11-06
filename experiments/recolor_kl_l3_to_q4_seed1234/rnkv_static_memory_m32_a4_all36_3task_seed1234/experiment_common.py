"""Pinned completed dynamic reference; an independent static-memory run."""
import importlib.util
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text());PARENT=Path(CFG['reference'])
spec=importlib.util.spec_from_file_location('static_dynamic_reference_common',PARENT/'experiment_common.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
R=N.R;M=N.M;K=N.K;O=N.O;AR=N.AR;C=N.C;T=N.T;torch=N.torch
ORIGINAL=N.ORIGINAL;GSM=N.GSM
ARM='static_memory_all36';DYNAMIC=N.ARM

def reference_module(name,file):
    previous=sys.modules.get('experiment_common');sys.modules['experiment_common']=N
    try:
        spec=importlib.util.spec_from_file_location(name,PARENT/file)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    finally:
        if previous is None:del sys.modules['experiment_common']
        else:sys.modules['experiment_common']=previous

def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**v):C.save(ROOT/name,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows(split):return C.read_jsonl(ROOT/'prepared'/(split+'.jsonl'))
def paused():return (ROOT/'controls/pause_requested.json').exists()
def done(name):
    p=ROOT/name
    if not p.exists():return False
    v=C.read(p);assert v['manifest_sha256']==identity() and v['status']=='completed',name
    return True
def verify():
    N.verify();assert N.identity()==CFG['reference_manifest_sha256']
    v=C.read(ROOT/'manifest.json');assert C.sha(ROOT/'config.json')==v['config_sha256']
    for n,d in v['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in v['sources'].items():assert C.sha(n)==d,n
def baseline_path(row):return PARENT/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json'
def baseline(row):
    v=C.read(baseline_path(row));assert v['manifest_sha256']==N.identity() and v['id']==row['id'] and v['question']==row['question'] and v['gold']==row['gold']
    return v['conditions']
