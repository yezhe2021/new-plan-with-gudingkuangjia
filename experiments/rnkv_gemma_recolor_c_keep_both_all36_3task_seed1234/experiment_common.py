"""Single-factor all36 vs19-30 Gemma Keep-both: same frozen translator, data, initialization."""
import importlib.util
import json
import sys
import time
from pathlib import Path
from functools import lru_cache
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text());PARENT=Path(CFG['reference'])
spec=importlib.util.spec_from_file_location('gemma_partial_reference_common',PARENT/'experiment_common.py')
PG=importlib.util.module_from_spec(spec);spec.loader.exec_module(PG)
N=PG.N;R=PG.R;M=PG.M;K=PG.K;O=PG.O;C=PG.C;T=PG.T;torch=PG.torch;AR=PG.AR
ORIGINAL=PG.ORIGINAL;GSM=PG.GSM
ARM='gemma_recolor_c_memory_all36';PARTIAL='gemma_recolor_c_memory19_30'

def parent_module(name,file):
    # The reference modules keep their original globals, avoiding accidental
    # substitution of the new experiment ROOT or cache ownership identity.
    previous=sys.modules.get('experiment_common');sys.modules['experiment_common']=PG
    try:
        spec=importlib.util.spec_from_file_location(name,PARENT/file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
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
    PG.verify();v=C.read(ROOT/'manifest.json');assert PG.identity()==CFG['reference_manifest_sha256']
    assert C.sha(ROOT/'config.json')==v['config_sha256']
    for n,d in v['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in v['sources'].items():assert C.sha(n)==d,n

@lru_cache(maxsize=1)
def map_sha():
    value=PG.map_sha();assert value==CFG['recolor_sha256'];return value

def load_c():
    assert C.sha(PARENT/'runs/C/epoch4_step800.pt')==CFG['c_checkpoint_sha256']
    return PG.load_c()

def reference_path(row):return PARENT/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json'
def baseline(row):
    v=C.read(reference_path(row));assert v['manifest_sha256']==CFG['reference_manifest_sha256'] and v['id']==row['id'] and v['gold']==row['gold'] and v['question']==row['question']
    return v['conditions']
