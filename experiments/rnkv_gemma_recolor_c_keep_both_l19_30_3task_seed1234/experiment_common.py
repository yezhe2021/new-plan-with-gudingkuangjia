"""Independent Gemma RAW Recolor+C redo; canonical Llama data and implementations only."""
import importlib.util
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text());REFERENCE=Path(CFG['data_reference'])
spec=importlib.util.spec_from_file_location('canonical_three_task_common',REFERENCE/'experiment_common.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
R=N.R;M=N.M;K=N.K;O=N.O;C=N.C;T=N.T;torch=N.torch;AR=N.AR
ORIGINAL=N.ORIGINAL;STRUCT=N.STRUCT;GSM=N.GSM
ARM='gemma_recolor_c_memory19_30'

def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**v):C.save(ROOT/name,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows(split):return C.read_jsonl(ROOT/'prepared'/(split+'.jsonl'))
def cal_rows(split):return C.read_jsonl(ROOT/'prepared'/('calibration_'+split+'.jsonl'))
def paused():return (ROOT/'controls/pause_requested.json').exists()
def done(name):
    path=ROOT/name
    if not path.exists():return False
    v=C.read(path);assert v['manifest_sha256']==identity() and v['status']=='completed',str(path)
    return True

def verify():
    N.verify();v=C.read(ROOT/'manifest.json');assert N.identity()==CFG['data_reference_manifest_sha256']
    assert C.sha(ROOT/'config.json')==v['config_sha256']
    for n,d in v['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in v['sources'].items():assert C.sha(n)==d,n

def map_sha():
    v=C.read(ROOT/'maps/completed.json');assert v['manifest_sha256']==identity() and v['status']=='completed'
    assert C.sha(ROOT/'maps/recolor.pt')==v['recolor_sha256'];return v['recolor_sha256']

def fresh_c():return C.make_corrections('linear',64,CFG['seed'])
def load_c():
    assert done('runs/C/completed.json');cp=torch.load(ROOT/'runs/C/epoch4_step800.pt',map_location='cpu',weights_only=True)
    assert cp['manifest_sha256']==identity() and cp['recolor_sha256']==map_sha() and cp['step']==800
    assert C.sha(ROOT/'runs/C/epoch4_step800.pt')==C.read(ROOT/'runs/C/completed.json')['checkpoint_sha256']
    result=fresh_c();result.load_state_dict(cp['state'],strict=True);return result.eval().requires_grad_(False)
