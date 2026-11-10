"""Matched dynamic versus trained static Memory on an immutable DST base."""
import importlib.util
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text());STATIC_ROOT=Path(CFG['static_reference'])
spec=importlib.util.spec_from_file_location('dst_dynamic_trained_static_reference',STATIC_ROOT/'experiment_common.py')
F=importlib.util.module_from_spec(spec);sys.modules[spec.name]=F;spec.loader.exec_module(F)
Z=F.Z;S=F.S;D=F.D;C=F.C;torch=F.torch;K=F.K;SM=F.SM
R=F.R;M=F.M;O=F.O;AR=F.AR;T=F.T;N=F.N
SOURCE=F.SOURCE;DST=F.DST;PARENT=F.PARENT;ORIGINAL=F.ORIGINAL;GSM=F.GSM;TRANSFER=F.TRANSFER
sys.path.append(str(C.ROOT));sys.path.append(str(ORIGINAL));sys.path.append(str(ORIGINAL/'cross_dataset5x64_seed1234'))
ARM='DST_dynamic_memory';STATIC=F.ARM;OLD=F.OLD;BASE=F.BASE

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
    F.verify();assert F.identity()==CFG['static_reference_manifest_sha256'] and D.identity()==CFG['dst_manifest_sha256']
    v=C.read(ROOT/'manifest.json');assert C.sha(ROOT/'config.json')==v['config_sha256']
    for n,d in v['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in v['sources'].items():assert C.sha(n)==d,n
def baseline_path(row):return STATIC_ROOT/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json'
def baseline(row):
    v=C.read(baseline_path(row));assert v['manifest_sha256']==F.identity() and v['id']==row['id'] and v['gold']==row['gold'] and v['question']==row['question']
    return v['conditions']
same_prediction=F.same_prediction
