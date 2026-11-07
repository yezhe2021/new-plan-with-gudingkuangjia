"""Fresh static-memory training with an immutable completed DST base."""
import importlib.util
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text())
TRANSFER=Path(CFG['transfer_reference']);spec=importlib.util.spec_from_file_location('dst_static_retrain_zero_shot_reference',TRANSFER/'experiment_common.py')
Z=importlib.util.module_from_spec(spec);sys.modules[spec.name]=Z;spec.loader.exec_module(Z)
S=Z.S;D=Z.D;C=Z.C;torch=Z.torch;K=Z.K;SM=Z.SM
R=S.R;M=S.M;O=S.O;AR=S.AR;T=S.T;N=S.N
SOURCE=Z.SOURCE;DST=Z.DST;PARENT=N.ROOT;ORIGINAL=S.ORIGINAL;GSM=S.GSM
sys.path.append(str(C.ROOT));sys.path.append(str(ORIGINAL));sys.path.append(str(ORIGINAL/'cross_dataset5x64_seed1234'))
ARM='DST_retrained_static_memory';OLD=Z.ARM;BASE=Z.BASE

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
    Z.verify();assert Z.identity()==CFG['transfer_manifest_sha256'] and D.identity()==CFG['dst_manifest_sha256']
    v=C.read(ROOT/'manifest.json');assert C.sha(ROOT/'config.json')==v['config_sha256']
    for n,d in v['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in v['sources'].items():assert C.sha(n)==d,n
def baseline_path(row):return TRANSFER/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json'
def baseline(row):
    v=C.read(baseline_path(row));assert v['manifest_sha256']==Z.identity() and v['id']==row['id'] and v['gold']==row['gold'] and v['question']==row['question']
    return v['conditions']
def same_prediction(value,reference,task,name):
    for k in ['prediction','correct','legacy_prediction','legacy_correct','em','f1','generation','token_ids','generated_tokens','reached_cap']:
        if k in reference:assert value[k]==reference[k],(task,name,k)
    if 'scores' in reference:assert len(value['scores'])==len(reference['scores']) and max(abs(a-b) for a,b in zip(value['scores'],reference['scores']))<1e-4,(task,name,'scores')
