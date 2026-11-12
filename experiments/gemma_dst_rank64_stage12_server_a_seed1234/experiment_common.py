"""Standalone DST; immutable data/protocol reference only, no Recolor weights."""
import importlib.util
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text());REFERENCE=Path(CFG['data_reference'])
spec=importlib.util.spec_from_file_location('dst_data_reference_common',REFERENCE/'experiment_common.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
C=N.C;R=N.R;M=N.M;K=N.K;torch=N.torch
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
    N.verify();assert N.identity()==CFG['data_reference_manifest_sha256']
    v=C.read(ROOT/'manifest.json');assert C.sha(ROOT/'config.json')==v['config_sha256']
    for n,d in v['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in v['sources'].items():assert C.sha(n)==d,n
def baseline(row):
    path=ROOT/f'baseline/{row["task"]}/per_sample/{row["id"]}.json';v=C.read(path)
    assert v['manifest_sha256']==CFG['baseline_manifest_sha256'] and v['id']==row['id'] and v['gold']==row['gold'] and v['question']==row['question']
    return dict(C=v['conditions']['recolor_c'],native=v['conditions']['native'])
# Parent modules deliberately import their own train/core; restore this experiment
# to first place only after those imports have completed.
sys.path.insert(0,str(ROOT))
