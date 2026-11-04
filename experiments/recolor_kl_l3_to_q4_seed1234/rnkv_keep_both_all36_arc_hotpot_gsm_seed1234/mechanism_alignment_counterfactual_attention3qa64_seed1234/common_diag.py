import importlib.util
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text());PARENT=Path(CFG['parent'])
sys.path.insert(0,str(PARENT))
import experiment_common as N
import runtime_new as RT
import evaluate_new as E
C=N.C;torch=N.torch;K=N.K
spec=importlib.util.spec_from_file_location('memory_mechanism_diagnostics',CFG['diagnostics']);D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)

def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**v):C.save(ROOT/name,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows():return C.read_jsonl(ROOT/'prepared/test.jsonl')
def paused():
    if (ROOT/'controls/pause_requested.json').exists():status(status='paused');raise SystemExit(75)
def verify():
    N.verify();v=C.read(ROOT/'manifest.json');assert C.sha(ROOT/'config.json')==v['config_sha256']
    for n,d in v['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in v['sources'].items():assert C.sha(n)==d,n
def prior(row):
    v=C.read(PARENT/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json')
    assert v['manifest_sha256']==CFG['parent_manifest_sha256'] and v['question']==row['question'] and v['gold']==row['gold']
    return v['conditions']
def load():
    engine,correction,module=RT.load(final=True);engine.mode='test';module.eval().requires_grad_(False)
    return engine,correction,module
def unchanged(engine,correction,module,snapshots):
    engine.unchanged();assert engine.student_test_native_prefills==0
    assert snapshots==[C.versions(correction),C.versions(module)]
    assert all(not p.requires_grad and p.grad is None for obj in [correction,module] for p in obj.parameters())
