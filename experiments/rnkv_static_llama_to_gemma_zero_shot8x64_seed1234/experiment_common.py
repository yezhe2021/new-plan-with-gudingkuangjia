"""Zero-update Sender transfer of a pinned learned static Memory checkpoint."""
import importlib.util
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text())
SOURCE=Path(CFG['static_source']);GEMMA=Path(CFG['gemma_reference'])

def scoped_module(name,path,common=None):
    old=sys.modules.get('experiment_common')
    if common is not None:sys.modules['experiment_common']=common
    try:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    finally:
        if old is None:sys.modules.pop('experiment_common',None)
        else:sys.modules['experiment_common']=old

S=scoped_module('zero_shot_static_reference_common',SOURCE/'experiment_common.py')
G=scoped_module('zero_shot_gemma_reference_common',GEMMA/'experiment_common.py')
C=S.C;torch=S.torch;K=S.K
SM=scoped_module('zero_shot_frozen_static_architecture',SOURCE/'static_memory.py',S)
GE=scoped_module('zero_shot_frozen_gemma_engine',GEMMA/'engine_frozen.py',G)
GR=scoped_module('zero_shot_gemma_dynamic_reference',GEMMA/'runtime_new.py',G)
ARM='llama_static_on_gemma';DYNAMIC=G.ARM

def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**v):C.save(ROOT/name,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows():return C.read_jsonl(ROOT/'prepared/test.jsonl')
def paused():return (ROOT/'controls/pause_requested.json').exists()
def done(name):
    p=ROOT/name
    if not p.exists():return False
    v=C.read(p);assert v['manifest_sha256']==identity() and v['status']=='completed',name
    return True
def baseline_path(row):return GEMMA/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json'
def baseline(row):
    v=C.read(baseline_path(row));assert v['manifest_sha256']==G.identity() and v['id']==row['id'] and v['gold']==row['gold'] and v['question']==row['question']
    return v['conditions']
def verify():
    S.verify();G.verify()
    assert S.identity()==CFG['static_manifest_sha256'] and G.identity()==CFG['gemma_manifest_sha256']
    v=C.read(ROOT/'manifest.json');assert C.sha(ROOT/'config.json')==v['config_sha256']
    for n,d in v['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in v['sources'].items():assert C.sha(n)==d,n
def static_checkpoint():
    path=SOURCE/'runs/epoch4_step1536.pt';assert C.sha(path)==CFG['static_checkpoint_sha256']
    cp=torch.load(path,map_location='cpu',weights_only=True)
    assert cp['manifest_sha256']==S.identity() and cp['step']==1536
    return cp
def load_static(engine):
    module=SM.StaticMemory(engine,S.CFG);cp=static_checkpoint()
    module.load_state_dict(cp['state'],strict=True);module.eval().requires_grad_(False)
    assert sum(p.numel() for p in module.parameters())==409600
    assert all(torch.equal(v.detach().cpu(),cp['state'][n]) for n,v in module.state_dict().items())
    return module
