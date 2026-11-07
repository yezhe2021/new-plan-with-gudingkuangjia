"""Frozen final DST Stage2 plus the unchanged Recolor-trained Static Memory."""
import importlib.util
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text())
DST=Path(CFG['dst_reference']);SOURCE=Path(CFG['static_source']);EXPORT=Path(CFG['export_reference'])

def scoped_module(name,path,common=None):
    old=sys.modules.get('experiment_common');paths=list(sys.path)
    if common is not None:sys.modules['experiment_common']=common
    sys.path.insert(0,str(path.parent))
    try:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module);return module
    finally:
        sys.path[:]=paths
        if old is None:sys.modules.pop('experiment_common',None)
        else:sys.modules['experiment_common']=old

# Import common modules before restoring the caller's path: their own verified
# globals continue pointing to the original immutable experiment directories.
S=scoped_module('dst_static_source_common',SOURCE/'experiment_common.py')
D=scoped_module('dst_static_translator_common',DST/'experiment_common.py')
C=D.C;torch=D.torch;K=D.K
SM=scoped_module('dst_static_source_architecture',SOURCE/'static_memory.py',S)
DE=scoped_module('dst_static_frozen_engine',DST/'engine.py',D)
DT=scoped_module('dst_static_frozen_translator',DST/'translator.py',D)
ARM='DST_stage2_static_memory';BASE='DST_stage2'

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
def dst_record(row):return DST/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json'
def static_record(row):return SOURCE/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json'
def baseline(row):
    a=C.read(dst_record(row));b=C.read(static_record(row))
    for v,expected in [(a,D.identity()),(b,S.identity())]:
        assert v['manifest_sha256']==expected and v['id']==row['id'] and v['gold']==row['gold'] and v['question']==row['question']
    aa,bb=a['conditions'],b['conditions']
    for k in ['correct','legacy_correct','em','f1','prediction','generation']:
        if k in aa['native']:assert aa['native'][k]==bb['native'][k],(row['id'],'Native reference mismatch',k)
        if k in aa['C']:assert aa['C'][k]==bb['C'][k],(row['id'],'C reference mismatch',k)
    return dict(C=aa['C'],C_static=bb['static_memory_all36'],DST_stage2=aa[BASE],native=aa['native'])
def verify():
    S.verify();D.verify();assert S.identity()==CFG['static_manifest_sha256'] and D.identity()==CFG['dst_manifest_sha256']
    v=C.read(ROOT/'manifest.json');assert C.sha(ROOT/'config.json')==v['config_sha256']
    for n,d in v['code'].items():assert C.sha(ROOT/n)==d,n
    for n,d in v['sources'].items():assert C.sha(n)==d,n
def load_frozen(engine):
    dp=DST/'runs/stage2/epoch4_step1536.pt';mp=SOURCE/'runs/epoch4_step1536.pt'
    assert C.sha(dp)==CFG['dst_checkpoint_sha256'] and C.sha(mp)==CFG['static_checkpoint_sha256']
    translator=DT.DepthStateTranslator(DT.DSTConfig(**D.CFG['model']),D.CFG['seed']).cuda().float()
    cp=torch.load(dp,map_location='cpu',weights_only=True)
    assert cp['manifest_sha256']==D.identity() and cp['stage']=='stage2' and cp['step']==1536 and cp['epoch']==4
    translator.load_state_dict(cp['state'],strict=True);translator.eval().requires_grad_(False)
    assert all(torch.equal(v.detach().cpu(),cp['state'][n]) for n,v in translator.state_dict().items());del cp
    module=SM.StaticMemory(engine,S.CFG);cp=torch.load(mp,map_location='cpu',weights_only=True)
    assert cp['manifest_sha256']==S.identity() and cp['step']==1536
    module.load_state_dict(cp['state'],strict=True);module.eval().requires_grad_(False)
    assert sum(p.numel() for p in module.parameters())==409600
    assert all(torch.equal(v.detach().cpu(),cp['state'][n]) for n,v in module.state_dict().items())
    assert all(not p.requires_grad for m in [translator,module] for p in m.parameters())
    return translator,module
