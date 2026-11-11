import importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text())
spec=importlib.util.spec_from_file_location('rank64_static_reference',Path(CFG['static_reference'])/'experiment_common.py')
F=importlib.util.module_from_spec(spec);sys.modules[spec.name]=F;spec.loader.exec_module(F)
D=F.D;Z=F.Z;C=F.C;R=F.R;M=F.M;K=F.K;torch=F.torch;N=F.N;REFERENCE=N.ROOT
ORIGINAL=F.ORIGINAL;GSM=F.GSM
for path in [C.ROOT,ORIGINAL,ORIGINAL/'cross_dataset5x64_seed1234']:
    if str(path) not in sys.path:sys.path.append(str(path))
def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**v):C.save(ROOT/name,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows(split):return C.read_jsonl(ROOT/'prepared'/(split+'.jsonl'))
def paused():return (ROOT/'controls/pause_requested.json').exists()
def done(name):
    p=ROOT/name
    if not p.exists():return False
    v=C.read(p);assert v['manifest_sha256']==identity() and v['status']=='completed';return True
def verify():
    F.verify();assert F.identity()==CFG['static_reference_manifest_sha256'] and D.identity()==CFG['dst_manifest_sha256']
    v=C.read(ROOT/'manifest.json');assert C.sha(ROOT/'config.json')==v['config_sha256']
    for n,h in v['code'].items():assert C.sha(ROOT/n)==h,n
    for n,h in v['sources'].items():assert C.sha(n)==h,n
def baseline(row):
    v=C.read(D.ROOT/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json')
    f=C.read(F.ROOT/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json')
    assert v['manifest_sha256']==D.identity() and f['manifest_sha256']==F.identity()
    assert v['id']==f['id']==row['id'] and v['gold']==f['gold']==row['gold']
    return {**v['conditions'],'F_static':f['conditions'][F.ARM],'C_static':f['conditions']['C_static']}
