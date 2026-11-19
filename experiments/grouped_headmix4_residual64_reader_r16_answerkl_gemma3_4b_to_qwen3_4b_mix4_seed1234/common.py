"""Bind proven Qwen training helpers to this independent Gemma experiment."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_OFFLINE']='1'
import sys,json,time,hashlib,importlib.util,fcntl
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
U=load_module('gemma_reference_ufd',Path(CFG['ufd_parent'])/'experiment.py')
C,torch=U.C,U.torch
T=load_module('gemma_reference_reconstruction',C.ROOT/'train.py')
# Restore local priority after historical imports inserted their own directories.
sys.path.insert(0,str(ROOT))
OLD_U_ROOT,OLD_C_ROOT=U.ROOT,C.ROOT
OLD_U_ID,OLD_C_ID=U.identity(),C.identity()
old_model=C.model
old_rows=U.rows

def identity():return C.sha(ROOT/'manifest.json')
def status(**kw):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**kw))
def mark(path,**kw):C.save(path,dict(manifest_sha256=identity(),**kw))
def done(path):
    if not path.exists():return False
    r=json.loads(path.read_text());assert r['manifest_sha256']==identity()
    assert r['status']=='completed';return True

def shared(state,result):
    C.save(ROOT/'latest_result.json',dict(status=state,result=result,manifest_sha256=identity()))

def rows(split,smoke=False):
    rr=C.read_jsonl(ROOT/f'prepared/{split}.jsonl')
    if not smoke:return rr
    out=[];seen=set()
    for r in rr:
        if r['task'] not in seen:out.append(r);seen.add(r['task'])
    return out

def source_path(row):return ROOT/'source'/(U.fname(row)+'.pt')
def loaded_source(row):
    x=torch.load(source_path(row),map_location='cpu',weights_only=True)
    assert x['manifest_sha256']==identity() and x['id']==row['id']
    assert x['row_sha256']==C.digest(row)
    assert x['key'].shape==x['value'].shape==(34,len(row['source_indices']),4,256)
    assert x['key'].dtype==x['value'].dtype==torch.float16
    return x

_TARGET_MODEL=None
@torch.no_grad()
def state(split,row,family):
    if family in ['llama','gemma']:return loaded_source(row)
    assert family=='qwen'
    global _TARGET_MODEL
    if _TARGET_MODEL is None:_,_TARGET_MODEL=old_model('qwen')
    ids=row['frames']['qwen']['body_ids'];anchor=row['frames']['qwen']['native_prefix_length']
    k,v,_=C.capture(_TARGET_MODEL,ids,len(ids),[0])
    assert torch.isfinite(k).all() and torch.isfinite(v).all()
    # Exact same raw post-k_norm / pre-RoPE target and translated slots as the parent.
    return dict(key=k[:,anchor:].contiguous(),value=v[:,anchor:].contiguous())

@torch.no_grad()
def native_prefix(model,tok,row,native):
    # Receiver frame/suffix are byte-for-byte identical to the historical snapshots.
    path=OLD_U_ROOT/'native_prefix'/(U.fname(row)+'.json')
    if path.exists():
        rec=json.loads(path.read_text());assert rec['manifest_sha256']==OLD_U_ID
        assert rec.get('id',row['id'])==row['id']
        return rec
    path=ROOT/'native_prefix'/(U.fname(row)+'.json')
    if path.exists():
        rec=json.loads(path.read_text());assert rec['manifest_sha256']==identity();return rec
    rec=U.rollout(model,tok,row,U.clone_cache(native,model),U.CFG['prefix_max_new_tokens'])
    mark(path,id=row['id'],**rec);return rec

# Bind only in this child process; historical files and weights remain read-only.
U.ROOT=ROOT;U.identity=identity;U.rows=rows;U.source_path=source_path;U.loaded_source=loaded_source
C.rows=rows;C.state=state

def stage_config(stage):
    import copy
    key='reader' if stage=='reader' else 'stage_'+stage
    cfg=copy.deepcopy(CFG[key])
    cfg['base_parent']=str(ROOT);cfg['residual_parent']=str(ROOT)
    if (ROOT/'manifest.json').exists():
        cfg['base_parent_manifest_sha256']=identity();cfg['residual_parent_manifest_sha256']=identity()
    if stage in ['b','reader']:
        assert done(ROOT/'runs/stage_a/completed.json')
        cfg['base_stage_a_sha256']=C.sha(ROOT/'runs/stage_a/best.pt')
        completed=json.loads((ROOT/'runs/stage_a/completed.json').read_text())
        assert completed['steps']==512
        if (ROOT/'stage_a_selected.json').exists():
            assert json.loads((ROOT/'stage_a_selected.json').read_text())['sha256']==cfg['base_stage_a_sha256']
        else:mark(ROOT/'stage_a_selected.json',sha256=cfg['base_stage_a_sha256'])
    if stage=='reader':
        assert done(ROOT/'runs/stage_b/completed.json')
        cfg['frozen_stage_b_sha256']=C.sha(ROOT/'runs/stage_b/last.pt')
        if (ROOT/'stage_b_selected.json').exists():
            assert json.loads((ROOT/'stage_b_selected.json').read_text())['sha256']==cfg['frozen_stage_b_sha256']
        else:mark(ROOT/'stage_b_selected.json',sha256=cfg['frozen_stage_b_sha256'])
    return cfg

def verify():
    m=json.loads((ROOT/'manifest.json').read_text())
    assert C.sha(ROOT/'config.json')==m['config_sha256']
    for p,d in m['code'].items():assert C.sha(ROOT/p)==d,('Code changed',p)
    for p,d in m['dependencies'].items():assert C.sha(p)==d,('Dependency changed',p)
    for s,d in m['prepared_sha256'].items():assert C.digest(rows(s))==d
    for p,d in m['model_metadata'].items():assert C.sha(p)==d
