"""Independent trainable-projection experiment on pinned frozen dual-source features."""
import sys,os,json,time,hashlib,importlib.util,random,shutil
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text());EP=Path(CFG['equal_parent'])
spec=importlib.util.spec_from_file_location('frozen_equal_reference',EP/'experiment.py');E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E)
C,U,torch=E.C,E.U,E.torch
sys.path.insert(0,str(ROOT))
from projection import ConcatProjection
CACHE=Path(CFG['feature_cache'])
def identity():return C.sha(ROOT/'manifest.json')
def save(name,obj):C.save(ROOT/name,obj)
def mark(name,**kw):save(name,dict(manifest_sha256=identity(),**kw))
def status(**kw):save('status.json',dict(time=time.strftime('%F %T'),**kw))
def done(name):
    p=ROOT/name
    if not p.exists():return False
    x=json.loads(p.read_text());assert x['manifest_sha256']==identity();return x['status']=='completed'
def rows(split):return C.read_jsonl(ROOT/'prepared'/(split+'.jsonl'))
def fname(row):return U.fname(row['gemma_row'])
def features_path(row):return CACHE/(fname(row)+'.pt')
def feature_digest(row):return C.digest(row)
def load_features(row):
    value=torch.load(features_path(row),map_location='cpu',weights_only=True)
    assert value['manifest_sha256']==identity() and value['row_sha256']==feature_digest(row)
    return [value[x].cuda() for x in ['lk','lv','gk','gv']]

def prepare():
    em=E.verify();items={};deps={}
    for split in ['train','validation']:
        lr={r['id']:r for r in C.read_jsonl(E.OLD_U_ROOT/f'prepared/{split}.jsonl')};gg=E.gemma_rows(split);out=[]
        for gr in gg:
            l=lr[gr['id']]
            for key in ['id','task','prompt','receiver_suffix','candidates','gold','normalize','cue','suffix_ids','target_indices','native_context_ids']:assert gr[key]==l[key]
            assert gr['frames']['qwen']==l['frames']['qwen']
            source=Path(l['source_ref']) if l.get('source_ref') else E.OLD_U_ROOT/'source'/(U.fname(l)+'.pt');assert source.exists()
            out.append(dict(gemma_row=gr,llama_source_path=str(source),llama_source_identity=l.get('source_identity',E.OLD_U_ID),llama_source_length=len(l['source_indices'])))
        assert dict(C.Counter(x['gemma_row']['task'] for x in out))==CFG[split+'_counts'];items[split]=out
        for p in [E.OLD_U_ROOT/f'prepared/{split}.jsonl',E.GP/f'prepared/{split}.jsonl']:deps[str(p)]=C.sha(p)
    items['test']=E.rows();assert len(items['test'])==544
    keys={s:{x['gemma_row']['question_key'] for x in rr} for s,rr in items.items()}
    assert len(keys['train'])==2048 and not keys['train']&(keys['validation']|keys['test']) and not keys['validation']&keys['test']
    for p in list(EP.glob('*.py'))+[EP/'manifest.json',EP/'config.json',EP/'summary.json',EP/'evaluation/per_sample.jsonl']:deps[str(p)]=C.sha(p)
    count=sum(len(x['gemma_row']['source_indices']) for s in ['train','validation'] for x in items[s])
    gib=count*36*1024*2*4/2**30
    CACHE.mkdir(exist_ok=True)
    assert shutil.disk_usage(CACHE).free/2**30>=gib+CFG['feature_cache_reserve_gib']
    assert shutil.disk_usage(ROOT).free/2**30>=CFG['disk_reserve_gib']+6
    m=dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={p.name:C.sha(p) for p in ROOT.glob('*.py')},dependencies=deps,equal_manifest_sha256=E.identity(),weights=em['weights'],prepared_sha256={s:C.digest(v) for s,v in items.items()},feature_cache_gib=gib,feature_cache_policy='Regenerable FP16 frozen A+B outputs on task-specific tmpfs; models/checkpoints/metrics remain on persistent disk.',all_receiver_inputs_identical=True)
    if (ROOT/'manifest.json').exists():assert json.loads((ROOT/'manifest.json').read_text())==m
    else:save('manifest.json',m)
    for s,rr in items.items():C.write_jsonl(ROOT/'prepared'/(s+'.jsonl'),rr)
    mark('prepared/completed.json',status='completed',feature_cache_gib=gib,train=2048,validation=256,test=544)
    status(status='prepared',phase='ready');print(json.dumps(dict(feature_cache_gib=gib,manifest_sha256=identity())))

def verify():
    em=E.verify();m=json.loads((ROOT/'manifest.json').read_text())
    assert m['equal_manifest_sha256']==E.identity() and C.sha(ROOT/'config.json')==m['config_sha256']
    for p,d in m['code'].items():assert C.sha(ROOT/p)==d,p
    for p,d in m['dependencies'].items():assert C.sha(p)==d,p
    for s,d in m['prepared_sha256'].items():assert C.digest(rows(s))==d
    return em

def fresh():
    m=ConcatProjection(CFG['layers'],CFG['heads'],CFG['dim'])
    assert sum(p.numel() for p in m.parameters())==CFG['expected_parameters']
    return m.cuda()

def epoch_order(epoch):
    rr=rows('train');bytask={t:[r for r in rr if r['gemma_row']['task']==t] for t in CFG['train_counts']}
    for i,ss in enumerate(bytask.values()):random.Random(CFG['seed']+epoch+i*100).shuffle(ss)
    result=[]
    for batch in range(256):
        for ss in bytask.values():
            for off in [0,1]:result.append((ss[batch*2+off],epoch>1 and (off+epoch)%2==0))
    assert len(result)==2048;return result

@torch.no_grad()
def native_prefix(model,tok,row,native):
    p=E.OLD_U_ROOT/'native_prefix'/(U.fname(row)+'.json')
    if p.exists():
        v=json.loads(p.read_text());assert v['manifest_sha256']==E.OLD_U_ID;return v
    p=ROOT/'native_prefix'/(U.fname(row)+'.json')
    if p.exists():v=json.loads(p.read_text());assert v['manifest_sha256']==identity();return v
    v=U.rollout(model,tok,row,U.clone_cache(native,model),CFG['prefix_max_new_tokens']);mark('native_prefix/'+p.name,**v);return v

def project(module,features):
    out=[]
    for start in range(0,features[0].shape[1],64):
        with torch.amp.autocast('cuda',dtype=torch.float16):out.append(module(*(x[:,start:start+64] for x in features)))
    return tuple(torch.cat([p[j] for p in out],dim=1) for j in range(2))

def example_loss(module,model,tok,item,epoch,student=False,validation=False):
    row=item['gemma_row'];features=load_features(item)
    with torch.no_grad():
        native=U.native_body(model,row)
        if student:
            k,v=project(module,features);trajectory=U.rollout(model,tok,row,U.translated_cache(model,row,native,k,v),64);del k,v
        else:trajectory=native_prefix(model,tok,row,native)
    positions=U.positions_for(trajectory['tokens'],row,epoch,validation);prefix=trajectory['tokens'][:max(positions)]
    with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16):teacher=U.suffix_logits(model,row,U.clone_cache(native,model),prefix,positions)
    k,v=project(module,features)
    with torch.amp.autocast('cuda',dtype=torch.float16):pred=U.suffix_logits(model,row,U.translated_cache(model,row,native,k,v),prefix,positions)
    loss=U.full_kl(teacher,pred);assert torch.isfinite(loss)
    return loss,dict(kl=float(loss.detach()),positions=len(positions),prefix_source='student' if student else 'native',rollout_reached_cap=int(trajectory['reached_cap']))
