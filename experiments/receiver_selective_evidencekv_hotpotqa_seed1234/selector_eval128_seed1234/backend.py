"""Frozen evidence-only Sender; native-question Receiver; packet selection pilot."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_OFFLINE']='1'
import copy,hashlib,json,sys,time,random,shutil,inspect
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
sys.path.insert(0,CFG['reference'])
import experiment as E
sys.path.insert(0,str(ROOT))
from pilot_metrics import seed_for, bootstrap
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(1)


def read(p):return json.loads(Path(p).read_text())
def save(p,x):E.save(p,x)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident():return sha(ROOT/'manifest.json')
def norm(s):return ' '.join(s.casefold().split())
def rows(split=None):
    records=[json.loads(line) for line in (ROOT/'prepared/questions.jsonl').read_text().splitlines()]
    return [r for r in records if split is None or r['split']==split]
def status(**kw):save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**kw))
def temp(row,kind):return ROOT/('temporary_'+kind)/(row['id']+'.pt')
def result_path(row):return ROOT/'evaluation'/row['split']/(row['id']+'.json')


def verify():
    E.verify()
    m=read(ROOT/'manifest.json')
    assert sha(ROOT/'config.json')==m['config_sha256']
    assert sha(ROOT/'prepared/questions.jsonl')==m['prepared_sha256']
    assert sha(ROOT/'prepared/reserved_test.jsonl')==m['reserved_test_sha256']
    assert E.identity()==m['reference_manifest_sha256']
    for name,digest in m['code'].items():assert sha(ROOT/name)==digest,name


def load_model(family):
    assert torch.cuda.mem_get_info()[0]/2**30>=CFG['free_gpu_gib']
    assert shutil.disk_usage(ROOT).free/2**30>=CFG['disk_reserve_gib']
    torch.manual_seed(CFG['seed'])
    model=E.AutoModelForCausalLM.from_pretrained(E.CFG['models'][family],dtype=torch.float16,device_map='cuda',local_files_only=True,
        attn_implementation=CFG['attention_backend']).eval().requires_grad_(False)
    assert not any('LoRA' in type(m).__name__ for m in model.modules())
    E.log(f'LOADED {family}; no Q/O Reader');return model


def load_translator():
    cfg=E.R.CFG;m=read(ROOT/'manifest.json')
    base=E.R.GroupedKVTranslator(**cfg['architecture'])
    state=torch.load(m['checkpoints']['stage_a']['path'],map_location='cpu',weights_only=True);base.load_state_dict(state['state'],strict=True);del state
    base=E.R.FrozenTranslatorWithResidual(base,cfg['residual_rank'])
    state=torch.load(m['checkpoints']['stage_b']['path'],map_location='cpu',weights_only=True);base.residual.load_state_dict(state['state'],strict=True);del state
    return base.cuda().eval().requires_grad_(False)


@torch.inference_mode()
def capture():
    verify();model=load_model('llama');records=rows()
    for i,row in enumerate(records,1):
        assert not temp(row,'source').exists()
        status(status='running',phase='capture',sample=i,total=len(records))
        ids=row['frames']['llama']['body_ids'];k,v,_=E.C.capture(model,ids,len(ids),[0]);ix=row['source_indices']
        out=temp(row,'source');out.parent.mkdir(exist_ok=True)
        torch.save(dict(key=k[:,ix].contiguous(),value=v[:,ix].contiguous(),manifest_sha256=ident()),out)
        if i%16==0:E.log(f'EVIDENCE CAPTURE {i}/{len(records)}')
    save(ROOT/'capture_completed.json',dict(manifest_sha256=ident()))


@torch.inference_mode()
def translate():
    verify();assert read(ROOT/'capture_completed.json')['manifest_sha256']==ident();base=load_translator();records=rows()
    for i,row in enumerate(records,1):
        status(status='running',phase='translate',sample=i,total=len(records))
        source=torch.load(temp(row,'source'),map_location='cpu',weights_only=True);assert source['manifest_sha256']==ident()
        parts=[]
        for start in range(0,source['key'].shape[1],64):
            with torch.amp.autocast('cuda',dtype=torch.float16):parts.append(base(source['key'][:,start:start+64][None].cuda(),source['value'][:,start:start+64][None].cuda()))
        k,v=(torch.cat([p[j][0] for p in parts],1).cpu().half().contiguous() for j in range(2))
        assert torch.isfinite(k).all() and torch.isfinite(v).all()
        out=temp(row,'mapped');out.parent.mkdir(exist_ok=True);assert not out.exists()
        torch.save(dict(key=k,value=v,manifest_sha256=ident()),out);temp(row,'source').unlink()
        if i%16==0:E.log(f'EVIDENCE TRANSLATED {i}/{len(records)}')
    save(ROOT/'translate_completed.json',dict(manifest_sha256=ident()))


def memory(row):
    data=torch.load(temp(row,'mapped'),map_location='cpu',weights_only=True);assert data['manifest_sha256']==ident()
    return data['key'],data['value']


def resized(k,v,n):
    ix=((2*torch.arange(n)+1)*k.shape[1]//(2*n)).clamp_max(k.shape[1]-1)
    return k[:,ix].contiguous(),v[:,ix].contiguous()


def corrupted(row,key,value,donor):
    dk,dv=resized(*donor,key.shape[1]);n=key.shape[1]
    ix=np.random.default_rng(seed_for(CFG['seed'],row['id'],'partial')).permutation(n)[:int(np.ceil(n*CFG['partial_corruption_fraction']))]
    pk,pv=key.clone(),value.clone();pk[:,ix]=dk[:,ix];pv[:,ix]=dv[:,ix]
    noise=[]
    for role,x in enumerate([key,value]):
        xf=x.float();g=torch.Generator().manual_seed(seed_for(CFG['seed'],row['id'],role,'gaussian'))
        z=torch.randn(x.shape,generator=g);z=(z-z.mean(1,keepdim=True))/z.std(1,unbiased=False,keepdim=True).clamp_min(1e-20)
        noise.append((xf.mean(1,keepdim=True)+xf.std(1,unbiased=False,keepdim=True)*z).half())
    return dict(clean=(key,value),wrong=(dk,dv),partial_wrong=(pk,pv),gaussian=tuple(noise)),ix.tolist()


@torch.inference_mode()
def native_cache(model,ids):
    t=torch.tensor([ids],device='cuda')
    return model.model(t,attention_mask=torch.ones_like(t),position_ids=torch.arange(len(ids),device='cuda')[None],use_cache=True).past_key_values


