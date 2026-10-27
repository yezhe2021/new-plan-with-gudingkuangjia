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
    E.verify();m=read(ROOT/'manifest.json')
    assert sha(ROOT/'config.json')==m['config_sha256']
    assert sha(ROOT/'prepared/questions.jsonl')==m['prepared_sha256']
    assert E.identity()==m['reference_manifest_sha256']
    for name,digest in m['code'].items():assert sha(ROOT/name)==digest,name


def load_translator():
    cfg=E.R.CFG;m=read(ROOT/'manifest.json')
    base=E.R.GroupedKVTranslator(**cfg['architecture'])
    state=torch.load(m['checkpoints']['stage_a']['path'],map_location='cpu',weights_only=True);base.load_state_dict(state['state'],strict=True);del state
    base=E.R.FrozenTranslatorWithResidual(base,cfg['residual_rank'])
    state=torch.load(m['checkpoints']['stage_b']['path'],map_location='cpu',weights_only=True);base.residual.load_state_dict(state['state'],strict=True);del state
    return base.cuda().eval().requires_grad_(False)


