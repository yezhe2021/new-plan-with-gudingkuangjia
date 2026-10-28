"""Reuse the original full-prompt Hotpot checkpoint and evaluation implementation."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_OFFLINE']='1'
import json,hashlib,importlib.util,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
spec=importlib.util.spec_from_file_location('original_hotpot_checkpoint_experiment',Path(CFG['parent'])/'experiment.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
H,U,C,torch=P.H,P.U,P.C,P.torch

def read(p):return json.loads(Path(p).read_text())
def save(p,obj):C.save(ROOT/p,obj)
def sha(p):return C.sha(Path(p))
def identity():return sha(ROOT/'manifest.json')
def rows():return [json.loads(s) for s in (ROOT/'prepared/questions.jsonl').read_text().splitlines()]
def status(**kw):save('status.json',dict(time=time.strftime('%F %T'),**kw))
def verify():
    P.verify();m=read(ROOT/'manifest.json')
    assert P.identity()==m['parent_manifest_sha256']
    assert sha(ROOT/'config.json')==m['config_sha256']
    assert sha(ROOT/'prepared/questions.jsonl')==m['prepared_sha256']
    for name,digest in m['code'].items():assert sha(ROOT/name)==digest,name
    for cp in m['checkpoints'].values():assert sha(cp['path'])==cp['sha256']
