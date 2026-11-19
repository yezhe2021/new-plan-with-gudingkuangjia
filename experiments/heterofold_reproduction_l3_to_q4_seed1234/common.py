import json
import random
import os
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
if os.environ.get('HETEROFOLD_FP32_COMPAT')=='1':
    CFG['model_dtype']='float32'
    CFG['mapped_feature_dtype']='float32'
    CFG['hardware_deviation']='Explicit FP32 model/mapped-feature V100 compatibility run; BF16 storage and FP32 RoPE. Not paper BF16 numerics.'


def seed():
    random.seed(CFG['seed']); torch.manual_seed(CFG['seed'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False


def save(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8'); tmp.replace(path)


def store(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); tmp.replace(path)


def load_model(role):
    from transformers import AutoModelForCausalLM
    dtype=getattr(torch,CFG['model_dtype'])
    if dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 model mode unavailable; no silent precision fallback')
    model=AutoModelForCausalLM.from_pretrained(CFG[role],dtype=dtype,
        attn_implementation='eager',local_files_only=True).cuda().eval().requires_grad_(False)
    assert model.config.num_hidden_layers==(28 if role=='sender' else 36)
    assert model.config.num_key_value_heads==8 and model.model.layers[0].self_attn.head_dim==128
    return model


def tokenizers():
    from transformers import AutoTokenizer
    return tuple(AutoTokenizer.from_pretrained(CFG[r],local_files_only=True) for r in ('sender','receiver'))
