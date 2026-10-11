import hashlib
import json
from pathlib import Path
import random
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT/'config.json').read_text())
sys.path.insert(0,str(ROOT/'base_code'))
from translator import NativeKVTranslator, ResidualKVAdapter
from protocol import make_cache, apply_rope


def save(path,obj):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'); tmp.replace(path)


def seed():
    random.seed(CFG['seed']); torch.manual_seed(CFG['seed']); torch.cuda.manual_seed_all(CFG['seed'])
    torch.set_num_threads(4)


def load_models():
    models = []
    for name,expected in [('llama',28),('qwen',36)]:
        m = AutoModelForCausalLM.from_pretrained(CFG['models'][name],local_files_only=True,
            dtype=torch.float16,attn_implementation='eager').cuda().eval().requires_grad_(False)
        assert m.config.num_hidden_layers==expected and m.config.num_key_value_heads==8 and m.config.head_dim==128
        models.append(m)
    return models


def tokenizers():
    return tuple(AutoTokenizer.from_pretrained(CFG['models'][x],local_files_only=True) for x in ('llama','qwen'))


def freeze(cache):
    return [(l.keys.detach().cpu().clone(),l.values.detach().cpu().clone()) for l in cache.layers]


def restore(receiver,state):
    from transformers import DynamicCache
    return DynamicCache(ddp_cache_data=[(k.cuda().clone(),v.cuda().clone()) for k,v in state],config=receiver.config)


def digest_state(state):
    h = hashlib.sha256()
    for pair in state:
        for x in pair: h.update(x.contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def prefill(receiver,ids,cache=None):
    n = cache.get_seq_length() if cache is not None else 0
    tensor = torch.tensor([ids],device='cuda',dtype=torch.long)
    out = receiver.model(input_ids=tensor,attention_mask=torch.ones((1,n+len(ids)),device='cuda',dtype=torch.long),
        position_ids=torch.arange(n,n+len(ids),device='cuda')[None],past_key_values=cache,use_cache=True)
    return receiver.lm_head(out.last_hidden_state[:,-1])[0].float(),out.past_key_values


@torch.no_grad()
def decode(receiver,tok,logits,cache):
    eos = receiver.generation_config.eos_token_id
    eos = {eos} if isinstance(eos,int) else set(eos)
    ids=[]
    for _ in range(CFG['max_new_tokens']):
        token = int(logits.argmax())
        if token in eos: return {'text':tok.decode(ids,skip_special_tokens=True).strip(),'ids':ids,'stop':'eos'}
        ids.append(token); logits,cache = prefill(receiver,[token],cache)
    return {'text':tok.decode(ids,skip_special_tokens=True).strip(),'ids':ids,'stop':'max_tokens'}


@torch.no_grad()
def answer_with_memory(receiver,tok,memory,query_ids):
    # Receiver API excludes evidence text and full prompt IDs.
    logits,cache = prefill(receiver,query_ids,restore(receiver,memory))
    first = logits.cpu(); result = decode(receiver,tok,logits,cache)
    return result,first


def checkpoint(path,model,epoch,step,metric,optimizer=None,scaler=None):
    data = {'protocol':CFG['protocol'],'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'epoch':epoch,'step':step,'metric':metric}
    if optimizer is not None: data['optimizer']=optimizer.state_dict()
    if scaler is not None: data['scaler']=scaler.state_dict()
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp'); torch.save(data,tmp); tmp.replace(path)


def base(path=None,old=False):
    m = NativeKVTranslator(hidden_dim=CFG['hidden_dim']).cuda()
    if path:
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if not old: assert payload['protocol']==CFG['protocol']
        m.load_state_dict(payload['state'],strict=True)
    return m


def adapter(path=None,old=False):
    m = ResidualKVAdapter(rank=CFG['rank']).cuda()
    if path:
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if not old: assert payload['protocol']==CFG['protocol']
        m.load_state_dict(payload['state'],strict=True)
    return m
