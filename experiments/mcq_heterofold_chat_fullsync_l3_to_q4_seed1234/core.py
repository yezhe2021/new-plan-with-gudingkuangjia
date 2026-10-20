"""Pinned inference interface, dataset protocol and durable artifacts."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import fcntl,hashlib,importlib.metadata,json,math,random,shutil,sys,time,zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,DynamicCache
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
sys.path.insert(0,CFG['hf_root'])
from heterofold.evaluation import runtime as R,short_context as S
from heterofold.modeling import chat_affixes,load_model
from translator import NativeKVTranslator,ResidualKVAdapter
torch.set_num_threads(1)

def log(value):print(f'[{time.strftime("%F %T")}] {value}',flush=True)
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def read_jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.jsonl.tmp');tmp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));tmp.replace(path)
def identity():return sha(ROOT/'manifest.json')
def status(**value):save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**value))
def shared(state,value):
    path=ROOT.parent/'EXPERIMENT_RESULTS.md'
    record=dict(experiment=CFG['name'],time=time.strftime('%F %T'),status=state,config=CFG,
        checkpoint_selection=CFG['checkpoint_selection'],results=value,
        raw_results=str(ROOT/'evaluation/per_sample.jsonl'),log=str(ROOT/'logs/pipeline.log'))
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write('\n\n## '+CFG['name']+'\n\n```json\n'+json.dumps(record,ensure_ascii=False,indent=2)+'\n```\n');f.flush();os.fsync(f.fileno())
    save(ROOT/'shared_summary_updated.json',dict(path=str(path),status=state))
def fail(exc):
    value=dict(type=type(exc).__name__,message=str(exc))
    save(ROOT/'failure.json',value);status(status='failed',**value);shared('failed_or_incomplete',value)
def seed():
    random.seed(CFG['seed']);torch.manual_seed(CFG['seed']);torch.cuda.manual_seed_all(CFG['seed'])
def guard(phase):
    assert torch.cuda.is_available(),'CUDA unavailable'
    free=torch.cuda.mem_get_info()[0]/2**30
    assert free>=CFG['min_free_gpu_gib'][phase],f'{phase} insufficient free GPU: {free:.2f} GiB'
    log(f'GPU {phase}: {free:.2f} GiB free')
def model(family):
    guard(family)
    tok,m=load_model(CFG['models'][family],'cuda',dtype=torch.float16)
    assert m.config._attn_implementation==CFG['attention']
    return tok,m.eval().requires_grad_(False)
def runtime(m,t):
    rt=R.Runtime.__new__(R.Runtime);rt.model=m;rt.tokenizer=t;rt.prefix,rt.assistant=chat_affixes(t,thinking=False);rt.mode='native'
    return rt
def encode_prefix(t,text):return ([t.bos_token_id] if t.bos_token_id is not None else [])+list(t.encode(text,add_special_tokens=False))

def rows(split,smoke=False):
    rr=read_jsonl(ROOT/f'prepared/{split}.jsonl')
    if smoke:
        counts=Counter();selected=[]
        for r in rr:
            if counts[r['task']]<2:selected.append(r);counts[r['task']]+=1
        return selected
    return rr

def cache_path(split,row,family):return ROOT/'cache'/family/split/(row['id'].replace(':','_')+'.pt')

def save_tensor(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pt.tmp');torch.save(value,temp);temp.replace(path)

def state(split,row,family):
    x=torch.load(cache_path(split,row,family),map_location='cpu',weights_only=True)
    assert x['manifest_sha256']==identity() and x['id']==row['id']
    return x

def fresh_base():return NativeKVTranslator('full28_mlp',hidden_dim=CFG['mlp_hidden_dim'])

def base_from_best():
    x=torch.load(ROOT/'runs/stage_a/best.pt',map_location='cpu',weights_only=True)
    assert x['manifest_sha256']==identity() and x['stage']=='a'
    m=fresh_base();m.load_state_dict(x['state'],strict=True)
    return m.cuda().eval().requires_grad_(False)

def adapter_from_best():
    x=torch.load(ROOT/'runs/stage_b/best.pt',map_location='cpu',weights_only=True)
    assert x['manifest_sha256']==identity() and x['stage']=='b'
    m=ResidualKVAdapter(rank=CFG['adapter_rank']);m.load_state_dict(x['state'],strict=True)
    return m.cuda().eval().requires_grad_(False)

@torch.no_grad()
def map_base(base,source):
    kk,vv=[],[]
    for start in range(0,source['key'].shape[1],CFG['chunk_tokens']):
        with torch.amp.autocast('cuda',dtype=torch.float16):
            k,v=base(source['key'][:,start:start+CFG['chunk_tokens']][None].cuda(),source['value'][:,start:start+CFG['chunk_tokens']][None].cuda())
        kk.append(k[0]);vv.append(v[0])
    return torch.cat(kk,1),torch.cat(vv,1)

def adapted(adapter,key,value):
    with torch.amp.autocast('cuda',dtype=torch.float16):k,v,_,_=adapter(key[None],value[None])
    return k[0],v[0]

def join(anchor,key,value):return torch.cat((anchor['anchor_k'].cuda(),key),1),torch.cat((anchor['anchor_v'].cuda(),value),1)

def candidate_tensor(model,tok,row,key,value,index):
    """Differentiable equivalent of the pinned transfer_candidate_scores."""
    suffix=list(tok.encode(row['receiver_suffix'],add_special_tokens=False))
    target=R.candidate_ids(tok,row['candidates'][index])
    ids=torch.tensor([suffix+target],device='cuda')
    length=key.shape[1]
    cache=make_cache(model,key,value,torch.arange(length,device='cuda'))
    output=model(ids,past_key_values=cache,
        attention_mask=torch.ones(1,length+ids.shape[1],device='cuda',dtype=torch.long),
        position_ids=torch.arange(length,length+ids.shape[1],device='cuda')[None],use_cache=False)
    logits=output.logits[:,len(suffix)-1:len(suffix)-1+len(target)]
    labels=torch.tensor([target],device='cuda')
    result=F.log_softmax(logits.float(),-1).gather(-1,labels[...,None]).sum()
    if row['normalize']:result=result/len(target)
    return result

@torch.no_grad()
def tensor_scores(model,tok,row,key,value):
    return torch.stack([candidate_tensor(model,tok,row,key,value,i) for i in range(len(row['candidates']))])

def choice_loss(teacher,student):return F.kl_div(student.float().log_softmax(-1),teacher.float().softmax(-1),reduction='sum')

def choice_gate(a,b):
    a,b=torch.tensor(a,dtype=torch.float64),torch.tensor(b,dtype=torch.float64)
    kl=float(F.kl_div(b.log_softmax(-1),a.softmax(-1),reduction='sum'))
    result=dict(prediction_identical=int(a.argmax())==int(b.argmax()),choice_kl=kl,max_score_abs_error=float((a-b).abs().max()))
    assert result['prediction_identical'] and kl<CFG['native_cache_choice_kl_max'],result
    return result

def record(scores,gold):
    values=list(map(float,scores));assert all(math.isfinite(x) for x in values)
    pred=max(range(len(values)),key=values.__getitem__)
    return dict(scores=values,prediction=pred,correct=int(pred==gold))

@dataclass(frozen=True)
class Span:
    index: int
    start: int
    end: int


def token_spans(tok, text, expected_ids):
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    ids = bos + list(encoded["input_ids"])
    if ids != list(expected_ids):
        raise RuntimeError("Offset tokenization mismatch")
    offsets = [(0, 0)] * len(bos) + [tuple(map(int, x)) for x in encoded["offset_mapping"]]
    return [Span(i, a, b) for i, (a, b) in enumerate(offsets)]


def source_rank(source_count, target_count, target_rank):
    if source_count > target_count:
        return math.ceil((target_rank + 1) * source_count / target_count) - 1
    return target_rank * source_count // target_count


def fullsync_map(row, source_tok, qwen_tok, family):
    text = row["body"]
    source = token_spans(source_tok, text, row["encoded"][family]["body"])
    target = token_spans(qwen_tok, text, row["encoded"]["qwen"]["body"])
    char_bytes = [0]
    for character in text:
        char_bytes.append(char_bytes[-1] + len(character.encode("utf-8")))
    def byte_tokens(spans):
        return [(x.index, char_bytes[x.start], char_bytes[x.end]) for x in spans
                if x.index != 0 and x.start != x.end]
    source_b, target_b = byte_tokens(source), byte_tokens(target)
    shared = sorted(({end for _, _, end in source_b} & {end for _, _, end in target_b}) | {0})
    option_indices = list(row["encoded"]["qwen"]["option_token_indices"])
    option_set, mapping, counts = set(option_indices), {}, Counter()
    for left, right in zip(shared, shared[1:]):
        si = [index for index, _, end in source_b if left < end <= right]
        ti = [index for index, _, end in target_b if left < end <= right]
        if not si or not ti:
            continue
        touched = [index for index in ti if index in option_set]
        if not touched:
            continue
        kind = "one_to_one" if len(si) == len(ti) == 1 else "copy" if len(si) == 1 else "drop" if len(ti) == 1 else "many_to_many"
        counts[kind + "_units"] += 1
        counts[kind + "_target_tokens"] += len(touched)
        for rank, target_index in enumerate(ti):
            if target_index in option_set:
                mapping[target_index] = si[source_rank(len(si), len(ti), rank)]
    if set(mapping) != option_set:
        raise RuntimeError(f"Unmapped Qwen option tokens for {row['id']}: {len(option_set - set(mapping))}")
    return option_indices, [mapping[index] for index in option_indices], dict(counts)

def text_config(model):
    return getattr(model.config, "text_config", model.config)


def backbone(model):
    return getattr(model.model, "language_model", model.model)


def rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rope(model, x, positions, layer_index):
    base = backbone(model)
    positions = positions.to(x.device).unsqueeze(0)
    config = text_config(model)
    types = getattr(config, "layer_types", None)
    if types is not None and isinstance(getattr(base.rotary_emb, "rope_type", None), dict):
        cos, sin = base.rotary_emb(x.unsqueeze(0), positions, types[layer_index])
    else:
        cos, sin = base.rotary_emb(x.unsqueeze(0), positions)
    return x * cos[0, :, None] + rotate_half(x) * sin[0, :, None]


@torch.no_grad()
def capture(model, ids, body_length, choice_ids):
    captured_k, captured_v, handles = {}, {}, []
    base = backbone(model)
    def hook(index):
        def apply(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            key = module.k_proj(hidden).view(shape)
            if hasattr(module, "k_norm"):
                try: key = module.k_norm(key.transpose(1, 2)).transpose(1, 2)
                except (RuntimeError, ValueError): key = module.k_norm(key)
            value = module.v_proj(hidden).view(shape)
            captured_k[index], captured_v[index] = key[0, :body_length].cpu(), value[0, :body_length].cpu()
        return apply
    for index, layer in enumerate(base.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        tensor = torch.tensor([ids], device="cuda", dtype=torch.long)
        output = base(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                      position_ids=torch.arange(len(ids), device="cuda")[None], use_cache=False)
        logits = model.lm_head(output.last_hidden_state[:, -1])[0].float()
    finally:
        for handle in handles: handle.remove()
    return torch.stack([captured_k[i] for i in range(len(base.layers))]), torch.stack([captured_v[i] for i in range(len(base.layers))]), logits[torch.tensor(choice_ids, device="cuda")].cpu()


def make_cache(model, key, value, positions):
    dtype = next(model.parameters()).dtype
    key, value = key.to(dtype), value.to(dtype)
    rotated = torch.stack([apply_rope(model, layer, positions, index) for index, layer in enumerate(key)])
    data = [(rotated[layer].permute(1, 0, 2).unsqueeze(0), value[layer].permute(1, 0, 2).unsqueeze(0))
            for layer in range(key.shape[0])]
    return DynamicCache(ddp_cache_data=data, config=text_config(model))

def frame(tokenizer, family, mode, content, end=None):
    if mode == 'raw':
        rendered = content
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        full = bos + list(encoded['input_ids'])
        offsets = [(0, 0)] * len(bos) + list(encoded['offset_mapping'])
        start = 0
    else:
        kw = {'enable_thinking': False} if family == 'qwen' else {'date_string': CFG['llama_template_date']}
        messages = [{'role': 'user', 'content': content}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        direct = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kw)
        if hasattr(direct, 'keys'):
            direct = direct['input_ids']
        full = list(tokenizer.encode(rendered, add_special_tokens=False))
        assert list(direct) == full, 'Duplicate special tokens or template/token mismatch'
        assert rendered.count(content) == 1
        start = rendered.index(content)
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        offsets = list(encoded['offset_mapping'])
        if family == 'qwen':
            prefix, suffix = chat_affixes(tokenizer, thinking=False)
            assert rendered == prefix + content + suffix, 'Not the reference HeteroFold Qwen template'
            assert '<think>\n\n</think>' in suffix
    assert not any(a < start < b for a, b in offsets)
    boundaries = {int(b) - start for a, b in offsets if a >= start and start < b <= start + len(content)}
    if end is None:
        return boundaries
    assert end in boundaries
    stop = start + end
    assert not any(a < stop < b for a, b in offsets)
    ix = [i for i, (a, b) in enumerate(offsets) if start <= a < b <= stop]
    assert ix == list(range(ix[0], ix[-1] + 1))
    prefix, core, suffix = full[:ix[0]], full[ix[0]:ix[-1]+1], full[ix[-1]+1:]
    assert core == list(tokenizer.encode(content[:end], add_special_tokens=False))
    assert prefix + core + suffix == full and core and suffix
    assert 'Answer:' not in rendered
    return dict(prefix_ids=prefix, content_ids=core, body_ids=prefix + core,
                suffix_ids=suffix, full_ids=full, rendered_full=rendered,
                rendered_prefix=rendered[:start], rendered_suffix=rendered[stop:],
                native_boundary_tail=content[end:], native_prefix_length=len(prefix) + 1)


