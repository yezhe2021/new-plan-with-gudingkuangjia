"""One frozen backbone on CUDA at a time; native cache interfaces are explicit."""
from collections import OrderedDict
from contextlib import contextmanager
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from common import ROOT, CFG, log, append, release


def backbone(model):
    return getattr(model.model, 'language_model', model.model)


def text_config(model):
    return getattr(model.config, 'text_config', model.config)


def geometry(model):
    c = text_config(model)
    return c.num_hidden_layers, c.num_key_value_heads, c.head_dim


class Pool:
    def __init__(self):
        self.models = {}
        self.active = None
        self.tokenizers = {f: AutoTokenizer.from_pretrained(p, local_files_only=True) for f, p in CFG['models'].items()}

    def get(self, family):
        if family not in self.models:
            log(f'LOAD frozen {family} float32 on CPU')
            m = AutoModelForCausalLM.from_pretrained(CFG['models'][family], local_files_only=True,
                  dtype=torch.float32, attn_implementation='eager').eval().requires_grad_(False)
            if family == 'gemma':
                for name in ('vision_tower', 'multi_modal_projector'):
                    if hasattr(m.model, name):
                        delattr(m.model, name)
            assert geometry(m) == {'llama': (28,8,128), 'gemma': (34,4,256), 'qwen': (36,8,128)}[family]
            self.models[family] = m
        return self.models[family]

    def cpu(self):
        if self.active is not None:
            # Reloading costs I/O but avoids retaining ~40 GiB of CPU weights
            # alongside volatile checkpoint storage inside the 94-GiB cgroup.
            self.models[self.active].cpu()
            del self.models[self.active]
            self.active = None
            release()

    def gpu(self, family):
        if self.active != family:
            self.cpu()
            m = self.get(family)
            needed = sum(p.numel() * p.element_size() for p in m.parameters())
            free = torch.cuda.mem_get_info()[0]
            if free < needed + CFG['model_activation_reserve_gib'] * 2**30:
                raise RuntimeError(f'Insufficient free GPU memory for {family}; unrelated tasks not touched')
            m.cuda()
            self.active = family
            append(ROOT / 'logs/model_loading.jsonl', {'family': family, 'dtype': 'float32',
                'weight_bytes': needed, 'free_before_bytes': free, 'sequential_loading': True})
        return self.models[family]


class FeatureCache:
    """Bounded RAM only. No captured K/V tensors are written to disk."""
    def __init__(self):
        self.items = OrderedDict()
        self.bytes = 0
        self.limit = int(CFG['cpu_kv_cache_gib'] * 2**30)

    def get(self, key):
        item = self.items.pop(key, None)
        if item is not None:
            self.items[key] = item
            return item[0]

    def add(self, key, value):
        def size(v):
            if torch.is_tensor(v):
                return v.numel() * v.element_size()
            if isinstance(v, dict):
                return sum(size(x) for x in v.values())
            return 0
        n = size(value)
        while self.items and self.bytes + n > self.limit:
            _, (_, old) = self.items.popitem(last=False)
            self.bytes -= old
        if n <= self.limit:
            self.items[key] = value, n
            self.bytes += n


@torch.no_grad()
def capture(model, ids, probes=()):
    """Capture actual raw and normalized pre-RoPE keys, values, optional queries."""
    records = [{} for _ in backbone(model).layers]
    handles = []
    cfg = text_config(model)
    for index, layer in enumerate(backbone(model).layers):
        attention = layer.self_attn
        def hook(name, heads, index=index):
            def save(module, args, output):
                x = output[0] if isinstance(output, tuple) else output
                # Gemma3 normalizes [B,H,T,D]; Qwen3 normalizes [B,T,H,D].
                # Projection outputs are [B,T,H*D] for both families.
                if name in ('k','q') and x.ndim==4 and cfg.model_type=='gemma3_text':
                    assert x.shape[1:]==(heads,len(ids),cfg.head_dim)
                    x=x.transpose(1,2)
                x = x.reshape(1, len(ids), heads, cfg.head_dim)[0]
                if name == 'q':
                    x = x[list(probes)]
                records[index][name] = x.detach().cpu().contiguous()
            return save
        handles.append(attention.k_proj.register_forward_hook(hook('raw_k', cfg.num_key_value_heads)))
        handles.append(attention.v_proj.register_forward_hook(hook('v', cfg.num_key_value_heads)))
        if getattr(attention, 'k_norm', None) is not None:
            handles.append(attention.k_norm.register_forward_hook(hook('k', cfg.num_key_value_heads)))
        if probes:
            handles.append(getattr(attention, 'q_norm', attention.q_proj).register_forward_hook(hook('q', cfg.num_attention_heads)))
    try:
        inp = torch.tensor([ids], device='cuda')
        backbone(model)(input_ids=inp, attention_mask=torch.ones_like(inp),
                        position_ids=torch.arange(len(ids), device='cuda')[None], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    for row in records:
        if 'k' not in row:
            row['k'] = row['raw_k']
        assert all(torch.isfinite(x).all() for x in row.values()), 'Non-finite native capture'
    return {name: torch.stack([r[name] for r in records]) for name in records[0]}


def rotate_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


def rope(model, x, positions):
    cos, sin = backbone(model).rotary_emb(x[None], positions[None])
    return x * cos[0, :, None] + rotate_half(x) * sin[0, :, None]


def cache(model, k, v):
    positions = torch.arange(k.shape[1], device='cuda')
    rotated = torch.stack([rope(model, layer.cuda(), positions) for layer in k])
    v = v.cuda()
    pairs = [(rotated[i].permute(1,0,2)[None], v[i].permute(1,0,2)[None]) for i in range(len(k))]
    return DynamicCache(ddp_cache_data=pairs, config=text_config(model))


def run(model, ids, past=None):
    n = 0 if past is None else past.get_seq_length()
    inp = torch.tensor([ids], device='cuda')
    return backbone(model)(input_ids=inp,
        attention_mask=torch.ones((1,n+len(ids)), device='cuda', dtype=torch.long),
        position_ids=torch.arange(n,n+len(ids),device='cuda')[None],
        past_key_values=past, use_cache=True, return_dict=True)


@torch.no_grad()
def generate(model, ids, memory=None, limit=None):
    past = None if memory is None else cache(model, *memory)
    out = run(model, ids, past)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    result = []
    for _ in range(limit or CFG['max_new_tokens']):
        token = int(model.lm_head(out.last_hidden_state[:, -1]).argmax(-1))
        result.append(token)
        if token in eos:
            break
        out = run(model, [token], out.past_key_values)
    return result, 'eos' if result[-1] in eos else 'length_limit'


def move_optimizer(opt, device):
    for state in opt.state.values():
        for name, value in state.items():
            if torch.is_tensor(value) and name != 'step':
                state[name] = value.to(device)


def offload(module, opt=None):
    module.cpu()
    if opt is not None:
        move_optimizer(opt, 'cpu')
    release()


def activate(module, opt=None):
    module.cuda()
    if opt is not None:
        move_optimizer(opt, 'cuda')
        assert {id(p) for g in opt.param_groups for p in g['params']} == {id(p) for p in module.parameters()}
