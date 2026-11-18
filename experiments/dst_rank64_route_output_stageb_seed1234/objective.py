"""Original functional supervision and sparse fixed-query attention distillation."""
from contextlib import ExitStack
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "dst_stage1_rank64_then_static_m32_a4_seed1234"
sys.path.insert(0, str(REFERENCE))
import experiment_common as G
import runtime as original
from engine import BodyReader, Engine
from score import score
from heterofold.attention import causal_mask, rotate_half, softcap_logits

C, K, torch, CFG = G.C, G.K, G.torch, G.CFG
PROBES = 4
METRICS = ("loss", "task_loss", "full_kl", "route_kl", "output_nmse")


@torch.no_grad()
def native_prefix(reader, row, native):
    for folder, pin in ((G.REFERENCE, G.N.identity()), (REFERENCE, G.identity())):
        path = folder / "native_prefix" / (row["id"] + ".json")
        if path.exists():
            value = C.read(path)
            assert value["manifest_sha256"] == pin and value["id"] == row["id"]
            return value["tokens"]
    return reader.rollout(row, native, rn=False, cap=CFG["prefix_max_new_tokens"])


@torch.no_grad()
def teacher_forward(engine, row, native, prefix, positions):
    length = native.get_seq_length()
    suffix = row["suffix_ids"]
    count = min(PROBES, len(prefix) + 1)
    take = torch.arange(len(suffix)-1, len(suffix)-1+count, device="cuda")
    absolute = length + take
    blocks = C.layers_of(engine.model)
    queries = {}

    def collect(layer):
        def capture(_module, _inputs, output):
            assert output.ndim == 4 and output.shape[0] == 1
            queries[layer] = output[0].index_select(0, take)
        return capture

    with ExitStack() as stack:
        for layer, block in enumerate(blocks):
            stack.callback(block.self_attn.q_norm.register_forward_hook(collect(layer)).remove)
        independent = C.DynamicCache(ddp_cache_data=K.pairs(native), config=engine.model.config)
        out = engine.forward(suffix + prefix, independent, length, use_cache=True)
    assert len(queries) == len(blocks) and native.get_seq_length() == length
    tables = C._rotary_tables(engine.model, absolute)
    for layer, query in queries.items():
        cos, sin = tables(layer)
        queries[layer] = query*cos[:, None] + rotate_half(query)*sin[:, None]
    select = torch.tensor([len(suffix)-1+i for i in positions], device="cuda")
    teacher = engine.model.lm_head(out.last_hidden_state[0, select]).float()
    end = length + len(suffix) - 1 + count
    pairs = [(k[:, :, :end], v[:, :, :end]) for k, v in K.pairs(out.past_key_values)]
    assert all(k.shape[2] == v.shape[2] for k,v in pairs)
    return teacher, queries, pairs, absolute


def scores(attn, query, key, mask):
    kv_heads, _, width = key.shape
    probes, heads, _ = query.shape
    grouped = query.transpose(0, 1).reshape(kv_heads, heads//kv_heads, probes, width).float()
    logits = torch.einsum("hgpd,htd->hgpt", grouped, key.float()) * float(attn.scaling)
    logits = softcap_logits(attn, logits)
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min).log_softmax(-1)


def readout(attn, log_attention, value):
    out = torch.einsum("hgpt,htd->phgd", log_attention.exp(), value.float()).flatten(1)
    return attn.o_proj(out.to(attn.o_proj.weight.dtype)).float()


def attention_losses(blocks, translated, queries, native, absolute):
    length = translated.get_seq_length()
    routes, outputs = [], []
    pairs = K.pairs(translated)
    for layer, (block, student, teacher) in enumerate(zip(blocks, pairs, native)):
        attn = block.self_attn
        mask = causal_mask(attn, absolute, teacher[0].shape[2])
        with torch.no_grad():
            keys, values = (x[0] for x in teacher)
            reference = scores(attn, queries[layer], keys, mask)
            reference_out = readout(attn, reference, values)
        mapped_k, mapped_v = (torch.cat((actual[0], target[0, :, length:]), dim=1)
                              for actual, target in zip(student, teacher))
        mapped = scores(attn, queries[layer], mapped_k, mask)
        output = readout(attn, mapped, mapped_v)
        routes.append((reference.exp() * (reference-mapped)).sum(-1).mean())
        outputs.append((output-reference_out).square().sum() / reference_out.square().sum().clamp_min(1e-8))
    return torch.stack(routes).mean(), torch.stack(outputs).mean()


def components(engine, reader, row, source, kind, epoch):
    native = engine.native_cache(row)
    cache = engine.translated(row, engine.correction, source)
    target = row["gold_tokens"]
    logits = reader.token_logits(row, cache, target, rn=False)
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor(target, device=logits.device))
    del logits
    with torch.no_grad():
        tokens = (target if kind == "gold" else native_prefix(reader, row, native) if kind == "native"
                  else reader.rollout(row, cache, rn=False, cap=CFG["prefix_max_new_tokens"]))
        positions = C.positions_for(tokens, row, epoch, validation=epoch == 0)
        prefix = tokens[:max(positions)]
        teacher, queries, kv, absolute = teacher_forward(engine, row, native, prefix, positions)
    del native
    student = reader.logits(row, cache, prefix, positions, rn=False)
    kl = C.vocab_kl(teacher, student)
    route, output = attention_losses(C.layers_of(engine.model), cache, queries, kv, absolute)
    return ce, kl, route, output


def objective(parts, weights):
    ce, kl, route, output = parts
    loss = ce + CFG["stage2"]["lambda_kl"]*kl + weights["route"]*route + weights["output"]*output
    return loss, dict(zip(METRICS, (float(x.detach()) for x in (loss, *parts))))
