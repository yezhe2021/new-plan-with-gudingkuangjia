"""Two declared protocol adaptations; receiver parameters remain frozen."""
import sys
import torch
import torch.nn.functional as F
from common import ROOT, CFG, signature
from protocol import build, sync_indices
from models import capture, FeatureCache, backbone, rope

sys.path.insert(0, str(ROOT / 'vendor'))
from heterofold.alignment import exact_end_pairs, receiver_to_sender_indices, stencil_taps
from heterofold.calibration import LayerCorrection, fold
from heterofold.attention import attention_probabilities, causal_mask
from vendor import our_llama, our_gemma


class Engine:
    def __init__(self, pool):
        self.pool = pool
        self.cache = FeatureCache()

    def pairs(self, rows, family, units=('0','1','joint')):
        jobs = [(row, unit, build(self.pool.tokenizers, row, unit)) for row in rows for unit in units]
        source_keys, target_keys = [], []
        for row, unit, proto in jobs:
            source_keys.append((family, signature(proto['source'][family]['ids'])))
            target_keys.append(('qwen', signature(proto['target']['ids'])))
        sources, targets = {}, {}
        for key, (_, _, proto) in zip(source_keys, jobs):
            value = self.cache.get(key)
            if value is None:
                source = proto['source'][family]
                value = capture(self.pool.gpu(family), source['ids'])
                value = {k: v[:, source['begin']:source['end']] for k,v in value.items()}
                self.cache.add(key, value)
            sources[key] = value
        for key, (_, _, proto) in zip(target_keys, jobs):
            value = self.cache.get(key)
            if value is None:
                target = proto['target']
                value = capture(self.pool.gpu('qwen'), target['ids'], target['probe_positions'])
                self.cache.add(key, value)
            targets[key] = value
        return [{'row': row, 'unit': unit, 'family': family, 'proto': proto, 'source': sources[sk], 'target': targets[tk]}
                for (row, unit, proto), sk, tk in zip(jobs, source_keys, target_keys)]


def our_model(family):
    return our_llama.NativeKVTranslator(hidden_dim=CFG['our_stage_a']['hidden_dim']) if family == 'llama' else our_gemma.NativeKVTranslator(hidden_dim=CFG['our_stage_a']['hidden_dim'])


def residual():
    return our_llama.ResidualKVAdapter(rank=CFG['our_stage_b']['rank'])


def our_inputs(pair):
    idx = sync_indices(pair['proto']['source'][pair['family']], pair['proto']['target'])
    return pair['source']['k'][:, idx], pair['source']['v'][:, idx]


def our_loss(model, pair, backward=False, divisor=1):
    source, target = pair['source'], pair['proto']['target']
    idx = sync_indices(pair['proto']['source'][pair['family']], target)
    n = len(idx)
    total = 0.
    for start in range(0, n, CFG['our_stage_a']['chunk_tokens']):
        stop = min(start + CFG['our_stage_a']['chunk_tokens'], n)
        selected = idx[start:stop]
        k, v = model(source['k'][:, selected][None].cuda(), source['v'][:, selected][None].cuda())
        tk = pair['target']['k'][:, target['begin']+start:target['begin']+stop][None].cuda()
        tv = pair['target']['v'][:, target['begin']+start:target['begin']+stop][None].cuda()
        loss, _ = our_llama.representation_loss(k, v, tk, tv)
        weight = (stop - start) / n
        if backward:
            (loss * weight / divisor).backward()
        total += float(loss.detach()) * weight
    return total


@torch.no_grad()
def our_map(model, pair):
    target = pair['proto']['target']
    idx = sync_indices(pair['proto']['source'][pair['family']], target)
    keys, values = [], []
    for start in range(0, len(idx), CFG['our_stage_a']['chunk_tokens']):
        selected = idx[start:start + CFG['our_stage_a']['chunk_tokens']]
        k, v = model(pair['source']['k'][:, selected][None].cuda(), pair['source']['v'][:, selected][None].cuda())
        keys.append(k[0].cpu())
        values.append(v[0].cpu())
    return torch.cat(keys, 1), torch.cat(values, 1)


def hf_features(pair, layer, role, indices):
    source = pair['source']['raw_k' if role == 'k' else 'v']
    taps = stencil_taps(source.shape[0], 36, span=1, stride=4)[layer]
    return torch.cat([source[tap, indices].flatten(1) for tap in taps], -1)


def hf_map_layer(mapping, pair, layer):
    source = receiver_to_sender_indices(pair['proto']['source'][pair['family']]['spans'], pair['proto']['target']['spans'])
    values = []
    for role in 'kv':
        weight = mapping[f'A_{role}'][layer]
        x = hf_features(pair, layer, role, source).to(weight.device)
        values.append(x / mapping[f'rms_{role}'][layer] @ weight + mapping[f'b_{role}'][layer])
    return values


@torch.no_grad()
def hf_map(mapping, pair, receiver):
    keys, values = [], []
    for layer in range(36):
        key, value = hf_map_layer(mapping, pair, layer)
        key = key.reshape(-1,8,128).to('cuda')
        key = backbone(receiver).layers[layer].self_attn.k_norm(key)
        keys.append(key.cpu())
        values.append(value.reshape(-1,8,128).cpu())
    return torch.stack(keys), torch.stack(values)


def hf_objective(receiver, mapping, corrections, pair, backward=False, divisor=1):
    """Paper K/V objectives, with only evidence positions mapped.

    Teacher question queries and non-evidence keys/values are fixed calibration
    probes. Inference never receives those native-evidence-conditioned suffix
    states: it recomputes its complete question suffix from the injected memory.
    """
    t = pair['proto']['target']
    probes = torch.tensor(t['probe_positions'], device='cuda')
    positions = torch.arange(len(t['ids']), device='cuda')
    metrics = {'routing_kl': 0., 'k_post_wo_mse': 0., 'post_wo_nmse': 0.}
    for layer, correction in enumerate(corrections):
        attention = backbone(receiver).layers[layer].self_attn
        with torch.no_grad():
            bk, bv = hf_map_layer(mapping, pair, layer)
            bk, bv = bk.cuda(), bv.cuda()
            native_k = pair['target']['k'][layer].cuda()
            native_v = pair['target']['v'][layer].cuda()
            query = rope(receiver, pair['target']['q'][layer].cuda(), probes)
            native_rotated = rope(receiver, native_k, positions)
            mask = causal_mask(attention, probes, len(t['ids']))
            teacher = attention_probabilities(attention, query, native_rotated, mask, 4)
            native_context = torch.einsum('hpt,thd->phd', teacher, native_v.repeat_interleave(4,1)).flatten(1)
            native_output = attention.o_proj(native_context)
        corrected_k = attention.k_norm(correction.key(bk).reshape(-1,8,128))
        corrected_v = correction.value(bv).reshape(-1,8,128)
        all_k = torch.cat((native_k[:t['begin']], corrected_k, native_k[t['end']:]), 0)
        all_v = torch.cat((native_v[:t['begin']], corrected_v, native_v[t['end']:]), 0)
        probability = attention_probabilities(attention, query, rope(receiver, all_k, positions), mask, 4)
        routing = (teacher * (teacher.clamp_min(1e-12).log() - probability.clamp_min(1e-20).log())).sum(-1).mean()
        error = torch.einsum('hpt,thd->phd', probability - teacher, native_v.repeat_interleave(4,1))
        wo = attention.o_proj.weight.reshape(2560,32,128).permute(1,0,2)
        k_output = torch.einsum('phd,hmd->phm', error, wo).square().mean()
        context = torch.einsum('hpt,thd->phd', probability.detach(), all_v.repeat_interleave(4,1)).flatten(1)
        mapped_output = attention.o_proj(context)
        v_output = (mapped_output - native_output).square().sum() / native_output.square().sum().clamp_min(1e-8)
        loss = routing + CFG['heterofold']['lambda_k_output'] * k_output + v_output
        assert torch.isfinite(loss), 'Non-finite HeteroFold objective'
        if backward:
            (loss / (36 * divisor)).backward()
        for name, value in [('routing_kl', routing), ('k_post_wo_mse', k_output), ('post_wo_nmse', v_output)]:
            metrics[name] += float(value.detach()) / 36
    metrics['loss'] = metrics['routing_kl'] + CFG['heterofold']['lambda_k_output'] * metrics['k_post_wo_mse'] + metrics['post_wo_nmse']
    return metrics


def prefix_memory(pair, body):
    n = pair['proto']['target']['begin']
    return torch.cat((pair['target']['k'][:, :n].cuda(), body[0]), 1), torch.cat((pair['target']['v'][:, :n].cuda(), body[1]), 1)


def suffix_hidden(receiver, proto, continuation, k=None, v=None):
    from models import cache, run
    t = proto['target']
    ids = (t['ids'] if k is None else t['suffix_ids']) + continuation[:-1]
    past = None if k is None else cache(receiver, k, v)
    return run(receiver, ids, past).last_hidden_state[0, -len(continuation):]


def token_kl(receiver, adapter, pair, base_k, base_v, continuation):
    from torch.utils.checkpoint import checkpoint
    with torch.no_grad():
        teacher = suffix_hidden(receiver, pair['proto'], continuation)
    k, v, _, _ = adapter(base_k[None].cuda(), base_v[None].cuda())
    k, v = prefix_memory(pair, (k[0], v[0]))
    student = checkpoint(lambda kk,vv: suffix_hidden(receiver, pair['proto'], continuation, kk,vv), k,v, use_reentrant=False)
    terms = student.new_zeros(())
    for start in range(0, len(continuation), CFG['our_stage_b']['projection_chunk_tokens']):
        end = start + CFG['our_stage_b']['projection_chunk_tokens']
        def project(s,t):
            slp = receiver.lm_head(s).log_softmax(-1)
            with torch.no_grad():
                tlp = receiver.lm_head(t).log_softmax(-1)
            return F.kl_div(slp, tlp, log_target=True, reduction='sum')
        terms = terms + checkpoint(project, student[start:end], teacher[start:end], use_reentrant=False)
    loss = terms / len(continuation)
    assert torch.isfinite(loss), 'Non-finite Writer stage-B KL'
    return loss
