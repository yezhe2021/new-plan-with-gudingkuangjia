import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
import sys
import json
import hashlib
import gc
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent.parent
sys.path.insert(0, str(EXP))
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from threadpoolctl import threadpool_limits
threadpool_limits(1)
import pyarrow
pyarrow.set_cpu_count(1)
pyarrow.set_io_thread_count(1)
from heterofold.modeling import load_model, load_tokenizer, layers_of, chat_affixes, kv_geometry
from heterofold.maps import load_map, sender_layers, describe
from heterofold.transfer import map_layer, source_indices
from heterofold.evaluation import short_context, longbench
from heterofold.evaluation.v100_sdpa import configure
configure()
CFG = json.loads((ROOT / 'config.json').read_text())
MODEL_ROOT = Path('/hy-tmp/yezhe/all_models/models')
SENDER = MODEL_ROOT / 'LLM-Research/Llama-3.2-3B-Instruct'
RECEIVER = MODEL_ROOT / 'Qwen/Qwen3-4B'
MAPS = EXP / 'maps/llama3_to_qwen4'
CHANNELS = ('K', 'V', 'K_raw')

def now():
    return datetime.now(timezone.utc).isoformat()

def save(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(p)

def log(message):
    print(f'[{now()}] {message}', flush=True)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def prepare(tokenizer, sender_tokenizer):
    result = {}
    prefix, _ = chat_affixes(tokenizer)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    for task in CFG['tasks']:
        if task == 'qasper':
            rows = longbench.load_jsonl(longbench.verify(EXP / 'data', longbench.FILES[task]))
        else:
            rows = short_context.load(task)
        # Match the completed evaluation's actual row indices, not a new sample draw.
        old = []
        for p in sorted((EXP / 'results/native' / task).glob('items_*.json')):
            old.extend(json.loads(p.read_text())['rows'])
        indices = sorted({r['index'] for r in old})
        assert indices == list(range(CFG['samples_per_task'])), (task, indices)
        items = []
        for index in indices:
            row = rows[index]
            if task == 'qasper':
                text = longbench.PROMPTS[task].format(context=row['context'], input=row['input'])
                text, original_tokens, truncated = longbench.truncate_prompt(text, tokenizer, longbench.SETTINGS[task][2])
            else:
                text = short_context.format_item(task, row)[0]
                original_tokens, truncated = None, False
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            body_ids = enc['input_ids']
            full_ids = tokenizer(prefix + text, add_special_tokens=False)['input_ids']
            assert full_ids == prefix_ids + body_ids, 'Chat/body token boundary changed'
            candidates = [i for i, (a, b) in enumerate(enc['offset_mapping']) if b > a]
            assert len(candidates) >= 2
            # Discrete midpoints of equal intervals; deterministic, unique and length-spanning.
            n = min(CFG['sampled_positions'], len(candidates))
            positions = [candidates[((2*j+1)*len(candidates)) // (2*n)] for j in range(n)]
            s = sender_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            source = source_indices(s['offset_mapping'], enc['offset_mapping'], 'cpu').tolist()
            items.append(dict(index=index, text=text, prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                body_ids=body_ids, full_ids=full_ids, sender_ids=s['input_ids'],
                native_positions=[len(prefix_ids)+i for i in positions],
                source_positions=[source[i] for i in positions], positions=positions,
                original_receiver_tokens=original_tokens, truncated=truncated,
                exact_end_fraction=sum(s['offset_mapping'][source[i]][1] == enc['offset_mapping'][i][1] for i in positions)/n))
        result[task] = items
    manifest = {'config': CFG, 'script_sha256': digest(__file__),
        'maps': {name: {'sha256': digest(MAPS / f'{name}.pt')} for name in ('recolor', 'heterofold')},
        'inputs': {task: [{k: v for k, v in item.items() if k not in ('text', 'body_ids', 'full_ids', 'sender_ids')}
                    for item in items] for task, items in result.items()}}
    previous = ROOT / 'manifest.json'
    if previous.exists():
        assert json.loads(previous.read_text()) == manifest, 'Manifest differs; preserve outputs and use a new run directory'
    else:
        save('manifest.json', manifest)
    return result

@torch.inference_mode()
def capture(model, ids, positions, layer_ids, normalized, chunk):
    heads, dim = kv_geometry(model)[1:]
    positions = torch.tensor(positions, device='cuda', dtype=torch.long)
    stores = {c: {} for c in CHANNELS if normalized or c != 'K'}
    # Sender has no k_norm; its K_raw/V feed the official map.
    hooks, offset = [], [0]
    def hook(channel, layer):
        def collect(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            x = x.detach().reshape(x.shape[1], heads, dim)
            mask = (positions >= offset[0]) & (positions < offset[0] + x.shape[0])
            selected = torch.where(mask)[0]
            if selected.numel():
                if layer not in stores[channel]:
                    stores[channel][layer] = torch.empty(len(positions), heads, dim, device='cuda', dtype=x.dtype)
                stores[channel][layer].index_copy_(0, selected, x.index_select(0, positions[selected] - offset[0]))
        return collect
    for layer in layer_ids:
        attn = layers_of(model)[layer].self_attn
        hooks.append(attn.k_proj.register_forward_hook(hook('K_raw', layer)))
        hooks.append(attn.v_proj.register_forward_hook(hook('V', layer)))
        if normalized:
            norm = getattr(attn, 'k_norm', None)
            hooks.append((norm if norm is not None else attn.k_proj).register_forward_hook(hook('K', layer)))
    cache = None
    try:
        for start in range(0, len(ids), chunk):
            offset[0] = start
            part = torch.tensor([ids[start:start+chunk]], device='cuda')
            output = model(part, past_key_values=cache, use_cache=True, logits_to_keep=1,
                attention_mask=torch.ones(1, start+part.shape[1], dtype=torch.long, device='cuda'),
                position_ids=torch.arange(start, start+part.shape[1], device='cuda')[None])
            cache = output.past_key_values
            del output
    finally:
        for h in hooks:
            h.remove()
        del cache
    return stores

def moments(x):
    x = x.double().permute(1, 0, 2)  # heads,tokens,dim
    return x.mean(1), x.transpose(-1,-2) @ x / x.shape[1]

def spectrum(cov):
    eig, basis = torch.linalg.eigh((cov + cov.transpose(-1,-2)) * .5)
    eig = eig.flip(-1).clamp_min(0)
    basis = basis.flip(-1)
    p = eig / eig.sum(-1, keepdim=True).clamp_min(1e-30)
    rank = {f'n{threshold}': ((p.cumsum(-1) < threshold/100).sum(-1)+1).clamp_max(p.shape[-1]) for threshold in (90,95,99)}
    rank.update(effective_rank=(-(p * p.clamp_min(1e-30).log()).sum(-1)).exp(),
                participation_rank=1/p.square().sum(-1).clamp_min(1e-30), total_variance=eig.sum(-1))
    return eig, basis, rank

@torch.inference_mode()
def smoke(sender, receiver, item):
    checks = {}
    for name, model, ids, positions, norm in [('sender', sender, item['sender_ids'], item['source_positions'], False),
                                           ('receiver', receiver, item['full_ids'], item['native_positions'], True)]:
        a = capture(model, ids, positions, range(kv_geometry(model)[0]), norm, len(ids))
        b = capture(model, ids, positions, range(kv_geometry(model)[0]), norm, 17)
        checks[name] = {}
        for c in a:
            x = torch.stack(list(a[c].values())).float().flatten(0,2)
            y = torch.stack(list(b[c].values())).float().flatten(0,2)
            cosine = float(torch.nn.functional.cosine_similarity(x,y,dim=-1).mean())
            nmse = float((x-y).square().sum()/x.square().sum().clamp_min(1e-20))
            checks[name][c] = dict(cosine=cosine, nmse=nmse)
            assert cosine > .995 and nmse < .01, checks
        del a, b
    # Known full-rank isotropic covariance and rank-one covariance.
    _, _, a = spectrum(torch.eye(128, device='cuda', dtype=torch.double)[None])
    assert int(a['n95'][0]) == 122
    x = torch.zeros(1,128,128, device='cuda', dtype=torch.double);x[:,0,0] = 1
    _, _, a = spectrum(x)
    assert int(a['n95'][0]) == 1
    save('smoke.json', {'status':'passed', 'chunked_capture':checks, 'math_tests':'passed'})
    log('SMOKE PASSED '+str(checks))

@torch.inference_mode()
def analyze(task, items, sender, receiver, mappings):
    L,H,D = kv_geometry(receiver)
    accum = {cond:{c:[torch.zeros(L,H,D,device='cuda',dtype=torch.double),
                          torch.zeros(L,H,D,D,device='cuda',dtype=torch.double)] for c in CHANNELS}
             for cond in CFG['conditions']}
    per_sample=[]
    for j,item in enumerate(items):
        save('status.json',dict(status='running',task=task,sample=j+1,total=len(items),timestamp=now()))
        src = capture(sender,item['sender_ids'], item['source_positions'], sender_layers(mappings['recolor']),False,CFG['prefill_chunk'])
        native = capture(receiver,item['full_ids'],item['native_positions'], range(L),True,CFG['prefill_chunk'])
        source = torch.arange(len(item['positions']),device='cuda')
        sk = {l:x.reshape(x.shape[0],-1) for l,x in src['K_raw'].items()}
        sv = {l:x.reshape(x.shape[0],-1) for l,x in src['V'].items()}
        record={'index':item['index'],'sampled_positions':len(source),'body_tokens':len(item['body_ids']),
                'exact_end_fraction':item['exact_end_fraction'],'conditions':{c:{} for c in CFG['conditions']}}
        for layer in range(L):
            values={'native':{c:native[c][layer] for c in CHANNELS}}
            for cond,mapping in mappings.items():
                k,v=map_layer(mapping,sk,sv,source,layer)
                k=k.reshape(-1,H,D);v=v.reshape(-1,H,D)
                knorm=getattr(layers_of(receiver)[layer].self_attn,'k_norm',None)
                values[cond]={'K_raw':k,'K':knorm(k) if knorm is not None else k,'V':v}
            for cond in CFG['conditions']:
                for c in CHANNELS:
                    x=values[cond][c]; target=values['native'][c]
                    assert torch.isfinite(x).all(),(task,j,cond,c,layer)
                    m,q=moments(x)
                    accum[cond][c][0][layer]+=m/len(items)
                    accum[cond][c][1][layer]+=q/len(items)
                    metrics=record['conditions'][cond].setdefault(c,{'rms':0.,'nmse_to_native':0.,'cosine_to_native':0.})
                    xf,yf=x.float(),target.float()
                    metrics['rms']+=float(xf.square().mean().sqrt())/L
                    metrics['nmse_to_native']+=float((xf-yf).square().sum()/yf.square().sum().clamp_min(1e-20))/L
                    metrics['cosine_to_native']+=float(torch.nn.functional.cosine_similarity(xf,yf,dim=-1).mean())/L
            del values
        per_sample.append(record)
        save(f'results/{task}/per_sample.json',per_sample)
        del src,native,sk,sv
        torch.cuda.empty_cache()
        log(f'{task} captured {j+1}/{len(items)}; body={len(item["body_ids"])} sampled={len(source)}')
    results={}; summary={}; curves={}
    for c in CHANNELS:
        data={}
        for cond in CFG['conditions']:
            mean,second=accum[cond][c]
            cov=second-mean.unsqueeze(-1)*mean.unsqueeze(-2)
            eig,u,ranks=spectrum(cov)
            data[cond]=(mean,cov,eig,u,ranks)
        results[c]={};summary[c]={};curves[c]={}
        mn,cn,en,un,_=data['native']
        for cond,(mean,cov,eig,u,ranks) in data.items():
            fields={name:x.tolist() for name,x in ranks.items()}
            fields['eigenvalues']=eig.tolist()
            fields['mean_offset_native_fluctuation_units']=((mean-mn).norm(dim=-1)/en.sum(-1).sqrt().clamp_min(1e-20)).tolist()
            coverage={};overlap={}
            for k in (16,32,64,96):
                uk=u[...,:k]
                coverage[str(k)]=((uk.transpose(-1,-2) @ cn @ uk).diagonal(dim1=-2,dim2=-1).sum(-1)/en.sum(-1).clamp_min(1e-20)).tolist()
                overlap[str(k)]=((un[...,:k].transpose(-1,-2) @ uk).square().sum((-1,-2))/k).tolist()
            fields.update(native_variance_coverage=coverage,native_subspace_overlap=overlap)
            results[c][cond]=fields
            summary[c][cond]={name:float(x.double().mean()) for name,x in ranks.items()}
            summary[c][cond]['native_variance_coverage']={k:float(torch.tensor(v,device='cuda').mean()) for k,v in coverage.items()}
            summary[c][cond]['mean_offset_native_fluctuation_units']=float((mean-mn).norm(dim=-1).div(en.sum(-1).sqrt().clamp_min(1e-20)).mean())
            curves[c][cond]=(eig.cumsum(-1)/eig.sum(-1,keepdim=True).clamp_min(1e-20)).mean((0,1)).tolist()
        del data
    save(f'results/{task}/layer_head.json',results)
    save(f'results/{task}/summary.json',summary)
    save(f'results/{task}/curves.json',curves)
    plots(task,results,curves)
    save(f'results/{task}/completed.json',{'status':'completed','samples':len(items),'timestamp':now()})
    del accum
    gc.collect();torch.cuda.empty_cache()
    log(f'{task} PCA COMPLETED '+str(summary))

def plots(task, results, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for c in CHANNELS:
        fig,ax=plt.subplots(figsize=(6,4))
        for cond,y in curves[c].items(): ax.plot(range(1,129),y,label=cond)
        ax.set(xlabel='Principal components',ylabel='Mean cumulative variance',title=f'{task} {c}',ylim=(0,1.02))
        ax.legend();fig.tight_layout();fig.savefig(ROOT/f'results/{task}/{c}_spectrum.png',dpi=130);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(11,5),sharey=True)
        for ax,cond in zip(axes,CFG['conditions']):
            im=ax.imshow(results[c][cond]['n95'],aspect='auto',vmin=1,vmax=128);ax.set(title=cond,xlabel='KV head')
        axes[0].set_ylabel('Qwen layer');fig.colorbar(im,ax=axes.ravel().tolist(),label='Components for 95% variance')
        fig.suptitle(f'{task} {c}');fig.savefig(ROOT/f'results/{task}/{c}_n95_heatmap.png',dpi=130);plt.close(fig)

def main():
    assert torch.cuda.is_available()
    torch.manual_seed(CFG['seed'])
    torch.backends.cuda.preferred_linalg_library('cusolver')
    free,_=torch.cuda.mem_get_info()
    assert free/2**30>=CFG['minimum_free_gpu_gib'],f'Insufficient free GPU memory: {free/2**30:.2f} GiB; do not preempt others'
    save('resource_policy.json',dict(torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        arrow_threads=pyarrow.cpu_count(),pca_device='cuda',cpu_used_for='tokenization, I/O and small plots',initial_free_gpu_gib=free/2**30))
    st,rt=load_tokenizer(str(SENDER)),load_tokenizer(str(RECEIVER))
    items=prepare(rt,st)
    _,sender=load_model(str(SENDER),'cuda',torch.float16)
    _,receiver=load_model(str(RECEIVER),'cuda',torch.float16)
    smoke(sender,receiver,items['arc'][0])
    if '--smoke' in sys.argv: return
    mappings={name:load_map(MAPS/f'{name}.pt',device='cuda',dtype=torch.float16,mmap=True) for name in ('recolor','heterofold')}
    save('map_metadata.json',{name:describe(m) for name,m in mappings.items()})
    assert mappings['recolor']['depth_stencil']==mappings['heterofold']['depth_stencil']
    for task,rows in items.items():
        if not (ROOT/f'results/{task}/completed.json').exists(): analyze(task,rows,sender,receiver,mappings)
    summary={task:json.loads((ROOT/f'results/{task}/summary.json').read_text()) for task in items}
    save('summary.json',summary)
    lines=['# HeteroFold KV PCA analysis','',
        'Equal-prompt weighted, per-layer/head centered PCA. K is post-k_norm/pre-RoPE; K_raw is supplementary.',
        'Descriptive 32-example/task analysis, not a benchmark or causal intervention. No new training/generation.','',
        '| Task | Role | Condition | Mean n90 | Mean n95 | Mean n99 | Effective rank | Native variance coverage@32/@64 |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for task,roles in summary.items():
        for c,conditions in roles.items():
            for cond,s in conditions.items():
                coverage=s['native_variance_coverage']
                lines.append(f'| {task} | {c} | {cond} | {s["n90"]:.2f} | {s["n95"]:.2f} | {s["n99"]:.2f} | {s["effective_rank"]:.2f} | {coverage["32"]:.4f}/{coverage["64"]:.4f} |')
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save('status.json',{'status':'completed','timestamp':now()})
    log('ALL PCA ANALYSES COMPLETED')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        save('failure.json',{'type':type(exc).__name__,'message':str(exc),'timestamp':now()})
        raise
