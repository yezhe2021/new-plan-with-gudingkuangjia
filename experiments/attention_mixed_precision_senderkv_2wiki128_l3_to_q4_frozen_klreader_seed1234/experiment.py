"""Frozen Sender-KV quantization, native and heterogeneous reading paths."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from quantization import encode, decode
from metrics import text_units, projection, normalize, to_token_scores, ranked_positions, answer_normalize, bootstrap

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT/'config.json').read_text())
torch.set_num_threads(1)
sys.path.insert(0, CFG['reference_reader'])
spec = importlib.util.spec_from_file_location('quant_frozen_reference', Path(CFG['reference_reader'])/'experiment.py')
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
C, U = R.C, R.U
ARMS = ['fp16', 'int4', 'attention_top', 'random_0', 'random_1', 'random_2', 'attention_bottom']


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(): return sha(ROOT/'manifest.json')
def log(x): print(f'[{time.strftime("%F %T")}] {x}', flush=True)
def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n'); tmp.replace(path)
def status(**kw): save(ROOT/'status.json', dict(time=time.strftime('%F %T'), **kw))
def rows(): return [json.loads(x) for x in (ROOT/'prepared/questions.jsonl').read_text().splitlines()]
def item_path(row): return ROOT/'source_cache'/(row['id']+'.pt')
def result_path(family, row): return ROOT/'evaluation'/family/(row['id']+'.json')
def tok(family): return AutoTokenizer.from_pretrained(CFG['models'][family], local_files_only=True, use_fast=True)


def prepare():
    assert CFG['reader_experiment'] == CFG['reference_reader'], 'Only original KL-only Reader authorized'
    assert read(Path(CFG['reference_reader'])/'completed.json')['status'] == 'completed'
    checkpoint = Path(CFG['reader_experiment'])/CFG['reader_checkpoint']
    reader_meta = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert reader_meta['manifest_sha256'] == R.identity() and reader_meta['step'] == 1024 and reader_meta['epoch'] == 4
    assert reader_meta['frozen_stage_a_sha256'] == R.CFG['base_stage_a_sha256']
    assert reader_meta['frozen_stage_b_sha256'] == R.CFG['frozen_stage_b_sha256']
    del reader_meta
    tokenizers = {f: tok(f) for f in ['llama', 'qwen']}
    docs = read(CFG['dataset']); random.Random(CFG['seed']).shuffle(docs)
    accepted, rejected, seen = [], Counter(), set(CFG['excluded_question_keys'])
    for doc in docs:
        if len(accepted) == CFG['n']: break
        answer, question = str(doc['answer']).strip(), doc['question'].strip()
        key = ' '.join(question.casefold().split())
        if doc['_id'] in CFG['excluded_ids'] or key in seen or not 0 < len(answer) <= CFG['max_answer_characters'] or answer.casefold() in ['yes', 'no']:
            rejected['excluded_duplicate_answer_filter'] += 1; continue
        supports = {(title, int(i)) for title, i in doc['supporting_facts']}
        facts, distractors = [], []
        for title, sentences in doc['context']:
            for i, sentence in enumerate(sentences):
                line = f'{title}: {sentence.strip()}'
                (facts if (title, i) in supports else distractors).append(line)
        if len(facts) != len(supports) or not facts:
            rejected['missing_support'] += 1; continue
        rng = random.Random(int(C.digest([CFG['seed'], doc['_id']])[:12], 16))
        rng.shuffle(distractors); lines = facts+distractors[:CFG['distractor_sentences']]; rng.shuffle(lines)
        evidence = '\n'.join(lines)
        if answer.casefold() not in evidence.casefold(): rejected['absent_answer'] += 1; continue
        prompt = ('Use only the supplied evidence to answer the question. Give only the short answer, without explanation.\n\nEvidence:\n'+evidence+'\n\nQuestion: '+question)
        try:
            row = U.make_row(doc['_id'], 'gsm8k', dict(question=question), tokenizers['llama'], tokenizers['qwen'],
                             prompt=prompt, candidates=[], gold=answer, normalize=False, cue=CFG['answer_prefix'])
            row.update(task='2wiki_controlled', answer=answer, question=question, evidence=evidence, units=text_units(evidence), layouts={})
            for family, tokenizer in tokenizers.items():
                frame = row['frames'][family]
                suffix = tokenizer.encode(frame['rendered_suffix']+CFG['answer_prefix'], add_special_tokens=False)
                full = frame['rendered_full']+CFG['answer_prefix']
                ids = tokenizer.encode(full, add_special_tokens=False)
                assert frame['body_ids']+suffix == ids and len(ids) <= CFG['max_prompt_tokens']
                combined = tokenizer.encode(full+' '+answer, add_special_tokens=False)
                assert combined[:len(ids)] == ids
                gold = combined[len(ids):]; assert 0 < len(gold) <= CFG['max_gold_tokens']
                offsets = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping'][:len(frame['body_ids'])]
                estart = len(frame['rendered_prefix'])+prompt.index('Evidence:\n')+len('Evidence:\n')
                row['layouts'][family] = dict(body_ids=frame['body_ids'], suffix_ids=suffix, gold_ids=gold,
                    offsets=offsets, evidence_start=estart, rendered_prompt=full, total_prompt_tokens=len(ids))
            positions = sorted(set(row['source_indices'])); reverse = {p:i for i,p in enumerate(positions)}
            row['wire_positions'] = positions; row['target_map'] = [reverse[i] for i in row['source_indices']]
            layout = row['layouts']['llama']
            matrix = projection([layout['offsets'][i] for i in positions], layout['evidence_start'], row['units'])
            row['eligible_wire_positions'] = np.flatnonzero(matrix.sum(1) > 0).tolist()
            row['protected_tokens'] = math.ceil(len(positions)*CFG['protected_fraction'])
            assert row['protected_tokens'] <= len(row['eligible_wire_positions'])
            accepted.append(row); seen.add(key)
        except (AssertionError, ValueError) as exc:
            rejected['length_boundary_mapping_filter'] += 1
    assert len(accepted) == CFG['n'], (len(accepted), rejected)
    out = ROOT/'prepared/questions.jsonl'; out.parent.mkdir(exist_ok=True)
    content = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in accepted)
    if out.exists(): assert out.read_text() == content
    else: out.write_text(content)
    checkpoints = dict(reader=checkpoint, stage_a=Path(R.CFG['base_parent'])/'runs/stage_a/best.pt',
                       stage_b=Path(R.CFG['residual_parent'])/'runs/stage_b/last.pt')
    dependencies = {str(p):sha(p) for p in [Path(R.__file__), Path(U.__file__), Path(C.__file__),
                    Path(inspect.getfile(R.GroupedKVTranslator)), Path(inspect.getfile(R.FrozenTranslatorWithResidual)),
                    Path(inspect.getfile(R.QOReader))]}
    models = {}
    for family, model_path in CFG['models'].items():
        model_path = Path(model_path)
        models[family] = dict(path=str(model_path), files={p.name:sha(p) for p in model_path.iterdir() if p.name in
            ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']},
            weights=[dict(name=p.name,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(model_path.glob('*.safetensors'))])
    manifest = dict(config=CFG, config_sha256=sha(ROOT/'config.json'), prepared_sha256=sha(out), dataset_sha256=sha(CFG['dataset']),
        code={p.name:sha(p) for p in ROOT.glob('*.py')}, dependencies=dependencies,
        checkpoints={name:dict(path=str(p),sha256=sha(p)) for name,p in checkpoints.items()},
        reference_reader_manifest_sha256=R.identity(), models=models, n=len(accepted), rejected=dict(rejected),
        official_benchmark=False, training=False, minmax_axes='Key: token groups64; Value: channel groups64',
        llama_template_date=C.CFG['llama_template_date'], environment=dict(torch=torch.__version__,python=sys.version))
    if (ROOT/'manifest.json').exists(): assert read(ROOT/'manifest.json') == manifest
    else: save(ROOT/'manifest.json', manifest)
    save(ROOT/'prepared/completed.json', dict(manifest_sha256=identity(), n=len(accepted)))
    status(status='prepared',phase='data',n=len(accepted)); log(f'PREPARED {len(accepted)} new questions; {dict(rejected)}')


def verify():
    m=read(ROOT/'manifest.json')
    assert sha(ROOT/'config.json')==m['config_sha256'] and sha(ROOT/'prepared/questions.jsonl')==m['prepared_sha256']
    assert R.identity()==m['reference_reader_manifest_sha256']
    for name,digest in m['code'].items(): assert sha(ROOT/name)==digest,name
    for name,digest in m['dependencies'].items(): assert sha(name)==digest,name
    for v in m['checkpoints'].values(): assert sha(v['path'])==v['sha256'],v['path']
    for family,v in m['models'].items():
        for name,digest in v['files'].items(): assert sha(Path(v['path'])/name)==digest
        for w in v['weights']:
            st=(Path(v['path'])/w['name']).stat(); assert st.st_size==w['size'] and st.st_mtime_ns==w['mtime_ns']


def load_model(family):
    assert torch.cuda.is_available()
    free=torch.cuda.mem_get_info()[0]/2**30
    assert free>=CFG['free_gpu_gib'],f'Insufficient GPU memory: {free:.2f} GiB'
    assert shutil.disk_usage(ROOT).free/2**30>=CFG['disk_reserve_gib']
    torch.manual_seed(CFG['seed'])
    model=AutoModelForCausalLM.from_pretrained(CFG['models'][family],dtype=torch.float16,device_map='cuda',
        local_files_only=True,attn_implementation=CFG['attention_backend']).eval().requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(); log(f'LOADED {family}; {free:.2f} GiB free before load')
    return tok(family),model


@torch.inference_mode()
def capture_body(model, ids):
    kk,vv,handles={},{},[]
    def hook(i):
        def apply(module,args,kwargs):
            x=kwargs.get('hidden_states',args[0] if args else None);shape=(*x.shape[:-1],-1,module.head_dim)
            k=module.k_proj(x).view(shape)
            if hasattr(module,'k_norm'): k=module.k_norm(k)
            kk[i]=k[0].cpu();vv[i]=module.v_proj(x).view(shape)[0].cpu()
        return apply
    for i,layer in enumerate(model.model.layers): handles.append(layer.self_attn.register_forward_pre_hook(hook(i),with_kwargs=True))
    inputs=torch.tensor([ids],device='cuda')
    try:
        output=model.model(inputs,attention_mask=torch.ones_like(inputs),position_ids=torch.arange(len(ids),device='cuda')[None],use_cache=True)
    finally:
        for h in handles:h.remove()
    return torch.stack([kk[i] for i in range(len(kk))]),torch.stack([vv[i] for i in range(len(vv))]),output.past_key_values


@torch.inference_mode()
def evaluate_cache(model, tokenizer, layout, body, answer, capture=False):
    b=len(layout['body_ids']);suffix=layout['suffix_ids'];gold=layout['gold_ids'];collected=[];handles=[]
    if capture:
        def hook(module,args,out):
            weights=out[1];assert weights is not None
            collected.append(weights[0,:,-1,:b].detach().float().mean(0).cpu().numpy())
        handles=[layer.self_attn.register_forward_hook(hook) for layer in model.model.layers]
    cache=copy.deepcopy(body);inputs=torch.tensor([suffix],device='cuda')
    try:
        out=model.model(inputs,past_key_values=cache,attention_mask=torch.ones(1,b+len(suffix),device='cuda',dtype=torch.long),
            position_ids=torch.arange(b,b+len(suffix),device='cuda')[None],use_cache=True)
    finally:
        for h in handles:h.remove()
    logits=model.lm_head(out.last_hidden_state[:,-1])[0].float();first=logits.clone();generated=[]
    eos=model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos or [tokenizer.eos_token_id])
    cache=out.past_key_values; current=logits
    for i in range(CFG['max_new_tokens']):
        token=int(current.argmax());generated.append(token)
        if token in eos:break
        if i+1==CFG['max_new_tokens']:break
        output=model.model(torch.tensor([[token]],device='cuda'),past_key_values=cache,
            attention_mask=torch.ones(1,b+len(suffix)+i+1,device='cuda',dtype=torch.long),
            position_ids=torch.tensor([[b+len(suffix)+i]],device='cuda'),use_cache=True)
        cache=output.past_key_values;current=model.lm_head(output.last_hidden_state[:,-1])[0].float()
    inputs=suffix+gold[:-1];tensor=torch.tensor([inputs],device='cuda')
    output=model.model(tensor,past_key_values=copy.deepcopy(body),attention_mask=torch.ones(1,b+len(inputs),device='cuda',dtype=torch.long),
        position_ids=torch.arange(b,b+len(inputs),device='cuda')[None],use_cache=False)
    prediction=model.lm_head(output.last_hidden_state[0,len(suffix)-1:len(suffix)-1+len(gold)]).float()
    loss=F.cross_entropy(prediction,torch.tensor(gold,device='cuda'),reduction='sum')
    text=tokenizer.decode(generated,skip_special_tokens=True)
    result=dict(gold_nll_sum=float(loss),gold_tokens=len(gold),nll_per_token=float(loss)/len(gold),
        nll_per_character=float(loss)/max(1,len(answer)),generation=text,generated_tokens=len(generated),
        reached_cap=int(len(generated)==CFG['max_new_tokens'] and generated[-1] not in eos),
        exact_match=int(answer_normalize(text)==answer_normalize(answer)))
    assert torch.isfinite(loss) and torch.isfinite(first).all()
    return result,first,(np.mean(collected,axis=0) if capture else None)


@torch.inference_mode()
def native_cache(model, original, key, value, positions):
    data=[];ix=torch.tensor(positions,device='cuda')
    for i,layer in enumerate(original.layers):
        k=layer.keys.detach().clone();v=layer.values.detach().clone()
        rotated=C.apply_rope(model,key[i].cuda(),ix,i).transpose(0,1)[None]
        k[:,:,ix]=rotated;v[:,:,ix]=value[i].cuda().transpose(0,1)[None]
        data.append((k,v))
    return DynamicCache(ddp_cache_data=data,config=model.config)


def selectors(row, raw_attention):
    layout=row['layouts']['llama'];matrix=projection(layout['offsets'],layout['evidence_start'],row['units'])
    mass=float((raw_attention@matrix).sum());unit=normalize(raw_attention@matrix)
    wire_matrix=matrix[row['wire_positions']];score,eligible=to_token_scores(unit,wire_matrix)
    assert eligible.tolist()==row['eligible_wire_positions'] and mass>0
    count=row['protected_tokens'];seed=int(C.digest([CFG['seed'],row['id']])[:12],16)
    selections=dict(fp16=[],int4=[],attention_top=ranked_positions(score,eligible,count,seed=seed),
                    attention_bottom=ranked_positions(score,eligible,count,reverse=True,seed=seed))
    for i,random_seed in enumerate(CFG['random_seeds']):
        rng=np.random.default_rng(seed+random_seed);selections[f'random_{i}']=sorted(rng.choice(eligible,count,replace=False).tolist())
    return selections,dict(evidence_attention_mass=mass,unit_scores=unit.tolist(),wire_scores=score.tolist(),
        units=row['units'],source_positions=row['wire_positions'],protected_tokens=count)


def packet_for(row,key,value,arm,selections):
    bits=16 if arm=='fp16' else (4 if arm=='int4' else 'mixed')
    packet,stats=encode(key,value,selections[arm],bits,CFG['group_size'],row['wire_positions'],row['target_map'])
    k,v,metadata=decode(packet)
    assert metadata['source_positions']==row['wire_positions'] and metadata['target_map']==row['target_map']
    assert metadata['high_positions']==selections[arm]
    stats['packet_sha256']=hashlib.sha256(packet).hexdigest()
    stats['key_nmse']=float((k.float()-key.float()).square().sum()/key.float().square().sum().clamp_min(1e-30))
    stats['value_nmse']=float((v.float()-value.float()).square().sum()/value.float().square().sum().clamp_min(1e-30))
    return k,v,stats


def add_delta(value,base,first,base_first):
    value.update(delta_nll_per_token=value['nll_per_token']-base['nll_per_token'],
                 delta_nll_per_character=value['nll_per_character']-base['nll_per_character'],
                 next_token_kl=float(F.kl_div(F.log_softmax(first,-1),F.softmax(base_first,-1),reduction='sum')))
    return value


@torch.inference_mode()
def sender(smoke=False):
    verify(); tokenizer,model=load_model('llama'); selected=rows()[:1] if smoke else rows(); summaries=[]
    for index,row in enumerate(selected,1):
        if not smoke and result_path('llama',row).exists(): raise RuntimeError('Existing native result; use explicit reviewed resume, not blind rerun')
        key,value,original=capture_body(model,row['layouts']['llama']['body_ids'])
        direct,direct_first,attention=evaluate_cache(model,tokenizer,row['layouts']['llama'],original,row['answer'],capture=True)
        selections,ranks=selectors(row,attention);positions=row['wire_positions'];key=key[:,positions].contiguous();value=value[:,positions].contiguous()
        results={};base=base_first=None;byte_counts=[]
        for arm in ARMS:
            status(status='running',phase='sender_smoke' if smoke else 'sender',sample=index,total=len(selected),arm=arm)
            k,v,stats=packet_for(row,key,value,arm,selections)
            cache=native_cache(model,original,k,v,positions)
            result,first,_=evaluate_cache(model,tokenizer,row['layouts']['llama'],cache,row['answer'])
            if arm=='fp16':
                base,base_first=result,first.clone()
                parity=dict(top1_equal=int(first.argmax())==int(direct_first.argmax()),max_logit_error=float((first-direct_first).abs().max()),
                    nll_error=abs(result['nll_per_token']-direct['nll_per_token']),generation_equal=result['generation']==direct['generation'])
                assert parity['top1_equal'] and parity['nll_error']<.03,parity
            results[arm]=dict(add_delta(result,base,first,base_first),wire=stats)
            if arm not in ['fp16','int4']:byte_counts.append(stats['wire_bytes'])
        assert len(set(byte_counts))==1,'Mixed arms not byte matched'
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        if smoke:
            (ROOT/'smoke').mkdir(exist_ok=True)
            torch.save(dict(manifest_sha256=identity(),key=key,value=value),ROOT/'smoke/source.pt')
            save(ROOT/'smoke/llama.json',dict(status='completed',manifest_sha256=identity(),fp16_cache_parity=parity,
                exact_byte_match=True,arms=results,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
            save(ROOT/'smoke/sender_row.json',dict(manifest_sha256=identity(),id=row['id'],selections=selections,arms=results))
        else:
            item_path(row).parent.mkdir(exist_ok=True)
            torch.save(dict(manifest_sha256=identity(),key=key,value=value),item_path(row))
            save(result_path('llama',row),dict(id=row['id'],manifest_sha256=identity(),answer=row['answer'],selections=selections,
                ranking=ranks,arms=results,fp16_cache_parity=parity))
        del original,cache,key,value,k,v
        log(f'LLAMA {"SMOKE" if smoke else "EVAL"} {index}/{len(selected)} FP16 EM={base["exact_match"]}')
    if not smoke:save(ROOT/'evaluation/llama/completed.json',dict(status='completed',manifest_sha256=identity(),n=len(selected)))


def load_frontend(model):
    meta=read(ROOT/'manifest.json');rcfg=R.CFG
    base=R.GroupedKVTranslator(**rcfg['architecture'])
    state=torch.load(meta['checkpoints']['stage_a']['path'],map_location='cpu',weights_only=True)
    base.load_state_dict(state['state'],strict=True);del state
    base=R.FrozenTranslatorWithResidual(base,rcfg['residual_rank'])
    state=torch.load(meta['checkpoints']['stage_b']['path'],map_location='cpu',weights_only=True)
    base.residual.load_state_dict(state['state'],strict=True);del state
    base=base.cuda().eval().requires_grad_(False)
    reader=R.QOReader(model,rcfg['rank'],rcfg['alpha'],rcfg['layers'])
    state=torch.load(meta['checkpoints']['reader']['path'],map_location='cpu',weights_only=True)
    assert state['manifest_sha256']==meta['reference_reader_manifest_sha256'] and state['step']==1024
    reader.load_state_dict(state['state']);del state
    model.eval().requires_grad_(False)
    assert all(not p.requires_grad for p in reader.parameters())
    return base,reader


@torch.inference_mode()
def translated(model,base,reader,row,native,key,value):
    mapping=row['target_map'];pieces=[]
    for start in range(0,len(mapping),64):
        ix=mapping[start:start+64]
        with torch.amp.autocast('cuda',dtype=torch.float16): pieces.append(base(key[:,ix][None].cuda(),value[:,ix][None].cuda()))
    k,v=(torch.cat([p[i][0] for p in pieces],dim=1) for i in range(2))
    assert torch.isfinite(k).all() and torch.isfinite(v).all()
    cache=U.translated_cache(model,row,native,k,v)
    return cache,k,v


@torch.inference_mode()
def receiver(smoke=False):
    verify();tokenizer,model=load_model('qwen');base,reader=load_frontend(model)
    selected=rows()[:1] if smoke else rows()
    if smoke:
        # Smoke gets the exact first-row Sender values from a temporary CPU-only file prepared by sender smoke.
        source=torch.load(ROOT/'smoke/source.pt',map_location='cpu',weights_only=True)
    for index,row in enumerate(selected,1):
        if not smoke and result_path('qwen',row).exists(): raise RuntimeError('Existing receiver result; explicit reviewed resume required')
        record=read(ROOT/'smoke/sender_row.json') if smoke else read(result_path('llama',row))
        source=source if smoke else torch.load(item_path(row),map_location='cpu',weights_only=True)
        assert source['manifest_sha256']==record['manifest_sha256']==identity()
        key,value=source['key'],source['value'];selections=record['selections']
        with reader.mode(False):native=U.native_body(model,row)
        results={};base_result=base_first=base_k=base_v=None;byte_counts=[]
        for arm in ARMS:
            status(status='running',phase='receiver_smoke' if smoke else 'receiver',sample=index,total=len(selected),arm=arm)
            k,v,stats=packet_for(row,key,value,arm,selections)
            assert stats['packet_sha256']==record['arms'][arm]['wire']['packet_sha256'],'Reading paths did not receive identical packet'
            with reader.mode(False):cache,mapped_k,mapped_v=translated(model,base,reader,row,native,k,v)
            with reader.mode(True):result,first,_=evaluate_cache(model,tokenizer,row['layouts']['qwen'],cache,row['answer'])
            if arm=='fp16':
                base_result,base_first=result,first.clone();base_k,base_v=mapped_k.clone(),mapped_v.clone()
            stats['mapped_key_nmse']=float((mapped_k.float()-base_k.float()).square().sum()/base_k.float().square().sum().clamp_min(1e-30))
            stats['mapped_value_nmse']=float((mapped_v.float()-base_v.float()).square().sum()/base_v.float().square().sum().clamp_min(1e-30))
            results[arm]=dict(add_delta(result,base_result,first,base_first),wire=stats)
            if arm not in ['fp16','int4']:byte_counts.append(stats['wire_bytes'])
        assert len(set(byte_counts))==1
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert all(not p.requires_grad and p.grad is None for p in base.parameters())
        output=dict(status='completed',id=row['id'],manifest_sha256=identity(),answer=row['answer'],arms=results,
                    same_packet_both_paths=True,parameters_frozen=True,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        save(ROOT/'smoke/qwen.json' if smoke else result_path('qwen',row),output)
        if not smoke:item_path(row).unlink()  # Only this run's regenerable temporary Sender tensor, after durable results.
        del native,cache,k,v,mapped_k,mapped_v,base_k,base_v,source
        log(f'QWEN {"SMOKE" if smoke else "EVAL"} {index}/{len(selected)} FP16 EM={base_result["exact_match"]}')
    if not smoke:save(ROOT/'evaluation/qwen/completed.json',dict(status='completed',manifest_sha256=identity(),n=len(selected)))


def summarize():
    verify();summary=dict(experiment=CFG['name'],status='completed',timestamp=time.strftime('%F %T'),manifest_sha256=identity(),
                         config=CFG,paths={},uncertainty='Paired question bootstrap, random repeats averaged within question; exploratory uncorrected95% intervals.')
    for family in ['llama','qwen']:
        assert read(ROOT/f'evaluation/{family}/completed.json')['manifest_sha256']==identity()
        records=[read(result_path(family,r)) for r in rows()];arms={}
        for arm in ARMS+['random_mean']:
            names=['random_0','random_1','random_2'] if arm=='random_mean' else [arm]
            values=[{key:float(np.mean([r['arms'][a][key] for a in names])) for key in
                ['delta_nll_per_token','delta_nll_per_character','exact_match','next_token_kl','reached_cap']} for r in records]
            arms[arm]=dict(n=len(values),accuracy=float(np.mean([v['exact_match'] for v in values])),
                cap_rate=float(np.mean([v['reached_cap'] for v in values])),
                delta_nll_per_token=bootstrap([v['delta_nll_per_token'] for v in values],CFG['seed']),
                delta_nll_per_character=bootstrap([v['delta_nll_per_character'] for v in values],CFG['seed']),
                next_token_kl=bootstrap([v['next_token_kl'] for v in values],CFG['seed']),
                mean_wire_bytes=float(np.mean([r['arms'][names[0]]['wire']['wire_bytes'] for r in records])),
                effective_bits_per_element=float(np.mean([r['arms'][names[0]]['wire']['effective_bits_per_element'] for r in records])))
        paired={}
        for control in ['random_mean','attention_bottom']:
            diff=[]
            for r in records:
                other=np.mean([r['arms'][a]['delta_nll_per_token'] for a in ['random_0','random_1','random_2']]) if control=='random_mean' else r['arms'][control]['delta_nll_per_token']
                diff.append(r['arms']['attention_top']['delta_nll_per_token']-other)
            paired['attention_top_minus_'+control]=bootstrap(diff,CFG['seed'])
        summary['paths'][family]=dict(arms=arms,paired_contrasts=paired)
    save(ROOT/'summary.json',summary)
    import fcntl
    with (ROOT.parent/'EXPERIMENT_RESULTS.md').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.write('\n\n## '+CFG['name']+'\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n');stream.flush();os.fsync(stream.fileno())
    save(ROOT/'completed.json',dict(status='completed',manifest_sha256=identity()));status(status='completed',phase='all');log('ALL QUANTIZATION EVALUATIONS COMPLETED')


if __name__=='__main__':
    try:
        action=sys.argv[1]
        {'prepare':prepare,'verify':verify,'sender_smoke':lambda:sender(True),'receiver_smoke':lambda:receiver(True),
         'sender':sender,'receiver':receiver,'summarize':summarize}[action]()
    except Exception as exc:
        import traceback
        save(ROOT/'failure.json',dict(time=time.strftime('%F %T'),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        status(status='failed',phase=sys.argv[1],error=str(exc));raise
