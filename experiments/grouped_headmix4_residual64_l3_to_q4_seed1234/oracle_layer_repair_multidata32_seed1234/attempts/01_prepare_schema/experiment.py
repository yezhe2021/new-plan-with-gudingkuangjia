"""Frozen multi-task translator: Oracle K/V replacement and direct-access masking."""
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT/'config.json').read_text())
sys.path.insert(0, CFG['parent'])
spec = importlib.util.spec_from_file_location('oracle_multidata_parent', Path(CFG['parent'])/'experiment.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
U, C, torch = P.U, P.C, P.torch
from transformers import DynamicCache
cache_layer = U._cache_layer


def read(path): return json.loads(Path(path).read_text())
def save(path, value): C.save(ROOT/path, value)
def identity(): return C.sha(ROOT/'manifest.json')
def mark(path, **value): save(path, dict(manifest_sha256=identity(), **value))
def status(**value): save('status.json', dict(time=time.strftime('%F %T'), **value))
def rows(task=None):
    return [r for r in C.read_jsonl(ROOT/'prepared/pilot.jsonl') if task is None or r['task']==task]
def arms():
    return [dict(name=n, kind=n, layers=[]) for n in ['translated','native','reject_all']] + [
        dict(name=f'g{g+1}_{kind}', kind=kind, layers=list(range(6*g,6*(g+1))))
        for g in range(6) for kind in ['K','V','KV','mask']]
def historical(row):
    return read(Path(CFG['parent'])/'evaluation'/row['task']/'per_sample'/(U.fname(row)+'.json'))


def prepare():
    if (ROOT/'manifest.json').exists(): verify(); return
    P.verify()
    assert P.identity()==CFG['parent_manifest_sha256']
    assert C.sha(Path(P.CFG['base_parent'])/'runs/stage_a/best.pt')==CFG['stage_a_sha256']
    assert C.sha(Path(CFG['parent'])/'runs/stage_b/last.pt')==CFG['stage_b_sha256']
    pool=U.rows('test'); selected=[]; reserved=[]; paths=[]; counts={}
    train_keys={r['question_key'] for split in ['train','validation'] for r in U.rows(split)}
    for task in CFG['tasks']:
        task_rows=[r for r in pool if r['task']==task]
        chosen=sorted(random.Random(CFG['seed']).sample(range(len(task_rows)), CFG['n_per_task']))
        counts[task]=dict(pilot=len(chosen), original_test=len(task_rows), existing_remaining=len(task_rows)-len(chosen))
        for i, original in enumerate(task_rows):
            if i not in chosen:
                reserved.append(dict(id=original['id'],task=task,question_key=original['question_key'])); continue
            assert original['question_key'] not in train_keys
            row=json.loads(json.dumps(original)); frame=row['frames']['qwen']
            assert frame['body_ids']+row['suffix_ids']==row['native_context_ids']
            anchor=frame['native_prefix_length']; length=len(frame['body_ids'])
            assert 0<anchor<length and length-anchor==len(row['source_indices'])
            row.update(cache_length=length, intervention_positions=list(range(anchor,length)),
                       original_row_sha256=C.digest(original))
            ref=historical(row)
            assert ref['manifest_sha256']==P.identity()
            for name in ['id','task','prompt','receiver_suffix','candidates','gold']: assert ref[name]==row[name]
            assert set(['grouped_residual','qwen_native']).issubset(ref['conditions'])
            selected.append(row)
            paths.append(Path(CFG['parent'])/'evaluation'/task/'per_sample'/(U.fname(row)+'.json'))
    assert len(selected)==192 and len({r['id'] for r in selected})==192
    C.write_jsonl(ROOT/'prepared/pilot.jsonl',selected)
    C.write_jsonl(ROOT/'prepared/remaining_historical_ids.jsonl',reserved)
    paths += [ROOT/'prepared/pilot.jsonl',ROOT/'prepared/remaining_historical_ids.jsonl',
              Path(CFG['parent'])/'manifest.json',Path(CFG['parent'])/'experiment.py',
              Path(CFG['parent'])/'runs/stage_b/last.pt',Path(P.CFG['base_parent'])/'runs/stage_a/best.pt',
              U.ROOT/'prepared/test.jsonl',U.ROOT/'prepared/train.jsonl',U.ROOT/'prepared/validation.jsonl',Path(U.__file__),Path(C.__file__)]
    save('manifest.json',dict(config=CFG, config_sha256=C.sha(ROOT/'config.json'),
        code={p.name:C.sha(p) for p in ROOT.glob('*.py')}, sources={str(p):C.sha(p) for p in paths},
        counts=counts,ids={t:[r['id'] for r in selected if r['task']==t] for t in CFG['tasks']},arms=arms(),
        source_cache_available=sum(U.source_path(r).exists() for r in selected),
        selection='Seed1234 sample32 indices/task without outcome/length filtering; historical evaluation examples, exploratory only.',
        intervention='All original translated body columns after native anchor, including Question/options. No evidence-only reformulation. Native anchor and suffix fixed.',
        teacher_use='Native prompt KV is permitted in this offline Oracle experiment; not a deployable inference method.',
        confirmation='Not run here. GSM8K original test subset has32 examples: new disjoint confirmation examples must be prepared later.',
        training=False,reader=False))
    mark('prepared/completed.json',status='completed',n=192,configurations=27,total_sample_configurations=5184,counts=counts)
    print(time.strftime('%F %T'),'PREPARED',json.dumps(counts),flush=True)


def verify():
    P.verify();m=read(ROOT/'manifest.json')
    assert C.sha(ROOT/'config.json')==m['config_sha256']
    for name,digest in m['code'].items():assert C.sha(ROOT/name)==digest,name
    for path,digest in m['sources'].items():assert C.sha(path)==digest,path


def versions(module):return [(n,id(p),p._version,p.requires_grad) for n,p in module.named_parameters()]


def mix_cache(model,translated,native,layers=(),roles=(),positions=()):
    data=[]
    for l in range(36):
        tk,tv=cache_layer(translated,l);nk,nv=cache_layer(native,l)
        k,v=tk.detach().clone(),tv.detach().clone()
        if l in layers:
            if 'K' in roles:k[:,:,positions,:]=nk[:,:,positions,:]
            if 'V' in roles:v[:,:,positions,:]=nv[:,:,positions,:]
        data.append((k,v))
    return DynamicCache(ddp_cache_data=data,config=model.config)


class Engine:
    def __init__(self):
        C.seed();self.tok,self.model=C.model('qwen')
        self.translator=P.load_weights(Path(CFG['parent'])/'runs/stage_b/last.pt').eval().requires_grad_(False)
        self.sender=None; self.snapshots=[versions(m) for m in [self.model,self.translator]]
        self.blocked=set();self.context=None;self.mask_calls=0;self.hooks=[]
        assert (self.model.config.num_hidden_layers,self.model.config.num_attention_heads,self.model.config.num_key_value_heads)==(36,32,8)
        for i,layer in enumerate(self.model.model.layers):
            def hook(module,args,kwargs,index=i):
                if index not in self.blocked:return
                assert self.context is not None
                base,count,cols=self.context
                q=torch.arange(base,base+count,device='cuda')[:,None]
                k=torch.arange(base+count,device='cuda')[None,:]
                allowed=(k<=q);allowed[:,cols]=False
                mask=torch.zeros(count,base+count,device='cuda',dtype=self.model.dtype).masked_fill(~allowed,float('-inf'))
                kwargs=dict(kwargs);kwargs['attention_mask']=mask[None,None];self.mask_calls+=1
                return args,kwargs
            self.hooks.append(layer.self_attn.register_forward_pre_hook(hook,with_kwargs=True))
        eos=self.model.generation_config.eos_token_id
        self.stops=set(eos if isinstance(eos,list) else [eos])
        self.stops.update([self.tok.eos_token_id,self.tok.convert_tokens_to_ids('<|im_end|>')]);self.stops.discard(None)

    def unchanged(self):
        assert [versions(m) for m in [self.model,self.translator]]==self.snapshots
        for m in [self.model,self.translator]+([self.sender] if self.sender is not None else []):
            assert all(not p.requires_grad and p.grad is None for p in m.parameters())

    @torch.no_grad()
    def caches(self,row):
        assert not self.blocked
        native=U.native_body(self.model,row)
        if U.source_path(row).exists():source=U.loaded_source(row)
        else:
            if self.sender is None:self.sender=C.model('llama')[1]
            ids=row['frames']['llama']['body_ids']
            k,v,_=C.capture(self.sender,ids,len(ids),[0]);ix=row['source_indices']
            source=dict(key=k[:,ix].contiguous(),value=v[:,ix].contiguous())
        k,v=P.map_source(self.translator,source)
        assert torch.isfinite(k).all() and torch.isfinite(v).all()
        translated=U.translated_cache(self.model,row,native,k,v)
        for l in range(36):
            for a,b in zip(cache_layer(translated,l),cache_layer(native,l)):
                assert a.shape==b.shape
                assert torch.equal(a[:,:,:row['intervention_positions'][0]],b[:,:,:row['intervention_positions'][0]])
        return native,translated

    def forward(self,ids,cache,base,row,layers=(),use_cache=True):
        self.context=(base,len(ids),row['intervention_positions']);self.blocked=set(layers)
        try:
            return self.model.model(torch.tensor([ids],device='cuda'),past_key_values=cache,
                attention_mask=torch.ones(1,base+len(ids),device='cuda',dtype=torch.long),
                position_ids=torch.arange(base,base+len(ids),device='cuda')[None],use_cache=use_cache)
        finally:self.context=None;self.blocked=set()

    @torch.no_grad()
    def first_logits(self,row,cache,layers=()):
        out=self.forward(row['suffix_ids'],U.clone_cache(cache,self.model),row['cache_length'],row,layers)
        return self.model.lm_head(out.last_hidden_state[0,-1]).float()

    @torch.no_grad()
    def score(self,row,cache,layers=()):
        scores=[]
        for candidate in row['candidates']:
            target=C.R.candidate_ids(self.tok,candidate)
            out=self.forward(row['suffix_ids']+target,U.clone_cache(cache,self.model),row['cache_length'],row,layers,use_cache=False)
            ix=torch.arange(len(row['suffix_ids'])-1,len(row['suffix_ids'])-1+len(target),device='cuda')
            logits=self.model.lm_head(out.last_hidden_state[0,ix]).float()
            scores.append(C.R._candidate_log_likelihood(logits[None],torch.tensor([target],device='cuda'),row['normalize']))
        return C.record(scores,row['gold'])

    @torch.no_grad()
    def generate(self,row,cache,layers=(),cap=None):
        cap=CFG['gsm8k_max_new_tokens'] if cap is None else cap
        out=self.forward(row['suffix_ids'],U.clone_cache(cache,self.model),row['cache_length'],row,layers)
        total=row['cache_length']+len(row['suffix_ids']);tokens=[];stopped=False
        for i in range(cap):
            logits=self.model.lm_head(out.last_hidden_state[0,-1]).float();assert torch.isfinite(logits).all()
            token=int(logits.argmax());tokens.append(token)
            if token in self.stops:stopped=True;break
            if i+1<cap:out=self.forward([token],out.past_key_values,total,row,layers);total+=1
        text=self.tok.decode(tokens,skip_special_tokens=False);pred=C.S.extract_numeric_answer(text)
        return dict(generation=text,prediction=pred,generated_tokens=len(tokens),reached_cap=int(len(tokens)>=cap),
                    correct=int(pred==C.S.normalize_number(row['gold'])))

    def result(self,row,cache,layers=()):
        return self.generate(row,cache,layers) if row['task']=='gsm8k' else self.score(row,cache,layers)


def cache_for(engine,row,native,translated,arm):
    kind=arm['kind'];cols=row['intervention_positions']
    if kind=='native':return mix_cache(engine.model,translated,native,range(36),'KV',cols),[]
    if kind in ['translated','reject_all','mask']:
        blocked=list(range(36)) if kind=='reject_all' else arm['layers'] if kind=='mask' else []
        return mix_cache(engine.model,translated,native),blocked
    return mix_cache(engine.model,translated,native,arm['layers'],kind,cols),[]


def baseline_parity(row,arm,value):
    expected=historical(row)['conditions']['grouped_residual' if arm=='translated' else 'qwen_native']
    assert value['prediction']==expected['prediction'] and value['correct']==expected['correct'],('Historical prediction mismatch',row['id'],arm)
    if row['task']=='gsm8k':
        assert value['generation']==expected['generation'] and value['generated_tokens']==expected['generated_tokens'],('Historical generation mismatch',row['id'],arm)
    else:
        error=max(abs(a-b) for a,b in zip(value['scores'],expected['scores']))
        assert error<CFG['candidate_score_parity_abs_max'],('Historical candidate-score mismatch',row['id'],arm,error)


@torch.no_grad()
def smoke():
    verify();engine=Engine();checks=[]
    for task in CFG['tasks']:
        row=rows(task)[0];native,translated=engine.caches(row)
        copies=[mix_cache(engine.model,translated,native),mix_cache(engine.model,translated,native,range(36),'KV',row['intervention_positions'])]
        for copy,reference in zip(copies,[translated,native]):
            assert all(torch.equal(a,b) for l in range(36) for a,b in zip(cache_layer(copy,l),cache_layer(reference,l)))
        for name,cache in [('translated',translated),('native',native)]:
            if task!='gsm8k':baseline_parity(row,name,engine.score(row,cache))
            else:
                value=engine.generate(row,cache,cap=16)
                reference,count=C.R.generate(C.runtime(engine.model,engine.tok),torch.tensor([row['suffix_ids']],device='cuda'),16,
                    past_key_values=U.clone_cache(cache,engine.model),past_length=row['cache_length'],temperature=0.)
                assert value['generation']==reference and value['generated_tokens']==count
                assert historical(row)['conditions']['grouped_residual' if name=='translated' else 'qwen_native']['generation'].startswith(value['generation'])
        out=engine.model.model(torch.tensor([row['native_context_ids']],device='cuda'),use_cache=False)
        full=engine.model.lm_head(out.last_hidden_state[0,-1]).float();cached=engine.first_logits(row,native)
        kl=float(U.full_kl(full[None],cached[None]));assert kl<U.CFG['native_parity_kl_max'] and int(full.argmax())==int(cached.argmax())
        cols=row['intervention_positions'];anchor=cols[0]
        for role,unchanged_role in [('K',1),('V',0),('KV',None)]:
            mixed=mix_cache(engine.model,translated,native,range(6),role,cols)
            for l in range(36):
                for j,(a,b) in enumerate(zip(cache_layer(mixed,l),cache_layer(translated,l))):
                    assert torch.equal(a[:,:,:anchor],b[:,:,:anchor])
                    if l>=6 or j==unchanged_role:assert torch.equal(a,b)
                    elif (j==0 and 'K' in role) or (j==1 and 'V' in role):assert torch.equal(a[:,:,cols],cache_layer(native,l)[j][:,:,cols])
            del mixed
        changed=U.clone_cache(translated,engine.model)
        for l in range(6):
            for x in cache_layer(changed,l):x[:,:,cols,:]=torch.randn_like(x[:,:,cols,:])
        masked=engine.first_logits(row,translated,range(6));other=engine.first_logits(row,changed,range(6))
        assert torch.equal(masked,other),'Masked content still affects outputs'
        assert engine.mask_calls>0
        checks.append(dict(task=task,id=row['id'],native_cached_full_kl=kl,mask_content_invariant=True,
                           role_and_layer_isolation=True,native_anchor_unchanged=True,baseline_interface_parity=True))
        del native,translated,copies,changed,out,full,cached,masked,other
        engine.unchanged();print(time.strftime('%F %T'),'SMOKE',task,'PASSED',flush=True)
    mark('smoke/completed.json',status='completed',checks=checks,gsm8k_smoke_generation_cap=16,
         gsm8k_formal_cap=4096,all_formal_baselines_checked_during_evaluation=True)


def paired(values):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(CFG['seed'])
    boot=x[rng.integers(0,len(x),size=(2000,len(x)))].mean(1)
    return dict(n=len(x),mean=float(x.mean()),ci95=[float(v) for v in np.quantile(boot,[.025,.975])])


@torch.no_grad()
def evaluate(task):
    verify();assert read(ROOT/'smoke/completed.json')['manifest_sha256']==identity()
    engine=Engine();records=[];start=time.monotonic();completed=0
    for i,row in enumerate(rows(task),1):
        native,translated=engine.caches(row);values={}
        for arm in arms():
            path=f'evaluation/{task}/per_arm/{arm["name"]}/{U.fname(row)}.json'
            if (ROOT/path).exists():
                item=read(ROOT/path);assert item['manifest_sha256']==identity();value=item['result']
            else:
                cache,blocked=cache_for(engine,row,native,translated,arm)
                value=engine.result(row,cache,blocked);del cache
                mark(path,id=row['id'],task=task,arm=arm,result=value)
            if arm['kind'] in ['translated','native']:baseline_parity(row,arm['kind'],value)
            values[arm['name']]=value;completed+=1
            status(status='running',phase='evaluation',task=task,sample=i,total=32,arm=arm['name'],
                   configurations_completed=completed,task_configurations=864,elapsed_seconds=time.monotonic()-start)
            if task=='gsm8k':print(time.strftime('%F %T'),f'{task} {i}/32 {arm["name"]} tokens={value["generated_tokens"]} correct={value["correct"]}',flush=True)
        records.append(dict(id=row['id'],task=task,gold=row['gold'],conditions=values))
        mark(f'evaluation/{task}/per_sample/{U.fname(row)}.json',**records[-1]);del native,translated
        engine.unchanged();print(time.strftime('%F %T'),f'EVALUATION {task} {i}/32',flush=True)
    verify();C.write_jsonl(ROOT/f'evaluation/{task}/per_sample.jsonl',records)
    metrics={a['name']:dict(n=len(records),correct=sum(r['conditions'][a['name']]['correct'] for r in records),
        accuracy=sum(r['conditions'][a['name']]['correct'] for r in records)/len(records),
        cap_count=sum(r['conditions'][a['name']].get('reached_cap',0) for r in records)) for a in arms()}
    comparisons={};gap=metrics['native']['accuracy']-metrics['translated']['accuracy']
    for a in arms()[3:]:
        name=a['name'];mask=name.split('_')[0]+'_mask';delta=metrics[name]['accuracy']-metrics['translated']['accuracy']
        comparisons[name]=dict(delta_accuracy=delta,recovery_ratio=delta/gap if gap>0 else None,
            paired_vs_translated=paired([r['conditions'][name]['correct']-r['conditions']['translated']['correct'] for r in records]),
            paired_vs_mask=paired([r['conditions'][name]['correct']-r['conditions'][mask]['correct'] for r in records]),
            rescued_ids=[r['id'] for r in records if r['conditions'][name]['correct'] and not r['conditions']['translated']['correct']],
            harmed_ids=[r['id'] for r in records if not r['conditions'][name]['correct'] and r['conditions']['translated']['correct']],
            strict_rescue_ids=[r['id'] for r in records if r['conditions'][name]['correct'] and not any(r['conditions'][z]['correct'] for z in ['translated','reject_all',mask])])
    mark(f'evaluation/{task}/summary.json',status='completed',task=task,metrics=metrics,comparisons=comparisons,
         all32_historical_baselines_checked=True,models_frozen=True)
    mark(f'evaluation/{task}/completed.json',status='completed',finished_at=time.strftime('%F %T'))


def summarize():
    verify();results={t:read(ROOT/f'evaluation/{t}/summary.json') for t in CFG['tasks']}
    assert all(v['manifest_sha256']==identity() and v['status']=='completed' for v in results.values())
    macro={a['name']:sum(v['metrics'][a['name']]['accuracy'] for v in results.values())/6 for a in arms()}
    mark('summary.json',status='completed',n=192,configurations=27,tasks=results,macro_accuracy=macro,
         training=False,reader=False,exploratory=True,confirmation_evaluated=False,caveats=CFG['caveats'])
    lines=['# Multi-dataset full-transferred-body Oracle repair pilot','',
           'Frozen existing multi-task Stage-A+B, no Reader or training.32 historical test questions/task; all six layer groups. Offline Native KV Oracle.','',
           '| Arm | '+' | '.join(CFG['tasks'])+' | Macro |','|---|'+'---:|'*7]
    for a in arms():
        n=a['name'];lines.append('| '+n+' | '+' | '.join(f'{results[t]["metrics"][n]["accuracy"]*100:.2f}%' for t in CFG['tasks'])+f' | {macro[n]*100:.2f}% |')
    lines+=['',*['- '+s for s in CFG['caveats']]]
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='smoke':smoke()
    elif action=='evaluate':evaluate(sys.argv[2])
    elif action=='summarize':summarize()
    else:raise ValueError(action)
