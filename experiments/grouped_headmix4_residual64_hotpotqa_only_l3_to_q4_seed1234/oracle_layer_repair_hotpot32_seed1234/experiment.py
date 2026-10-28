"""Evidence-only post-RoPE oracle replacement and true layer attention masking."""
import importlib.util
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
spec=importlib.util.spec_from_file_location('oracle_frozen_parent',Path(CFG['parent'])/'experiment.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
H,R,U,C,torch=P.H,P.R,P.U,P.C,P.torch
from transformers import DynamicCache
F=torch.nn.functional


def read(p):return json.loads(Path(p).read_text())
def save(n,v):C.save(ROOT/n,v)
def identity():return C.sha(ROOT/'manifest.json')
def mark(path_name,**v):save(path_name,dict(manifest_sha256=identity(),**v))
def status(**v):save('status.json',dict(time=time.strftime('%F %T'),**v))
def rows():return C.read_jsonl(ROOT/'prepared/pilot.jsonl')
def arms():
    return [dict(name=n,kind=n,layers=[]) for n in ['translated','native','reject_all']]+[
        dict(name=f'g{g+1}_{kind}',kind=kind,layers=list(range(g*6,(g+1)*6)))
        for g in range(6) for kind in ['K','V','KV','mask']]


def prepare():
    if (ROOT/'manifest.json').exists():verify();return
    P.verify();assert P.identity()==CFG['parent_manifest_sha256']
    assert C.sha(Path(CFG['parent'])/'runs/stage_a/best.pt')==CFG['stage_a_sha256']
    assert C.sha(Path(CFG['parent'])/'runs/stage_b/last.pt')==CFG['stage_b_sha256']
    selected=P.rows('test')[:32];reserved=P.rows('test')[32:160]
    raw={d['_id']:d for d in read(H.CFG['source_json'])}
    tok={f:C.AutoTokenizer.from_pretrained(C.CFG['models'][f],local_files_only=True) for f in ['llama','qwen']}
    prepared=[]
    for old in selected:
        d=raw[old['source_id']]
        evidence='Reference information:\n'+'\n\n'.join(f'Passage {i}:\n{title}\n'+''.join(sentences).strip() for i,(title,sentences) in enumerate(d['context'],1))+'\nEnd of reference'
        row=U.make_row(old['id'],'gsm8k',dict(question=d['question']),tok['llama'],tok['qwen'],prompt=evidence,candidates=[],gold=d['answer'],normalize=False,cue='')
        assert not row['native_tail']
        frame=row['frames']['qwen']
        query='\n\nAnswer the question using the reference information. Output only the short answer, without an explanation.\n\nQuestion: '+d['question']+'\nGive only the answer.'
        _,assistant=C.chat_affixes(tok['qwen'],thinking=False)
        rendered=frame['rendered_prefix']+evidence+query+assistant
        ids=tok['qwen'].encode(rendered,add_special_tokens=False)
        body=frame['body_ids'];assert ids[:len(body)]==body
        gold=tok['qwen'].encode(rendered+d['answer'],add_special_tokens=False)
        assert gold[:len(ids)]==ids and len(gold)>len(ids)
        assert max(len(ids),len(row['frames']['llama']['full_ids']))+CFG['max_new_tokens']<CFG['max_context_tokens']
        row.update(task='oracle_hotpot_evidence',source_id=d['_id'],question=d['question'],answer=d['answer'],
            paragraph_count=len(d['context']),continuation_ids=ids[len(body):],full_native_ids=ids,
            gold_ids=gold[len(ids):],evidence_start=frame['native_prefix_length'],evidence_end=len(body))
        prepared.append(row)
    assert len({r['source_id'] for r in prepared})==32
    assert not {r['source_id'] for r in selected}&{r['source_id'] for r in reserved}
    C.write_jsonl(ROOT/'prepared/pilot.jsonl',prepared)
    C.write_jsonl(ROOT/'prepared/reserved_confirmation_ids.jsonl',[dict(id=r['id'],source_id=r['source_id']) for r in reserved])
    paths=[ROOT/'prepared/pilot.jsonl',ROOT/'prepared/reserved_confirmation_ids.jsonl',Path(CFG['parent'])/'manifest.json',Path(CFG['parent'])/'experiment.py',
        Path(CFG['parent'])/'runs/stage_a/best.pt',Path(CFG['parent'])/'runs/stage_b/last.pt',Path(H.__file__),Path(H.CFG['source_json'])]
    save('manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={p.name:C.sha(p) for p in ROOT.glob('*.py')},
        sources={str(p):C.sha(p) for p in paths},arms=arms(),pilot_ids=[r['source_id'] for r in prepared],
        selection='First32 of the previously fixed512 dev rows, independent of predictions; next128 reserved without inference',
        interface='Only evidence is prefetched/translated. Full native query and assistant suffix recomputed for every condition. Native anchor plus first content token held common.',
        replacement_space='Physical receiver cache after k_norm and RoPE, same token/position grid; no re-rotation',training=False,reader=False))
    mark('prepared/completed.json',status='completed',n=32,configurations=27,generations=864,reserved_n=128)


def verify():
    P.verify();m=read(ROOT/'manifest.json')
    assert C.sha(ROOT/'config.json')==m['config_sha256']
    for n,d in m['code'].items():assert C.sha(ROOT/n)==d,n
    for p,d in m['sources'].items():assert C.sha(p)==d,p


def mix_cache(model,translated,native,layers=(),roles=()):
    data=[]
    for layer in range(36):
        tk,tv=H._cache_layer(translated,layer);nk,nv=H._cache_layer(native,layer)
        # Both caches have the same immutable native anchor; outside the evidence replacement span they are equal.
        data.append(((nk if layer in layers and 'K' in roles else tk).detach().clone(),
                     (nv if layer in layers and 'V' in roles else tv).detach().clone()))
    return DynamicCache(ddp_cache_data=data,config=model.config)


class Engine:
    def __init__(self):
        self.tok,self.model,self.sender=P.models()
        self.translator=P.fresh_b().eval()
        cp=torch.load(Path(CFG['parent'])/'runs/stage_b/last.pt',map_location='cpu',weights_only=True)
        assert cp['manifest_sha256']==P.identity() and cp['frozen_stage_a_sha256']==CFG['stage_a_sha256']
        self.translator.residual.load_state_dict(cp['state'],strict=True);del cp
        self.translator.requires_grad_(False)
        assert self.model.config.num_hidden_layers==36 and self.model.config.num_key_value_heads==8
        assert self.model.config.num_attention_heads==32
        self.versions=[H.frozen_versions(m) for m in [self.model,self.sender,self.translator]]
        self.blocked_layers=set();self.context=None;self.hooks=[];self.mask_calls=0
        for i,layer in enumerate(self.model.model.layers):
            def hook(module,args,kwargs,layer=i):
                if layer not in self.blocked_layers:return
                assert self.context is not None
                base,count,start,end=self.context
                q=torch.arange(base,base+count,device='cuda')[:,None]
                k=torch.arange(base+count,device='cuda')[None,:]
                allowed=(k<=q)&~((k>=start)&(k<end))
                mask=torch.zeros((count,base+count),dtype=self.model.dtype,device='cuda').masked_fill(~allowed,float('-inf'))
                kwargs=dict(kwargs);kwargs['attention_mask']=mask[None,None];self.mask_calls+=1
                return args,kwargs
            self.hooks.append(layer.self_attn.register_forward_pre_hook(hook,with_kwargs=True))
        eos=self.model.generation_config.eos_token_id
        self.stops=set(eos if isinstance(eos,list) else [eos]);self.stops.update([self.tok.eos_token_id,self.tok.convert_tokens_to_ids('<|im_end|>')]);self.stops.discard(None)

    def unchanged(self):
        assert [H.frozen_versions(m) for m in [self.model,self.sender,self.translator]]==self.versions
        assert all(p.grad is None and not p.requires_grad for m in [self.model,self.sender,self.translator] for p in m.parameters())

    @torch.no_grad()
    def caches(self,row):
        assert not self.blocked_layers
        native=U.native_body(self.model,row)
        source=P.capture_pair(self.sender,self.model,row)
        k,v=P.map_source(self.translator,source)
        assert torch.isfinite(k).all() and torch.isfinite(v).all()
        translated=U.translated_cache(self.model,row,native,k,v)
        for layer in range(36):
            for a,b in zip(H._cache_layer(translated,layer),H._cache_layer(native,layer)):
                assert a.shape==b.shape and torch.equal(a[:,:,:row['evidence_start']],b[:,:,:row['evidence_start']])
        return native,translated

    def forward(self,ids,cache,base,row,layers=(),use_cache=True):
        self.context=(base,len(ids),row['evidence_start'],row['evidence_end']);self.blocked_layers=set(layers)
        try:
            t=torch.tensor([ids],device='cuda')
            return self.model.model(t,past_key_values=cache,attention_mask=torch.ones(1,base+len(ids),device='cuda',dtype=torch.long),
                position_ids=torch.arange(base,base+len(ids),device='cuda')[None],use_cache=use_cache)
        finally:self.context=None;self.blocked_layers=set()

    @torch.no_grad()
    def first_logits(self,row,cache,layers=()):
        out=self.forward(row['continuation_ids'],U.clone_cache(cache,self.model),row['evidence_end'],row,layers)
        return self.model.lm_head(out.last_hidden_state[0,-1]).float()

    @torch.no_grad()
    def generate(self,row,cache,layers=()):
        out=self.forward(row['continuation_ids'],U.clone_cache(cache,self.model),row['evidence_end'],row,layers)
        total=row['evidence_end']+len(row['continuation_ids']);tokens=[];stopped=False
        for i in range(CFG['max_new_tokens']):
            logits=self.model.lm_head(out.last_hidden_state[0,-1]).float();assert torch.isfinite(logits).all()
            token=int(logits.argmax());tokens.append(token)
            if token in self.stops:stopped=True;break
            if i+1<CFG['max_new_tokens']:
                out=self.forward([token],out.past_key_values,total,row,layers);total+=1
        text=self.tok.decode(tokens,skip_special_tokens=False);pred=H.clean_prediction(text)
        return dict(generation=text,prediction=pred,generated_ids=tokens,generated_tokens=len(tokens),reached_cap=int(not stopped),**H.answer_metrics(pred,row['answer']))

    @torch.no_grad()
    def nll(self,row,cache,layers=()):
        prefix=row['continuation_ids'];gold=row['gold_ids']
        out=self.forward(prefix+gold[:-1],U.clone_cache(cache,self.model),row['evidence_end'],row,layers,use_cache=False)
        logits=self.model.lm_head(out.last_hidden_state[0,len(prefix)-1:]).float()
        value=F.cross_entropy(logits,torch.tensor(gold,device='cuda'),reduction='none')
        assert torch.isfinite(value).all()
        return dict(gold_answer_nll=float(value.mean()),gold_answer_sum_nll=float(value.sum()),gold_answer_tokens=len(gold))


def cache_for(engine,row,native,translated,arm):
    kind=arm['kind'];layers=arm['layers']
    if kind=='native':return mix_cache(engine.model,translated,native,range(36),'KV'),[]
    if kind=='translated':return mix_cache(engine.model,translated,native),[]
    if kind=='reject_all':return mix_cache(engine.model,translated,native),list(range(36))
    if kind=='mask':return mix_cache(engine.model,translated,native),layers
    return mix_cache(engine.model,translated,native,layers,kind),[]


@torch.no_grad()
def smoke(engine):
    checks=[]
    for row in {r['id']:r for r in [rows()[0],max(rows(),key=lambda r:len(r['full_native_ids']))]}.values():
        native,translated=engine.caches(row)
        copy_t=mix_cache(engine.model,translated,native)
        copy_n=mix_cache(engine.model,translated,native,range(36),'KV')
        assert all(torch.equal(a,b) for cache,ref in [(copy_t,translated),(copy_n,native)] for layer in range(36) for a,b in zip(H._cache_layer(cache,layer),H._cache_layer(ref,layer)))
        assert engine.generate(row,copy_t)==engine.generate(row,translated)
        assert engine.generate(row,copy_n)==engine.generate(row,native)
        for cache in [native,translated]:
            reference,count=C.R.generate(C.runtime(engine.model,engine.tok),torch.tensor([row['continuation_ids']],device='cuda'),CFG['max_new_tokens'],
                past_key_values=U.clone_cache(cache,engine.model),past_length=row['evidence_end'],temperature=0.,seed=CFG['seed'])
            direct=engine.generate(row,cache)
            assert direct['generation']==reference and direct['generated_tokens']==count, 'Existing generation interface mismatch'
        out=engine.model.model(torch.tensor([row['full_native_ids']],device='cuda'),use_cache=False)
        full=engine.model.lm_head(out.last_hidden_state[0,-1]).float();cached=engine.first_logits(row,native)
        parity=float(U.full_kl(full[None],cached[None]));assert parity<U.CFG['native_parity_kl_max'] and int(full.argmax())==int(cached.argmax())
        altered=U.clone_cache(translated,engine.model)
        for layer in range(6):
            for tensor in H._cache_layer(altered,layer):tensor[:,:,row['evidence_start']:row['evidence_end']].normal_()
        assert torch.equal(engine.first_logits(row,translated,range(6)),engine.first_logits(row,altered,range(6))), 'Masked KV affected logits'
        assert engine.mask_calls>0
        # K-only and V-only swaps must leave the other role bitwise unchanged.
        for role,other in [('K',1),('V',0)]:
            mixed=mix_cache(engine.model,translated,native,range(6),role)
            assert all(torch.equal(H._cache_layer(mixed,i)[other],H._cache_layer(translated,i)[other]) for i in range(36))
            value=engine.generate(row,mixed);value.update(engine.nll(row,mixed));del mixed
        checks.append(dict(id=row['id'],native_cache_kl=parity,mask_content_invariant=True,endpoint_generation_exact=True))
        del native,translated,copy_t,copy_n,altered,out,full,cached
    engine.unchanged();mark('smoke/completed.json',status='completed',checks=checks)
    print(time.strftime('%F %T'),'SMOKE PASSED',json.dumps(checks),flush=True)


def evaluate():
    verify();engine=Engine();status(status='running',phase='smoke');smoke(engine)
    records=[];started=time.monotonic()
    for i,row in enumerate(rows(),1):
        native,translated=engine.caches(row);values={}
        for arm in arms():
            name=arm['name'];path='evaluation/per_arm/'+name+'/'+row['id']+'.json'
            if (ROOT/path).exists():
                item=read(ROOT/path);assert item['manifest_sha256']==identity();value=item['result']
            else:
                cache,blocked=cache_for(engine,row,native,translated,arm)
                value=engine.generate(row,cache,blocked);value.update(engine.nll(row,cache,blocked))
                mark(path,id=row['id'],arm=arm,result=value);del cache
            values[name]=value
            status(status='running',phase='evaluation',sample=i,total=32,arm=name,elapsed_seconds=time.monotonic()-started)
        records.append(dict(id=row['id'],source_id=row['source_id'],question=row['question'],gold=row['answer'],conditions=values))
        mark('evaluation/per_sample/'+row['id']+'.json',**records[-1]);del native,translated
        print(time.strftime('%F %T'),f'EVALUATION {i}/32',flush=True)
    engine.unchanged();verify();C.write_jsonl(ROOT/'evaluation/per_sample.jsonl',records)
    metrics={a['name']:{k:sum(r['conditions'][a['name']][k] for r in records)/32 for k in ['em','f1','gold_answer_nll','reached_cap']} for a in arms()}
    gap=metrics['native']['f1']-metrics['translated']['f1'];comparisons={}
    for a in arms()[3:]:
        name=a['name'];mask_name=name.split('_')[0]+'_mask'
        improvements=[r['conditions'][name]['f1']-r['conditions']['translated']['f1'] for r in records]
        comparisons[name]=dict(delta_f1=metrics[name]['f1']-metrics['translated']['f1'],
            replaced_kv_fraction=(len(a['layers'])/36*len(a['kind'])/2) if a['kind'] in ['K','V','KV'] else 0.,
            recovery_ratio=(metrics[name]['f1']-metrics['translated']['f1'])/gap if gap>0 else None,
            paired_vs_translated=H.paired_interval(improvements,CFG['seed']),
            paired_vs_same_location_mask=H.paired_interval([r['conditions'][name]['f1']-r['conditions'][mask_name]['f1'] for r in records],CFG['seed']),
            strict_rescue_ids=[r['id'] for r in records if not r['conditions']['translated']['em'] and not r['conditions']['reject_all']['em'] and r['conditions'][name]['em']],
            translated_correct_harmed_ids=[r['id'] for r in records if r['conditions']['translated']['em'] and not r['conditions'][name]['em']])
    mark('summary.json',status='completed',n=32,metrics=metrics,comparisons=comparisons,training=False,reader=False,
         exploratory=True,reserved_confirmation_evaluated=False,caveats=CFG['caveats'])
    lines=['# Evidence-only layer-group Oracle repair pilot','','Exploratory32 questions,27 configurations. Frozen Hotpot A+B, no Reader; all distractor paragraphs preserved. All baselines recomputed on evidence-only transfer.','',
           '| Arm | EM | F1 | Gold token-mean NLL |','|---|---:|---:|---:|']
    for name,v in metrics.items():lines.append(f'| {name} | {v["em"]*100:.2f}% | {v["f1"]*100:.2f}% | {v["gold_answer_nll"]:.4f} |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='evaluate':evaluate()
    else:raise ValueError(sys.argv[1])
