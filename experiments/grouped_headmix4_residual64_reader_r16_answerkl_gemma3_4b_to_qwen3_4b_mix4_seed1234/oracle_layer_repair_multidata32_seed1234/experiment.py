"""Same five-task layer-group Oracle protocol, using frozen Gemma-to-Qwen A+B."""
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
GROOT=Path(CFG['parent'])


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


O=load_module('frozen_llama_oracle_protocol',Path(CFG['reference_oracle'])/'experiment.py')
U,C,torch=O.U,O.C,O.torch
G=load_module('frozen_gemma_grouped_architecture',GROOT/'grouped_translator.py')
R=load_module('frozen_gemma_residual_architecture',GROOT/'residual_translator.py')
cache_layer=U._cache_layer


def read(path):return json.loads(Path(path).read_text())
def save(path,value):C.save(ROOT/path,value)
def identity():return C.sha(ROOT/'manifest.json')
def mark(path,**value):save(path,dict(manifest_sha256=identity(),**value))
def status(**value):save('status.json',dict(time=time.strftime('%F %T'),**value))
def rows(task=None):return [r for r in C.read_jsonl(ROOT/'prepared/pilot.jsonl') if task is None or r['task']==task]
def arms():return O.arms()
def historical(row):return read(GROOT/'evaluation'/row['task']/'per_sample'/(U.fname(row)+'.json'))


def verify_gemma():
    m=read(GROOT/'manifest.json');assert C.sha(GROOT/'manifest.json')==CFG['parent_manifest_sha256']
    assert C.sha(GROOT/'config.json')==m['config_sha256']
    for name,d in m['code'].items():assert C.sha(GROOT/name)==d,name
    for path,d in m['dependencies'].items():assert C.sha(path)==d,path
    for path,d in m['model_metadata'].items():assert C.sha(path)==d,path
    for split,d in m['prepared_sha256'].items():assert C.digest(C.read_jsonl(GROOT/'prepared'/(split+'.jsonl')))==d,split
    for stage,key in [('a','stage_a_sha256'),('b','stage_b_sha256')]:
        assert C.sha(GROOT/('runs/stage_'+stage)/('best.pt' if stage=='a' else 'last.pt'))==CFG[key]


def prepare():
    if (ROOT/'manifest.json').exists():verify();return
    O.verify();assert O.identity()==CFG['reference_oracle_manifest_sha256'];verify_gemma()
    gemma_rows={r['id']:r for r in C.read_jsonl(GROOT/'prepared/test.jsonl')}
    reference=[r for r in O.rows() if r['task'] in CFG['tasks']]
    selected=[];paths=[]
    for old in reference:
        original=gemma_rows[old['id']]
        for key in ['id','task','prompt','receiver_suffix','candidates','gold','normalize','cue','suffix_ids','native_context_ids','question_key']:
            assert original[key]==old[key],('Receiver protocol mismatch',old['id'],key)
        assert original['frames']['qwen']==old['frames']['qwen'],old['id']
        row=json.loads(json.dumps(original));frame=row['frames']['qwen']
        assert frame['body_ids']+row['suffix_ids']==row['native_context_ids']
        cols=list(range(frame['native_prefix_length'],len(frame['body_ids'])))
        assert cols==old['intervention_positions'] and len(cols)==len(row['source_indices'])
        source=GROOT/'source'/(U.fname(row)+'.pt');assert source.exists(),('Missing original Gemma source',row['id'])
        row.update(cache_length=len(frame['body_ids']),intervention_positions=cols,source_row_sha256=C.digest(original))
        h=historical(row);assert h['manifest_sha256']==CFG['parent_manifest_sha256']
        for key in ['id','task','prompt','gold']:assert h[key]==row[key]
        assert set(['qwen_native','grouped_residual']).issubset(h['conditions'])
        selected.append(row);paths.append(GROOT/'evaluation'/row['task']/'per_sample'/(U.fname(row)+'.json'))
    assert len(selected)==160 and all(sum(r['task']==t for r in selected)==32 for t in CFG['tasks'])
    assert [r['id'] for r in selected]==[r['id'] for r in reference]
    C.write_jsonl(ROOT/'prepared/pilot.jsonl',selected)
    paths += [ROOT/'prepared/pilot.jsonl',GROOT/'manifest.json',GROOT/'config.json',GROOT/'prepared/test.jsonl',
              GROOT/'runs/stage_a/best.pt',GROOT/'runs/stage_b/last.pt',GROOT/'grouped_translator.py',GROOT/'residual_translator.py',
              Path(O.__file__),Path(CFG['reference_oracle'])/'manifest.json',Path(CFG['reference_oracle'])/'summary.json',
              Path(CFG['reference_oracle'])/'prepared/pilot.jsonl']
    save('manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),
        code={p.name:C.sha(p) for p in ROOT.glob('*.py')},sources={str(p):C.sha(p) for p in paths},arms=arms(),
        ids={t:[r['id'] for r in selected if r['task']==t] for t in CFG['tasks']},n=160,
        protocol='Exact same five-task32 IDs and receiver prompt/frame/suffix/scorer as Llama Oracle. Frozen Gemma Stage-A+B only, no Reader.',
        source_policy='Reuse identity/row-hash-checked original Gemma FP32-compute/FP16-storage source caches. No Gemma forward, recapture or training.',
        intervention='Physical post-RoPE Qwen K/V, all originally translated body positions including Question/options; native anchor and suffix untouched.',
        training=False,reader=False,gsm8k=False))
    mark('prepared/completed.json',status='completed',n=160,configurations=27,total_sample_configurations=4320)
    print(time.strftime('%F %T'),'PREPARED 160, SAME LLAMA IDS',flush=True)


def verify():
    O.verify();verify_gemma();m=read(ROOT/'manifest.json')
    assert C.sha(ROOT/'config.json')==m['config_sha256']
    for name,d in m['code'].items():assert C.sha(ROOT/name)==d,name
    for path,d in m['sources'].items():assert C.sha(path)==d,path


def load_translator():
    config=read(GROOT/'config.json')
    a=torch.load(GROOT/'runs/stage_a/best.pt',map_location='cpu',weights_only=True)
    assert a['manifest_sha256']==CFG['parent_manifest_sha256'] and a['stage']=='a'
    base=G.GroupedKVTranslator(**config['architecture']);assert sum(p.numel() for p in base.parameters())==146800640
    base.load_state_dict(a['state'],strict=True);del a
    module=R.FrozenTranslatorWithResidual(base,64)
    b=torch.load(GROOT/'runs/stage_b/last.pt',map_location='cpu',weights_only=True)
    assert b['manifest_sha256']==CFG['parent_manifest_sha256'] and b['state_scope']=='cross_head_residual_only'
    assert b['frozen_stage_a_sha256']==CFG['stage_a_sha256'] and b['step']==1024
    module.residual.load_state_dict(b['state'],strict=True);del b
    return module.cuda().eval().requires_grad_(False)


class Engine(O.Engine):
    def __init__(self):
        C.seed();self.tok,self.model=C.model('qwen');self.translator=load_translator()
        self.sender=None;self.snapshots=[O.versions(m) for m in [self.model,self.translator]]
        assert (self.model.config.num_hidden_layers,self.model.config.num_attention_heads,self.model.config.num_key_value_heads)==(36,32,8)
        assert all('LoRA' not in type(m).__name__ for m in self.model.modules())
        self.blocked=set();self.context=None;self.mask_calls=0;self.hooks=[]
        for i,layer in enumerate(self.model.model.layers):
            def hook(module,args,kwargs,index=i):
                if index not in self.blocked:return
                assert self.context is not None
                base,count,cols=self.context
                q=torch.arange(base,base+count,device='cuda')[:,None];k=torch.arange(base+count,device='cuda')[None,:]
                allowed=(k<=q);allowed[:,cols]=False
                mask=torch.zeros(count,base+count,device='cuda',dtype=self.model.dtype).masked_fill(~allowed,float('-inf'))
                kwargs=dict(kwargs);kwargs['attention_mask']=mask[None,None];self.mask_calls+=1
                return args,kwargs
            self.hooks.append(layer.self_attn.register_forward_pre_hook(hook,with_kwargs=True))

    @torch.no_grad()
    def caches(self,row):
        assert not self.blocked
        native=U.native_body(self.model,row)
        source=torch.load(GROOT/'source'/(U.fname(row)+'.pt'),map_location='cpu',weights_only=True)
        assert source['manifest_sha256']==CFG['parent_manifest_sha256'] and source['id']==row['id']
        assert source['row_sha256']==row['source_row_sha256']
        assert source['key'].shape==source['value'].shape==(34,len(row['source_indices']),4,256)
        assert source['key'].dtype==source['value'].dtype==torch.float16
        pieces=[]
        for start in range(0,source['key'].shape[1],64):
            with torch.amp.autocast('cuda',dtype=torch.float16):
                pair=self.translator(source['key'][:,start:start+64][None].cuda(),source['value'][:,start:start+64][None].cuda())
            pieces.append(pair)
        k,v=tuple(torch.cat([p[j][0] for p in pieces],dim=1) for j in range(2))
        assert k.shape==v.shape==(36,len(row['intervention_positions']),8,128)
        assert torch.isfinite(k).all() and torch.isfinite(v).all()
        translated=U.translated_cache(self.model,row,native,k,v)
        for l in range(36):
            for a,b in zip(cache_layer(translated,l),cache_layer(native,l)):
                assert a.shape==b.shape and torch.equal(a[:,:,:row['intervention_positions'][0]],b[:,:,:row['intervention_positions'][0]])
        return native,translated


def baseline_parity(row,arm,value):
    expected=historical(row)['conditions']['grouped_residual' if arm=='translated' else 'qwen_native']
    assert value['prediction']==expected['prediction'] and value['correct']==expected['correct'],('Gemma historical prediction mismatch',row['id'],arm)
    error=max(abs(a-b) for a,b in zip(value['scores'],expected['scores']))
    assert error<CFG['candidate_score_parity_abs_max'],('Gemma historical score mismatch',row['id'],arm,error)


@torch.no_grad()
def smoke():
    verify();engine=Engine();checks=[]
    for task in CFG['tasks']:
        row=rows(task)[0];native,translated=engine.caches(row);cols=row['intervention_positions'];anchor=cols[0]
        for layers,roles,reference in [((),(),translated),(range(36),'KV',native)]:
            copy=O.mix_cache(engine.model,translated,native,layers,roles,cols)
            assert all(torch.equal(a,b) for l in range(36) for a,b in zip(cache_layer(copy,l),cache_layer(reference,l)));del copy
        for name,cache in [('translated',translated),('native',native)]:baseline_parity(row,name,engine.score(row,cache))
        out=engine.model.model(torch.tensor([row['native_context_ids']],device='cuda'),use_cache=False)
        full=engine.model.lm_head(out.last_hidden_state[0,-1]).float();cached=engine.first_logits(row,native)
        kl=float(U.full_kl(full[None],cached[None]));assert kl<U.CFG['native_parity_kl_max'] and int(full.argmax())==int(cached.argmax())
        for role,other in [('K',1),('V',0),('KV',None)]:
            mixed=O.mix_cache(engine.model,translated,native,range(6),role,cols)
            for l in range(36):
                for j,(a,b) in enumerate(zip(cache_layer(mixed,l),cache_layer(translated,l))):
                    assert torch.equal(a[:,:,:anchor],b[:,:,:anchor])
                    if l>=6 or j==other:assert torch.equal(a,b)
                    else:assert torch.equal(a[:,:,cols],cache_layer(native,l)[j][:,:,cols])
            del mixed
        changed=U.clone_cache(translated,engine.model)
        for l in range(6):
            for tensor in cache_layer(changed,l):tensor[:,:,cols,:]=torch.randn_like(tensor[:,:,cols,:])
        assert torch.equal(engine.first_logits(row,translated,range(6)),engine.first_logits(row,changed,range(6)))
        assert engine.mask_calls>0;engine.unchanged()
        checks.append(dict(task=task,id=row['id'],historical_baselines_exact=True,native_cached_full_kl=kl,
                           role_and_layer_isolation=True,anchor_unchanged=True,masked_content_invariant=True,reader_absent=True))
        del native,translated,changed,out,full,cached
        print(time.strftime('%F %T'),'SMOKE',task,'PASSED',flush=True)
    mark('smoke/completed.json',status='completed',checks=checks,frozen_models=True)


@torch.no_grad()
def evaluate(task):
    verify();assert read(ROOT/'smoke/completed.json')['manifest_sha256']==identity()
    engine=Engine();records=[];started=time.monotonic();done=0
    for i,row in enumerate(rows(task),1):
        native,translated=engine.caches(row);values={}
        for arm in arms():
            path=f'evaluation/{task}/per_arm/{arm["name"]}/{U.fname(row)}.json'
            if (ROOT/path).exists():
                item=read(ROOT/path);assert item['manifest_sha256']==identity();value=item['result']
            else:
                cache,blocked=O.cache_for(engine,row,native,translated,arm)
                value=engine.score(row,cache,blocked);del cache
                mark(path,id=row['id'],task=task,arm=arm,result=value)
            if arm['kind'] in ['translated','native']:baseline_parity(row,arm['kind'],value)
            values[arm['name']]=value;done+=1
            status(status='running',phase='evaluation',task=task,sample=i,total=32,arm=arm['name'],configurations_completed=done,
                   task_configurations=864,elapsed_seconds=time.monotonic()-started)
        result=dict(id=row['id'],task=task,gold=row['gold'],conditions=values);records.append(result)
        mark(f'evaluation/{task}/per_sample/{U.fname(row)}.json',**result)
        del native,translated;engine.unchanged();print(time.strftime('%F %T'),f'EVALUATION {task} {i}/32',flush=True)
    verify();C.write_jsonl(ROOT/f'evaluation/{task}/per_sample.jsonl',records)
    metrics={a['name']:dict(n=32,correct=sum(r['conditions'][a['name']]['correct'] for r in records),
        accuracy=sum(r['conditions'][a['name']]['correct'] for r in records)/32) for a in arms()}
    comparisons={};gap=metrics['native']['accuracy']-metrics['translated']['accuracy']
    for arm in arms()[3:]:
        name=arm['name'];mask=name.split('_')[0]+'_mask';delta=metrics[name]['accuracy']-metrics['translated']['accuracy']
        comparisons[name]=dict(delta_accuracy=delta,recovery_ratio=delta/gap if gap>0 else None,
            paired_vs_translated=O.paired([r['conditions'][name]['correct']-r['conditions']['translated']['correct'] for r in records]),
            paired_vs_mask=O.paired([r['conditions'][name]['correct']-r['conditions'][mask]['correct'] for r in records]),
            rescued_ids=[r['id'] for r in records if r['conditions'][name]['correct'] and not r['conditions']['translated']['correct']],
            harmed_ids=[r['id'] for r in records if not r['conditions'][name]['correct'] and r['conditions']['translated']['correct']],
            strict_rescue_ids=[r['id'] for r in records if r['conditions'][name]['correct'] and not any(r['conditions'][z]['correct'] for z in ['translated','reject_all',mask])])
    mark(f'evaluation/{task}/summary.json',status='completed',task=task,metrics=metrics,comparisons=comparisons,all32_historical_baselines_checked=True)
    mark(f'evaluation/{task}/completed.json',status='completed',finished_at=time.strftime('%F %T'))


def summarize():
    verify();results={t:read(ROOT/f'evaluation/{t}/summary.json') for t in CFG['tasks']}
    assert all(x['manifest_sha256']==identity() and x['status']=='completed' for x in results.values())
    macro={a['name']:sum(v['metrics'][a['name']]['accuracy'] for v in results.values())/len(results) for a in arms()}
    previous=read(Path(CFG['reference_oracle'])/'summary.json')
    mark('summary.json',status='completed',n=160,configurations=27,tasks=results,macro_accuracy=macro,
         reference_llama_macro=previous['macro_accuracy'],same_receiver_and_test_ids=True,
         training=False,reader=False,gsm8k=False,exploratory=True,confirmation_evaluated=False,caveats=CFG['caveats'])
    lines=['# Gemma to Qwen Oracle layer replacement, five tasks','',
           'Existing frozen Gemma Stage-A+B, no Reader/training/GSM8K. Same160 questions and receiver protocol as previous Llama Oracle.','',
           '| Arm | '+' | '.join(CFG['tasks'])+' | Macro |','|---|'+'---:|'*6]
    for a in arms():
        name=a['name'];lines.append('| '+name+' | '+' | '.join(f'{results[t]["metrics"][name]["accuracy"]*100:.2f}%' for t in CFG['tasks'])+f' | {macro[name]*100:.2f}% |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='smoke':smoke()
    elif action=='evaluate':evaluate(sys.argv[2])
    elif action=='summarize':summarize()
    else:raise ValueError(action)
