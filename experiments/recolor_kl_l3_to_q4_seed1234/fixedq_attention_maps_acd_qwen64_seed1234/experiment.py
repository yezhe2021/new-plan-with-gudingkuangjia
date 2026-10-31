"""Four-way attention maps on one shared native Q/position/continuation grid."""
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text())
PARENT=Path(CFG['parent']);PREVIOUS=Path(CFG['wait_for_experiment'])
spec=importlib.util.spec_from_file_location('fixedq_previous',PREVIOUS/'experiment.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
C=P.C;O=P.O;D=P.D;torch=C.torch
METHODS=['native','A','C','D'];LAYERS=list(range(36))


def identity():return C.sha(ROOT/'manifest.json')
def mark(path,**v):C.save(ROOT/path,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows(task=None):return [r for r in C.read_jsonl(ROOT/'prepared/test.jsonl') if task is None or r['task']==task]


def prepare():
    if (ROOT/'manifest.json').exists():verify();return
    P.verify();assert P.identity()==CFG['wait_for_manifest_sha256']
    assert C.read(PREVIOUS/'completed.json')['status']=='completed'
    rr=P.rows();assert len(rr)==128
    refs=[PREVIOUS/f for f in ['manifest.json','config.json','experiment.py','diagnostics.py','prepared/test.jsonl','completed.json','smoke/completed.json']]
    refs += [PARENT/'runs'/a/'epoch4_step800.pt' for a in ['C','D']]
    for r in rr:
        for a in ['A','C','D']:
            p,_=P.previous(r,a);refs.append(p)
            for group in P.GROUPS:
                p=PREVIOUS/f'evaluation/{r["task"]}/per_group/{a}/gold_answer/{group}/{r["id"]}.json'
                v=C.read(p);assert v['manifest_sha256']==P.identity() and v['id']==r['id'];refs.append(p)
        refs.append(O.ROOT/f'evaluation/{r["task"]}/per_config/native/{r["id"]}.json')
    C.write_jsonl(ROOT/'prepared/test.jsonl',rr);refs.append(ROOT/'prepared/test.jsonl')
    examples={t:[P.rows(t)[i]['id'] for i in CFG['example_indices']] for t in CFG['test_counts']}
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),
        code={p.name:C.sha(p) for p in ROOT.glob('*.py')},sources={str(p):C.sha(p) for p in refs},
        test_ids={t:[r['id'] for r in rr if r['task']==t] for t in CFG['test_counts']},examples=examples,
        probe_context='All gold-answer prediction positions; same native Query and native nonbody tail for all4 KV sources.',
        training=False,actual_query_comparison=False))
    mark('prepared/completed.json',status='completed',n=128,layers=36,query_heads=32,examples=examples)


def verify():
    C.verify();m=C.read(ROOT/'manifest.json')
    assert C.sha(ROOT/'config.json')==m['config_sha256']
    for name,d in m['code'].items():assert C.sha(ROOT/name)==d,name
    for path,d in m['sources'].items():assert C.sha(path)==d,path


def pause_gate():
    if (ROOT/'controls/pause_requested.json').exists():status(status='paused');sys.exit(75)


def pair_metrics(lp,lq,mask,body_start,body_end):
    p,q=lp.exp(),lq.exp();diff=(lp-lq).masked_fill(~mask[None],0.)
    m=(p+q)/2;lm=m.clamp_min(1e-30).log()
    js=.5*((p*(lp-lm).masked_fill(~mask[None],0.)).sum(-1)+(q*(lq-lm).masked_fill(~mask[None],0.)).sum(-1))
    cosine=(p*q).sum(-1)/(p.square().sum(-1)*q.square().sum(-1)).sqrt().clamp_min(1e-30)
    k=min(10,body_end-body_start)
    top_p=p[:,:,body_start:body_end].topk(k,dim=-1).indices
    top_q=q[:,:,body_start:body_end].topk(k,dim=-1).indices
    overlap=(top_p[:,:,:,None]==top_q[:,:,None,:]).any(-1).float().mean(-1)
    arrays=dict(kl=(p*diff).sum(-1).clamp_min(0),js=js.clamp_min(0),tv=(p-q).abs().sum(-1)/2,
        cosine=cosine,body_top10_overlap=overlap,top1_agreement=(p.argmax(-1)==q.argmax(-1)).float())
    return dict(mean={k:float(v.mean()) for k,v in arrays.items()},
        per_head={k:v.mean(-1).cpu().tolist() for k,v in arrays.items()})


def region_labels(engine,row):
    enc=engine.tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True)
    assert enc['input_ids']==row['content_ids']
    prompt=row['prompt'];start=prompt.rfind(row['question']);assert start>=0
    marker=prompt.rfind('Question:',0,start+1)
    question_begin=marker if marker>=0 and start-marker<20 else start
    spans=[('question',question_begin,start+len(row['question']))]
    if row['task']=='arc':
        pattern=r'(?m)^('+ '|'.join(re.escape(x) for x in row['candidates'])+r')\. '
        matches=list(re.finditer(pattern,prompt));limit=prompt.find('\n\nAnswer with')
        for i,m in enumerate(matches):spans.append(('choice_'+m.group(1),m.start(),matches[i+1].start() if i+1<len(matches) else limit))
    else:
        matches=list(re.finditer(r'(?m)^Passage (\d+):',prompt))
        for i,m in enumerate(matches):spans.append(('passage_'+m.group(1),m.start(),matches[i+1].start() if i+1<len(matches) else question_begin))
    labels=[]
    for a,b in enc['offset_mapping']:
        labels.append(next((label for label,s,e in spans if b>s and a<e),'instruction'))
    return labels


def bins(prob,start,end,count=64):
    body=prob[:,:,start:end];indices=(torch.arange(end-start,device='cuda')*count//(end-start)).clamp_max(count-1)
    out=torch.zeros(body.shape[0],body.shape[1],count,device=body.device,dtype=body.dtype)
    out.scatter_add_(-1,indices[None,None].expand_as(body),body)
    return out.mean(1).cpu().tolist() # head x body-bin; sum mass, never renormalize


@torch.no_grad()
def row_maps(engine,modules,row,check=False):
    P.settings(engine,row);base=engine.base(row);native=O.native_cache(engine,row)
    bodies={a:engine.cache(row,base,m) for a,m in modules.items()}
    text=row['candidates'][row['gold']] if row['task']=='arc' else str(row['gold'])
    tokens=engine.tok.encode(text,add_special_tokens=False);positions=list(range(len(tokens)));assert tokens
    trace=D.forward_trace(C,engine,native,tokens,positions,LAYERS)
    start=len(engine.prefix_ids);end=engine._diagnostic_body_end;regions=region_labels(engine,row)
    examples=C.read(ROOT/'manifest.json')['examples'][row['task']];example=row['id'] in examples
    metrics=[];example_maps={};anchor_error=0.
    previous={}
    for a in ['A','C','D']:
        previous[a]={}
        for group,ls in P.GROUPS.items():
            v=C.read(PREVIOUS/f'evaluation/{row["task"]}/per_group/{a}/gold_answer/{group}/{row["id"]}.json')
            for item in v['diagnostics']['fixed_native_q']['per_layer']:
                previous[a][item['layer']-1]=item['cases']['TT']['route_kl']
    for layer in LAYERS:
        pause_gate();common=C._cache_layer(trace.cache,layer)[0]
        lp={};probs={};mask=None
        for a in METHODS:
            keys=common.clone()
            if a!='native':keys[:,:,start:end]=C._cache_layer(bodies[a],layer)[0][:,:,start:end]
            if check:
                assert torch.equal(keys[:,:,:start],common[:,:,:start]) and torch.equal(keys[:,:,end:],common[:,:,end:])
                expected=common if a=='native' else C._cache_layer(bodies[a],layer)[0]
                assert torch.equal(keys[:,:,start:end],expected[:,:,start:end])
            lp[a],mask=D.distribution(C,engine,layer,trace.q[layer],keys,trace.qpos);probs[a]=lp[a].exp();del keys
        distances={a:pair_metrics(lp['native'],lp[a],mask,start,end) for a in METHODS}
        assert distances['native']['mean']['kl']<1e-10 and distances['native']['mean']['tv']<1e-10
        assert abs(distances['native']['mean']['cosine']-1)<1e-5
        for a in ['A','C','D']:
            if layer in previous[a]:
                error=abs(distances[a]['mean']['kl']-previous[a][layer]);anchor_error=max(anchor_error,error)
                assert error<1e-5,('Existing fixed-Q KL mismatch',row['id'],a,layer,error)
        pairs={};residual={}
        for i,a in enumerate(METHODS):
            for b in METHODS[i+1:]:pairs[a+'__'+b]=pair_metrics(lp[a],lp[b],mask,start,end)['mean']
        for i,a in enumerate(['A','C','D']):
            for b in ['A','C','D'][i+1:]:
                da=probs[a]-probs['native'];db=probs[b]-probs['native']
                norm=(da.square().sum(-1)*db.square().sum(-1)).sqrt();valid=norm>1e-20
                value=((da*db).sum(-1)/norm.clamp_min(1e-20))[valid]
                residual[a+'__'+b]=dict(cosine=float(value.mean()) if value.numel() else None,valid_head_probes=int(valid.sum()))
        mass={}
        for a,p in probs.items():
            mass[a]=dict(header=float(p[:,:,:start].sum(-1).mean()),body=float(p[:,:,start:end].sum(-1).mean()),
                continuation=float(p[:,:,end:].sum(-1).mean()))
            for name in sorted(set(regions)):
                idx=torch.tensor([start+i for i,s in enumerate(regions) if s==name],device='cuda')
                mass[a][name]=float(p.index_select(-1,idx).sum(-1).mean())
        metrics.append(dict(layer=layer+1,to_native=distances,pairwise=pairs,residual_alignment=residual,attention_mass=mass))
        if example and layer+1 in CFG['example_layers']:
            # Same token columns for all4 maps: union of each method's top8 body
            # tokens at the first answer query, ranked by mean Query-head mass.
            chosen=set()
            for p in probs.values():chosen.update((p[:,0,start:end].mean(0).topk(min(8,end-start)).indices+start).cpu().tolist())
            chosen=sorted(chosen);index=torch.tensor(chosen,device='cuda')
            example_maps[str(layer+1)]=dict(body_bin_mass={a:bins(p,start,end,CFG['body_bins']) for a,p in probs.items()},
                token_positions=chosen,token_labels=[str(i-start)+':'+engine.tok.decode([row['content_ids'][i-start]],skip_special_tokens=False).replace('\n','\\n') for i in chosen],
                first_query_token_mass={a:p[:,0].index_select(-1,index).cpu().tolist() for a,p in probs.items()})
    free={a:P.previous(row,a)[1] for a in ['A','C','D']}
    native_record=C.read(O.ROOT/f'evaluation/{row["task"]}/per_config/native/{row["id"]}.json')
    assert native_record['manifest_sha256']==O.identity()
    free['native']=native_record['result']
    engine.unchanged()
    return dict(id=row['id'],task=row['task'],gold=row['gold'],gold_tokens=tokens,
        probe_positions=positions,body_tokens=end-start,layers=metrics,example_maps=example_maps,
        historical_task_results=free,phase1_max_route_kl_difference=anchor_error)


@torch.no_grad()
def smoke():
    verify();engine,modules,_=O.models();selected=[rows(t)[0] for t in CFG['test_counts']]
    selected.append(max(rows('hotpotqa'),key=lambda r:len(r['content_ids'])*len(engine.tok.encode(str(r['gold']),add_special_tokens=False))))
    checks=[]
    for row in {r['id']:r for r in selected}.values():
        v=row_maps(engine,modules,row,check=True)
        if v['example_maps'] and not (ROOT/'smoke/preview.png').exists():
            sys.path.insert(0,str(ROOT));from plot_results import example_figure
            example_figure(v,ROOT/'smoke/preview.png')
        checks.append(dict(id=row['id'],all36_layers=True,native_self_zero=True,
            shared_query_and_nonbody_keys=True,phase1_max_route_kl_difference=v['phase1_max_route_kl_difference']))
        C.log('FIXEDQ MAP SMOKE',row['id'],'PASSED')
    mark('smoke/completed.json',status='completed',checks=checks,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)


@torch.no_grad()
def evaluate(task):
    verify();assert C.read(ROOT/'smoke/completed.json')['manifest_sha256']==identity()
    engine,modules,_=O.models()
    for i,row in enumerate(rows(task),1):
        path=f'evaluation/{task}/per_sample/{row["id"]}.json'
        if (ROOT/path).exists():assert C.read(ROOT/path)['manifest_sha256']==identity();continue
        value=row_maps(engine,modules,row);mark(path,**value)
        status(status='running',phase='fixed_native_q_attention_maps',task=task,sample=i,total=64)
        if i%4==0:C.log('FIXEDQ MAP',task,f'{i}/64')
    verify();mark(f'evaluation/{task}/completed.json',status='completed',n=64,finished_at=time.strftime('%F %T'))


def summarize():
    verify();sys.path.insert(0,str(ROOT));from plot_results import generate
    generate(C,ROOT,CFG,identity)
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')


if __name__=='__main__':
    a=sys.argv[1]
    if a=='prepare':prepare()
    elif a=='smoke':smoke()
    elif a=='evaluate':evaluate(sys.argv[2])
    elif a=='summarize':summarize()
    else:raise ValueError(a)
