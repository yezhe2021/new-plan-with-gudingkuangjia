"""C-only32/task; error atlas + logit contribution analysis, no dimension repair."""
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text())
PARENT=Path(CFG['parent']);PREVIOUS=PARENT/CFG['mechanism_experiment'];MAPS=PARENT/CFG['maps_experiment']
spec=importlib.util.spec_from_file_location('finegrained_previous',PREVIOUS/'experiment.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
C=P.C;D=P.D;O=P.O;torch=C.torch
sys.path.insert(0,str(ROOT));import math_ops as M


def identity():return C.sha(ROOT/'manifest.json')
def mark(path,**v):C.save(ROOT/path,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows(task=None):return [r for r in C.read_jsonl(ROOT/'prepared/test.jsonl') if task is None or r['task']==task]


def source(row,context):
    p=PREVIOUS/f'evaluation/{row["task"]}/per_group/C/{context}/g1/{row["id"]}.json'
    v=C.read(p);assert v['manifest_sha256']==CFG['mechanism_manifest_sha256'] and v['id']==row['id']
    assert v['method']=='C' and v['context']==context
    return p,v


def prepare():
    if (ROOT/'manifest.json').exists():verify();return
    P.verify();assert P.identity()==CFG['mechanism_manifest_sha256']
    assert C.read(PREVIOUS/'completed.json')['status']=='completed'
    assert C.sha(MAPS/'manifest.json')==CFG['maps_manifest_sha256']
    assert C.read(MAPS/'completed.json')['status']=='completed'
    rr=[r for task,n in CFG['test_counts'].items() for r in P.rows(task)[:n]];assert len(rr)==64
    refs=[PREVIOUS/f for f in ['manifest.json','experiment.py','diagnostics.py','config.json','prepared/test.jsonl','completed.json','smoke/completed.json']]
    refs += [MAPS/f for f in ['manifest.json','completed.json']]
    refs += [PARENT/f for f in ['manifest.json','core.py','train.py','evaluate.py','runs/C/epoch4_step800.pt']]
    refs.append(ROOT/'design.md')
    for r in rr:
        for context in CFG['contexts']:
            p,_=source(r,context);refs.append(p)
        p=MAPS/f'evaluation/{r["task"]}/per_sample/{r["id"]}.json'
        v=C.read(p);assert v['manifest_sha256']==CFG['maps_manifest_sha256'] and v['id']==r['id'];refs.append(p)
    C.write_jsonl(ROOT/'prepared/test.jsonl',rr);refs.append(ROOT/'prepared/test.jsonl')
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),
        code={p.name:C.sha(p) for p in ROOT.glob('*.py')},sources={str(p):C.sha(p) for p in refs},
        test_ids={t:[r['id'] for r in rr if r['task']==t] for t in CFG['test_counts']},
        example_ids={t:[P.rows(t)[i]['id'] for i in CFG['example_indices']] for t in CFG['test_counts']},
        training=False,method='C',experiments=[1,2],dimension_interventions=False,independent_validation=False,
        preregistered_at=time.strftime('%F %T')))
    mark('prepared/completed.json',status='completed',n=64,method='C',layers=36,kv_heads=8,dimensions=128)


def verify():
    C.verify();m=C.read(ROOT/'manifest.json')
    assert C.sha(ROOT/'config.json')==m['config_sha256']
    for name,d in m['code'].items():assert C.sha(ROOT/name)==d,name
    for path,d in m['sources'].items():assert C.sha(path)==d,path


def pause_gate():
    if (ROOT/'controls/pause_requested.json').exists():status(status='paused');sys.exit(75)


def models():
    engine=C.Engine();engine.mode='test'
    module=O.T.load_final('C').eval().requires_grad_(False)
    return engine,module


def scalar_heads(values):
    output=[]
    arrays={k:v.detach().cpu().tolist() for k,v in values.items()}
    for h in range(8):output.append({k:(float(v[h]) if C.np.isfinite(v[h]) else None) for k,v in arrays.items()})
    return output


def push(arrays,key,value):arrays.setdefault(key,[]).append(value.detach().cpu().numpy().astype('float32'))


def atlas_layer(arrays,name,delta,native):
    channel,token,head=M.error_stats(delta,native,CFG['native_scale_floor_fraction'])
    for k,v in channel.items():push(arrays,name+'__dimension__'+k,v)
    for k,v in token.items():push(arrays,name+'__token__'+k,v)
    return head


def critical_indices(delta,contrib,p_native,h):
    length=delta.shape[1];fixed=torch.linspace(0,length-1,min(64,length),device='cuda').round().long().unique()
    error=delta[h].square().sum(-1);logit=contrib['delta_logits'][h].abs().mean((0,1))
    native=p_native.reshape(8,4,p_native.shape[1],-1)[h].mean((0,1))
    count=min(16,length)
    chosen=torch.cat([fixed,error.topk(count).indices,logit.topk(count).indices,native.topk(count).indices]).unique().sort().values
    return fixed,chosen


@torch.no_grad()
def calculate(engine,module,row,check=False):
    P.settings(engine,row);base=engine.base(row);native=O.native_cache(engine,row);translated=engine.cache(row,base,module)
    start=len(engine.prefix_ids);end=engine._diagnostic_body_end;length=end-start
    tables=C._rotary_tables(engine.model,torch.arange(start,end,device='cuda'))
    arrays={};features=[];error_heads={};representative={};anchors=[];checks=[]
    is_example=row['id'] in C.read(ROOT/'manifest.json')['example_ids'][row['task']]
    traces={};contexts={}
    for context in CFG['contexts']:
        _,old=source(row,context);tokens=old['tokens'];positions=old['probe_positions']
        if context=='gold_answer':
            text=row['candidates'][row['gold']] if row['task']=='arc' else str(row['gold'])
            assert tokens==engine.tok.encode(text,add_special_tokens=False) and positions==list(range(len(tokens)))
        contexts[context]=dict(tokens=tokens,probe_positions=positions,source='immutable mechanism C prefix')
        traces[context]={
            'native_q':D.forward_trace(C,engine,native,tokens,positions,list(range(36))),
            'translated_q':D.forward_trace(C,engine,translated,tokens,positions,list(range(36)))}
    maps=C.read(MAPS/f'evaluation/{row["task"]}/per_sample/{row["id"]}.json')
    for layer in range(36):
        pause_gate();cos,sin=tables(layer)
        kt,vt=[x[0,:,start:end].float() for x in C._cache_layer(translated,layer)]
        kn,vn=[x[0,:,start:end].float() for x in C._cache_layer(native,layer)]
        delta=kt-kn;dv=vt-vn;pre=M.undo_rope(delta,cos,sin);npre=M.undo_rope(kn,cos,sin)
        errors={name:atlas_layer(arrays,name,d,n) for name,d,n in [('k_post',delta,kn),('k_pre',pre,npre),('v',dv,vn)]}
        error_lists={name:scalar_heads(v) for name,v in errors.items()}
        for h in range(8):
            error_heads.setdefault(h,[]).append(dict(layer=layer+1,**{name:v[h] for name,v in error_lists.items()}))
        if check:
            back=pre*cos.float()[None]+M.rotate_half(pre)*sin.float()[None]
            relative=D.relative_l2(back,delta);assert relative<2e-6,relative
            checks.append(dict(layer=layer+1,unrotate_rerotate_relative_l2=relative))
        for context,refs in traces.items():
            for query_mode,trace in refs.items():
                fixed=M.attention_metrics(C,D,engine,layer,trace,translated,native)
                att=C.layers_of(engine.model)[layer].self_attn
                assert getattr(att,'attn_logit_softcapping',None) is None
                contribution=M.contributions(delta,trace.q[layer],float(att.scaling),fixed['p_native'],start,end,
                    CFG['token_chunk'],CFG['probe_chunk'],check=check)
                prefix=context+'__'+query_mode
                for k,v in contribution['coordinate'].items():push(arrays,prefix+'__coordinate__'+k,v)
                push(arrays,prefix+'__pair__mean_abs',contribution['pair_mean_abs'])
                push(arrays,prefix+'__pair__error_rms',contribution['pair_error_rms'])
                for k,v in contribution['token'].items():push(arrays,prefix+'__token__'+k,v)
                heads=scalar_heads({**contribution['head'],**fixed['head']})
                for h,value in enumerate(heads):
                    features.append(dict(layer=layer+1,kv_head=h+1,context=context,query_mode=query_mode,
                        **error_lists['k_post'][h],**{ 'v_'+k:v for k,v in error_lists['v'][h].items()},**value))
                if context=='gold_answer' and query_mode=='native_q':
                    expected=maps['layers'][layer]['to_native']['C']['mean']['kl']
                    difference=abs(fixed['route_kl']-expected);anchors.append(difference)
                    assert difference<1e-5,('Prior full-map KL mismatch',row['id'],layer,difference)
                if check:
                    assert fixed['reconstruction_relative_l2']<.02,fixed['reconstruction_relative_l2']
                    # Pairwise contributions are invariant when both vectors are
                    # expressed in the key-token frame (transpose-R for Q,
                    # inverse-R for K); scalar coordinates need not be invariant.
                    q=trace.q[layer][:,0].reshape(8,4,128).float();di=delta[:,:3];dpi=pre[:,:3]
                    c=cos[:3].float();s=sin[:3].float()
                    qframe=q[:,:,None,:]*c[None,None]-M.rotate_half(q[:,:,None,:])*s[None,None]
                    post=q[:,:,None,:]*di[:,None];before=qframe*dpi[:,None]
                    pair_post=post[...,:64]+post[...,64:];pair_pre=before[...,:64]+before[...,64:]
                    pair_error=float((pair_post-pair_pre).abs().max());assert pair_error<2e-5*(1+float(pair_post.abs().max()))
                if is_example and context=='gold_answer' and layer+1 in CFG['example_layers']:
                    for h in CFG['example_kv_heads_zero_based']:
                        fixed_ix,ix=critical_indices(delta,contribution,fixed['p_native'][:,:,start:end],h)
                        key=f'L{layer+1:02d}_H{h+1:02d}'
                        if key not in representative:
                            representative[key]=dict(token_positions=ix.cpu().tolist(),fixed_plot_positions=fixed_ix.cpu().tolist(),
                                token_labels=[engine.tok.decode([row['content_ids'][i]],skip_special_tokens=False) for i in ix.cpu().tolist()])
                            for name,values in [('k_post_delta',delta),('k_pre_delta',pre),('v_delta',dv)]:
                                arrays['example__'+key+'__'+name]=values[h].index_select(0,ix).cpu().numpy()
                            scale=kn[h].square().mean(0).sqrt();floor=kn[h].square().mean().sqrt()*CFG['native_scale_floor_fraction']
                            arrays['example__'+key+'__k_normalizer']=torch.maximum(scale,floor.clamp_min(1e-6)).cpu().numpy()
                        else:assert representative[key]['token_positions']==ix.cpu().tolist() or query_mode=='translated_q'
                        # Keep exactly the SAME token grid for both Q references.
                        ix=torch.tensor(representative[key]['token_positions'],device='cuda')
                        q0=trace.q[layer].reshape(8,4,trace.qpos.numel(),128)[h,:,0].float()
                        values=q0[:,None,:]*delta[h].index_select(0,ix)[None]*float(att.scaling)
                        arrays['example__'+key+'__'+query_mode+'__first_probe_signed_contribution']=values.cpu().numpy()
                del contribution,fixed
        status(status='running',phase='finegrained_error_and_contribution',task=row['task'],id=row['id'],layer=layer+1,total_layers=36)
    summary_errors={name:[dict(layer=l+1,heads=[error_heads[h][l][name] for h in range(8)]) for l in range(36)] for name in ['k_post','k_pre','v']}
    prepared={k:(C.np.stack(v) if isinstance(v,list) else v) for k,v in arrays.items()}
    prepared.update(manifest_sha256=C.np.array(identity()),sample_id=C.np.array(row['id']))
    engine.unchanged()
    return dict(id=row['id'],task=row['task'],method='C',body_tokens=length,
        contexts=contexts,error_summary=summary_errors,functional_features=features,representatives=representative,
        phase1_native_functional={context:D.functional(C,v['native_q'],v['native_q'],contexts[context]['tokens'],contexts[context]['probe_positions']) for context,v in traces.items()},
        phase1_translated_functional={context:D.functional(C,v['translated_q'],v['native_q'],contexts[context]['tokens'],contexts[context]['probe_positions']) for context,v in traces.items()},
        max_prior_route_kl_difference=max(anchors),checks=checks),prepared


def write_arrays(path,arrays):
    assert shutil.disk_usage(ROOT).free/2**30>CFG['disk_reserve_gib']
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with temp.open('wb') as stream:C.np.savez_compressed(stream,**arrays)
    temp.replace(path)


@torch.no_grad()
def smoke():
    verify();engine,module=models();checks=[]
    selected=[rows(t)[0] for t in CFG['test_counts']]
    selected.append(max(rows('hotpotqa'),key=lambda r:len(r['content_ids'])*len(engine.tok.encode(str(r['gold']),add_special_tokens=False))))
    for row in {r['id']:r for r in selected}.values():
        value,arrays=calculate(engine,module,row,check=True)
        checks.append(dict(id=row['id'],all36_layers=True,all8KV_heads=True,all128dimensions=True,
            coordinate_roundtrip_and_paired_dot_invariance=True,exact_GQA_logit_contribution_sum=True,
            prior_fixed_query_kl_max_difference=value['max_prior_route_kl_difference']))
        if value['representatives'] and not (ROOT/'smoke/example_stats.npz').exists():
            write_arrays(ROOT/'smoke/example_stats.npz',arrays);mark('smoke/example.json',**value)
        mark('smoke/progress.json',status='running',checks=checks)
        C.log('FINEGRAINED SMOKE',row['id'],'PASSED');del value,arrays
    mark('smoke/completed.json',status='completed',checks=checks,training=False,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)


@torch.no_grad()
def evaluate(task):
    verify();assert C.read(ROOT/'smoke/completed.json')['manifest_sha256']==identity()
    engine,module=models();started=time.monotonic()
    for i,row in enumerate(rows(task),1):
        pause_gate();path=f'evaluation/{task}/per_sample/{row["id"]}.json';stats=f'evaluation/{task}/error_stats/{row["id"]}.npz'
        if (ROOT/path).exists():
            assert C.read(ROOT/path)['manifest_sha256']==identity() and (ROOT/stats).exists();continue
        value,arrays=calculate(engine,module,row)
        write_arrays(ROOT/stats,arrays);mark(path,stats_file=stats,**value);del arrays,value
        status(status='running',phase='sample_completed',task=task,sample=i,total=32,elapsed_seconds=time.monotonic()-started)
        C.log('FINEGRAINED',task,f'{i}/32')
    verify();mark(f'evaluation/{task}/completed.json',status='completed',n=32,finished_at=time.strftime('%F %T'))


def summarize():
    verify();from summarize import generate
    generate(C,ROOT,CFG,identity)
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')


if __name__=='__main__':
    a=sys.argv[1]
    if a=='prepare':prepare()
    elif a=='smoke':smoke()
    elif a=='evaluate':evaluate(sys.argv[2])
    elif a=='summarize':summarize()
    else:raise ValueError(a)
