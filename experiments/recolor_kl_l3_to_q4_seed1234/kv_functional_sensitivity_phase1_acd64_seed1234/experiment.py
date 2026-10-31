"""Stage1 mechanism diagnostic: same questions, fixed contexts, frozen A/C/D."""
import ast
import importlib.util
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text())
PARENT=Path(CFG['parent']);ORACLE=PARENT/CFG['oracle_experiment'];DONOR=PARENT/CFG['phasefixed_donor_experiment']
sys.path.insert(0,str(PARENT))
import core as C
spec=importlib.util.spec_from_file_location('mechanism_oracle',ORACLE/'experiment.py')
O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)
sys.path.insert(0,str(ROOT))
import diagnostics as D
torch=C.torch;GROUPS=CFG['groups'];LAYERS=sorted({l for g in GROUPS.values() for l in g})
CASES={'TT':'none','NT':'k','TN':'v','NN':'kv'}


def identity():return C.sha(ROOT/'manifest.json')
def mark(path,**v):C.save(ROOT/path,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows(task=None):return [r for r in C.read_jsonl(ROOT/'prepared/test.jsonl') if task is None or r['task']==task]


def previous(row,method,group=None,case='TT'):
    name='unmasked' if case=='TT' else group+'_'+CASES[case]
    path=ORACLE/'evaluation'/row['task']/'per_config'/method/name/(row['id']+'.json')
    item=C.read(path);assert item['manifest_sha256']==CFG['oracle_manifest_sha256']
    assert item['id']==row['id'] and item['task']==row['task'] and item['method']==method
    assert item['condition']['layers']==([] if case=='TT' else GROUPS[group])
    return path,item['result']


def prepare():
    if (ROOT/'manifest.json').exists():verify();return
    O.verify();assert C.identity()==CFG['parent_manifest_sha256'] and O.identity()==CFG['oracle_manifest_sha256']
    assert C.read(ORACLE/'completed.json')['status']=='completed'
    rr=[r for r in O.rows() if r['task'] in CFG['test_counts']]
    assert len(rr)==128 and all(sum(r['task']==t for r in rr)==64 for t in CFG['test_counts'])
    dm=C.read(DONOR/'manifest.json');assert C.sha(DONOR/'manifest.json')==CFG['phasefixed_donor_manifest_sha256']
    assert C.read(DONOR/'completed.json')['status']=='completed'
    assert C.sha(DONOR/'config.json')==dm['config_sha256']
    for name,d in dm['code'].items():assert C.sha(DONOR/name)==d,name
    refs=[PARENT/f for f in ['manifest.json','core.py','train.py','evaluate.py','prepared/test.jsonl']]
    refs += [PARENT/'runs'/a/'epoch4_step800.pt' for a in ['C','D']]
    refs += [ORACLE/f for f in ['manifest.json','config.json','experiment.py','prepared/test.jsonl','completed.json','smoke/completed.json']]
    refs += [DONOR/f for f in ['manifest.json','config.json','experiment.py','completed.json','smoke/completed.json']]
    for r in rr:
        for a in CFG['methods']:
            p,_=previous(r,a);refs.append(p)
            for g in GROUPS:
                for case in ['NT','TN','NN']:
                    p,_=previous(r,a,g,case);refs.append(p)
    C.write_jsonl(ROOT/'prepared/test.jsonl',rr);refs.append(ROOT/'prepared/test.jsonl')
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),
        code={p.name:C.sha(p) for p in ROOT.glob('*.py')},sources={str(p):C.sha(p) for p in refs},
        test_ids={t:[r['id'] for r in rr if r['task']==t] for t in CFG['test_counts']},
        n=128,training=False,heldout_confirmation=False,phase=1,
        preregistered_at=time.strftime('%F %T'),functional_modes=['actual_q','clamped_native_q'],
        contexts=['gold_answer','student_prefix'],native_and_oracle_task_scores='reused immutable prior results'))
    mark('prepared/completed.json',status='completed',n=128,methods=3,groups=3,contexts=2)
    C.log('MECHANISM PREPARED: existing ARC64/Hotpot64, no training')


def verify():
    C.verify();m=C.read(ROOT/'manifest.json')
    assert C.sha(ROOT/'config.json')==m['config_sha256']
    for name,d in m['code'].items():assert C.sha(ROOT/name)==d,name
    for path,d in m['sources'].items():assert C.sha(path)==d,path


def pause_gate():
    if (ROOT/'controls/pause_requested.json').exists():
        status(status='paused',phase='paused_by_user');sys.exit(75)


@torch.no_grad()
def phase_audit(engine):
    # Execute ONLY the two reviewed mathematical functions, not the donor runner.
    tree=ast.parse((DONOR/'experiment.py').read_text())
    chosen=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['rotate_half','reposition_key']]
    assert len(chosen)==2
    namespace={'torch':torch};exec(compile(ast.Module(body=chosen,type_ignores=[]),str(DONOR/'experiment.py'),'exec'),namespace)
    positions=torch.arange(7,23,device='cuda');other=positions+11
    from heterofold.attention import rotary_cos_sin,apply_rope
    cos,sin=rotary_cos_sin(engine.model,positions,0,torch.float32)
    dstc,dsts=rotary_cos_sin(engine.model,other,0,torch.float32)
    raw=torch.randn(16,8,128,device='cuda',dtype=torch.float32)
    source=apply_rope(raw,cos,sin).permute(1,0,2)[None]
    corrected=namespace['reposition_key'](source,cos,sin,dstc,dsts)
    direct=apply_rope(raw,dstc,dsts).permute(1,0,2)[None]
    error=D.relative_l2(corrected,direct);assert error<1e-5,error
    mark('smoke/donor_phase_audit.json',status='completed',source_manifest_sha256=CFG['phasefixed_donor_manifest_sha256'],
        source_code_sha256=C.sha(DONOR/'experiment.py'),source_phasefix_completed=C.read(DONOR/'completed.json'),
        fp32_direct_position_reconstruction_relative_l2=error,
        old_unrephased_donor_results_excluded=True,new_donor_evaluations=False,
        caveat='Phase correction removes rotation-position contamination, not all length/resampling/content confounds.')


def settings(engine,row):
    engine._diagnostic_suffix=row['suffix_ids'];engine._diagnostic_body_end=len(engine.prefix_ids)+len(row['content_ids'])
    if row['task']=='arc':
        ids=[engine.tok.encode(v,add_special_tokens=False) for v in row['candidates']]
        assert all(len(v)==1 for v in ids),'First-probe choice margins require single-token labels'
        engine._diagnostic_candidate_ids=[v[0] for v in ids];engine._diagnostic_gold_index=row['gold']
    else:engine._diagnostic_candidate_ids=None;engine._diagnostic_gold_index=None


@torch.no_grad()
def contexts(engine,row,original,method):
    gold=row['candidates'][row['gold']] if row['task']=='arc' else str(row['gold'])
    gold_ids=engine.tok.encode(gold,add_special_tokens=False);assert gold_ids
    rollout=engine.rollout(row,C.clone_cache(original,engine.model),cap=64);assert rollout
    if row['task']=='hotpotqa':
        _,old=previous(row,method)
        assert len(rollout)==old['generated_tokens']
        assert engine.tok.decode(rollout,skip_special_tokens=False)==old['generation'],('Student prefix does not replay ordinary inference',row['id'],method)
    positions=C.positions_for(rollout,row,0,validation=True)
    return dict(gold_answer=(gold_ids,list(range(len(gold_ids)))),student_prefix=(rollout,positions))


def distances(original,native,layers,start,end):
    numerator=[0.,0.];denominator=[0.,0.];per_layer=[]
    for l in layers:
        values={}
        for i,name in enumerate(['k','v']):
            a=C._cache_layer(original,l)[i][:,:,start:end].float()
            b=C._cache_layer(native,l)[i][:,:,start:end].float()
            n=float((a-b).square().sum());d=float(b.square().sum());numerator[i]+=n;denominator[i]+=d
            values[name+'_relative_l2']=(n/max(d,1e-20))**.5
        per_layer.append(dict(layer=l+1,**values))
    return dict(k_relative_l2=(numerator[0]/max(denominator[0],1e-20))**.5,
        v_relative_l2=(numerator[1]/max(denominator[1],1e-20))**.5,
        kv_relative_l2=(sum(numerator)/max(sum(denominator),1e-20))**.5,per_layer=per_layer)


def fixed_metrics(engine,original,native,trace,layers):
    per_layer=[dict(layer=l+1,**D.fixed_grid(C,engine,l,trace,original,native)) for l in layers]
    return dict(cases={case:D.mean_scalars([p['cases'][case] for p in per_layer]) for case in CASES},
        output_interaction_nmse=sum(p['output_interaction_nmse'] for p in per_layer)/len(layers),
        output_damage_synergy=sum(p['output_damage_synergy'] for p in per_layer)/len(layers),per_layer=per_layer)


def actual_metrics(engine,trace,native_trace,translated_trace,layers):
    per_layer=[dict(layer=l+1,**D.actual_metrics(C,engine,l,trace,native_trace,translated_trace)) for l in layers]
    return dict(mean=D.mean_scalars(per_layer),per_layer=per_layer)


def synergy(values,key):return values['NT'][key]+values['TN'][key]-values['TT'][key]-values['NN'][key]


@torch.no_grad()
def group_run(engine,row,original,native,nt,tt,tokens,positions,group,check=False):
    layers=GROUPS[group]
    output=dict(fixed_native_q=fixed_metrics(engine,original,native,nt,layers),
        fixed_translated_q=fixed_metrics(engine,original,native,tt,layers),actual_q={},clamped_native_q={})
    for branch in ['actual_q','clamped_native_q']:
        for case,mode in CASES.items():
            pause_gate()
            if branch=='actual_q' and case=='TT':trace=tt
            else:
                body=C.clone_cache(original,engine.model) if case=='TT' else O.replace(engine,row,original,native,layers,mode)
                if check:
                    for l in range(36):
                        for i,t in enumerate(C._cache_layer(body,l)):
                            old=C._cache_layer(original,l)[i];s=len(engine.prefix_ids)
                            assert torch.equal(t[:,:,:s],old[:,:,:s])
                            selected=l in layers and case!='TT' and (case=='NN' or i==(0 if case=='NT' else 1))
                            expected=C._cache_layer(native,l)[i] if selected else old
                            assert torch.equal(t[:,:,s:],expected[:,:,s:])
                trace=D.forward_trace(C,engine,body,tokens,positions,LAYERS,
                    clamp=nt if branch=='clamped_native_q' else None,clamp_layers=layers)
                del body
            f=D.functional(C,trace,nt,tokens,positions)
            a=actual_metrics(engine,trace,nt,tt,layers)
            if check:
                assert max(v['reconstruction_relative_l2'] for v in a['per_layer'])<.02,a
                assert all(C.np.isfinite(v) for v in f.values() if isinstance(v,(int,float)))
            output[branch][case]=dict(functional=f,attention=a)
            if trace is not tt:del trace
    output['functional_interactions']={branch:{key:synergy({c:output[branch][c]['functional'] for c in CASES},key)
        for key in ['next_token_kl','target_token_nll']} for branch in ['actual_q','clamped_native_q']}
    return output


@torch.no_grad()
def smoke():
    verify();engine,modules,_=O.models();phase_audit(engine);checks=[]
    selected=[min(rows(t),key=lambda r:len(r['content_ids'])) for t in CFG['test_counts']]
    selected.append(max(rows('hotpotqa'),key=lambda r:len(r['content_ids'])))
    selected.append(max(rows('hotpotqa'),key=lambda r:len(r['content_ids'])*len(engine.tok.encode(str(r['gold']),add_special_tokens=False))))
    for row in {r['id']:r for r in selected}.values():
        settings(engine,row);base=engine.base(row);native=O.native_cache(engine,row)
        for method,module in modules.items():
            original=engine.cache(row,base,module)
            for context,(tokens,positions) in contexts(engine,row,original,method).items():
                nt=D.forward_trace(C,engine,native,tokens,positions,LAYERS)
                tt=D.forward_trace(C,engine,original,tokens,positions,LAYERS)
                # Hooks/cloning preserve ordinary logits; replaying one's own Q
                # is an exact no-op, not a new decoding protocol.
                regular=engine.logits(row,C.clone_cache(original,engine.model),tokens[:max(positions)],positions)
                assert torch.equal(regular,tt.logits),('Trace changes model logits',row['id'],context)
                selfclamp=D.forward_trace(C,engine,original,tokens,positions,LAYERS,clamp=tt,clamp_layers=LAYERS)
                assert torch.equal(selfclamp.logits,tt.logits),'Self-query clamp changes logits'
                del selfclamp,regular
                for group in GROUPS:
                    result=group_run(engine,row,original,native,nt,tt,tokens,positions,group,check=True)
                    # Native/native reference has zero local fixed-Q damage.
                    assert result['fixed_native_q']['cases']['NN']['wo_nmse']<1e-10
                    del result
                del nt,tt
            engine.unchanged();del original
        del base,native
        checks.append(dict(id=row['id'],task=row['task'],all_methods_contexts_and_groups_checked=True,
            trace_logits_bitwise_parity=True,self_query_clamp_bitwise_parity=True,
            fixed_query_KV_factorial_identities=True,attention_reconstruction_relative_l2_max_bound=.02,
            query_clamp_exact_pre_rope=True,header_and_untouched_components_exact=True))
        mark('smoke/progress.json',status='running',checks=checks)
        C.log('MECHANISM SMOKE',row['id'],'PASSED')
    engine.unchanged()
    mark('smoke/completed.json',status='completed',checks=checks,training=False,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)


@torch.no_grad()
def evaluate(task):
    verify();assert C.read(ROOT/'smoke/completed.json')['manifest_sha256']==identity()
    engine,modules,_=O.models();started=time.monotonic()
    for i,row in enumerate(rows(task),1):
        pause_gate();settings(engine,row);base=engine.base(row);native=O.native_cache(engine,row)
        for method,module in modules.items():
            original=engine.cache(row,base,module)
            for context,(tokens,positions) in contexts(engine,row,original,method).items():
                missing=[g for g in GROUPS if not (ROOT/f'evaluation/{task}/per_group/{method}/{context}/{g}/{row["id"]}.json').exists()]
                if not missing:continue
                nt=D.forward_trace(C,engine,native,tokens,positions,LAYERS)
                tt=D.forward_trace(C,engine,original,tokens,positions,LAYERS)
                for group in missing:
                    result=group_run(engine,row,original,native,nt,tt,tokens,positions,group)
                    s=len(engine.prefix_ids);end=engine._diagnostic_body_end
                    _,baseline=previous(row,method)
                    oracle={case:(baseline if case=='TT' else previous(row,method,group,case)[1]) for case in CASES}
                    gold_scores=None
                    if task=='arc':
                        gold_scores={c:dict(gold_score=v['scores'][row['gold']],
                            gold_margin=v['scores'][row['gold']]-max(s for j,s in enumerate(v['scores']) if j!=row['gold'])) for c,v in oracle.items()}
                    mark(f'evaluation/{task}/per_group/{method}/{context}/{group}/{row["id"]}.json',id=row['id'],task=task,
                        method=method,context=context,group=group,physical_layers=[l+1 for l in GROUPS[group]],
                        tokens=tokens,probe_positions=positions,probe_absolute_positions=nt.qpos.cpu().tolist(),
                        gold_token_nll_is_full_answer=context=='gold_answer',
                        distances=distances(original,native,GROUPS[group],s,end),
                        native_functional=D.functional(C,nt,nt,tokens,positions),oracle_free=oracle,
                        historical_choice_scores=gold_scores,diagnostics=result)
                    status(status='running',phase='evaluation',task=task,sample=i,total=64,method=method,context=context,group=group,
                        elapsed_seconds=time.monotonic()-started)
                    del result
                del nt,tt
            del original;engine.unchanged()
        del native,base
        if i%2==0:C.log('MECHANISM EVALUATE',task,f'{i}/64')
    verify();mark(f'evaluation/{task}/completed.json',status='completed',n=64,finished_at=time.strftime('%F %T'))


def summarize():
    verify()
    from statistics_report import report
    report(C,ROOT,CFG,identity)
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'));status(status='completed')


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='smoke':smoke()
    elif action=='evaluate':evaluate(sys.argv[2])
    elif action=='summarize':summarize()
    else:raise ValueError(action)
