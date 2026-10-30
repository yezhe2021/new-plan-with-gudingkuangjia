"""Cumulative native KV prefixes, immutable A/C/D and reused64-question anchors."""
import importlib.util
import json
import sys
import time
from pathlib import Path
from analysis import classify,overall,check_rules
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
PARENT=Path(CFG['parent']);ORACLE=PARENT/CFG['oracle_experiment']
sys.path.insert(0,str(PARENT))
import core as C
spec=importlib.util.spec_from_file_location('previous_oracle',ORACLE/'experiment.py')
O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)
torch=C.torch;METHODS=CFG['methods']


def identity():return C.sha(ROOT/'manifest.json')
def mark(path,**v):C.save(ROOT/path,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows(task=None):return [r for r in C.read_jsonl(ROOT/'prepared/test.jsonl') if task is None or r['task']==task]
def conditions():
    return [dict(name='unmasked',mode='none',layers=[])]+[
        dict(name=f'prefix{k:02d}',mode='kv',layers=list(range(k))) for k in CFG['prefix_cuts']]


def source(row,method,condition='unmasked'):
    if method=='native':p=ORACLE/'evaluation'/row['task']/'per_config/native'/(row['id']+'.json')
    else:p=ORACLE/'evaluation'/row['task']/'per_config'/method/condition/(row['id']+'.json')
    item=C.read(p)
    assert item['manifest_sha256']==CFG['oracle_manifest_sha256']
    assert item['id']==row['id'] and item['task']==row['task']
    if method!='native':
        assert item['method']==method and item['condition']['name']==condition
        layers=[] if condition=='unmasked' else list(range(18,24))
        mode='none' if condition=='unmasked' else 'v' if condition=='g4_v' else 'kv'
        assert item['condition']['layers']==layers and item['condition']['mode']==mode
    else:
        # Historical native schema has id/task/result, no method/condition fields.
        assert 'method' not in item or item['method']=='native'
    return p,item['result']


def prepare():
    if (ROOT/'manifest.json').exists():verify();return
    O.verify();assert C.identity()==CFG['parent_manifest_sha256'] and O.identity()==CFG['oracle_manifest_sha256']
    assert C.read(ORACLE/'completed.json')['status']=='completed'
    rr=O.rows();assert len(rr)==192
    assert all(sum(r['task']==t for r in rr)==n==64 for t,n in CFG['test_counts'].items())
    assert [r['id'] for r in rr]==[r['id'] for r in C.rows('test') if r['task'] in CFG['test_counts'] and
        r['id'] in {v['id'] for v in rr}]
    assert CFG['prefix_cuts']==[6,12,18,24,30,36] and CFG['mode']=='kv' and METHODS==['A','C','D']
    assert CFG['seed']==C.CFG['seed'] and not CFG['gsm8k'] and not CFG['training']
    refs=[ORACLE/f for f in ['manifest.json','config.json','experiment.py','completed.json','prepared/test.jsonl','smoke/completed.json']]
    refs += [PARENT/f for f in ['manifest.json','core.py','train.py','evaluate.py','prepared/test.jsonl']]
    refs += [PARENT/'runs'/a/'epoch4_step800.pt' for a in ['C','D']]
    refs.append(ROOT/'design.md');reuse=[]
    for row in rr:
        for method in ['native']+METHODS:
            p,value=source(row,method);refs.append(p)
            reuse.append((row,method,str(p),value))
        for method in METHODS:
            for con in ['g4_v','g4_kv']:
                p,_=source(row,method,con);refs.append(p)
    C.write_jsonl(ROOT/'prepared/test.jsonl',rr);refs.append(ROOT/'prepared/test.jsonl')
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),
        code={p.name:C.sha(p) for p in ROOT.glob('*.py')},sources={str(p):C.sha(p) for p in refs},
        test_ids={t:[r['id'] for r in rr if r['task']==t] for t in CFG['test_counts']},
        conditions=conditions(),prefix00_alias='unmasked',baseline_recomputed=False,
        n=192,new_configurations=3456,reused_configurations=768,total_configurations=4224,
        decision_rule_checks=check_rules(),preregistered_at=time.strftime('%F %T'),
        decision_unit='Per method x task; accuracy for ARC/HellaSwag, EM for HotpotQA. F1 secondary.',
        overall_rule='Consensus across curves whose native score exceeds translated; otherwise ambiguous. Nonpositive-gap curves retain descriptive labels but do not imply recovery.'))
    for row,method,p,value in reuse:
        con=dict(name='native',mode='native',layers=list(range(36))) if method=='native' else conditions()[0]
        path=f'evaluation/{row["task"]}/per_config/native/{row["id"]}.json' if method=='native' else f'evaluation/{row["task"]}/per_config/{method}/unmasked/{row["id"]}.json'
        mark(path,id=row['id'],task=row['task'],method=method,condition=con,result=value,
             reused_from=p,source_manifest_sha256=O.identity())
    mark('prepared/completed.json',status='completed',n=192,new_configurations=3456,reused_configurations=768)
    C.log('PREFIX PREPARED',CFG['test_counts'],'rules frozen before inference')


def verify():
    C.verify();m=C.read(ROOT/'manifest.json')
    assert C.sha(ROOT/'config.json')==m['config_sha256']
    for name,d in m['code'].items():assert C.sha(ROOT/name)==d,name
    for path,d in m['sources'].items():assert C.sha(path)==d,path


def anchor(row,method):
    path=f'evaluation/{row["task"]}/per_config/native/{row["id"]}.json' if method=='native' else f'evaluation/{row["task"]}/per_config/{method}/unmasked/{row["id"]}.json'
    item=C.read(ROOT/path);assert item['manifest_sha256']==identity()
    assert item['id']==row['id'] and item['task']==row['task'] and item['method']==method
    assert item['condition']['layers']==(list(range(36)) if method=='native' else [])
    _,original=source(row,method);assert item['result']==original
    return item['result']


def endpoint(row,value,native):
    assert value==native,('prefix36 differs from reused native result',row['id'],value,native)


@torch.no_grad()
def smoke():
    verify();engine,modules,mask=O.models();checks=[]
    selected=[rows(t)[0] for t in CFG['test_counts']]
    selected.append(max(rows('hotpotqa'),key=lambda r:len(r['content_ids'])))
    inherited=C.read(ORACLE/'smoke/completed.json')
    assert inherited['manifest_sha256']==O.identity()
    assert all(c['native_cached_direct_prefill_parity'] for c in inherited['checks'])
    for row in {r['id']:r for r in selected}.values():
        base=engine.base(row);native=O.native_cache(engine,row);reference=anchor(row,'native')
        start=len(engine.prefix_ids);end=native.get_seq_length()
        for method,module in modules.items():
            original=engine.cache(row,base,module);O.M.parity(row,method,anchor(row,method))
            zero=O.replace(engine,row,original,native,[],'kv')
            assert all(torch.equal(a,b) for l in range(36) for a,b in zip(C._cache_layer(original,l),C._cache_layer(zero,l)))
            a=O.M.first_logits(engine,mask,row,original,[]);b=O.M.first_logits(engine,mask,row,zero,[])
            assert torch.equal(a,b),('prefix00 differs from unmasked',row['id'],method)
            del zero,a,b
            same=O.replace(engine,row,original,original,list(range(36)),'kv')
            a=O.M.first_logits(engine,mask,row,original,[]);b=O.M.first_logits(engine,mask,row,same,[])
            assert torch.equal(a,b);del same,a,b
            for con in conditions()[1:]:
                mixed=O.replace(engine,row,original,native,con['layers'],'kv')
                for layer in range(36):
                    for index,tensor in enumerate(C._cache_layer(mixed,layer)):
                        orig=C._cache_layer(original,layer)[index]
                        assert torch.equal(tensor[:,:,:start,:],orig[:,:,:start,:])
                        expected=C._cache_layer(native,layer)[index] if layer in con['layers'] else orig
                        assert torch.equal(tensor[:,:,start:end,:],expected[:,:,start:end,:])
                logits=O.M.first_logits(engine,mask,row,mixed,[]);assert torch.isfinite(logits).all();del logits
                if con['name']=='prefix36':
                    value=O.M.score(engine,mask,row,mixed,[]);endpoint(row,value,reference)
                del mixed
            del original
        engine.unchanged();del base,native
        checks.append(dict(id=row['id'],task=row['task'],prefix00_cache_and_logits_exact=True,
            prefix36_scores_or_full_generation_exact_to_reused_native=True,
            all_prefix_tensor_components_and_header_checked=True,self_substitution_logits_exact=True,
            native_and_unmasked_baselines_recomputed=False))
        C.log('PREFIX SMOKE',row['id'],'PASSED')
    assert engine.student_test_native_prefills==0
    mark('smoke/completed.json',status='completed',checks=checks,
        inherited_native_cached_direct_prefill_parity=str(ORACLE/'smoke/completed.json'),
        inherited_manifest_sha256=O.identity(),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)


@torch.no_grad()
def evaluate(task):
    verify();assert C.read(ROOT/'smoke/completed.json')['manifest_sha256']==identity()
    engine,modules,mask=O.models();records=[];done=0;started=time.monotonic()
    for i,row in enumerate(rows(task),1):
        native_value=anchor(row,'native');done+=1
        base=engine.base(row);native=O.native_cache(engine,row);values={}
        for method,module in modules.items():
            original=engine.cache(row,base,module);values[method]={}
            for con in conditions():
                name=con['name'];path=f'evaluation/{task}/per_config/{method}/{name}/{row["id"]}.json'
                if name=='unmasked':value=anchor(row,method);O.M.parity(row,method,value)
                elif (ROOT/path).exists():
                    item=C.read(ROOT/path);assert item['manifest_sha256']==identity()
                    assert item['id']==row['id'] and item['method']==method and item['condition']==con
                    value=item['result']
                else:
                    mixed=O.replace(engine,row,original,native,con['layers'],'kv')
                    value=O.M.score(engine,mask,row,mixed,[]);del mixed
                    if name=='prefix36':endpoint(row,value,native_value)
                    mark(path,id=row['id'],task=task,method=method,condition=con,result=value)
                if name=='prefix36':endpoint(row,value,native_value)
                values[method][name]=value;done+=1
                status(status='running',phase='evaluation',task=task,sample=i,total=64,method=method,
                    condition=name,task_configurations_completed=done,task_configurations=1408,
                    elapsed_seconds=time.monotonic()-started)
            del original
        record=dict(id=row['id'],task=task,gold=row['gold'],native=native_value,conditions=values)
        mark(f'evaluation/{task}/per_sample/{row["id"]}.json',**record);records.append(record)
        engine.unchanged();del base,native
        if i%4==0:C.log('PREFIX EVALUATE',task,f'{i}/64')
    assert engine.student_test_native_prefills==0;verify()
    C.write_jsonl(ROOT/f'evaluation/{task}/per_sample.jsonl',records);make_summary(task,records)


def make_summary(task,records):
    keys=['em','f1'] if task=='hotpotqa' else ['correct'];primary=keys[0]
    lookup={r['id']:r for r in rows(task)}
    metrics={};comparisons={};marginals={};decisions={};mismatch={};recovery={}
    native={k:sum(r['native'][k] for r in records)/64 for k in keys}
    for method in METHODS:
        metrics[method]={};comparisons[method]={};marginals[method]={};decisions[method]={};recovery[method]={}
        for con in conditions():
            name=con['name'];metrics[method][name]={k:sum(r['conditions'][method][name][k] for r in records)/64 for k in keys}
            metrics[method][name]['cap_count']=sum(r['conditions'][method][name].get('reached_cap',0) for r in records)
            comparisons[method][name]=dict(delta_vs_unmasked={k:C.paired([r['conditions'][method][name][k]-r['conditions'][method]['unmasked'][k] for r in records]) for k in keys},
                rescued_ids=[r['id'] for r in records if r['conditions'][method][name][primary] and not r['conditions'][method]['unmasked'][primary]],
                harmed_ids=[r['id'] for r in records if not r['conditions'][method][name][primary] and r['conditions'][method]['unmasked'][primary]])
            recovery[method][name]={k:((metrics[method][name][k]-metrics[method]['unmasked'][k])/(native[k]-metrics[method]['unmasked'][k])
                if native[k]>metrics[method]['unmasked'][k] else None) for k in keys}
        for key in keys:
            decisions[method][key]=classify([metrics[method][con['name']][key] for con in conditions()],[0]+CFG['prefix_cuts'])
            marginals[method][key]=[];prev='unmasked';prev_k=0
            for cut in CFG['prefix_cuts']:
                name=f'prefix{cut:02d}';paired=C.paired([r['conditions'][method][name][key]-r['conditions'][method][prev][key] for r in records])
                marginals[method][key].append(dict(k_from=prev_k,k_to=cut,delta=paired['mean'],paired=paired))
                prev,prev_k=name,cut
        old={con:[source(lookup[r['id']],method,con)[1] for r in records] for con in ['g4_v','g4_kv']}
        mismatch[method]=dict(
            prefix24_minus_g4_v={k:C.paired([r['conditions'][method]['prefix24'][k]-v[k] for r,v in zip(records,old['g4_v'])]) for k in keys},
            prefix24_minus_g4_kv={k:C.paired([r['conditions'][method]['prefix24'][k]-v[k] for r,v in zip(records,old['g4_kv'])]) for k in keys},
            lower_prefix_by_g4_kv_interaction={k:C.paired([
                (r['conditions'][method]['prefix24'][k]-r['conditions'][method]['prefix18'][k])-
                (v[k]-r['conditions'][method]['unmasked'][k]) for r,v in zip(records,old['g4_kv'])]) for k in keys},
            interpretation='prefix24 vs g4_v changes both K and lower layers, so cannot alone establish mismatch. Interaction holds intervention type KV fixed but does not prove all earlier substitutions are lower bounds.')
    mark(f'evaluation/{task}/summary.json',status='completed',task=task,n=64,primary_metric=primary,
        native=native,metrics=metrics,comparisons=comparisons,marginals=marginals,decisions=decisions,
        recovery_ratio=recovery,mismatch_comparisons=mismatch,all_prefix36_exact=True,baselines_recomputed=False,
        native_usage='Offline oracle body cache only; shared header, no model training.')
    mark(f'evaluation/{task}/completed.json',status='completed',finished_at=time.strftime('%F %T'))


def summarize():
    verify();tasks={t:C.read(ROOT/f'evaluation/{t}/summary.json') for t in CFG['test_counts']}
    assert all(d['status']=='completed' and d['manifest_sha256']==identity() for d in tasks.values())
    decision=overall([d['decisions'][a][d['primary_metric']] for d in tasks.values() for a in METHODS])
    mark('summary.json',status='completed',tasks=tasks,overall_decision=decision,config=CFG,
        suggested_commit_message=f'Prefix native KV sweep: {decision} (A/C/D,64 per task)',
        interpretation='Descriptive cumulative hybrid-cache diagnostic; no proof of intrinsic error localization, repair ceiling, or deployable90% recovery.')
    lines=['# Prefix Native Sweep: A/C/D,64 per task','',f'Overall preregistered descriptive label: **{decision}**.','',
        '| Method | Native prefix layers | ARC64 | HellaSwag64 | HotpotQA64 EM | HotpotQA64 F1 | Recovery ARC |',
        '|---|---:|---:|---:|---:|---:|---:|']
    native_values=[tasks[t]['native']['correct'] for t in ['arc','hellaswag']]+[tasks['hotpotqa']['native'][k] for k in ['em','f1']]
    lines.append('| Qwen native |36|'+ '|'.join(f'{v*100:.2f}%' for v in native_values)+'| — |')
    for method in METHODS:
        for con in conditions():
            name=con['name'];values=[tasks[t]['metrics'][method][name]['correct'] for t in ['arc','hellaswag']]+[tasks['hotpotqa']['metrics'][method][name][k] for k in ['em','f1']]
            ratio=tasks['arc']['recovery_ratio'][method][name]['correct'];rr='—' if ratio is None else f'{ratio*100:.2f}%'
            lines.append('|'+method+'|'+str(len(con['layers']))+'|'+'|'.join(f'{v*100:.2f}%' for v in values)+'|'+rr+'|')
    lines+=['','Conditional marginals, percentage points:','',
        '| Method | Newly native layers | ARC Δ | HellaSwag Δ | Hotpot EM Δ | Hotpot F1 Δ |','|---|---|---:|---:|---:|---:|']
    for a in METHODS:
        for i,k in enumerate(CFG['prefix_cuts']):
            values=[tasks[t]['marginals'][a]['correct'][i]['delta'] for t in ['arc','hellaswag']]+[tasks['hotpotqa']['marginals'][a][m][i]['delta'] for m in ['em','f1']]
            lines.append('|'+a+f'|{k-5:02d}–{k:02d}|'+'|'.join(f'{v*100:+.2f}' for v in values)+'|')
    lines+=['','Preregistered primary labels (accuracy / Hotpot EM):','',
        '| Method | Task | Label | Linear R² | Monotone | Native better |','|---|---|---|---:|---|---|']
    for a in METHODS:
        for task,d in tasks.items():
            v=d['decisions'][a][d['primary_metric']];r2='undefined' if v['linear_r2'] is None else f'{v["linear_r2"]:.4f}'
            lines.append(f'|{a}|{task}|{v["label"]}|{r2}|{v["monotone"]}|{v["recovery_interpretation_applicable"]}|')
    lines+=['','Paired ARC mismatch diagnostics (percentage points):','',
        '| Method | prefix24 − isolated G4 V | prefix24 − isolated G4 KV | Lower-prefix × G4 KV interaction |',
        '|---|---:|---:|---:|']
    for a in METHODS:
        v=tasks['arc']['mismatch_comparisons'][a]
        values=[v[name]['correct']['mean'] for name in ['prefix24_minus_g4_v','prefix24_minus_g4_kv','lower_prefix_by_g4_kv_interaction']]
        lines.append('|'+a+'|'+'|'.join(f'{x*100:+.2f}' for x in values)+'|')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',overall_decision=decision,finished_at=time.strftime('%F %T'));status(status='completed',overall_decision=decision)


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='smoke':smoke()
    elif action=='evaluate':evaluate(sys.argv[2])
    elif action=='summarize':summarize()
    else:raise ValueError(action)
