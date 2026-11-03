"""Four frozen checkpoints on one new, fixed GSM8K cohort; no training."""
import json
import random
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text());PARENT=Path(CFG['parent'])
sys.path.insert(0,str(PARENT))
import common as R
import model as M
import cache as K
STRUCT=PARENT/'structural_ablations_seed1234'
sys.path.insert(0,str(STRUCT))
import ablation_common as A
import ablation_runtime as AR
from numeric import extract,normalize,checks
C=R.C;torch=R.torch

def identity():return C.sha(ROOT/'manifest.json')
def mark(name,**v):C.save(ROOT/name,dict(manifest_sha256=identity(),**v))
def status(**v):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def rows():return C.read_jsonl(ROOT/'prepared/test.jsonl')
def pause():
    if (ROOT/'controls/pause_requested.json').exists():status(status='paused');raise SystemExit(75)

def prepare():
    if (ROOT/'manifest.json').exists():verify();return
    R.verify();A.verify();assert R.identity()==CFG['parent_manifest_sha256'];assert A.identity()==CFG['structural_manifest_sha256']
    cp=PARENT/'runs/epoch4_step1024.pt';kp=STRUCT/'arms/keep_both/runs/epoch4_step1024.pt'
    assert C.sha(cp)==CFG['rnkv_checkpoint_sha256'];assert C.sha(kp)==CFG['keep_both_checkpoint_sha256']
    from datasets import Dataset
    raw=Dataset.from_file(CFG['source']);assert len(raw)==CFG['expected_source_count']
    refs=[Path(CFG['source']),cp,kp,PARENT/'manifest.json',STRUCT/'manifest.json']
    excluded=[]
    for root in [C.ROOT,PARENT]:
        for split in ['train','validation','test']:
            path=root/f'prepared/{split}.jsonl';refs.append(path);excluded+=C.read_jsonl(path)
    path=STRUCT/'prepared/test.jsonl';refs.append(path);excluded+=C.read_jsonl(path)
    blocked={C.norm(r['question']) for r in excluded};old_gsm={C.norm(r['question']) for r in C.rows('test') if r['task']=='gsm8k'}
    tok=C.AutoTokenizer.from_pretrained(C.CFG['models']['receiver'],local_files_only=True)
    affix=C.read(C.ROOT/'manifest.json');order=list(range(len(raw)));random.Random(CFG['seed']).shuffle(order)
    selected=[];rejected=0
    for index in order:
        doc=raw[index];question=doc['question'];key=C.norm(question)
        if key in blocked:rejected+=1;continue
        prompt,_,gold,_,_=C.SC.format_item('gsm8k',doc);assert normalize(gold) is not None
        row=dict(id=f'gsm8k_test_{index}',task='gsm8k',source_id=str(index),source_index=index,question=question,prompt=prompt,
            gold=gold,gold_normalized=normalize(gold),continuation_suffix=affix['assistant'],source_doc_sha256=C.digest(doc))
        row['content_ids']=tok.encode(prompt,add_special_tokens=False);row['suffix_ids']=tok.encode(affix['assistant'],add_special_tokens=False)
        assert tok.encode(affix['prefix']+prompt+affix['assistant'],add_special_tokens=False)==affix['prefix_ids']+row['content_ids']+row['suffix_ids']
        assert len(affix['prefix_ids'])+len(row['content_ids'])+32+len(row['suffix_ids'])+CFG['extension_max_new_tokens']<CFG['max_context_tokens']
        blocked.add(key);selected.append(row)
        if len(selected)==CFG['test_count']:break
    assert len(selected)==64 and not old_gsm & {C.norm(r['question']) for r in selected}
    C.write_jsonl(ROOT/'prepared/test.jsonl',selected);refs+=[ROOT/'prepared/test.jsonl']
    refs+=list(PARENT.glob('*.py'))+list(STRUCT.glob('*.py'))
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={f.name:C.sha(f) for f in ROOT.glob('*.py')},
        sources={str(f):C.sha(f) for f in refs},selected_indices=[r['source_index'] for r in selected],excluded_candidates=rejected,
        historical_parent_gsm_questions=len(old_gsm),normalized_question_overlap=0,training=False,numeric_checks=checks()))
    mark('prepared/completed.json',status='completed',n=64,maximum_body_tokens=max(len(r['content_ids']) for r in selected))
    C.log('GSM8K PREPARED',64,'max_body',max(len(r['content_ids']) for r in selected))

def verify():
    R.verify();A.verify();value=C.read(ROOT/'manifest.json');assert C.sha(ROOT/'config.json')==value['config_sha256']
    for name,digest in value['code'].items():assert C.sha(ROOT/name)==digest,name
    for name,digest in value['sources'].items():assert C.sha(name)==digest,name

def unchanged(engine,correction,modules,snapshots):
    engine.unchanged();assert engine.student_test_native_prefills==0
    for obj,before in zip([correction,*modules],snapshots):
        assert C.versions(obj)==before
        assert all(not p.requires_grad and p.grad is None for p in obj.parameters())

@torch.no_grad()
def score(module,row,cache,rn,cap):
    started=time.time();tokens=module.rollout(row,cache,rn=rn,cap=cap)
    eos=module.engine.model.generation_config.eos_token_id;stops=set(eos if isinstance(eos,list) else [eos])
    stops.add(module.engine.tok.convert_tokens_to_ids('<|im_end|>'))
    stopped=bool(tokens and tokens[-1] in stops)
    text=module.engine.tok.decode(tokens,skip_special_tokens=True);answer=extract(text)
    legacy=C.SC.extract_numeric_answer(text)
    return dict(**answer,correct=int(answer['prediction']==row['gold_normalized']),legacy_prediction=legacy,
        legacy_correct=int(normalize(legacy)==row['gold_normalized']),generation=text,token_ids=tokens,generated_tokens=len(tokens),
        stopped_on_eos=stopped,reached_cap=int(len(tokens)==cap and not stopped),max_new_tokens=cap,seconds=time.time()-started)

@torch.no_grad()
def smoke(engine,correction,full,keep,snapshots):
    row=rows()[0];base=engine.base(row);translated=engine.cache(row,base,correction)
    own=full.logits(row,translated,rn=False)
    existing=engine.logits(row,C.clone_cache(translated,engine.model))
    assert torch.equal(own,existing),'C continuation changed'
    assert torch.equal(own,keep.logits(row,translated,rn=False)),'Inactive second routing changes baseline'
    native=engine.native_cache(row,baseline=True)
    assert torch.equal(full.logits(row,native,rn=False),engine.logits(row,C.clone_cache(native,engine.model)))
    L=translated.get_seq_length();h=len(engine.prefix_ids);q=torch.tensor([L+32],device='cuda')
    for layer in range(1,37):
        a=full.routing.allowed('answer',layer,h,L,32,q,L+33)[0];b=keep.routing.allowed('answer',layer,h,L,32,q,L+33)[0]
        assert a[:h].all() and b[:h].all() and a[-1] and b[-1]
        if 19<=layer<=30:assert not a[h:L].any() and a[L:L+32].all() and b[h:L+32].all()
        else:assert a[h:L].all() and b[h:L].all() and not a[L:L+32].any() and not b[L:L+32].any()
    for module in [full,keep]:
        mem=module.construct(row,translated);assert mem.get_seq_length()==L+32
        for l in range(36):
            for actual,original in zip(C._cache_layer(mem,l),C._cache_layer(translated,l)):assert torch.equal(actual[:,:,:L],original)
        assert torch.isfinite(module.logits(row,mem,rn=True)).all()
        del mem
    unchanged(engine,correction,[full,keep],snapshots)
    mark('smoke/completed.json',status='completed',baseline_logits_exact=True,both_routing_masks_checked_all36_layers=True,
        original_cache_prefix_unchanged=True,memory_positions=[L,L+31],answer_suffix_start=L+32,all_parameters_frozen=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    C.log('GSM8K SMOKE PASSED')

def summarize(cap):
    records=[C.read(ROOT/f'evaluation/cap{cap}/per_sample/{r["id"]}.json') for r in rows()]
    assert all(v['manifest_sha256']==identity() and set(v['conditions'])==set(CFG['arms']) for v in records)
    metrics={a:{k:sum(v['conditions'][a][k] for v in records)/64 for k in ['correct','legacy_correct','valid_final_answer','generated_tokens','reached_cap','stopped_on_eos']} for a in CFG['arms']}
    delta={a:{b:C.paired([v['conditions'][a]['correct']-v['conditions'][b]['correct'] for v in records]) for b in ['C','native'] if a!=b} for a in ['rnkv','keep_both']}
    delta['keep_both']['rnkv']=C.paired([v['conditions']['keep_both']['correct']-v['conditions']['rnkv']['correct'] for v in records])
    changes={a:{b:{name:[v['id'] for v in records if (v['conditions'][a]['correct']-v['conditions'][b]['correct'])*sign>0] for name,sign in [('rescued_ids',1),('harmed_ids',-1)]} for b in ['C','rnkv','native'] if a!=b} for a in ['rnkv','keep_both']}
    counts={a:sum(v['conditions'][a]['reached_cap'] for v in records) for a in CFG['arms']}
    mark(f'evaluation/cap{cap}/summary.json',status='completed',n=64,max_new_tokens=cap,metrics=metrics,paired_differences=delta,
        rescued_harmed=changes,cap_counts=counts,length_limited=any(v>=CFG['extension_trigger_count'] for v in counts.values()))
    return C.read(ROOT/f'evaluation/cap{cap}/summary.json')

@torch.no_grad()
def evaluate(engine,correction,full,keep,snapshots,cap):
    start=time.time()
    for index,row in enumerate(rows(),1):
        pause();path=ROOT/f'evaluation/cap{cap}/per_sample/{row["id"]}.json'
        if path.exists():
            value=C.read(path);assert value['manifest_sha256']==identity()
            if set(value['conditions'])==set(CFG['arms']):continue
        else:value=dict(manifest_sha256=identity(),id=row['id'],source_id=row['source_id'],question=row['question'],gold=row['gold'],conditions={})
        todo=set(CFG['arms'])-set(value['conditions']);base=translated=None
        if todo & {'C','rnkv','keep_both'}:base=engine.base(row);translated=engine.cache(row,base,correction)
        for arm in CFG['arms']:
            if arm not in todo:continue
            pause();status(status='running',phase='evaluation',max_new_tokens=cap,sample=index,total=64,arm=arm,elapsed_seconds=time.time()-start)
            if arm=='C':result=score(full,row,translated,False,cap)
            elif arm in ['rnkv','keep_both']:
                module=full if arm=='rnkv' else keep;mem=module.construct(row,translated)
                result=score(module,row,mem,True,cap);del mem
            else:
                native=engine.native_cache(row,baseline=True);result=score(full,row,native,False,cap);del native
            value['conditions'][arm]=result;C.save(path,value)
            unchanged(engine,correction,[full,keep],snapshots)
            C.log('GSM8K',f'cap{cap}',f'{index}/64',arm,'correct',result['correct'],'tokens',result['generated_tokens'],'cap',result['reached_cap'])
        del base,translated
    return summarize(cap)

def run():
    verify();checks();status(status='running',phase='loading_frozen_models')
    engine,correction,full=M.load('final');engine.mode='test';full.eval().requires_grad_(False)
    keep=AR.AblationModel(engine,'keep_both');cp=torch.load(STRUCT/'arms/keep_both/runs/epoch4_step1024.pt',map_location='cpu',weights_only=True)
    assert cp['manifest_sha256']==CFG['structural_manifest_sha256'] and cp['step']==1024 and cp['arm']=='keep_both'
    keep.load_state_dict(cp['state'],strict=True);keep.eval().requires_grad_(False);del cp
    snapshots=[C.versions(v) for v in [correction,full,keep]];smoke(engine,correction,full,keep,snapshots)
    results={};cap=CFG['initial_max_new_tokens'];results[str(cap)]=evaluate(engine,correction,full,keep,snapshots,cap)
    extend=results[str(cap)]['length_limited'];mark('length_decision.json',status='completed',trigger_count=CFG['extension_trigger_count'],cap_counts=results[str(cap)]['cap_counts'],extend_all_four_arms=extend)
    if extend:
        cap=CFG['extension_max_new_tokens'];results[str(cap)]=evaluate(engine,correction,full,keep,snapshots,cap)
    unchanged(engine,correction,[full,keep],snapshots);verify()
    mark('summary.json',status='completed',budgets=results,training=False,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    lines=['# Frozen GSM8K: C / RN-KV / Keep-both / Native','','64 identical questions; explicit final numeric answer is primary.','',
        '| Budget | Method | Accuracy | Legacy fallback accuracy | Valid final answer | Unfinished at cap | Mean tokens |','|---|---|---:|---:|---:|---:|---:|']
    for budget,result in results.items():
        for arm,m in result['metrics'].items():lines.append(f'| {budget} | {arm} | {m["correct"]*100:.2f}% | {m["legacy_correct"]*100:.2f}% | {m["valid_final_answer"]*100:.2f}% | {result["cap_counts"][arm]} | {m["generated_tokens"]:.1f} |')
    lines+=['',*['- '+s for s in CFG['caveats']]];(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    mark('completed.json',status='completed',finished_at=time.strftime('%F %T'),budgets=list(results));status(status='completed');C.log('GSM8K ALL COMPLETED')

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='run':run()
    else:raise ValueError(sys.argv[1])
