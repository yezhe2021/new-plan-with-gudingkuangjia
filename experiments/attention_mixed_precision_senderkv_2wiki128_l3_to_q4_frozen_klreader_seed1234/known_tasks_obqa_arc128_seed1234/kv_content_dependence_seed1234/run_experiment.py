"""Causal dependence on transmitted KV content, frozen old KL-only Reader."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_OFFLINE']='1'
import copy,hashlib,json,sys,time,shutil
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import run_experiment as K
E=K.E
sys.path.insert(0,str(ROOT))
from controls import seed_for,permute_options,donor_maps,resize_pair,matched_noise,visibility
CFG=json.loads((ROOT/'config.json').read_text())
torch.set_num_threads(1)


def read(p):return json.loads(Path(p).read_text())
def save(p,x):E.save(p,x)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity():return sha(ROOT/'manifest.json')
def status(**kw):save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**kw))
def rows():return [json.loads(x) for x in (ROOT/'prepared/questions.jsonl').read_text().splitlines()]
def fname(r):return E.U.fname(r)
def cache_path(row,variant):return ROOT/'temporary_kv'/variant/(fname(row)+'.pt')
def source_path(row):return ROOT/'temporary_source'/(fname(row)+'.pt')
def result_path(row):return ROOT/'evaluation'/row['task']/(fname(row)+'.json')


def layout(row,tokenizer):
    labels=[E.C.R.candidate_ids(tokenizer,label) for label in row['candidates']]
    assert all(len(x)==1 for x in labels)
    label_ids=[x[0] for x in labels];assert len(set(label_ids))==len(label_ids)
    return dict(body_ids=row['frames']['qwen']['body_ids'],suffix_ids=row['suffix_ids'],label_ids=label_ids)


def prepare():
    K.verify();original=K.rows();tokenizers={f:E.tok(f) for f in ['llama','qwen']}
    assert dict(Counter(r['task'] for r in original))==CFG['tasks']
    donors=donor_maps(original,CFG['repeats'],CFG['seed']);records=[];anchors=set();suffixes=set()
    for row in original:
        changed,perm,texts=permute_options(row['prompt'],row['candidates'],seed_for(CFG['seed'],row['id'],'options'))
        reordered=E.U.make_row(row['id']+'__reordered','gsm8k',dict(question=row['question_key']),tokenizers['llama'],tokenizers['qwen'],
            prompt=changed,candidates=row['candidates'],gold=perm.index(row['gold']),normalize=row['normalize'],cue=row['cue'])
        reordered.update(task=row['task'],question_key=row['question_key'])
        assert reordered['gold']!=row['gold'] and texts[perm[reordered['gold']]]==texts[row['gold']]
        row=copy.deepcopy(row)
        row['content_layout']=layout(row,tokenizers['qwen']);reordered['content_layout']=layout(reordered,tokenizers['qwen'])
        for item in [row,reordered]:
            frame=item['frames']['qwen'];anchors.add(tuple(frame['body_ids'][:frame['native_prefix_length']]))
            suffixes.add(tuple(item['suffix_ids']))
            assert frame['native_prefix_length']+len(item['source_indices'])==len(frame['body_ids'])
        records.append(dict(id=row['id'],task=row['task'],original=row,reordered=reordered,permutation=perm,option_texts=texts,donors=donors[row['id']]))
    assert len(anchors)==len(suffixes)==1,'All direct textual inputs must be common across questions'
    anchor=list(next(iter(anchors)));suffix=list(next(iter(suffixes)))
    out=ROOT/'prepared/questions.jsonl';out.parent.mkdir(exist_ok=True)
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records)
    if out.exists():assert out.read_text()==data
    else:out.write_text(data)
    manifest=dict(config=CFG,config_sha256=sha(ROOT/'config.json'),prepared_sha256=sha(out),code={p.name:sha(p) for p in ROOT.glob('*.py')},
        parent_manifest_sha256=K.identity(),parent_code_sha256=sha(Path(K.__file__)),checkpoints=read(E.ROOT/'manifest.json')['checkpoints'],
        direct_inputs=dict(anchor_ids=anchor,suffix_ids=suffix,anchor_text=tokenizers['qwen'].decode(anchor),suffix_text=tokenizers['qwen'].decode(suffix),
            question_option_text_directly_prefilled=False,dummy_candidate=records[0]['original']['content_layout']['label_ids'][0]),
        source_reference_backend=E.C.CFG['attention'],reordered_capture_backend=CFG['attention_backend'],n=len(records),
        donor_fallback_pairs=sum(not d['same_option_count'] for r in records for d in r['donors']),
        donor_exact_length_pairs=sum(d['exact_length'] for r in records for d in r['donors']))
    assert manifest['source_reference_backend']==CFG['attention_backend']=='sdpa'
    if(ROOT/'manifest.json').exists():assert read(ROOT/'manifest.json')==manifest
    else:save(ROOT/'manifest.json',manifest)
    save(ROOT/'prepared/completed.json',dict(status='completed',manifest_sha256=identity(),counts=CFG['tasks']))
    status(status='prepared',phase='data');E.log('KV CONTENT DEPENDENCE PREPARED')


def verify():
    K.verify();m=read(ROOT/'manifest.json')
    assert K.identity()==m['parent_manifest_sha256'] and sha(K.__file__)==m['parent_code_sha256']
    assert sha(ROOT/'config.json')==m['config_sha256'] and sha(ROOT/'prepared/questions.jsonl')==m['prepared_sha256']
    for name,digest in m['code'].items():assert sha(ROOT/name)==digest,name


def load_model(family):
    free=torch.cuda.mem_get_info()[0]/2**30;assert free>=CFG['free_gpu_gib'],free
    assert shutil.disk_usage(ROOT).free/2**30>=CFG['disk_reserve_gib']
    torch.manual_seed(CFG['seed'])
    model=E.AutoModelForCausalLM.from_pretrained(E.CFG['models'][family],dtype=torch.float16,device_map='cuda',local_files_only=True,
        attn_implementation=CFG['attention_backend']).eval().requires_grad_(False)
    E.log(f'LOADED {family} SDPA; free before load {free:.2f} GiB')
    return E.tok(family),model


@torch.inference_mode()
def capture_reordered():
    verify();_,model=load_model('llama')
    for i,record in enumerate(rows(),1):
        row=record['reordered'];status(status='running',phase='capture_reordered',sample=i,total=256)
        assert not source_path(record).exists(),'Existing temporary source requires reviewed resume'
        ids=row['frames']['llama']['body_ids'];key,value,_=E.C.capture(model,ids,len(ids),[0]);ix=row['source_indices']
        source_path(record).parent.mkdir(exist_ok=True)
        torch.save(dict(key=key[:,ix].contiguous(),value=value[:,ix].contiguous(),manifest_sha256=identity()),source_path(record))
        if i%16==0:E.log(f'REORDERED SOURCE {i}/256')
    save(ROOT/'capture_completed.json',dict(status='completed',manifest_sha256=identity(),n=256))


@torch.inference_mode()
def map_source(base,source):
    parts=[]
    for start in range(0,source['key'].shape[1],64):
        with torch.amp.autocast('cuda',dtype=torch.float16):
            parts.append(base(source['key'][:,start:start+64][None].cuda(),source['value'][:,start:start+64][None].cuda()))
    key,value=(torch.cat([v[i][0] for v in parts],dim=1).cpu().half().contiguous() for i in range(2))
    assert torch.isfinite(key).all() and torch.isfinite(value).all()
    return key,value


@torch.inference_mode()
def map_all():
    verify();assert read(ROOT/'capture_completed.json')['manifest_sha256']==identity()
    _,model=load_model('qwen');base,reader=E.load_frontend(model)
    for i,record in enumerate(rows(),1):
        status(status='running',phase='map_kv',sample=i,total=256)
        for variant in ['original','reordered']:
            row=record[variant]
            source=E.U.loaded_source(row) if variant=='original' else torch.load(source_path(record),map_location='cpu',weights_only=True)
            if variant=='reordered':assert source['manifest_sha256']==identity()
            key,value=map_source(base,source);out=cache_path(record,variant);out.parent.mkdir(parents=True,exist_ok=True)
            assert not out.exists(),'Existing mapped cache requires reviewed resume'
            torch.save(dict(key=key,value=value,manifest_sha256=identity(),row_id=record['id'],variant=variant),out)
        source_path(record).unlink()
        if i%16==0:E.log(f'MAPPED ORIGINAL/REORDERED {i}/256')
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(not p.requires_grad and p.grad is None for p in base.parameters())
    save(ROOT/'mapping_completed.json',dict(status='completed',manifest_sha256=identity(),n=512))


def load_pair(record,variant):
    v=torch.load(cache_path(record,variant),map_location='cpu',weights_only=True)
    assert v['manifest_sha256']==identity() and v['row_id']==record['id'] and v['variant']==variant
    return v['key'],v['value']


@torch.inference_mode()
def anchor_cache(model):
    inputs=read(ROOT/'manifest.json')['direct_inputs'];ids=inputs['anchor_ids']
    tensor=torch.tensor([ids],device='cuda')
    return model.model(tensor,attention_mask=torch.ones_like(tensor),position_ids=torch.arange(len(ids),device='cuda')[None],use_cache=True).past_key_values


@torch.inference_mode()
def scores(model,row,anchor,key,value,blocked=False,dummy=None):
    frame=row['frames']['qwen'];b=len(frame['body_ids']);a=frame['native_prefix_length'];suffix=row['suffix_ids'];labels=row['content_layout']['label_ids']
    assert key.shape==value.shape and key.shape[1]==b-a
    cache=E.U.translated_cache(model,row,anchor,key.cuda(),value.cuda())
    dummy=labels[0] if dummy is None else dummy;ids=suffix+[dummy]
    mask=visibility(a,b,len(ids),blocked).cuda()
    if blocked:assert int((mask==0).sum())==b-a
    out=model.model(torch.tensor([ids],device='cuda'),past_key_values=cache,attention_mask=mask,
        position_ids=torch.arange(b,b+len(ids),device='cuda')[None],use_cache=False)
    logits=model.lm_head(out.last_hidden_state[:,len(suffix)-1])[0].float();lp=logits.log_softmax(-1)[labels]
    choice_lp=lp.log_softmax(-1);prediction=int(lp.argmax())
    assert torch.isfinite(logits).all()
    return dict(scores=lp.cpu().tolist(),prediction=prediction,correct=int(prediction==row['gold']),choice_nll=float(-choice_lp[row['gold']]),
        full_vocab_nll=float(-lp[row['gold']]),gold_probability=float(choice_lp[row['gold']].exp())),logits


@torch.inference_mode()
def smoke():
    verify();assert read(ROOT/'mapping_completed.json')['manifest_sha256']==identity()
    tokenizer,model=load_model('qwen');base,reader=E.load_frontend(model)
    with reader.mode(False):anchor=anchor_cache(model)
    checks=[]
    for task in CFG['tasks']:
        record=next(r for r in rows() if r['task']==task);row=record['original'];key,value=load_pair(record,'original')
        nk,nv=matched_noise(key,value,seed_for(CFG['seed'],row['id'],'smoke'))
        with reader.mode(True):
            true,logits=scores(model,row,anchor,key,value)
            canonical=E.U.candidate_scores(model,tokenizer,row,anchor,key.cuda(),value.cuda())
            masked,masked_logits=scores(model,row,anchor,key,value,True)
            noisy_masked,noisy_masked_logits=scores(model,row,anchor,nk,nv,True)
            _,dummy_logits=scores(model,row,anchor,key,value,dummy=row['content_layout']['label_ids'][1])
        score_error=max(abs(a-b) for a,b in zip(true['scores'],canonical['scores']))
        check=dict(task=task,id=row['id'],legacy_candidate_score_error=score_error,
            legacy_prediction_equal=true['prediction']==canonical['prediction'],
            masked_content_max_logit_error=float((masked_logits-noisy_masked_logits).abs().max()),
            future_dummy_max_logit_error=float((logits-dummy_logits).abs().max()),native_anchor_length=anchor.layers[0].keys.shape[-2])
        assert score_error<1e-4 and check['legacy_prediction_equal'],check
        assert check['masked_content_max_logit_error']==check['future_dummy_max_logit_error']==0,check
        assert check['native_anchor_length']==row['frames']['qwen']['native_prefix_length'],check
        checks.append(check)
    baseline_results=[]
    for i,record in enumerate(rows(),1):
        row=record['original'];key,value=load_pair(record,'original')
        status(status='running',phase='true_baseline',sample=i,total=256,task=row['task'])
        with reader.mode(True):result,_=scores(model,row,anchor,key,value)
        save(ROOT/'baseline'/row['task']/(fname(record)+'.json'),dict(id=row['id'],manifest_sha256=identity(),result=result))
        baseline_results.append((row,result))
        if i%32==0:E.log(f'TRUE KV BASELINE {i}/256')
    baseline_tasks={}
    for task,n in CFG['tasks'].items():
        correct=sum(result['correct'] for row,result in baseline_results if row['task']==task)
        baseline_tasks[task]=dict(n=n,correct=correct,accuracy=correct/n)
        assert correct/n>=.5,('True FP16 KV baseline unusable; stop before controls',task,baseline_tasks[task])
    save(ROOT/'baseline/completed.json',dict(status='completed',manifest_sha256=identity(),tasks=baseline_tasks))
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    save(ROOT/'smoke/completed.json',dict(status='completed',manifest_sha256=identity(),checks=checks,all_parameters_frozen=True,
        direct_inputs=read(ROOT/'manifest.json')['direct_inputs'],baseline_tasks=baseline_tasks))
    E.log('KV CONTENT MASK / NO LEAK / LEGACY SCORER SMOKE PASSED')


@torch.inference_mode()
def evaluate():
    verify();assert read(ROOT/'smoke/completed.json')['manifest_sha256']==identity()
    tokenizer,model=load_model('qwen');base,reader=E.load_frontend(model)
    with reader.mode(False):anchor=anchor_cache(model)
    records=rows();lookup={r['id']:r for r in records}
    for i,record in enumerate(records,1):
        assert not result_path(record).exists(),'Existing result requires reviewed resume'
        row=record['original'];key,value=load_pair(record,'original');arms={};base_result=None
        def run(name,r,k,v,blocked=False):
            status(status='running',phase='evaluation',sample=i,total=256,task=row['task'],condition=name)
            with reader.mode(True):result,_=scores(model,r,anchor,k,v,blocked)
            if base_result is not None:result['delta_choice_nll']=result['choice_nll']-base_result['choice_nll']
            return result
        baseline=read(ROOT/'baseline'/row['task']/(fname(record)+'.json'))
        assert baseline['manifest_sha256']==identity()
        arms['true']=baseline['result'];base_result=arms['true'];arms['true']['delta_choice_nll']=0.
        arms['masked']=run('masked',row,key,value,True)
        donor_diagnostics=[]
        for rep,donor in enumerate(record['donors']):
            dk,dv=load_pair(lookup[donor['id']],'original');dk,dv,indices=resize_pair(dk,dv,key.shape[1])
            arms[f'wrong_{rep}']=run(f'wrong_{rep}',row,dk,dv)
            noise_seed=seed_for(CFG['seed'],record['id'],rep,'noise')
            nk,nv=matched_noise(key,value,noise_seed)
            arms[f'random_target_stats_{rep}']=run(f'random_target_stats_{rep}',row,nk,nv)
            nk,nv=matched_noise(dk,dv,noise_seed)
            arms[f'random_donor_stats_{rep}']=run(f'random_donor_stats_{rep}',row,nk,nv)
            donor_diagnostics.append(dict(**donor,resample_indices=indices))
            del dk,dv,nk,nv
        reordered=record['reordered'];rk,rv=load_pair(record,'reordered')
        arms['reordered_true']=run('reordered_true',reordered,rk,rv)
        arms['reordered_masked']=run('reordered_masked',reordered,rk,rv,True)
        oldk,oldv,indices=resize_pair(key,value,rk.shape[1])
        arms['reordered_old_kv']=run('reordered_old_kv',reordered,oldk,oldv)
        save(result_path(record),dict(id=record['id'],task=record['task'],manifest_sha256=identity(),gold=row['gold'],reordered_gold=reordered['gold'],
            permutation=record['permutation'],donors=donor_diagnostics,original_body_length=len(row['frames']['qwen']['body_ids']),
            reordered_body_length=len(reordered['frames']['qwen']['body_ids']),reordered_old_kv_indices=indices,arms=arms))
        del key,value,rk,rv,oldk,oldv
        if i%8==0:E.log(f'KV CONTENT EVAL {i}/256')
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    save(ROOT/'evaluation/completed.json',dict(status='completed',manifest_sha256=identity(),n=256))


def summarize():
    verify();assert read(ROOT/'evaluation/completed.json')['manifest_sha256']==identity()
    summary=dict(experiment=CFG['name'],status='completed',timestamp=time.strftime('%F %T'),manifest_sha256=identity(),config=CFG,
        checkpoint_identity=read(ROOT/'manifest.json')['checkpoints'],direct_inputs=read(ROOT/'manifest.json')['direct_inputs'],tasks={},
        interpretation='Target-moment Gaussian noise retains aggregate target information; compare donor-moment noise and mismatched natural KV before attributing effects to model priors. MCQ questions/options themselves enter through KV.',
        uncertainty='Question-level paired bootstrap, average repeats within question; uncorrected exploratory95% intervals.')
    for task,n in CFG['tasks'].items():
        prepared=[r for r in rows() if r['task']==task];records=[read(result_path(r)) for r in prepared]
        gold_counts=Counter(r['gold'] for r in records)
        chance=float(np.mean([1/len(r['original']['candidates']) for r in prepared]))
        task_result=dict(n=n,uniform_random_accuracy=chance,majority_label_accuracy=max(gold_counts.values())/n,gold_label_counts=dict(gold_counts),arms={})
        groups=dict(true=['true'],masked=['masked'],wrong=[f'wrong_{i}' for i in range(CFG['repeats'])],
            random_target_stats=[f'random_target_stats_{i}' for i in range(CFG['repeats'])],random_donor_stats=[f'random_donor_stats_{i}' for i in range(CFG['repeats'])],
            reordered_true=['reordered_true'],reordered_masked=['reordered_masked'],reordered_old_kv=['reordered_old_kv'])
        for name,names in groups.items():
            val=lambda key:[float(np.mean([r['arms'][a][key] for a in names])) for r in records]
            correct=val('correct');nll=val('choice_nll')
            difference=[float(np.mean([r['arms'][a]['choice_nll'] for a in names]))-r['arms']['true']['choice_nll'] for r in records]
            changed=[float(np.mean([r['arms'][a]['prediction']!=r['arms']['true']['prediction'] for a in names])) for r in records]
            loss=[r['arms']['true']['correct']-v for r,v in zip(records,correct)]
            task_result['arms'][name]=dict(n=n,correct=sum(correct),accuracy=float(np.mean(correct)),mean_choice_nll=float(np.mean(nll)),
                delta_choice_nll=K.bootstrap(difference,CFG['seed']),accuracy_drop_vs_true=K.bootstrap(loss,CFG['seed']),
                prediction_changed_vs_true=float(np.mean(changed)),predicted_label_counts=dict(Counter(r['arms'][a]['prediction'] for r in records for a in names)))
        true_correct=[r for r in records if r['arms']['true']['correct']]
        tracking=[int(r['permutation'][r['arms']['reordered_true']['prediction']]==r['arms']['true']['prediction']) for r in records]
        task_result['option_reordering']=dict(gold_label_changed=n,content_prediction_tracking=K.bootstrap(tracking,CFG['seed']),
            same_label_rate=float(np.mean([r['arms']['reordered_true']['prediction']==r['arms']['true']['prediction'] for r in records])),
            original_correct_n=len(true_correct),original_correct_retained=sum(r['arms']['reordered_true']['correct'] for r in true_correct),
            old_kv_keeps_original_prediction=float(np.mean([r['arms']['reordered_old_kv']['prediction']==r['arms']['true']['prediction'] for r in records])))
        for subset in ['exact_length','same_option_count']:
            usable=[];pairs=0
            for r in records:
                names=[f'wrong_{i}' for i,d in enumerate(r['donors']) if d[subset]]
                if names:usable.append((r,names));pairs+=len(names)
            task_result['wrong_'+subset+'_subset']=dict(question_count=len(usable),donor_pair_count=pairs,
                true_accuracy=float(np.mean([r['arms']['true']['correct'] for r,names in usable])) if usable else None,
                wrong_accuracy=float(np.mean([np.mean([r['arms'][a]['correct'] for a in names]) for r,names in usable])) if usable else None)
        # An interpretable operating-point check; not used to choose samples or tune controls.
        task_result['true_baseline_usable']=task_result['arms']['true']['accuracy']>=.5
        summary['tasks'][task]=task_result
    save(ROOT/'summary.json',summary)
    import fcntl
    shared=ROOT.parents[2]/'EXPERIMENT_RESULTS.md'
    with shared.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.write('\n\n## '+CFG['name']+'\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n');stream.flush();os.fsync(stream.fileno())
    for r in rows():
        for variant in ['original','reordered']:cache_path(r,variant).unlink()
    save(ROOT/'completed.json',dict(status='completed',manifest_sha256=identity(),temporary_kv_cleaned=True));status(status='completed',phase='all');E.log('KV CONTENT DEPENDENCE COMPLETE')


if __name__=='__main__':
    try:{'prepare':prepare,'capture':capture_reordered,'map':map_all,'smoke':smoke,'evaluate':evaluate,'summarize':summarize}[sys.argv[1]]()
    except Exception as exc:
        import traceback
        save(ROOT/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        status(status='failed',phase=sys.argv[1],error=str(exc));raise
