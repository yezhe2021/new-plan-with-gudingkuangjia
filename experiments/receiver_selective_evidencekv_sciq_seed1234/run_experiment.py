"""Frozen evidence-only Sender; native-question Receiver; packet selection pilot."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_OFFLINE']='1'
import copy,hashlib,json,sys,time,random,shutil,inspect
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
sys.path.insert(0,CFG['reference'])
import experiment as E
sys.path.insert(0,str(ROOT))
from gate_rules import seed_for,feature,decide,fit,bootstrap
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(1)


def read(p):return json.loads(Path(p).read_text())
def save(p,x):E.save(p,x)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident():return sha(ROOT/'manifest.json')
def norm(s):return ' '.join(s.casefold().split())
def rows(split=None):
    records=[json.loads(line) for line in (ROOT/'prepared/questions.jsonl').read_text().splitlines()]
    return [r for r in records if split is None or r['split']==split]
def status(**kw):save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**kw))
def temp(row,kind):return ROOT/('temporary_'+kind)/(row['id']+'.pt')
def result_path(row):return ROOT/'evaluation'/row['split']/(row['id']+'.json')


def prepare():
    E.verify()
    import pyarrow.parquet as pq
    tokenizers={f:E.tok(f) for f in ['llama','qwen']};qt=tokenizers['qwen']
    old=[json.loads(line) for line in Path(CFG['previous_sciq']).read_text().splitlines()]
    blocked_questions={norm(r['question']) for r in old};blocked_supports={norm(r['evidence']) for r in old}
    for split in ['train','validation']:
        blocked_questions.update(r['question_key'] for r in E.U.rows(split))
    raw=pq.read_table(CFG['source_parquet']).to_pylist();assert len(raw)==1000
    order=list(range(len(raw)));random.Random(CFG['seed']).shuffle(order)
    counts=[('calibration',CFG['calibration_count']),('test',CFG['test_count'])]
    chosen=[];skips=Counter();seen_q=set(blocked_questions);seen_e=set(blocked_supports)
    for index in order:
        if len(chosen)==sum(n for _,n in counts):break
        d=raw[index];q=d['question'].strip();support=d['support'].strip()
        options=[d['correct_answer'],d['distractor1'],d['distractor2'],d['distractor3']]
        if not support or norm(q) in seen_q or norm(support) in seen_e or len({norm(v) for v in options})!=4 or not all(v.strip() for v in options):
            skips['empty_duplicate_or_excluded']+=1;continue
        permutation=list(range(4));random.Random(seed_for(CFG['seed'],index,'options')).shuffle(permutation)
        texts=[options[i].strip() for i in permutation];gold=permutation.index(0)
        evidence_prompt='Reference information:\n'+support+'\nEnd of reference'
        try:
            row=E.U.make_row('sciq_'+str(index),'gsm8k',dict(question=q),tokenizers['llama'],qt,
                prompt=evidence_prompt,candidates=list('ABCD'),gold=gold,normalize=False,cue='')
            frame=row['frames']['qwen'];assert row['prompt']==evidence_prompt and not row['native_tail']
            assert len(frame['body_ids'])<=CFG['max_evidence_tokens'] and len(row['frames']['llama']['body_ids'])<=CFG['max_evidence_tokens']
            before='\n\nAnswer the multiple-choice question. Use the reference only when relevant.\n\nQuestion: '
            query=before+q+'\n\nChoices:\n'+'\n'.join(label+'. '+text for label,text in zip('ABCD',texts))+'\n\nAnswer with only the option label.'
            rendered=frame['rendered_prefix']+evidence_prompt+query+frame['rendered_suffix']+'The correct option is '
            encoded=qt(rendered,add_special_tokens=False,return_offsets_mapping=True)
            allids=list(encoded['input_ids']);body=frame['body_ids'];assert allids[:len(body)]==body
            assert len(allids)<=CFG['max_total_tokens']
            qstart=len(frame['rendered_prefix'])+len(evidence_prompt)+len(before);qend=qstart+len(q)
            qpos=[i-len(body) for i,(a,b) in enumerate(encoded['offset_mapping']) if qstart<=a<b<=qend]
            assert qpos and min(qpos)>0
            label_ids=[E.C.R.candidate_ids(qt,label) for label in 'ABCD'];assert all(len(v)==1 for v in label_ids)
            compact=E.C.chat_affixes(qt,thinking=False)
            query_only=compact[0]+query.lstrip()+compact[1]+'The correct option is '
            row.update(question=q,support=support,choice_texts=texts,label_ids=[v[0] for v in label_ids],
                continuation_ids=allids[len(body):],question_positions=qpos,compact_ids=qt.encode(query_only,add_special_tokens=False),
                raw_index=index,raw_sha256=E.C.digest(d),evidence_key=norm(support),task='sciq')
            chosen.append(row);seen_q.add(norm(q));seen_e.add(norm(support))
        except (AssertionError,ValueError):skips['token_boundary_or_length']+=1
    assert len(chosen)==sum(n for _,n in counts),skips
    start=0
    for split,n in counts:
        group=chosen[start:start+n];start+=n
        lengths=np.array([len(r['source_indices']) for r in group]);cost=np.abs(lengths[:,None]-lengths[None,:]).astype(float)
        cost+=np.random.default_rng(seed_for(CFG['seed'],split,'donors')).uniform(0,.5,cost.shape)
        np.fill_diagonal(cost,1e9);a,b=linear_sum_assignment(cost)
        for i,j in zip(a,b):group[i].update(split=split,donor_id=group[j]['id']);assert i!=j
    anchors={tuple(r['frames']['qwen']['body_ids'][:r['frames']['qwen']['native_prefix_length']]) for r in chosen}
    assert len(anchors)==1
    out=ROOT/'prepared/questions.jsonl';out.parent.mkdir(exist_ok=True)
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen)
    if out.exists():assert out.read_text()==payload
    else:out.write_text(payload)
    checkpoints={k:v for k,v in read(E.ROOT/'manifest.json')['checkpoints'].items() if k in ['stage_a','stage_b']}
    m=dict(config=CFG,config_sha256=sha(ROOT/'config.json'),prepared_sha256=sha(out),source_sha256=sha(CFG['source_parquet']),
        previous_sciq_sha256=sha(CFG['previous_sciq']),reference_manifest_sha256=E.identity(),
        code={p.name:sha(p) for p in ROOT.glob('*.py') if not p.name.startswith('._')},checkpoints=checkpoints,
        anchor_ids=list(next(iter(anchors))),anchor_text=qt.decode(next(iter(anchors))),
        no_reader_installed=True,no_target_answer_in_sender_prompt=True,counts=dict(counts),skips=dict(skips),
        known_question_exclusion_count=len(blocked_questions),known_support_exclusion_count=len(blocked_supports),
        split_policy='Seeded disjoint subsets of cached official test parquet; calibration is not official validation split.',
        corruption_policy='Donors assigned within each split by support length without gold. Partial replacement affects identical K/V positions. Gaussian retains per-layer/head/channel target moments.',
        environment=dict(torch=torch.__version__,python=sys.version))
    if (ROOT/'manifest.json').exists():assert read(ROOT/'manifest.json')==m
    else:save(ROOT/'manifest.json',m)
    save(ROOT/'prepared/completed.json',dict(manifest_sha256=ident(),counts=dict(counts)))
    status(status='prepared',phase='prepare');E.log(f'PREPARED {dict(counts)}; skips {dict(skips)}')


def verify():
    E.verify();m=read(ROOT/'manifest.json')
    assert m['reference_manifest_sha256']==E.identity()
    assert sha(ROOT/'config.json')==m['config_sha256'] and sha(ROOT/'prepared/questions.jsonl')==m['prepared_sha256']
    for name,digest in m['code'].items():assert sha(ROOT/name)==digest,name


def load_model(family):
    assert torch.cuda.mem_get_info()[0]/2**30>=CFG['free_gpu_gib']
    assert shutil.disk_usage(ROOT).free/2**30>=CFG['disk_reserve_gib']
    torch.manual_seed(CFG['seed'])
    model=E.AutoModelForCausalLM.from_pretrained(E.CFG['models'][family],dtype=torch.float16,device_map='cuda',local_files_only=True,
        attn_implementation=CFG['attention_backend']).eval().requires_grad_(False)
    assert not any('LoRA' in type(m).__name__ for m in model.modules())
    E.log(f'LOADED {family}; no Q/O Reader');return model


def load_translator():
    cfg=E.R.CFG;m=read(ROOT/'manifest.json')
    base=E.R.GroupedKVTranslator(**cfg['architecture'])
    state=torch.load(m['checkpoints']['stage_a']['path'],map_location='cpu',weights_only=True);base.load_state_dict(state['state'],strict=True);del state
    base=E.R.FrozenTranslatorWithResidual(base,cfg['residual_rank'])
    state=torch.load(m['checkpoints']['stage_b']['path'],map_location='cpu',weights_only=True);base.residual.load_state_dict(state['state'],strict=True);del state
    return base.cuda().eval().requires_grad_(False)


@torch.inference_mode()
def capture():
    verify();model=load_model('llama');records=rows()
    for i,row in enumerate(records,1):
        assert not temp(row,'source').exists()
        status(status='running',phase='capture',sample=i,total=len(records))
        ids=row['frames']['llama']['body_ids'];k,v,_=E.C.capture(model,ids,len(ids),[0]);ix=row['source_indices']
        out=temp(row,'source');out.parent.mkdir(exist_ok=True)
        torch.save(dict(key=k[:,ix].contiguous(),value=v[:,ix].contiguous(),manifest_sha256=ident()),out)
        if i%16==0:E.log(f'EVIDENCE CAPTURE {i}/{len(records)}')
    save(ROOT/'capture_completed.json',dict(manifest_sha256=ident()))


@torch.inference_mode()
def translate():
    verify();assert read(ROOT/'capture_completed.json')['manifest_sha256']==ident();base=load_translator();records=rows()
    for i,row in enumerate(records,1):
        status(status='running',phase='translate',sample=i,total=len(records))
        source=torch.load(temp(row,'source'),map_location='cpu',weights_only=True);assert source['manifest_sha256']==ident()
        parts=[]
        for start in range(0,source['key'].shape[1],64):
            with torch.amp.autocast('cuda',dtype=torch.float16):parts.append(base(source['key'][:,start:start+64][None].cuda(),source['value'][:,start:start+64][None].cuda()))
        k,v=(torch.cat([p[j][0] for p in parts],1).cpu().half().contiguous() for j in range(2))
        assert torch.isfinite(k).all() and torch.isfinite(v).all()
        out=temp(row,'mapped');out.parent.mkdir(exist_ok=True);assert not out.exists()
        torch.save(dict(key=k,value=v,manifest_sha256=ident()),out);temp(row,'source').unlink()
        if i%16==0:E.log(f'EVIDENCE TRANSLATED {i}/{len(records)}')
    save(ROOT/'translate_completed.json',dict(manifest_sha256=ident()))


def memory(row):
    data=torch.load(temp(row,'mapped'),map_location='cpu',weights_only=True);assert data['manifest_sha256']==ident()
    return data['key'],data['value']


def resized(k,v,n):
    ix=((2*torch.arange(n)+1)*k.shape[1]//(2*n)).clamp_max(k.shape[1]-1)
    return k[:,ix].contiguous(),v[:,ix].contiguous()


def corrupted(row,key,value,donor):
    dk,dv=resized(*donor,key.shape[1]);n=key.shape[1]
    ix=np.random.default_rng(seed_for(CFG['seed'],row['id'],'partial')).permutation(n)[:int(np.ceil(n*CFG['partial_corruption_fraction']))]
    pk,pv=key.clone(),value.clone();pk[:,ix]=dk[:,ix];pv[:,ix]=dv[:,ix]
    noise=[]
    for role,x in enumerate([key,value]):
        xf=x.float();g=torch.Generator().manual_seed(seed_for(CFG['seed'],row['id'],role,'gaussian'))
        z=torch.randn(x.shape,generator=g);z=(z-z.mean(1,keepdim=True))/z.std(1,unbiased=False,keepdim=True).clamp_min(1e-20)
        noise.append((xf.mean(1,keepdim=True)+xf.std(1,unbiased=False,keepdim=True)*z).half())
    return dict(clean=(key,value),wrong=(dk,dv),partial_wrong=(pk,pv),gaussian=tuple(noise)),ix.tolist()


@torch.inference_mode()
def native_cache(model,ids):
    t=torch.tensor([ids],device='cuda')
    return model.model(t,attention_mask=torch.ones_like(t),position_ids=torch.arange(len(ids),device='cuda')[None],use_cache=True).past_key_values


def summarize_logits(logits,row,question_logp):
    lp=logits.float().log_softmax(-1)[row['label_ids']];cp=lp.log_softmax(-1);pred=int(cp.argmax())
    return dict(prediction=pred,correct=int(pred==row['gold']),choice_nll=float(-cp[row['gold']]),scores=lp.cpu().tolist(),
                choice_entropy=float(-(cp.exp()*cp).sum()),question_logp=float(question_logp))


@torch.inference_mode()
def score(model,row,cache,blocked=False):
    frame=row['frames']['qwen'];b=len(frame['body_ids']);a=frame['native_prefix_length']
    ids=row['continuation_ids']+[row['label_ids'][0]];mask=torch.ones(1,b+len(ids),device='cuda',dtype=torch.long)
    if blocked:mask[:,a:b]=0
    out=model.model(torch.tensor([ids],device='cuda'),past_key_values=copy.deepcopy(cache),attention_mask=mask,
        position_ids=torch.arange(b,b+len(ids),device='cuda')[None],use_cache=False)
    logits=model.lm_head(out.last_hidden_state[0,len(ids)-2]).float()
    positions=row['question_positions'];qlogits=model.lm_head(out.last_hidden_state[0,torch.tensor(positions,device='cuda')-1]).float()
    targets=torch.tensor([ids[p] for p in positions],device='cuda')
    qlp=-F.cross_entropy(qlogits,targets,reduction='mean')
    assert torch.isfinite(logits).all() and torch.isfinite(qlp)
    return summarize_logits(logits,row,qlp),logits


@torch.inference_mode()
def preflight():
    verify();assert read(ROOT/'translate_completed.json')['manifest_sha256']==ident()
    model=load_model('qwen');anchor=native_cache(model,read(ROOT/'manifest.json')['anchor_ids']);checks=[]
    lookup={r['id']:r for r in rows('calibration')}
    for row in rows('calibration')[:8]:
        k,v=memory(row);variants,_=corrupted(row,k,v,memory(lookup[row['donor_id']]))
        transferred=E.U.translated_cache(model,row,anchor,k.cuda(),v.cuda())
        accept,al=score(model,row,transferred);reject,rl=score(model,row,transferred,True)
        nk,nv=variants['gaussian'];noisy=E.U.translated_cache(model,row,anchor,nk.cuda(),nv.cuda())
        _,nl=score(model,row,noisy,True);error=float((rl-nl).abs().max());assert error==0,error
        full_native=native_cache(model,row['frames']['qwen']['body_ids']);native,cl=score(model,row,full_native)
        ids=row['frames']['qwen']['body_ids']+row['continuation_ids']+[row['label_ids'][0]]
        t=torch.tensor([ids],device='cuda');out=model.model(t,attention_mask=torch.ones_like(t),use_cache=False)
        fl=model.lm_head(out.last_hidden_state[0,-2]).float();score_error=float((cl[row['label_ids']]-fl[row['label_ids']]).abs().max())
        assert score_error<.1,score_error
        compact=row['compact_ids'];ct=torch.tensor([compact],device='cuda');co=model.model(ct,attention_mask=torch.ones_like(ct),use_cache=False)
        compact_result=summarize_logits(model.lm_head(co.last_hidden_state[0,-1]).float(),row,0)
        checks.append(dict(id=row['id'],masked_content_max_logit_error=error,native_cache_full_label_logit_error=score_error,
            native_cache_full_prediction_equal=int(cl[row['label_ids']].argmax())==int(fl[row['label_ids']].argmax()),
            transferred=accept,masked=reject,native_evidence=native,question_only_compact=compact_result))
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    save(ROOT/'preflight/completed.json',dict(manifest_sha256=ident(),n=len(checks),checks=checks,no_reader_installed=True))
    E.log('PREFLIGHT PASSED '+json.dumps({name:sum(r[name]['correct'] for r in checks) for name in ['transferred','masked','native_evidence','question_only_compact']}))


@torch.inference_mode()
def evaluate(split):
    verify();assert read(ROOT/'preflight/completed.json')['manifest_sha256']==ident()
    if split=='test':
        locked=read(ROOT/'selector.json');assert locked['manifest_sha256']==ident()
    model=load_model('qwen');anchor=native_cache(model,read(ROOT/'manifest.json')['anchor_ids'])
    selected=rows(split);lookup={r['id']:r for r in selected}
    for i,row in enumerate(selected,1):
        assert not result_path(row).exists()
        key,value=memory(row);variants,indices=corrupted(row,key,value,memory(lookup[row['donor_id']]))
        native=native_cache(model,row['frames']['qwen']['body_ids']);native_result,_=score(model,row,native)
        cache=E.U.translated_cache(model,row,anchor,key.cuda(),value.cuda());reject,_=score(model,row,cache,True)
        conditions={}
        for condition,(k,v) in variants.items():
            status(status='running',phase='evaluate_'+split,sample=i,total=len(selected),condition=condition)
            cache=E.U.translated_cache(model,row,anchor,k.cuda(),v.cuda());accept,_=score(model,row,cache)
            features=feature(accept,reject)
            decision={}
            if split=='test':decision={name:decide(features,rule) for name,rule in locked['rules'].items()}
            conditions[condition]=dict(accept=accept,reject=reject,features=features,decisions=decision)
        save(result_path(row),dict(id=row['id'],split=split,manifest_sha256=ident(),selector_sha256=sha(ROOT/'selector.json') if split=='test' else None,gold=row['gold'],donor_id=row['donor_id'],
            partial_replaced_indices=indices,native_evidence=native_result,conditions=conditions))
        if i%8==0:E.log(f'SELECTIVE {split} {i}/{len(selected)}')
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    save(ROOT/'evaluation'/split/'completed.json',dict(manifest_sha256=ident(),n=len(selected)))


def results(split):
    assert read(ROOT/'evaluation'/split/'completed.json')['manifest_sha256']==ident()
    return [read(result_path(row)) for row in rows(split)]


def calibrate():
    verify();records=results('calibration')
    assert not list((ROOT/'evaluation/test').glob('sciq_*.json')),'Selector must be fixed before test inference'
    samples=[r['conditions'][c] for r in records for c in CFG['conditions']]
    selected=fit(samples);selected.update(manifest_sha256=ident(),locked_at=time.strftime('%F %T'),
        calibration_only=True,thresholds_before_test=True,question_count=len(records))
    assert not (ROOT/'selector.json').exists();save(ROOT/'selector.json',selected)
    E.log('LOCKED SELECTOR '+json.dumps(selected,ensure_ascii=False))


def summarize():
    verify();records=results('test');locked=read(ROOT/'selector.json');n=len(records);rules=locked['rules']
    assert all(r['selector_sha256']==sha(ROOT/'selector.json') for r in records)
    summary=dict(experiment=CFG['name'],status='completed',timestamp=time.strftime('%F %T'),manifest_sha256=ident(),config=CFG,
        selector=locked,n_questions=n,condition_count=len(CFG['conditions']),conditions={},balanced_mixture={},
        caveats=['No Reader installed; Translator frozen. Evidence-only interface differs from Translator training interface.',
            'Candidate entropy and question likelihood are uncalibrated proxies; neither guarantees truth.',
            'Gaussian retains target aggregate moments; partial replacement is synthetic corruption.',
            'Calibration32 and evaluation128 come from disjoint subsets of official SciQ test, excluding prior64; not leaderboard scoring.',
            'Selectors require accept and reject evaluations; this pilot does not claim inference compute savings.',
            'Oracle is diagnostic and reads gold, unavailable at inference. Bootstrap by question; exploratory intervals.'])
    for condition in CFG['conditions']:
        group=[r['conditions'][condition] for r in records];metric={}
        for name,rule in rules.items():
            accepted=[decide(r['features'],rule) for r in group]
            values=[r['accept' if a else 'reject'] for r,a in zip(group,accepted)]
            metric[name]=dict(n=n,correct=sum(v['correct'] for v in values),accuracy=float(np.mean([v['correct'] for v in values])),
                mean_choice_nll=float(np.mean([v['choice_nll'] for v in values])),accept_rate=float(np.mean(accepted)),
                accuracy_gain_vs_accept=bootstrap([v['correct']-r['accept']['correct'] for v,r in zip(values,group)]),
                accuracy_gain_vs_reject=bootstrap([v['correct']-r['reject']['correct'] for v,r in zip(values,group)]))
        metric['oracle_best_of_two']=dict(correct=sum(max(r['accept']['correct'],r['reject']['correct']) for r in group),n=n)
        metric['useful_packet_count']=sum(r['accept']['correct'] and not r['reject']['correct'] for r in group)
        metric['harmful_packet_count']=sum(r['reject']['correct'] and not r['accept']['correct'] for r in group)
        summary['conditions'][condition]=metric
    for name,rule in rules.items():
        correct=[];gain_a=[];gain_r=[];acceptance=[];random_expected=[]
        overall_acceptance=np.mean([decide(r['conditions'][c]['features'],rule) for r in records for c in CFG['conditions']])
        for r in records:
            cs=[r['conditions'][c] for c in CFG['conditions']]
            selected=[v['accept' if decide(v['features'],rule) else 'reject']['correct'] for v in cs]
            correct.append(np.mean(selected));gain_a.append(np.mean([s-v['accept']['correct'] for s,v in zip(selected,cs)]))
            gain_r.append(np.mean([s-v['reject']['correct'] for s,v in zip(selected,cs)]))
            acceptance.extend(decide(v['features'],rule) for v in cs)
            random_expected.append(np.mean([overall_acceptance*v['accept']['correct']+(1-overall_acceptance)*v['reject']['correct'] for v in cs]))
        summary['balanced_mixture'][name]=dict(accuracy=float(np.mean(correct)),accept_rate=float(np.mean(acceptance)),
            question_bootstrap_accuracy=bootstrap(correct),accuracy_gain_vs_accept=bootstrap(gain_a),accuracy_gain_vs_reject=bootstrap(gain_r),
            rate_matched_random_expected_accuracy=float(np.mean(random_expected)))
    summary['native_clean_evidence']=dict(n=n,correct=sum(r['native_evidence']['correct'] for r in records))
    save(ROOT/'summary.json',summary)
    import fcntl
    with (ROOT.parent/'EXPERIMENT_RESULTS.md').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write('\n\n## '+CFG['name']+'\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n');f.flush();os.fsync(f.fileno())
    for r in rows():temp(r,'mapped').unlink()
    save(ROOT/'completed.json',dict(status='completed',manifest_sha256=ident(),temporary_tensors_cleaned=True))
    status(status='completed',phase='all');E.log('SELECTIVE RECEIVER COMPLETE')


if __name__=='__main__':
    action=sys.argv[1]
    try:
        actions=dict(prepare=prepare,capture=capture,translate=translate,preflight=preflight,calibration=lambda:evaluate('calibration'),
            calibrate=calibrate,test=lambda:evaluate('test'),summarize=summarize)
        actions[action]()
    except Exception as exc:
        import traceback
        save(ROOT/'failure.json',dict(action=action,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        status(status='failed',phase=action,error=str(exc));raise
