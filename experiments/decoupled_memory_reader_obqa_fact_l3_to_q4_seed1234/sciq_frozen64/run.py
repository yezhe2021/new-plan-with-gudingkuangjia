"""Evaluation-only SciQ transfer of the three frozen OBQA Reader checkpoints."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,sys,time,random,hashlib,urllib.request,fcntl,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
PARENT=Path(CFG['parent'])
sys.path.insert(0,str(PARENT))
spec=importlib.util.spec_from_file_location('evidence_reader_pilot',PARENT/'run.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
C=P.C
import torch
from transformers import AutoTokenizer


def ident():return C.sha(ROOT/'manifest.json')
def status(**kw):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**kw))
def rows(smoke=False):
    rr=C.read_jsonl(ROOT/'prepared/test.jsonl')
    if not smoke:return rr
    return [rr[0],next(r for r in rr[1:] if r['evidence_key']!=rr[0]['evidence_key'])]
def norm(s):return ' '.join(s.casefold().split())
def checkpoint(arm):return PARENT/'runs'/arm/'best.pt'


def fetch():
    """Only public, credential-free test parquet; pin repository revision first."""
    if (ROOT/'data/source.json').exists():
        source=json.loads((ROOT/'data/source.json').read_text())
        assert C.sha(ROOT/'data/test.parquet')==source['parquet_sha256'];return
    error=[]
    for endpoint in ['https://huggingface.co','https://hf-mirror.com']:
        try:
            def get(url):
                req=urllib.request.Request(url,headers={'User-Agent':'KV-Reader-SciQ-evaluation/1.0'})
                with urllib.request.urlopen(req,timeout=20) as resp:return resp.read()
            api=json.loads(get(endpoint+'/api/datasets/'+CFG['dataset_repo']))
            revision=api['sha']
            names=[x['rfilename'] for x in api['siblings'] if x['rfilename'].startswith('data/test-') and x['rfilename'].endswith('.parquet')]
            assert len(names)==1,names
            url=endpoint+'/datasets/'+CFG['dataset_repo']+'/resolve/'+revision+'/'+names[0]
            payload=get(url)
            assert payload[:4]==b'PAR1' and len(payload)<20_000_000,'Unexpected dataset payload'
            p=ROOT/'data/test.parquet';p.parent.mkdir(parents=True,exist_ok=True)
            temporary=p.with_suffix('.download');temporary.write_bytes(payload);temporary.replace(p)
            C.save(ROOT/'data/source.json',dict(repo=CFG['dataset_repo'],revision=revision,file=names[0],download_url=url,
                official_card=CFG['dataset_card'],parquet_sha256=C.sha(p),bytes=len(payload),license='cc-by-nc-3.0',prior_connection_errors=error))
            C.log(f'SciQ official test downloaded: revision={revision}, bytes={len(payload)}');return
        except Exception as e:error.append(dict(endpoint=endpoint,type=type(e).__name__,message=str(e)))
    raise RuntimeError('Cannot download public SciQ test: '+str(error))


def prepare():
    assert (PARENT/'completed.json').exists(),'Reader training not completed'
    for arm,sha in CFG['reader_sha256'].items():assert C.sha(checkpoint(arm))==sha,('Frozen Reader changed',arm)
    for stage in ['a','b']:assert C.sha(Path(P.CFG['parent'])/f'runs/stage_{stage}/best.pt')==P.CFG[f'stage_{stage}_sha256']
    fetch()
    import pyarrow.parquet as pq
    raw=pq.read_table(ROOT/'data/test.parquet').to_pylist();assert len(raw)==1000
    blocked_q=set();blocked_e=set()
    for r in P.rows('train'):blocked_q.add(r['question_key']);blocked_e.add(r['evidence_key'])
    for r in C.read_jsonl(C.ROOT/'prepared/train.jsonl'):
        d=r['doc'];q=d.get('question_stem',d.get('question'));q=q['stem'] if isinstance(q,dict) else q
        blocked_q.add(norm(q))
    chosen=[];skips=[];seen=set()
    for index,d in enumerate(raw):
        options=[d['distractor1'],d['distractor2'],d['distractor3'],d['correct_answer']]
        reason=None
        if not d['support'].strip():reason='empty_support'
        elif not all(x.strip() for x in options) or len({norm(x) for x in options})!=4:reason='non_distinct_or_empty_answers'
        elif norm(d['question']) in blocked_q:reason='known_training_question_overlap'
        elif norm(d['support']) in blocked_e:reason='reader_training_support_overlap'
        elif norm(d['question']) in seen:reason='duplicate_test_question'
        if reason:skips.append(dict(index=index,reason=reason));continue
        chosen.append((index,d));seen.add(norm(d['question']))
        if len(chosen)==CFG['count']:break
    assert len(chosen)==CFG['count']
    lt,qt=[AutoTokenizer.from_pretrained(C.CFG['models'][f],local_files_only=True) for f in ['llama','qwen']]
    positions=list(range(4))*(CFG['count']//4);random.Random(CFG['seed']).shuffle(positions)
    prepared=[]
    for ordinal,(index,d) in enumerate(chosen):
        options=[d['distractor1'],d['distractor2'],d['distractor3']]
        random.Random(CFG['seed']+index).shuffle(options);gold=positions[ordinal];options.insert(gold,d['correct_answer'])
        doc=dict(id='sciq_'+str(index),question=d['question'],choices=dict(label=list('ABCD'),text=options),answerKey='ABCD'[gold])
        prompt,candidates,gold2,kind,_=C.S.format_item('arc',doc);assert gold==gold2
        evidence=d['support'].strip()
        frames={f:C.frame(t,f,'chat',evidence,len(evidence)) for f,t in [('llama',lt),('qwen',qt)]}
        indices=P.align_all(lt,qt,evidence,frames)
        prefix,assistant=C.chat_affixes(qt,thinking=False);cue=C.S.ANSWER_CUE[kind]
        labels=[qt.encode(c,add_special_tokens=False) for c in candidates];assert all(len(x)==1 for x in labels)
        rendered=prefix+prompt+assistant+cue
        prepared.append(dict(id='sciq_'+str(index),index=index,doc_sha256=C.digest(d),question=d['question'],evidence=evidence,evidence_key=norm(evidence),
            prompt=prompt,candidates=candidates,gold=gold,choice_texts=options,label_ids=[x[0] for x in labels],context_ids=qt.encode(rendered,add_special_tokens=False),
            rendered_receiver=rendered,frames=frames,source_indices=indices,
            full_evidence_context_ids=qt.encode(prefix+'Evidence:\n'+evidence+'\n\n'+prompt+assistant+cue,add_special_tokens=False)))
    manifest=dict(config=CFG,parent_manifest_sha256=C.sha(PARENT/'manifest.json'),parent_summary_sha256=C.sha(PARENT/'summary.json'),
        source=json.loads((ROOT/'data/source.json').read_text()),prepared_digest=C.digest(prepared),selected_indices=[i for i,_ in chosen],skips=skips,
        parent_code={str(p):C.sha(p) for p in [PARENT/'run.py',PARENT/'reader.py',C.ROOT/'core.py']},
        code={p.name:C.sha(p) for p in ROOT.glob('*.py')},model_configs={f:C.sha(Path(p)/'config.json') for f,p in C.CFG['models'].items()},
        versions={k:C.importlib.metadata.version(k) for k in ['torch','transformers','pyarrow']},dtype='float16',attention=C.CFG['attention'],
        writer_sha256={s:P.CFG[f'stage_{s}_sha256'] for s in ['a','b']},
        training=False,exact_question_overlap_with_known_writer_reader_train=0,exact_support_overlap_with_reader_train=0,
        audit_limits='Exact normalized overlap only; not semantic near-duplicate or pretraining contamination audit',gold_positions={k:positions.count(i) for i,k in enumerate('ABCD')},
        tokens=dict(min=min(len(r['source_indices']) for r in prepared),max=max(len(r['source_indices']) for r in prepared),mean=sum(len(r['source_indices']) for r in prepared)/len(prepared)))
    p=ROOT/'manifest.json'
    if p.exists():assert json.loads(p.read_text())==manifest,'Immutable evaluation protocol changed'
    else:C.save(p,manifest)
    C.write_jsonl(ROOT/'prepared/test.jsonl',prepared);C.log(f'PREPARED SciQ {len(prepared)}; evidence tokens {manifest["tokens"]}; skips={len(skips)}')


@torch.no_grad()
def capture():
    todo=[r for r in rows() if not (ROOT/'source'/(r['id']+'.pt')).exists()]
    if not todo:return
    tok,model=C.model('llama')
    for i,r in enumerate(todo,1):
        ids=r['frames']['llama']['body_ids'];k,v,_=C.capture(model,ids,len(ids),[0])
        C.save_tensor(ROOT/'source'/(r['id']+'.pt'),dict(key=k[:,r['source_indices']].contiguous(),value=v[:,r['source_indices']].contiguous(),manifest_sha256=ident()))
        if i%8==0:C.log(f'SciQ support-only Sender capture {i}/{len(todo)}')


@torch.no_grad()
def translate():
    todo=[r for r in rows() if not (ROOT/'memory'/(r['id']+'.pt')).exists()]
    if not todo:return
    base=C.base_from_best();adapter=C.adapter_from_best()
    for i,r in enumerate(todo,1):
        x=torch.load(ROOT/'source'/(r['id']+'.pt'),map_location='cpu',weights_only=True);assert x['manifest_sha256']==ident()
        k,v=C.map_base(base,x);k,v=C.adapted(adapter,k,v)
        C.save_tensor(ROOT/'memory'/(r['id']+'.pt'),dict(key=k.cpu().contiguous(),value=v.cpu().contiguous(),manifest_sha256=ident()))
        if i%8==0:C.log(f'SciQ frozen Writer mapping {i}/{len(todo)}')


def memory(r):
    x=torch.load(ROOT/'memory'/(r['id']+'.pt'),map_location='cpu',weights_only=True);assert x['manifest_sha256']==ident()
    return x['key'].cuda(),x['value'].cuda()


@torch.no_grad()
def evaluate(arm,smoke=False):
    mode='smoke' if smoke else 'formal';folder=ROOT/mode/arm
    if (folder/'completed.json').exists():return
    tok,model=C.model('qwen');rr=rows(smoke);anchor=P.native_anchor(model,rr);reader=None
    if arm!='baselines':
        assert C.sha(checkpoint(arm))==CFG['reader_sha256'][arm]
        reader=P.make_reader(model,arm);x=torch.load(checkpoint(arm),map_location='cpu',weights_only=True)
        assert x['manifest_sha256']==P.identity() and x['arm']==arm
        P.restore(reader,x['state']);reader.eval()
    for i,r in enumerate(rr):
        path=folder/'per_sample'/(r['id']+'.json')
        if path.exists():assert json.loads(path.read_text())['manifest_sha256']==ident();continue
        status(status='running',phase=mode,arm=arm,sample=i+1,total=len(rr))
        wrong=P.wrong_row(rr,i);conditions={}
        with torch.amp.autocast('cuda',dtype=torch.float16):
            if arm=='baselines':
                conditions['question_only']=P.result(P.context_forward(model,r),r)
                conditions['native_full_evidence_text']=P.result(P.context_forward(model,r,full=True),r)
                conditions['frozen_stage_b_physical_prefix']=P.result(P.context_forward(model,r,memory(r),anchor=anchor,physical=True),r)
            else:
                if smoke:
                    baseline=P.context_forward(model,r)
                    disabled=P.context_forward(model,r,None,reader,arm,anchor)
                    assert torch.equal(baseline,disabled),'Disabled Reader native parity failed'
                for name,mem in [('correct_memory',memory(r)),('no_memory',None),('wrong_memory',memory(wrong))]:
                    conditions[name]=P.result(P.context_forward(model,r,mem,reader,arm,anchor),r)
        C.save(path,dict(id=r['id'],index=r['index'],question=r['question'],support=r['evidence'],choice_texts=r['choice_texts'],gold=r['gold'],
            evidence_key=r['evidence_key'],wrong_memory_id=wrong['id'],wrong_support=wrong['evidence'],conditions=conditions,manifest_sha256=ident(),
            reader_checkpoint_sha256=CFG['reader_sha256'].get(arm),evidence_tokens=len(r['source_indices'])))
        if (i+1)%8==0 or smoke:C.log(f'SciQ {mode} {arm} {i+1}/{len(rr)}')
    records=[json.loads((folder/'per_sample'/(r['id']+'.json')).read_text()) for r in rr]
    C.write_jsonl(folder/'per_sample.jsonl',records);C.save(folder/'completed.json',dict(status='completed',samples=len(rr),manifest_sha256=ident()))
    if reader is not None:P.detach(model,reader,arm)


def summarize():
    merged={r['id']:dict(id=r['id'],question=r['question'],support=r['evidence'],gold=r['gold'],choices=r['choice_texts'],conditions={}) for r in rows()}
    for arm in ['baselines']+CFG['arms']:
        assert (ROOT/f'formal/{arm}/completed.json').exists()
        for r in C.read_jsonl(ROOT/f'formal/{arm}/per_sample.jsonl'):
            merged[r['id']]['conditions'].update({arm+'/'+k:v for k,v in r['conditions'].items()})
    records=list(merged.values());keys=list(records[0]['conditions']);n=len(records)
    metrics={k:dict(n=n,correct=sum(r['conditions'][k]['correct'] for r in records),accuracy=sum(r['conditions'][k]['correct'] for r in records)/n,
        gold_ce=sum(r['conditions'][k]['gold_ce'] for r in records)/n) for k in keys}
    paired={}
    for arm in CFG['arms']:
        paired[arm]={}
        for control in ['no_memory','wrong_memory']:
            a=[r['conditions'][arm+'/correct_memory'] for r in records];b=[r['conditions'][arm+'/'+control] for r in records]
            paired[arm][control]=dict(both_correct=sum(x['correct'] and y['correct'] for x,y in zip(a,b)),correct_memory_only=sum(x['correct'] and not y['correct'] for x,y in zip(a,b)),
                control_only=sum(not x['correct'] and y['correct'] for x,y in zip(a,b)),both_wrong=sum(not x['correct'] and not y['correct'] for x,y in zip(a,b)),
                prediction_identical=sum(x['prediction']==y['prediction'] for x,y in zip(a,b)))
    summary=dict(experiment=CFG['name'],status='completed',training=False,metrics=metrics,paired=paired,config=CFG,
        caveats=['Gold support supplied; not retrieval. No-memory bypasses Reader; wrong supports may differ in length and are not guaranteed contradictory.',
        'Reader and Writer are frozen OBQA/MCQ models. This is a cross-dataset evidence protocol diagnostic, not unmodified lm-eval SciQ scoring.'])
    C.write_jsonl(ROOT/'formal/per_sample.jsonl',records);C.save(ROOT/'summary.json',summary)
    shared=ROOT.parent.parent/'EXPERIMENT_RESULTS.md'
    entry=dict(summary,time=time.strftime('%F %T'),manifest=str(ROOT/'manifest.json'),raw_results=str(ROOT/'formal/per_sample.jsonl'),log=str(ROOT/'logs/pipeline.log'))
    with shared.open('a',encoding='utf-8') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write('\n\n## '+CFG['name']+'\n\n```json\n'+json.dumps(entry,ensure_ascii=False,indent=2)+'\n```\n');f.flush();os.fsync(f.fileno())
    C.save(ROOT/'shared_summary_updated.json',dict(status='completed',path=str(shared)))
    C.save(ROOT/'completed.json',dict(status='completed',manifest_sha256=ident()));status(status='completed',phase='all');C.log('ALL SCIQ FROZEN READER EVALUATIONS COMPLETED')


if __name__=='__main__':
    try:
        C.seed();action=sys.argv[1]
        if action=='prepare':prepare()
        elif action=='capture':capture()
        elif action=='translate':translate()
        elif action=='smoke':evaluate(sys.argv[2],True)
        elif action=='evaluate':evaluate(sys.argv[2])
        elif action=='summarize':summarize()
        else:raise ValueError(action)
    except Exception as e:
        error=dict(status='failed_or_incomplete',type=type(e).__name__,message=str(e))
        C.save(ROOT/'failure.json',error)
        entry=dict(experiment=CFG['name'],time=time.strftime('%F %T'),status='failed_or_incomplete',config=CFG,error=error,
            raw_results=str(ROOT/'formal'),log=str(ROOT/'logs/pipeline.log'))
        with (ROOT.parent.parent/'EXPERIMENT_RESULTS.md').open('a',encoding='utf-8') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.write('\n\n## '+CFG['name']+'\n\n```json\n'+json.dumps(entry,ensure_ascii=False,indent=2)+'\n```\n');f.flush();os.fsync(f.fileno())
        status(status='failed',error=str(e));raise
