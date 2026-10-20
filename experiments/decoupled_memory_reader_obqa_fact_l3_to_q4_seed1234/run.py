"""Evidence-only frozen Writer + question-only Reader training pilot."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import sys,json,time,zipfile,random,math,importlib.util,fcntl,shutil
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
sys.path.insert(0,CFG['parent'])
import core as C
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from reader import DecoupledReader,trainable,state,restore


def log(x):C.log(x)
def identity():return C.sha(ROOT/'manifest.json')
def status(**kw):C.save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**kw))
def rows(split):return C.read_jsonl(ROOT/f'prepared/{split}.jsonl')
def cachepath(row):return ROOT/'memory'/(row['id']+'.pt')
def qkey(row):return ' '.join(row['question']['stem'].casefold().split())
def fkey(row):return ' '.join(row['fact1'].casefold().split())
def shared(status0,result):
    p=ROOT.parent/'EXPERIMENT_RESULTS.md'
    record=dict(experiment=CFG['name'],time=time.strftime('%F %T'),status=status0,config=CFG,
        results=result,manifest=str(ROOT/'manifest.json'),raw_results=str(ROOT/'evaluation/per_sample.jsonl'),log=str(ROOT/'logs/pipeline.log'))
    with p.open('a',encoding='utf-8') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write('\n\n## '+CFG['name']+'\n\n```json\n'+json.dumps(record,ensure_ascii=False,indent=2)+'\n```\n');f.flush();os.fsync(f.fileno())
    C.save(ROOT/'shared_summary_updated.json',dict(status=status0,path=str(p)))


def align_all(lt,qt,text,frames):
    # Full-Sync byte-boundary copy/drop, including the first evidence token.
    # No native content anchor exists inside the independent external bank.
    data=[]
    for tok in [lt,qt]:
        enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        data.append([(i,int(a),int(b)) for i,(a,b) in enumerate(enc['offset_mapping']) if b>a])
    source,target=data
    common=sorted(({b for _,_,b in source}&{b for _,_,b in target})|{0})
    selected={}
    for left,right in zip(common,common[1:]):
        si=[i for i,_,b in source if left<b<=right];ti=[i for i,_,b in target if left<b<=right]
        if not si or not ti:continue
        for rank,i in enumerate(ti):selected[i]=si[C.source_rank(len(si),len(ti),rank)]
    n=len(frames['qwen']['content_ids'])
    assert sorted(selected)==list(range(n)),('Incomplete evidence alignment',text)
    offset=len(frames['llama']['prefix_ids'])
    return [offset+selected[i] for i in range(n)]


def prepare():
    assert Path(CFG['prerequisite']).exists(),'Preceding Attention-Output diagnostic not complete'
    parent=Path(CFG['parent'])
    for stage in ['a','b']:
        assert C.sha(parent/f'runs/stage_{stage}/best.pt')==CFG[f'stage_{stage}_sha256']
    assert Path(CFG['old_reader']).exists()
    with zipfile.ZipFile(CFG['archive']) as z:
        raw={}
        members={}
        for split,file in [('train','train'),('validation','dev'),('test','test')]:
            n=next(n for n in z.namelist() if n.endswith(f'Additional/{file}_complete.jsonl'))
            raw[split]=[json.loads(x) for x in z.read(n).decode().splitlines() if x.strip()];members[split]=n
    old=json.loads((Path(C.CFG['legacy_root'])/'config.json').read_text())
    mainroot=Path(old['datasets']['openbookqa_root'])
    for split,file in [('train','train'),('validation','dev'),('test','test')]:
        byid={str(x['id']):x for x in C.read_jsonl(mainroot/(file+'.jsonl'))}
        for d in raw[split]:
            m=byid[str(d['id'])];assert d['question']==m['question'] and d['answerKey']==m['answerKey']
            assert isinstance(d['fact1'],str) and d['fact1'].strip()
    blocked_q={qkey(x) for s in ['validation','test'] for x in raw[s]}
    blocked_f={fkey(x) for s in ['validation','test'] for x in raw[s]}
    pool=[x for x in raw['train'] if qkey(x) not in blocked_q and fkey(x) not in blocked_f]
    random.Random(CFG['seed']).shuffle(pool)
    unique=[];seen=set()
    for d in pool:
        if qkey(d) not in seen:unique.append(d);seen.add(qkey(d))
    assert len(unique)>=CFG['counts']['train'],('Not enough leakage-filtered official train samples',len(unique))
    chosen=dict(train=unique[:CFG['counts']['train']],validation=raw['validation'][:CFG['counts']['validation']],test=raw['test'][:CFG['counts']['test']])
    assert not ({qkey(x) for x in chosen['validation']}&{qkey(x) for x in chosen['test']}),'Dev/test questions overlap'
    lt,qt=[AutoTokenizer.from_pretrained(C.CFG['models'][f],local_files_only=True) for f in ['llama','qwen']]
    rr={}
    for split,dd in chosen.items():
        rr[split]=[]
        for d in dd:
            question=dict(id=str(d['id']),question=d['question']['stem'],choices={k:[c[k] for c in d['question']['choices']] for k in ['label','text']},answerKey=d['answerKey'])
            prompt,candidates,gold,kind,_=C.S.format_item('arc',question)
            evidence=d['fact1'].strip()
            frames={f:C.frame(t,f,'chat',evidence,len(evidence)) for f,t in [('llama',lt),('qwen',qt)]}
            indices=align_all(lt,qt,evidence,frames)
            rt=C.R.Runtime.__new__(C.R.Runtime);rt.tokenizer=qt
            prefix,assistant=C.chat_affixes(qt,thinking=False)
            context=prefix+prompt+assistant+C.S.ANSWER_CUE[kind]
            ids=qt.encode(context,add_special_tokens=False)
            labels=[qt.encode(c,add_special_tokens=False) for c in candidates];assert all(len(x)==1 for x in labels)
            assert len(frames['qwen']['content_ids'])>0
            row=dict(id=split+'_'+str(d['id']),official_id=str(d['id']),split=split,doc_sha256=C.digest(d),question_key=qkey(d),evidence_key=fkey(d),
                evidence=evidence,question=d['question']['stem'],prompt=prompt,candidates=candidates,gold=gold,label_ids=[x[0] for x in labels],
                context_ids=ids,rendered_receiver=context,source_indices=indices,frames=frames,
                full_evidence_context_ids=qt.encode(prefix+'Evidence:\n'+evidence+'\n\n'+prompt+assistant+C.S.ANSWER_CUE[kind],add_special_tokens=False))
            rr[split].append(row)
    audit=dict(official_counts={s:len(v) for s,v in raw.items()},selected={s:len(v) for s,v in rr.items()},eligible_train_after_question_fact_filter=len(unique),
        train_vs_all_official_dev_test_question_overlap=0,train_vs_all_official_dev_test_normalized_fact_overlap=0,
        fact_filtered_training=True,validation_test_fact_overlap=len({fkey(x) for x in chosen['validation']}&{fkey(x) for x in chosen['test']}),
        train_question_duplicates=0,pretraining_contamination_not_audited=True,
        upstream_writer='Frozen parent MCQ Writer trained on official OBQA/ARC train; its dev/test question audit retained; not retrained on fact-only prompts',
        same_evidence_multiple_test_questions=sum(n>1 for n in Counter(x['evidence_key'] for x in rr['test']).values()))
    estimate=sum(2*2*36*len(x['source_indices'])*8*128*2 for ss in rr.values() for x in ss)/2**30
    assert shutil.disk_usage(ROOT).free/2**30>estimate+CFG['disk_reserve_gib']
    manifest=dict(config=CFG,parent_manifest_sha256=C.sha(parent/'manifest.json'),archive_sha256=C.sha(CFG['archive']),members=members,
        old_reader_sha256=C.sha(CFG['old_reader']),code={p.name:C.sha(p) for p in ROOT.glob('*.py')},
        parent_core_sha256=C.sha(parent/'core.py'),model_configs={f:C.sha(Path(p)/'config.json') for f,p in C.CFG['models'].items()},
        prepared={s:C.digest(v) for s,v in rr.items()},audit=audit,estimated_memory_gib=estimate,
        protocol_change='Evidence-only/Question-only intentional new protocol. External bank has no native header/content-anchor/tail and no autoregressive positions; frozen Writer weights unchanged. All evidence content tokens mapped with Full-Sync copy/drop.',
        layer_policy='Preregistered sparse 0-based [7,15,23,31], not selected from test or Attention-Output outcomes; not claimed optimal. Layer-location search deferred.',
        query_arms='Native frozen Q + zero-up low-rank correction vs independently initialized rank-matched factorized Q; both use frozen native O plus trainable zero-up low-rank correction and zero scalar gate.',
        old_baseline='Original coupled external reader algorithm; physical token0+memory+native question cache, selected-layer updates only; native/external joint softmax. Parameter count and shifted question positions recorded, not silently treated as matched.')
    p=ROOT/'manifest.json'
    if p.exists():assert json.loads(p.read_text())==manifest,'Immutable experiment definition changed'
    else:C.save(p,manifest)
    for s,ss in rr.items():C.write_jsonl(ROOT/f'prepared/{s}.jsonl',ss)
    C.save(ROOT/'data_audit.json',audit);C.save(ROOT/'prepared/completed.json',dict(status='completed',manifest_sha256=identity()))
    log(f'DATA AUDIT PASSED: {audit}; memory cache estimate {estimate:.2f} GiB')


@torch.no_grad()
def capture():
    rr=sum([rows(s) for s in CFG['counts']],[])
    todo=[r for r in rr if not (ROOT/'source'/(r['id']+'.pt')).exists()]
    if not todo:return
    tok,model=C.model('llama')
    for i,r in enumerate(todo,1):
        ids=r['frames']['llama']['body_ids'];k,v,_=C.capture(model,ids,len(ids),[0])
        selected=r['source_indices']
        C.save_tensor(ROOT/'source'/(r['id']+'.pt'),dict(key=k[:,selected].contiguous(),value=v[:,selected].contiguous(),manifest_sha256=identity(),doc_sha256=r['doc_sha256']))
        if i%32==0 or i==len(todo):log(f'Evidence-only Sender capture {i}/{len(todo)}')
        status(status='running',phase='sender_capture',sample=i,total=len(todo))


@torch.no_grad()
def translate():
    todo=[r for s in CFG['counts'] for r in rows(s) if not cachepath(r).exists()]
    if not todo:return
    base=C.base_from_best();adapter=C.adapter_from_best()
    for i,r in enumerate(todo,1):
        x=torch.load(ROOT/'source'/(r['id']+'.pt'),map_location='cpu',weights_only=True)
        assert x['manifest_sha256']==identity() and x['doc_sha256']==r['doc_sha256']
        k,v=C.map_base(base,x);bk,bv=C.adapted(adapter,k,v)
        C.save_tensor(cachepath(r),dict(stage_a_k=k.cpu().contiguous(),stage_a_v=v.cpu().contiguous(),key=bk.cpu().contiguous(),value=bv.cpu().contiguous(),
            doc_sha256=r['doc_sha256'],manifest_sha256=identity()))
        if i%32==0 or i==len(todo):log(f'Frozen Stage-A+B fact mapping {i}/{len(todo)}')
        status(status='running',phase='frozen_writer_mapping',sample=i,total=len(todo))
    C.save(ROOT/'memory/completed.json',dict(status='completed',samples=sum(CFG['counts'].values()),manifest_sha256=identity()))


def memory(r,stage='b'):
    x=torch.load(cachepath(r),map_location='cpu',weights_only=True)
    assert x['manifest_sha256']==identity() and x['doc_sha256']==r['doc_sha256']
    k,v=(x['key'],x['value']) if stage=='b' else (x['stage_a_k'],x['stage_a_v'])
    return k.cuda(),v.cuda()


def make_reader(model,arm):
    model.eval().requires_grad_(False)
    if arm!='old_coupled_reader':return DecoupledReader(model,CFG['layers_zero_based'],CFG['rank'],arm).cuda()
    spec=importlib.util.spec_from_file_location('original_coupled_reader',CFG['old_reader']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reader=module.ExternalReader(model.config,rank=CFG['old_reader_rank']).cuda()
    for i,branch in enumerate(reader.layers):
        if i not in CFG['layers_zero_based']:branch.requires_grad_(False)
    reader.attach(model)
    return reader


def detach(model,reader,arm):
    if arm!='old_coupled_reader':reader.detach()
    else:
        for block,old in zip(model.model.layers,reader.originals):block.self_attn.forward=old
        reader.originals=[]


def context_forward(model,r,mem=None,reader=None,arm=None,anchor=None,physical=False,full=False):
    ids=r['full_evidence_context_ids'] if full else r['context_ids']
    if (arm=='old_coupled_reader' or physical) and mem is not None:
        k,v=mem;length=k.shape[1]+1
        k,v=torch.cat((anchor[0],k),1),torch.cat((anchor[1],v),1)
        cache=C.make_cache(model,k,v,torch.arange(length,device='cuda'))
        tensor=torch.tensor([ids[1:]],device='cuda')
        kwargs=dict(past_key_values=cache,attention_mask=torch.ones(1,length+tensor.shape[1],device='cuda',dtype=torch.long),position_ids=torch.arange(length,length+tensor.shape[1],device='cuda')[None])
        if reader is not None:
            with reader.scope(length):return model(tensor,use_cache=False,**kwargs).logits[0,-1].float()
        return model(tensor,use_cache=False,**kwargs).logits[0,-1].float()
    tensor=torch.tensor([ids],device='cuda')
    if reader is None:return model(tensor,use_cache=False).logits[0,-1].float()
    if arm=='old_coupled_reader':
        with reader.scope(None):return model(tensor,use_cache=False).logits[0,-1].float()
    with reader.scope(mem):return model(tensor,use_cache=False).logits[0,-1].float()


@torch.no_grad()
def native_anchor(model,rr):
    token=rr[0]['context_ids'][0];assert all(r['context_ids'][0]==token for r in rr)
    k,v,_=C.capture(model,[token],1,[0]);return k.cuda(),v.cuda()


def objective(logits,row):return F.cross_entropy(logits[None],torch.tensor([row['label_ids'][row['gold']]],device='cuda'))
def result(logits,row):
    logs=logits.log_softmax(-1);scores=logs[torch.tensor(row['label_ids'],device='cuda')]
    return dict(C.record(scores.tolist(),row['gold']),gold_ce=float(objective(logits,row)))


@torch.no_grad()
def validate(model,reader,arm,anchor):
    reader.eval();total=correct=0.;rr=rows('validation')
    for r in rr:
        with torch.amp.autocast('cuda',dtype=torch.float16):out=context_forward(model,r,memory(r),reader,arm,anchor)
        value=result(out,r);total+=value['gold_ce'];correct+=value['correct']
    return dict(gold_ce=total/len(rr),correct=int(correct),n=len(rr),accuracy=correct/len(rr))


def checkpoint(path,reader,arm,epoch,step,position,completed,optimizer,scaler,history):
    C.save_tensor(path,dict(manifest_sha256=identity(),arm=arm,state=state(reader),epoch=epoch,step=step,position=position,epoch_completed=completed,
        optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),history=history))


def smoke():
    if (ROOT/'smoke/completed.json').exists():return
    tok,model=C.model('qwen');rr=rows('train')[:2];anchor=native_anchor(model,rr);checks={}
    frozen={n:p.detach().flatten()[:8].clone() for n,p in model.named_parameters()}
    for arm in CFG['arms']:
        reader=make_reader(model,arm);pp=trainable(reader);assert pp
        for r in rr:
            with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16):
                expected=context_forward(model,r)
                disabled=context_forward(model,r,None,reader,arm,anchor)
                assert torch.equal(expected,disabled),'Disabled Reader changed question-only baseline'
                initial=context_forward(model,r,memory(r),reader,arm,anchor)
                if arm!='old_coupled_reader':assert torch.equal(initial,expected),'Zero gate is not exactly question-only'
        optimizer=torch.optim.AdamW(pp.values(),lr=CFG['learning_rate'],weight_decay=0)
        norms=[];gate_grad=None
        for i in range(3):
            optimizer.zero_grad(set_to_none=True);r=rr[i%2]
            with torch.amp.autocast('cuda',dtype=torch.float16):out=context_forward(model,r,memory(r),reader,arm,anchor);loss=objective(out,r)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(pp.values(),CFG['gradient_clip'])
            assert torch.isfinite(norm) and norm>0 and torch.isfinite(loss),'Invalid Reader smoke gradient'
            if i==0 and arm!='old_coupled_reader':
                gate_grad=sum(float(p.grad.abs().sum()) for n,p in pp.items() if n.endswith('gate') and p.grad is not None)
                assert gate_grad>0,'Zero-gate initialization is dead'
            if i==2 and arm!='old_coupled_reader':assert sum(float(p.grad.abs().sum()) for n,p in pp.items() if 'query.' in n and p.grad is not None)>0,'Query projection has no gradient after gate opens'
            optimizer.step();norms.append(float(norm))
            assert all(p.grad is None for p in model.parameters()),'Frozen Qwen received parameter gradients'
        checks[arm]=dict(trainable_parameters=sum(p.numel() for p in pp.values()),initial_identity=True,gradient_norms=norms,first_gate_gradient=gate_grad,
            gate_values={n:float(p.detach()) for n,p in pp.items() if n.endswith('gate') and p.numel()==1})
        detach(model,reader,arm);del reader,optimizer,pp;torch.cuda.empty_cache()
    for n,p in model.named_parameters():assert torch.equal(p.detach().flatten()[:8],frozen[n]),('Frozen model mutated',n)
    C.save(ROOT/'smoke/completed.json',dict(status='completed',checks=checks,manifest_sha256=identity()))
    log(f'SMOKE PASSED: {checks}')


def train(arm):
    folder=ROOT/'runs'/arm
    if (folder/'completed.json').exists():return
    assert (ROOT/'smoke/completed.json').exists()
    C.seed();tok,model=C.model('qwen');rr=rows('train');anchor=native_anchor(model,rr)
    reader=make_reader(model,arm);pp=trainable(reader)
    optimizer=torch.optim.AdamW(pp.values(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=64.,growth_interval=1000000)
    start_epoch=1;step=position0=0;history=[]
    if (folder/'last.pt').exists():
        x=torch.load(folder/'last.pt',map_location='cpu',weights_only=True);assert x['manifest_sha256']==identity() and x['arm']==arm
        restore(reader,x['state']);optimizer.load_state_dict(x['optimizer']);scaler.load_state_dict(x['scaler']);step=x['step'];history=x['history']
        start_epoch=x['epoch']+int(x['epoch_completed']);position0=0 if x['epoch_completed'] else x['position'];del x
        log(f'RESUME {arm} epoch={start_epoch} step={step} position={position0}')
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'steps.jsonl').open('a') as stream:
        for epoch in range(start_epoch,CFG['epochs']+1):
            reader.train();order=list(range(len(rr)));random.Random(CFG['seed']+epoch).shuffle(order)
            optimizer.zero_grad(set_to_none=True);accum=loss_sum=0
            for position,index in enumerate(order,1):
                if epoch==start_epoch and position<=position0:continue
                r=rr[index]
                with torch.amp.autocast('cuda',dtype=torch.float16):out=context_forward(model,r,memory(r),reader,arm,anchor);loss=objective(out,r)
                assert torch.isfinite(loss),'Nonfinite gold CE'
                scaler.scale(loss/CFG['effective_batch']).backward();loss_sum+=float(loss.detach());accum+=1
                if accum==CFG['effective_batch']:
                    scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(pp.values(),CFG['gradient_clip'])
                    assert torch.isfinite(norm) and norm>0,'Invalid Reader gradient'
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);step+=1
                    assert all(p.grad is None for p in model.parameters()),'Frozen backbone gradient'
                    rec=dict(arm=arm,epoch=epoch,step=step,samples=position,total=len(rr),gold_ce=loss_sum/accum,gradient_norm=float(norm))
                    stream.write(json.dumps(rec)+'\n');stream.flush();status(status='running',phase='train',**rec)
                    if step%8==0:log(f'{arm} epoch={epoch}/{CFG["epochs"]} step={step}/512 samples={position}/{len(rr)} GoldCE={rec["gold_ce"]:.5f}')
                    loss_sum=accum=0
                    if step%CFG['checkpoint_every_steps']==0 and position<len(rr):checkpoint(folder/'last.pt',reader,arm,epoch,step,position,False,optimizer,scaler,history)
            metrics=validate(model,reader,arm,anchor);entry=dict(epoch=epoch,step=step,validation=metrics)
            history=[h for h in history if h['epoch']!=epoch]+[entry]
            best=min(history,key=lambda x:(x['validation']['gold_ce'],x['epoch']))
            if entry==best:C.save_tensor(folder/'best.pt',dict(state=state(reader),manifest_sha256=identity(),arm=arm,epoch=epoch,step=step,validation=metrics))
            C.save(folder/'selection.json',dict(rule=CFG['selection'],best=best,history=history))
            checkpoint(folder/'last.pt',reader,arm,epoch,step,len(rr),True,optimizer,scaler,history)
            log(f'{arm} EPOCH {epoch} COMPLETED: {metrics}')
    assert step==CFG['counts']['train']//CFG['effective_batch']*CFG['epochs']
    C.save(folder/'completed.json',dict(status='completed',steps=step,epochs=CFG['epochs'],trainable_parameters=sum(p.numel() for p in pp.values()),
        best_sha256=C.sha(folder/'best.pt'),manifest_sha256=identity()))
    detach(model,reader,arm);log(f'{arm} TRAINING COMPLETED')


def wrong_row(rr,index):
    for offset in range(1,len(rr)):
        other=rr[(index+offset)%len(rr)]
        if other['evidence_key']!=rr[index]['evidence_key']:return other
    raise AssertionError('No disjoint wrong evidence')


@torch.no_grad()
def evaluate(arm):
    folder=ROOT/'evaluation'/arm
    if (folder/'completed.json').exists():return
    tok,model=C.model('qwen');rr=rows('test');anchor=native_anchor(model,rr)
    reader=None
    if arm!='baselines':
        reader=make_reader(model,arm);x=torch.load(ROOT/f'runs/{arm}/best.pt',map_location='cpu',weights_only=True)
        assert x['manifest_sha256']==identity() and x['arm']==arm;restore(reader,x['state']);reader.eval()
    for i,r in enumerate(rr):
        p=folder/'per_sample'/(r['id']+'.json')
        if p.exists():assert json.loads(p.read_text())['manifest_sha256']==identity();continue
        status(status='running',phase='evaluate',arm=arm,sample=i+1,total=len(rr))
        wrong=wrong_row(rr,i);conditions={}
        with torch.amp.autocast('cuda',dtype=torch.float16):
            if arm=='baselines':
                conditions['question_only']=result(context_forward(model,r),r)
                conditions['native_full_evidence_text']=result(context_forward(model,r,full=True),r)
                for stage in ['a','b']:
                    conditions['frozen_stage_'+stage+'_physical_prefix']=result(context_forward(model,r,memory(r,stage),anchor=anchor,physical=True),r)
            else:
                for name,mem in [('correct_memory',memory(r)),('no_memory',None),('wrong_memory',memory(wrong))]:
                    conditions[name]=result(context_forward(model,r,mem,reader,arm,anchor),r)
        C.save(p,dict(id=r['id'],official_id=r['official_id'],question=r['question'],prompt=r['prompt'],fact1=r['evidence'],
            evidence_key=r['evidence_key'],candidates=r['candidates'],gold=r['gold'],wrong_memory_id=wrong['id'],wrong_fact1=wrong['evidence'],
            manifest_sha256=identity(),conditions=conditions,checkpoint_sha256=None if arm=='baselines' else C.sha(ROOT/f'runs/{arm}/best.pt')))
        if (i+1)%16==0:log(f'EVALUATE {arm} {i+1}/{len(rr)}')
    records=[json.loads((folder/'per_sample'/(r['id']+'.json')).read_text()) for r in rr]
    C.write_jsonl(folder/'per_sample.jsonl',records);C.save(folder/'completed.json',dict(status='completed',samples=len(rr),manifest_sha256=identity()))
    if reader is not None:detach(model,reader,arm)


def summarize():
    merged={r['id']:dict(id=r['id'],official_id=r['official_id'],question=r['question'],fact1=r['evidence'],evidence_key=r['evidence_key'],gold=r['gold'],conditions={}) for r in rows('test')}
    selections={}
    for arm in ['baselines']+CFG['arms']:
        assert (ROOT/f'evaluation/{arm}/completed.json').exists()
        for r in C.read_jsonl(ROOT/f'evaluation/{arm}/per_sample.jsonl'):
            merged[r['id']]['conditions'].update({arm+'/'+k:v for k,v in r['conditions'].items()})
        if arm!='baselines':selections[arm]=json.loads((ROOT/f'runs/{arm}/selection.json').read_text())['best']
    records=list(merged.values());keys=list(records[0]['conditions'])
    metrics={k:dict(n=len(records),correct=sum(r['conditions'][k]['correct'] for r in records),accuracy=sum(r['conditions'][k]['correct'] for r in records)/len(records),gold_ce=sum(r['conditions'][k]['gold_ce'] for r in records)/len(records)) for k in keys}
    def paired(a,b):
        aa=[r['conditions'][a]['correct'] for r in records];bb=[r['conditions'][b]['correct'] for r in records]
        return dict(both_correct=sum(x and y for x,y in zip(aa,bb)),first_only=sum(x and not y for x,y in zip(aa,bb)),second_only=sum(not x and y for x,y in zip(aa,bb)),both_wrong=sum(not x and not y for x,y in zip(aa,bb)))
    pairs={arm:{control:paired(arm+'/correct_memory',arm+'/'+control) for control in ['no_memory','wrong_memory']} for arm in CFG['arms']}
    for arm in CFG['arms']:pairs[arm]['vs_frozen_writer']=paired(arm+'/correct_memory','baselines/frozen_stage_b_physical_prefix')
    grouped=defaultdict(list)
    for r in records:grouped[r['evidence_key']].append(r)
    reusable={e:[r['id'] for r in rr] for e,rr in grouped.items() if len(rr)>1}
    summary=dict(status='completed',experiment=CFG['name'],metrics=metrics,paired=pairs,validation_selection=selections,
        same_evidence_multiple_question_groups=reusable,counts=CFG['counts'],steps_per_arm=512,
        caveats=['Gold fact1 is supplied, not retrieved. OpenBookQA can often be solved from prior knowledge; memory ablations do not establish causal dependence by themselves.',
        'Independent Reader has question-only autoregressive positions; coupled baseline shifts question positions after external memory. This is an intentional interface difference.',
        'Frozen Writer is evaluated out-of-distribution on fact-only input; no new Stage-A, Writer training, shared new Sender, compression or joint optimization in this pilot.',
        'Layer pattern preregistered, not claimed optimal; old coupled baseline has different trainable parameter count, recorded in smoke and completed files.'])
    C.write_jsonl(ROOT/'evaluation/per_sample.jsonl',records);C.save(ROOT/'summary.json',summary);shared('completed',summary)
    C.save(ROOT/'completed.json',dict(status='completed',manifest_sha256=identity()));status(status='completed',phase='all');log('ALL READER-ONLY EXPERIMENTS COMPLETED')


if __name__=='__main__':
    try:
        C.seed();action=sys.argv[1]
        if action=='prepare':prepare()
        elif action=='capture':capture()
        elif action=='translate':translate()
        elif action=='smoke':smoke()
        elif action=='train':train(sys.argv[2])
        elif action=='evaluate':evaluate(sys.argv[2])
        elif action=='summarize':summarize()
        else:raise ValueError(action)
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc));C.save(ROOT/'failure.json',error);status(status='failed',**error)
        if (ROOT/'manifest.json').exists():shared('failed_or_incomplete',error)
        raise
