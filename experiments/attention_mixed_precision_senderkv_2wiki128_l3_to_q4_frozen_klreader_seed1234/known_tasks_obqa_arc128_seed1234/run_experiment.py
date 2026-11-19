"""Quantization on fixed, previously adapted held-out MCQ tasks; no training."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import copy,hashlib,json,math,sys,time,random,inspect
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import experiment as E
from quantization import decode
from metrics import ranked_positions,bootstrap
sys.path.insert(0,str(ROOT))
from codec import FixedCodec
CFG=json.loads((ROOT/'config.json').read_text())
ARMS=E.ARMS


def read(path):return json.loads(Path(path).read_text())
def save(path,x):E.save(path,x)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity():return sha(ROOT/'manifest.json')
def status(**kw):save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**kw))
def rows():return [json.loads(x) for x in (ROOT/'prepared/questions.jsonl').read_text().splitlines()]
def fname(row):return E.U.fname(row)
def source_path(row):return ROOT/'source_cache'/(fname(row)+'.pt')
def result_path(family,row):return ROOT/'evaluation'/family/(fname(row)+'.json')


def prepare():
    E.verify(); tokenizers={f:E.tok(f) for f in ['llama','qwen']}
    original=[r for r in E.U.rows('test') if r['task'] in CFG['tasks']]
    from collections import Counter
    assert dict(Counter(r['task'] for r in original))==CFG['tasks']
    records=[];historical={}
    for original_row in original:
        row=copy.deepcopy(original_row);row['layouts']={}
        for family,tokenizer in tokenizers.items():
            frame=row['frames'][family]
            full=frame['rendered_full']+row['cue']
            suffix=tokenizer.encode(frame['rendered_suffix']+row['cue'],add_special_tokens=False)
            assert frame['body_ids']+suffix==tokenizer.encode(full,add_special_tokens=False)
            labels=[E.C.R.candidate_ids(tokenizer,c) for c in row['candidates']]
            assert all(len(x)==1 for x in labels)
            label_ids=[x[0] for x in labels];assert len(set(label_ids))==len(labels)
            row['layouts'][family]=dict(body_ids=frame['body_ids'],suffix_ids=suffix,label_ids=label_ids)
        assert row['suffix_ids']==row['layouts']['qwen']['suffix_ids']
        positions=sorted(set(row['source_indices']));mapping={p:i for i,p in enumerate(positions)}
        row['wire_positions']=positions;row['target_map']=[mapping[x] for x in row['source_indices']]
        row['protected_tokens']=math.ceil(CFG['protected_fraction']*len(positions))
        file=Path(E.CFG['reference_reader'])/'evaluation'/row['task']/'per_sample'/(fname(row)+'.json')
        prior=read(file);assert prior['reader_manifest_sha256']==E.R.identity()
        assert all(prior[k]==row[k] for k in ['id','task','prompt','gold'])
        row['historical_fp16']=prior['conditions']['grouped_residual_reader']
        historical[str(file)]=sha(file);records.append(row)
    out=ROOT/'prepared/questions.jsonl';out.parent.mkdir(exist_ok=True)
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records)
    if out.exists():assert out.read_text()==data
    else:out.write_text(data)
    manifest=dict(config=CFG,config_sha256=sha(ROOT/'config.json'),prepared_sha256=sha(out),
        code={p.name:sha(p) for p in ROOT.glob('*.py')},parent_manifest_sha256=E.identity(),
        historical_sources=historical,source_split_sha256=sha(E.U.ROOT/'prepared/test.jsonl'),
        checkpoints=read(E.ROOT/'manifest.json')['checkpoints'],historical_test_subsets_previously_inspected=True)
    if(ROOT/'manifest.json').exists():assert read(ROOT/'manifest.json')==manifest
    else:save(ROOT/'manifest.json',manifest)
    save(ROOT/'prepared/completed.json',dict(status='completed',manifest_sha256=identity(),counts=CFG['tasks']))
    status(status='prepared',phase='data',counts=CFG['tasks']);E.log('KNOWN TASKS PREPARED')


def verify():
    E.verify();m=read(ROOT/'manifest.json')
    assert E.identity()==m['parent_manifest_sha256']
    assert sha(ROOT/'config.json')==m['config_sha256'] and sha(ROOT/'prepared/questions.jsonl')==m['prepared_sha256']
    for name,digest in m['code'].items():assert sha(ROOT/name)==digest,name


@torch.inference_mode()
def scores(model,row,layout,cache,capture=False):
    attention=[];handles=[];b=len(layout['body_ids']);suffix=layout['suffix_ids']
    if capture:
        def hook(module,args,out):
            assert out[1] is not None
            attention.append(out[1][0,:,-1,:b].float().mean(0).cpu().numpy())
        handles=[layer.self_attn.register_forward_hook(hook) for layer in model.model.layers]
    inputs=torch.tensor([suffix],device='cuda')
    try:
        out=model.model(inputs,past_key_values=copy.deepcopy(cache),attention_mask=torch.ones(1,b+len(suffix),device='cuda',dtype=torch.long),
            position_ids=torch.arange(b,b+len(suffix),device='cuda')[None],use_cache=False)
    finally:
        for h in handles:h.remove()
    logits=model.lm_head(out.last_hidden_state[:,-1])[0].float()
    label_scores=F.log_softmax(logits,-1)[layout['label_ids']]
    choice_lp=F.log_softmax(label_scores,-1);prediction=int(label_scores.argmax())
    result=dict(scores=label_scores.cpu().tolist(),prediction=prediction,correct=int(prediction==row['gold']),
        choice_nll=float(-choice_lp[row['gold']]),full_vocab_nll=float(-label_scores[row['gold']]))
    assert torch.isfinite(logits).all()
    return result,logits,(np.mean(attention,axis=0) if capture else None)


@torch.inference_mode()
def baseline():
    verify();tokenizer,model=E.load_model('qwen');base,reader=E.load_frontend(model);records=[]
    for i,row in enumerate(rows(),1):
        status(status='running',phase='baseline_replay',sample=i,total=256,task=row['task'])
        native,k,v=E.R.fixed_state(model,base,reader,row)
        cache=E.U.translated_cache(model,row,native,k,v)
        with reader.mode(True):
            result,_,_=scores(model,row,row['layouts']['qwen'],cache)
            if i==1:
                canonical=E.U.candidate_scores(model,tokenizer,row,native,k,v)
                assert canonical['prediction']==result['prediction']
                assert max(abs(a-b) for a,b in zip(canonical['scores'],result['scores']))<.03
        prior=row['historical_fp16'];result['historical_score_max_error']=max(abs(a-b) for a,b in zip(prior['scores'],result['scores']))
        result['historical_prediction_match']=prior['prediction']==result['prediction']
        save(ROOT/'baseline'/row['task']/(fname(row)+'.json'),dict(id=row['id'],manifest_sha256=identity(),**result))
        records.append((row,result));del native,k,v,cache
        if i%16==0:E.log(f'BASELINE REPLAY {i}/256')
    task_results={}
    for task,n in CFG['tasks'].items():
        values=[v for r,v in records if r['task']==task]
        task_results[task]=dict(n=n,correct=sum(v['correct'] for v in values),accuracy=sum(v['correct'] for v in values)/n,
            historical_prediction_matches=sum(v['historical_prediction_match'] for v in values),
            max_historical_score_error=max(v['historical_score_max_error'] for v in values))
        assert task_results[task]['accuracy']>=.5,('FP16 task baseline failed; do not run quantization',task,task_results[task])
    save(ROOT/'baseline/completed.json',dict(status='completed',manifest_sha256=identity(),tasks=task_results))
    E.log('FP16 BASELINES ACCEPTED '+json.dumps(task_results))


def selections(row,attention):
    positions=row['wire_positions'];s=attention[positions];eligible=np.arange(len(positions));count=row['protected_tokens']
    seed=int(E.C.digest([CFG['seed'],row['id']])[:12],16)
    chosen=dict(fp16=[],int4=[],attention_top=ranked_positions(s,eligible,count,seed=seed),
        attention_bottom=ranked_positions(s,eligible,count,reverse=True,seed=seed))
    for i,rseed in enumerate(CFG['random_seeds']):chosen[f'random_{i}']=sorted(np.random.default_rng(seed+rseed).choice(eligible,count,replace=False).tolist())
    return chosen,s.tolist()


def packet(row,factory,arm,selected):
    bits=16 if arm=='fp16' else 4 if arm=='int4' else 'mixed'
    data=factory.encode(selected[arm],bits);k,v,meta=decode(data)
    assert meta['source_positions']==row['wire_positions'] and meta['target_map']==row['target_map'] and meta['high_positions']==selected[arm]
    return k,v,dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),effective_bits=8*len(data)/(factory.key.numel()+factory.value.numel()),
        key_nmse=float((k.float()-factory.key.float()).square().sum()/factory.key.float().square().sum().clamp_min(1e-30)),
        value_nmse=float((v.float()-factory.value.float()).square().sum()/factory.value.float().square().sum().clamp_min(1e-30)))


def delta(result,baseline,logits,base_logits):
    result.update(delta_choice_nll=result['choice_nll']-baseline['choice_nll'],
        delta_full_vocab_nll=result['full_vocab_nll']-baseline['full_vocab_nll'],
        full_vocab_kl=float(F.kl_div(F.log_softmax(logits,-1),F.softmax(base_logits,-1),reduction='sum')))
    return result


@torch.inference_mode()
def sender():
    verify();assert read(ROOT/'baseline/completed.json')['manifest_sha256']==identity()
    tokenizer,model=E.load_model('llama')
    for i,row in enumerate(rows(),1):
        assert not result_path('llama',row).exists(),'Existing result requires explicit reviewed resume'
        fullk,fullv,original=E.capture_body(model,row['layouts']['llama']['body_ids'])
        direct,direct_logits,attention=scores(model,row,row['layouts']['llama'],original,capture=True)
        selected,ranking=selections(row,attention);positions=row['wire_positions']
        key,value=fullk[:,positions].contiguous(),fullv[:,positions].contiguous();del fullk,fullv
        factory=FixedCodec(key,value,CFG['group_size'],positions,row['target_map']);results={};base_result=base_logits=None
        for arm in ARMS:
            status(status='running',phase='sender',sample=i,total=256,task=row['task'],arm=arm)
            k,v,wire=packet(row,factory,arm,selected);cache=E.native_cache(model,original,k,v,positions)
            result,logits,_=scores(model,row,row['layouts']['llama'],cache)
            if arm=='fp16':
                base_result,base_logits=result,logits.clone()
                assert result['prediction']==direct['prediction'] and abs(result['full_vocab_nll']-direct['full_vocab_nll'])<.03
            results[arm]=dict(delta(result,base_result,logits,base_logits),wire=wire)
        assert len({results[a]['wire']['bytes'] for a in ARMS[2:]})==1
        source_path(row).parent.mkdir(exist_ok=True);torch.save(dict(key=key,value=value,manifest_sha256=identity()),source_path(row))
        save(result_path('llama',row),dict(id=row['id'],task=row['task'],gold=row['gold'],manifest_sha256=identity(),arms=results,selections=selected,ranking=ranking))
        del original,cache,key,value,k,v,factory
        if i%8==0:E.log(f'LLAMA QUANT {i}/256')
    save(ROOT/'evaluation/llama/completed.json',dict(status='completed',manifest_sha256=identity(),n=256))


@torch.inference_mode()
def receiver():
    verify();tokenizer,model=E.load_model('qwen');base,reader=E.load_frontend(model)
    for i,row in enumerate(rows(),1):
        assert not result_path('qwen',row).exists(),'Existing result requires explicit reviewed resume'
        prior=read(result_path('llama',row));source=torch.load(source_path(row),map_location='cpu',weights_only=True)
        assert source['manifest_sha256']==prior['manifest_sha256']==identity()
        factory=FixedCodec(source['key'],source['value'],CFG['group_size'],row['wire_positions'],row['target_map'])
        with reader.mode(False):native=E.U.native_body(model,row)
        results={};base_result=base_logits=None
        for arm in ARMS:
            status(status='running',phase='receiver',sample=i,total=256,task=row['task'],arm=arm)
            k,v,wire=packet(row,factory,arm,prior['selections']);assert wire['sha256']==prior['arms'][arm]['wire']['sha256']
            with reader.mode(False):cache,mappedk,mappedv=E.translated(model,base,reader,row,native,k,v)
            with reader.mode(True):result,logits,_=scores(model,row,row['layouts']['qwen'],cache)
            if arm=='fp16':
                baseline=read(ROOT/'baseline'/row['task']/(fname(row)+'.json'))
                discrepancy=max(abs(a-b) for a,b in zip(result['scores'],baseline['scores']))
                assert discrepancy<.1,('New Sender capture differs from verified original FP16 baseline',row['id'],discrepancy)
                result['baseline_score_max_error']=discrepancy;result['baseline_prediction_match']=result['prediction']==baseline['prediction']
                base_result,base_logits=result,logits.clone()
            results[arm]=dict(delta(result,base_result,logits,base_logits),wire=wire)
        assert len({results[a]['wire']['bytes'] for a in ARMS[2:]})==1
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        save(result_path('qwen',row),dict(id=row['id'],task=row['task'],gold=row['gold'],manifest_sha256=identity(),arms=results))
        source_path(row).unlink();del factory,source,native,cache,k,v,mappedk,mappedv
        if i%8==0:E.log(f'QWEN QUANT {i}/256')
    save(ROOT/'evaluation/qwen/completed.json',dict(status='completed',manifest_sha256=identity(),n=256))


def summarize():
    verify();summary=dict(experiment=CFG['name'],timestamp=time.strftime('%F %T'),status='completed',manifest_sha256=identity(),config=CFG,
        checkpoints=read(ROOT/'manifest.json')['checkpoints'],baseline=read(ROOT/'baseline/completed.json'),tasks={},
        uncertainty='Paired question bootstrap; average random repetitions within each question; exploratory uncorrected95% intervals.')
    for task,n in CFG['tasks'].items():
        summary['tasks'][task]={}
        for family in ['llama','qwen']:
            assert read(ROOT/f'evaluation/{family}/completed.json')['manifest_sha256']==identity()
            records=[read(result_path(family,r)) for r in rows() if r['task']==task];arms={}
            for arm in ARMS+['random_mean']:
                names=['random_0','random_1','random_2'] if arm=='random_mean' else [arm]
                def values(k):return [float(np.mean([r['arms'][a][k] for a in names])) for r in records]
                arms[arm]=dict(n=n,correct=sum(values('correct')),accuracy=np.mean(values('correct')),
                    delta_choice_nll=bootstrap(values('delta_choice_nll'),CFG['seed']),
                    delta_full_vocab_nll=bootstrap(values('delta_full_vocab_nll'),CFG['seed']),
                    mean_wire_bytes=float(np.mean([r['arms'][names[0]]['wire']['bytes'] for r in records])),
                    mean_effective_bits=float(np.mean([r['arms'][names[0]]['wire']['effective_bits'] for r in records])))
            contrasts={}
            for control in ['random_mean','attention_bottom']:
                differences=[]
                for r in records:
                    other=np.mean([r['arms'][a]['delta_choice_nll'] for a in ['random_0','random_1','random_2']]) if control=='random_mean' else r['arms'][control]['delta_choice_nll']
                    differences.append(r['arms']['attention_top']['delta_choice_nll']-other)
                contrasts['top_minus_'+control]=bootstrap(differences,CFG['seed'])
            summary['tasks'][task][family]=dict(arms=arms,paired_choice_nll_contrasts=contrasts)
    save(ROOT/'summary.json',summary)
    import fcntl
    with (ROOT.parent.parent/'EXPERIMENT_RESULTS.md').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.write('\n\n## '+E.CFG['name']+' / '+CFG['name']+'\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n');stream.flush();os.fsync(stream.fileno())
    save(ROOT/'completed.json',dict(status='completed',manifest_sha256=identity()));status(status='completed',phase='all');E.log('KNOWN TASK QUANTIZATION COMPLETE')


if __name__=='__main__':
    try:{'prepare':prepare,'baseline':baseline,'sender':sender,'receiver':receiver,'summarize':summarize}[sys.argv[1]]()
    except Exception as exc:
        import traceback
        save(ROOT/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        status(status='failed',phase=sys.argv[1],error=str(exc));raise
