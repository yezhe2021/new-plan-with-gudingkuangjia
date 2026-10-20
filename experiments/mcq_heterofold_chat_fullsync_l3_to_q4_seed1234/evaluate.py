"""Same immutable ARC128/HellaSwag64; only retrained MCQ Writer is newly scored."""
import itertools,sys
from core import *


def test_rows():
    pp=Path(CFG['previous_root'])
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert sha(pp/'comparison/per_sample.jsonl')==manifest['previous_comparison_sha256']
    rr=json.loads((pp/'previous_prepared.json').read_text())
    assert len(rr)==192
    return rr


def test_path(row,kind):return ROOT/f'evaluation/{kind}/{row["id"]}.pt'


def prepare_eval():
    assert (ROOT/'runs/stage_b/completed.json').exists()
    value=dict(manifest_sha256=identity(),stage_a_sha256=sha(ROOT/'runs/stage_a/best.pt'),
        stage_b_sha256=sha(ROOT/'runs/stage_b/best.pt'),prepared_sha256=digest(test_rows()))
    path=ROOT/'evaluation/manifest.json'
    if path.exists():assert json.loads(path.read_text())==value
    else:save(path,value)


@torch.no_grad()
def capture_test():
    prepare_eval()
    rr=[r for r in test_rows() if not test_path(r,'source').exists()]
    if not rr:return
    tok,m=model('llama')
    for i,row in enumerate(rr,1):
        ids=row['frames']['llama']['body_ids'];k,v,_=capture(m,ids,len(ids),[0])
        assert list(tok.encode(row['frames']['llama']['rendered_full'],add_special_tokens=False))==row['frames']['llama']['full_ids']
        ix=row['source_indices']
        save_tensor(test_path(row,'source'),dict(key=k[:,ix].contiguous(),value=v[:,ix].contiguous(),manifest_sha256=identity()))
        if i%16==0 or i==len(rr):log(f'Evaluation Sender capture {i}/{len(rr)}')


@torch.no_grad()
def translate_test():
    prepare_eval();rr=[r for r in test_rows() if not test_path(r,'translated').exists()]
    if not rr:return
    guard('stage_a');base,adapter=base_from_best(),adapter_from_best()
    for i,row in enumerate(rr,1):
        source=torch.load(test_path(row,'source'),map_location='cpu',weights_only=True)
        assert source['manifest_sha256']==identity()
        k,v=map_base(base,source);k,v=adapted(adapter,k,v)
        assert torch.isfinite(k).all() and torch.isfinite(v).all()
        save_tensor(test_path(row,'translated'),dict(key=k.cpu().contiguous(),value=v.cpu().contiguous(),manifest_sha256=identity()))
        if i%16==0 or i==len(rr):log(f'Evaluation new MCQ Writer {i}/{len(rr)}')


@torch.no_grad()
def score_test():
    prepare_eval()
    if (ROOT/'evaluation/completed.json').exists():return
    old={r['id']:r for r in read_jsonl(Path(CFG['previous_root'])/'comparison/per_sample.jsonl')}
    tok,m=model('qwen');rt=runtime(m,tok);results=[]
    for i,row in enumerate(test_rows(),1):
        path=ROOT/f'evaluation/per_sample/{row["id"]}.json'
        if path.exists():
            result=json.loads(path.read_text());assert result['manifest_sha256']==identity()
            results.append(result);continue
        previous=old[row['id']]
        for key in ['id','row_sha256','prompt','candidates','gold','normalize','cue']:assert row[key]==previous[key]
        frame=row['frames']['qwen'];ids=frame['body_ids']
        k,v,_=capture(m,ids,len(ids),[0]);n=frame['native_prefix_length']
        mapped=torch.load(test_path(row,'translated'),map_location='cpu',weights_only=True)
        assert mapped['manifest_sha256']==identity()
        k=torch.cat((k[:,:n].cuda(),mapped['key'].cuda()),1);v=torch.cat((v[:,:n].cuda(),mapped['value'].cuda()),1)
        assert k.shape[1]==len(ids)
        cache=make_cache(m,k,v,torch.arange(len(ids),device='cuda'))
        scores,_=R.transfer_candidate_scores(rt,cache,len(ids),row['candidates'],row['normalize'],row['receiver_suffix'])
        conditions={name:dict(value,reused=True) for name,value in previous['conditions'].items()}
        conditions['mcq_retrained']=dict(record(scores,row['gold']),reused=False)
        result=dict(row,conditions=conditions,manifest_sha256=identity())
        save(path,result);results.append(result)
        if i%8==0 or i==192:
            save(ROOT/'evaluation/partial_summary.json',summarize(results))
            status(status='running',phase='evaluation',sample=i,total=192)
            log(f'Evaluation scored {i}/192')
    assert len(results)==192
    write_jsonl(ROOT/'evaluation/per_sample.jsonl',results)
    summary=summarize(results)
    summary.update(status='completed',training_manifest_sha256=identity(),
        selections={stage:json.loads((ROOT/f'runs/stage_{stage}/selection.json').read_text()) for stage in ['a','b']},
        scope='New MCQ Writer trained and evaluated under fixed HeteroFold chat/task protocol; no GSM8K training',
        caveats=['Previous methods reused on same rows; only mcq_retrained is new',
                 'HeteroFold training data and Sender interface differ; not a same-training-budget architecture comparison'])
    save(ROOT/'evaluation/summary.json',summary)
    shared('completed_retraining_and_evaluation',summary)
    save(ROOT/'evaluation/completed.json',dict(status='completed',rows=192,manifest_sha256=identity()))
    status(status='completed',phase='all')
    log('ALL MCQ RETRAINING AND EVALUATION COMPLETED')


def summarize(rr):
    output={}
    for task in ['arc','hellaswag']:
        subset=[r for r in rr if r['task']==task]
        if not subset:continue
        names=sorted(subset[0]['conditions']);metrics={}
        for name in names:
            correct=sum(r['conditions'][name]['correct'] for r in subset)
            metrics[name]=dict(n=len(subset),correct=correct,accuracy=correct/len(subset))
        pairs={}
        for a,b in itertools.combinations(names,2):
            x=[bool(r['conditions'][a]['correct']) for r in subset];y=[bool(r['conditions'][b]['correct']) for r in subset]
            pairs[a+'__vs__'+b]=dict(both_correct=sum(i and j for i,j in zip(x,y)),only_a=sum(i and not j for i,j in zip(x,y)),only_b=sum(not i and j for i,j in zip(x,y)),both_wrong=sum(not i and not j for i,j in zip(x,y)))
        output[task]=dict(metrics=metrics,paired=pairs)
    return dict(experiment=CFG['name'],tasks=output)


if __name__=='__main__':
    try:{'capture':capture_test,'translate':translate_test,'score':score_test}[sys.argv[1]]()
    except Exception as exc:fail(exc);raise
