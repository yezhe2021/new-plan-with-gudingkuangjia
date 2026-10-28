"""Same original128, one-factor-at-a-time prompt and cache boundary variants."""
import random,json,time
from common import ROOT,CFG,P,H,U,C,read,save,sha,verify


def make(row,prompt,tok):
    r=U.make_row(row['id'],'gsm8k',dict(question=row['question']),tok['llama'],tok['qwen'],
        prompt=prompt,candidates=[],gold=row['answer'],normalize=False,cue='')
    return r


def prepare():
    if (ROOT/'manifest.json').exists():verify();return
    P.verify();parent=Path(CFG['parent'])
    assert read(parent/'completed.json')['manifest_sha256']==P.identity()
    original=P.rows('test');assert len(original)==512
    selected=sorted(random.Random(CFG['seed']).sample(original,CFG['n']),key=lambda r:r['source_index'])
    excluded=P.rows('train')+P.rows('validation')
    assert not {r['source_id'] for r in selected}&{r['source_id'] for r in excluded}
    assert not {H.question_key(r['question']) for r in selected}&{H.question_key(r['question']) for r in excluded}
    source=Path(H.CFG['source_json']);raw={r['_id']:r for r in read(source)}
    prior={r['source_id']:r for r in C.read_jsonl(parent/'evaluation/per_sample.jsonl')}
    tok={f:C.AutoTokenizer.from_pretrained(C.CFG['models'][f],local_files_only=True) for f in ['llama','qwen']}
    output=[];max_tokens={c:0 for c in CFG['conditions']}
    for row in selected:
        doc=raw[row['source_id']];assert row['prompt']==H.prompt_for(doc)
        titles={fact[0] for fact in doc['supporting_facts']}
        supports=[(i,title,sentences) for i,(title,sentences) in enumerate(doc['context'],1) if title in titles]
        assert len(supports)==len(titles) and supports
        # Preserve original passage numbers when removing distractors.
        passages=['Passage '+str(i)+':\n'+title+'\n'+''.join(sentences).strip() for i,title,sentences in supports]
        support_prompt=('Answer the question using the passages below. Output only the short answer, without an explanation.\n\n'
            +'\n\n'.join(passages)+'\n\nQuestion: '+row['question']+'\nGive only the answer.')
        evidence='Reference information:\n'+'\n\n'.join(f'Document {j}: {title}\n'+''.join(sentences).strip()
            for j,(_,title,sentences) in enumerate(supports,1))+'\nEnd of reference'
        # Match the previous selector preparation's question.strip(); Llama's
        # chat template trims trailing whitespace, so untrimmed questions at
        # the very end otherwise fail the shared exact-content framing check.
        query='\n\nAnswer the question. Use the reference only when relevant. Give only the short answer, without explanation.\n\nQuestion: '+row['question'].strip()
        b=make(row,support_prompt,tok);c=make(row,evidence+query,tok);d=make(row,evidence,tok)
        assert not d['native_tail']
        f=d['frames']['qwen'];body=f['body_ids']
        rendered=f['rendered_prefix']+evidence+query+f['rendered_suffix']
        ids=tok['qwen'].encode(rendered,add_special_tokens=False)
        assert ids[:len(body)]==body and ids==c['native_context_ids']
        d['continuation_ids']=ids[len(body):]
        variants=dict(A_original=row,B_support_only=b,C_selector_prompt=c,D_evidence_only=d)
        for name,r in variants.items():
            total=len(r['frames']['qwen']['body_ids'])+len(r.get('continuation_ids',r['suffix_ids']))
            assert total+CFG['max_new_tokens']<CFG['max_context_tokens']
            assert len(r['frames']['llama']['body_ids'])+CFG['max_new_tokens']<CFG['max_context_tokens']
            max_tokens[name]=max(max_tokens[name],total)
        assert prior[row['source_id']]['question']==row['question'] and prior[row['source_id']]['gold']==row['answer']
        output.append(dict(id=row['id'],source_id=row['source_id'],question=row['question'],answer=row['answer'],
            question_type=doc['type'],original_paragraphs=len(doc['context']),support_paragraphs=len(supports),
            variants=variants,historical=prior[row['source_id']]['conditions']))
    max_tokens['E_diagnostic_decoder']=max_tokens['D_evidence_only']
    (ROOT/'prepared').mkdir(exist_ok=True)
    path=ROOT/'prepared/questions.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
    cps={}
    for stage in ['stage_a','stage_b']:
        done=read(parent/'runs'/stage/'completed.json');assert done['manifest_sha256']==P.identity()
        cps[stage]=dict(path=done['selected'],sha256=done['selected_sha256']);assert sha(done['selected'])==done['selected_sha256']
    save('manifest.json',dict(config_sha256=sha(ROOT/'config.json'),parent_manifest_sha256=P.identity(),
        prepared_sha256=sha(path),source_sha256=sha(source),checkpoints=cps,source_ids=[r['source_id'] for r in output],
        code={p.name:sha(p) for p in ROOT.glob('*.py')},train_validation_overlap=0,no_reader=True,selection_uses_predictions=False,
        native_C_D_input_tokens_identical=True,n=CFG['n'],max_prompt_tokens=max_tokens,frozen_at=time.strftime('%F %T')))
    print('PREPARED',len(output),'no length/type/outcome filters',max_tokens,flush=True)


if __name__=='__main__':
    from pathlib import Path
    prepare()
