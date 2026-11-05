"""Pin unchanged Llama cohorts and original calibration; validate Gemma RAW alignment on CPU."""
import shutil
from collections import Counter
from experiment_common import *
from runtime_new import batches
from heterofold.alignment import exact_end_pairs

def main():
    if (ROOT/'manifest.json').exists():verify();return
    N.verify();assert N.identity()==CFG['data_reference_manifest_sha256']
    data={s:N.rows(s) for s in ['train','validation','test']};cal={s:C.rows(s) for s in ['train','validation']}
    assert Counter((s,r['dataset']) for s,rr in cal.items() for r in rr)==Counter({('train','Open-R1 prompt-only'):800,('validation','Open-R1 prompt-only'):200,('train','HotpotQA distractor train full-context'):800,('validation','HotpotQA distractor train full-context'):200})
    for s,rr in data.items():
        assert {t:sum(r['task']==t for r in rr) for t in CFG[s+'_counts']}==CFG[s+'_counts']
        path=ROOT/f'prepared/{s}.jsonl';path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REFERENCE/f'prepared/{s}.jsonl',path)
    for s in cal:shutil.copyfile(C.ROOT/f'prepared/{s}.jsonl',ROOT/f'prepared/calibration_{s}.jsonl')
    cal_questions={C.norm(r['question']) for rr in cal.values() for r in rr}
    questions={s:{C.norm(r['question']) for r in rr} for s,rr in data.items()}
    assert all(not q&cal_questions for q in questions.values()),'Downstream/calibration overlap'
    assert not questions['train']&questions['validation'] and not questions['train']&questions['test'] and not questions['validation']&questions['test']
    st=C.AutoTokenizer.from_pretrained(CFG['sender'],local_files_only=True);rt=C.AutoTokenizer.from_pretrained(CFG['receiver'],local_files_only=True)
    alignment={};minimum=10**9
    for s,rr in cal.items():
        counts=[]
        for row in rr:
            off=st(row['prompt'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
            enc=rt(row['prompt'],add_special_tokens=False,return_offsets_mapping=True);assert enc['input_ids']==row['content_ids']
            n=len(exact_end_pairs(off,enc['offset_mapping']));assert n>=8,(row['id'],n)
            counts.append(n);minimum=min(minimum,n)
        alignment[s]=dict(min_pairs=min(counts),total_pairs=sum(counts),n=len(counts))
    refs=[REFERENCE/'manifest.json',C.ROOT/'manifest.json',Path(CFG['calibration'])]
    refs+=list(REFERENCE.glob('*.py'))+[ROOT/f'prepared/{n}.jsonl' for n in ['train','validation','test','calibration_train','calibration_validation']]
    refs+=[Path(CFG[m])/n for m in ['sender','receiver'] for n in ['config.json','tokenizer_config.json','tokenizer.json'] if (Path(CFG[m])/n).exists()]
    C.save(ROOT/'manifest.json',dict(config=CFG,config_sha256=C.sha(ROOT/'config.json'),code={f.name:C.sha(f) for f in ROOT.glob('*.py')},sources={str(f):C.sha(f) for f in refs},
        prefix_ids=C.read(C.ROOT/'manifest.json')['prefix_ids'],calibration_alignment=alignment,min_aligned_pairs=minimum,
        counts={s:len(rr) for s,rr in data.items()},calibration_counts={s:len(rr) for s,rr in cal.items()},
        canonical_split_byte_identical=True,question_overlap=0,memory_trainable_parameters=409600,c_trainable_parameters=9437184,
        batch_schedules={str(e):C.digest([[(r['id'],k) for r,k in b] for b in batches(e)]) for e in range(1,5)}))
    mark('prepared/completed.json',status='completed',counts={s:len(rr) for s,rr in data.items()},calibration_alignment=alignment)
    assert not torch.cuda.is_initialized();C.log('GEMMA RECOLOR REDO PREPARED',alignment)
if __name__=='__main__':main()
