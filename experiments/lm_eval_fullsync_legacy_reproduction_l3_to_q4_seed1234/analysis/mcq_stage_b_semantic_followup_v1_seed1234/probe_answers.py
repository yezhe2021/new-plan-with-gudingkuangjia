"""Held-out answer-token diagnosis; no benchmark inputs or checkpoint selection."""
from pathlib import Path
import importlib.util
import json
import sys
import time

import torch

HERE=Path(__file__).resolve().parent
PARENT=HERE.parents[1]
TRAJ=PARENT/'analysis/mcq_stage_b_trajectorykd_v1_seed1234'
sys.path.insert(0,str(TRAJ))
import trajectory_common as tc
from kv_backend import assemble,encoding
from trajectory_data import ANSWER_RE

spec=importlib.util.spec_from_file_location('frozen_trajectory_train',TRAJ/'train.py')
train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)


def summary(rows):
    result={}
    for view in ('direct_answer_cue','reference_first_answer'):
        selected=[r for r in rows if r['view']==view]
        result[view]={}
        for arm in ('native','scheme3','trajectory'):
            values=[r[arm] for r in selected]
            result[view][arm]={'n':len(values),'correct':sum(v['prediction']==r['gold'] for v,r in zip(values,selected)),
                              'mean_gold_probability':sum(v['gold_probability'] for v in values)/max(1,len(values)),
                              'mean_gold_ce':sum(v['gold_ce'] for v in values)/max(1,len(values))}
    return result


@torch.no_grad()
def main():
    tc.verify_manifest();tc.gpu_check()
    receiver,sender,base,scheme3,source_tok,hf=tc.model_bundle(False)
    trajectory=tc.load_new_adapter(TRAJ/'stage_b/best.pt')
    refs={r['base_id']:r for r in tc.read_rows(TRAJ/'prepared/validation_trajectories.jsonl')}
    attempts=tc.read_rows(TRAJ/'prepared/validation_attempts.jsonl')
    anchors={};results=[]
    for index,q in enumerate(attempts,1):
        context=q['context'];ctx=hf.tok_encode(context)
        item,bk,bv=train.prefix(sender,receiver,base,source_tok,hf,{'context':context,'context_ids':ctx},anchors)
        caches={arm:assemble(item,bk,bv,a) for arm,a in (('scheme3',scheme3),('trajectory',trajectory))}
        probes=[('direct_answer_cue',encoding(hf.tokenizer,' The answer is ('),[encoding(hf.tokenizer,c)[0] for c in 'ABCDEFGHIJ'[:len(q['choices'])]])]
        if q['base_id'] in refs:
            ref=refs[q['base_id']];ids=ref['continuation_ids']
            text=hf.tokenizer.decode(ids,skip_special_tokens=True)
            match=ANSWER_RE.search(text);start,end=match.span(1)
            ends=[len(hf.tokenizer.decode(ids[:i+1],skip_special_tokens=True)) for i in range(len(ids))]
            starts=[0]+ends[:-1]
            positions=[i for i,(a,b) in enumerate(zip(starts,ends)) if b>start and a<end]
            assert len(positions)==1,'Answer letter spans multiple tokens; implement sequence scoring before proceeding'
            pos=positions[0];token_text=hf.tokenizer.decode([ids[pos]],skip_special_tokens=True)
            assert token_text.strip()==q['gold'],repr(token_text)
            candidates=[encoding(hf.tokenizer,token_text.replace(q['gold'],c)) for c in 'ABCDEFGHIJ'[:len(q['choices'])]]
            assert all(len(x)==1 for x in candidates)
            assert candidates['ABCDEFGHIJ'.index(q['gold'])][0]==ids[pos]
            probes.append(('reference_first_answer',ids[:pos],[x[0] for x in candidates]))
        for view,continuation,candidates in probes:
            # Add a dummy target only to retrieve the distribution AFTER prefix.
            target_prefix=continuation+[candidates[0]]
            entry={'base_id':q['base_id'],'task':q['task'],'view':view,'gold':q['gold'],'candidate_token_ids':candidates}
            for arm in ('native','scheme3','trajectory'):
                cache=(None,None) if arm=='native' else caches[arm]
                hidden=train.suffix_hidden(receiver,ctx,target_prefix,*cache)[-1]
                lp=receiver.lm_head(hidden).float().log_softmax(-1)
                chosen=lp[torch.tensor(candidates,device='cuda')]
                gold='ABCDEFGHIJ'.index(q['gold']);pred=int(chosen.argmax())
                entry[arm]={'prediction':'ABCDEFGHIJ'[pred],'gold_probability':float(lp[candidates[gold]].exp()),
                            'gold_ce':float(-lp[candidates[gold]]),'candidate_probabilities':chosen.exp().cpu().tolist(),
                            'full_vocab_top_token_id':int(lp.argmax())}
            results.append(entry)
        tc.write_rows(HERE/'diagnostic/answer_tokens.jsonl',results)
        tc.log(f'ANSWER DIAGNOSIS {index}/{len(attempts)}')
        del caches,item,bk,bv
    report={'VALIDATION_DIAGNOSIS_NOT_BENCHMARK':True,'n_fixed_validation_questions':len(attempts),
            'checkpoint_sha256':tc.digest(TRAJ/'stage_b/best.pt'),
            'reference_note':'First matching answer letter only; no earlier official answer phrase is supplied. Correct reference reasoning can still reveal the answer semantically.',
            'comparisons':summary(results)}
    tc.save(HERE/'diagnostic/answer_token_summary.json',report)
    tc.log('ANSWER DIAGNOSIS COMPLETED '+json.dumps(report))


if __name__=='__main__':main()
