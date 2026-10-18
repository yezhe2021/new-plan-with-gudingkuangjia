"""Training-validation mechanism diagnostic; separate from official test metrics."""
from collections import Counter
import json
import time

from trajectory_common import (HERE,CONFIG,model_bundle,verify_manifest,gpu_check,load_new_adapter,
                               FullSyncHFLM,save,read_rows,write_rows,digest,log)
from trajectory_data import prompt,ANSWER_RE
from prepare import generate


def main():
    verify_manifest(); gpu_check()
    receiver,sender,base,initial,_,_=model_bundle(False)
    new=load_new_adapter(HERE/'stage_b/best.pt')
    attempts=read_rows(HERE/'prepared/validation_attempts.jsonl')
    chosen=[]
    for task in sorted({r['task'] for r in attempts}):
        chosen.extend([r for r in attempts if r['task']==task][:4])
    demos={r['base_id']:r for r in read_rows(HERE/'prepared/demos.jsonl')}
    records=[]; summary={}
    for arm,adapter in (('scheme3',initial),('trajectory_kd',new)):
        hf=FullSyncHFLM(receiver,'translated',sender,base,adapter)
        arm_rows=[]; consistency=[]
        for q in chosen:
            n=len(q['choices'])
            rotated=dict(q,choices=q['choices'][1:]+q['choices'][:1],gold='ABCDEFGHIJ'[('ABCDEFGHIJ'.index(q['gold'])-1)%n])
            pair=[]
            for variant,view in (('original',q),('rotated',rotated)):
                context=q['context'] if variant=='original' else prompt(view,[demos[i] for i in q['demo_ids']])
                ids,text=generate(hf,context,CONFIG['validation_generation_max_new_tokens'])
                found=ANSWER_RE.findall(text); pred=found[0] if found else '[invalid]'
                eos=receiver.generation_config.eos_token_id
                terminal='eos' if ids[-1] in eos else ('question_stop' if 'Question:' in text else 'length_limit')
                row={'arm':arm,'variant':variant,'base_id':q['base_id'],'task':q['task'],'context':context,
                     'choices':view['choices'],'gold':view['gold'],'prediction':pred,'correct':pred==view['gold'],
                     'terminal':terminal,'answer_phrase_count':len(found),'generated_tokens':len(ids),'ids':ids,'response':text}
                records.append(row); arm_rows.append(row); pair.append(row)
                write_rows(HERE/'diagnostics/generations.jsonl',records)
                log(f'DIAGNOSTIC {arm} {len(arm_rows)}/{2*len(chosen)}: gold={view["gold"]} pred={pred} stop={terminal}')
            options=[]
            for r in pair:
                options.append(r['choices']['ABCDEFGHIJ'.index(r['prediction'])] if r['prediction'] in 'ABCDEFGHIJ'[:n] else None)
            consistency.append(options[0] is not None and options[0]==options[1])
        summary[arm]={'n_generations':len(arm_rows),'correct':sum(r['correct'] for r in arm_rows),
                      'valid_extracted':sum(r['prediction']!='[invalid]' for r in arm_rows),
                      'terminal_counts':dict(Counter(r['terminal'] for r in arm_rows)),
                      'prediction_histogram':dict(Counter(r['prediction'] for r in arm_rows)),
                      'answer_phrase_repeated_at_least_10':sum(r['answer_phrase_count']>=10 for r in arm_rows),
                      'option_semantic_consistency_pairs':sum(consistency),'n_pairs':len(consistency)}
    save(HERE/'diagnostics/summary.json',{'MECHANISM_DIAGNOSTIC_NOT_OFFICIAL_BENCHMARK':True,'validation_only':True,
                                       'max_new_tokens':CONFIG['validation_generation_max_new_tokens'],
                                       'selected_checkpoint_sha256':digest(HERE/'stage_b/best.pt'),'comparisons':summary})
    log('GENERATION DIAGNOSTIC COMPLETED '+json.dumps(summary))


if __name__=='__main__': main()
