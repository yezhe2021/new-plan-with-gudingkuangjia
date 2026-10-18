"""Reuse all trusted data; freeze held-out initial-student prefixes before training."""
import argparse
import json
from followup_common import HERE,TRAJ,CONFIG,parent,base_train,gpu_check,model_bundle,FullSyncHFLM,read_rows,write_rows,save,log,digest
from student_kd import rollout


def main():
    parent.verify_manifest();gpu_check()
    folder=HERE/'prepared';folder.mkdir(exist_ok=True)
    for name in ('train_trajectories.jsonl','validation_trajectories.jsonl','validation_questions.jsonl','validation_attempts.jsonl','demos.jsonl'):
        source=TRAJ/'prepared'/name;target=folder/name
        if target.exists():assert target.read_bytes()==source.read_bytes()
        else:target.write_bytes(source.read_bytes())
    ident={'initial_stage_b_sha256':digest(TRAJ/'stage_b/best.pt'),'config_sha256':digest(HERE/'config.json')}
    if (folder/'identity.json').exists():assert json.loads((folder/'identity.json').read_text())==ident
    else:save(folder/'identity.json',ident)
    receiver,sender,base,adapter,source_tok,hf=model_bundle(False)
    gen=FullSyncHFLM(receiver,'translated',sender,base,adapter)
    dest=folder/'validation_student_prefixes.jsonl'
    rows=read_rows(dest) if dest.exists() else [];done={r['base_id'] for r in rows};anchors={}
    for q in read_rows(folder/'validation_attempts.jsonl'):
        if q['base_id'] in done:continue
        row=dict(q,context_ids=hf.tok_encode(q['context']))
        item,bk,bv=base_train.prefix(sender,receiver,base,source_tok,hf,row,anchors)
        ids,text=rollout(gen,adapter,row,item,bk,bv,CONFIG['validation_rollout_max_tokens'])
        rows.append({'base_id':q['base_id'],'task':q['task'],'gold':q['gold'],'context':q['context'],
                     'context_ids':row['context_ids'],'continuation_ids':ids,'response':text,
                     'validation_prefix_only_not_gold_targets':True,'initial_stage_b_sha256':ident['initial_stage_b_sha256']})
        write_rows(dest,rows);log(f'FIXED VALIDATION STUDENT PREFIX {len(rows)}/32')
    assert len(rows)==32
    save(folder/'summary.json',{'train_references':256,'reference_validation':len(read_rows(folder/'validation_trajectories.jsonl')),
                              'fixed_validation_student_prefixes':len(rows),'student_prefix_hard_labels':False,
                              'reused_original_data':True,'benchmark_training':False})


if __name__=='__main__':main()
