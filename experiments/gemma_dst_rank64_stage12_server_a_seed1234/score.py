"""Unchanged conditional likelihood and generation scorers, without Memory."""
from experiment_common import *
ORIGINAL=N.ORIGINAL;GSM=N.GSM
sys.path.append(str(C.ROOT))
sys.path.append(str(ORIGINAL/'cross_dataset5x64_seed1234'))
spec=importlib.util.spec_from_file_location('dst_cross_scorer',ORIGINAL/'cross_dataset5x64_seed1234/experiment.py')
X=importlib.util.module_from_spec(spec);spec.loader.exec_module(X)
spec=importlib.util.spec_from_file_location('dst_unchanged_gsm_numeric',GSM/'numeric.py')
NUM=importlib.util.module_from_spec(spec);spec.loader.exec_module(NUM)

@torch.no_grad()
def score(module,row,cache,*,rn=False):
    if row['task']!='gsm8k':
        return module.score(row,cache,rn=rn) if row['task'] in ['arc','hotpotqa'] else X.score(module,row,cache,rn=rn)
    cap=CFG['evaluation_generation_caps']['gsm8k'];started=time.time();tokens=module.rollout(row,cache,rn=rn,cap=cap)
    eos=module.engine.model.generation_config.eos_token_id;stops=set(eos if isinstance(eos,list) else [eos]);stops.add(module.engine.tok.convert_tokens_to_ids('<|im_end|>'))
    text=module.engine.tok.decode(tokens,skip_special_tokens=True);answer=NUM.extract(text);legacy=C.SC.extract_numeric_answer(text)
    gold=NUM.normalize(row['gold']);stopped=bool(tokens and tokens[-1] in stops)
    return dict(**answer,correct=int(answer['prediction']==gold),legacy_prediction=legacy,legacy_correct=int(NUM.normalize(legacy)==gold),generation=text,
        token_ids=tokens,generated_tokens=len(tokens),stopped_on_eos=stopped,reached_cap=int(len(tokens)==cap and not stopped),max_new_tokens=cap,seconds=time.time()-started)

