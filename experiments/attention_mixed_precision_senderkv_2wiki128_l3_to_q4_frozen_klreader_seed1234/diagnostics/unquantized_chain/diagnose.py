"""Bounded 16-row diagnostic. Does not modify the completed quantization experiment."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='1'
import sys,json,time,copy,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import experiment as E
OUT=Path(__file__).resolve().parent


def nmse(x,y):return float((x.float()-y.float()).square().sum()/y.float().square().sum().clamp_min(1e-30))


@torch.inference_mode()
def main():
    E.verify()
    selected=E.rows()[:16]
    assert not (OUT/'summary.json').exists(),'Diagnostic already complete; do not rerun'
    lt,llama=E.load_model('llama');qt,qwen=E.load_model('qwen');base,reader=E.load_frontend(qwen)
    E.CFG['max_new_tokens']=128  # Process-local diagnostic override, saved files unchanged.
    results=[]
    for i,row in enumerate(selected,1):
        E.save(OUT/'status.json',dict(status='running',sample=i,total=16,time=time.strftime('%F %T')))
        sk,sv,sc=E.capture_body(llama,row['layouts']['llama']['body_ids'])
        del sc
        native_audit={}
        if i==1:
            ck,cv,_=E.C.capture(llama,row['layouts']['llama']['body_ids'],len(row['layouts']['llama']['body_ids']),[0])
            native_audit.update(sender_key_capture_max_error=float((sk-ck).abs().max()),sender_value_capture_max_error=float((sv-cv).abs().max()))
            del ck,cv
        with reader.mode(False):
            qk,qv,native=E.capture_body(qwen,row['layouts']['qwen']['body_ids'])
            native_result,_,_=E.evaluate_cache(qwen,qt,row['layouts']['qwen'],native,row['answer'])
            key=sk[:,row['wire_positions']].contiguous();value=sv[:,row['wire_positions']].contiguous()
            cache,k,v=E.translated(qwen,base,reader,row,native,key,value)
        anchor=row['frames']['qwen']['native_prefix_length']
        map_error=dict(key_nmse=nmse(k,qk[:,anchor:].cuda()),value_nmse=nmse(v,qv[:,anchor:].cuda()))
        if i==1:
            old_loader=E.U.loaded_source
            try:
                E.U.loaded_source=lambda _:dict(key=sk[:,row['source_indices']].contiguous(),value=sv[:,row['source_indices']].contiguous())
                _,refk,refv=E.R.fixed_state(qwen,base,reader,row)
                native_audit.update(canonical_mapped_key_max_error=float((k-refk).abs().max()),canonical_mapped_value_max_error=float((v-refv).abs().max()))
                del refk,refv
            finally:E.U.loaded_source=old_loader
        with reader.mode(False):
            ab_result,_,_=E.evaluate_cache(qwen,qt,row['layouts']['qwen'],cache,row['answer'])
        with reader.mode(True):
            reader_result,_,_=E.evaluate_cache(qwen,qt,row['layouts']['qwen'],cache,row['answer'])
            if i==1:
                gold=row['layouts']['qwen']['gold_ids']
                logits=E.U.suffix_logits(qwen,row,copy.deepcopy(cache),gold[:-1],list(range(len(gold))))
                canonical_nll=float(F.cross_entropy(logits,torch.tensor(gold,device='cuda')))
                native_audit['canonical_vs_new_gold_nll_error']=abs(canonical_nll-reader_result['nll_per_token'])
        prior=E.read(E.result_path('qwen',row))['arms']['fp16']
        native_audit.update(formal_nll_replay_error=abs(reader_result['nll_per_token']-prior['nll_per_token']),
                           formal_generation_is_prefix=reader_result['generation'].startswith(prior['generation']))
        assert native_audit['formal_nll_replay_error']<.03,native_audit
        rec=dict(id=row['id'],question=row['question'],gold=row['answer'],native_qwen=native_result,
                 stage_ab=ab_result,stage_ab_reader=reader_result,mapped_cache_error=map_error,implementation_audit=native_audit)
        E.save(OUT/(row['id']+'.json'),rec);results.append(rec)
        print(time.strftime('%F %T'),i,'/16',json.dumps({a:dict(em=rec[a]['exact_match'],tokens=rec[a]['generated_tokens'],nll=rec[a]['nll_per_token']) for a in ['native_qwen','stage_ab','stage_ab_reader']}),flush=True)
        del sk,sv,key,value,qk,qv,native,cache,k,v
    summary=dict(status='completed',timestamp=time.strftime('%F %T'),n=16,max_new_tokens=128,
        original_experiment_manifest_sha256=E.identity(),sample_selection='First16 predetermined prepared questions; diagnostic subset, not a new benchmark.',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),arms={})
    for name in ['native_qwen','stage_ab','stage_ab_reader']:
        summary['arms'][name]=dict(correct=sum(r[name]['exact_match'] for r in results),
            capped=sum(r[name]['reached_cap'] for r in results),mean_nll=sum(r[name]['nll_per_token'] for r in results)/16,
            mean_tokens=sum(r[name]['generated_tokens'] for r in results)/16)
    summary['implementation_audit']=results[0]['implementation_audit']
    summary['mapped_key_nmse_mean']=sum(r['mapped_cache_error']['key_nmse'] for r in results)/16
    summary['mapped_value_nmse_mean']=sum(r['mapped_cache_error']['value_nmse'] for r in results)/16
    E.save(OUT/'summary.json',summary);E.save(OUT/'status.json',dict(status='completed',time=time.strftime('%F %T')))
    # Keep shared results automatic; append a separately labelled diagnostic, never rewrite the formal experiment.
    import fcntl
    with (ROOT.parent/'EXPERIMENT_RESULTS.md').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        stream.write('\n\n## '+E.CFG['name']+' / unquantized-chain diagnostic\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        E.save(OUT/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
