"""Real CUDA protocol/attention/gradient tests, gated before any training."""
import argparse
import json
import torch
from runtime import CFG,ROOT,seed,save,load_models,tokenizers,adapter,base,freeze,make_cache,prefill,decode,answer_with_memory,digest_state
from kv_backend import PaperCache
from attention_probe import collect,losses
from train_stage_a import chunk_losses


def run(diagnostic=False):
    # Full/split GEMMs use different shapes. Audit that difference in FP32,
    # then require exact reconstruction of the native FP16 split baseline.
    precision_done=json.loads((ROOT/'smoke/native_precision_completed.json').read_text())
    assert precision_done['protocol']==CFG['protocol']
    precision_records=json.loads((ROOT/'smoke/native_precision_audit.json').read_text())
    precision={(r['paper_id'],r['question_id']):r for r in precision_records if r['dtype']=='torch.float32'}
    seed(); sender,receiver=load_models(); st,rt=tokenizers(); cache=PaperCache(sender,receiver,st,rt)
    if diagnostic:
        rows=sorted(json.loads((ROOT/'available_papers.json').read_text())['train'],key=lambda p:p['body_tokens'])[:2]
    else: rows=json.loads((ROOT/'prepared.json').read_text())['smoke']
    output=ROOT/('smoke_diagnostic' if diagnostic else 'smoke')
    cal=adapter(); records=[]
    for number,row in enumerate(rows,1):
        item=cache.get(row,teacher=False)
        assert 'native_k' not in item and 'target_k' not in item, 'Source-only path calculated native evidence'
        item=cache.get(row); proto=item['protocol']; prefix=len(proto['prefix_ids'])
        _,nc=prefill(receiver,proto['prefix_ids']); native=freeze(nc); del nc
        rebuilt=freeze(make_cache(receiver,item['native_k'].cuda(),item['native_v'].cuda(),torch.arange(prefix,device='cuda')))
        hashes=(digest_state(native),digest_state(rebuilt))
        for q in proto['queries'][:2]:
            audited=precision[(row['paper_id'],q['question_id'])]
            assert audited['full_split_mae']<1e-4 and audited['split_rebuild_mae']<1e-4
            assert audited['full_split_argmax'] and audited['split_rebuild_argmax']
            with torch.no_grad():
                logits,nc=prefill(receiver,q['full_ids']); first=logits.cpu(); full=decode(receiver,rt,logits,nc); del nc
                checks={}
                for name,state in [('native_cache',native),('native_rebuild',rebuilt)]:
                    pred,actual=answer_with_memory(receiver,rt,state,q['suffix_ids'])
                    mae=float((actual-first).abs().mean())
                    same=int(actual.argmax())==int(first.argmax())
                    checks[name]={'generation_identical':pred['ids']==full['ids'],'argmax_identical':same,
                                  'logit_mean_abs_error':mae,'logit_max_abs_error':float((actual-first).abs().max())}
                    assert same, checks
                    if name=='native_cache':
                        native_first=actual
                    else:
                        assert torch.equal(actual,native_first), 'Rebuilt cache differs from native FP16 cache'
                if checks['native_cache']['logit_mean_abs_error']>CFG['smoke_first_logit_mae_tolerance']:
                    print('AUDITED FP16 full/split roundoff',row['paper_id'],q['question_id'],checks,flush=True)
                probe=collect(receiver,item,q); worst=0.
                for layer in range(36):
                    lk,lv,stats=losses(receiver,item,probe,layer,item['target_k'][layer].cuda(),item['target_v'][layer].cuda(),cal)
                    assert torch.isfinite(lk+lv) and abs(float(lk+lv))<1e-4, 'Identity attention replay loss'
                    worst=max(worst,stats['native_replay_nmse'])
                assert worst<=CFG['smoke_replay_nmse_tolerance'], f'Native GQA/W_O/RoPE replay NMSE {worst}'
            records.append({'paper_id':row['paper_id'],'question_id':q['question_id'],'checks':checks,
                            'worst_native_attention_replay_nmse':worst,'all_evidence_positions_covered':True})
            save(output/'protocol.json',records)
        assert hashes==(digest_state(native),digest_state(rebuilt)), 'Query mutated source memory'
        assert cache.capture_counts[row['paper_id']]==1
        print(f'PROTOCOL SMOKE paper={number}/{len(rows)} passed',flush=True)
        if number==1:
            probe=collect(receiver,item,proto['queries'][0])
            bk=item['target_k'][0].cuda()+0.05*torch.randn_like(item['target_k'][0].cuda())
            bv=item['target_v'][0].cuda()+0.05*torch.randn_like(item['target_v'][0].cuda())
            cal.zero_grad(set_to_none=True); lk,_,_=losses(receiver,item,probe,0,bk,bv,cal); lk.backward()
            assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in cal.k.parameters())
            assert all(p.grad is None for p in cal.v.parameters()), 'Key loss leaked to Value'
            cal.zero_grad(set_to_none=True); _,lv,_=losses(receiver,item,probe,0,bk,bv,cal); lv.backward()
            assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in cal.v.parameters())
            assert all(p.grad is None for p in cal.k.parameters()), 'Value loss leaked to Key'
            assert all(p.grad is None for p in receiver.parameters()) and all(p.grad is None for p in sender.parameters())
            cal.zero_grad(set_to_none=True)
            mapper=base(); tiny={k:v for k,v in item.items()}
            for k in ('source_k','source_v','target_k','target_v'): tiny[k]=tiny[k][:,:64]
            tiny['tokens']=64
            optimizer=torch.optim.SGD(mapper.parameters(),lr=1e-5)
            for loss in chunk_losses(mapper,tiny): loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(mapper.parameters(),30)
            assert torch.isfinite(norm) and norm>0
            optimizer.step(); del mapper,optimizer,tiny,bk,bv
            save(output/'gradient.json',{'K_only':True,'V_only':True,'base_models_frozen':True,
                'stage_a_backward_update_finite':True,'GQA_heads':receiver.config.num_attention_heads,
                'KV_heads':receiver.config.num_key_value_heads,'K_domain':CFG['key_domain']})
            torch.cuda.empty_cache()
    save(output/'completed.json',{'papers':len(rows),'queries':len(records),'diagnostic_only':diagnostic,
        'native_argmax_passed':True,'FP32_full_split_mae_lt':1e-4,
        'FP16_native_rebuild_logits_exact':True,'full_generation_exact_count':{
            name:sum(r['checks'][name]['generation_identical'] for r in records) for name in ('native_cache','native_rebuild')},
        'FP16_generation_differences_preserved':True,'all_native_attention_layers_replayed':36})
    print('ALL PROTOCOL AND GRADIENT TESTS PASSED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--diagnostic',action='store_true')
    run(parser.parse_args().diagnostic)
