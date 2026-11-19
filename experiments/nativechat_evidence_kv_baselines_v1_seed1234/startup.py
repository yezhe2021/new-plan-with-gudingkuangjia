"""Real-model native cache identity and evidence-only source correctness."""
import copy
import torch
from common import ROOT, CFG, rows, save, seed, gpu_check, log, release
from protocol import build, encode, audit_row, sync_indices
from models import Pool, capture, cache, run, generate, backbone


def main():
    seed()
    gpu_check()
    pool = Pool()
    train = rows(ROOT / 'prepared/train.jsonl')
    selected = [train[0], max(train, key=lambda r: len(r['documents'][0]) + len(r['documents'][1]))]
    records = []
    receiver = pool.gpu('qwen')
    for row in selected:
        proto = build(pool.tokenizers, row)
        t = proto['target']
        features = capture(receiver, t['ids'][:t['end']])
        memory = features['k'], features['v']
        with torch.no_grad():
            direct = run(receiver, t['ids'])
            injected = run(receiver, t['suffix_ids'], cache(receiver, *memory))
            a = receiver.lm_head(direct.last_hidden_state[:, -1])
            b = receiver.lm_head(injected.last_hidden_state[:, -1])
            drift = float((a-b).abs().max())
            assert drift <= CFG['native_logit_atol'] and a.argmax() == b.argmax(), drift
        del direct, injected, a, b
        native_ids, _ = generate(receiver, t['ids'], limit=16)
        rebuilt_ids, _ = generate(receiver, t['suffix_ids'], memory, limit=16)
        assert native_ids == rebuilt_ids, 'Native cache injection changes greedy generation'
        inp = torch.tensor([t['ids']],device='cuda')
        with torch.no_grad():
            hf_ids = receiver.generate(input_ids=inp, attention_mask=torch.ones_like(inp), max_new_tokens=16,
                  do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.)[0,len(t['ids']):].tolist()
        assert hf_ids == native_ids, 'Manual greedy decoder differs from native generate'
        records.append({'id': row['id'], 'qwen_body_tokens':len(t['body_ids']), 'max_logit_drift': drift,
                        'greedy_16_token_identity': True, 'native_generate_identity': True})
        log('NATIVE PARITY ' + str(records[-1]))
    del receiver
    pool.cpu()
    sources = []
    for family in ('llama','gemma'):
        model = pool.gpu(family)
        row = selected[-1]
        proto = build(pool.tokenizers, row)
        changed = build(pool.tokenizers, dict(row, question='A different question which is never sent to the sender.'))
        assert proto['source'][family]['ids'] == changed['source'][family]['ids']
        features = capture(model, proto['source'][family]['ids'])
        normalization_drift=0.
        with torch.no_grad():
            for layer,attention_layer in enumerate(backbone(model).layers):
                raw=features['raw_k'][layer].cuda()
                norm=getattr(attention_layer.self_attn,'k_norm',None)
                expected=raw if norm is None else norm(raw)
                drift=float((expected.cpu()-features['k'][layer]).abs().max())
                normalization_drift=max(normalization_drift,drift)
                assert drift<1e-5,'Native normalized-key capture axis mismatch'
        del raw,expected
        sources.append({'family':family,'finite_native_capture':True,'question_absent_from_sender':True,
                        'native_key_normalization_drift':normalization_drift,
                        'body_tokens':len(proto['source'][family]['body_ids']),
                        'fullsync_mapped_tokens':len(sync_indices(proto['source'][family],proto['target']))})
        del features
        del model
        pool.cpu()
    save(ROOT / 'reports/native_startup.json', {'passed':True,'native':records,'sources':sources,
         'model_dtype':'float32','native_cache_injection_compared_to_same_joint_prefill':True,
         'native_independent_is_not_required_to_equal_joint':True})
    log('NATIVE PROTOCOL STARTUP PASSED')


if __name__ == '__main__':
    main()
