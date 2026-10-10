"""Isolate full-vs-split native prefill roundoff, without Sender or Writer."""
import json
import torch
from transformers import AutoModelForCausalLM
from runtime import CFG, ROOT, seed, tokenizers, prefill, freeze, restore, save, make_cache
from prompt_protocol import paper_protocol
from kv_backend import capture


@torch.no_grad()
def main():
    seed()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    st, rt = tokenizers()
    rows = json.loads((ROOT/'prepared.json').read_text())['smoke']
    records = []
    for dtype in (torch.float16, torch.float32):
        model = AutoModelForCausalLM.from_pretrained(CFG['models']['qwen'],
            local_files_only=True, dtype=dtype, attn_implementation='eager').cuda().eval().requires_grad_(False)
        for number, row in enumerate(rows, 1):
            proto = paper_protocol(st, rt, row['paper'], row['questions'])
            _, cache = prefill(model, proto['prefix_ids'])
            state = freeze(cache); del cache
            k, v = capture(model, proto['prefix_ids'])
            rebuilt = freeze(make_cache(model, k.cuda(), v.cuda(), torch.arange(len(proto['prefix_ids']), device='cuda')))
            for q in proto['queries'][:2]:
                full, cache = prefill(model, q['full_ids']); del cache
                split, cache = prefill(model, q['suffix_ids'], restore(model, state)); del cache
                replay, cache = prefill(model, q['suffix_ids'], restore(model, rebuilt)); del cache
                entry = {'dtype':str(dtype), 'paper_id':row['paper_id'], 'question_id':q['question_id'],
                    'full_split_mae':float((full-split).abs().mean()),
                    'full_split_max':float((full-split).abs().max()),
                    'split_rebuild_mae':float((split-replay).abs().mean()),
                    'split_rebuild_max':float((split-replay).abs().max()),
                    'full_split_argmax':int(full.argmax())==int(split.argmax()),
                    'split_rebuild_argmax':int(split.argmax())==int(replay.argmax())}
                records.append(entry); save(ROOT/'smoke/native_precision_audit.json', records)
                print(entry, flush=True)
                if dtype == torch.float32:
                    assert entry['full_split_mae'] < 1e-4 and entry['split_rebuild_mae'] < 1e-4, entry
                    assert entry['full_split_argmax'] and entry['split_rebuild_argmax'], entry
            del state, rebuilt, k, v
            print('PRECISION AUDIT',str(dtype),number,len(rows),flush=True)
        del model
        torch.cuda.empty_cache()
    save(ROOT/'smoke/native_precision_completed.json', {'protocol':CFG['protocol'],
        'papers':len(rows), 'queries_per_dtype':len(records)//2,
        'FP32_native_split_and_rebuild_mae_lt':1e-4, 'tf32':False})


if __name__ == '__main__':
    main()
