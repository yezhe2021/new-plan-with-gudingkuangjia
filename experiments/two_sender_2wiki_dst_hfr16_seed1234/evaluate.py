"""Zero-shot split-evidence evaluation, retaining the original checkpoint alignments."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RECOLOR = PROJECT / 'recolor_kl_l3_to_q4_seed1234'
UTIL = PROJECT / 'herb_evidence_kv_pilot_20261119'
sys.path[:0] = [str(UTIL), str(RECOLOR / 'reference'),
                 str(PROJECT / 'rnkv_c_native_regeneration_v1_seed1234/cross_dataset5x64_seed1234')]
import torch
from transformers import AutoTokenizer, DynamicCache
from evaluate_multisender import make_writer, load_model, save, sha, read
from herb_metrics import clean_prediction
from qa_metrics import answer_metrics
from heterofold.capture import capture_kv
from heterofold.maps import load_map
from heterofold.modeling import decoder, chat_affixes
from heterofold.transfer import _cache_layer, _rotary_tables, finalize_layer, capture_sender, map_layer, source_indices


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    model_dir = Path('/hy-tmp/yezhe/data/all_models/models')
    paths = [model_dir / 'Qwen/Qwen3-4B', model_dir / 'LLM-Research/Llama-3.2-3B-Instruct']
    target, source = [AutoTokenizer.from_pretrained(p, local_files_only=True) for p in paths]
    model, sender = [load_model(p, f, 'dst_rank64', torch) for p, f in zip(paths, ['qwen', 'llama'])]
    dst, pins = make_writer(SimpleNamespace(project=PROJECT, canonical_dst=True), 'dst_rank64', 'llama',
                            source, sender, target, None, torch)
    file = ROOT / 'inputs/heterofold.pt'
    mode = sys.argv[1]
    assert mode in ('base','paper')
    pin = read(RECOLOR / 'config.json')['historical_heterofold_sha256']
    if mode == 'paper':
        assert sha(file) == pin
        mapping = load_map(str(file), device='cuda', dtype=torch.float32, mmap=True)
        assert mapping['functional_variant'] == 'output_aware_rank16' and mapping['functional_correction']['rank'] == 16
    prefix, assistant = chat_affixes(target, thinking=False)
    def encode(s):
        return target.encode(s, add_special_tokens=False)
    hids = encode(prefix)
    nhead = len(hids)
    def forward(ids, cache=None):
        start = 0 if cache is None else cache.get_seq_length()
        return decoder(model)(input_ids=torch.tensor([ids], device='cuda'), past_key_values=cache,
                              position_ids=torch.arange(start, start+len(ids), device='cuda')[None],
                              attention_mask=torch.ones(1, start+len(ids), dtype=torch.long, device='cuda'), use_cache=True)
    header = forward(hids).past_key_values
    layers = range(model.config.num_hidden_layers)
    def native(seg):
        _, k, v = capture_kv(target, model, prefix + seg['text'], layers, device='cpu')
        return tuple(torch.stack([group[l][nhead:] for l in layers]) for group in (k,v))
    def paper(seg):
        off, k, v = capture_sender(source, sender, mapping, seg['text'], device='cuda')
        dest = target(seg['text'], add_special_tokens=False, return_offsets_mapping=True)
        assert dest['input_ids'] == seg['ids']
        idx = source_indices(off, dest['offset_mapping'], 'cuda')
        pairs = [tuple(x.cpu() for x in map_layer(mapping,k,v,idx,l)) for l in layers]
        return tuple(torch.stack(xs) for xs in zip(*pairs))
    def assemble(parts):
        size = sum(k.shape[1] for k,v in parts)
        tables = _rotary_tables(model,torch.arange(nhead,nhead+size,device='cuda'))
        pairs = []
        for l in layers:
            k, v = [torch.cat([part[i][l] for part in parts]).cuda() for i in (0,1)]
            k, v = finalize_layer(model,l,k,v,*tables(l))
            hk, hv = _cache_layer(header,l)
            pairs.append((torch.cat((hk,k[None]),2),torch.cat((hv,v[None]),2)))
        return DynamicCache(ddp_cache_data=pairs,config=model.config)
    eos = model.generation_config.eos_token_id
    stops = set(eos if isinstance(eos,list) else [eos]) | {target.eos_token_id,target.convert_tokens_to_ids('<|im_end|>')}
    def evaluate(cache, suffix, gold):
        state = forward(suffix,cache)
        out = []
        for i in range(64):
            token = int(model.lm_head(state.last_hidden_state[:,-1]).float().argmax())
            out.append(token)
            if token in stops or i == 63:
                break
            state = forward([token],state.past_key_values)
        text = target.decode(out,skip_special_tokens=False)
        pred = clean_prediction(text)
        return dict(prediction=pred,tokens=out,generation=text,**answer_metrics(pred,[gold],'2wiki'))
    manifest = dict(cohort=sha(ROOT/'cohort.json'),dst=pins,heterofold=pin,code=sha(Path(__file__)),
                    seed=1234,cap=64,methods=['native','dst_rank64','heterofold_r16'],conditions=['joint','split','a_only','b_only'])
    if (ROOT/'manifest.json').exists():
        assert read(ROOT/'manifest.json') == manifest
    save(ROOT/'manifest.json',manifest)
    records = []
    for num,row in enumerate(read(ROOT/'cohort.json'),1):
        path = ROOT/'per_sample'/(row['id']+'.json')
        old = read(path) if path.exists() else {}
        results = old.get('scores', {})
        needed = ['native','dst_rank64'] if mode == 'base' else ['heterofold_r16']
        if all(m in results for m in needed):
            records.append(old)
            continue
        suffix_text = 'Question: '+row['question']+'\nOutput only the short answer, without an explanation.'+assistant
        suffix=encode(suffix_text)
        a,b,joint = [dict(text=t,ids=encode(t)) for t in [row['a'],row['b'],row['a']+row['b']]]
        assert joint['ids']==a['ids']+b['ids']
        for seg in (a,b,joint):
            assert encode(prefix+seg['text']+suffix_text)==hids+seg['ids']+suffix
        for name,translate in [('native',native),('dst_rank64',dst),('heterofold_r16',paper)]:
            if name not in needed or name in results:
                continue
            left,right=[translate(s) for s in (a,b)]
            score=lambda parts: evaluate(assemble(parts),suffix,row['answer'])
            results[name]=dict(split=score([left,right]),a_only=score([left]),b_only=score([right]))
            del left,right
            if name == 'native':
                if num == 1:
                    def first_logits(cache):
                        return model.lm_head(forward(suffix,cache).last_hidden_state[:,-1]).float()
                    ref=first_logits(forward(hids+joint['ids']).past_key_values)
                    rec=first_logits(assemble([native(joint)]))
                    kl=torch.nn.functional.kl_div(rec.log_softmax(-1),ref.log_softmax(-1),log_target=True,reduction='sum')
                    assert float(kl)<0.001 and ref.argmax()==rec.argmax()
                    save(ROOT/'native_parity.json',dict(kl=float(kl),top1=True))
                cache=forward(hids+joint['ids']).past_key_values
                results[name]['joint']=evaluate(cache,suffix,row['answer'])
                del cache
            else:
                results[name]['joint']=score([translate(joint)])
        rec=dict(id=row['id'],gold=row['answer'],tokens_a=len(a['ids']),tokens_b=len(b['ids']),scores=results)
        save(path,rec)
        records.append(rec)
        print(f'Completed {num}/64 {row["id"]}',flush=True)
    if not all(len(r['scores']) == 3 for r in records):
        return
    summary={}
    for m in manifest['methods']:
        arms=[r['scores'][m] for r in records]
        summary[m]={s:{metric:sum(x[s][metric] for x in arms)/len(arms) for metric in ['em','f1']}
                    for s in manifest['conditions']}
        summary[m]['both_sides_wrong_split_correct']=sum(x['split']['em']==1 and x['a_only']['em']==x['b_only']['em']==0 for x in arms)
        summary[m]['joint_correct_split_wrong']=sum(x['joint']['em']>x['split']['em'] for x in arms)
        summary[m]['joint_wrong_split_correct']=sum(x['joint']['em']<x['split']['em'] for x in arms)
    save(ROOT/'summary.json',dict(n=len(records),methods=summary))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
