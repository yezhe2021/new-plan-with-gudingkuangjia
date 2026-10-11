import argparse, hashlib, json, random, time, gc
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import AutoTokenizer
from kv_backend import CFG, ROOT, pair, map_base, load_base, load_model, seed_all
from metrics import compare, tests

def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def qhash(q):return hashlib.sha256(' '.join(q.split()).encode()).hexdigest()
def context(body):return CFG['prompt_template'].format(body=body).rstrip()
def prepare():
    assets=json.loads((ROOT/'heldout_assets.json').read_text()); blocked=set(assets['blocked_question_hashes'])
    tok=AutoTokenizer.from_pretrained(CFG['models']['qwen'],local_files_only=True)
    rng=random.Random(CFG['seed']); result=[]; rejected=Counter(); seen=set()
    for corpus in ('openbookqa','arc_challenge','gsm8k'):
        candidates=[d for d in assets['rows'] if d['corpus']==corpus];rng.shuffle(candidates);chosen=[]
        for d in candidates:
            key=qhash(d['question'])
            if key in blocked or key in seen:rejected['overlap_or_duplicate']+=1;continue
            body='Question:\n'+d['question']
            if d['options']:body+='\n\nOptions:\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(d['options']))
            text=context(body);n=len(tok.encode(text,add_special_tokens=False))
            if not CFG['min_tokens']<=n<=CFG['max_tokens']:rejected['length']+=1;continue
            seen.add(key);chosen.append(dict(d,context=text,qwen_tokens=n,question_hash=key,split='official_test'))
            if len(chosen)==CFG['counts'][corpus]:break
        assert len(chosen)==CFG['counts'][corpus],(corpus,len(chosen));result+=chosen
    import pyarrow.parquet as pq
    wiki=pq.read_table(CFG['wiki_test']).to_pylist();rng.shuffle(wiki)
    # Match Wiki length targets to an equal mixture of held-out MCQ/GSM, not artificially long documents.
    mcq=[r['qwen_tokens'] for r in result if r['corpus']!='gsm8k'];gsm=[r['qwen_tokens'] for r in result if r['corpus']=='gsm8k']
    desired=mcq[::2]+gsm[::2];rng.shuffle(desired);wi=[];used=set();cursor=0
    for target in desired:
        while cursor<len(wiki):
            text=wiki[cursor]['text'].strip();cursor+=1
            if not text or text.startswith('=') or qhash(text) in used:continue
            enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
            count=max(8,target-len(tok.encode(context(''),add_special_tokens=False)))
            if len(enc['input_ids'])<count:continue
            frag=text[:enc['offset_mapping'][count-1][1]].rstrip();prompt=context(frag)
            n=len(tok.encode(prompt,add_special_tokens=False))
            if not CFG['min_tokens']<=n<=CFG['max_tokens']:continue
            used.add(qhash(text));wi.append(dict(corpus='wikitext',id=f'paragraph_{cursor}',context=prompt,qwen_tokens=n,
                length_target=target,question_hash=qhash(frag),split='official_test',source_text_hash=qhash(text)));break
        else:raise RuntimeError('Insufficient Wiki test paragraphs')
    assert len(wi)==128;result+=wi
    assert len(result)==384 and not ({r['question_hash'] for r in result}&blocked)
    save(ROOT/'manifest.json',result)
    save(ROOT/'data_audit.json',{'count':384,'source_assets_sha256':sha(ROOT/'heldout_assets.json'),
        'wiki_test_sha256':sha(CFG['wiki_test']),'exclusion_hash_count':len(blocked),'selected_overlap':0,
        'rejected':dict(rejected),'prompt_template':CFG['prompt_template'],
        'lengths':{c:{'min':min(r['qwen_tokens'] for r in result if r['corpus']==c),
            'max':max(r['qwen_tokens'] for r in result if r['corpus']==c),
            'mean':float(np.mean([r['qwen_tokens'] for r in result if r['corpus']==c]))} for c in CFG['counts']}})
    print('PREPARED 384 HELD-OUT INPUTS',flush=True)

def aggregate(rows):
    summaries={};heads={}
    groups={c:[r for r in rows if r['corpus']==c] for c in CFG['counts']}
    groups['mcq']=groups['openbookqa']+groups['arc_challenge'];groups['all']=rows
    for name,rr in groups.items():
        if not rr:continue
        summaries[name]={'n':len(rr),'mean_translated_tokens':float(np.mean([r['tokens'] for r in rr]))}
        heads[name]={}
        for kind in ('K','V'):
            summaries[name][kind]={k:float(np.mean([r[kind]['scalar'][k] for r in rr])) for k in rr[0][kind]['scalar']}
            heads[name][kind]={k:np.mean([r[kind]['layer_head'][k] for r in rr],axis=0).tolist() for k in rr[0][kind]['layer_head']}
    bins={}
    for c,rr in groups.items():
        if c=='all':continue
        bins[c]={}
        for b in sorted({r['qwen_tokens']//32 for r in rr}):
            part=[r for r in rr if r['qwen_tokens']//32==b]
            bins[c][str(b)]={'n':len(part),**{kind:{k:float(np.mean([r[kind]['scalar'][k] for r in part])) for k in part[0][kind]['scalar']} for kind in ('K','V')}}
    common=set(bins.get('mcq',{}))&set(bins.get('gsm8k',{}))&set(bins.get('wikitext',{}))
    standardized={c:{kind:{k:float(np.mean([bins[c][b][kind][k] for b in common])) for k in rows[0][kind]['scalar']} for kind in ('K','V')} for c in ('mcq','gsm8k','wikitext')} if common else {}
    save(ROOT/'results/summary.json',summaries);save(ROOT/'results/layer_head.json',heads)
    save(ROOT/'results/length_stratified.json',{'bin_width':32,'bins':bins,'common_bins':sorted(common),
        'standardized_equal_bin_weight':standardized,'caution':'Length bins mitigate but do not eliminate content/prompt confounding.'})
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind in ('K','V'):
        fig,ax=plt.subplots(figsize=(9,12));data=np.asarray(heads['all'][kind]['mg_symmetric_nmse'])
        im=ax.imshow(data,aspect='auto',origin='upper');ax.set_xticks(range(data.shape[1]));ax.set_xlabel('Qwen KV head (0-based)');ax.set_ylabel('Qwen layer (0-based)')
        ax.set_title(f'{kind}: MCQ vs GSM8K symmetric NMSE (sample macro-average)');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(ROOT/f'results/{kind}_difference_heatmap.png',dpi=160);plt.close(fig)
    lines=['# Cross-Corpus Stage-A Mapping Consistency','', 'No training, no Stage-B. Same Sender and Native state per sample. Macro-average over samples; K post-k_norm/pre-RoPE. Neutral common wrapper: analysis, not accuracy benchmark.','',
        '|Corpus|N|KV|M/G cosine|M/G symmetric NMSE|MCQ→Native NMSE|GSM→Native NMSE|Error cosine|','|---|---:|---|---:|---:|---:|---:|---:|']
    for c,s in summaries.items():
        for k in ('K','V'):
            a=s[k];lines.append(f"|{c}|{s['n']}|{k}|{a['mg_cosine']:.5f}|{a['mg_symmetric_nmse']:.5f}|{a['mcq_native_nmse']:.5f}|{a['gsm8k_native_nmse']:.5f}|{a['error_cosine']:.5f}|")
    (ROOT/'results/REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

@torch.inference_mode()
def run(smoke=False):
    tests();seed_all(CFG['seed']);assert torch.cuda.is_available()
    provenance={}
    for name,path in CFG['checkpoints'].items():
        p=ROOT/path;digest=sha(p)
        assert digest==CFG['checkpoint_sha256'][name],(name,'checkpoint hash mismatch')
        payload=torch.load(p,map_location='cpu',weights_only=True)
        assert payload['step']==512 and payload['protocol']==CFG['protocol']
        assert payload['stage'] in ('a','stage_a')
        provenance[name]={'path':str(p),'sha256':digest,'epoch':payload['epoch'],'step':payload['step']};del payload
    tok_s=AutoTokenizer.from_pretrained(CFG['models']['llama'],local_files_only=True)
    tok_t=AutoTokenizer.from_pretrained(CFG['models']['qwen'],local_files_only=True)
    sender=load_model(CFG,'llama');receiver=load_model(CFG,'qwen')
    bm=load_base(ROOT/CFG['checkpoints']['mcq']);bg=load_base(ROOT/CFG['checkpoints']['gsm8k'])
    assert all(not p.requires_grad for x in (sender,receiver,bm,bg) for p in x.parameters())
    inputs=json.loads((ROOT/'manifest.json').read_text()); folder=ROOT/('smoke' if smoke else 'results');folder.mkdir(exist_ok=True)
    signature={'writers':provenance,'manifest_sha256':sha(ROOT/'manifest.json'),'config_sha256':sha(ROOT/'config.json')}
    if (folder/'provenance.json').exists():
        old=json.loads((folder/'provenance.json').read_text())
        assert all(old[k]==v for k,v in signature.items()),'Refuse resume with changed inputs/checkpoints/config'
    save(folder/'provenance.json',{**signature,
        'key_domain':'post-k_norm/pre-RoPE','translated_positions':'1..N-2 (zero-based), excludes native anchor and final native readout',
        'cosine':'Frobenius flatten per sample and per layer/head; token-vector cosine also provided',
        'aggregation':'per-sample metrics then arithmetic corpus means, no token pooling','model_dtype':'FP16','metric_reductions':'CPU FP64',
        'error_zero_norm':'cosine set to0 with explicit zero-norm fraction; do not interpret as error-direction evidence',
        'target_layers':receiver.config.num_hidden_layers,'target_heads':receiver.config.num_key_value_heads})
    if smoke:inputs=[next(r for r in inputs if r['corpus']==c) for c in CFG['counts']]
    output=folder/'per_sample.jsonl';done=[]
    if output.exists():done=[json.loads(x) for x in output.read_text().splitlines() if x.strip()]
    keys={(r['corpus'],r['id']) for r in done};start=time.monotonic()
    for r in inputs:
        if (r['corpus'],r['id']) in keys:continue
        t=time.monotonic();item=pair(sender,receiver,tok_s,tok_t,r['context']);assert item['tokens']==len(item['ids'])-2
        mk,mv=map_base(bm,item,CFG['chunk_tokens']);gk,gv=map_base(bg,item,CFG['chunk_tokens'])
        row={k:r[k] for k in ('corpus','id','qwen_tokens')};row.update(tokens=item['tokens'],sync_counts=item['counts'],context_sha256=qhash(r['context']))
        for kind,m,g,target in [('K',mk,gk,item['target_k']),('V',mv,gv,item['target_v'])]:
            scalar,matrix=compare(m,g,target);row[kind]={'scalar':scalar,'layer_head':matrix}
        if smoke:
            rk,rv=map_base(bm,item,CFG['chunk_tokens']);assert torch.equal(mk,rk) and torch.equal(mv,rv)
            del rk,rv
        row['seconds']=time.monotonic()-t
        with output.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        done.append(row);print(f"DONE {len(done)}/{len(inputs)} {r['corpus']} tokens={item['tokens']} seconds={row['seconds']:.2f}",flush=True)
        del item,mk,mv,gk,gv,m,g,target;gc.collect()
    if not smoke:aggregate(done)
    save(folder/'completed.json',{'n':len(done),'seconds':time.monotonic()-start,'smoke':smoke})
    print('SMOKE PASSED' if smoke else 'ALL ANALYSIS COMPLETED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','smoke','run']);a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:run(a.mode=='smoke')
