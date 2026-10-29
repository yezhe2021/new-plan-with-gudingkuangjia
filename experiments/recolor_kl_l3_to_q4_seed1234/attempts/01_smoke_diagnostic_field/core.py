"""Frozen official RAW interface and one differentiable pre-norm residual cache path."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ.update(TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HETEROFOLD_DTYPE='float16',HETEROFOLD_V100_SDPA='1',HETEROFOLD_CAPTURE_DEVICE='cpu')
import contextlib
import hashlib
import json
import random
import sys
import time
import unicodedata
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
sys.path.insert(0,str(ROOT/'reference'))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,DynamicCache
from heterofold import calibration as HC
from heterofold.maps import load_map,atomic_save
from heterofold.modeling import load_model,chat_affixes,layers_of,decoder
from heterofold.transfer import finalize_layer,_rotary_tables,_cache_layer,capture_sender,build_receiver_cache
from heterofold.attention import attention_probabilities,causal_mask,output_projection
from heterofold.evaluation import runtime as RT,short_context as SC
from heterofold.evaluation.v100_sdpa import configure
configure();torch.set_num_threads(1)
from residuals import make_corrections


def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def identity():return sha(ROOT/'manifest.json')
def mark(path,**v):save(ROOT/path,dict(manifest_sha256=identity(),**v))
def status(**v):save(ROOT/'status.json',dict(time=time.strftime('%F %T'),**v))
def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def read_jsonl(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def rows(split):return read_jsonl(ROOT/'prepared'/(split+'.jsonl'))
def norm(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
def log(*v):print(time.strftime('%F %T'),*v,flush=True)
def fresh(arm):
    a=CFG['arms'][arm];return make_corrections(a['kind'],a['rank'],CFG['seed'])
def versions(model):return [(n,id(p),p._version) for n,p in model.named_parameters()]


def prepare():
    if (ROOT/'manifest.json').exists():verify();return
    assert sha(CFG['recolor'])==CFG['recolor_sha256']
    assert sha(CFG['historical_heterofold'])==CFG['historical_heterofold_sha256']
    original=read(CFG['calibration']);cal=original['rows']
    assert Counter((r['split'],r['dataset']) for r in cal)==Counter({('train','Open-R1 prompt-only'):800,('validation','Open-R1 prompt-only'):200,('train','HotpotQA distractor train full-context'):800,('validation','HotpotQA distractor train full-context'):200})
    m=load_map(CFG['recolor']);assert m['source_space']=='kv_prerope_joint' and m['target_space']=='kv_pre_norm'
    assert m['calibration']['train_prompts']==1600 and m['calibration']['text_count']==1600
    assert m['calibration']['prompt_sha256_sequence']==original['prompt_sha256_sequence']
    geometry={k:m[k] for k in ['Ls','Lr','kv_width','source_space','target_space']};del m
    tok=AutoTokenizer.from_pretrained(CFG['models']['receiver'],local_files_only=True)
    prefix,assistant=chat_affixes(tok,thinking=False);prefix_ids=tok.encode(prefix,add_special_tokens=False)
    prepared={s:[] for s in ['train','validation','test']};maxima=Counter()
    for i,r in enumerate(cal):
        row=dict(r,id=f'cal_{i:04d}',continuation_suffix=assistant)
        assert row['prompt'][row['question_char_start']:row['question_char_end']]==row['question']
        encoding=tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        row['content_ids']=encoding['input_ids'];row['suffix_ids']=tok.encode(assistant,add_special_tokens=False)
        row['probe_positions']=HC.question_token_positions(encoding['offset_mapping'],row['question_char_start'],row['question_char_end'])
        assert tok.encode(prefix+row['prompt'],add_special_tokens=False)==prefix_ids+row['content_ids']
        assert tok.encode(prefix+row['prompt']+assistant,add_special_tokens=False)==prefix_ids+row['content_ids']+row['suffix_ids']
        maxima['content_tokens']=max(maxima['content_tokens'],len(row['content_ids']))
        maxima['question_probes']=max(maxima['question_probes'],len(row['probe_positions']))
        prepared[row['split']].append(row)
    for task,path in [('arc',CFG['arc_input']),('hellaswag',CFG['hellaswag_input'])]:
        data=read(path);assert len(data['rows'])==CFG['test_counts'][task]
        for i,doc in enumerate(data['rows']):
            prompt,candidates,gold,kind,_=SC.format_item(task,doc)
            prepared['test'].append(dict(id=f'{task}_{i}',task=task,prompt=prompt,candidates=candidates,gold=gold,normalize=kind=='continuation',continuation_suffix=assistant+SC.ANSWER_CUE[kind],question=doc['question'] if task=='arc' else doc['ctx']))
    gsm=[r for r in read_jsonl(CFG['historical_test_rows']) if r['task']=='gsm8k'];assert len(gsm)==32
    for i,r in enumerate(gsm):
        question=r['prompt'].split('\n\nProblem: ',1)[1]
        prompt,_,gold,_,_=SC.format_item('gsm8k',dict(question=question,answer='#### '+str(r['gold'])))
        assert prompt==r['prompt'] and SC.normalize_number(gold)==SC.normalize_number(r['gold'])
        prepared['test'].append(dict(id=r['id'],task='gsm8k',prompt=prompt,gold=r['gold'],question=question,continuation_suffix=assistant))
    hotpot=read_jsonl(CFG['hotpot_test_rows']);assert len(hotpot)==512
    for r in hotpot:
        prepared['test'].append(dict(id=r['id'],source_id=r['source_id'],task='hotpotqa',prompt=r['prompt'],gold=r['answer'],question=r['question'],paragraph_count=r['paragraph_count'],continuation_suffix=assistant))
    cal_questions={norm(r['question']) for r in cal};cal_ids={r.get('hotpot_id') for r in cal if r.get('hotpot_id')}
    for row in prepared['test']:
        assert norm(row['question']) not in cal_questions,('Calibration/test question overlap',row['id'])
        assert row.get('source_id') not in cal_ids or row.get('source_id') is None
        row['content_ids']=tok.encode(row['prompt'],add_special_tokens=False)
        row['suffix_ids']=tok.encode(row['continuation_suffix'],add_special_tokens=False)
        assert tok.encode(prefix+row['prompt']+row['continuation_suffix'],add_special_tokens=False)==prefix_ids+row['content_ids']+row['suffix_ids']
    assert Counter(r['task'] for r in prepared['test'])==Counter(CFG['test_counts'])
    assert not {norm(r['question']) for r in prepared['train']} & {norm(r['question']) for r in prepared['validation']}
    for s,rr in prepared.items():write_jsonl(ROOT/'prepared'/(s+'.jsonl'),rr)
    sources=[CFG[k] for k in ['recolor','historical_heterofold','calibration','arc_input','hellaswag_input','historical_test_rows','hotpot_test_rows']]
    references={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'reference').rglob('*.py')}
    value=dict(config=CFG,config_sha256=sha(ROOT/'config.json'),code={p.name:sha(p) for p in ROOT.glob('*.py')},references=references,
        sources={str(p):sha(p) for p in sources},prepared={s:sha(ROOT/'prepared'/(s+'.jsonl')) for s in prepared},
        counts={s:len(rr) for s,rr in prepared.items()},geometry=geometry,prefix_ids=prefix_ids,prefix=prefix,assistant=assistant,
        calibration_train_counts=original['counts'],maxima=dict(maxima),stage1_fit_prompts=1600,
        calibration_test_id_overlap=0,calibration_test_question_overlap=0,
        stage2_validation_selection='fixed800 primary; own validation best diagnostic',training_order=CFG['training_order'])
    save(ROOT/'manifest.json',value);mark('prepared/completed.json',status='completed',counts=value['counts'],maxima=dict(maxima));log('PREPARED',value['counts'],dict(maxima))


def verify():
    m=read(ROOT/'manifest.json');assert sha(ROOT/'config.json')==m['config_sha256']
    for name,d in m['code'].items():assert sha(ROOT/name)==d,name
    for name,d in m['references'].items():assert sha(ROOT/name)==d,name
    for path,d in m['sources'].items():assert sha(path)==d,path
    for s,d in m['prepared'].items():assert sha(ROOT/'prepared'/(s+'.jsonl'))==d,s


def clone_cache(cache,model):
    return DynamicCache(ddp_cache_data=[(k.detach().clone(),v.detach().clone()) for k,v in [_cache_layer(cache,l) for l in range(36)]],config=model.config)


class Engine:
    def __init__(self):
        self.tok,self.model=load_model(CFG['models']['receiver'],'cuda',dtype=torch.float16)
        self.sender_tok,self.sender=load_model(CFG['models']['sender'],'cuda',dtype=torch.float16)
        self.base_map=load_map(CFG['recolor'])
        self.builder=HC.ExampleBuilder((self.sender_tok,self.sender),(self.tok,self.model),self.base_map,'cuda')
        self.mapping=self.builder.mapping
        self.prefix_ids=read(ROOT/'manifest.json')['prefix_ids'];assert self.prefix_ids==self.builder.prefix_ids.tolist()
        with torch.no_grad():
            ids=torch.tensor([self.prefix_ids],device='cuda')
            self.header=decoder(self.model)(ids,use_cache=True).past_key_values
        self.prefix_keys=[_cache_layer(self.header,l)[0][0].transpose(0,1).detach() for l in range(36)]
        self.prefix_values=[_cache_layer(self.header,l)[1][0].transpose(0,1).detach() for l in range(36)]
        self.snapshots=[versions(m) for m in [self.model,self.sender]]
        self.map_tensors=[x for role in 'kv' for name in ['A_'+role,'b_'+role] for x in self.mapping[name]]
        self.map_versions=[x._version for x in self.map_tensors]
        self.mode='train';self.student_test_native_prefills=0;self.norm_counts=None;self.norm_hooks=[]
        for l,block in enumerate(layers_of(self.model)):
            def count(module,args,index=l):
                if self.norm_counts is not None:self.norm_counts[index]+=1
            self.norm_hooks.append(block.self_attn.k_norm.register_forward_pre_hook(count))
        self.runtime=RT.Runtime.__new__(RT.Runtime);self.runtime.model=self.model;self.runtime.tokenizer=self.tok
        self.runtime.prefix,self.runtime.assistant=chat_affixes(self.tok,thinking=False)
        self.runtime.mode='native'

    def unchanged(self):
        assert [versions(m) for m in [self.model,self.sender]]==self.snapshots
        assert [x._version for x in self.map_tensors]==self.map_versions
        assert all(not p.requires_grad and p.grad is None for m in [self.model,self.sender] for p in m.parameters())

    @torch.no_grad()
    def base(self,row):
        encoding=self.tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        assert encoding['input_ids']==row['content_ids']
        k,v=self.builder.recolor_outputs(row['prompt'],encoding['offset_mapping'])
        assert k.shape==v.shape==(36,len(row['content_ids']),1024)
        assert torch.isfinite(k).all() and torch.isfinite(v).all()
        return k,v

    @torch.no_grad()
    def native_cache(self,row,*,baseline=False):
        if self.mode=='test' and not baseline:
            self.student_test_native_prefills+=1;raise RuntimeError('Native body prefill forbidden for student test')
        ids=torch.tensor([self.prefix_ids+row['content_ids']],device='cuda')
        return decoder(self.model)(ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(ids.shape[1],device='cuda')[None],use_cache=True).past_key_values

    @torch.no_grad()
    def attention_example(self,row,base):
        p=len(self.prefix_ids);probes=torch.tensor(row['probe_positions'],device='cuda')+p
        q,a,o,v=self.builder.native_targets(torch.tensor(row['content_ids']),probes)
        return HC.CalibrationExample(prefix_length=p,content_length=len(row['content_ids']),probe_positions=probes,
            base_key=base[0],base_value=base[1],native_query=q,native_attention=a,native_output=o,native_value=v)

    def cache(self,row,base,corrections):
        p=len(self.prefix_ids);positions=torch.arange(p,p+len(row['content_ids']),device='cuda')
        tables=_rotary_tables(self.model,positions);data=[]
        for l in range(36):
            k,v=base[0][l].float(),base[1][l].float()
            if corrections is not None:k,v=corrections[l].key(k),corrections[l].value(v)
            cosine,sine=tables(l);k,v=finalize_layer(self.model,l,k,v,cosine,sine)
            hk,hv=_cache_layer(self.header,l)
            data.append((torch.cat([hk.detach(),k[None]],2),torch.cat([hv.detach(),v[None]],2)))
        return DynamicCache(ddp_cache_data=data,config=self.model.config)

    def forward(self,ids,cache,past_length,*,use_cache=True):
        return decoder(self.model)(torch.tensor([ids],device='cuda'),past_key_values=cache,
            attention_mask=torch.ones(1,past_length+len(ids),device='cuda',dtype=torch.long),
            position_ids=torch.arange(past_length,past_length+len(ids),device='cuda')[None],use_cache=use_cache)

    def logits(self,row,cache,prefix=(),positions=(0,)):
        ids=row['suffix_ids']+list(prefix);length=len(self.prefix_ids)+len(row['content_ids'])
        out=self.forward(ids,cache,length,use_cache=False)
        ix=torch.tensor([len(row['suffix_ids'])-1+i for i in positions],device='cuda')
        return self.model.lm_head(out.last_hidden_state[0,ix]).float()

    @torch.no_grad()
    def rollout(self,row,cache,cap=64):
        length=len(self.prefix_ids)+len(row['content_ids'])
        out=self.forward(row['suffix_ids'],cache,length);total=length+len(row['suffix_ids']);tokens=[]
        eos=self.model.generation_config.eos_token_id;stops=set(eos if isinstance(eos,list) else [eos])
        stops.update([self.tok.eos_token_id,self.tok.convert_tokens_to_ids('<|im_end|>')]);stops.discard(None)
        for i in range(cap):
            t=int(self.model.lm_head(out.last_hidden_state[0,-1]).float().argmax());tokens.append(t)
            if t in stops:break
            if i+1<cap:out=self.forward([t],out.past_key_values,total);total+=1
        return tokens


def positions_for(tokens,row,epoch,validation=False):
    n=min(len(tokens),CFG['supervised_positions_max'])
    if n==1:return [0]
    chosen={0,len(tokens)-1};remaining=[i for i in range(len(tokens)) if i not in chosen]
    rng=random.Random(int(digest(dict(id=row['id'],epoch=0 if validation else epoch,seed=CFG['seed']))[:16],16));rng.shuffle(remaining)
    chosen.update(remaining[:n-len(chosen)]);return sorted(chosen)


@torch.no_grad()
def native_prefix(engine,row,native):
    path=ROOT/'native_prefix'/(row['id']+'.json')
    if path.exists():
        x=read(path);assert x['manifest_sha256']==identity();return x['tokens']
    tokens=engine.rollout(row,clone_cache(native,engine.model),CFG['prefix_max_new_tokens'])
    mark(str(path.relative_to(ROOT)),tokens=tokens,id=row['id']);return tokens


def vocab_kl(teacher,student):
    target=teacher.float().log_softmax(-1).detach();pred=student.float().log_softmax(-1)
    return F.kl_div(pred,target,reduction='none',log_target=True).sum(-1).mean()


def kl_loss(engine,corrections,row,epoch,online=False,validation=False,base=None,native=None):
    if base is None:base=engine.base(row)
    if native is None:native=engine.native_cache(row)
    with torch.no_grad():
        tokens=engine.rollout(row,engine.cache(row,base,corrections),CFG['prefix_max_new_tokens']) if online else native_prefix(engine,row,native)
    positions=positions_for(tokens,row,epoch,validation);prefix=tokens[:max(positions)]
    with torch.no_grad():teacher=engine.logits(row,clone_cache(native,engine.model),prefix,positions)
    student=engine.logits(row,engine.cache(row,base,corrections),prefix,positions)
    loss=vocab_kl(teacher,student);assert torch.isfinite(loss)
    return loss,dict(full_kl=float(loss.detach()),next_token_agreement=float((teacher.argmax(-1)==student.argmax(-1)).float().mean()),positions=len(positions),rollout_tokens=len(tokens),online=online)


def attention_objective(engine,corrections,example,*,scaler=None,divisor=1):
    """Official objective formula and stop-grad; shared deployment finalization."""
    total=dict(routing_kl=0.,k_post_wo_mse=0.,post_wo_nmse=0.)
    length=example.prefix_length+example.content_length
    positions=torch.arange(example.prefix_length,length,device='cuda');tables=_rotary_tables(engine.model,positions)
    for l,block in enumerate(layers_of(engine.model)):
        k,v=example.base_key[l].float(),example.base_value[l].float()
        if corrections is not None:k,v=corrections[l].key(k),corrections[l].value(v)
        cosine,sine=tables(l);k,v=finalize_layer(engine.model,l,k,v,cosine,sine)
        keys=torch.cat([engine.prefix_keys[l],k.transpose(0,1)],0)
        values=torch.cat([engine.prefix_values[l],v.transpose(0,1)],0)
        attn=block.self_attn;mask=causal_mask(attn,example.probe_positions,length)
        prob=attention_probabilities(attn,example.native_query[l],keys,mask,4)
        target=example.native_attention[l].float().clamp_min(0);target=target/target.sum(-1,keepdim=True).clamp_min(1e-20)
        routing=(target*(target.clamp_min(1e-12).log()-prob.float().clamp_min(1e-20).log())).sum(-1).mean()
        native_v=example.native_value[l].float().repeat_interleave(4,dim=1)
        context_error=torch.einsum('hpt,thd->phd',prob.float()-target,native_v)
        projection=output_projection(attn);wo=projection.weight.float().reshape(engine.model.config.hidden_size,32,128).permute(1,0,2)
        key_output=torch.einsum('phd,hmd->phm',context_error,wo).square().mean()
        context=torch.einsum('hpt,thd->phd',prob.detach(),values.repeat_interleave(4,dim=1)).reshape(example.native_query[l].shape[0],4096)
        mapped=projection(context).float();native_o=example.native_output[l].float()
        value_output=(mapped-native_o).square().sum()/native_o.square().sum().clamp_min(1e-8)
        loss=routing+CFG['lambda_k_output']*key_output+value_output
        assert torch.isfinite(loss)
        if scaler is not None:scaler.scale(loss/(36*divisor)).backward()
        for name,value in [('routing_kl',routing),('k_post_wo_mse',key_output),('post_wo_nmse',value_output)]:total[name]+=float(value.detach())/36
        del k,v,keys,values,prob,target,context_error,context,mapped,loss
    total['attention_loss']=total['routing_kl']+CFG['lambda_k_output']*total['k_post_wo_mse']+total['post_wo_nmse']
    return total


def update(module,optimizer,scaler):
    scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_(module.parameters(),CFG['gradient_clip'])
    assert torch.isfinite(grad) and grad>0
    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);return float(grad)


def paired(values):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(CFG['seed'])
    boot=x[rng.integers(0,len(x),size=(2000,len(x)))].mean(1)
    return dict(n=len(x),mean=float(x.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:raise ValueError(sys.argv[1])
