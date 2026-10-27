"""Frozen co-resident models; bounded CPU KV cache; receiver-observable features."""
import backend as B
from backend import CFG,E,torch,np,read,ROOT
from pilot_metrics import seed_for,normalize,answer_metrics
from collections import OrderedDict,Counter
import copy,time,math
import torch.nn.functional as F


def resized(k,v,n):
    ix=((2*torch.arange(n)+1)*k.shape[1]//(2*n)).clamp_max(k.shape[1]-1)
    return k[:,ix].contiguous(),v[:,ix].contiguous()


def perturb_positions(row,spec):
    n=len(row['source_indices']);count=max(1,int(round(n*spec['fraction'])))
    rng=np.random.default_rng(seed_for(CFG['seed'],row['id'],spec['kind'],spec['fraction'],spec['seed_index'],'perturb'))
    if spec['kind']=='random':ix=rng.choice(n,count,replace=False)
    else:
        start=int(rng.integers(0,n-count+1));ix=np.arange(start,start+count)
    anchor=row['frames']['qwen']['native_prefix_length'];cols=sorted((ix+anchor).tolist())
    assert len(cols)==count and min(cols)>=anchor and max(cols)<len(row['frames']['qwen']['body_ids'])
    return cols


def output_features(a,r):
    out={}
    for prefix,v in [('a',a),('r',r)]:
        for k in ['question_logp','first_entropy','mean_entropy','first_margin','mean_logp','generated_tokens','eos']:
            out[prefix+'_'+k]=float(v[k])
    out['question_gain']=a['question_logp']-r['question_logp'];out['entropy_gain']=r['first_entropy']-a['first_entropy']
    out['mean_logp_gain']=a['mean_logp']-r['mean_logp']
    aa=normalize(a['prediction']);rr=normalize(r['prediction'])
    out['answer_disagreement']=float(aa!=rr)
    common=sum((Counter(aa.split())&Counter(rr.split())).values());den=len(aa.split())+len(rr.split())
    out['answer_overlap']=2*common/den if den else 1.
    return out


def pair_features(a,r):
    out={}
    for branch,values in [('accept',a),('reject',r)]:
        for j,v in enumerate(values):
            for metric in ['mean','sum']:out[f'{branch}_candidate{j}_{metric}']=v[metric]
        out[branch+'_margin']=values[0]['mean']-values[1]['mean']
    out['margin_gain']=out['accept_margin']-out['reject_margin']
    out['accept_answer_support_gain']=a[0]['mean']-r[0]['mean']
    out['reject_answer_support_gain']=a[1]['mean']-r[1]['mean']
    return out


def stability_features(full,perturbed,count):
    subset=perturbed[:count];base=full[0]['mean']-full[1]['mean'];out={}
    groups={'all':subset,'random':[p for p in subset if p['spec']['kind']=='random'],'block':[p for p in subset if p['spec']['kind']=='block']}
    for name,group in groups.items():
        if not group:continue
        margins=np.array([p['scores'][0]['mean']-p['scores'][1]['mean'] for p in group])
        aa=np.array([p['scores'][0]['mean'] for p in group]);rr=np.array([p['scores'][1]['mean'] for p in group])
        values=dict(margin_mean=margins.mean(),margin_std=margins.std(),margin_min=margins.min(),margin_max=margins.max(),
            margin_shift=(margins-base).mean(),margin_abs_shift=np.abs(margins-base).mean(),
            sign_flip=np.mean((margins>0)!=(base>0)),accept_preference=np.mean(margins>0),
            accept_support_shift=(aa-full[0]['mean']).mean(),accept_support_std=aa.std(),
            reject_support_shift=(rr-full[1]['mean']).mean(),reject_support_std=rr.std())
        out.update({name+'_'+k:float(v) for k,v in values.items()})
    return out


class Engine:
    def __init__(self):
        assert torch.cuda.mem_get_info()[0]/2**30>=CFG['free_gpu_gib'],'GPU is not free enough to start'
        self.models={}
        for family in ['llama','qwen']:
            model=E.AutoModelForCausalLM.from_pretrained(E.CFG['models'][family],dtype=torch.float16,device_map='cuda',local_files_only=True,
                attn_implementation=CFG['attention_backend']).eval().requires_grad_(False)
            assert not any('LoRA' in type(m).__name__ for m in model.modules())
            self.models[family]=model;E.log('LOADED '+family+' frozen, no Reader')
        self.translator=B.load_translator();self.qwen=self.models['qwen'];self.qt=E.tok('qwen')
        eos=self.qwen.generation_config.eos_token_id;self.stops=set(eos if isinstance(eos,list) else [eos])
        self.stops.update([self.qt.eos_token_id,self.qt.convert_tokens_to_ids('<|im_end|>')]);self.stops.discard(None)
        self.anchor=self.native(read(ROOT/'manifest.json')['anchor_ids']);self.cpu_cache=OrderedDict();self.cache_bytes=0

    @torch.inference_mode()
    def forward(self,ids,cache=None,base=0,blocked=(),use_cache=True):
        t=torch.tensor([ids],device='cuda');mask=torch.ones(1,base+len(ids),device='cuda',dtype=torch.long)
        if len(blocked):mask[:,blocked]=0
        return self.qwen.model(t,past_key_values=cache,attention_mask=mask,
            position_ids=torch.arange(base,base+len(ids),device='cuda')[None],use_cache=use_cache)

    def native(self,ids):return self.forward(ids).past_key_values

    @torch.inference_mode()
    def mapped(self,row):
        rid=row['id']
        if rid in self.cpu_cache:self.cpu_cache.move_to_end(rid);return self.cpu_cache[rid]
        ids=row['frames']['llama']['body_ids'];k,v,_=E.C.capture(self.models['llama'],ids,len(ids),[0])
        ix=row['source_indices'];k=k[:,ix].contiguous();v=v[:,ix].contiguous();parts=[[],[]]
        for start in range(0,k.shape[1],64):
            with torch.amp.autocast('cuda',dtype=torch.float16):
                result=self.translator(k[:,start:start+64][None].cuda(),v[:,start:start+64][None].cuda())
            for j in range(2):parts[j].append(result[j][0].cpu().half())
        result=tuple(torch.cat(p,1).contiguous() for p in parts)
        assert all(torch.isfinite(x).all() for x in result)
        size=sum(x.numel()*x.element_size() for x in result)
        while self.cpu_cache and self.cache_bytes+size>CFG['cache_cpu_gib']*2**30:
            _,old=self.cpu_cache.popitem(last=False);self.cache_bytes-=sum(x.numel()*x.element_size() for x in old)
        self.cpu_cache[rid]=result;self.cache_bytes+=size;return result

    def variant(self,row,condition,own,donor):
        k,v=own;n=k.shape[1]
        if condition=='clean':return own
        if condition=='gaussian':
            result=[]
            for role,x in enumerate(own):
                xf=x.float();g=torch.Generator().manual_seed(seed_for(CFG['seed'],row['id'],role,'gaussian'))
                z=torch.randn(x.shape,generator=g);z=(z-z.mean(1,keepdim=True))/z.std(1,unbiased=False,keepdim=True).clamp_min(1e-20)
                result.append((xf.mean(1,keepdim=True)+xf.std(1,unbiased=False,keepdim=True)*z).half())
            return tuple(result)
        dk,dv=resized(*donor,n)
        if condition=='wrong':return dk,dv
        fraction={'partial25':.25,'partial50':.5,'partial75':.75}[condition]
        ix=np.random.default_rng(seed_for(CFG['seed'],row['id'],'corruption')).permutation(n)[:int(np.ceil(n*fraction))]
        pk,pv=k.clone(),v.clone();pk[:,ix]=dk[:,ix];pv[:,ix]=dv[:,ix];return pk,pv

    def cache(self,row,kv):return E.U.translated_cache(self.qwen,row,self.anchor,kv[0].cuda(),kv[1].cuda())

    def reject_columns(self,row):
        f=row['frames']['qwen'];return list(range(f['native_prefix_length'],len(f['body_ids'])))

    def prompt(self,row,cache,blocked=()):
        base=len(row['frames']['qwen']['body_ids'])
        out=self.forward(row['continuation_ids'],copy.deepcopy(cache),base,blocked)
        return out,base+len(row['continuation_ids'])

    @torch.inference_mode()
    def generate(self,row,cache=None,blocked=(),compact=False):
        start=time.time()
        if compact:out=self.forward(row['compact_ids']);total=len(row['compact_ids']);qlp=None
        else:
            out,total=self.prompt(row,cache,blocked);positions=row['question_positions']
            logits=self.qwen.lm_head(out.last_hidden_state[0,torch.tensor(positions,device='cuda')-1]).float()
            qlp=float(-F.cross_entropy(logits,torch.tensor([row['continuation_ids'][p] for p in positions],device='cuda')))
        logits=self.qwen.lm_head(out.last_hidden_state[0,-1]).float();top=logits.topk(2).values
        first_margin=float(top[0]-top[1]);memory=out.past_key_values;tokens=[];lps=[];entropies=[];eos=False
        for i in range(CFG['max_new_tokens']):
            lp=logits.log_softmax(-1);token=int(logits.argmax());tokens.append(token);lps.append(float(lp[token]));entropies.append(float(-(lp.exp()*lp).sum()))
            if token in self.stops:eos=True;break
            if i+1<CFG['max_new_tokens']:
                out=self.forward([token],memory,total,blocked);total+=1;memory=out.past_key_values;logits=self.qwen.lm_head(out.last_hidden_state[0,-1]).float()
        prediction=self.qt.decode(tokens,skip_special_tokens=True).strip()
        assert np.isfinite(lps).all()
        return dict(prediction=prediction,raw_output=self.qt.decode(tokens,skip_special_tokens=False),generated_ids=tokens,
            generated_tokens=len(tokens),eos=eos,question_logp=qlp,first_entropy=entropies[0],mean_entropy=float(np.mean(entropies)),
            first_margin=first_margin,mean_logp=float(np.mean(lps)),seconds=time.time()-start)

    def candidate_ids(self,generation):
        ids=list(generation['generated_ids'])
        if ids and ids[-1] in self.stops:ids=ids[:-1]
        # Empty generation is represented by its actual EOS, not by a gold-dependent fallback.
        return ids or [generation['generated_ids'][-1]]

    @torch.inference_mode()
    def pair_scores(self,row,cache,candidates,blocked=()):
        out,total=self.prompt(row,cache,blocked);memory=out.past_key_values
        initial=self.qwen.lm_head(out.last_hidden_state[0,-1]).float().log_softmax(-1)
        results=[];memo={}
        for ids in candidates:
            key=tuple(ids)
            if key in memo:results.append(memo[key]);continue
            scores=[float(initial[ids[0]])]
            if len(ids)>1:
                tail=self.forward(ids[:-1],copy.deepcopy(memory),total,blocked,use_cache=False)
                logits=self.qwen.lm_head(tail.last_hidden_state[0]).float()
                lps=-F.cross_entropy(logits,torch.tensor(ids[1:],device='cuda'),reduction='none');scores.extend(lps.cpu().tolist())
            result=dict(mean=float(np.mean(scores)),sum=float(np.sum(scores)),tokens=len(ids));memo[key]=result;results.append(result)
        assert all(math.isfinite(x[k]) for x in results for k in ['mean','sum']);return results

    @torch.inference_mode()
    def gold_nll(self,row,cache=None,blocked=(),compact=False):
        prefix=row['compact_ids'] if compact else row['continuation_ids'];base=0 if compact else len(row['frames']['qwen']['body_ids'])
        ids=prefix+row['gold_ids'][:-1];out=self.forward(ids,copy.deepcopy(cache),base,blocked,use_cache=False)
        logits=self.qwen.lm_head(out.last_hidden_state[0,len(prefix)-1:]).float()
        return float(F.cross_entropy(logits,torch.tensor(row['gold_ids'],device='cuda')))

    def annotate(self,row,prediction,cache=None,blocked=(),compact=False):
        return dict(prediction,**answer_metrics(prediction['prediction'],row['answer']),answer_token_nll=self.gold_nll(row,cache,blocked,compact))

    def extract(self,row,cache,reject_cache,accept,reject,audit=False):
        start=time.time();candidates=[self.candidate_ids(accept),self.candidate_ids(reject)]
        pair_a=self.pair_scores(row,cache,candidates);pair_r=self.pair_scores(row,reject_cache,candidates,self.reject_columns(row))
        perturb=[]
        for spec in CFG['perturbations']:
            cols=perturb_positions(row,spec);scores=self.pair_scores(row,cache,candidates,cols)
            record=dict(spec=spec,blocked_positions=cols,scores=scores)
            if audit:
                g=self.generate(row,cache,cols);record['generation']=g
                record['same_as_full']=normalize(g['prediction'])==normalize(accept['prediction'])
            perturb.append(record)
        groups=dict(output=output_features(accept,reject),cross=pair_features(pair_a,pair_r))
        groups.update({f'stability{n}':stability_features(pair_a,perturb,n) for n in [2,4,8]})
        assert all(math.isfinite(v) for d in groups.values() for v in d.values())
        return dict(feature_groups=groups,candidate_scores=dict(accept=pair_a,reject=pair_r),perturbations=perturb,
            feature_seconds=time.time()-start,forward_budget=dict(perturbations=len(perturb),candidate_sequences_per_probe=2),
            generation_audit=audit)
