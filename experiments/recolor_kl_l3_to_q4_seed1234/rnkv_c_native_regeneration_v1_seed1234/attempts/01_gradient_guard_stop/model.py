"""Native Qwen memory generation followed by mask-based functional replacement."""
import random
from common import *
from memory import Memory
import cache as K
from attention import Routing
from torch.utils.checkpoint import checkpoint

class RNKV(torch.nn.Module):
    def __init__(self,engine,config=None):
        super().__init__();self.engine=engine;self.cfg=CFG if config is None else config
        self.memory=Memory(engine.model.config.hidden_size,self.cfg['memory_tokens'],self.cfg['adapter_rank'],self.cfg['adapter_after_layers'],self.cfg['seed']).to(next(engine.model.parameters()).device).float()
        self.routing=Routing(engine,self.memory,self.cfg['replace_layers']);self.activation_checkpointing=True

    @property
    def M(self):return self.cfg['memory_tokens']

    def construct(self,row,translated):
        L=translated.get_seq_length();header=len(self.engine.prefix_ids);model=self.engine.model
        frozen=K.flat(translated)
        def build(prompt):
            cache=K.from_values(frozen,model)
            q=torch.arange(L,L+self.M,device=prompt.device)
            with self.routing.context('construct',header,L,self.M,q):
                result=C.decoder(model)(inputs_embeds=prompt[None].to(model.dtype),past_key_values=cache,
                    attention_mask=torch.ones(1,L+self.M,device=prompt.device,dtype=torch.long),position_ids=q[None],use_cache=True)
            data=K.pairs(result.past_key_values)
            assert len(data)==model.config.num_hidden_layers
            assert all(k.shape[2]==L+self.M and v.shape[2]==L+self.M for k,v in data)
            return tuple(x[:,:,L:L+self.M,:].clone() for pair in data for x in pair)
        memories=checkpoint(build,self.memory.prompt,use_reentrant=False,preserve_rng_state=True) if self.activation_checkpointing and torch.is_grad_enabled() else build(self.memory.prompt)
        return K.append(translated,memories,model)

    def logits(self,row,cache,tokens=(),positions=(0,),*,rn=True):
        engine=self.engine;model=engine.model;length=cache.get_seq_length()
        ids=row['suffix_ids']+list(tokens)
        ix=torch.tensor([len(row['suffix_ids'])-1+i for i in positions],device=next(model.parameters()).device)
        L=len(engine.prefix_ids)+len(row['content_ids'])
        assert length==L+(self.M if rn else 0)
        def forward(*values):
            current=K.from_values(values,model)
            q=torch.arange(length,length+len(ids),device=ix.device)
            with self.routing.context('answer' if rn else 'plain',len(engine.prefix_ids),L,self.M,q):
                result=engine.forward(ids,current,length,use_cache=False)
                return model.lm_head(result.last_hidden_state[0,ix]).float()
        values=K.flat(cache)
        if self.activation_checkpointing and torch.is_grad_enabled() and any(x.requires_grad for x in values):
            return checkpoint(forward,*values,use_reentrant=False,preserve_rng_state=True)
        return forward(*values)

    @torch.no_grad()
    def rollout(self,row,cache,*,rn,cap=64):
        engine=self.engine;model=engine.model;L=len(engine.prefix_ids)+len(row['content_ids']);length=cache.get_seq_length()
        assert length==L+(self.M if rn else 0)
        ids=row['suffix_ids'];tokens=[];current=K.clone(cache,model,detach=True)
        eos=model.generation_config.eos_token_id;stops=set(eos if isinstance(eos,list) else [eos]);stops.add(engine.tok.convert_tokens_to_ids('<|im_end|>'))
        for i in range(cap):
            q=torch.arange(length,length+len(ids),device=next(model.parameters()).device)
            with self.routing.context('answer' if rn else 'plain',len(engine.prefix_ids),L,self.M,q):
                result=engine.forward(ids,current,length,use_cache=True)
                token=int(model.lm_head(result.last_hidden_state[0,-1]).float().argmax())
            tokens.append(token);length+=len(ids);current=result.past_key_values;ids=[token]
            if token in stops:break
        return tokens

    def token_logits(self,row,cache,tokens,*,rn):
        # Include the whole target, matching official candidate-scoring inputs;
        # the final hidden state is unscored and causality protects all targets.
        return self.logits(row,cache,tokens,range(len(tokens)),rn=rn)

    @torch.no_grad()
    def score(self,row,cache,*,rn):
        if row['task']=='arc':
            scores=[]
            for label in row['candidates']:
                tokens=self.engine.tok.encode(label,add_special_tokens=False)
                logits=self.token_logits(row,cache,tokens,rn=rn)
                target=torch.tensor(tokens,device=logits.device)
                scores.append(float(logits.log_softmax(-1).gather(-1,target[:,None]).sum()))
            pred=max(range(len(scores)),key=scores.__getitem__)
            return dict(scores=scores,prediction=pred,correct=int(pred==row['gold']))
        from metrics import clean_prediction,answer_metrics
        tokens=self.rollout(row,cache,rn=rn);text=self.engine.tok.decode(tokens,skip_special_tokens=False);pred=clean_prediction(text)
        return dict(prediction=pred,generation=text,generated_tokens=len(tokens),reached_cap=int(len(tokens)>=64),**answer_metrics(pred,row['gold']))

def load(which='initial'):
    engine=Engine();correction=T.load_final('C').requires_grad_(False)
    model=RNKV(engine);assert sum(p.numel() for p in model.parameters())==409600
    if which=='final':
        cp=torch.load(ROOT/'runs/epoch4_step1024.pt',map_location='cpu',weights_only=True)
        assert cp['manifest_sha256']==identity() and cp['step']==1024
        model.load_state_dict(cp['state'],strict=True)
    return engine,correction,model

@torch.no_grad()
def native_prefix(model,row,native):
    path=ROOT/'native_prefix'/(row['id']+'.json')
    if path.exists():
        x=C.read(path);assert x['manifest_sha256']==identity();return x['tokens']
    tokens=model.rollout(row,native,rn=False)
    mark(str(path.relative_to(ROOT)),id=row['id'],tokens=tokens)
    return tokens

def losses(engine,model,row,translated,native,prefix_kind,epoch):
    regenerated=model.construct(row,translated)
    target=row['gold_tokens'];gold=model.token_logits(row,regenerated,target,rn=True)
    task_loss=torch.nn.functional.cross_entropy(gold,torch.tensor(target,device=gold.device))
    with torch.no_grad():
        tokens=target if prefix_kind=='gold' else (native_prefix(model,row,native) if prefix_kind=='native' else model.rollout(row,regenerated,rn=True))
    assert tokens
    positions=C.positions_for(tokens,row,epoch,validation=epoch==0);prefix=tokens[:max(positions)]
    with torch.no_grad():teacher=model.logits(row,native,prefix,positions,rn=False)
    student=model.logits(row,regenerated,prefix,positions,rn=True)
    kl=C.vocab_kl(teacher,student);loss=task_loss+CFG['lambda_kl']*kl
    assert torch.isfinite(loss)
    return loss,dict(task_loss=float(task_loss.detach()),full_kl=float(kl.detach()),loss=float(loss.detach()),
        prefix_kind=prefix_kind,positions=len(positions),prefix_tokens=len(tokens),gold_tokens=len(target))

def batches(epoch):
    data=rows('train');bytask={t:[r for r in data if r['task']==t] for t in CFG['train_counts']}
    for i,t in enumerate(bytask):random.Random(CFG['seed']+epoch*17+i).shuffle(bytask[t])
    result=[];q=CFG['prefix_quotas_per_batch'][epoch-1]
    for b in range(256):
        batch=[r for t in bytask for r in bytask[t][b*4:b*4+4]]
        labels=[kind for kind,n in q.items() for _ in range(n)]
        random.Random(CFG['seed']+epoch*10000+b).shuffle(labels)
        result.append(list(zip(batch,labels)))
    assert len(result)==256 and all(len(x)==8 for x in result)
    return result
