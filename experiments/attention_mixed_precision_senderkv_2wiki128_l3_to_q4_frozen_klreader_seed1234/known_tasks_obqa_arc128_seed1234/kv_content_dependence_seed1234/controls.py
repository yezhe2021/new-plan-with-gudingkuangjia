"""Deterministic controls; no gold-dependent randomization."""
import hashlib,json,re
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def seed_for(*parts):
    return int(hashlib.sha256(json.dumps(parts,ensure_ascii=False).encode()).hexdigest()[:15],16)


def option_block(prompt,labels):
    head,block=prompt.split('\n\nChoices:\n',1)
    choices,tail=block.rsplit('\n\nAnswer with only the option label.',1)
    assert len(set(labels))==len(labels)
    pattern=r'^('+ '|'.join(re.escape(label) for label in labels)+r')\. '
    matches=list(re.finditer(pattern,choices,re.M))
    assert [m.group(1) for m in matches]==labels
    texts=[choices[m.end():matches[i+1].start() if i+1<len(matches) else len(choices)].rstrip('\n') for i,m in enumerate(matches)]
    rebuilt=head+'\n\nChoices:\n'+'\n'.join(label+'. '+text for label,text in zip(labels,texts))+'\n\nAnswer with only the option label.'+tail
    assert rebuilt==prompt and all(texts)
    return head,texts,tail


def permute_options(prompt,labels,seed):
    head,texts,tail=option_block(prompt,labels)
    rng=np.random.default_rng(seed);n=len(labels)
    assert n>=2
    while True:
        perm=rng.permutation(n).tolist()
        if all(i!=v for i,v in enumerate(perm)):break
    changed=head+'\n\nChoices:\n'+'\n'.join(labels[i]+'. '+texts[old] for i,old in enumerate(perm))+'\n\nAnswer with only the option label.'+tail
    return changed,perm,texts


def donor_maps(rows,repeats,seed):
    by_task=defaultdict(list)
    for r in rows:by_task[r['task']].append(r)
    result={r['id']:[] for r in rows}
    for task,task_rows in by_task.items():
        groups=defaultdict(list)
        for r in task_rows:groups[len(r['candidates'])].append(r)
        previous=defaultdict(set)
        for rep in range(repeats):
            for count,group in groups.items():
                rng=np.random.default_rng(seed_for(seed,task,count,rep));n=len(group)
                if n==1:
                    row=group[0]
                    pool=[r for r in task_rows if r['question_key']!=row['question_key']]
                    pool.sort(key=lambda d:(abs(len(d['source_indices'])-len(row['source_indices'])),d['id']))
                    available=[r for r in pool[:8] if r['id'] not in previous[row['id']]] or pool[:8]
                    donor=available[int(rng.integers(len(available)))]
                    assignments=[(row,donor)]
                else:
                    lengths=np.array([len(r['source_indices']) for r in group])
                    cost=np.abs(lengths[:,None]-lengths[None,:]).astype(float)+rng.uniform(0,.5,(n,n))
                    for i,row in enumerate(group):
                        for j,donor in enumerate(group):
                            if row['question_key']==donor['question_key']:cost[i,j]=1e9
                            elif donor['id'] in previous[row['id']]:cost[i,j]+=4
                    left,right=linear_sum_assignment(cost)
                    assignments=[(group[i],group[j]) for i,j in zip(left,right)]
                for row,donor in assignments:
                    assert row['id']!=donor['id'] and row['question_key']!=donor['question_key']
                    previous[row['id']].add(donor['id'])
                    result[row['id']].append(dict(id=donor['id'],length=len(donor['source_indices']),
                        target_length=len(row['source_indices']),same_option_count=len(donor['candidates'])==len(row['candidates']),
                        exact_length=len(donor['source_indices'])==len(row['source_indices'])))
    assert all(len(v)==repeats for v in result.values())
    return result


def resize_pair(key,value,length):
    assert key.shape==value.shape and key.ndim==4 and length>0
    n=key.shape[1]
    indices=((2*torch.arange(length)+1)*n//(2*length)).clamp_max(n-1)
    return key[:,indices].contiguous(),value[:,indices].contiguous(),indices.tolist()


def matched_noise(key,value,seed):
    outputs=[]
    for role,x in enumerate([key,value]):
        x=x.detach().cpu();xf=x.float();mu=xf.mean(1,keepdim=True);sd=xf.std(1,unbiased=False,keepdim=True)
        gen=torch.Generator(device='cpu').manual_seed(seed_for(seed,role))
        z=torch.randn(x.shape,generator=gen,dtype=torch.float32)
        z=(z-z.mean(1,keepdim=True))/z.std(1,unbiased=False,keepdim=True).clamp_min(1e-20)
        out=(mu+sd*z).to(x.dtype)
        assert torch.isfinite(out).all();outputs.append(out)
    return tuple(outputs)


def visibility(anchor,body_length,continuation,blocked):
    assert 0<anchor<=body_length and continuation>0
    mask=torch.ones(1,body_length+continuation,dtype=torch.long)
    if blocked:mask[:,anchor:body_length]=0
    assert torch.all(mask[:,:anchor]==1) and torch.all(mask[:,body_length:]==1)
    return mask
