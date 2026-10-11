"""Float64 reductions; all inputs [layers, tokens, heads, dim]."""
import torch

def compare(m,g,r):
    assert m.shape==g.shape==r.shape and m.ndim==4 and m.shape[1]>0
    # Chunk outputs moved to CPU; 512-token input remains modest. FP64 is CPU only.
    m,g,r=(x.detach().cpu().double() for x in (m,g,r))
    assert all(torch.isfinite(x).all() for x in (m,g,r))
    axes=(1,3)
    norm=lambda x:x.square().sum(axes)
    dot=lambda x,y:(x*y).sum(axes)
    cosine=lambda x,y:dot(x,y)/(norm(x)*norm(y)).sqrt().clamp_min(1e-20)
    em,eg=m-r,g-r
    matrices={
        'mg_cosine':cosine(m,g),
        'mg_symmetric_nmse':norm(m-g)/((norm(m)+norm(g))/2).clamp_min(1e-20),
        'mcq_native_nmse':norm(em)/norm(r).clamp_min(1e-20),
        'gsm8k_native_nmse':norm(eg)/norm(r).clamp_min(1e-20),
        'mcq_native_cosine':cosine(m,r),
        'gsm8k_native_cosine':cosine(g,r),
        'error_cosine':cosine(em,eg),
        'error_zero_norm_fraction':((norm(em)<1e-20)|(norm(eg)<1e-20)).double(),
        'mg_vector_cosine_mean':torch.nn.functional.cosine_similarity(m,g,dim=-1,eps=1e-20).mean(1)
    }
    n=lambda x:x.square().sum()
    c=lambda x,y:float((x*y).sum()/(n(x)*n(y)).sqrt().clamp_min(1e-20))
    scalar={'mg_cosine':c(m,g),'mg_symmetric_nmse':float(n(m-g)/((n(m)+n(g))/2).clamp_min(1e-20)),
        'mcq_native_nmse':float(n(em)/n(r).clamp_min(1e-20)),
        'gsm8k_native_nmse':float(n(eg)/n(r).clamp_min(1e-20)),
        'mcq_native_cosine':c(m,r),'gsm8k_native_cosine':c(g,r),'error_cosine':c(em,eg),
        'error_zero_norm_fraction':float(matrices['error_zero_norm_fraction'].mean()),
        'mg_vector_cosine_mean':float(matrices['mg_vector_cosine_mean'].mean())}
    return scalar,{k:v.tolist() for k,v in matrices.items()}

def tests():
    torch.manual_seed(1234)
    r=torch.randn(3,7,2,4)
    s,h=compare(r,r,r)
    assert abs(s['mg_cosine']-1)<1e-10 and s['mg_symmetric_nmse']==0
    assert s['mcq_native_nmse']==0 and s['error_zero_norm_fraction']==1
    s,_=compare(2*r,r,r)
    assert abs(s['mg_symmetric_nmse']-.4)<1e-10 and abs(s['mcq_native_nmse']-1)<1e-10
    e=torch.randn_like(r)
    s,_=compare(r+e,r-e,r)
    assert abs(s['error_cosine']+1)<1e-10
    t,_=compare(r-e,r+e,r)
    assert abs(s['mg_symmetric_nmse']-t['mg_symmetric_nmse'])<1e-10
    print('METRIC TESTS PASSED',flush=True)

if __name__=='__main__':tests()
