"""Weighted population moments; FP64 accumulation; explicit PSD stabilization."""
import torch


class Moments:
    def __init__(self,dx,dy):
        self.weight=0.; self.n=0
        self.sx=torch.zeros(dx,dtype=torch.float64); self.sy=torch.zeros(dy,dtype=torch.float64)
        self.xx=torch.zeros(dx,dx,dtype=torch.float64); self.yy=torch.zeros(dy,dy,dtype=torch.float64)
        self.xy=torch.zeros(dx,dy,dtype=torch.float64)

    def add(self,x,y,weight):
        x=x.double(); y=y.double(); w=torch.as_tensor(weight,dtype=torch.float64)
        if w.ndim==0: w=w.expand(len(x))
        assert len(x)==len(y)==len(w) and bool((w>=0).all())
        self.weight+=float(w.sum()); self.n+=len(x)
        self.sx+=(x*w[:,None]).sum(0); self.sy+=(y*w[:,None]).sum(0)
        self.xx+=x.T@(x*w[:,None]); self.yy+=y.T@(y*w[:,None]); self.xy+=x.T@(y*w[:,None])

    def covariance(self):
        if self.weight<=0: raise ValueError('Empty statistics')
        mx,my=self.sx/self.weight,self.sy/self.weight
        xx=self.xx/self.weight-torch.outer(mx,mx)
        yy=self.yy/self.weight-torch.outer(my,my)
        xy=self.xy/self.weight-torch.outer(mx,my)
        rms=(self.xx.diagonal().sum()/(self.weight*len(mx))).clamp_min(1e-30).sqrt()
        return mx/rms,my,xx/rms.square(),yy,xy/rms,rms


def roots(cov,relative_floor,absolute_floor):
    eigen,v=torch.linalg.eigh((cov+cov.T)/2)
    floor=max(absolute_floor,float(eigen.max())*relative_floor)
    clipped=eigen.clamp_min(floor)
    root=(v*clipped.sqrt())@v.T; inv=(v*clipped.rsqrt())@v.T
    return root,inv,{'min_eigenvalue':float(eigen.min()),'max_eigenvalue':float(eigen.max()),
        'floor':floor,'clipped_dimensions':int((eigen<floor).sum())}


def fit_recolor(stats,relative_floor=1e-6,absolute_floor=1e-10,ridge_lambda=.01):
    mx,my,cx,cy,cxy,rms=stats.covariance()
    _,ix,dx=roots(cx,relative_floor,absolute_floor)
    ry,iy,dy=roots(cy,relative_floor,absolute_floor)
    u,_,vh=torch.linalg.svd(ix@cxy@iy,full_matrices=False)
    a=ix@(u@vh)@ry; b=my-mx@a
    ridge=torch.linalg.solve(cx+ridge_lambda*torch.eye(len(mx),dtype=cx.dtype),cxy)
    bias_ridge=my-mx@ridge
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    report={'source':dx,'target':dy,'rms':float(rms),'aligned_tokens':stats.n,
        'mean_max_error':float((mx@a+b-my).abs().max()),
        'covariance_relative_error':float(torch.linalg.vector_norm(a.T@cx@a-cy)/torch.linalg.vector_norm(cy).clamp_min(1e-30))}
    return {'A':a.float(),'b':b.float(),'rms':rms.float()}, {'A':ridge.float(),'b':bias_ridge.float(),'rms':rms.float()}, report
