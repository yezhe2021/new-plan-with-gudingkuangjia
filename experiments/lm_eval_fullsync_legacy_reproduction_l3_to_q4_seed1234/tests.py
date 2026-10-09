"""Small mathematical and architecture invariants independent of GPU performance."""
import torch
import torch.nn.functional as F
from kv_backend import NativeKVTranslator, ResidualKVAdapter

torch.manual_seed(1234)
base=NativeKVTranslator(source_layers=3,target_layers=4,heads=2,dim=8,hidden_dim=16)
zero=torch.zeros(1,3,5,2,8)
k,v=base(zero,zero)
assert torch.count_nonzero(k)==0 and torch.count_nonzero(v)==0, "Writer(0) != 0"
assert all(m.bias is None for m in base.modules() if isinstance(m,torch.nn.Linear))
adapter=ResidualKVAdapter(layers=4,heads=2,dim=8,rank=4)
x,y=base(torch.randn_like(zero),torch.randn_like(zero))
a,b,dk,dv=adapter(x,y)
assert torch.equal(a,x) and torch.equal(b,y), "Residual not exact identity initially"
assert torch.count_nonzero(dk)==0 and torch.count_nonzero(dv)==0
teacher=torch.tensor([0.2,-1.1,2.3,0.4])
student=torch.tensor([-0.3,-1.0,1.4,0.8],requires_grad=True)
loss=F.kl_div(student.log_softmax(-1),teacher.softmax(-1),reduction="sum")
loss.backward()
assert torch.allclose(student.grad,student.detach().softmax(-1)-teacher.softmax(-1),atol=1e-6)
print("PASS: Writer zero, no bias, residual identity, exact streamed choice-KL gradient",flush=True)
