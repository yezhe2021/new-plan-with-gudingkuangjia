"""Same wire representation as parent experiment, amortized fixed codebooks."""
import json,struct
import numpy as np
import torch
from quantization import MAGIC, parameters, codes, pack4


class FixedCodec:
    def __init__(self,key,value,group,positions,mapping):
        self.key,self.value=key.cpu().half().contiguous(),value.cpu().half().contiguous()
        self.shape=list(key.shape);self.group=group;self.positions=positions;self.mapping=mapping
        self.posbytes=np.asarray(positions,dtype='<u4').tobytes();self.mapbytes=np.asarray(mapping,dtype='<u4').tobytes()
        self.tables=[]
        for role,x in [('key',self.key),('value',self.value)]:
            p=parameters(x,role,group)
            self.tables.append((p.numpy().astype('<f2',copy=False).tobytes(),codes(x,p,role,group,15).numpy(),codes(x,p,role,group,255).numpy()))

    def encode(self,high_positions,bits):
        assert bits in [4,16,'mixed']
        n=self.shape[1];mask=np.zeros(n,dtype=bool);mask[high_positions]=True
        assert mask.sum()==len(high_positions) and (bits=='mixed' or not mask.any())
        sections=[np.packbits(mask,bitorder='little').tobytes(),self.posbytes,self.mapbytes]
        if bits==16:
            sections.extend(v.numpy().astype('<f2',copy=False).tobytes() for v in [self.key,self.value])
        else:
            for params,q4,q8 in self.tables:
                sections += [params,pack4(q4[:,~mask]),q8[:,mask].tobytes()]
        header=dict(version=1,shape=self.shape,bits=bits,group=self.group,target_tokens=len(self.mapping),lengths=[len(s) for s in sections])
        meta=json.dumps(header,separators=(',',':'),sort_keys=True).encode()
        return MAGIC+struct.pack('<I',len(meta))+meta+b''.join(sections)
