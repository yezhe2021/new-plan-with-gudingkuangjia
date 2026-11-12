"""Original Sync units + ordered Copy/Drop, adapted to complete RAW bodies."""
import math
from bisect import bisect_right
from dataclasses import dataclass
from reference.legacy_offsets import TokenSpan
from reference.legacy_units import build_synchronized_units,char_byte_boundaries,SyncUnit,byte_tokens

def synchronized_units(text,source_spans,target_spans):
    """Same legacy units, indexed by monotone ends to avoid quadratic scans."""
    cb=char_byte_boundaries(text);bc={b:i for i,b in enumerate(cb)}
    source,target=byte_tokens(source_spans,cb),byte_tokens(target_spans,cb)
    se=[x.end for x in source];te=[x.end for x in target]
    if se!=sorted(se) or te!=sorted(te):raise ValueError('Nonmonotone tokenizer endpoints')
    shared=sorted(set(se)&set(te)|{0});out=[]
    for left,right in zip(shared,shared[1:]):
        si=tuple(x.index for x in source[bisect_right(se,left):bisect_right(se,right)])
        ti=tuple(x.index for x in target[bisect_right(te,left):bisect_right(te,right)])
        if not si or not ti:continue
        cs,ce=bc[left],bc[right]
        out.append(SyncUnit(len(out),left,right,cs,ce,text[cs:ce],si,ti,si[-1],ti[-1]))
    if not out:raise ValueError('No synchronized units')
    return out

def source_rank(m,n,target_rank):
    if m<1 or n<1 or not 0<=target_rank<n:raise ValueError((m,n,target_rank))
    return math.ceil((target_rank+1)*m/n)-1 if m>n else target_rank*m//n

@dataclass(frozen=True)
class Alignment:
    source_indices: tuple
    source_ids: tuple
    receiver_ids: tuple
    audit: dict

def spans(text,encoded):
    # The legacy unit builder reserves index0 for a zero-width BOS. A synthetic
    # sentinel preserves that contract without adding any actual model token.
    ids=encoded['input_ids'];offsets=encoded['offset_mapping']
    out=[TokenSpan(0,-1,0,0)]
    for i,(token,(a,b)) in enumerate(zip(ids,offsets),1):
        if not 0<=a<b<=len(text):raise ValueError(('Unmappable RAW body token',i-1,token,a,b))
        out.append(TokenSpan(i,int(token),int(a),int(b)))
    if len(out)!=len(ids)+1:raise ValueError('Token/offset count mismatch')
    return out

def fullsync(text,sender_tok,receiver_tok,expected_receiver_ids=None):
    if not text:raise ValueError('Empty body')
    se=sender_tok(text,add_special_tokens=False,return_offsets_mapping=True)
    re=receiver_tok(text,add_special_tokens=False,return_offsets_mapping=True)
    if expected_receiver_ids is not None and list(re['input_ids'])!=list(expected_receiver_ids):raise ValueError('Receiver token-grid mismatch')
    ss,rs=spans(text,se),spans(text,re);units=synchronized_units(text,ss,rs)
    mapping={};lookahead=[];max_unit=0;char_bytes=char_byte_boundaries(text)
    for unit in units:
        source,target=unit.llama_indices,unit.qwen_indices
        max_unit=max(max_unit,unit.byte_end-unit.byte_start)
        for rank,index in enumerate(target):
            if index in mapping:raise ValueError('Receiver token mapped twice')
            selected=source[source_rank(len(source),len(target),rank)]
            assert ss[selected].end<=unit.char_end and rs[index].end<=unit.char_end
            mapping[index]=selected
            lookahead.append(max(0,char_bytes[ss[selected].end]-char_bytes[rs[index].end]))
    if set(mapping)!=set(range(1,len(rs))):raise ValueError('Full-Sync failed to cover every Receiver body token')
    if units[-1].char_end!=len(text):raise ValueError('Full-Sync omitted text suffix')
    indices=tuple(mapping[i]-1 for i in range(1,len(rs)))
    if list(indices)!=sorted(indices) or min(indices)<0 or max(indices)>=len(se['input_ids']):raise ValueError('Invalid ordered mapping')
    return Alignment(indices,tuple(se['input_ids']),tuple(re['input_ids']),dict(units=len(units),source_tokens=len(se['input_ids']),receiver_tokens=len(re['input_ids']),full_coverage=True,max_unit_bytes=max_unit,intra_unit_lookahead_tokens=sum(x>0 for x in lookahead),max_intra_unit_lookahead_bytes=max(lookahead,default=0),causality='shared-unit-boundary; not guaranteed strict per target token'))
