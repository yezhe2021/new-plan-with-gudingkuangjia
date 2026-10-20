"""Lossless fixed-input caches, sequential model residency and no test capture."""
import sys
from core import *


@torch.no_grad()
def main(family,smoke=False):
    todo=[(s,r) for s in ['train','validation'] for r in rows(s,smoke) if not cache_path(s,r,family).exists()]
    if not todo:log(f'{family} cache already complete');return
    tok,m=model(family);rt=runtime(m,tok) if family=='qwen' else None
    for i,(split,row) in enumerate(todo,1):
        free=shutil.disk_usage(ROOT).free/2**30
        assert free>CFG['disk_reserve_gib'],f'Disk reserve reached: {free:.2f} GiB'
        status(status='running',phase=f'{family}_capture',sample=i,total=len(todo),split=split)
        ids=row['frames'][family]['body_ids']
        k,v,_=capture(m,ids,len(ids),[0])
        payload=dict(id=row['id'],manifest_sha256=identity(),dtype=str(k.dtype))
        assert k.dtype==v.dtype==torch.float16
        if family=='llama':
            ix=row['source_indices'];payload.update(key=k[:,ix].contiguous(),value=v[:,ix].contiguous())
        else:
            n=row['frames']['qwen']['native_prefix_length'];ix=row['target_indices']
            offset=len(row['frames']['qwen']['prefix_ids'])
            payload.update(key=k[:,[offset+x for x in ix]].contiguous(),value=v[:,[offset+x for x in ix]].contiguous(),
                anchor_k=k[:,:n].contiguous(),anchor_v=v[:,:n].contiguous())
            teacher,_=R.native_candidate_scores(rt,row['prompt'],row['candidates'],False,rt.assistant+row['cue'])
            payload['teacher_scores']=torch.tensor(teacher,dtype=torch.float32)
        save_tensor(cache_path(split,row,family),payload)
        if smoke:
            saved=state(split,row,family)
            for name,value in payload.items():
                if torch.is_tensor(value):assert torch.equal(saved[name],value),(row['id'],name,'Cache changed precision/value')
        if i%16==0 or smoke or i==len(todo):log(f'{family} capture {i}/{len(todo)} {split} {row["id"]}')
    if not smoke:save(ROOT/f'cache/{family}/completed.json',dict(status='completed',manifest_sha256=identity()))
    log(f'{family.upper()} CAPTURE COMPLETED smoke={smoke}')


if __name__=='__main__':
    try:main(sys.argv[1],len(sys.argv)>2 and sys.argv[2]=='smoke')
    except Exception as exc:fail(exc);raise
