"""Sender-only GPU phase: original Recolor outputs, no Qwen weights resident."""
import shutil
from types import SimpleNamespace
from experiment_common import *

def main():
    verify();assert done('cpu_checks.json');probe=len(sys.argv)>1 and sys.argv[1]=='probe'
    marker='base_cache/probe_completed.json' if probe else 'base_cache/completed.json'
    if done(marker):return
    budget=gpu_budget();st,sender=C.load_model(C.CFG['models']['sender'],'cuda',dtype=torch.float16)
    tok=C.AutoTokenizer.from_pretrained(C.CFG['models']['receiver'],local_files_only=True)
    assert C.sha(C.CFG['recolor'])==C.CFG['recolor_sha256']
    mapping=C.load_map(C.CFG['recolor']);builder=C.HC.ExampleBuilder((st,sender),(tok,SimpleNamespace(dtype=torch.float16)),mapping,'cuda')
    before=C.versions(sender);rr=C.read_jsonl(ROOT/'prepared/probe.jsonl') if probe else rows('train')+rows('validation')+rows('test')
    started=time.time();saved=0
    for i,row in enumerate(rr,1):
        if paused():raise SystemExit(75)
        path=cache_file(row)
        if path.exists():saved_base(row);continue
        assert shutil.disk_usage(ROOT).free>CFG['disk_reserve_gib']*2**30
        enc=tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True);assert enc['input_ids']==row['content_ids']
        with torch.no_grad():k,v=builder.recolor_outputs(row['prompt'],enc['offset_mapping'])
        assert torch.isfinite(k).all() and torch.isfinite(v).all() and k.shape==v.shape==(36,len(row['content_ids']),1024)
        C.atomic_save(dict(manifest_sha256=identity(),prompt_sha256=C.digest(dict(prompt=row['prompt'],content_ids=row['content_ids'])),recolor_sha256=C.CFG['recolor_sha256'],kv=(k.cpu(),v.cpu())),path)
        if probe:
            a,b=saved_base(row);assert torch.equal(a,k.cpu()) and torch.equal(b,v.cpu())
        del k,v;saved+=1
        status(status='running',phase='source_only_cache_probe' if probe else 'source_only_cache_full',sample=i,total=len(rr),new_entries=saved,elapsed_seconds=time.time()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,**budget)
        if i==1 or i%16==0:C.log('M64 SOURCE CACHE',i,len(rr),time.time()-started)
    assert C.versions(sender)==before and all(p.grad is None and not p.requires_grad for p in sender.parameters())
    for row in rr:assert cache_file(row).exists()
    mark(marker,status='completed',n=len(rr),new_entries=saved,elapsed_seconds=time.time()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,**budget)
if __name__=='__main__':main()
