"""Exact completed Gemma Recolor+C engine; existing cache read-only, no translator retraining."""
from experiment_common import *
PE=parent_module('gemma_partial_reference_engine','engine_new.py')

class Engine(PE.Engine):
    @torch.no_grad()
    def base(self,row):
        key=C.digest(dict(prompt=row['prompt'],ids=row['content_ids']));path=self.cache_root/(key+'.pt')
        if path.exists():return super().base(row)
        # Cache misses use the reference RAW projection mapping exactly; no
        # extra disk cache is created on the nearly full server.
        self.cache_misses+=1;k,v=self.compute_base(row)
        assert k.shape==v.shape==(36,len(row['content_ids']),1024) and k.dtype==v.dtype==torch.float16
        assert torch.isfinite(k).all() and torch.isfinite(v).all();return k,v
