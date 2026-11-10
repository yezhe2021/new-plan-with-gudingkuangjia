"""Existing RNKV construction reads the current full frozen translated body."""
from experiment_common import *

class DynamicMemory(M.RNKV):
    def __init__(self,engine,config=None):
        cfg=dict(CFG if config is None else config);cfg['replace_layers']=list(range(1,37))
        super().__init__(engine,cfg)
        self.routing.close();self.routing=AR.KeepBothRouting(engine,self.memory,cfg['replace_layers'])
    # Inherit the original full-body construct(), logits(), rollout() and score().
    # No canonical fixed-header branch or static K relocation is used here.
