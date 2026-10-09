"""CPU schema-only test; dummy predictions must never enter benchmark reports."""
import json
from lm_eval.api.model import LM
from lm_eval import evaluator
from tasks_native import install_local_data, ROOT

class Dummy(LM):
    def loglikelihood(self,requests,**kwargs): return [(-float(len(r.args[1])),False) for r in requests]
    def loglikelihood_rolling(self,requests,**kwargs): raise NotImplementedError
    def generate_until(self,requests,**kwargs): return ["#### 0" for r in requests]

install_local_data()
result=evaluator.simple_evaluate(model=Dummy(),tasks=["openbookqa","arc_challenge","gsm8k"],
                                 limit=1,bootstrap_iters=0,log_samples=True,
                                 random_seed=1234,numpy_random_seed=1234,torch_random_seed=1234,fewshot_random_seed=1234)
schema={"SCHEMA_ONLY_NOT_MODEL_RESULTS":True,"result_keys":list(result),
        "sample_keys":{k:list(v[0]) for k,v in result.get("samples",{}).items()},
        "sample_metric_types":{k:{name:type(value).__name__ for name,value in v[0].items()} for k,v in result.get("samples",{}).items()}}
(ROOT/"schema_test.json").write_text(json.dumps(schema,indent=2),encoding="utf-8")
print(json.dumps(schema),flush=True)
