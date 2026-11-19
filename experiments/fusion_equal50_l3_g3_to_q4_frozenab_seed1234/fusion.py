"""Token-local equal-weight fusion in Qwen's raw pre-RoPE KV space."""
import torch

def fuse(llama_key,llama_value,gemma_key,gemma_value,weight=0.5):
    if not 0<=weight<=1:raise ValueError('Weight must lie in [0,1]')
    if not (llama_key.shape==llama_value.shape==gemma_key.shape==gemma_value.shape):
        raise ValueError('Both sources must share [layers,tokens,heads,dim]')
    if llama_key.ndim!=4:raise ValueError('Expected [layers,tokens,heads,dim]')
    if len({x.device for x in [llama_key,llama_value,gemma_key,gemma_value]})!=1:raise ValueError('Devices differ')
    if len({x.dtype for x in [llama_key,llama_value,gemma_key,gemma_value]})!=1:raise ValueError('Dtypes differ')
    # FP32 arithmetic avoids two separate FP16 product roundings; store physical cache in original dtype.
    def average(a,b):return (weight*a.float()+(1-weight)*b.float()).to(a.dtype)
    return average(llama_key,gemma_key),average(llama_value,gemma_value)
