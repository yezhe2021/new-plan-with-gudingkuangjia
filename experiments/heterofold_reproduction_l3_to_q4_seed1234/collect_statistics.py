import gc
import torch
from common import CFG,load_model,store,save
from capture_kv import capture,protocol
from layer_alignment import features
from recolor import Moments,fit_recolor


def paired(row,st,rt,probes=False,keep_receiver=False):
    """Sequential models: bounded one-example CPU KVs, no persistent KV corpus."""
    proto=protocol(st,rt,row)
    sender=load_model('sender'); source=capture(sender,proto['source']['ids'])
    del sender; gc.collect(); torch.cuda.empty_cache()
    receiver=load_model('receiver')
    native=capture(receiver,proto['target']['ids'],proto['target']['probe_positions'] if probes else ())
    source={role:source[role][:,proto['source']['head']:][:,proto['mapping']] for role in ('k','v')}
    if keep_receiver: return proto,source,native,receiver
    del receiver; gc.collect(); torch.cuda.empty_cache()
    return proto,source,native


def collect_paired_statistics(prepared,st,rt,folder):
    # ~8GiB FP64 sufficient statistics, NOT all calibration KV tensors.
    stats={role:[Moments(3072,1024) for _ in range(36)] for role in ('k','v')}
    for number,row in enumerate(prepared['train'],1):
        proto,source,native=paired(row,st,rt)
        weight=1/len(prepared['token_totals']['train'])/prepared['token_totals']['train'][row['corpus']]
        for role in ('k','v'):
            for layer in range(36):
                x=features(source[role],layer,36)
                y=native[role][layer,proto['target']['head']:].flatten(1)
                stats[role][layer].add(x,y,weight)
        print('MOMENTS',number,len(prepared['train']),flush=True)
        del source,native
    recolor={}; ridge={}; reports={}
    for role in ('k','v'):
        recolor[role]=[]; ridge[role]=[]; reports[role]=[]
        for layer,moment in enumerate(stats[role]):
            a,r,report=fit_recolor(moment,CFG['eigenvalue_relative_floor'],CFG['eigenvalue_absolute_floor'],CFG['ridge_lambda'])
            recolor[role].append(a); ridge[role].append(r); reports[role].append(report)
            stats[role][layer]=None
            store(folder/f'recolor_{role}_{layer:02d}.pt',a)
            store(folder/f'ridge_{role}_{layer:02d}.pt',r)
            save(folder/'moment_report.json',reports)
            print('RECOLOR FIT',role,layer,report,flush=True)
    store(folder/'recolor.pt',recolor); store(folder/'ridge.pt',ridge)
    return recolor
