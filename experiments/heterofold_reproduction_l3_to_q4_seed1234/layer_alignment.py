import math


def select_sender_layers(target_layer, sender_layers=28, receiver_layers=36):
    """Zero-based, half-up tie rule, retain clipped duplicates (3 slots)."""
    if not 0<=target_layer<receiver_layers: raise ValueError(target_layer)
    center = math.floor(target_layer*(sender_layers-1)/max(1,receiver_layers-1)+0.5)
    return tuple(min(sender_layers-1,max(0,center+d)) for d in (-4,0,4))


def features(source, layer, receiver_layers):
    import torch
    return torch.cat([source[i].flatten(1) for i in select_sender_layers(layer,len(source),receiver_layers)],-1)
