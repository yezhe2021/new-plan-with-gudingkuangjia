"""Actual portable INT4/INT8 wire format. No GPU or model dependencies."""
import json
import math
import struct
import numpy as np
import torch

MAGIC = b'AKV1'


def pack4(x):
    x = np.asarray(x, dtype=np.uint8).reshape(-1)
    assert (x <= 15).all()
    if len(x) % 2:
        x = np.pad(x, (0, 1))
    return (x[::2] | (x[1::2] << 4)).tobytes()


def unpack4(data, count):
    x = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(len(x) * 2, dtype=np.uint8)
    out[::2], out[1::2] = x & 15, x >> 4
    return out[:count]


def parameters(x, role, group):
    """Store one common FP16 min/range, independent of the bit allocation."""
    l, n, h, d = x.shape
    xf = x.float()
    if role == 'key':
        pad = (-n) % group
        padded = torch.cat((xf, torch.zeros(l, pad, h, d)), 1) if pad else xf
        blocked = padded.reshape(l, -1, group, h, d)
        valid = (torch.arange(blocked.shape[1] * group).reshape(1, -1, group, 1, 1) < n)
        low = blocked.masked_fill(~valid, float('inf')).amin(2)
        high = blocked.masked_fill(~valid, -float('inf')).amax(2)
    else:
        assert d % group == 0
        blocked = xf.reshape(l, n, h, d // group, group)
        low, high = blocked.amin(-1), blocked.amax(-1)
    low = low.half()
    span = (high - low.float()).clamp_min(0).half()
    assert torch.isfinite(low).all() and torch.isfinite(span).all()
    return torch.stack((low, span), -1).contiguous()


def expanded(params, role, shape, group):
    low, span = params[..., 0].float(), params[..., 1].float()
    if role == 'key':
        low, span = (v.repeat_interleave(group, 1)[:, :shape[1]] for v in (low, span))
    else:
        low, span = (v.repeat_interleave(group, -1) for v in (low, span))
    return low, span


def codes(x, params, role, group, qmax):
    low, span = expanded(params, role, x.shape, group)
    return (((x.float() - low) / span.clamp_min(1e-30)).clamp(0, 1) * qmax).round().byte()


def encode(key, value, high_positions, bits, group, source_positions, target_map):
    assert bits in [4, 16, 'mixed'] and key.shape == value.shape and key.ndim == 4
    key, value = (v.detach().cpu().half().contiguous() for v in (key, value))
    assert torch.isfinite(key).all() and torch.isfinite(value).all()
    l, n, h, d = key.shape
    assert len(source_positions) == n and len(set(source_positions)) == n
    assert all(0 <= int(x) < n for x in target_map)
    mask = np.zeros(n, dtype=np.uint8)
    mask[list(high_positions)] = 1
    assert int(mask.sum()) == len(high_positions)
    assert bits == 'mixed' or not mask.any()
    high = torch.from_numpy(mask.astype(bool))
    sections = [np.packbits(mask, bitorder='little').tobytes(),
                np.asarray(source_positions, dtype='<u4').tobytes(),
                np.asarray(target_map, dtype='<u4').tobytes()]
    if bits == 16:
        sections += [key.numpy().astype('<f2', copy=False).tobytes(), value.numpy().astype('<f2', copy=False).tobytes()]
    else:
        for role, x in [('key', key), ('value', value)]:
            params = parameters(x, role, group)
            q4 = codes(x, params, role, group, 15)
            q8 = codes(x, params, role, group, 255) if mask.any() else None
            sections += [params.numpy().astype('<f2', copy=False).tobytes(), pack4(q4[:, ~high].numpy()),
                         q8[:, high].numpy().tobytes() if q8 is not None else b'']
    header = dict(version=1, shape=list(key.shape), bits=bits, group=group,
                  target_tokens=len(target_map), lengths=[len(v) for v in sections])
    metadata = json.dumps(header, separators=(',', ':'), sort_keys=True).encode()
    packet = MAGIC + struct.pack('<I', len(metadata)) + metadata + b''.join(sections)
    payload_bytes = (sum(header['lengths'][3:]) if bits == 16 else
                     sum(header['lengths'][i] for i in [4, 5, 7, 8]))
    stats = dict(wire_bytes=len(packet), tensor_elements=key.numel()+value.numel(),
                 protected_tokens=int(mask.sum()), transmitted_tokens=n,
                 data_bits_per_element=16 if bits == 16 else 4+4*int(mask.sum())/n,
                 effective_bits_per_element=8*len(packet)/(key.numel()+value.numel()),
                 code_bytes=payload_bytes, overhead_bytes=len(packet)-payload_bytes)
    return packet, stats


def decode(packet):
    assert packet[:4] == MAGIC
    size = struct.unpack('<I', packet[4:8])[0]
    header = json.loads(packet[8:8+size])
    assert header['version'] == 1
    sections, cursor = [], 8+size
    for length in header['lengths']:
        sections.append(packet[cursor:cursor+length]); cursor += length
    assert cursor == len(packet)
    l, n, h, d = shape = header['shape']
    mask = np.unpackbits(np.frombuffer(sections[0], dtype=np.uint8), bitorder='little')[:n].astype(bool)
    positions = np.frombuffer(sections[1], dtype='<u4').astype(np.int64).tolist()
    mapping = np.frombuffer(sections[2], dtype='<u4').astype(np.int64).tolist()
    assert len(positions) == n and len(mapping) == header['target_tokens']
    if header['bits'] == 16:
        tensors = [torch.from_numpy(np.frombuffer(b, dtype='<f2').copy().reshape(shape)) for b in sections[3:]]
    else:
        tensors = []
        high = torch.from_numpy(mask)
        for role, i in [('key', 3), ('value', 6)]:
            pshape = (l, math.ceil(n/header['group']), h, d, 2) if role == 'key' else (l, n, h, d//header['group'], 2)
            params = torch.from_numpy(np.frombuffer(sections[i], dtype='<f2').copy().reshape(pshape))
            low, span = expanded(params, role, shape, header['group'])
            q = torch.empty(shape, dtype=torch.float32)
            low_count = l * int((~mask).sum()) * h * d
            q[:, ~high] = torch.from_numpy(unpack4(sections[i+1], low_count).copy().reshape(l, -1, h, d)).float()/15
            if mask.any():
                q[:, high] = torch.from_numpy(np.frombuffer(sections[i+2], dtype=np.uint8).copy().reshape(l, -1, h, d)).float()/255
            tensors.append((low + q*span).half())
    assert len(tensors) == 2 and all(torch.isfinite(v).all() for v in tensors)
    return *tensors, dict(source_positions=positions, target_map=mapping, high_positions=np.flatnonzero(mask).tolist())
