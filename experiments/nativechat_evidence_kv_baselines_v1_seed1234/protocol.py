"""Native templates with an exact Qwen token grid and evidence-only senders."""
import math
from common import CFG, signature

INSTRUCTION = 'Use the supplied evidence to answer the question. Return only the short answer, without explanation.'
HEADER = INSTRUCTION + '\n\nEvidence:\n\n'
FOOTER = 'End of evidence.'


class ProtocolError(ValueError):
    pass


def document(title, sentences, index):
    return f'[Document {index}]\nTitle: {title}\n' + ''.join(sentences).strip() + '\n\n'


def encode(tok, evidence, question=None):
    content = HEADER + evidence + FOOTER
    if question is not None:
        content += '\n\nQuestion: ' + question + '\nAnswer:'
    messages = [{'role': 'user', 'content': content}]
    kwargs = {'add_generation_prompt': question is not None, 'enable_thinking': False,
              'date_string': CFG['native_template_date']}
    text = tok.apply_chat_template(messages, tokenize=False, **kwargs)
    if text.count(content) != 1:
        raise ProtocolError('native_template_changed_user_content')
    begin = text.index(content) + len(HEADER)
    end = begin + len(evidence)
    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded['input_ids']), [tuple(x) for x in encoded['offset_mapping']]
    native = tok.apply_chat_template(messages, tokenize=True, **kwargs)
    native_ids = native['input_ids'] if hasattr(native, 'keys') else native
    if ids != list(native_ids):
        raise ProtocolError('native_template_token_ids_mismatch')
    for a, b in offsets:
        if a < begin < b or a < end < b:
            raise ProtocolError('token_crosses_evidence_boundary')
    body = [i for i, (a, b) in enumerate(offsets) if a >= begin and b <= end and b > a]
    if not body or body != list(range(body[0], body[-1] + 1)):
        raise ProtocolError('noncontiguous_evidence_tokens')
    start, stop = body[0], body[-1] + 1
    if any(ids[i] in tok.all_special_ids for i in body):
        raise ProtocolError('special_token_inside_evidence')
    spans = [(a - begin, b - begin) for a, b in offsets[start:stop]]
    if spans[0][0] != 0 or spans[-1][1] != len(evidence):
        raise ProtocolError('incomplete_evidence_span')
    probes = []
    if question is not None:
        qstart = end + len(FOOTER + '\n\nQuestion: ')
        qend = qstart + len(question)
        probes = [i for i, (a, b) in enumerate(offsets) if b > qstart and a < qend and b > a]
        if not probes or min(probes) < stop:
            raise ProtocolError('question_leaks_into_evidence')
    return {'ids': ids, 'prefix_ids': ids[:start], 'body_ids': ids[start:stop],
            'suffix_ids': ids[stop:], 'spans': spans, 'begin': start, 'end': stop,
            'probe_positions': probes, 'text': text, 'template_sha256': signature({'prefix': text[:begin], 'suffix': text[end:] if question is None else 'question-dependent'})}


def sync_indices(source, target):
    """Original copy/drop rule within common UTF-8 end-boundary units."""
    ss, ts = source['spans'], target['spans']
    boundaries = sorted(({b for _, b in ss} & {b for _, b in ts}) | {0})
    mapping = {}
    si = ti = 0
    for left, right in zip(boundaries, boundaries[1:]):
        src, dst = [], []
        while si < len(ss) and ss[si][1] <= right:
            if ss[si][1] > left:
                src.append(si)
            si += 1
        while ti < len(ts) and ts[ti][1] <= right:
            if ts[ti][1] > left:
                dst.append(ti)
            ti += 1
        if not src or not dst:
            continue
        for rank, index in enumerate(dst):
            pick = math.ceil((rank + 1) * len(src) / len(dst)) - 1 if len(src) > len(dst) else rank * len(src) // len(dst)
            mapping[index] = src[pick]
    if set(mapping) != set(range(len(ts))):
        raise ProtocolError('incomplete_fullsync_mapping')
    return [mapping[i] for i in range(len(ts))]


def build(tok, row, unit='joint'):
    evidence = row['documents'][0] + row['documents'][1] if unit == 'joint' else row['documents'][int(unit)]
    target = encode(tok['qwen'], evidence, row['question'])
    sources = {family: encode(tok[family], evidence) for family in ('llama', 'gemma')}
    if len(target['body_ids']) > CFG['max_evidence_qwen_tokens']:
        raise ProtocolError('qwen_evidence_length')
    for source in sources.values():
        if len(source['ids']) > CFG['max_sender_tokens']:
            raise ProtocolError('sender_context_length')
        sync_indices(source, target)
        if len({b for _, b in source['spans']} & {b for _, b in target['spans']}) < 8:
            raise ProtocolError('fewer_than_eight_exact_end_anchors')
    return {'evidence': evidence, 'target': target, 'source': sources}


def audit_row(tok, row):
    units = {name: build(tok, row, name) for name in ('0', '1', 'joint')}
    a, b, joint = [units[name]['target'] for name in ('0', '1', 'joint')]
    if a['prefix_ids'] != b['prefix_ids'] or a['prefix_ids'] != joint['prefix_ids']:
        raise ProtocolError('receiver_prefix_not_shared')
    if joint['body_ids'] != a['body_ids'] + b['body_ids']:
        raise ProtocolError('independent_token_grid_differs_from_joint')
    if a['suffix_ids'] != joint['suffix_ids'] or b['suffix_ids'] != joint['suffix_ids']:
        raise ProtocolError('receiver_suffix_boundary_differs')
    return {'qwen_body_tokens': {u: len(x['target']['body_ids']) for u, x in units.items()},
            'sender_tokens': {f: {u: len(x['source'][f]['ids']) for u, x in units.items()} for f in ('llama','gemma')}}
