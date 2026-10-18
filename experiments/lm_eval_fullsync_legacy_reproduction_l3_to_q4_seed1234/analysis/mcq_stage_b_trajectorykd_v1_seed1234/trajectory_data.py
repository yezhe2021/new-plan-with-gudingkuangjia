"""Training-only long reasoning prompts, remapped choices, strict terminal audit."""
import hashlib
import random
import re

ANSWER_RE = re.compile(r'answer is \(?([ABCDEFGHIJ])\)?')
DESCRIPTION = ('The following are multiple choice questions (with answers) about science. '
               'Think step by step and then finish your answer with "the answer is (X)" '
               'where X is the correct letter choice.\n')


def rng_for(*parts):
    return random.Random(int(hashlib.sha256('|'.join(map(str, (1234, *parts))).encode()).hexdigest()[:16], 16))


def question(row):
    return row['doc'].get('question', row['doc'].get('question_stem')).strip()


def make_question(row, split):
    choices = row['doc']['choices']
    labels = list(map(str, choices['label']))
    gold = labels.index(str(row['doc']['answerKey']))
    permutation = list(range(len(choices['text'])))
    rng_for(split, row['id']).shuffle(permutation)
    return {'base_id': row['id'], 'task': row['task'], 'question': question(row),
            'choices': [choices['text'][i] for i in permutation],
            'gold': 'ABCDEFGHIJ'[permutation.index(gold)], 'permutation': permutation}


def render(q):
    return 'Question:\n' + q['question'] + '\nOptions:\n' + '\n'.join(f'{"ABCDEFGHIJ"[i]}. {s}' for i,s in enumerate(q['choices'])) + "\nAnswer: Let's think step by step."


def prompt(q, demos=(), verbose=False):
    head = DESCRIPTION
    if verbose:
        head += 'Explain the relevant principle and compare the options in about 100 to 150 words before the final answer.\n'
    segments = []
    for demo in demos:
        assert demo['base_id'] != q['base_id'] and demo['question'] != q['question'], 'Question appears in its own demonstrations'
        segments.append(render(demo) + demo['response'] + '\n\n')
    return head + ''.join(segments) + render(q)


def acceptance(text, ids, eos_ids, gold, max_matches=2):
    found = ANSWER_RE.findall(text)
    if ids and ids[-1] in eos_ids:
        terminal = 'eos'
    elif 'Question:' in text:
        terminal = 'question_stop'
    else:
        terminal = 'length_limit'
    accepted = (terminal != 'length_limit' and 0 < len(found) <= max_matches and all(x == gold for x in found))
    return {'accepted': accepted, 'terminal': terminal, 'extracted_answers': found,
            'reason': 'accepted' if accepted else ('truncated' if terminal == 'length_limit' else 'wrong_or_repetitive_answer')}


def training_record(q, context, demos, ids, text, stop, tokenizer):
    # Keep real generated IDs including the observed terminal; never append synthetic EOS.
    # Prefix decoding recovers exact token spans even when whitespace is merged.
    clean = tokenizer.decode(ids, skip_special_tokens=True)
    matches = list(ANSWER_RE.finditer(clean))
    assert stop['accepted'] and matches
    final = matches[-1]
    ends = [len(tokenizer.decode(ids[:i + 1], skip_special_tokens=True)) for i in range(len(ids))]
    starts = [0] + ends[:-1]
    answer_positions = [i for i,(a,b) in enumerate(zip(starts, ends)) if b > final.start() and a < final.end()]
    if stop['terminal'] == 'eos':
        terminal_positions = [len(ids) - 1]
        response = clean.strip()
    else:
        at = clean.index('Question:')
        terminal_positions = [i for i,(a,b) in enumerate(zip(starts, ends)) if b > at]
        response = clean[:at].strip()
    assert answer_positions and terminal_positions and not set(answer_positions) & set(terminal_positions)
    return {**q, 'context': context, 'demo_ids': [x['base_id'] for x in demos],
            'context_ids': tokenizer.encode(context, add_special_tokens=False),
            'continuation_ids': ids, 'response': ' ' + response,
            'answer_positions': answer_positions, 'terminal_positions': terminal_positions,
            'stop': stop, 'shots': len(demos)}
