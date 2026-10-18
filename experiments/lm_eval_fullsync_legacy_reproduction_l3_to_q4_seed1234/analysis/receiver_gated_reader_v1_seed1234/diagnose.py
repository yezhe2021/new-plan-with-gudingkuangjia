"""Held-out, post-training interventions; never used for checkpoint selection.

Gate-only/read-only are inference ablations of the same trained checkpoint,
not independently trained comparison arms. Wrong-memory replaces the entire
translated prompt bank, so it tests prompt dependence, not isolated evidence.
"""
from collections import Counter
import hashlib
import json
import torch
from experiment import (HERE, CONFIG, base_train, parent_prepare, model_bundle,
    ReaderHFLM, load_reader, read_rows, write_rows, save, log, verify_manifest,
    digest, gpu_check)
from trajectory_data import prompt, ANSWER_RE
from kv_backend import assemble


def main():
    verify_manifest()
    gpu_check()
    receiver, sender, base, adapter, source_tok, hf, reader = model_bundle(False)
    p = torch.load(HERE / 'reader/best.pt', map_location='cpu', weights_only=True)
    reader.load_state_dict(p['state'])
    gen = ReaderHFLM(receiver, 'translated', sender, base, adapter)
    attempts = read_rows(HERE / 'prepared/validation_attempts.jsonl')
    chosen = []
    for task in sorted({r['task'] for r in attempts}):
        chosen += [r for r in attempts if r['task'] == task][:CONFIG['diagnostic_questions_per_task']]
    demos = {r['base_id']: r for r in read_rows(HERE / 'prepared/demos.jsonl')}
    anchors, records = {}, []
    raw_path = HERE / 'diagnostics/generations.jsonl'
    if raw_path.exists():
        records = read_rows(raw_path)
    done = {(r['base_id'], r['condition']) for r in records}
    for index, question in enumerate(chosen):
        for condition in ('full', 'disabled', 'gate_only', 'read_only', 'wrong_memory', 'masked', 'rotated'):
            if (question['base_id'], condition) in done:
                continue
            q = dict(question)
            if condition == 'rotated':
                n = len(q['choices'])
                q['choices'] = q['choices'][1:] + q['choices'][:1]
                q['gold'] = 'ABCDEFGHIJ'[('ABCDEFGHIJ'.index(q['gold']) - 1) % n]
                q['context'] = prompt(q, [demos[i] for i in q['demo_ids']])
            q['context_ids'] = hf.tok_encode(q['context'])
            item, bk, bv = base_train.prefix(sender, receiver, base, source_tok, hf, q, anchors)
            if condition == 'wrong_memory':
                other = dict(chosen[(index + 1) % len(chosen)])
                other['context_ids'] = hf.tok_encode(other['context'])
                oi, ok, ov = base_train.prefix(sender, receiver, base, source_tok, hf, other, anchors)
                # Map another question's already-translated pre-RoPE bank to the
                # same slot count. Keep current anchor and original positions.
                with torch.no_grad():
                    wk, wv = assemble(oi, ok, ov, adapter)
                    slots = torch.linspace(1, wk.shape[1] - 1, item['tokens'], device='cuda').round().long()
                    key = torch.cat((item['anchor_k'].cuda(), wk[:, slots]), 1)
                    value = torch.cat((item['anchor_v'].cuda(), wv[:, slots]), 1)
                del oi, ok, ov, wk, wv
            else:
                with torch.no_grad():
                    key, value = assemble(item, bk, bv, adapter)
            gen._cached_context = hashlib.sha256(('translated' + q['context'].rstrip()).encode()).hexdigest()
            gen._cached_kv = key, value
            mode = condition if condition in ('disabled', 'gate_only', 'read_only', 'masked') else 'full'
            reader.statistics = [] if condition == 'full' else None
            try:
                with reader.scope(None, mode):
                    ids, text = parent_prepare.generate(gen, q['context'], CONFIG['validation_generation_max_new_tokens'])
                statistics = reader.statistics
            finally:
                reader.statistics = None
                gen._cached_context = gen._cached_kv = None
            found = ANSWER_RE.findall(text)
            prediction = found[0] if found else '[invalid]'
            eos = receiver.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            stop = 'eos' if ids[-1] in eos else ('question_stop' if 'Question:' in text else 'length_limit')
            records.append({'base_id': q['base_id'], 'condition': condition, 'task': q['task'],
                            'gold': q['gold'], 'prediction': prediction, 'correct': prediction == q['gold'],
                            'choices': q['choices'], 'response': text, 'ids': ids, 'terminal': stop,
                            'mean_external_attention_mass': (sum(s['external_mass'] for s in statistics) / len(statistics)) if statistics else None})
            write_rows(raw_path, records)
            del item, bk, bv, key, value
            log(f'DIAGNOSTIC {len(records)}/{len(chosen)*7} {condition}: gold={q["gold"]} pred={prediction}')
    summary = {}
    for condition in sorted({r['condition'] for r in records}):
        rows = [r for r in records if r['condition'] == condition]
        summary[condition] = {'n': len(rows), 'correct': sum(r['correct'] for r in rows),
                              'valid': sum(r['prediction'] != '[invalid]' for r in rows),
                              'terminal_counts': dict(Counter(r['terminal'] for r in rows)),
                              'predicted_letters': dict(Counter(r['prediction'] for r in rows))}
    pairs = []
    for q in chosen:
        pair = [next(r for r in records if r['base_id'] == q['base_id'] and r['condition'] == c) for c in ('full', 'rotated')]
        texts = [r['choices']['ABCDEFGHIJ'.index(r['prediction'])] if r['prediction'] in 'ABCDEFGHIJ'[:len(r['choices'])] else None for r in pair]
        pairs.append(texts[0] is not None and texts[0] == texts[1])
    save(HERE / 'diagnostics/summary.json', {'validation_only_not_official_benchmark': True,
         'inference_ablations_not_separately_trained': True,
         'wrong_memory_replaces_question_and_evidence': True,
         'selected_reader_sha256': digest(HERE / 'reader/best.pt'),
         'conditions': summary, 'option_semantic_consistency': {'consistent': sum(pairs), 'pairs': len(pairs)}})


if __name__ == '__main__':
    main()
