"""Independent reader experiment; all predecessor sources/checkpoints are read-only."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[1]
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
PREVIOUS = PARENT / CONFIG['previous_directory']
sys.path.insert(0, str(PREVIOUS))
import followup_common as previous
from kv_backend import CFG, FullSyncHFLM, load_base, load_model, seed_all
from reader import ExternalReader

log, save, plain, digest = previous.log, previous.save, previous.plain, previous.digest
read_rows, write_rows = previous.read_rows, previous.write_rows
base_train = previous.base_train
mcq_train = previous.parent.previous_train
parent_prepare = previous.parent_prepare


def gpu_check():
    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info()
    log(f'Server B {torch.cuda.get_device_name(0)} free={free / 2**30:.2f}/{total / 2**30:.2f} GiB')
    if free / 2**30 < CONFIG['min_free_gpu_gib']:
        raise RuntimeError('Insufficient free GPU memory; unrelated jobs left intact')


def identity():
    previous.verify_manifest()
    files = sorted(HERE.glob('*.py')) + [HERE / 'config.json']
    return {'configuration': CONFIG, 'source_sha256': {str(p): digest(p) for p in files},
            'previous_manifest_sha256': digest(PREVIOUS / 'manifest.json'),
            'stage_a': {'path': str(PARENT / CONFIG['parent_stage_a']), 'sha256': digest(PARENT / CONFIG['parent_stage_a'])},
            'frozen_stage_b': {'path': str(PREVIOUS / 'stage_b/best.pt'), 'sha256': digest(PREVIOUS / 'stage_b/best.pt')}}


def freeze_manifest():
    manifest = {'identities': identity(),
                'prepared_sha256': {str(p.relative_to(HERE)): digest(p) for p in sorted((HERE / 'prepared').glob('*.json*'))},
                'smoke_sha256': digest(HERE / 'reports/smoke.json'),
                'unit_tests_sha256': digest(HERE / 'reports/tests.json')}
    path = HERE / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        save(path, manifest)
    return manifest


def verify_manifest():
    m = json.loads((HERE / 'manifest.json').read_text())
    assert m['identities'] == identity(), 'Reader source/parent identity changed'
    for rel, sha in m['prepared_sha256'].items():
        assert digest(HERE / rel) == sha, rel
    assert digest(HERE / 'reports/smoke.json') == m['smoke_sha256']
    assert digest(HERE / 'reports/tests.json') == m['unit_tests_sha256']
    return m


def load_frozen_adapter():
    return previous.load_new_adapter(PREVIOUS / 'stage_b/best.pt', False)


def load_reader(receiver, path=None, trainable=False):
    reader = ExternalReader(receiver.config, CONFIG['reader_rank']).cuda()
    if path is not None:
        m = verify_manifest()
        p = torch.load(path, map_location='cpu', weights_only=True)
        assert p['protocol'] == CONFIG['protocol']
        assert p['frozen_stage_b_sha256'] == m['identities']['frozen_stage_b']['sha256']
        reader.load_state_dict(p['state'], strict=True)
    reader.train(trainable).requires_grad_(trainable)
    reader.attach(receiver)
    # Not a child nn.Module of the receiver: frozen-base checks stay unambiguous.
    object.__setattr__(receiver, '_external_reader', reader)
    return reader


def model_bundle(trainable=False):
    seed_all(CONFIG['seed'])
    receiver, sender, base, adapter, source_tok, hf = previous.model_bundle(False)
    del adapter
    adapter = load_frozen_adapter()
    reader = load_reader(receiver, trainable=trainable)
    return receiver, sender, base, adapter, source_tok, hf, reader


# Frozen helpers retain exactly their original loss formulas. Scope only the
# translated student forward, including checkpoint recomputation. Teacher/native
# forwards cannot see the reader even when invoked within another student scope.
_suffix_hidden = base_train.suffix_hidden
_suffix_logits = mcq_train.suffix_logits


def scoped_hidden(receiver, context, continuation, key=None, value=None):
    reader = receiver._external_reader
    with reader.scope(None if key is None else len(context) - 1):
        return _suffix_hidden(receiver, context, continuation, key, value)


def scoped_logits(receiver, context, answer, key=None, value=None):
    reader = receiver._external_reader
    with reader.scope(None if key is None else len(context) - 1):
        return _suffix_logits(receiver, context, answer, key, value)


base_train.suffix_hidden = scoped_hidden
base_train.CONFIG = CONFIG
mcq_train.suffix_logits = scoped_logits


class ReaderHFLM(FullSyncHFLM):
    def __init__(self, receiver, mode='native', sender=None, base=None, adapter=None, **kwargs):
        super().__init__(receiver, mode, sender, base, adapter, **kwargs)
        if not hasattr(receiver, '_external_reader'):
            load_reader(receiver, getattr(adapter, '_reader_checkpoint', None))

    def _loglikelihood_tokens(self, requests, **kwargs):
        if self.mode == 'native':
            with self.model._external_reader.scope(None):
                return super()._loglikelihood_tokens(requests, **kwargs)
        results = []
        for request in requests:
            with self.model._external_reader.scope(len(request[1]) - 1):
                results.extend(super()._loglikelihood_tokens([request], **kwargs))
        return results

    def _model_generate(self, context, max_length, stop, **kwargs):
        end = context.shape[1] - 1 if self.mode != 'native' else None
        with self.model._external_reader.scope(end):
            return super()._model_generate(context, max_length, stop, **kwargs)


def load_new_adapter(path, trainable=False):
    assert not trainable
    adapter = load_frozen_adapter()
    object.__setattr__(adapter, '_reader_checkpoint', path)
    return adapter
