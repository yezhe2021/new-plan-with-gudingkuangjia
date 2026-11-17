"""Batch-2 QuALITY handoff latency with Sender capture outside the measurement."""
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
from pathlib import Path
import random
from time import perf_counter

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("latency_reference", ROOT.parent / "benchmark.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
from benchmark import Engine, fullsync
from heterofold.attention import rotate_half, rotary_cos_sin
from transformers.cache_utils import DynamicCache

C, K, torch = R.C, R.K, R.torch
LENGTHS = (4096, 16384, 32768)
BATCH, CHUNK, TRIALS, STEPS = 2, 2048, 3, 32
SENDER_CHUNK = 4096


def status(phase, **details):
    C.save(ROOT / "status.json", dict(status=phase, **details))


def inputs(engine):
    data = C.read(R.E4 / "data_manifest.json")
    archive = R.E4 / "quality_dev.jsonl"
    assert C.sha(archive) == data["source_sha256"]
    articles = {}
    for row in C.read_jsonl(archive):
        text = row["article"]
        digest = hashlib.sha256(text.encode()).hexdigest()
        articles[digest] = str(row["article_id"]), text
    header = C.read(C.ROOT / "manifest.json")
    tok = engine.tok
    def encode(text):
        return tok.encode(text, add_special_tokens=False, verbose=False)
    suffix_ids = encode(header["assistant"])
    assert suffix_ids == data["suffix_ids"] and engine.prefix_ids == data["prefix_ids"]
    # Disjoint article streams. The same streams supply the nested prefixes at every length.
    keys = sorted(articles)
    random.Random(1234).shuffle(keys)
    iterator = iter(keys)
    groups = []
    for stream in range(4):
        selected, pieces = [], []
        for key in iterator:
            article, text = articles[key]
            selected.append((article, key))
            pieces.append(text)
            if len(encode("\n\n".join(pieces))) >= max(LENGTHS) + 64:
                break
        text = "\n\n".join(pieces)
        offsets = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
        assert len(offsets) >= max(LENGTHS) + 64
        rows = {}
        for length in LENGTHS:
            end = offsets[length - 1][1]
            prompt = text[:end]
            ids = encode(prompt)
            if len(ids) != length or encode(header["prefix"] + prompt + header["assistant"]) != engine.prefix_ids + ids + suffix_ids:
                for end in dict.fromkeys(e for _, e in offsets[length-12:length+12]):
                    prompt = text[:end]
                    ids = encode(prompt)
                    if len(ids) == length and encode(header["prefix"] + prompt + header["assistant"]) == engine.prefix_ids + ids + suffix_ids:
                        break
                else:
                    raise ValueError(("Cannot fit exact QuALITY token boundary", stream, length))
            rows[length] = dict(prompt=prompt, content_ids=ids, article_ids=selected,
                                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), suffix_ids=suffix_ids)
        assert all(rows[y]["prompt"].startswith(rows[x]["prompt"]) for x, y in zip(LENGTHS, LENGTHS[1:]))
        groups.append(rows)
    return {length: [[groups[i][length], groups[i+1][length]] for i in (0, 2)] for length in LENGTHS}


@torch.no_grad()
def capture(engine, row):
    sender = engine.sender
    ids = engine.sender_tok.encode(row["prompt"], add_special_tokens=False, verbose=False)
    assert len(ids) <= sender.config.max_position_embeddings
    ids_cuda = torch.tensor([ids], device="cuda")
    positions = torch.arange(len(ids), device="cuda")
    layers = C.layers_of(sender)
    stores = tuple([torch.empty((len(ids), 1024), dtype=sender.dtype) for _ in layers] for _ in range(2))
    offset = 0

    def hook_for(target):
        def hook(_module, _args, output):
            width = output.shape[1]
            target[offset:offset+width].copy_(output[0].detach())
        return hook

    with ExitStack() as stack:
        for i, layer in enumerate(layers):
            for role, projection in enumerate((layer.self_attn.k_proj, layer.self_attn.v_proj)):
                stack.callback(projection.register_forward_hook(hook_for(stores[role][i])).remove)
        cache = None
        for offset in range(0, len(ids), SENDER_CHUNK):
            end = min(offset + SENDER_CHUNK, len(ids))
            result = C.decoder(sender)(input_ids=ids_cuda[:, offset:end], position_ids=positions[None, offset:end],
                                       past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            assert cache.get_seq_length() == end
    return ids, stores


def place(engine, pairs):
    result = DynamicCache(config=engine.model.config)
    # Initial update keeps the input storage. All answer updates retain the stock cache API.
    for l, (k, v) in enumerate(pairs):
        layer = result.layers[l]
        layer.lazy_initialization(k, v)
        layer.keys, layer.values = k, v
    return result


def build_batch(engine, rows, source, clock):
    lengths = [len(row["content_ids"]) for row in rows]
    assert len(set(lengths)) == 1 and len(rows) == BATCH
    length = lengths[0]
    h = len(engine.prefix_ids)
    rank = engine.correction
    model = engine.model
    base = rank.base
    dtype = next(base.parameters()).dtype

    def align():
        indices = []
        for row, (ids, _) in zip(rows, source):
            a = fullsync(row["prompt"], engine.sender_tok, engine.tok, row["content_ids"])
            assert list(a.source_ids) == ids
            indices.append(torch.tensor(a.source_indices, device="cpu"))
        return indices

    indices = clock.measure("alignment", align)

    def allocate():
        pairs = []
        for k, v in K.pairs(engine.header):
            holder = []
            for x in (k, v):
                buf = x.new_empty((BATCH, x.shape[1], length + h, x.shape[-1]))
                buf[:, :, :h].copy_(x.expand(BATCH, -1, -1, -1))
                holder.append(buf)
            pairs.append(tuple(holder))
        return pairs

    pairs = clock.measure("cache_allocate", allocate)
    taps = tuple(sorted(set(base.taps_tuple)))
    for begin in range(0, length, CHUNK):
        end = min(begin + CHUNK, length)
        width = end - begin
        def upload():
            return tuple({
                tap: torch.cat([cache[role][tap].index_select(0, ix[begin:end])
                                for ix, (_, cache) in zip(indices, source)], dim=0).to("cuda")
                for tap in taps} for role in (0, 1))
        keys, values = clock.measure("gather_and_upload", upload)

        def translate():
            z = torch.zeros(BATCH * width, base.cfg.state_width, device="cuda", dtype=dtype)
            raw = []
            for l, (tap, layer) in enumerate(zip(base.taps_tuple, base.layers)):
                k, v, z = layer(keys[tap].to(dtype), values[tap].to(dtype), z)
                raw.append((k + rank.keys[l](k), v + rank.values[l](v)))
            return raw
        assert base.cfg.carry_state
        raw = clock.measure("translation", translate)
        del keys, values

        def finalize():
            positions = torch.arange(h + begin, h + end, device="cuda").repeat(BATCH)
            tables = C._rotary_tables(model, positions)
            for l, (key, value) in enumerate(raw):
                result = C.finalize_layer(model, l, key, value, *tables(l))
                for ready, dest in zip(result, pairs[l]):
                    heads, _, dim = ready.shape
                    dest[:, :, h + begin:h + end].copy_(ready.reshape(heads, BATCH, width, dim).transpose(0, 1))
        clock.measure("cache_finalize", finalize)
        del raw
    return clock.measure("cache_handle", place, engine, pairs)


class ConstructionCache(DynamicCache):
    """Reuse the body storage and retain only the newly generated memory states."""
    def __init__(self, engine, body):
        super().__init__(config=engine.model.config)
        self.memory = {}
        for l, (k, v) in enumerate(K.pairs(body)):
            layer = self.layers[l]
            layer.lazy_initialization(k, v)
            layer.keys, layer.values = k, v

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        layer = self.layers[layer_idx]
        self.memory[layer_idx] = key_states, value_states
        return torch.cat((layer.keys, key_states), dim=-2), torch.cat((layer.values, value_states), dim=-2)


def construct(memory, body, batch):
    engine = memory.engine
    length = body.get_seq_length()
    positions = torch.arange(length, length + memory.M, device="cuda")
    cache = ConstructionCache(engine, body)
    prompt = memory.memory.prompt[None].expand(batch, -1, -1).to(engine.model.dtype)
    with memory.routing.context("construct", len(engine.prefix_ids), length, memory.M, positions):
        C.decoder(engine.model)(inputs_embeds=prompt, past_key_values=cache,
                                attention_mask=torch.ones(batch, length + memory.M, device="cuda", dtype=torch.long),
                                position_ids=positions[None].expand(batch, -1), use_cache=True)
    data = []
    assert len(cache.memory) == 36 and body.get_seq_length() == length
    for l in range(36):
        key, value = cache.memory[l]
        assert key.shape[0] == batch and key.shape[2] == memory.M
        cosine, sine = rotary_cos_sin(engine.model, positions, l, engine.model.dtype)
        cosine, sine = cosine[None, None].float(), sine[None, None].float()
        rotated = key.float()
        raw = (rotated * cosine - rotate_half(rotated) * sine) / (cosine.square() + sine.square()).clamp_min(1e-12)
        data.extend((raw.to(key.dtype), value.contiguous()))
    return tuple(data)


def receiver_step(engine, body, ids):
    batch, width = ids.shape
    begin = body.get_seq_length()
    pos = torch.arange(begin, begin + width, device="cuda")
    out = C.decoder(engine.model)(input_ids=ids, past_key_values=body, use_cache=True,
                                  attention_mask=torch.ones(batch, begin + width, device="cuda", dtype=torch.long),
                                  position_ids=pos[None].expand(batch, -1))
    assert out.past_key_values.get_seq_length() == begin + width
    return engine.model.lm_head(out.last_hidden_state[:, -1]).float()


@torch.no_grad()
def measure(engine, memory, rows, source, *, profile=False, decode=False):
    stopwatch = R.Stopwatch(profile)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    body = build_batch(engine, rows, source, stopwatch)
    cached_bytes = R.tensor_bytes(K.flat(body))
    bank = stopwatch.measure("memory_construct", construct, memory, body, BATCH)
    ids = torch.tensor([row["suffix_ids"] for row in rows], device="cuda")
    with memory.answering(bank, memory.gates):
        logits = stopwatch.measure("receiver_suffix_and_head", receiver_step, engine, body, ids)
        token = logits.argmax(-1)
        first = token.tolist()
        ttft = (perf_counter() - start) * 1000
        peak = torch.cuda.max_memory_allocated()
        tpot = None
        if decode:
            start = perf_counter()
            for _ in range(STEPS):
                logits = receiver_step(engine, body, token[:, None])
                token = logits.argmax(-1)
            torch.cuda.synchronize()
            tpot = (perf_counter() - start) * 1000 / STEPS
    assert engine.student_test_native_prefills == 0
    return dict(ttft_ms=ttft, decode_ms_per_step=tpot, first_tokens=first, peak_allocated_bytes=peak,
                peak_decode_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                body_bytes=cached_bytes, memory_bytes=R.tensor_bytes(bank), phases_ms=stopwatch.stages)


@torch.no_grad()
def verify_batch(engine, memory, rows, source):
    body = build_batch(engine, rows, source, R.Stopwatch(False))
    bank = construct(memory, body, BATCH)
    errors = []
    for i, row in enumerate(rows):
        single = place(engine, [tuple(x[i:i+1].contiguous() for x in pair) for pair in K.pairs(body)])
        expected = memory.construct(row, single)
        for reference, current in zip(expected.bank, bank):
            x, y = reference.float(), current[i:i+1].float()
            errors.append(float((x-y).norm()/x.norm().clamp_min(1e-12)))
    assert max(errors) <= .025, max(errors)
    return max(errors)


def summary(records, manifest):
    lengths = {}
    for length in LENGTHS:
        rows = [r for r in records if r["length"] == length]
        if not rows:
            continue
        trials = [v for row in rows for v in row["trials"]]
        p = [r["profile"]["phases_ms"] for r in rows]
        lengths[str(length)] = dict(
            batches=len(rows), ttft_ms=R.stats([t["ttft_ms"] for t in trials]),
            decode_ms_per_step=R.stats([t["decode_ms_per_step"] for t in trials]),
            memory_construct_ms=R.stats([t["memory_construct"] for t in p]),
            phases_ms={k:R.stats([m[k] for m in p]) for k in p[0]},
            peak_allocated_bytes=max(t["peak_allocated_bytes"] for t in trials),
            peak_reserved_bytes=max(t["peak_reserved_bytes"] for t in trials),
            peak_decode_bytes=max(t["peak_decode_bytes"] for t in trials),
            body_bytes=trials[0]["body_bytes"], memory_bytes=trials[0]["memory_bytes"])
    C.save(ROOT / "summary.json", dict(status="completed" if len(records)==len(LENGTHS)*2 else "running",
                                       manifest=manifest, lengths=lengths))


def main():
    status("loading")
    engine = Engine()
    memory, weights = R.load_memory(engine)
    engine.mode = "test"
    assert len(engine.prefix_ids) + max(LENGTHS) + max(STEPS + 12, memory.M) <= engine.model.config.max_position_embeddings
    cohorts = inputs(engine)
    manifest = dict(script_sha256=C.sha(Path(__file__)), quality_archive_sha256=C.sha(R.E4 / "quality_dev.jsonl"),
                    checkpoint=C.sha(weights), device=torch.cuda.get_device_name(), batch=BATCH, chunk=CHUNK,
                    trials=TRIALS, decode_steps=STEPS, lengths=list(LENGTHS), sender_chunk_excluded=SENDER_CHUNK,
                    prompt_policy="nested prefixes of disjoint concatenations of QuALITY dev articles",
                    prompts={str(n): [[{k:row[k] for k in ("article_ids","prompt_sha256")} for row in g]
                                      for g in groups] for n,groups in cohorts.items()},
                    timing="starts with CPU Sender KV available, includes Full-Sync and GPU upload",
                    model_residency="both frozen models stay on GPU; Sender execution is outside the timed region")
    path = ROOT / "manifest.json"
    if path.exists():
        assert C.read(path) == manifest
    else:
        C.save(path, manifest)
    pin = C.sha(path)
    records = []
    for length, groups in cohorts.items():
        for g, rows in enumerate(groups):
            record_path = ROOT / "batches" / f"{length}_{g}.json"
            if record_path.exists():
                record = C.read(record_path)
                assert record["manifest_sha256"] == pin
            else:
                status("preparing_sender_kv", length=length, batch_index=g+1)
                source = [capture(engine, row) for row in rows]
                torch.cuda.empty_cache()
                status("measuring_handoff", length=length, batch_index=g+1)
                if length==LENGTHS[0] and g==0:
                    error = verify_batch(engine, memory, rows, source)
                    C.save(ROOT / "batch_parity.json", dict(max_memory_kv_relative_l2=error))
                warm = measure(engine, memory, rows, source)
                trials = [measure(engine, memory, rows, source, decode=True) for _ in range(TRIALS)]
                assert all(t["first_tokens"] == warm["first_tokens"] for t in trials)
                profile = measure(engine, memory, rows, source, profile=True)
                assert profile["first_tokens"] == warm["first_tokens"]
                record = dict(manifest_sha256=pin, length=length, batch_index=g+1, trials=trials, profile=profile)
                C.save(record_path, record)
                del source
            records.append(record)
            summary(records, manifest)
            print("BATCH",length,g+1,[round(v["ttft_ms"],1) for v in record["trials"]],flush=True)
    status("completed")


if __name__ == "__main__":
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception as exc:
            status("failed", error=repr(exc))
            raise
