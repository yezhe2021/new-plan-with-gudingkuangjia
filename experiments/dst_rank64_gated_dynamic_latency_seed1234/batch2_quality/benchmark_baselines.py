"""Measure Native and Recolor+C on the pinned QuALITY batch-2 workload."""
import fcntl
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import benchmark_batch as B
import resume_32k as S

from heterofold.alignment import receiver_to_sender_indices

C, K, torch, ROOT = B.C, B.K, B.torch, B.ROOT
OUT = ROOT / "baselines"


def status(state, **details):
    C.save(OUT / "status.json", dict(status=state, **details))


@torch.no_grad()
def native_first(engine, rows, block):
    ids = [engine.prefix_ids + engine.tok.encode(row["prompt"], add_special_tokens=False, verbose=False)
           + row["suffix_ids"] for row in rows]
    assert all(seq == engine.prefix_ids + row["content_ids"] + row["suffix_ids"] for seq, row in zip(ids, rows))
    inputs = torch.tensor(ids, device="cuda")
    positions = torch.arange(inputs.shape[1], device="cuda")[None].expand(B.BATCH, -1)
    cache = None
    width = inputs.shape[1] if block is None else block
    for start in range(0, inputs.shape[1], width):
        end = min(start + width, inputs.shape[1])
        out = C.decoder(engine.model)(input_ids=inputs[:, start:end], position_ids=positions[:, start:end],
                                      past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        assert cache.get_seq_length() == end
    return cache, engine.model.lm_head(out.last_hidden_state[:, -1]).float()


class Recolor:
    def __init__(self, engine, prior):
        map_path = Path(C.CFG["recolor"])
        checkpoint = C.ROOT / "runs/C/epoch4_step800.pt"
        assert C.sha(map_path) == C.CFG["recolor_sha256"]
        assert C.sha(checkpoint) == prior["recolor_checkpoint_sha256"]
        self.engine = engine
        self.mapping = C.load_map(map_path, device="cuda", dtype=torch.float32)
        self.correction = C.fresh("C")
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert weights["manifest_sha256"] == C.identity() and weights["step"] == 800 and weights["arm"] == "C"
        assert weights["recolor_sha256"] == C.CFG["recolor_sha256"]
        self.correction.load_state_dict(weights["state"], strict=True)
        self.correction.eval().requires_grad_(False)
        stencil = self.mapping["depth_stencil"]
        self.taps = tuple(tuple(tuple(map(int, layer)) for layer in stencil["taps_" + role]) for role in ("k","v"))
        self.rms = tuple(tuple(map(float, self.mapping["rms_" + role])) for role in ("k","v"))
        self.weights = tuple(tuple(self.mapping["A_" + role]) for role in ("k","v"))
        self.bias = tuple(tuple(self.mapping["b_" + role]) for role in ("k","v"))
        self.used = tuple(sorted({tap for layer in role for tap in layer}) for role in self.taps)

    def align(self, rows, sources):
        indices = []
        for row, (ids, _) in zip(rows, sources):
            sender = self.engine.sender_tok(row["prompt"], add_special_tokens=False,
                                            return_offsets_mapping=True, verbose=False)
            receiver = self.engine.tok(row["prompt"], add_special_tokens=False,
                                       return_offsets_mapping=True, verbose=False)
            assert sender["input_ids"] == ids and receiver["input_ids"] == row["content_ids"]
            indices.append(torch.tensor(receiver_to_sender_indices(sender["offset_mapping"],
                                                                   receiver["offset_mapping"]), device="cpu"))
        return indices

    @torch.no_grad()
    def build(self, rows, sources, clock):
        engine = self.engine
        model = engine.model
        length = len(rows[0]["content_ids"])
        header = len(engine.prefix_ids)
        indices = clock.measure("alignment", self.align, rows, sources)
        assert all(len(index) == length for index in indices)

        def allocate():
            pairs = []
            for pair in K.pairs(engine.header):
                expanded = []
                for x in pair:
                    buf = x.new_empty((B.BATCH, x.shape[1], length + header, x.shape[-1]))
                    buf[:, :, :header].copy_(x.expand(B.BATCH, -1, -1, -1))
                    expanded.append(buf)
                pairs.append(tuple(expanded))
            return pairs

        pairs = clock.measure("cache_allocate", allocate)
        for begin in range(0, length, B.CHUNK):
            end = min(begin + B.CHUNK, length)
            width = end - begin

            def upload():
                return tuple({tap: torch.cat([
                    raw[role][tap].index_select(0, index[begin:end]) for index, (_, raw) in zip(indices, sources)
                ]).to("cuda") for tap in taps} for role, taps in enumerate(self.used))

            features = clock.measure("gather_and_upload", upload)

            def translate():
                mapped = []
                for layer in range(36):
                    result = []
                    for role, correct in enumerate((self.correction[layer].key, self.correction[layer].value)):
                        x = torch.cat([features[role][tap].float() for tap in self.taps[role][layer]], dim=-1)
                        recon = (x / self.rms[role][layer]) @ self.weights[role][layer] + self.bias[role][layer]
                        result.append(correct(recon.to(model.dtype).float()))
                    mapped.append(tuple(result))
                return mapped

            mapped = clock.measure("translation", translate)
            del features

            def finalize():
                positions = torch.arange(header+begin, header+end, device="cuda").repeat(B.BATCH)
                tables = C._rotary_tables(model, positions)
                for layer,(key,value) in enumerate(mapped):
                    data = C.finalize_layer(model, layer, key, value, *tables(layer))
                    for source, dest in zip(data, pairs[layer]):
                        heads, _, dim = source.shape
                        dest[:, :, header+begin:header+end].copy_(
                            source.reshape(heads, B.BATCH, width, dim).transpose(0, 1))

            clock.measure("cache_finalize", finalize)
            del mapped
        return clock.measure("cache_handle", B.place, engine, pairs)


@torch.no_grad()
def measure(engine, rows, *, block=None, recolor=None, sources=None, profile=False, decode=False):
    clock = B.R.Stopwatch(profile)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    if recolor is None:
        body, logits = clock.measure("native_tokenizer_prefill_and_head", native_first, engine, rows, block)
    else:
        body = recolor.build(rows, sources, clock)
        ids = torch.tensor([row["suffix_ids"] for row in rows], device="cuda")
        logits = clock.measure("receiver_suffix_and_head", B.receiver_step, engine, body, ids)
    next_token = logits.argmax(-1)
    first_tokens = next_token.tolist()
    ttft = (perf_counter()-start) * 1000
    peak = torch.cuda.max_memory_allocated()
    per_step = None
    if decode:
        start = perf_counter()
        for _ in range(B.STEPS):
            next_token = B.receiver_step(engine, body, next_token[:, None]).argmax(-1)
        torch.cuda.synchronize()
        per_step = (perf_counter()-start) * 1000 / B.STEPS
    assert engine.student_test_native_prefills == 0
    return dict(ttft_ms=ttft, decode_ms_per_step=per_step, first_tokens=first_tokens,
                peak_allocated_bytes=peak, peak_with_decode_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), phases_ms=clock.stages)


def native_test(engine, rows):
    for block in (None, 4096, 2048, 1024):
        try:
            warm = measure(engine, rows, block=block)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        else:
            trials = [measure(engine, rows, block=block, decode=True) for _ in range(B.TRIALS)]
            assert all(t["first_tokens"] == warm["first_tokens"] for t in trials)
            return dict(native_prefill_block=block, trials=trials)
    raise RuntimeError("Native prefill could not fit at the smallest block")


@torch.no_grad()
def verify_recolor(engine, rows, recolor, sources):
    one = rows[0]
    builder = C.HC.ExampleBuilder((engine.sender_tok, engine.sender),
                                   (engine.tok, engine.model), recolor.mapping, "cuda")
    offsets = engine.tok(one["prompt"], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    expected = C.Engine.cache(engine, one, builder.recolor_outputs(one["prompt"], offsets), recolor.correction)
    actual = recolor.build(rows, sources, B.R.Stopwatch(False))
    errors = []
    for a, b in zip(K.flat(actual), K.flat(expected)):
        x, y = a[:1].float(), b.float()
        errors.append(float((x-y).norm()/y.norm().clamp_min(1e-12)))
    assert max(errors) <= .003, max(errors)
    return max(errors)


def recolor_test(engine, rows, recolor, *, check=False):
    with TemporaryDirectory(prefix="recolor_kv_", dir=OUT) as work:
        sources = [S.capture(engine, row, Path(work) / f"{i}.bin") for i,row in enumerate(rows)]
        torch.cuda.empty_cache()
        parity = verify_recolor(engine, rows, recolor, sources) if check else None

        def trial(**kwargs):
            for _, raw in sources:
                raw.view(-1)[::2048].sum().item()
            before = S.counters()
            result = measure(engine, rows, recolor=recolor, sources=sources, **kwargs)
            after = S.counters()
            result.update(timed_storage_read_bytes=after[0]-before[0], timed_major_page_faults=after[1]-before[1])
            return result

        warm = trial()
        trials = [trial(decode=True) for _ in range(B.TRIALS)]
        profile = trial(profile=True)
        assert all(r["first_tokens"] == warm["first_tokens"] for r in [*trials,profile])
        del sources
        return dict(trials=trials, profile=profile, max_cache_relative_l2=parity)


def summarize(records, pin, complete=False):
    gated = C.read(ROOT / "summary.json")
    lengths = {}
    for length in map(str, B.LENGTHS):
        arms = {}
        for name in ("native","recolor_c"):
            items = [v for (arm,n,_),v in records.items() if n==int(length) and arm==name]
            if not items:
                continue
            flat = [t for x in items for t in x["trials"]]
            arms[name] = dict(batches=len(items), ttft_ms=B.R.stats([t["ttft_ms"] for t in flat]),
                              decode_ms_per_step=B.R.stats([t["decode_ms_per_step"] for t in flat]),
                              peak_allocated_bytes=max(t["peak_allocated_bytes"] for t in flat))
            if name=="native":
                arms[name]["prefill_blocks"] = [v["native_prefill_block"] for v in items]
            else:
                arms[name]["timed_storage_read_bytes"] = [t["timed_storage_read_bytes"] for t in flat]
        if length in gated["lengths"]:
            measured = gated["lengths"][length]
            arms["gated_dynamic"] = measured
            if measured["batches"] == 2:
                arms["gated_over_baseline"] = {
                    arm: dict(ttft_ratio=measured["ttft_ms"]["p50"]/v["ttft_ms"]["p50"],
                              delta_ms=measured["ttft_ms"]["p50"]-v["ttft_ms"]["p50"])
                    for arm,v in arms.items() if arm in {"native","recolor_c"} and v["batches"]==2
                }
        if arms:
            lengths[length] = arms
    C.save(OUT / "summary.json", dict(status="completed" if complete else "running", manifest_sha256=pin,
                                      lengths=lengths, gated_status=gated["status"],
                                      gated_storage_32k=gated.get("storage_32k")))


def main():
    trial_manifest = C.read(ROOT / "manifest.json")
    prior = C.read(B.R.E4 / "run_manifest.json")
    status("loading")
    engine = B.Engine()
    engine.mode = "test"
    del engine.correction
    del engine.dst
    cohorts = B.inputs(engine)
    prompts = {str(n): [[{k:r[k] for k in ("article_ids","prompt_sha256")} for r in g]
                        for g in items] for n,items in cohorts.items()}
    assert C.digest(prompts) == C.digest(trial_manifest["prompts"])
    manifest = dict(pinned_gated_manifest=C.sha(ROOT/"manifest.json"), baseline_script=C.sha(Path(__file__)),
                    inputs=prompts, device=torch.cuda.get_device_name(), chunk=B.CHUNK, batch=B.BATCH,
                    trials=B.TRIALS, decode_steps=B.STEPS, recolor_checkpoint=prior["recolor_checkpoint_sha256"],
                    modes={"native":"Qwen original text", "recolor_c":"Sender KV already available on CPU"},
                    native_prefill_fallback=[4096,2048,1024], recolor_storage="warmed file-backed CPU")
    path = OUT / "manifest.json"
    if path.exists():
        assert C.digest(manifest)==C.digest(C.read(path))
    else:
        C.save(path,manifest)
    pin = C.sha(path)
    records = {}
    for arm in ("native","recolor_c"):
        recolor = Recolor(engine, prior) if arm=="recolor_c" else None
        for length, groups in cohorts.items():
            for i, rows in enumerate(groups):
                status("measuring", arm=arm, length=length, batch_index=i+1)
                file = OUT / "batches" / f"{arm}_{length}_{i}.json"
                if file.exists():
                    result = C.read(file)
                    assert result["manifest_sha256"]==pin
                else:
                    result = (native_test(engine, rows) if recolor is None else
                              recolor_test(engine, rows, recolor, check=length == B.LENGTHS[0] and i == 0))
                    result.update(manifest_sha256=pin, length=length, batch_index=i+1, arm=arm)
                    C.save(file,result)
                records[arm,length,i] = result
                summarize(records,pin)
                print(arm,length,i+1,[round(t["ttft_ms"],1) for t in result["trials"]],flush=True)
    summarize(records,pin,complete=True)
    status("completed")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    status("queued")
    with (ROOT/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            main()
        except Exception as exc:
            status("failed",error=repr(exc))
            raise
