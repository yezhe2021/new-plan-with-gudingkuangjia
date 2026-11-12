"""Measure Qwen prefill and two exact-arithmetic cache handoff paths."""

import json
from pathlib import Path
import random
from statistics import median
import sys
from time import perf_counter

from prepare import ROOT, PROJECT, save, sha

RANK64 = PROJECT / "dst_stage1_rank64_then_static_m32_a4_seed1234"
sys.path.insert(0, str(RANK64 / "static_memory"))
from experiment_common import B, C, K, torch
from engine_frozen import Engine

sys.path.insert(0, str(B.D.ROOT))
from alignment import fullsync
from heterofold.alignment import receiver_to_sender_indices
from heterofold.capture import capture_kv
from heterofold.transfer import map_layer

TRIALS = 3
DECODE_STEPS = 32
METHODS = ("native", "recolor_c", "dst_rank64")
MODES = {
    "native": ("ready", "text_ttft"),
    "recolor_c": ("ready", "handoff_ttft", "text_ttft"),
    "dst_rank64": ("ready", "handoff_ttft", "text_ttft"),
}


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


class Stopwatch:
    def __init__(self, split):
        self.split = split
        self.stages = {}

    def measure(self, name, fn, *args):
        if not self.split:
            return fn(*args)
        torch.cuda.synchronize()
        start = perf_counter()
        value = fn(*args)
        torch.cuda.synchronize()
        self.stages[name] = self.stages.get(name, 0.0) + 1000 * (perf_counter() - start)
        return value


class Benchmark:
    def __init__(self):
        self.engine = Engine()
        self.engine.mode = "test"
        self.model = self.engine.model
        self.body_decoder = C.decoder(self.model)
        self.device = next(self.model.parameters()).device
        self.header_length = len(self.engine.prefix_ids)
        self.builder = C.HC.ExampleBuilder(
            (self.engine.sender_tok, self.engine.sender), (self.engine.tok, self.model),
            C.load_map(C.CFG["recolor"]), str(self.device),
        )
        assert self.builder.prefix_ids.tolist() == self.engine.prefix_ids
        self.recolor = self.builder.mapping
        path = C.ROOT / "runs/C/epoch4_step800.pt"
        cp = torch.load(path, map_location="cpu", weights_only=True)
        assert cp["manifest_sha256"] == C.identity() and cp["arm"] == "C" and cp["step"] == 800
        assert cp["recolor_sha256"] == C.CFG["recolor_sha256"] == C.sha(C.CFG["recolor"])
        self.calibration = C.fresh("C")
        self.calibration.load_state_dict(cp["state"], strict=True)
        self.calibration.eval().requires_grad_(False)
        self.all_layers = tuple(range(len(C.layers_of(self.engine.sender))))
        self.chunk = B.D.CFG["token_chunk"]

    def capture(self, row, method):
        layers = self.all_layers if method == "dst_rank64" else self.builder.source_layers
        return capture_kv(self.engine.sender_tok, self.engine.sender, row["prompt"], layers,
                          device="cpu", dtype=self.engine.sender.dtype)

    def align(self, row, method, sender_offsets):
        if method == "dst_rank64":
            return fullsync(row["prompt"], self.engine.sender_tok, self.engine.tok,
                            row["content_ids"]).source_indices
        encoded = self.engine.tok(row["prompt"], add_special_tokens=False, return_offsets_mapping=True)
        assert encoded["input_ids"] == row["content_ids"]
        return receiver_to_sender_indices(sender_offsets, encoded["offset_mapping"])

    def upload(self, keys, values, method):
        layers = self.all_layers if method == "dst_rank64" else self.builder.source_layers
        return tuple({l: group[l].to(self.device) for l in layers} for group in (keys, values))

    def assemble(self, pieces):
        layers = K.pairs(self.engine.header)
        return C.DynamicCache(ddp_cache_data=[
            (torch.cat([key, *pieces[l][0]], dim=2), torch.cat([value, *pieces[l][1]], dim=2))
            for l, (key, value) in enumerate(layers)
        ], config=self.model.config)

    def cache_from_source(self, row, method, source, clock):
        offsets, keys, values = source
        indices = clock.measure("alignment", self.align, row, method, offsets)
        keys, values = clock.measure("host_to_device", self.upload, keys, values, method)
        pieces = [([], []) for _ in C.layers_of(self.model)]
        if method == "dst_rank64":
            source_k = tuple(keys[l] for l in self.all_layers)
            source_v = tuple(values[l] for l in self.all_layers)
            for start in range(0, len(indices), self.chunk):
                end = min(start + self.chunk, len(indices))
                key, value = clock.measure("mapping", self.engine.correction, source_k, source_v, indices[start:end])

                def finalize():
                    positions = torch.arange(self.header_length + start, self.header_length + end, device=self.device)
                    return B.Z.DE.finalize_raw(self.model, key, value, positions)

                key, value = clock.measure("cache_finalize", finalize)
                for layer, (kk, vv) in enumerate(pieces):
                    kk.append(key[layer][None])
                    vv.append(value[layer][None])
        else:
            def setup():
                return torch.tensor(indices, device=self.device), C._rotary_tables(
                    self.model, torch.arange(self.header_length, self.header_length + len(indices), device=self.device))

            index, tables = clock.measure("cache_finalize", setup)
            for layer, (kk, vv) in enumerate(pieces):
                def translate():
                    key, value = map_layer(self.recolor, keys, values, index, layer)
                    key, value = key.to(self.model.dtype).float(), value.to(self.model.dtype).float()
                    correction = self.calibration[layer]
                    return correction.key(key), correction.value(value)

                key, value = clock.measure("mapping", translate)

                def finalize():
                    return C.finalize_layer(self.model, layer, key, value, *tables(layer))

                key, value = clock.measure("cache_finalize", finalize)
                kk.append(key[None])
                vv.append(value[None])
        return clock.measure("cache_assemble", self.assemble, pieces)

    def native(self, row, ready, clock):
        ids = clock.measure("receiver_tokenize", lambda: self.engine.tok.encode(row["prompt"], add_special_tokens=False))
        assert ids == row["content_ids"]
        if not ready:
            ids += row["suffix_ids"]
        return clock.measure("receiver_prefill", lambda: self.engine.forward(
            ids, K.clone(self.engine.header, self.model, detach=True), self.header_length,
        ))

    def first_token(self, output):
        return int(self.model.lm_head(output.last_hidden_state[:, -1:]).argmax())

    @torch.no_grad()
    def measure(self, row, method, mode, source=None, *, split=False, decode=False):
        clock = Stopwatch(split)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        if method == "native":
            output = self.native(row, mode == "ready", clock)
            cache = output.past_key_values
        else:
            if source is None:
                source = clock.measure("sender_prefill_capture", self.capture, row, method)
            cache = self.cache_from_source(row, method, source, clock)
            if mode != "ready":
                output = clock.measure("receiver_suffix", lambda: self.engine.forward(
                    row["suffix_ids"], cache, self.header_length + row["length"],
                ))
                cache = output.past_key_values
        token = None if mode == "ready" else clock.measure("first_token_head", self.first_token, output)
        torch.cuda.synchronize()
        elapsed = 1000 * (perf_counter() - start)
        assert cache.get_seq_length() == self.header_length + row["length"] + (
            0 if mode == "ready" else len(row["suffix_ids"]))
        result = dict(
            elapsed_ms=elapsed, phase_ms=clock.stages, first_token=token,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_delta_bytes=torch.cuda.max_memory_allocated() - baseline,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            receiver_cache_bytes=tensor_bytes(K.flat(cache)),
            source_available_bytes=0 if source is None else tensor_bytes(
                tensor for group in source[1:] for tensor in group.values()),
        )
        if decode:
            assert token is not None
            del output
            start = perf_counter()
            for _ in range(DECODE_STEPS):
                output = self.engine.forward([token], cache, cache.get_seq_length())
                token = self.first_token(output)
                cache = output.past_key_values
            torch.cuda.synchronize()
            result["decode_ms_per_token"] = 1000 * (perf_counter() - start) / DECODE_STEPS
        return result

    @torch.no_grad()
    def smoke(self, row):
        source = self.capture(row, "dst_rank64")
        offsets, keys, values = source
        for method in METHODS[1:]:
            measured = self.cache_from_source(row, method, source, Stopwatch(False))
            if method == "dst_rank64":
                indices = self.align(row, method, offsets)
                expected = self.engine.translated(dict(row, sync_source_indices=indices),
                                                  self.engine.correction,
                                                  source=(tuple(keys.values()), tuple(values.values())),
                                                  checkpointed=False)
            else:
                receiver = self.engine.tok(row["prompt"], add_special_tokens=False, return_offsets_mapping=True)
                key, value = self.builder.recolor_outputs(row["prompt"], receiver["offset_mapping"])
                expected = C.Engine.cache(self.engine, row, (key, value), self.calibration)
            assert len(K.flat(measured)) == len(K.flat(expected)) == 72
            for actual, old in zip(K.flat(measured), K.flat(expected)):
                torch.testing.assert_close(actual, old, atol=0.003, rtol=0.003)
            for cache in (measured, expected):
                assert cache.get_seq_length() == self.header_length + row["length"]
            out = self.engine.forward(row["suffix_ids"], measured, measured.get_seq_length())
            reference = self.engine.forward(row["suffix_ids"], expected, expected.get_seq_length())
            assert self.first_token(out) == self.first_token(reference)
        native = self.native(row, False, Stopwatch(False))
        direct = self.body_decoder(torch.tensor([self.engine.prefix_ids + row["content_ids"] + row["suffix_ids"]],
                                               device=self.device), use_cache=True)
        assert self.first_token(native) == self.first_token(direct)
        print("Smoke passed: paired cache and first-token parity", flush=True)


def stats(values):
    ordered = sorted(values)
    return dict(n=len(values), p50=median(values), p95=ordered[(95 * len(values) + 99) // 100 - 1])


def summarize(rows, manifest):
    summary = {}
    for length in manifest["lengths"]:
        entries = [x for x in rows if x["length"] == length]
        if not entries:
            continue
        methods = {}
        for method in METHODS:
            methods[method] = {}
            for mode in MODES[method]:
                runs = [r for e in entries for r in e["trials"] if r["method"] == method and r["mode"] == mode]
                phases = [r["phase_ms"] for e in entries for r in e["diagnostics"] if r["method"] == method and r["mode"] == mode]
                report = dict(
                    latency_ms=stats([r["elapsed_ms"] for r in runs]),
                    peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in runs),
                    peak_delta_bytes=max(r["peak_delta_bytes"] for r in runs),
                    peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in runs),
                    receiver_cache_bytes=median(r["receiver_cache_bytes"] for r in runs),
                    source_available_bytes=median(r["source_available_bytes"] for r in runs),
                )
                if phases:
                    report["phase_p50_ms"] = {k: median(p[k] for p in phases) for k in phases[0]}
                decode = [r["decode_ms_per_token"] for e in entries for r in e["diagnostics"]
                          if r["method"] == method and r["mode"] == mode and "decode_ms_per_token" in r]
                if decode:
                    report["decode_ms_per_token"] = stats(decode)
                if method != "native":
                    native_mode = "ready" if mode == "ready" else "text_ttft"
                    ratios = []
                    for entry in entries:
                        baseline = [r["elapsed_ms"] for r in entry["trials"] if r["method"] == "native" and r["mode"] == native_mode]
                        current = [r["elapsed_ms"] for r in entry["trials"] if r["method"] == method and r["mode"] == mode]
                        ratios.append(median(baseline) / median(current))
                    report["paired_speedup"] = stats(ratios)
                methods[method][mode] = report
        summary[length] = dict(articles=len(entries), methods=methods)
    return summary


@torch.no_grad()
def main():
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    assert manifest["input_sha256"] == sha(ROOT / "inputs.json")
    inputs = json.loads((ROOT / "inputs.json").read_text())
    bench = Benchmark()
    bench.smoke(inputs[0])
    plan = [(method, mode) for method in METHODS for mode in MODES[method]]
    run_info = dict(
        batch=1, device=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
        receiver_attention=bench.model.config._attn_implementation,
        sender_attention=bench.engine.sender.config._attn_implementation,
        dtype=str(bench.model.dtype), dst_mapping_dtype=str(next(bench.engine.correction.parameters()).dtype),
        trials=TRIALS, decode_steps=DECODE_STEPS, data_manifest_sha256=sha(ROOT / "data_manifest.json"),
        code_sha256=sha(Path(__file__)), dst_checkpoint_sha256=C.read(RANK64 / "runs/stage2/completed.json")["checkpoint_sha256"],
        recolor_checkpoint_sha256=C.sha(C.ROOT / "runs/C/epoch4_step800.pt"),
        residency="All checkpoints resident; report allocated/reserved peak and peak above resident baseline",
        handoff_start="compatible pre-RoPE CPU K/V available; upload once, on a single GPU",
        timing="Unsegmented wall time is primary; synchronized stage profile and 32 decode steps measured separately",
    )
    manifest_file = ROOT / "run_manifest.json"
    if manifest_file.exists():
        assert C.read(manifest_file) == run_info
    else:
        save(manifest_file, run_info)
    records_dir = ROOT / "per_prefix"
    records_dir.mkdir(exist_ok=True)
    records = []
    for length in manifest["lengths"]:
        group = [row for row in inputs if row["length"] == length]
        warm = group[0]
        if not all((records_dir / (row["id"] + ".json")).exists() for row in group):
            source = bench.capture(warm, "dst_rank64")
            for _ in range(2):
                for method, mode in plan:
                    ready = source if method != "native" and mode != "text_ttft" else None
                    bench.measure(warm, method, mode, ready)
            del source
        for row in group:
            path = records_dir / (row["id"] + ".json")
            if not path.exists():
                source = bench.capture(row, "dst_rank64")
                results = []
                for repeat in range(TRIALS):
                    order = plan.copy()
                    random.Random(f"{row['id']}:{repeat}").shuffle(order)
                    for method, mode in order:
                        ready = source if method != "native" and mode != "text_ttft" else None
                        result = bench.measure(row, method, mode, ready)
                        results.append(dict(method=method, mode=mode, trial=repeat, **result))
                diagnostics = []
                for method, mode in plan:
                    ready = source if method != "native" and mode != "text_ttft" else None
                    decode = mode == ("text_ttft" if method == "native" else "handoff_ttft")
                    diagnostics.append(dict(method=method, mode=mode,
                                            **bench.measure(row, method, mode, ready, split=True, decode=decode)))
                save(path, dict(id=row["id"], length=length, trials=results, diagnostics=diagnostics))
                del source
            records.append(C.read(path))
            save(ROOT / "summary.json", dict(status="running", n=len(records), total=len(inputs), lengths=summarize(records, manifest)))
            print(f"{len(records)}/{len(inputs)} {row['id']}", flush=True)
    save(ROOT / "summary.json", dict(status="completed", n=len(records), total=len(inputs), lengths=summarize(records, manifest)))


if __name__ == "__main__":
    main()
