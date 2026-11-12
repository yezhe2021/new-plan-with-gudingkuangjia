"""Paired 8K inference sweep using the E4 inputs and frozen DST checkpoint."""

from collections import defaultdict
import fcntl
import gc
import json
from pathlib import Path
import random
from statistics import median
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from benchmark import B, Benchmark, C, K, Stopwatch, stats, torch
from prepare import save, sha

CHUNKS = (128, 512, 1024, 2048, 8192)
PATHS = ("legacy", "gpu_index", "gpu_aligned")
TRIALS = 3
MODES = ("ready", "handoff_ttft")


class ChunkBenchmark(Benchmark):
    def cache_from_source(self, row, method, source, clock):
        if self.path == "legacy":
            return super().cache_from_source(row, method, source, clock)
        offsets, keys, values = source
        indices = clock.measure("alignment", self.align, row, method, offsets)
        keys, values = clock.measure("host_to_device", self.upload, keys, values, method)
        index = clock.measure("gpu_index_upload", lambda: torch.tensor(indices, device=self.device, dtype=torch.long))
        module = self.engine.correction
        base = module.base
        dtype = next(base.parameters()).dtype
        aligned = self.path == "gpu_aligned"
        if aligned:
            def gather():
                return tuple({
                    layer: group[layer].index_select(0, index).to(dtype=dtype)
                    for layer in sorted(set(base.taps_tuple))
                } for group in (keys, values))
            keys, values = clock.measure("gpu_prealignment", gather)
        pieces = [([], []) for _ in base.layers]

        for start in range(0, len(indices), self.chunk):
            stop = min(start + self.chunk, len(indices))

            def translate():
                local_index = index[start:stop]
                state = torch.zeros(stop - start, base.cfg.state_width, device=self.device, dtype=dtype)
                translated_k = []
                translated_v = []
                for l, (tap, block) in enumerate(zip(base.taps_tuple, base.layers)):
                    if aligned:
                        key, value = keys[tap][start:stop], values[tap][start:stop]
                    else:
                        key = keys[tap].index_select(0, local_index).to(dtype=dtype)
                        value = values[tap].index_select(0, local_index).to(dtype=dtype)
                    previous = state if base.cfg.carry_state else torch.zeros_like(state)
                    key, value, state = block(key, value, previous)
                    translated_k.append(key + module.keys[l](key))
                    translated_v.append(value + module.values[l](value))
                return torch.stack(translated_k), torch.stack(translated_v)

            key, value = clock.measure("mapping", translate)

            def finalize():
                positions = torch.arange(self.header_length + start, self.header_length + stop, device=self.device)
                return B.Z.DE.finalize_raw(self.model, key, value, positions)

            key, value = clock.measure("cache_finalize", finalize)
            for l, (kk, vv) in enumerate(pieces):
                kk.append(key[l][None])
                vv.append(value[l][None])
        return clock.measure("cache_assemble", self.assemble, pieces)


def status(state, **details):
    save(ROOT / "status.json", dict(status=state, time=time.strftime("%F %T"), **details))


def probe(fn):
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    return None


@torch.no_grad()
def reference(bench, row, source):
    bench.path, bench.chunk = "legacy", 128
    cache = bench.cache_from_source(row, "dst_rank64", source, Stopwatch(False))
    values = tuple(t.cpu() for t in K.flat(cache))
    out = bench.engine.forward(row["suffix_ids"], cache, cache.get_seq_length())
    return values, bench.first_token(out)


@torch.no_grad()
def check(bench, row, source, expected, expected_token):
    cache = bench.cache_from_source(row, "dst_rank64", source, Stopwatch(False))
    assert cache.get_seq_length() == bench.header_length + row["length"]
    tensors = K.flat(cache)
    assert len(tensors) == len(expected)
    max_error = torch.zeros((), device=bench.device)
    all_close = torch.ones((), dtype=torch.bool, device=bench.device)
    for current, ref in zip(tensors, expected):
        ref = ref.to(bench.device)
        error = (current.float() - ref.float()).abs()
        max_error = torch.maximum(max_error, error.max())
        all_close &= (error <= 0.003 + 0.003 * ref.float().abs()).all()
    out = bench.engine.forward(row["suffix_ids"], cache, cache.get_seq_length())
    token = bench.first_token(out)
    return dict(cache_within_tolerance=bool(all_close), max_abs_cache_error=float(max_error),
                first_token=token, reference_token=expected_token, first_token_matches=token == expected_token)


def variant_id(path, chunk):
    return f"{path}_{chunk}"


def summarize(records, manifest, complete):
    names = [(p, n) for p in PATHS for n in CHUNKS]
    native = {}
    for mode in MODES:
        runs = [run for r in records for run in r["native"][mode]]
        native[mode] = stats([r["elapsed_ms"] for r in runs])
    variants = {}
    for p, n in names:
        key = variant_id(p, n)
        attempts = [row["variants"][key] for row in records]
        flags = [r["status"] for r in attempts]
        report = dict(path=p, chunk=n, per_article_status=flags, articles=flags.count("completed"),
                      parity=[r.get("parity") for r in attempts])
        for mode in MODES:
            comparisons = [(r, a) for r, a in zip(records, attempts) if a["status"] == "completed"]
            runs = [v for _, a in comparisons for v in a["trials"][mode]]
            if not runs:
                continue
            ratios = [
                median(t["elapsed_ms"] for t in row["native"][mode]) /
                median(t["elapsed_ms"] for t in arm["trials"][mode])
                for row, arm in comparisons
            ]
            report[mode] = dict(
                latency_ms=stats([v["elapsed_ms"] for v in runs]),
                paired_native_speedup=stats(ratios),
                peak_allocated_bytes=max(v["peak_allocated_bytes"] for v in runs),
                peak_delta_bytes=max(v["peak_delta_bytes"] for v in runs),
                peak_reserved_bytes=max(v["peak_reserved_bytes"] for v in runs),
            )
        profiles = [a["profile"] for a in attempts if a["status"] == "completed"]
        if profiles:
            report["handoff_stage_p50_ms"] = {k: median(p[k] for p in profiles) for k in profiles[0]}
        variants[key] = report
    save(ROOT / "summary.json", dict(status="completed" if complete else "running",
         manifest=manifest, articles=len(records), native=native, variants=variants))


@torch.no_grad()
def main():
    parent = ROOT.parent
    data = json.loads((parent / "data_manifest.json").read_text())
    prior = json.loads((parent / "run_manifest.json").read_text())
    assert data["input_sha256"] == sha(parent / "inputs.json")
    assert prior["data_manifest_sha256"] == sha(parent / "data_manifest.json")
    assert prior["code_sha256"] == sha(parent / "benchmark.py")
    rows = [r for r in json.loads((parent / "inputs.json").read_text()) if r["length"] == 8192]
    assert len(rows) == 2
    manifest = dict(input_ids=[r["id"] for r in rows], chunks=CHUNKS, paths=PATHS, trials=TRIALS,
                    prior_run_sha256=sha(parent / "run_manifest.json"), script_sha256=sha(Path(__file__)),
                    dst_checkpoint_sha256=prior["dst_checkpoint_sha256"], device=torch.cuda.get_device_name(),
                    input_residency="compatible CPU raw K/V, upload included in all handoff arms",
                    cache_check="first token exact; all K/V values within atol=rtol=0.003")
    manifest = json.loads(json.dumps(manifest))
    path = ROOT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        save(path, manifest)
    bench = ChunkBenchmark()
    records = []
    for i, row in enumerate(rows, 1):
        path = ROOT / (row["id"] + ".json")
        if path.exists():
            records.append(json.loads(path.read_text()))
            continue
        source = bench.capture(row, "dst_rank64")
        expected, token = reference(bench, row, source)
        native = defaultdict(list)
        variants = {}
        valid = []
        for p in PATHS:
            for n in CHUNKS:
                key = variant_id(p, n)
                bench.path, bench.chunk = p, n
                status("running", phase="parity", article=i, total=len(rows), variant=key)
                parity = probe(lambda: check(bench, row, source, expected, token))
                if parity is None:
                    variants[key] = dict(status="oom", stage="parity")
                elif not (parity["cache_within_tolerance"] and parity["first_token_matches"]):
                    variants[key] = dict(status="parity_failed", parity=parity)
                else:
                    variants[key] = dict(status="running", parity=parity, trials={m: [] for m in MODES})
                    valid.append((p, n))
                print(i, key, parity if parity is not None else "OOM", flush=True)
        del expected
        for m in MODES:
            bench.measure(row, "native", "ready" if m == "ready" else "text_ttft")
        for trial in range(TRIALS):
            plans = [(p, n, m) for p, n in valid for m in MODES] + [("native", 0, m) for m in MODES]
            random.Random(f"{row['id']}:{trial}").shuffle(plans)
            for p, n, mode in plans:
                key = variant_id(p, n)
                status("running", phase="timing", article=i, total=len(rows), variant=key, mode=mode, trial=trial + 1)
                if p == "native":
                    native[mode].append(bench.measure(row, "native", "ready" if mode == "ready" else "text_ttft"))
                elif variants[key]["status"] == "running":
                    bench.path, bench.chunk = p, n
                    result = probe(lambda: bench.measure(row, "dst_rank64", mode, source))
                    if result is None:
                        variants[key].update(status="oom", stage="timing")
                    else:
                        variants[key]["trials"][mode].append(result)
            print("Article", i, "timing trial", trial + 1, "/", TRIALS, flush=True)
        for p, n in valid:
            arm = variants[variant_id(p, n)]
            if arm["status"] != "running":
                continue
            bench.path, bench.chunk = p, n
            result = probe(lambda: bench.measure(row, "dst_rank64", "handoff_ttft", source, split=True))
            if result is None:
                arm.update(status="oom", stage="profiling")
            else:
                arm.update(status="completed", profile=result["phase_ms"])
        record = dict(id=row["id"], native=native, variants=variants)
        save(path, record)
        records.append(record)
        summarize(records, manifest, complete=False)
    summarize(records, manifest, complete=True)
    status("completed", articles=len(records))


if __name__ == "__main__":
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status("running", phase="loading")
            main()
        except Exception as error:
            status("failed", error=repr(error))
            raise
