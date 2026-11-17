"""Time the frozen Gated Dynamic Memory model on the existing 8K E4 workload."""
import fcntl
from pathlib import Path
import random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent
E4 = ROOT.parent / "dst_e4_quality_latency_seed1234"
JOINT = ROOT.parent / "dst_rank64_gated_joint_m32_a4_seed1234"
sys.path.insert(0, str(E4 / "chunk_sweep"))
from sweep import B, C, K, Stopwatch, ChunkBenchmark, stats, torch
from benchmark import tensor_bytes

METHOD = "dst_rank64"
TRIALS = 3
DECODE_STEPS = 32


def load_memory(engine):
    settings = C.read(JOINT / "config.json")
    source = JOINT / "initial"
    weights = source / "runs/epoch4_step1536.pt"
    joint_manifest = C.read(JOINT / "manifest.json")
    original = source / "gated_memory.py"
    for path in (weights, original, source / "manifest.json"):
        assert C.sha(path) == joint_manifest["pins"][str(path)], path
    assert C.sha(source / "manifest.json") == settings["gated_manifest_sha256"]
    assert C.sha(weights) == settings["gated_checkpoint_sha256"]
    module = B.Z.scoped_module("gated_latency_common", JOINT / "experiment_common.py")
    dynamic_path = str(module.reference.DYNAMIC_ROOT)
    sys.path.insert(0, dynamic_path)
    try:
        implementation = B.Z.scoped_module("gated_latency_original", original, module)
    finally:
        sys.path.remove(dynamic_path)
    data = torch.load(weights, map_location="cpu", weights_only=True)
    assert data["manifest_sha256"] == settings["gated_manifest_sha256"] and data["step"] == 1536
    memory = implementation.GatedDynamicMemory(engine, module.CFG)
    memory.load_state_dict(data["state"], strict=True)
    memory.eval().requires_grad_(False)
    assert memory.M == 32 and memory.gates.numel() == 36 and torch.count_nonzero(memory.gates) > 0
    return memory, weights


@torch.no_grad()
def measure(bench, memory, row, source=None, *, profile=False, decode=False):
    clock = Stopwatch(profile)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    start = perf_counter()
    if source is None:
        source = clock.measure("sender_prefill", bench.capture, row, METHOD)
    body = bench.cache_from_source(row, METHOD, source, clock)
    body_len = body.get_seq_length()
    assert body_len == bench.header_length + row["length"]
    if profile:
        torch.cuda.synchronize()
        construct_baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    bank = clock.measure("memory_construct", memory.construct, row, body)
    construct_extra = torch.cuda.max_memory_allocated() - construct_baseline if profile else None
    assert bank.body is body and len(bank.bank) == 72 and all(t.shape[2] == memory.M for t in bank.bank)
    body_bytes = tensor_bytes(K.flat(body))
    memory_bytes = tensor_bytes(bank.bank)
    engine = bench.engine
    with memory.answering(bank.bank, memory.gates):
        output = clock.measure("receiver_suffix", engine.forward, row["suffix_ids"], body, body_len)
        token = clock.measure("first_token_head", bench.first_token, output)
        torch.cuda.synchronize()
        ttft = (perf_counter() - start) * 1000
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        decode_ms = None
        if decode:
            cache = output.past_key_values
            last = token
            start = perf_counter()
            for _ in range(DECODE_STEPS):
                output = engine.forward([last], cache, cache.get_seq_length())
                last = bench.first_token(output)
                cache = output.past_key_values
            torch.cuda.synchronize()
            decode_ms = (perf_counter() - start) * 1000 / DECODE_STEPS
    bench.engine.unchanged()
    assert engine.student_test_native_prefills == 0
    return dict(ttft_ms=ttft, first_token=token, decode_ms_per_token=decode_ms, stage_ms=clock.stages,
                peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                profile_construct_extra_bytes=construct_extra,
                peak_delta_bytes=allocated - baseline, peak_with_decode_bytes=torch.cuda.max_memory_allocated(),
                translated_kv_bytes=body_bytes, memory_bank_bytes=memory_bytes)


def write_summary(rows, manifest, done):
    out = dict(status="completed" if done else "running", manifest=manifest, articles=len(rows))
    for mode in ("handoff", "text"):
        trials = [t for row in rows for t in row[mode]]
        out[mode] = dict(ttft_ms=stats([t["ttft_ms"] for t in trials]),
                         peak_allocated_bytes=max(t["peak_allocated_bytes"] for t in trials),
                         peak_reserved_bytes=max(t["peak_reserved_bytes"] for t in trials),
                         peak_delta_bytes=max(t["peak_delta_bytes"] for t in trials),
                         peak_with_decode_bytes=max(t["peak_with_decode_bytes"] for t in trials))
    out["decode_ms_per_token"] = stats([t["decode_ms_per_token"] for r in rows for t in r["handoff"]])
    profiles = [row["profile"] for row in rows]
    out["profile_stage_ms"] = {name: stats([p[name] for p in profiles]) for name in profiles[0]}
    out["construct_peak_extra_bytes"] = max(row["construct_peak_extra_bytes"] for row in rows)
    out["translated_kv_bytes"] = rows[0]["handoff"][0]["translated_kv_bytes"]
    out["memory_bank_bytes"] = rows[0]["handoff"][0]["memory_bank_bytes"]
    C.save(ROOT / "summary.json", out)
    return out


def main():
    data = C.read(E4 / "data_manifest.json")
    prior = C.read(E4 / "run_manifest.json")
    old = C.read(E4 / "chunk_sweep/summary.json")
    assert C.sha(E4 / "inputs.json") == data["input_sha256"]
    assert C.sha(E4 / "run_manifest.json") == old["manifest"]["prior_run_sha256"]
    assert C.sha(E4 / "benchmark.py") == prior["code_sha256"]
    assert C.sha(E4 / "chunk_sweep/sweep.py") == old["manifest"]["script_sha256"]
    assert C.sha(B.ROOT / "runs/stage2/epoch4_step1536.pt") == prior["dst_checkpoint_sha256"]
    rows = [row for row in C.read(E4 / "inputs.json") if row["length"] == 8192]
    assert [row["id"] for row in rows] == old["manifest"]["input_ids"]
    assert old["variants"]["gpu_index_2048"]["articles"] == len(rows) == 2
    bench = ChunkBenchmark()
    bench.path, bench.chunk = "gpu_index", 2048
    memory, checkpoint = load_memory(bench.engine)
    manifest = dict(script_sha256=C.sha(Path(__file__)), e4_input_sha256=data["input_sha256"],
                    historical_summary_sha256=C.sha(E4 / "chunk_sweep/summary.json"),
                    rank64_sha256=prior["dst_checkpoint_sha256"], gated_sha256=C.sha(checkpoint),
                    device=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                    body_tokens=8192, chunk=bench.chunk, trials=TRIALS, decode_steps=DECODE_STEPS,
                    source_on_handoff="CPU raw Sender K/V available", input_ids=[r["id"] for r in rows],
                    resident_models="Same E4 benchmark engine", batch=1)
    assert manifest["device"] == old["manifest"]["device"]
    C.save(ROOT / "manifest.json", manifest)
    pin = C.sha(ROOT / "manifest.json")
    results = []
    for row in rows:
        path = ROOT / "per_article" / (row["id"] + ".json")
        if path.exists():
            item = C.read(path)
            assert item["manifest_sha256"] == pin
        else:
            source = bench.capture(row, METHOD)
            warm = measure(bench, memory, row, source)
            item = dict(id=row["id"], manifest_sha256=pin, handoff=[], text=[])
            for trial in range(TRIALS):
                modes = ["handoff", "text"]
                random.Random(f"{row['id']}:{trial}").shuffle(modes)
                for mode in modes:
                    values = measure(bench, memory, row, source if mode == "handoff" else None,
                                     decode=mode == "handoff")
                    assert values["first_token"] == warm["first_token"]
                    item[mode].append(values)
            # Keep synchronized profiling separate from wall-clock trials.
            profile = measure(bench, memory, row, source, profile=True)
            item["profile"] = profile["stage_ms"]
            item["construct_peak_extra_bytes"] = profile["profile_construct_extra_bytes"]
            C.save(path, item)
            del source
        results.append(item)
        print(row["id"], write_summary(results, manifest, len(results) == len(rows)), flush=True)


if __name__ == "__main__":
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
