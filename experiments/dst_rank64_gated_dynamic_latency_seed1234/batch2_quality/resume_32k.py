"""Finish the 32K batches with reclaimable file-backed Sender KV buffers."""
import fcntl
from pathlib import Path
import os
import resource
from tempfile import TemporaryDirectory
from contextlib import ExitStack

import benchmark_batch as B

C, torch, ROOT = B.C, B.torch, B.ROOT
LENGTH = 32768


@torch.no_grad()
def capture(engine, row, file):
    sender = engine.sender
    ids = engine.sender_tok.encode(row["prompt"], add_special_tokens=False, verbose=False)
    assert len(ids) <= sender.config.max_position_embeddings
    ids_cuda = torch.tensor([ids], device="cuda")
    positions = torch.arange(len(ids), device="cuda")
    layers = C.layers_of(sender)
    shape = (2, len(layers), len(ids), 1024)
    count = 2 * len(layers) * len(ids) * 1024
    with file.open("wb") as f:
        f.truncate(count * torch.empty((), dtype=sender.dtype).element_size())
    raw = torch.from_file(str(file), shared=True, size=count, dtype=sender.dtype).view(shape)
    offset = 0

    def save(target):
        def hook(_module, _args, output):
            width = output.shape[1]
            target[offset:offset+width].copy_(output[0].detach())
        return hook

    with ExitStack() as hooks:
        for i, layer in enumerate(layers):
            for role, projection in enumerate((layer.self_attn.k_proj, layer.self_attn.v_proj)):
                hooks.callback(projection.register_forward_hook(save(raw[role, i])).remove)
        cache = None
        for offset in range(0, len(ids), B.SENDER_CHUNK):
            end = min(offset + B.SENDER_CHUNK, len(ids))
            result = C.decoder(sender)(input_ids=ids_cuda[:, offset:end], position_ids=positions[None, offset:end],
                                       past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            assert cache.get_seq_length() == end
    with file.open("rb") as f:
        os.fsync(f.fileno())
    return ids, raw


def counters():
    with open("/proc/self/io") as f:
        read = next(int(s.split()[1]) for s in f if s.startswith("read_bytes:"))
    return read, resource.getrusage(resource.RUSAGE_SELF).ru_majflt


def timed(engine, memory, rows, source, **kwargs):
    # Touch mapped pages outside the timed region. Report any actual I/O during it.
    for _, raw in source:
        raw.view(-1)[::2048].sum().item()
    before = counters()
    record = B.measure(engine, memory, rows, source, **kwargs)
    after = counters()
    record.update(timed_storage_read_bytes=after[0] - before[0],
                  timed_major_page_faults=after[1] - before[1])
    return record


def main():
    path = ROOT / "manifest.json"
    manifest = C.read(path)
    assert C.sha(ROOT / "benchmark_batch.py") == manifest["script_sha256"]
    assert C.sha(B.R.E4 / "quality_dev.jsonl") == manifest["quality_archive_sha256"]
    pin = C.sha(path)
    B.status("loading_32k")
    engine = B.Engine()
    memory, weights = B.R.load_memory(engine)
    engine.mode = "test"
    assert C.sha(weights) == manifest["checkpoint"]
    groups = B.inputs(engine)[LENGTH]
    prompts = [[{k: r[k] for k in ("article_ids", "prompt_sha256")} for r in batch] for batch in groups]
    assert C.digest(prompts) == C.digest(manifest["prompts"][str(LENGTH)])
    records = [C.read(p) for p in sorted((ROOT / "batches").glob("*.json"))
               if not p.name.startswith(str(LENGTH))]
    assert len(records) == 4 and all(r["manifest_sha256"] == pin for r in records)
    for g, rows in enumerate(groups):
        record_path = ROOT / "batches" / f"{LENGTH}_{g}.json"
        if record_path.exists():
            record = C.read(record_path)
            assert record["manifest_sha256"] == pin
        else:
            with TemporaryDirectory(prefix="sender_kv_", dir=ROOT) as work:
                B.status("preparing_sender_kv", length=LENGTH, batch_index=g+1)
                source = [capture(engine, row, Path(work) / f"{i}.bin") for i, row in enumerate(rows)]
                torch.cuda.empty_cache()
                B.status("measuring_handoff", length=LENGTH, batch_index=g+1)
                warm = timed(engine, memory, rows, source)
                trials = [timed(engine, memory, rows, source, decode=True) for _ in range(B.TRIALS)]
                profile = timed(engine, memory, rows, source, profile=True)
                assert all(t["first_tokens"] == warm["first_tokens"] for t in [*trials, profile])
                record = dict(manifest_sha256=pin, length=LENGTH, batch_index=g+1, trials=trials,
                              profile=profile, source_storage="warmed file-backed CPU memory",
                              resume_script_sha256=C.sha(Path(__file__)))
                C.save(record_path, record)
                del source
        records.append(record)
        B.summary(records, manifest)
        combined = C.read(ROOT / "summary.json")
        combined["storage_32k"] = {
            "mode": "warmed file-backed CPU memory",
            "trial_storage_read_bytes": [t["timed_storage_read_bytes"] for r in records
                                         if r["length"] == LENGTH for t in r["trials"]],
            "trial_major_page_faults": [t["timed_major_page_faults"] for r in records
                                        if r["length"] == LENGTH for t in r["trials"]],
        }
        C.save(ROOT / "summary.json", combined)
        print("BATCH", LENGTH, g+1, [(round(t["ttft_ms"], 1), t["timed_storage_read_bytes"])
                                      for t in record["trials"]], flush=True)
    B.status("completed")


if __name__ == "__main__":
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception as exc:
            B.status("failed", error=repr(exc))
            raise
