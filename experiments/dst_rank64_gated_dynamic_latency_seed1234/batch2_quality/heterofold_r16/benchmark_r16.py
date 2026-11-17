"""Paired folded HeteroFold-R16 and fused DST handoff timing on QuALITY."""
import fcntl
import gc
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from affine_folding import benchmark as F
from paper_map import load_r16

B, C, torch = F.B, F.C, F.torch
ARMS = {"heterofold_r16": "recolor_folded", "dst_rank64_gated_dynamic": "dst_fused"}


class FoldedMap:
    align = F.H.Recolor.align

    def __init__(self, engine, mapping):
        self.engine = engine
        stencil = mapping["depth_stencil"]
        self.taps = tuple(stencil[f"taps_{role}"] for role in "kv")
        self.rms = tuple(tuple(map(float, mapping[f"rms_{role}"])) for role in "kv")
        self.weights = tuple(mapping[f"A_{role}"] for role in "kv")
        self.bias = tuple(mapping[f"b_{role}"] for role in "kv")
        self.used = tuple(sorted({tap for layer in role for tap in layer}) for role in self.taps)


class PairedMapping(F.Mapping):
    def fold(self, layer, role, key):
        # The calibrated checkpoint already contains the final affine weights.
        return self.recolor.weights[role][layer], self.recolor.bias[role][layer]

    def project_recolor(self, features, folded):
        assert folded
        mapped = []
        for layer in range(len(self.folded[0])):
            pair = []
            for role in range(2):
                x = torch.cat([features[role][tap].float() for tap in self.recolor.taps[role][layer]], dim=-1)
                weight, bias = self.folded[role][layer]
                pair.append(torch.addmm(bias, x / self.recolor.rms[role][layer], weight))
            mapped.append(tuple(pair))
        return mapped


@torch.no_grad()
def verify_r16(mapping, checkpoint, rows, sources):
    from heterofold.transfer import build_receiver_cache

    engine = mapping.engine
    ids, raw = sources[0]
    encoded = engine.sender_tok(rows[0]["prompt"], add_special_tokens=False,
                                return_offsets_mapping=True, verbose=False)
    assert encoded["input_ids"] == ids
    keys, values = ({tap: raw[role][tap].to("cuda") for tap in taps}
                    for role, taps in enumerate(mapping.recolor.used))
    official, length = build_receiver_cache(engine.tok, engine.model, checkpoint, encoded["offset_mapping"],
                                            keys, values, rows[0]["prompt"], engine.prefix_ids)
    batched = mapping.build(ARMS["heterofold_r16"], rows, sources, B.R.Stopwatch(False))
    assert length == batched.get_seq_length()
    relative = [[], []]
    for ref, opt in zip(F.K.pairs(official), F.K.pairs(batched)):
        for role, (x, y) in enumerate(zip(ref, opt)):
            a, b = x[0].float(), y[0].float()
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            relative[role].append(float((a - b).norm() / a.norm().clamp_min(1e-12)))
    assert max(map(max, relative)) < 0.002
    return dict(key_max_relative_l2=max(relative[0]), value_max_relative_l2=max(relative[1]))


def summarize(batches, pin):
    lengths = {}
    for length in B.LENGTHS:
        cohort = [x for x in batches if x["length"] == length]
        if not cohort:
            continue
        lengths[str(length)] = {"batches": len(cohort), "arms": {}}
        for label in ARMS:
            trials = [t for row in cohort for t in row["trials"][label]]
            profiles = [row["profiles"][label]["phases_ms"] for row in cohort]
            lengths[str(length)]["arms"][label] = dict(
                handoff_ms=B.R.stats([t["ttft_ms"] for t in trials]),
                phases_ms={stage: B.R.stats([p[stage] for p in profiles]) for stage in profiles[0]},
                post_upload_ms=B.R.stats([sum(p.get(stage, 0.0) for stage in (
                    "translation", "cache_finalize", "cache_handle", "memory_construct", "receiver_suffix_and_head"
                )) for p in profiles]),
                peak_allocated_bytes=max(t["peak_allocated_bytes"] for t in trials),
                timed_storage_read_bytes=[t["timed_storage_read_bytes"] for t in trials],
                first_tokens=[row["profiles"][label]["first_tokens"] for row in cohort],
            )
    C.save(ROOT / "latency_summary.json", dict(status="completed" if len(batches) == 6 else "running",
                                               manifest_sha256=pin, lengths=lengths))


def status(phase, **fields):
    C.save(ROOT / "latency_status.json", dict(status=phase, **fields))


def main():
    status("loading")
    parent_path = B.ROOT / "manifest.json"
    parent = C.read(parent_path)
    affine_path = B.ROOT / "affine_folding/manifest.json"
    affine = C.read(affine_path)
    assert C.sha(B.ROOT / "benchmark_batch.py") == parent["script_sha256"]
    assert C.sha(Path(F.__file__)) == affine["script_sha256"] and C.sha(parent_path) == affine["parent"]
    assert C.sha(B.R.E4 / "quality_dev.jsonl") == parent["quality_archive_sha256"]
    engine = B.Engine()
    engine.mode = "test"
    memory, weights = B.R.load_memory(engine)
    assert C.sha(weights) == parent["checkpoint"] == affine["gated_checkpoint"]
    cohorts = B.inputs(engine)
    prompts = {str(n): [[{k: r[k] for k in ("article_ids", "prompt_sha256")} for r in rows]
                       for rows in groups] for n, groups in cohorts.items()}
    assert C.digest(prompts) == C.digest(parent["prompts"])
    official, checkpoint = load_r16(C)
    assert tuple(B.LENGTHS) == tuple(affine["lengths"]) and B.BATCH == affine["batch"]
    manifest = dict(parent=C.sha(parent_path), timing_reference=C.sha(affine_path), checkpoint=checkpoint,
                    gated_checkpoint=C.sha(weights), gpu=torch.cuda.get_device_name(), methods=list(ARMS),
                    batch=B.BATCH, trials=B.TRIALS, chunk=B.CHUNK, lengths=list(B.LENGTHS),
                    code={p.name: C.sha(p) for p in (Path(__file__), ROOT / "paper_map.py")},
                    sender_capture_timed=False, sender_weights_resident_during_measurement=False,
                    storage="warmed file-backed CPU K/V", both_mappers_resident_during_measurement=True)
    manifest_path = ROOT / "latency_manifest.json"
    if manifest_path.exists():
        assert C.read(manifest_path) == manifest
    else:
        C.save(manifest_path, manifest)
    pin = C.sha(manifest_path)
    batches = []
    with TemporaryDirectory(dir=ROOT, prefix="sender_kv_") as work:
        staged = {}
        for length, groups in cohorts.items():
            for i, rows in enumerate(groups):
                path = ROOT / "batches" / f"{length}_{i}.json"
                if not path.exists():
                    status("preparing_sender_kv", length=length, batch_index=i+1)
                    staged[length, i] = [F.S.capture(engine, row, Path(work) / f"{length}_{i}_{j}.bin")
                                         for j, row in enumerate(rows)]
                    print("SENDER KV READY", length, i+1, flush=True)
        del engine.sender
        gc.collect()
        torch.cuda.empty_cache()
        mapping = PairedMapping(engine, FoldedMap(engine, official))
        for length, groups in cohorts.items():
            for i, rows in enumerate(groups):
                path = ROOT / "batches" / f"{length}_{i}.json"
                if path.exists():
                    record = C.read(path)
                    assert record["manifest_sha256"] == pin
                else:
                    source = staged.pop((length, i))
                    status("measuring", length=length, batch_index=i+1)
                    if length == B.LENGTHS[0] and i == 0:
                        parity = verify_r16(mapping, official, rows, source)
                        C.save(ROOT / "latency_cache_parity.json", parity)
                        print("OFFICIAL CACHE PARITY", parity, flush=True)
                    warm = {label: F.measure(mapping, memory, arm, rows, source) for label, arm in ARMS.items()}
                    trials = {label: [] for label in ARMS}
                    for j in range(B.TRIALS):
                        order = list(ARMS)
                        random.Random(f"{length}:{i}:{j}").shuffle(order)
                        for label in order:
                            result = F.measure(mapping, memory, ARMS[label], rows, source)
                            assert result["first_tokens"] == warm[label]["first_tokens"]
                            trials[label].append(result)
                    profiles = {label: F.measure(mapping, memory, arm, rows, source, profile=True)
                                for label, arm in ARMS.items()}
                    record = dict(manifest_sha256=pin, length=length, batch_index=i+1,
                                  trials=trials, profiles=profiles)
                    C.save(path, record)
                    del source
                    print("BATCH COMPLETED", length, i+1, flush=True)
                batches.append(record)
                summarize(batches, pin)
    status("completed", batches=len(batches))


if __name__ == "__main__":
    with (ROOT.parent / "run.lock").open("a") as resource:
        fcntl.flock(resource, fcntl.LOCK_EX)
        try:
            main()
        except Exception as exc:
            status("failed", error=repr(exc))
            raise
