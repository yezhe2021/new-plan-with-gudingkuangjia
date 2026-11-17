"""Paired QuALITY timing with folded Recolor and fused nonlinear DST readout."""
import fcntl
import gc
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import benchmark_baselines as H
import benchmark_batch as B
import resume_32k as S

C, K, torch = B.C, B.K, B.torch
ARMS = ("recolor_original", "recolor_folded", "dst_original", "dst_fused")
PROBE_STEPS = 4


def status(state, **details):
    C.save(ROOT / "status.json", dict(status=state, **details))


def fuse_residual(x, block):
    # Keep the exact trained GELU and accumulate the up projection into the readout.
    hidden = block[1](block[0](x))
    return x.addmm_(hidden, block[2].weight.T)


class Mapping:
    def __init__(self, engine, recolor):
        self.engine, self.recolor = engine, recolor
        self.rank = engine.correction
        self.base = self.rank.base
        assert self.base.cfg.carry_state
        assert all(isinstance(m[1], torch.nn.GELU) and m[1].approximate == "none"
                   and m[0].bias is None and m[2].bias is None for m in [*self.rank.keys, *self.rank.values])
        self.dst_taps = tuple(sorted(set(self.base.taps_tuple)))
        self.folded = tuple(
            tuple(self.fold(layer, role, key) for layer in range(36)) for role, key in enumerate("kv")
        )

    @torch.no_grad()
    def fold(self, layer, role, key):
        correction = self.recolor.correction[layer]
        left = getattr(correction, f"left_{key}")
        right = getattr(correction, f"right_{key}")
        weight = self.recolor.weights[role][layer]
        bias = self.recolor.bias[role][layer]
        return weight + (weight @ left) @ right, bias + (bias @ left) @ right

    def align_dst(self, rows, sources):
        indices = []
        for row, (ids, _) in zip(rows, sources):
            aligned = B.fullsync(row["prompt"], self.engine.sender_tok, self.engine.tok, row["content_ids"])
            assert list(aligned.source_ids) == ids
            indices.append(torch.tensor(aligned.source_indices, device="cpu"))
        return indices

    def project_dst(self, features, width, fused):
        dtype = next(self.base.parameters()).dtype
        z = torch.zeros(B.BATCH * width, self.base.cfg.state_width, device="cuda", dtype=dtype)
        output = []
        for layer, (tap, recurrent) in enumerate(zip(self.base.taps_tuple, self.base.layers)):
            k, v, z = recurrent(features[0][tap].to(dtype), features[1][tap].to(dtype), z)
            rk, rv = self.rank.keys[layer], self.rank.values[layer]
            output.append((fuse_residual(k, rk), fuse_residual(v, rv)) if fused else (k + rk(k), v + rv(v)))
        return output

    def project_recolor(self, features, folded):
        output = []
        base = self.recolor
        dtype = self.engine.model.dtype
        for layer, correction in enumerate(base.correction):
            mapped = []
            for role, correct in enumerate((correction.key, correction.value)):
                x = torch.cat([features[role][tap].float() for tap in base.taps[role][layer]], dim=-1)
                normalized = x / base.rms[role][layer]
                if folded:
                    weight, bias = self.folded[role][layer]
                    mapped.append(torch.addmm(bias, normalized, weight))
                else:
                    y = (normalized @ base.weights[role][layer]) + base.bias[role][layer]
                    mapped.append(correct(y.to(dtype).float()))
            output.append(tuple(mapped))
        return output

    def allocate(self, length, h):
        pairs = []
        for old in K.pairs(self.engine.header):
            tensors = []
            for x in old:
                dest = x.new_empty((B.BATCH, x.shape[1], length+h, x.shape[-1]))
                dest[:, :, :h].copy_(x.expand(B.BATCH, -1, -1, -1))
                tensors.append(dest)
            pairs.append(tuple(tensors))
        return pairs

    @torch.no_grad()
    def build(self, arm, rows, sources, clock):
        engine, model = self.engine, self.engine.model
        length = len(rows[0]["content_ids"])
        h = len(engine.prefix_ids)
        dst = arm.startswith("dst")
        align = self.align_dst if dst else self.recolor.align
        used = (self.dst_taps, self.dst_taps) if dst else self.recolor.used
        indices = clock.measure("alignment", align, rows, sources)
        pairs = clock.measure("cache_allocate", self.allocate, length, h)
        for begin in range(0, length, B.CHUNK):
            end = min(begin+B.CHUNK, length)
            width = end-begin

            def upload():
                return tuple({
                    tap: torch.cat([raw[role][tap].index_select(0, ix[begin:end])
                                    for ix, (_, raw) in zip(indices, sources)]).to("cuda")
                    for tap in taps
                } for role, taps in enumerate(used))
            features = clock.measure("gather_and_upload", upload)
            if dst:
                mapped = clock.measure("translation", self.project_dst, features, width, arm == "dst_fused")
            else:
                mapped = clock.measure("translation", self.project_recolor, features, arm == "recolor_folded")
            del features

            def finalize():
                positions = torch.arange(h+begin, h+end, device="cuda").repeat(B.BATCH)
                tables = C._rotary_tables(model, positions)
                for layer, (key, value) in enumerate(mapped):
                    for a, target in zip(C.finalize_layer(model, layer, key, value, *tables(layer)), pairs[layer]):
                        heads, _, dim = a.shape
                        target[:, :, h+begin:h+end].copy_(a.reshape(heads, B.BATCH, width, dim).transpose(0, 1))
            clock.measure("cache_finalize", finalize)
            del mapped
        return clock.measure("cache_handle", B.place, engine, pairs)


@torch.no_grad()
def measure(mapping, memory, arm, rows, source, *, profile=False, probe=False):
    engine = mapping.engine
    for _, raw in source:
        raw.view(-1)[::2048].sum().item()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    io_before, faults_before = S.counters()
    clock = B.R.Stopwatch(profile)
    start = perf_counter()
    body = mapping.build(arm, rows, source, clock)
    if arm.startswith("dst"):
        bank = clock.measure("memory_construct", B.construct, memory, body, B.BATCH)
        context = memory.answering(bank, memory.gates)
    else:
        from contextlib import nullcontext
        context = nullcontext()
    ids = torch.tensor([r["suffix_ids"] for r in rows], device="cuda")
    with context:
        logits = clock.measure("receiver_suffix_and_head", B.receiver_step, engine, body, ids)
        token = logits.argmax(-1)
        first = token.tolist()
        ttft = (perf_counter()-start) * 1000
        read, faults = S.counters()
        result = dict(ttft_ms=ttft, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      timed_storage_read_bytes=read-io_before, timed_major_page_faults=faults-faults_before,
                      first_tokens=first, phases_ms=clock.stages)
        if probe:
            trace = [first]
            for _ in range(PROBE_STEPS-1):
                token = B.receiver_step(engine, body, token[:, None]).argmax(-1)
                trace.append(token.tolist())
            return result, logits.cpu(), trace
    assert engine.student_test_native_prefills == 0
    return result


def comparison(original, optimized, a, b):
    rms = ((a-b).norm(dim=1) / a.norm(dim=1).clamp_min(1e-12)).tolist()
    assert all(torch.isfinite(x).all() for x in (a,b))
    return dict(first_logits_relative_l2=rms, first_token_matches=[x==y for x,y in zip(original[0],optimized[0])],
                greedy_token_matches=sum(x==y for u,v in zip(original,optimized) for x,y in zip(u,v)),
                greedy_token_total=PROBE_STEPS*B.BATCH)


@torch.no_grad()
def verify_cache(mapping, rows, sources):
    results = {}
    for method, original, optimized in (("recolor", *ARMS[:2]), ("dst", *ARMS[2:])):
        ref = mapping.build(original, rows, sources, B.R.Stopwatch(False))
        got = mapping.build(optimized, rows, sources, B.R.Stopwatch(False))
        roles = [[], []]
        for p, q in zip(K.pairs(ref), K.pairs(got)):
            for role, (x, y) in enumerate(zip(p,q)):
                a,b = x.float(), y.float()
                if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
                    raise FloatingPointError(method)
                roles[role].append(float((a-b).norm()/a.norm().clamp_min(1e-12)))
        results[method] = dict(key_max_relative_l2=max(roles[0]), value_max_relative_l2=max(roles[1]))
        del ref, got
    return results


def report(batches, pin):
    lengths = {}
    for n in B.LENGTHS:
        rows = [r for r in batches if r["length"] == n]
        if not rows:
            continue
        arms = {}
        for arm in ARMS:
            trials = [t for row in rows for t in row["trials"][arm]]
            profiles = [r["profiles"][arm]["phases_ms"] for r in rows]
            arms[arm] = dict(
                handoff_ms=B.R.stats([t["ttft_ms"] for t in trials]),
                translation_ms=B.R.stats([p["translation"] for p in profiles]),
                post_upload_ms=B.R.stats([sum(p[k] for k in p if k in
                    ("translation","cache_finalize","cache_handle","memory_construct","receiver_suffix_and_head"))
                    for p in profiles]),
                peak_allocated_bytes=max(t["peak_allocated_bytes"] for t in trials),
                timed_storage_read_bytes=[t["timed_storage_read_bytes"] for t in trials],
                profile_storage_read_bytes=[r["profiles"][arm]["timed_storage_read_bytes"] for r in rows],
            )
        lengths[str(n)] = dict(batches=len(rows), arms=arms, parity=[r["parity"] for r in rows])
    completed = len(batches)==len(B.LENGTHS)*2
    C.save(ROOT / "summary.json", dict(status="completed" if completed else "running", manifest_sha256=pin,
                                       lengths=lengths))


def main():
    status("loading")
    parent = C.read(B.ROOT / "manifest.json")
    assert parent["script_sha256"] == C.sha(B.ROOT / "benchmark_batch.py")
    assert parent["quality_archive_sha256"] == C.sha(B.R.E4 / "quality_dev.jsonl")
    prior = C.read(B.R.E4 / "run_manifest.json")
    parent_baselines = C.read(B.ROOT / "baselines/manifest.json")
    assert parent_baselines["baseline_script"] == C.sha(B.ROOT / "benchmark_baselines.py")
    assert parent_baselines["recolor_checkpoint"] == prior["recolor_checkpoint_sha256"]
    engine = B.Engine()
    memory, weights = B.R.load_memory(engine)
    assert C.sha(weights) == parent["checkpoint"]
    engine.mode = "test"
    cohorts = B.inputs(engine)
    prompts = {str(n): [[{k:r[k] for k in ("article_ids","prompt_sha256")} for r in rows]
                       for rows in groups] for n,groups in cohorts.items()}
    assert C.digest(prompts) == C.digest(parent["prompts"])
    manifest = dict(script_sha256=C.sha(Path(__file__)), parent=C.sha(B.ROOT / "manifest.json"),
                    recolor_checkpoint=prior["recolor_checkpoint_sha256"], gated_checkpoint=C.sha(weights),
                    lengths=list(B.LENGTHS), chunk=B.CHUNK, batch=B.BATCH, trials=B.TRIALS, arms=list(ARMS),
                    sender_outside_timed_region=True, sender_weights_resident_during_measurement=False,
                    folded_recolor_removes_intermediate_fp16_rounding=True,
                    gpu=torch.cuda.get_device_name(), storage="warmed file-backed CPU K/V")
    manifest_path = ROOT / "manifest.json"
    if manifest_path.exists():
        assert C.read(manifest_path)==manifest
    else:
        C.save(manifest_path,manifest)
    pin = C.sha(manifest_path)

    with TemporaryDirectory(dir=ROOT, prefix="sender_kv_") as work:
        staged = {}
        for n, groups in cohorts.items():
            for i, rows in enumerate(groups):
                dest = ROOT / "batches" / f"{n}_{i}.json"
                if not dest.is_file():
                    status("preparing_sender_kv", length=n, batch_index=i+1)
                    staged[n,i] = [S.capture(engine,row,Path(work) / f"{n}_{i}_{j}.bin") for j,row in enumerate(rows)]
                    print("SENDER KV READY", n, i+1, flush=True)
        del engine.sender
        gc.collect()
        torch.cuda.empty_cache()
        mapping = Mapping(engine, H.Recolor(engine, prior))
        batches = []
        for n, groups in cohorts.items():
            for i, rows in enumerate(groups):
                dest = ROOT / "batches" / f"{n}_{i}.json"
                if dest.is_file():
                    item = C.read(dest)
                    assert item["manifest_sha256"] == pin
                else:
                    sources = staged.pop((n,i))
                    status("measuring", length=n, batch_index=i+1)
                    if n==B.LENGTHS[0] and i==0:
                        C.save(ROOT / "cache_parity_4k.json", verify_cache(mapping,rows,sources))
                    probes = {arm:measure(mapping,memory,arm,rows,sources,probe=True) for arm in ARMS}
                    parity = {}
                    for label, orig, new in (("recolor", *ARMS[:2]), ("dst", *ARMS[2:])):
                        _, x, old_tokens = probes[orig]
                        _, y, new_tokens = probes[new]
                        parity[label] = comparison(old_tokens,new_tokens,x,y)
                    trials = {arm:[] for arm in ARMS}
                    for j in range(B.TRIALS):
                        order = list(ARMS)
                        random.Random(f"{n}:{i}:{j}").shuffle(order)
                        for arm in order:
                            result = measure(mapping,memory,arm,rows,sources)
                            assert result["first_tokens"] == probes[arm][0]["first_tokens"]
                            trials[arm].append(result)
                    profiles = {arm:measure(mapping,memory,arm,rows,sources,profile=True) for arm in ARMS}
                    item = dict(length=n, batch_index=i+1, manifest_sha256=pin, trials=trials,
                                profiles=profiles, parity=parity)
                    C.save(dest,item)
                    del sources
                    print("BATCH COMPLETED", n, i+1, parity, flush=True)
                batches.append(item)
                report(batches,pin)
    status("completed", batches=len(batches))


if __name__ == "__main__":
    with (B.ROOT / "run.lock").open("a") as resource, (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(resource, fcntl.LOCK_EX)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception as exc:
            status("failed",error=repr(exc))
            raise
