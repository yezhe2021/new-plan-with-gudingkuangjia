"""Five frozen transfer methods on the same complete long-document QA inputs."""
from contextlib import nullcontext
import fcntl
from pathlib import Path
import re
import sys
from time import perf_counter

from cohort import COUNT, LETTERS, TASKS, prepare

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
LATENCY = PROJECT / "dst_rank64_gated_dynamic_latency_seed1234" / "batch2_quality"
sys.path.insert(0, str(LATENCY))
import benchmark_batch as B
from affine_folding.benchmark import fuse_residual
from heterofold.alignment import receiver_to_sender_indices

C, K, torch = B.C, B.K, B.torch
ARMS = ("native", "heterofold_r64", "dst_rank64", "gated_dynamic", "recolor_c")
PREFILL = 2048
CHUNK = 1024
GEN_LIMIT = 16


def status(state, **details):
    C.save(ROOT / "status.json", dict(status=state, **details))


class Recolor:
    def __init__(self, group):
        self.group = group
        correction = C.fresh("C")
        if group == "heterofold_r64":
            map_path = Path(C.CFG["recolor"])
            run = C.ROOT / "runs/C"
            cp_path = run / "best_validation.pt"
            completion = C.read(run / "completed.json")
            best = min(completion["history"], key=lambda cfg: cfg["validation"]["selection_loss"])
            expected_map = C.CFG["recolor_sha256"]
            cp = torch.load(cp_path, map_location="cpu", weights_only=True)
            assert cp["step"] == best["step"] and cp["manifest_sha256"] == C.identity() and cp["arm"] == "C"
            assert cp["recolor_sha256"] == expected_map
        else:
            run = PROJECT / "recolor_c_dst_data_l3_to_q4_seed1234"
            map_path = run / "maps/recolor.pt"
            cp_path = run / "runs/epoch4_step1536.pt"
            completion = C.read(run / "runs/completed.json")
            expected_map = C.read(run / "maps/completed.json")["map_sha256"]
            cp = torch.load(cp_path, map_location="cpu", weights_only=True)
            assert C.sha(cp_path) == completion["checkpoint_sha256"] and cp["step"] == 1536
            assert cp["manifest_sha256"] == completion["manifest_sha256"] == C.sha(run / "manifest.json")
            assert cp["map_sha256"] == expected_map
        assert C.sha(map_path) == expected_map
        correction.load_state_dict(cp["state"], strict=True)
        correction.eval().requires_grad_(False)
        mapping = C.load_map(map_path, device="cuda", dtype=torch.float32, mmap=True)
        self.stencil = tuple(mapping["depth_stencil"][f"taps_{key}"] for key in "kv")
        self.taps = tuple(sorted({x for taps in role for x in taps}) for role in self.stencil)
        self.rms = tuple(tuple(map(float, mapping[f"rms_{key}"])) for key in "kv")
        self.weights = [mapping[f"A_{key}"] for key in "kv"]
        self.bias = [mapping[f"b_{key}"] for key in "kv"]
        self.correction = correction if group == "recolor_c" else None
        if self.correction is None:
            with torch.no_grad():
                for role, key in enumerate("kv"):
                    for layer, block in enumerate(correction):
                        left, right = getattr(block, f"left_{key}"), getattr(block, f"right_{key}")
                        w, b = self.weights[role][layer], self.bias[role][layer]
                        self.weights[role][layer] = w + (w @ left) @ right
                        self.bias[role][layer] = b + (b @ left) @ right
        self.identity = dict(map_sha256=expected_map, correction_sha256=C.sha(cp_path), step=cp["step"],
                             folded=self.correction is None, rank=64)

    def project(self, layer, features, dtype):
        pair = []
        for role in range(2):
            x = torch.cat([features[role][tap].float() for tap in self.stencil[role][layer]], dim=-1)
            w, b = self.weights[role][layer], self.bias[role][layer]
            normalized = x / self.rms[role][layer]
            if self.correction is None:
                y = torch.addmm(b, normalized, w)
            else:
                block = self.correction[layer]
                residual = block.key if role == 0 else block.value
                y = residual(((normalized @ w) + b).to(dtype).float())
            pair.append(y)
        return pair


def encode(tokenizer, text):
    return tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)


def encode_row(engine, header, task, prompt):
    suffix = header["assistant"] + ("The correct option is " if task == "quality" else "Answer: ")
    s = encode(engine.sender_tok, prompt)
    r = encode(engine.tok, prompt)
    suffix_ids = engine.tok.encode(suffix, add_special_tokens=False)
    assert engine.tok.encode(header["prefix"] + prompt + suffix, add_special_tokens=False, verbose=False) == (
        engine.prefix_ids + r["input_ids"] + suffix_ids)
    total = len(engine.prefix_ids) + len(r["input_ids"]) + len(suffix_ids) + GEN_LIMIT + 32
    assert total <= engine.model.config.max_position_embeddings, (task, total, "receiver_context_limit")
    assert len(s["input_ids"]) <= engine.sender.config.max_position_embeddings, (task, "sender_context_limit")
    return dict(prompt=prompt, content_ids=r["input_ids"], sender_ids=s["input_ids"], suffix_ids=suffix_ids,
                sender_offsets=s["offset_mapping"], receiver_offsets=r["offset_mapping"])


def prefill(engine, row, block_size):
    ids = engine.prefix_ids + row["content_ids"]
    inputs = torch.tensor([ids], device="cuda")
    positions = torch.arange(len(ids), device="cuda")
    cache = None
    for start in range(0, len(ids), block_size):
        end = min(start + block_size, len(ids))
        output = C.decoder(engine.model)(input_ids=inputs[:, start:end], position_ids=positions[None, start:end],
                                        past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        assert cache.get_seq_length() == end
    return cache


def transferred(engine, length, raw, indices, recolor, chunk_size):
    model = engine.model
    start = len(engine.prefix_ids)
    rank = engine.correction
    rank_dtype = next(rank.base.parameters()).dtype
    taps = tuple(sorted(set(rank.base.taps_tuple)))
    used = (taps, taps) if recolor is None else recolor.taps
    pairs = []
    for pair in K.pairs(engine.header):
        buf = [x.new_empty((1, x.shape[1], start+length, x.shape[-1])) for x in pair]
        for dst, src in zip(buf, pair):
            dst[:, :, :start].copy_(src)
        pairs.append(buf)
    for begin in range(0, length, chunk_size):
        end = min(begin+chunk_size, length)
        index = indices[begin:end]
        features = tuple({tap: raw[role][tap].index_select(0, index).to("cuda") for tap in layer_ids}
                         for role, layer_ids in enumerate(used))
        z = torch.zeros((end-begin, rank.base.cfg.state_width), device="cuda", dtype=rank_dtype) if recolor is None else None
        position = torch.arange(start+begin, start+end, device="cuda")
        tables = C._rotary_tables(model, position)
        for layer, pair in enumerate(pairs):
            if recolor is None:
                tap = rank.base.taps_tuple[layer]
                k, v, z = rank.base.layers[layer](features[0][tap].to(rank_dtype), features[1][tap].to(rank_dtype), z)
                k, v = fuse_residual(k, rank.keys[layer]), fuse_residual(v, rank.values[layer])
            else:
                k, v = recolor.project(layer, features, model.dtype)
            ready = C.finalize_layer(model, layer, k, v, *tables(layer))
            for dest, src in zip(pair, ready):
                dest[:, :, start+begin:start+end].copy_(src[None])
    return B.place(engine, pairs)


def prediction(task, text):
    clean = re.sub(r"^\s*(?:the\s+(?:correct\s+)?)?answer\s*[:=]\s*", "", text, flags=re.I).strip()
    if task == "sfiction":
        match = re.search(r"\b(true|false)\b", clean, re.I)
        return match[1].title() if match else None
    match = re.match(r"([A-D]+(?:[\s,;]+[A-D]+)*)(?![A-Za-z])", clean)
    return "".join(sorted({x for x in match[1] if x in LETTERS})) if match else None


def answer(engine, task, row, body, gold, memory=None, bank=None):
    begin = len(engine.prefix_ids) + len(row["content_ids"])
    cache = B.place(engine, K.pairs(body))
    assert cache.get_seq_length() == begin
    ids = torch.tensor([row["suffix_ids"]], device="cuda")
    with (memory.answering(bank, memory.gates) if memory is not None else nullcontext()):
        logits = B.receiver_step(engine, cache, ids)
        if task == "quality":
            labels = [engine.tok.encode(c, add_special_tokens=False) for c in LETTERS]
            assert all(len(x) == 1 for x in labels)
            scores = logits[0].log_softmax(-1)[[x[0] for x in labels]].tolist()
            choice = LETTERS[max(range(len(scores)), key=scores.__getitem__)]
            return dict(prediction=choice, correct=int(choice == gold), scores=scores)
        stop_ids = engine.model.generation_config.eos_token_id
        stops = set(stop_ids if isinstance(stop_ids, list) else [stop_ids])
        stops.add(engine.tok.convert_tokens_to_ids("<|im_end|>"))
        tokens = []
        for i in range(GEN_LIMIT):
            token = logits.argmax(-1)
            token_id = token.item()
            tokens.append(token_id)
            if token_id in stops or i + 1 == GEN_LIMIT:
                break
            logits = B.receiver_step(engine, cache, token[:, None])
    text = engine.tok.decode(tokens, skip_special_tokens=True)
    choice = prediction(task, text)
    return dict(prediction=choice, correct=int(choice == gold), output=text, generated_tokens=len(tokens),
                reached_cap=len(tokens) == GEN_LIMIT and tokens[-1] not in stops)


def score_row(engine, memory, maps, row, task, gold, prefill_chunk, transfer_chunk):
    body = prefill(engine, row, prefill_chunk)
    result = {"native": answer(engine, task, row, body, gold)}
    del body
    B.SENDER_CHUNK = prefill_chunk
    ids, raw = B.capture(engine, row)
    assert ids == row["sender_ids"]
    length = len(row["content_ids"])
    fullsync = B.fullsync(row["prompt"], engine.sender_tok, engine.tok, row["content_ids"])
    assert list(fullsync.source_ids) == ids
    dst = torch.tensor(fullsync.source_indices, device="cpu")
    ta = torch.tensor(receiver_to_sender_indices(row["sender_offsets"], row["receiver_offsets"]), device="cpu")
    assert len(dst) == len(ta) == length
    body = transferred(engine, length, raw, dst, None, transfer_chunk)
    result["dst_rank64"] = answer(engine, task, row, body, gold)
    bank = B.construct(memory, body, 1)
    result["gated_dynamic"] = answer(engine, task, row, body, gold, memory, bank)
    del bank, body
    for name, recolor in maps.items():
        body = transferred(engine, length, raw, ta, recolor, transfer_chunk)
        result[name] = answer(engine, task, row, body, gold)
        del body
    assert engine.student_test_native_prefills == 0 and result.keys() == set(ARMS)
    return result


def summarize(progress, pin):
    tasks = {}
    for task, saved in progress.items():
        if not saved:
            continue
        keys = ("correct", "loyalty_correct", "fact_correct") if task == "sfiction" else ("correct",)
        metrics = {arm: {key: sum(r["conditions"][arm][key] for r in saved) / len(saved) for key in keys}
                   for arm in ARMS}
        paired = {arm: C.paired([r["conditions"][arm]["correct"] - r["conditions"]["heterofold_r64"]["correct"]
                                 for r in saved]) for arm in ARMS if arm != "heterofold_r64"}
        lengths = [r["receiver_body_tokens"] for r in saved]
        tasks[task] = dict(n=len(saved), min_body_tokens=min(lengths), max_body_tokens=max(lengths),
                           scores=metrics, paired_minus_heterofold_r64=paired)
    complete = all(len(progress[t]) == COUNT for t in TASKS)
    C.save(ROOT / "summary.json", dict(status="completed" if complete else "running", manifest_sha256=pin, tasks=tasks))


@torch.no_grad()
def main():
    status("loading")
    samples, info = prepare(C)
    engine = B.Engine()
    engine.mode = "test"
    memory, memory_path = B.R.load_memory(engine)
    latency = C.read(LATENCY / "affine_folding/manifest.json")
    assert C.sha(memory_path) == latency["gated_checkpoint"]
    maps = {arm: Recolor(arm) for arm in ("heterofold_r64", "recolor_c")}
    header = C.read(C.ROOT / "manifest.json")
    assert engine.prefix_ids == header["prefix_ids"]
    rank_source = PROJECT / "dst_stage1_rank64_then_static_m32_a4_seed1234" / "runs/stage2/epoch4_step1536.pt"
    rank_hash = C.sha(rank_source)
    assert rank_hash == C.read(B.R.E4 / "run_manifest.json")["dst_checkpoint_sha256"]
    manifest_path = ROOT / "manifest.json"
    manifest = dict(cohort=info, arms=list(ARMS), gpu=torch.cuda.get_device_name(), prefill_chunk=PREFILL,
                    transfer_chunk=CHUNK, headers=engine.prefix_ids, generation_cap=GEN_LIMIT,
                    recolor={k: m.identity for k, m in maps.items()}, memory_sha256=C.sha(memory_path),
                    models=C.CFG["models"], rank_sha256=rank_hash,
                    code={p.name: C.sha(p) for p in (Path(__file__), ROOT / "cohort.py")})
    if manifest_path.exists():
        assert C.read(manifest_path) == manifest
    else:
        C.save(manifest_path, manifest)
    pin = C.sha(manifest_path)
    progress = {task: [] for task in TASKS}
    # Exercise the longest document from each task before processing the remaining cases.
    longest = {task: max(group, key=lambda item: len(item["context"])) for task, group in samples.items()}
    order = list(longest.items()) + [(task, item) for task, group in samples.items()
                                    for item in group if item is not longest[task]]
    for task, item in order:
        if (ROOT / "controls/pause_requested.json").exists():
            status("paused", completed=sum(map(len, progress.values())))
            return
        dest = ROOT / "evaluation" / task / "per_sample" / f"{item['id']}.json"
        if dest.exists():
            saved = C.read(dest)
            assert saved["manifest_sha256"] == pin and saved["id"] == item["id"]
        else:
            prompt = item["before"] + item["context"] + item["after"]
            row = encode_row(engine, header, task, prompt)
            started = perf_counter()
            status("evaluating", task=task, id=item["id"], body_tokens=len(row["content_ids"]),
                   completed=sum(map(len, progress.values())), total=COUNT * len(TASKS))
            for prefill_chunk, transfer_chunk in ((PREFILL, CHUNK), (512, 256)):
                try:
                    conditions = score_row(engine, memory, maps, row, task, item["gold"], prefill_chunk, transfer_chunk)
                    if task == "sfiction":
                        fact_row = encode_row(engine, header, task, item["fact_prompt"])
                        factual = score_row(engine, memory, maps, fact_row, task, item["fact"], prefill_chunk, transfer_chunk)
                        for arm, scores in conditions.items():
                            scores["loyalty_correct"] = scores["correct"]
                            scores["fact_correct"] = factual[arm]["correct"]
                            scores["fact"] = factual[arm]
                            scores["correct"] = (scores["loyalty_correct"] + scores["fact_correct"]) / 2
                    break
                except torch.OutOfMemoryError:
                    if prefill_chunk < PREFILL:
                        raise
                    print("RETRY OOM", task, item["id"], flush=True)
                torch.cuda.empty_cache()
            saved = dict(id=item["id"], manifest_sha256=pin, task=task, gold=item["gold"],
                         fact_gold=item.get("fact"), hard=item.get("hard"), document=item["document"],
                         context_sha256=C.digest(item["context"]), prompt_sha256=C.digest(prompt),
                         sender_body_tokens=len(row["sender_ids"]), receiver_body_tokens=len(row["content_ids"]),
                         conditions=conditions, elapsed_seconds=perf_counter()-started, clipped=False,
                         prefill_chunk=prefill_chunk, transfer_chunk=transfer_chunk)
            C.save(dest, saved)
        progress[task].append(saved)
        summarize(progress, pin)
        print("SAMPLE", task, len(progress[task]), item["id"], saved["receiver_body_tokens"],
              round(saved["elapsed_seconds"], 2), flush=True)
    status("completed", samples=sum(map(len, progress.values())))


if __name__ == "__main__":
    with (LATENCY / "run.lock").open("a") as gpu_lock, (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception as error:
            status("failed", error=repr(error))
            raise
