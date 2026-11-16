"""Frozen, paired LongBench v1 QA evaluation for Llama to Qwen on server A."""
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter, strftime
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "dst_rank64_gated_dynamic_m32_a4_server_a_seed1234"
CFG = json.loads((ROOT / "config.json").read_text())
ARMS = ("native", "dst_rank64", "gated_dynamic")

sys.path.insert(0, str(REFERENCE))
import experiment as G
from alignment import fullsync
from heterofold.evaluation.longbench import qa_f1

C, K, torch = G.C, G.G.K, G.torch


def status(state, **details):
    C.save(ROOT / "status.json", dict(status=state, time=strftime("%F %T"), **details))


def check_pause():
    if (ROOT / "controls/pause_requested.json").is_file():
        status("paused")
        raise SystemExit(75)


def load_examples():
    path = ROOT / "inputs/longbench.zip"
    assert C.sha(path) == CFG["archive_sha256"]
    with ZipFile(path) as archive:
        result = {task: [json.loads(line) for line in archive.read(f"data/{task}.jsonl").splitlines()]
                  for task in CFG["tasks"]}
    for task, rows in result.items():
        assert len(rows) == len({r["_id"] for r in rows}) == CFG["tasks"][task]["samples"]
        assert all(r["input"] and r["context"] and r["answers"] for r in rows)
    return result


class Protocol:
    def __init__(self, engine):
        self.engine = engine
        manifest = C.read(C.ROOT / "manifest.json")
        self.header = manifest["prefix"]
        self.header_ids = engine.prefix_ids
        self.suffix = manifest["assistant"]
        self.suffix_ids = self.encode(engine.tok, self.suffix)
        assert self.encode(engine.tok, self.header) == self.header_ids

    @staticmethod
    def encode(tokenizer, text):
        return tokenizer.encode(text, add_special_tokens=False, verbose=False)

    def lengths(self, prompt):
        e = self.engine
        return (len(self.encode(e.tok, prompt)) + len(self.header_ids) + len(self.suffix_ids),
                len(self.encode(e.sender_tok, prompt)))

    def row(self, task, item, cap):
        context, question = item["context"], item["input"]
        template = CFG["tasks"][task]["prompt"]

        def render(body):
            return template.format(context=body, input=question)

        prompt = render(context)
        original = self.lengths(prompt)
        clipped = max(original) > cap
        if clipped:
            offsets = self.engine.tok(context, add_special_tokens=False,
                                      return_offsets_mapping=True, verbose=False)["offset_mapping"]
            lo, hi = 0, len(offsets) - 1
            prompt = render("")
            while lo <= hi:
                count = (lo + hi) // 2
                left, right = (count + 1) // 2, count // 2
                front = context[:offsets[left - 1][1]] if left else ""
                back = context[offsets[-right][0]:] if right else ""
                candidate = render(front + "\n\n" + back)
                if max(self.lengths(candidate)) <= cap:
                    prompt = candidate
                    lo = count + 1
                else:
                    hi = count - 1

        engine = self.engine
        alignment = fullsync(prompt, engine.sender_tok, engine.tok)
        receiver, sender = list(alignment.receiver_ids), list(alignment.source_ids)
        assert max(len(receiver) + len(self.header_ids) + len(self.suffix_ids), len(sender)) <= cap
        assert self.encode(engine.tok, self.header + prompt + self.suffix) == (
            self.header_ids + receiver + self.suffix_ids
        )
        return dict(id=item["_id"], prompt=prompt, suffix_ids=self.suffix_ids,
                    sender_ids=sender, content_ids=receiver, sync_source_indices=alignment.source_indices,
                    clipped=clipped, original_qwen_tokens=original[0], original_sender_tokens=original[1],
                    qwen_tokens=len(receiver) + len(self.header_ids) + len(self.suffix_ids),
                    sender_tokens=len(sender))


def measured(fn):
    torch.cuda.synchronize()
    start = perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, perf_counter() - start


@torch.no_grad()
def eval_row(engine, rank, memory, reader, row, answers, cap):
    tok = engine.tok
    torch.cuda.reset_peak_memory_stats()

    def score(model, cache, rn, seconds):
        tokens, generation_time = measured(lambda: model.rollout(row, cache, rn=rn, cap=cap))
        prediction = tok.decode(tokens, skip_special_tokens=True).strip()
        return dict(prediction=prediction, f1=max(qa_f1(prediction, gold) for gold in answers),
                    generated_tokens=len(tokens), seconds=seconds + generation_time)

    native_cache, native_time = measured(lambda: engine.native_cache(row, baseline=True))
    native = score(reader, native_cache, False, native_time)
    del native_cache
    source, source_time = measured(lambda: engine.source(row))
    body, transfer_time = measured(lambda: engine.cache(row, source, rank))
    del source
    base_time = source_time + transfer_time
    plain = score(reader, body, False, base_time)
    bank, bank_time = measured(lambda: memory.construct(row, body))
    assert bank.body is body and len(bank.bank) == 2 * engine.model.config.num_hidden_layers
    assert all(value.shape[2] == memory.M for value in bank.bank)
    gated = score(memory, bank, True, base_time + bank_time)
    engine.unchanged()
    assert engine.student_test_native_prefills == 0
    return dict(native=native, dst_rank64=plain, gated_dynamic=gated), torch.cuda.max_memory_allocated() / 2**30


def aggregate(records):
    if not records:
        return None
    samples = np.array([[r["conditions"][arm]["f1"] for arm in ARMS] for r in records]) * 100
    draws = np.random.default_rng(1234).integers(len(samples), size=(10000, len(samples)))
    diff = samples[:, 2] - samples[:, 1]
    return dict(n=len(records), clipped=sum(r["clipped"] for r in records),
                f1={arm: float(samples[:, i].mean()) for i, arm in enumerate(ARMS)},
                gated_minus_dst_pp=float(diff.mean()), ci95_pp=np.quantile(diff[draws].mean(1), [.025, .975]).tolist())


def summarize(task, records):
    grouped = [
        ("<=4K", [r for r in records if r["qwen_tokens"] <= 4096]),
        ("4K-8K", [r for r in records if 4096 < r["qwen_tokens"] <= 8192]),
        (">8K", [r for r in records if r["qwen_tokens"] > 8192]),
    ]
    value = dict(status="completed" if len(records) == CFG["tasks"][task]["samples"] else "running",
                 overall=aggregate(records), length_buckets={label: aggregate(rows) for label, rows in grouped if rows},
                 fully_retained=aggregate([r for r in records if not r["clipped"]]))
    C.save(ROOT / f"evaluation/{task}/summary.json", value)
    return value


def probe(protocol, examples, engine, rank, memory, reader, cap):
    longest = max(((task, item) for task, rows in examples.items() for item in rows),
                  key=lambda pair: pair[1]["length"])
    task, item = longest
    row = protocol.row(task, item, cap)
    conditions, peak = eval_row(engine, rank, memory, reader, row, item["answers"], CFG["tasks"][task]["output_cap"])
    return dict(task=task, id=item["_id"], input_cap=cap, qwen_tokens=row["qwen_tokens"],
                sender_tokens=row["sender_tokens"], peak_gpu_gib=peak, seconds={a: v["seconds"] for a, v in conditions.items()})


def main():
    G.G.verify()
    examples = load_examples()
    status("loading")
    engine, rank, memory = G.load(final=True)
    memory.eval().requires_grad_(False)
    reader = G.BodyReader(engine)
    engine.mode = "test"
    protocol = Protocol(engine)
    manifest = ROOT / "manifest.json"
    pins = dict(
        archive=CFG["archive_sha256"], config=C.sha(ROOT / "config.json"),
        script=C.sha(Path(__file__)), reference=G.G.identity(), gated_checkpoint=C.sha(G.FINAL),
        fullsync={str(p.relative_to(ROOT)): C.sha(p) for p in [ROOT / "alignment.py", *sorted((ROOT / "reference").glob("legacy_*.py"))]},
        tasks=list(CFG["tasks"]), arms=list(ARMS),
        decoding="Qwen non-thinking greedy, upstream LongBench QA prompts and generation caps",
        length_policy="same raw body; cap both tokenizers; retain context head/tail and complete question",
    )
    if manifest.exists():
        settings = C.read(manifest)
        assert settings["pins"] == pins
        cap = settings["input_cap"]
    else:
        for cap in CFG["input_caps"]:
            check_pause()
            status("gpu_probe", input_cap=cap)
            try:
                result = probe(protocol, examples, engine, rank, memory, reader, cap)
            except torch.OutOfMemoryError:
                if cap == CFG["input_caps"][-1]:
                    raise
                print("GPU memory limit during probe", cap, flush=True)
            else:
                C.save(manifest, dict(input_cap=cap, pins=pins, probe=result))
                print("PROBE PASSED", json.dumps(result), flush=True)
                break
            torch.cuda.empty_cache()
    pin = C.sha(manifest)
    total_completed = 0
    final = {}
    for task, rows in examples.items():
        records = []
        for i, item in enumerate(rows, 1):
            check_pause()
            identifier = hashlib.sha256(item["_id"].encode()).hexdigest()[:20]
            path = ROOT / f"evaluation/{task}/per_sample/{identifier}.json"
            if path.exists():
                record = C.read(path)
                assert record["manifest_sha256"] == pin and record["id"] == item["_id"]
            else:
                row = protocol.row(task, item, cap)
                scores, peak = eval_row(engine, rank, memory, reader, row, item["answers"],
                                        CFG["tasks"][task]["output_cap"])
                record = {k: row[k] for k in ("id", "clipped", "qwen_tokens", "sender_tokens",
                                               "original_qwen_tokens", "original_sender_tokens")}
                record.update(manifest_sha256=pin, question=item["input"], answers=item["answers"],
                              conditions=scores, peak_gpu_gib=peak)
                C.save(path, record)
            records.append(record)
            total_completed += 1
            status("evaluating", task=task, task_completed=i, task_total=len(rows), total_completed=total_completed,
                   input_cap=cap, last_tokens=record["qwen_tokens"], peak_gpu_gib=record["peak_gpu_gib"])
            if i == 1 or i % 10 == 0 or i == len(rows):
                summary = summarize(task, records)
                print(task, f"{i}/{len(rows)}", json.dumps(summary["overall"]), flush=True)
        final[task] = summarize(task, records)
    C.save(ROOT / "summary.json", dict(status="completed", manifest_sha256=pin, input_cap=cap, tasks=final))
    status("completed", total_completed=total_completed, input_cap=cap)


if __name__ == "__main__":
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception as exc:
            status("failed", error=repr(exc))
            raise
