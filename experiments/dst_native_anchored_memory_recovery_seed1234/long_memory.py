"""Compare banks on the same long-context DST body, then score trained Recovery."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "long_qa_quality_leval_full64_seed1234"
sys.path.insert(0, str(REFERENCE))
import evaluate as L
from cohort import COUNT, TASKS, prepare
from recovery import attach

C, B, torch = L.C, L.B, L.torch
FINAL = ROOT / "runs/epoch4_step1536.pt"


def summarize(destination, records, arm, identity):
    tasks = {}
    for task, entries in records.items():
        if entries:
            values = [x["conditions"] for x in entries]
            metrics = {key: sum(row[key]["correct"] for row in values)/len(values) for key in ("dst_memory", arm)}
            tasks[task] = dict(n=len(values), scores=metrics,
                               paired_minus_dst=C.paired([row[arm]["correct"]-row["dst_memory"]["correct"] for row in values]))
    complete = all(len(records[task]) == COUNT for task in TASKS)
    C.save(destination / "summary.json", dict(status="completed" if complete else "running", identity=identity, tasks=tasks))


def translated(engine, row, prefill_chunk, cache_chunk):
    B.SENDER_CHUNK = prefill_chunk
    ids, raw = B.capture(engine, row)
    assert ids == row["sender_ids"]
    alignment = B.fullsync(row["prompt"], engine.sender_tok, engine.tok, row["content_ids"])
    assert list(alignment.source_ids) == ids
    indices = torch.tensor(alignment.source_indices, device="cpu")
    return L.transferred(engine, len(row["content_ids"]), raw, indices, None, cache_chunk)


def score_prompt(engine, memory, task, row, gold, recovered, prefill_chunk, cache_chunk, prior=None):
    body = translated(engine, row, prefill_chunk, cache_chunk)
    assert body.get_seq_length() == len(engine.prefix_ids) + len(row["content_ids"])
    student_bank = B.construct(memory, body, 1)
    if recovered:
        result = L.answer(engine, task, row, body, gold, memory, student_bank)
        return dict(dst_memory=prior["conditions"]["dst_memory"], recovery_memory=result)
    current = L.answer(engine, task, row, body, gold, memory, student_bank)
    del student_bank
    native = L.prefill(engine, row, prefill_chunk)
    assert native.get_seq_length() == body.get_seq_length()
    native_bank = B.construct(memory, native, 1)
    del native
    oracle = L.answer(engine, task, row, body, gold, memory, native_bank)
    return dict(dst_memory=current, native_built_memory=oracle)


@torch.no_grad()
def main(recovered):
    tag = "recovery_eval" if recovered else "oracle"
    destination = ROOT / tag
    cohorts, info = prepare(C)
    cohort_ref = REFERENCE / "manifest.json"
    assert info == C.read(cohort_ref)["cohort"]
    engine = B.Engine()
    memory, original = B.R.load_memory(engine)
    original_hash = C.sha(original)
    weights = original_hash
    if recovered:
        completion = C.read(ROOT / "runs/completed.json")
        weights = C.sha(FINAL)
        assert completion["status"] == "completed" and completion["checkpoint_sha256"] == weights
        data = torch.load(FINAL, map_location="cpu", weights_only=True)
        assert data["manifest_sha256"] == completion["manifest_sha256"] == C.sha(ROOT / "manifest.json")
        attach(memory).load_state_dict(data["state"], strict=True)
        memory.eval().requires_grad_(False)
    engine.mode = "test"
    header = C.read(C.ROOT / "manifest.json")
    manifest = dict(cohort_sha256=C.sha(cohort_ref), script_sha256=C.sha(Path(__file__)), m32_sha256=original_hash,
                    checkpoint_sha256=weights, sfiction_metric="original document judgment", recovered=recovered,
                    answer_body="DST Rank64 translated KV", memory_inputs=["translated", "native"] if not recovered else ["translated"],
                    gpu=torch.cuda.get_device_name())
    path = destination / "manifest.json"
    if path.exists():
        assert C.read(path) == manifest
    else:
        C.save(path, manifest)
    pin = C.sha(path)
    arm = "recovery_memory" if recovered else "native_built_memory"
    records = {task: [] for task in TASKS}
    ordered = [(task, row) for task in TASKS for row in cohorts[task]]
    longest = max(ordered, key=lambda entry: len(entry[1]["context"]))
    ordered.remove(longest)
    ordered.insert(0, longest)
    for task, example in ordered:
        if (ROOT / "controls/pause_requested.json").exists():
            raise SystemExit(75)
        file = destination / "per_sample" / task / (example["id"] + ".json")
        prompt = example["before"] + example["context"] + example["after"]
        digest = C.digest(prompt)
        if file.exists():
            result = C.read(file)
            assert result["identity"] == pin and result["prompt_sha256"] == digest
        else:
            C.save(ROOT / "status.json", dict(status="running", phase=tag, task=task, id=example["id"],
                                              sample=sum(map(len, records.values())), total=COUNT*len(TASKS)))
            row = L.encode_row(engine, header, task, prompt)
            prior = None
            if recovered:
                prior = C.read(ROOT / "oracle/per_sample" / task / file.name)
                assert prior["prompt_sha256"] == digest and prior["id"] == example["id"]
            for p, t in ((L.PREFILL, L.CHUNK), (512, 256)):
                try:
                    values = score_prompt(engine, memory, task, row, example["gold"], recovered, p, t, prior)
                    break
                except torch.OutOfMemoryError:
                    if p == 512:
                        raise
                    print("Retry with smaller forward chunks", task, example["id"], flush=True)
                torch.cuda.empty_cache()
            assert engine.student_test_native_prefills == 0
            result = dict(identity=pin, id=example["id"], task=task, body_tokens=len(row["content_ids"]),
                          prompt_sha256=digest, gold=example["gold"], conditions=values)
            C.save(file, result)
        records[task].append(result)
        summarize(destination, records, arm, pin)
        if len(records[task]) == 1 or len(records[task]) % 8 == 0:
            print(tag, task, len(records[task]), "/", COUNT, flush=True)
    C.save(ROOT / "status.json", dict(status="running", phase=tag+"_completed",
                                      sample=COUNT*len(TASKS), total=COUNT*len(TASKS)))


if __name__ == "__main__":
    main({"oracle": False, "recovered": True}[sys.argv[1]])
