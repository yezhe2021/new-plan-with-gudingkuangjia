"""Evaluate folded HeteroFold-R16 on the locked eight-task DST cohort."""
from collections import Counter
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
DST = PROJECT / "depth_state_translator_fullsync_l3_to_q4_seed1234"
RANK = PROJECT / "dst_stage1_rank64_then_static_m32_a4_seed1234"
sys.path.insert(0, str(DST))
from experiment_common import C, CFG, K, torch
from engine import BodyReader, Engine
from score import score
from heterofold.maps import sender_layers
from heterofold.transfer import capture_sender, map_layer, source_indices
from paper_map import load_r16

METHOD = "heterofold_r16"
QA = {"hotpotqa", "2wiki", "musique"}


@torch.no_grad()
def receiver_cache(engine, mapping, row):
    offsets, keys, values = capture_sender(engine.sender_tok, engine.sender, mapping, row["prompt"], device="cuda")
    encoded = engine.tok(row["prompt"], add_special_tokens=False, return_offsets_mapping=True)
    assert encoded["input_ids"] == row["content_ids"]
    source = source_indices(offsets, encoded["offset_mapping"], "cuda")
    header = engine.header.get_seq_length()
    positions = torch.arange(header, header + len(source), device="cuda")
    tables = C._rotary_tables(engine.model, positions)
    pairs = []
    for layer, (hk, hv) in enumerate(K.pairs(engine.header)):
        k, v = map_layer(mapping, keys, values, source, layer)
        k, v = C.finalize_layer(engine.model, layer, k, v, *tables(layer))
        pairs.append((torch.cat((hk, k[None]), dim=2), torch.cat((hv, v[None]), dim=2)))
    return C.DynamicCache(ddp_cache_data=pairs, config=engine.model.config)


class ReceiverTrace:
    def __init__(self, row, header):
        self.header = header
        self.start = header + len(row["content_ids"])
        self.suffix = row["suffix_ids"]
        self.generation = row["kind"] == "generation" or row["task"] in QA | {"gsm8k"}
        self.suffix_tokens = self.candidate_tokens = self.history_tokens = 0

    def __call__(self, _module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        assert ids is not None and kwargs.get("inputs_embeds") is None
        pos = kwargs["position_ids"]
        start = int(pos[0, 0])
        width = ids.shape[-1]
        assert start >= self.start
        assert kwargs["past_key_values"].get_seq_length() == start
        if start == self.start:
            assert ids[0, :len(self.suffix)].tolist() == self.suffix
            self.suffix_tokens += len(self.suffix)
            self.candidate_tokens += width - len(self.suffix)
        else:
            assert self.generation and width == 1
            self.history_tokens += width

    def report(self):
        assert self.suffix_tokens >= len(self.suffix)
        return dict(native_body_indices=[], native_header_tokens=0,
                    cached_public_header_tokens=self.header,
                    native_suffix_tokens=self.suffix_tokens,
                    candidate_forward_tokens=self.candidate_tokens,
                    decode_history_tokens=self.history_tokens, synthetic_memory_tokens=0,
                    answer_start=self.start)


def comparable(row, rank_pin, gated):
    path = RANK / "evaluation" / row["task"] / "per_sample" / (row["id"] + ".json")
    saved = C.read(path)
    assert saved["manifest_sha256"] == rank_pin
    assert all(saved[k] == row[k] for k in ("id", "task", "question", "gold"))
    v = saved["conditions"]
    entry = gated["samples"][row["id"]]
    assert all(entry[k] == row[k] for k in ("question", "gold", "task"))
    return dict(native=v["native"], recolor_c=v["C"], dst_rank64=v["B_rank64"], gated_dynamic=entry["scores"])


def add_raw_hellaswag(values, row):
    value = values[METHOD]
    raw = value["raw_sum_scores"]
    value["raw_prediction"] = max(range(len(raw)), key=raw.__getitem__)
    value["raw_correct"] = int(value["raw_prediction"] == row["gold"])
    for label, score in values.items():
        if label != METHOD and "raw_sum_scores" in score:
            raw = score["raw_sum_scores"]
            score["raw_correct"] = int(max(range(len(raw)), key=raw.__getitem__) == row["gold"])


def summarize(task, records):
    metrics = ["em", "f1"] if task in QA else ["correct"]
    if task == "gsm8k":
        metrics.append("legacy_correct")
    if task == "hellaswag":
        metrics.append("raw_correct")
    results = {}
    for method in records[0]:
        results[method] = {metric: sum(row[method][metric] for row in records) / len(records)
                           for metric in metrics if all(metric in row[method] for row in records)}
    paired = {method: {metric: C.paired([row[method][metric] - row[METHOD][metric] for row in records])
                       for metric in results[method]}
              for method in results if method != METHOD}
    return dict(n=len(records), scores=results, paired_minus_heterofold_r16=paired)


@torch.no_grad()
def main():
    cohort = DST / "prepared/test.jsonl"
    lock = C.read(ROOT / "DST_PROTOCOL_V1.lock.json")
    assert C.sha(cohort) == lock["prepared_sha256"]["llama"]["test"]
    assert C.sha(DST / "manifest.json") == lock["reference_manifest_sha256"]["llama_dst"]
    rows = C.read_jsonl(cohort)
    assert Counter(row["task"] for row in rows) == Counter(CFG["test_counts"])
    calibration = C.read(C.CFG["calibration"])["rows"]
    assert {C.norm(r["question"]) for r in rows}.isdisjoint(C.norm(r["question"]) for r in calibration)
    rank_pin = C.sha(RANK / "manifest.json")
    gated = C.read(ROOT / "gated_reference.json")
    assert len(gated["samples"]) == len(rows)
    saved = {r["id"]: comparable(r, rank_pin, gated) for r in rows}
    engine = Engine()
    engine.mode = "test"
    assert engine.prefix_ids == lock["public_header_ids"]
    mapping, checkpoint = load_r16(C)
    assert len(sender_layers(mapping)) <= engine.sender.config.num_hidden_layers
    method = BodyReader(engine)
    manifest = dict(protocol=lock["protocol"], cohort_role=lock["cohort_role"], test_sha256=C.sha(cohort),
                    protocol_lock_sha256=C.sha(ROOT / "DST_PROTOCOL_V1.lock.json"), checkpoint=checkpoint,
                    scorer_sha256=C.sha(DST / "score.py"), rank_manifest_sha256=rank_pin,
                    gated_reference_sha256=C.sha(ROOT / "gated_reference.json"),
                    gated_checkpoint_sha256=gated["checkpoint_sha256"],
                    code={p.name: C.sha(p) for p in (Path(__file__), ROOT / "paper_map.py")})
    manifest_path = ROOT / "tasks_manifest.json"
    if manifest_path.exists():
        assert C.read(manifest_path) == manifest
    else:
        C.save(manifest_path, manifest)
    pin = C.sha(manifest_path)
    tasks = {}
    for task, count in CFG["test_counts"].items():
        records = []
        for i, row in enumerate((r for r in rows if r["task"] == task), 1):
            path = ROOT / "tasks" / task / "per_sample" / f"{row['id']}.json"
            if path.exists():
                entry = C.read(path)
                assert entry["manifest_sha256"] == pin
            else:
                trace = ReceiverTrace(row, len(engine.prefix_ids))
                with C.decoder(engine.model).register_forward_pre_hook(trace, with_kwargs=True):
                    cache = receiver_cache(engine, mapping, row)
                    assert cache.get_seq_length() == trace.start
                    prediction = score(method, row, cache)
                del cache
                values = dict(saved[row["id"]], **{METHOD: prediction})
                if task == "hellaswag":
                    add_raw_hellaswag(values, row)
                entry = dict(id=row["id"], task=task, gold=row["gold"], manifest_sha256=pin,
                             conditions=values, receiver_trace=trace.report())
                C.save(path, entry)
            records.append(entry["conditions"])
            C.save(ROOT / "tasks_status.json", dict(status="running", task=task, sample=i, total=count))
        assert len(records) == count
        tasks[task] = summarize(task, records)
        C.save(ROOT / "tasks_summary.json", dict(status="running", manifest_sha256=pin, tasks=tasks))
        print("TASK COMPLETED", task, tasks[task]["scores"][METHOD], flush=True)
    C.save(ROOT / "tasks_summary.json", dict(status="completed", manifest_sha256=pin, tasks=tasks))
    C.save(ROOT / "tasks_status.json", dict(status="completed", samples=len(rows)))


if __name__ == "__main__":
    with (ROOT.parent / "run.lock").open("a") as resource:
        fcntl.flock(resource, fcntl.LOCK_EX)
        try:
            main()
        except Exception as exc:
            C.save(ROOT / "tasks_status.json", dict(status="failed", error=repr(exc)))
            raise
