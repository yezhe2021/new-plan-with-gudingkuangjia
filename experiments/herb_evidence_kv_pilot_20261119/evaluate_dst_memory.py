"""Evaluate the pinned DST Rank64 and trained Gated Dynamic Memory on HERB."""

import importlib.util
import json
from pathlib import Path

from herb_metrics import answer_metrics


ROOT = Path(__file__).resolve().parent
LATENCY = ROOT.parent / "dst_rank64_gated_dynamic_latency_seed1234" / "benchmark.py"
DST_SHA = "284ab876778ff41e615ae70e58955c93a908d37c561ab603692c364664fa7809"
RANK_SHA = "aaae0362577395681095d5f72fbc07c4bc9920e0506f79dc21e531ac618958a9"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(torch, engine, reader, row, cache, answer, cap, *, memory):
    tokens = reader.rollout(row, cache, rn=memory, cap=cap)
    tok = engine.tok
    prediction = tok.decode(tokens, skip_special_tokens=True).strip()
    target = tok.encode(answer, add_special_tokens=False)
    logits = reader.logits(row, cache, target[:-1], range(len(target)), rn=memory)
    nll = torch.nn.functional.cross_entropy(logits, torch.tensor(target, device=logits.device))
    eos = engine.model.generation_config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    stops.update((tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")))
    return dict(tokens=tokens, prediction=prediction, reached_cap=tokens[-1] not in stops,
                reference_token_nll=float(nll), **answer_metrics(prediction, answer))


def run():
    gated = load_module("herb_gated_loader", LATENCY)
    B, C, torch = gated.B, gated.C, gated.torch
    output = ROOT / "dst_rank64_gated_memory_cap192"
    reference_path = ROOT / "cap192/summary.json"
    inputs_path = ROOT / "recolor_inputs.json"
    samples_path = ROOT / "samples.json"
    reference, inputs = C.read(reference_path), C.read(inputs_path)
    assert C.sha(reference_path) == inputs["baseline_summary_sha256"]
    assert C.sha(samples_path) == reference["samples_sha256"] == inputs["samples_sha256"]
    items = {item["id"]: item for item in C.read(samples_path)["samples"]}
    assert len(items) == len(inputs["rows"]) == reference["n"]

    from alignment import fullsync
    from engine_frozen import Engine

    model_paths = reference["model_files"]
    for family, role in (("qwen", "receiver"), ("llama", "sender")):
        meta = model_paths[family]
        for name, expected in meta["sha256"].items():
            assert C.sha(Path(meta["path"]) / name) == expected, (family, name)
        C.CFG["models"][role] = meta["path"]
    dst_path = B.D.ROOT / "runs/stage1/epoch4_step1536.pt"
    rank_path = B.ROOT / "runs/stage2/epoch4_step1536.pt"
    assert C.sha(dst_path) == DST_SHA
    assert C.sha(rank_path) == RANK_SHA

    with torch.no_grad():
        engine = Engine()
        engine.mode = "test"
        reader, checkpoint = gated.load_memory(engine)
        assert C.sha(checkpoint) == C.read(gated.JOINT / "config.json")["gated_checkpoint_sha256"]
        assert all(not p.requires_grad for p in reader.parameters())
        cap = reference["max_new_tokens"]
        results = []
        for frozen in inputs["rows"]:
            item = items[frozen["id"]]
            assert engine.prefix_ids + frozen["content_ids"] + frozen["suffix_ids"] == frozen["full_ids"]
            alignment = fullsync(frozen["prompt"], engine.sender_tok, engine.tok, frozen["content_ids"])
            row = dict(frozen, sender_ids=list(alignment.source_ids), sync_source_indices=list(alignment.source_indices))
            length = len(engine.prefix_ids) + len(row["content_ids"])
            source = engine.base(row)
            body = engine.cache(row, source, engine.correction)
            assert body.get_seq_length() == length
            memory = reader.construct(row, body)
            assert memory.body is body and len(memory.bank) == 72
            assert all(t.shape[2] == reader.M for t in memory.bank)
            answer = item["ground_truth"]
            conditions = {
                "dst_rank64": score(torch, engine, reader, row, body, answer, cap, memory=False),
                "dst_rank64_gated_memory": score(torch, engine, reader, row, memory, answer, cap, memory=True),
            }
            assert body.get_seq_length() == length
            native = engine.native_cache(row, baseline=True)
            native_tokens = reader.rollout(row, native, rn=False, cap=cap)
            prior = C.read(ROOT / "cap192/per_sample" / (item["id"] + ".json"))
            assert native_tokens == prior["conditions"]["native_kv"]["tokens"], item["id"]
            result = dict(id=item["id"], conditions=conditions, native_matches_original_tokens=True,
                          body_cache_tokens=length, memory_slots=reader.M, alignment=alignment.audit)
            C.save(output / "per_sample" / (item["id"] + ".json"), result)
            results.append(result)
            engine.unchanged()
            assert engine.student_test_native_prefills == 0
            print(f"COMPLETED {len(results)}/{len(items)} {item['id']}", flush=True)
    arms = tuple(results[0]["conditions"])
    metrics = {arm: {metric: sum(r["conditions"][arm][metric] for r in results) / len(results)
                     for metric in ("em", "f1", "reference_token_nll")} for arm in arms}
    C.save(output / "summary.json", dict(
        n=len(results), max_new_tokens=cap, metrics=metrics, memory_slots=reader.M,
        native_matching_generations=sum(r["native_matches_original_tokens"] for r in results),
        cap_counts={arm: sum(r["conditions"][arm]["reached_cap"] for r in results) for arm in arms},
        dst_sha256=DST_SHA, rank64_sha256=RANK_SHA, memory_sha256=C.sha(checkpoint),
        samples_sha256=C.sha(samples_path), prepared_sha256=C.sha(inputs_path),
        baseline_summary_sha256=C.sha(reference_path), gpu=torch.cuda.get_device_name(),
        student_test_native_prefills=engine.student_test_native_prefills,
    ))


if __name__ == "__main__":
    run()
