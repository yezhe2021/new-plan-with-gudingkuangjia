"""Evaluate the two frozen Stage-As with the same DST scorer and HERB cohort."""

import sys

from experiment_common import BASE, C, CFG, D, PROJECT, ROOT, Z, done, identity, load_module, mark, rows, torch, verify
from engine import BodyReader, Engine, ResidualTranslator

HERB = PROJECT / "herb_official_field_prefill_20_seed1234"
LEGACY = PROJECT / "heterogeneous_multidoc_fullsync_writers_v1_seed1234"
score = load_module("locked_dst_scorer", BASE / "score.py").score
sys.path.insert(0, str(D.ROOT))
try:
    alignment = load_module("locked_dst_alignment", D.ROOT / "alignment.py")
finally:
    sys.path.remove(str(D.ROOT))


def load_dst():
    stage_a = Z.DT.DepthStateTranslator(Z.DT.DSTConfig(**D.CFG["model"]), CFG["seed"]).cuda().float()
    for path, module, expected_manifest, stage in (
        (D.ROOT / "runs/stage1/epoch4_step1536.pt", stage_a, D.identity(), "stage1"),
        (BASE / "runs/stage2/epoch4_step1536.pt",
         ResidualTranslator(stage_a, rank=64, seed=CFG["seed"]).cuda().float(),
         C.sha(BASE / "manifest.json"), "stage2"),
    ):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        assert payload["manifest_sha256"] == expected_manifest and payload["stage"] == stage
        assert payload["epoch"] == 4 and payload["step"] == 1536
        module.load_state_dict(payload["state"], strict=True)
        module.eval().requires_grad_(False)
        if stage == "stage2":
            return stage_a, module


def load_ridge_b(engine):
    assert done("runs/stage2/completed.json")
    checkpoint = ROOT / "runs/stage2/epoch4_step1536.pt"
    assert C.sha(checkpoint) == C.read(ROOT / "runs/stage2/completed.json")["checkpoint_sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    assert payload["manifest_sha256"] == identity() and payload["epoch"] == 4 and payload["step"] == 1536
    engine.correction.load_state_dict(payload["state"], strict=True)
    return engine.correction.eval().requires_grad_(False)


class HistoricalMLP:
    """The archived post-k_norm Stage-A with its original Sender prefix."""

    def __init__(self, herb):
        self.legacy = load_module("fixed_eval_legacy_fullsync", LEGACY / "experiment.py")
        if str(LEGACY) not in sys.path:
            sys.path.append(str(LEGACY))
        module = load_module("fixed_eval_legacy_mlp", LEGACY / "fullsync_llama_translator.py")
        checkpoint = HERB / "inputs/llama_stage_a.pt"
        pins = herb.PINS["fullsync_ab"]
        assert C.sha(checkpoint) == pins["stage_a"] and C.sha(module.__file__) == pins["module"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        spec = self.legacy.CFG["writers"]["llama"]
        assert (payload["stage"], payload["protocol"], payload["epoch"], payload["step"]) == (
            "stage_a", spec["protocol"], spec["stage_a_epoch"], spec["stage_a_step"])
        with torch.device("meta"):
            self.module = module.NativeKVTranslator()
        self.module.load_state_dict(payload["state"], strict=True, assign=True)
        self.module.cuda().eval().requires_grad_(False)
        self.assemble = herb.assemble

    def cache(self, engine, row):
        src, dst = [self.legacy.unit(t, row["prompt"]) for t in (engine.sender_tok, engine.tok)]
        assert dst["body_ids"] == row["content_ids"]
        selected = self.legacy.align(src, dst)
        sender = self.legacy.capture(engine.sender, src["ids"])
        chunks = [[], []]
        for start in range(0, len(selected), 64):
            indices = selected[start:start + 64]
            inputs = [x[:, indices].unsqueeze(0).cuda() for x in sender]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                raw = self.module(*inputs)
            for group, value in zip(chunks, raw):
                group.append(value[0].flatten(2))
        raw = [torch.cat(group, dim=1) for group in chunks]
        return C.DynamicCache(ddp_cache_data=self.assemble(engine.model, engine.header, raw, normalized=True),
                              config=engine.model.config)


def kv_errors(engine, modules, stage):
    totals = {name: [] for name in modules}
    for row in rows("validation"):
        path = ROOT / f"{stage}/kv_validation/{row['id']}.json"
        if path.exists():
            record = C.read(path)
            assert record["manifest_sha256"] == identity()
            results = record["conditions"]
        else:
            src = engine.source(row)
            raw_k, raw_v, native = engine.teacher(row)
            del native
            expected = [raw_k.cuda().float(), raw_v.cuda().float()]
            energies = [x.square().mean((1, 2)).clamp_min(1e-6) for x in expected]
            indices = row["sync_source_indices"]
            results = {}
            for name, module in modules.items():
                errors = torch.zeros(2, device="cuda")
                for start in range(0, len(indices), CFG["token_chunk"]):
                    stop = min(len(indices), start + CFG["token_chunk"])
                    prediction = module(*src, indices[start:stop])
                    for role in range(2):
                        error = (prediction[role] - expected[role][:, start:stop]).square().mean((1, 2))
                        errors[role] += (error / energies[role]).mean() * ((stop - start) / len(indices))
                results[name] = dict(k_nmse=float(errors[0]), v_nmse=float(errors[1]),
                                     kv_nmse=float(errors.mean()))
            mark(str(path.relative_to(ROOT)), id=row["id"], task=row["task"], conditions=results)
        for name, result in results.items():
            totals[name].append(result)
    summary = {name: {key: sum(r[key] for r in cases) / len(cases) for key in cases[0]}
               for name, cases in totals.items()}
    mark(f"{stage}/kv_validation/summary.json", status="completed", n=len(rows("validation")), metrics=summary)
    return summary


def conditions(engine, row, modules, historical, evaluate):
    output = {"native": evaluate(engine.native_cache(row, baseline=True))}
    source = engine.source(row)
    for name, module in modules.items():
        output[name] = evaluate(engine.translated(row, module, source, checkpointed=False))
    if historical is not None:
        output["historical_mlp_stage_a"] = evaluate(historical.cache(engine, row))
    assert engine.student_test_native_prefills == 0
    engine.unchanged()
    return output


def case(path, row, compute):
    artifact = ROOT / path
    if artifact.exists():
        record = C.read(artifact)
        assert record["manifest_sha256"] == identity() and record["id"] == row["id"]
        return record["conditions"]
    data = compute()
    mark(path, id=row["id"], question=row["question"], gold=row["gold"], conditions=data)
    return data


def report(results, fields, target):
    names = results[0]
    return dict(n=len(results),
                metrics={name: {f: sum(r[name][f] for r in results) / len(results) for f in fields}
                         for name in names},
                paired={name: {f: C.paired([r[target][f] - r[name][f] for r in results]) for f in fields}
                        for name in names if name != target})


def score_locked(reader, row, cache):
    result = score(reader, row, cache)
    if row["task"] == "hellaswag":
        sums = result["raw_sum_scores"]
        result["raw_correct"] = int(max(range(len(sums)), key=sums.__getitem__) == row["gold"])
    return result


@torch.no_grad()
def run(phase, smoke=False):
    verify()
    assert C.sha(HERB / "cohort.json") == "1f5d84fad280e7bdb94141243b9a3156d1f6e5e67918844aee14160a83412393"
    stage = "evaluation_smoke" if smoke else ("evaluation_stage_a" if phase == "stage-a" else "evaluation")
    if done(f"{stage}/summary.json"):
        return
    engine = Engine()
    reader = BodyReader(engine)
    dst_a, dst_b = load_dst()
    modules = {"dst_stage1": dst_a, "ridge_stage_a": engine.ridge}
    if phase == "full":
        modules.update(dst_rank64=dst_b, ridge_rank64=load_ridge_b(engine))
    herb = load_module("fixed_eval_herb", HERB / "evaluate.py")
    historical = HistoricalMLP(herb) if phase == "full" else None
    kv_validation = None
    if not smoke:
        kv_validation = kv_errors(engine, {"dst_stage1": dst_a, "ridge_stage_a": engine.ridge}, stage)
    engine.mode = "test"
    results = {}
    target = "ridge_rank64" if phase == "full" else "ridge_stage_a"
    for task, count in CFG["test_counts"].items():
        items = [row for row in rows("test") if row["task"] == task]
        assert len(items) == count
        cases = []
        for i, row in enumerate(items[:1] if smoke else items, 1):
            path = f"{stage}/{task}/per_sample/{row['id']}.json"
            cases.append(case(path, row, lambda: conditions(engine, row, modules, historical,
                                                              lambda cache: score_locked(reader, row, cache))))
            if i == 1 or i % 8 == 0:
                C.log("DST fixed evaluation", stage, task, i, len(items))
        fields = (("em", "f1") if task in ("hotpotqa", "2wiki", "musique") else
                  ("correct", "legacy_correct") if task == "gsm8k" else
                  ("correct", "raw_correct") if task == "hellaswag" else ("correct",))
        results[task] = report(cases, fields, target)
        mark(f"{stage}/{task}/summary.json", status="completed", **results[task])

    herb_rows = list(herb.render_rows(C.read(HERB / "cohort.json"), engine.tok))
    assert len(herb_rows) == 20
    cases = []
    for original in herb_rows[:1] if smoke else herb_rows:
        aligned = alignment.fullsync(original["segment"]["text"], engine.sender_tok, engine.tok,
                                     original["segment"]["ids"])
        row = dict(id=original["id"], prompt=original["segment"]["text"], question=original["question"],
                   gold=original["ground_truth"], sender_ids=list(aligned.source_ids),
                   content_ids=list(aligned.receiver_ids), sync_source_indices=list(aligned.source_indices))
        engine.prefix_ids = original["header_ids"]
        engine.header = herb.model_forward(engine.model, engine.prefix_ids).past_key_values
        path = f"{stage}/herb/per_sample/{row['id']}.json"
        cases.append(case(path, row, lambda: conditions(engine, row, modules, historical,
                         lambda cache: herb.generate(engine.model, engine.tok, original["suffix_ids"],
                                                     cache, 192, original))))
        C.log("DST fixed evaluation", stage, "herb", row["id"])
    results["herb"] = report(cases, ("exact", "set_f1", "recall", "capped"), target)
    mark(f"{stage}/summary.json", status="completed", protocol="dst-fullbody-chat-v1",
         results=results, kv_validation=kv_validation, ridge_sha256=engine.ridge_sha256,
         historical_mlp="archived Sender prefix and post-k_norm cache writer" if historical else None,
         herb_protocol="frozen 20, existing Evidence→Question prompt, standard Full-Sync; no field prefill")
    C.log("DST fixed evaluation complete", stage)


if __name__ == "__main__":
    assert sys.argv[1:] in (["--smoke"], ["--stage-a"], ["--full"])
    run("full" if sys.argv[1] == "--full" else "stage-a", smoke=sys.argv[1] == "--smoke")
