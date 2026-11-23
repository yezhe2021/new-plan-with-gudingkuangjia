"""Free generation on all frozen HERB questions using the same three conditions."""

from statistics import mean

from run import ARMS, COHORT_SHA, ROOT, BodyAttention, C, K, attention, fixed, torch


@torch.no_grad()
def run():
    fixed.verify()
    assert C.sha(fixed.HERB / "cohort.json") == COHORT_SHA
    engine = fixed.Engine()
    adapter = fixed.load_ridge_b(engine)
    engine.mode = "test"
    herb = fixed.load_module("attention_mass_generate_herb", fixed.HERB / "evaluate.py")
    originals = list(herb.render_rows(C.read(fixed.HERB / "cohort.json"), engine.tok))
    assert len(originals) == 20
    records = []
    cap = 192
    for original in originals:
        path = ROOT / "herb_generation" / "per_sample" / (original["id"] + ".json")
        if path.exists():
            records.append(C.read(path))
            continue
        segment = original["segment"]
        aligned = fixed.alignment.fullsync(segment["text"], engine.sender_tok, engine.tok, segment["ids"])
        engine.prefix_ids = original["header_ids"]
        engine.header = herb.model_forward(engine.model, engine.prefix_ids).past_key_values
        row = dict(id=original["id"], prompt=segment["text"], sender_ids=list(aligned.source_ids),
                   content_ids=list(aligned.receiver_ids), sync_source_indices=list(aligned.source_indices))
        native = engine.native_cache(row, baseline=True)
        translated = engine.translated(row, adapter, checkpointed=False)
        start, end = len(engine.prefix_ids), native.get_seq_length()
        assert end == translated.get_seq_length() == start + len(segment["ids"])
        reference = [key for key, _ in K.pairs(native)]
        results = {}
        for name in ARMS:
            base = native if name == "native" else translated
            cache = C.DynamicCache(ddp_cache_data=K.pairs(base), config=engine.model.config)
            controller = BodyAttention(start, end, reference if name == "mass_oracle" else None)
            controller.collect = False
            with attention(engine.model, controller):
                results[name] = herb.generate(engine.model, engine.tok, original["suffix_ids"], cache, cap, original)
            assert base.get_seq_length() == end
            del cache
        record = dict(id=original["id"], question=original["question"], type=original["type"],
                      gold=original["ground_truth"], results=results)
        C.save(path, record)
        records.append(record)
        C.log("HERB generation", len(records), len(originals), original["id"],
              {name: (v["exact"], round(v["set_f1"], 3), len(v["tokens"])) for name, v in results.items()})

    assert engine.student_test_native_prefills == 0
    engine.unchanged()
    metrics = ("exact", "set_f1", "recall", "capped")
    values = {name: {metric: mean(r["results"][name][metric] for r in records) for metric in metrics}
              for name in ARMS}
    comparison = {metric: C.paired([r["results"]["mass_oracle"][metric] - r["results"]["translated"][metric]
                                    for r in records]) for metric in metrics}
    C.save(ROOT / "herb_generation" / "summary.json",
           dict(status="completed", n=len(records), decoder="greedy", cap=cap, arms=values,
                oracle_minus_translated=comparison, cohort_sha256=COHORT_SHA,
                code_sha256=C.sha(__file__), oracle_code_sha256=C.sha(ROOT / "run.py"),
                rank64_sha256=C.read(fixed.ROOT / "runs/stage2/completed.json")["checkpoint_sha256"]))


if __name__ == "__main__":
    run()
