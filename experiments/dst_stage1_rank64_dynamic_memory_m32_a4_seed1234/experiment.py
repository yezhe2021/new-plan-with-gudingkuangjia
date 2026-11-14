"""Run the existing Dynamic Memory training and checks on the frozen Rank64 base."""

import sys

import experiment_common as G

# Local experiment_common and engine_frozen take precedence. The previously
# validated Dynamic Memory implementation supplies the architecture and trainer.
sys.path.append(str(G.DYNAMIC_ROOT))


def prepare():
    if (G.ROOT / "manifest.json").exists():
        G.verify()
        return
    from runtime_new import batches

    C = G.C
    G.F.verify()
    assert G.F.identity() == G.SETTINGS["static_reference_manifest_sha256"]
    assert G.F.done("completed.json") and G.B.done("runs/stage2/completed.json")
    rank_hash = C.sha(G.RANK64)
    static_hash = C.sha(G.STATIC_CHECKPOINT)
    assert rank_hash == G.SETTINGS["rank64_checkpoint_sha256"]
    assert static_hash == G.SETTINGS["static_reference_checkpoint_sha256"]
    static_manifest = C.read(G.STATIC_ROOT / "manifest.json")
    assert static_manifest["b_checkpoint_sha256"] == rank_hash
    assert G.CFG["epochs"] * G.CFG["updates_per_epoch"] == G.CFG["updates"] == 1536
    schedules = {
        str(epoch): C.digest([[(row["id"], kind) for row, kind in batch] for batch in batches(epoch)])
        for epoch in range(1, G.CFG["epochs"] + 1)
    }
    assert schedules == static_manifest["batch_schedules"]
    for row in G.rows("test"):
        G.baseline(row)
    files = [G.ROOT / "config.json", *G.ROOT.glob("*.py"),
             G.STATIC_ROOT / "manifest.json", G.STATIC_ROOT / "engine_frozen.py", G.RANK64, G.STATIC_CHECKPOINT]
    files.extend(G.DYNAMIC_ROOT / name for name in (
        "checks.py", "import_check.py", "memory_dynamic.py", "runtime_new.py", "score.py", "train_new.py",
    ))
    files.extend(G.STATIC_ROOT / "prepared" / f"{split}.jsonl" for split in ("train", "validation", "test"))
    files.extend(G.baseline_path(row) for row in G.rows("test"))
    prefixes = {str(path): C.sha(path) for path in (G.PARENT / "native_prefix").glob("*.json")}
    assert prefixes == static_manifest["native_prefix_sources"]
    C.save(G.ROOT / "manifest.json", dict(
        config=G.CFG, b_checkpoint_sha256=rank_hash,
        static_checkpoint_sha256=static_hash,
        sources={str(path): C.sha(path) for path in files}, batch_schedules=schedules,
        native_prefix_sources=prefixes, trainable_parameters=409600,
    ))
    G.mark("prepared/completed.json", status="completed", same_static_data_and_schedule=True)
    C.log("RANK64 DYNAMIC PREPARED", G.identity())


if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "prepare":
        prepare()
    elif phase == "imports":
        import runpy
        runpy.run_module("import_check", run_name="__main__")
    elif phase in ("cpu", "parity", "smoke"):
        import checks
        getattr(checks, phase)()
    elif phase == "train":
        import train_new
        train_new.main()
    elif phase == "evaluate":
        import evaluate_new
        evaluate_new.main()
