"""Launch the pinned DST Rank64 training code against the Ridge engine."""

import sys

from experiment_common import BASE, C, CFG, K, done, load_module, mark, rows, torch, verify
from engine import BodyReader, Engine, raw_cache

runtime = load_module("runtime", BASE / "runtime.py")
train = load_module("ridge_dst_rank_train", BASE / "train.py")
assert runtime.Engine is Engine and train.Engine is Engine


def check():
    verify()
    if done("checks/gpu_completed.json"):
        return
    engine = Engine()
    reader = BodyReader(engine)
    row = next(row for row in rows("train") if row["task"] == "arc")
    with torch.no_grad():
        raw_k, raw_v, native = engine.teacher(row)
        engine.norm_counts = [0] * 36
        oracle = raw_cache(engine.model, engine.header, raw_k.cuda(), raw_v.cuda())
        assert engine.norm_counts == [1] * 36
        engine.norm_counts = None
        target = row["gold_tokens"][:1]
        native_logits = reader.logits(row, native, target, (0, 1), rn=False)
        oracle_logits = reader.logits(row, oracle, target, (0, 1), rn=False)
        kl = float(C.vocab_kl(native_logits, oracle_logits))
        assert abs(kl) < 1e-4, kl
        del native, oracle, raw_k, raw_v, native_logits, oracle_logits
        source = engine.source(row)
        before = engine.translated(row, engine.ridge, source, checkpointed=False)
        after = engine.translated(row, engine.correction, source, checkpointed=False)
        assert all(torch.equal(a, b) for x, y in zip(K.pairs(before), K.pairs(after))
                   for a, b in zip(x, y))
        del before, after, source
    C.log("Ridge native raw-cache parity and zero residual verified", kl)
    module = engine.correction
    optimizer = torch.optim.AdamW(module.parameters(), lr=CFG["stage2"]["learning_rate"],
                                  weight_decay=CFG["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=64.0, growth_interval=1000000)
    batch = runtime.batches(1, "stage2")[0]
    _, outcome = runtime.backward_batch(engine, reader, module, optimizer, scaler, batch, "stage2", 1)
    engine.unchanged()
    mark("checks/gpu_completed.json", status="completed", raw_native_cache_kl=kl,
         zero_residual_exact_identity=True, exactly_once_k_norm=True,
         disposable_rank64_update=outcome, effective_batch=len(batch), ridge_sha256=engine.ridge_sha256)
    C.log("Ridge Rank64 smoke accepted", outcome)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check()
    else:
        assert not sys.argv[1:]
        train.main("stage2")
