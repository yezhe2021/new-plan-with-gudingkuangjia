"""Benchmark proposed runtime changes against the running experiment's numerical path."""

import importlib.util
import sys
from pathlib import Path
from statistics import mean
from time import monotonic

ROOT = Path(__file__).resolve().parent
path = ROOT / "train_fast.py"
spec = importlib.util.spec_from_file_location("rgst_speed_train", path if path.exists() else ROOT/"train.py")
train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train
spec.loader.exec_module(train)

torch, C = train.torch, train.C


def timed(fn):
    torch.cuda.synchronize()
    begin = monotonic()
    output = fn()
    torch.cuda.synchronize()
    return output, monotonic() - begin


def take_gradients(models):
    gradients = {}
    for name, model in models.items():
        gradients[name] = [p.grad.detach().flatten()[::113].float().cpu() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
    return gradients


def check():
    torch.set_float32_matmul_precision("highest")
    x = train.Experiment()
    x.identity = C.sha(ROOT/"manifest.json")
    x.stats = torch.load(ROOT/"statistics.pt", map_location="cuda", weights_only=True)
    state = torch.load(ROOT/"stage1/last.pt", map_location="cpu", weights_only=True, mmap=True)
    assert state["manifest_sha256"] == x.identity
    models = train.new_models()
    for n, m in models.items():
        m.load_state_dict(state["model"][n])
    del state
    corpus = train.fixed.rows("train")
    records = {}
    for task in train.CFG["train_counts"]:
        rows = sorted((r for r in corpus if r["task"] == task), key=lambda r:len(r["content_ids"]))
        row = rows[len(rows)//2]
        measurements = dict(body_tokens=len(row["content_ids"]))
        pairs = []
        for active in (False, True, True, False):
            x.fast_capture = active
            data, sec = timed(lambda: (x.source(row), x.target(row)))
            pairs.append((active, sec, data))
        measurements["capture_exact"] = all(torch.equal(a, b) for j in (1,2,3)
            for raw, other in zip(pairs[0][2],pairs[j][2]) for a,b in zip(raw,other))
        for active in (False, True):
            measurements["capture_fast" if active else "capture_old"] = mean(s for a,s,_ in pairs if a == active)
        source = pairs[0][2][0]
        del pairs
        with torch.no_grad():
            caches = [x.make_cache(x.predict(source, model=m)) for m in models.values()]
            old, sec = timed(lambda: train.decode.rollout(x.engine, row, caches, 64))
            measurements["decoder_old"] = sec
            try:
                new, sec = timed(lambda: train.speed.rollout(x.engine, row, caches, 64, C, train.K))
                measurements["decoder_fast"] = sec
                measurements["decoder_exact"] = old == new
            except (RuntimeError, AssertionError, ValueError) as exc:
                measurements["decoder_fast"] = None
                measurements["decoder_exact"] = False
                measurements["decoder_error"] = str(exc)
            del caches
        for kind in ("gold", "student"):
            old, fast = [], []
            for improved in (False, True):
                x.fast_functional = improved
                x.fast_decode = improved and measurements["decoder_exact"]
                current, sec = timed(lambda: train.functional(x, models, row, source, kind, 1,
                                                               backward=lambda n, loss:loss.backward()))
                gradients = take_gradients(models)
                (fast if improved else old).append((current, sec, gradients))
            original, t0, g0 = old[0]
            proposed, t1, g1 = fast[0]
            delta = max(abs(original[n][f] - proposed[n][f]) for n in models for f in ("ce","kl"))
            grad_delta = max(float((a-b).norm()/a.norm().clamp_min(1e-6)) for n in models
                             for a,b in zip(g0[n], g1[n]))
            measurements[kind] = dict(old=t0, fast=t1, loss_delta=delta, relative_gradient_delta=grad_delta,
                close=delta < 0.01 and grad_delta < 0.03)
        records[task] = measurements
        C.log("runtime parity and timing", task, measurements)
    timing = records.values()
    capture = all(r["capture_exact"] for r in timing) and sum(r["capture_fast"] for r in timing) < sum(r["capture_old"] for r in timing)
    decoder = all(r["decoder_exact"] for r in timing) and sum(r["decoder_fast"] for r in timing) < sum(r["decoder_old"] for r in timing)
    functional = all(r[k]["close"] for r in timing for k in ("gold","student"))
    C.save(ROOT/"speed_check.json", dict(manifest_sha256=x.identity, measurements=records,
                                        enabled=dict(capture=capture, decoder=decoder, functional=functional)))


if __name__ == "__main__":
    check()
