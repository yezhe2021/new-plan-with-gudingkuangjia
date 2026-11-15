"""Train a Memory-guided block selector and a deployable local KV residual."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
ORACLE = ROOT.parent / "dst_rank64_oracle_block_scan_seed1234"
spec = importlib.util.spec_from_file_location("repair_oracle", ORACLE / "experiment.py")
O = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = O
spec.loader.exec_module(O)
C, K, torch = O.C, O.K, O.torch
from heterofold.attention import apply_rope, rotate_half
from torch import nn
from torch.nn import functional as F

REF = O.REFERENCE
MEMORY = REF / "runs/epoch4_step1536.pt"
LAST = ROOT / "runs/last.pt"
FINAL = ROOT / "runs/final.pt"
TASKS = tuple(O.CFG["tasks"])


def fingerprint():
    return C.sha(ROOT / "manifest.json")


def save(name, **data):
    C.save(ROOT / name, dict(manifest_sha256=fingerprint(), **data))


def read(name):
    data = C.read(ROOT / name)
    assert data["manifest_sha256"] == fingerprint(), name
    return data


def status(phase, **data):
    C.save(ROOT / "status.json", dict(time=time.strftime("%F %T"), phase=phase, **data))


def pause():
    if (ROOT / "controls/pause_requested.json").exists():
        raise SystemExit(75)


def cohort(split):
    return C.read_jsonl(ROOT / f"prepared/{split}.jsonl")


def prepare():
    path = ROOT / "manifest.json"
    if path.exists():
        verify()
        return
    O.verify()
    assert C.sha(MEMORY) == CFG["memory_sha256"]
    assert O.ref.done("completed.json")
    for split in ("train", "validation", "test"):
        if split == "train":
            rows = [r for r in O.ref.rows(split) if r["task"] in TASKS]
            assert all(sum(r["task"] == task for r in rows) == 1024 for task in TASKS)
        else:
            rows = C.read_jsonl(ORACLE / f"prepared/{split}.jsonl")
        C.write_jsonl(ROOT / f"prepared/{split}.jsonl", rows)
    identities = [{C.norm(r["question"]) for r in cohort(s)} for s in ("train", "validation", "test")]
    assert all(not a & b for i, a in enumerate(identities) for b in identities[i+1:])
    files = [*ROOT.glob("*.py"), ROOT / "config.json", *ROOT.glob("prepared/*.jsonl"),
             ORACLE / "manifest.json", ORACLE / "experiment.py", O.ref.RANK64, MEMORY]
    C.save(path, dict(config=CFG, pins={str(p): C.sha(p) for p in files},
                      block_budget="One 6-layer by floor(body tokens / 4) block",
                      inference="Frozen Dynamic Memory probes; top-one translated-body residual; ordinary Rank64 answer suffix",
                      train_samples=len(cohort("train"))))
    C.log("REPAIR PREPARED", len(cohort("train")))


def verify():
    for path, expected in C.read(ROOT / "manifest.json")["pins"].items():
        assert C.sha(path) == expected, path


def cutoff():
    budget = C.read(ROOT / "budget.json")
    return budget["deadline"] - CFG["evaluation_reserve_minutes"] * 60


def label_path(row):
    return f'labels/{row["id"]}.json'


@torch.no_grad()
def label():
    verify()
    if (ROOT / "labels_completed.json").exists():
        return
    engine = O.Engine()
    engine.mode = "test"
    reader = O.BodyReader(engine)
    training = cohort("train")
    per_task = [[r for r in training if r["task"] == task] for task in TASKS]
    rows = [row for pair in zip(*per_task) for row in pair]
    for i, row in enumerate(rows, 1):
        pause()
        path = label_path(row)
        if (ROOT / path).exists():
            read(path)
            continue
        # Reserve a substantial part of the run for training even on slow hosts.
        if time.time() >= C.read(ROOT / "budget.json")["started_at"] + 3 * 3600:
            break
        body, native = O.row_caches(engine, row)
        nll = O.gold_nll(reader, row, body)
        gains = [nll - O.gold_nll(reader, row, O.block_patch(engine, row, body, native, block))
                 for block in range(len(O.BLOCKS))]
        save(path, id=row["id"], gains=gains, base_nll=nll)
        if i == 1 or i % 16 == 0:
            status("label", sample=i, total=len(rows))
            C.log("REPAIR LABEL", i, len(rows))
    count = sum(all((ROOT / label_path(row)).is_file() for row in pair) for pair in zip(*per_task))
    assert count > 0
    save("labels_completed.json", samples=count*len(TASKS), per_task=count, requested=len(rows))
    C.log("REPAIR LABEL COMPLETE", count*len(TASKS), len(rows))


class DynamicProbe(O.ref.M.RNKV):
    def __init__(self, engine):
        cfg = O.ref.CFG
        super().__init__(engine, cfg)
        self.routing.close()
        self.routing = O.ref.AR.KeepBothRouting(engine, self.memory, cfg["replace_layers"])
        cp = torch.load(MEMORY, map_location="cpu", weights_only=True)
        assert cp["manifest_sha256"] == O.ref.identity() and cp["step"] == 1536
        self.load_state_dict(cp["state"], strict=True)
        self.eval().requires_grad_(False)


@contextmanager
def capture_queries(engine):
    captured = {}
    hooks = []
    for index, block in enumerate(C.layers_of(engine.model)):
        def record(_module, _args, value, layer=index):
            captured[layer] = value[0].detach()
        hooks.append(block.self_attn.q_norm.register_forward_hook(record))
    try:
        yield captured
    finally:
        for hook in hooks:
            hook.remove()


def bounds(engine, row, block):
    layer, sector = O.BLOCKS[block]
    count = len(row["content_ids"])
    start = len(engine.prefix_ids) + sector * count // O.SECTORS
    return layer, start, start + count // O.SECTORS


def as_tokens(x):
    return x[0].transpose(0, 1).flatten(1).float()


def log_stats(values):
    return torch.stack((values.mean(), values.amax(), values.std(unbiased=False)))


@torch.no_grad()
def features(engine, probe, row, body):
    model = engine.model
    length = body.get_seq_length()
    header = len(engine.prefix_ids)
    with capture_queries(engine) as queries:
        built = probe.construct(row, body)
    assert len(queries) == 36 and all(q.shape[0] == probe.M for q in queries.values())
    cfg = model.config
    tables = C._rotary_tables(model, torch.arange(length + probe.M, device="cuda"))
    stats, contexts = [], []
    for l, ((key, value), (_, memory)) in enumerate(zip(K.pairs(body), K.pairs(built))):
        cosine, sine = tables(l)
        query = apply_rope(queries[l], cosine[length:], sine[length:]).float()
        groups = cfg.num_attention_heads // cfg.num_key_value_heads
        sk = key[0, :, header:].float()
        q = query.view(probe.M, cfg.num_key_value_heads, groups, cfg.head_dim)
        logits = torch.einsum("mhgd,htd->hgmt", q, sk) * C.layers_of(model)[l].self_attn.scaling
        prob = logits.softmax(dim=-1)
        width = (length - header) // O.SECTORS
        layer_features = []
        for sector in range(O.SECTORS):
            start = sector * (length - header) // O.SECTORS
            end = start + width
            key_norm = sk[:, start:end].square().mean(-1).sqrt()
            value_norm = value[0, :, header+start:header+end].float().square().mean(-1).sqrt()
            selected = prob[..., start:end].sum(-1)
            layer_features.append(torch.cat((log_stats(key_norm), log_stats(value_norm),
                                             log_stats(selected), log_stats(logits[..., start:end]))))
        stats.append(torch.stack(layer_features))
        contexts.append(as_tokens(memory[:, :, length:]).mean(0))
    stacked = torch.stack(stats)
    features = [stacked[l:l+O.GROUP, s].mean(0) for l,s in O.BLOCKS]
    out = torch.stack(features)
    assert torch.isfinite(out).all()
    return out, torch.stack(contexts)


class Repair(nn.Module):
    def __init__(self, kv_width):
        super().__init__()
        h = CFG["width"]
        self.block = nn.Embedding(len(O.BLOCKS), 16)
        self.selector = nn.Sequential(nn.Linear(12 + 16, h), nn.GELU(), nn.Linear(h, 1))
        self.layer = nn.Embedding(36, 16)
        self.context = nn.Linear(kv_width, 32)
        self.residual = nn.Sequential(nn.Linear(2 * kv_width + 48, h), nn.GELU(), nn.Linear(h, 2 * kv_width))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def rank(self, feats):
        idx = torch.arange(len(O.BLOCKS), device=feats.device)
        return self.selector(torch.cat((feats, self.block(idx)), dim=-1)).squeeze(-1)

    def apply_block(self, engine, row, body, contexts, block, position_tables):
        first, start, end = bounds(engine, row, block)
        pairs = K.pairs(body)
        for l in range(first, first + O.GROUP):
            key, val = pairs[l]
            k, v = key[:, :, start:end], val[:, :, start:end]
            cos, sin = (x[start:end][None, None].float() for x in position_tables(l))
            raw = (k.float() * cos - rotate_half(k.float()) * sin) / (cos.square() + sin.square()).clamp_min(1e-12)
            token_k, token_v = as_tokens(raw), as_tokens(v)
            emb = torch.cat((self.context(contexts[l]), self.layer.weight[l])).expand(end-start, -1)
            inp = torch.cat((F.layer_norm(token_k, token_k.shape[-1:]),
                             F.layer_norm(token_v, token_v.shape[-1:]), emb), dim=-1)
            dk, dv = self.residual(inp).chunk(2, dim=-1)
            def heads(x):
                return x.reshape(end-start, k.shape[1], k.shape[-1]).transpose(0, 1)[None]
            # Apply only the delta in the original position coordinates.
            dk = heads(dk)
            key_new = (k.float() + dk * cos + rotate_half(dk) * sin).to(k.dtype)
            val_new = (v.float() + heads(dv)).to(v.dtype)
            pairs[l] = (torch.cat((key[:, :, :start], key_new, key[:, :, end:]), dim=2),
                        torch.cat((val[:, :, :start], val_new, val[:, :, end:]), dim=2))
        return C.DynamicCache(ddp_cache_data=pairs, config=engine.model.config)


def setup():
    engine = O.Engine()
    probe = DynamicProbe(engine)
    model = Repair(engine.model.config.num_key_value_heads * engine.model.config.head_dim).cuda()
    return engine, probe, O.BodyReader(engine), model


def targets(reader, row):
    gold = row["candidates"][row["gold"]] if row["task"] == "arc" else row["gold"]
    tokens = reader.engine.tok.encode(gold, add_special_tokens=False)
    assert tokens
    return tokens


def kv_loss(engine, row, student, native, block):
    layer, start, end = bounds(engine, row, block)
    pairs = zip(K.pairs(student)[layer:layer+O.GROUP], K.pairs(native)[layer:layer+O.GROUP])
    values = []
    for predicted, gold in pairs:
        for a,b in zip(predicted, gold):
            t = b[:, :, start:end].float()
            values.append((a[:, :, start:end].float() - t).square().mean() / t.square().mean().clamp_min(1e-5))
    return torch.stack(values).mean()


def update(engine, probe, reader, model, row, explore):
    body, native = O.row_caches(engine, row)
    feats, contexts = features(engine, probe, row, body)
    ranks = model.rank(feats)
    gains = torch.tensor(read(label_path(row))["gains"], device=ranks.device)
    distribution = (gains / CFG["temperature_nll"]).softmax(0)
    rank_loss = -(distribution * ranks.log_softmax(0)).sum()
    selected = int(gains.argmax()) if explore else int(ranks.argmax())
    positions = torch.arange(body.get_seq_length(), device=ranks.device)
    tables = C._rotary_tables(engine.model, positions)
    repaired = model.apply_block(engine, row, body, contexts, selected, tables)
    tokens = targets(reader, row)
    actual = reader.token_logits(row, repaired, tokens, rn=False)
    with torch.no_grad():
        teacher = reader.token_logits(row, native, tokens, rn=False)
    ce = F.cross_entropy(actual, torch.tensor(tokens, device=ranks.device))
    kl = C.vocab_kl(teacher, actual)
    kv = kv_loss(engine, row, repaired, native, selected)
    loss = ce + CFG["lambda_kl"]*kl + CFG["lambda_rank"]*rank_loss + CFG["lambda_kv"]*kv
    assert torch.isfinite(loss)
    return loss, dict(loss=float(loss.detach()), ce=float(ce.detach()), kl=float(kl.detach()),
                      rank=float(rank_loss.detach()), kv=float(kv.detach()))


def train():
    verify()
    if (ROOT / "runs/completed.json").exists():
        read("runs/completed.json")
        return
    assert (ROOT / "labels_completed.json").exists()
    torch.manual_seed(CFG["seed"])
    engine, probe, reader, model = setup()
    per_task = read("labels_completed.json")["per_task"]
    eligible = cohort("train")
    rows = [row for task in TASKS for row in [r for r in eligible if r["task"] == task][:per_task]]
    assert len(rows) == per_task * len(TASKS)
    batch_size = CFG["effective_batch"]
    batches = []
    for epoch in range(CFG["epochs"]):
        order = list(rows)
        random.Random(CFG["seed"] + epoch).shuffle(order)
        batches.extend(order[i:i+batch_size] for i in range(0, len(order), batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"])
    scaler = O.ref.T.scaler_for()
    step = 0
    if LAST.exists():
        cp = torch.load(LAST, map_location="cpu", weights_only=True)
        assert cp["manifest_sha256"] == fingerprint()
        model.load_state_dict(cp["model"], strict=True)
        optimizer.load_state_dict(cp["optimizer"])
        scaler.load_state_dict(cp["scaler"])
        step = cp["step"]
        del cp
    else:
        with torch.no_grad():
            row = max(rows, key=lambda r: len(r["content_ids"]))
            body = engine.cache(row, engine.base(row), engine.correction)
            feats, contexts = features(engine, probe, row, body)
            tables = C._rotary_tables(engine.model, torch.arange(body.get_seq_length(), device="cuda"))
            assert O.same_pairs(body, model.apply_block(engine, row, body, contexts, 0, tables))
            del body, feats, contexts
    log = ROOT / "runs/steps.jsonl"
    history = [r for r in C.read_jsonl(log) if r["step"] <= step] if log.exists() else []
    assert [r["step"] for r in history] == list(range(1, step+1))
    C.write_jsonl(log, history)

    def checkpoint():
        C.atomic_save(dict(manifest_sha256=fingerprint(), step=step, model=model.state_dict(),
                           optimizer=optimizer.state_dict(), scaler=scaler.state_dict()), LAST)

    started = time.monotonic()
    with log.open("a") as stream:
        for batch in batches[step:]:
            if (ROOT / "controls/pause_requested.json").exists():
                checkpoint()
                pause()
            if step and time.time() >= cutoff():
                break
            for retry in range(9):
                optimizer.zero_grad(set_to_none=True)
                records = []
                for i,row in enumerate(batch):
                    loss, values = update(engine, probe, reader, model, row, (step*batch_size+i) % 4 == 0)
                    scaler.scale(loss/len(batch)).backward()
                    records.append(values)
                    del loss
                outcome = O.ref.O.finish_backward(model, optimizer, scaler)
                if outcome["accepted"]:
                    break
                C.log("REPAIR AMP RETRY", step+1, retry+1, outcome["new_scale"])
            else:
                raise FloatingPointError("Unable to recover scaled repair gradients")
            engine.unchanged()
            assert all(p.grad is None and not p.requires_grad for p in probe.parameters())
            step += 1
            metrics = {k:sum(r[k] for r in records)/len(records) for k in records[0]}
            record = dict(step=step, total=len(batches), elapsed_seconds=time.monotonic()-started,
                          metrics=metrics, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            status("train", **record)
            if step == 1 or step % 4 == 0:
                C.log("REPAIR TRAIN", record)
            if step <= 4 or step % 16 == 0:
                checkpoint()
    assert step > 0
    checkpoint()
    C.atomic_save(dict(manifest_sha256=fingerprint(), step=step, model=model.state_dict()), FINAL)
    save("runs/completed.json", step=step, planned=len(batches), labels=len(rows), sha256=C.sha(FINAL))


@torch.no_grad()
def evaluate():
    verify()
    if (ROOT / "summary.json").exists():
        read("summary.json")
        return
    completed = read("runs/completed.json")
    assert C.sha(FINAL) == completed["sha256"]
    engine, probe, reader, model = setup()
    cp = torch.load(FINAL, map_location="cpu", weights_only=True)
    assert cp["manifest_sha256"] == fingerprint()
    model.load_state_dict(cp["model"], strict=True)
    model.eval()
    engine.mode = "test"
    records = {task: [] for task in TASKS}
    rows = cohort("test")
    for i,row in enumerate(rows, 1):
        pause()
        task = row["task"]
        path = f'evaluation/{task}/per_sample/{row["id"]}.json'
        if not (ROOT / path).exists():
            body = engine.cache(row, engine.base(row), engine.correction)
            feats, contexts = features(engine, probe, row, body)
            selected = int(model.rank(feats).argmax())
            tables = C._rotary_tables(engine.model, torch.arange(body.get_seq_length(), device="cuda"))
            repaired = model.apply_block(engine, row, body, contexts, selected, tables)
            prediction = reader.score(row, repaired, rn=False)
            nll = O.gold_nll(reader, row, repaired)
            assert engine.student_test_native_prefills == 0
            oracle = O.read_mark(f'evaluation/{task}/per_sample/{row["id"]}.json')
            available = O.read_mark(f'scan/test/{row["id"]}.json')["patched_nll"]
            assert oracle["id"] == row["id"] and len(available) == len(O.BLOCKS)
            save(path, id=row["id"], block=selected, scores={**oracle["scores"], "repair": prediction},
                 nll={**oracle["nll"], "repair": nll, "selected_native_patch": available[selected]})
            engine.unchanged()
        records[task].append(read(path))
        if i == 1 or i % 8 == 0 or i == len(rows):
            status("evaluation", sample=i, total=len(rows), task=task)
            C.log("REPAIR EVAL", i, len(rows))
    arms = ("rank64", "native", "fixed", "random", "oracle", "repair")
    results = {}
    lines = ["# Memory-guided KV repair", "", "| Task | Metric | Rank64 | Native | Fixed Oracle patch | Random Oracle patch | Per-question Oracle | Learned repair |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for task,group in records.items():
        stats, paired = {}, {}
        binary = "correct" if task == "arc" else "em"
        for key in (("nll", "correct") if task == "arc" else ("nll", "em", "f1")):
            values = {arm:[r["nll"][arm] if key == "nll" else r["scores"][arm][key] for r in group] for arm in arms}
            stats[key] = {arm:sum(v)/len(v) for arm,v in values.items()}
            paired[key] = {arm:C.paired([a-b for a,b in zip(values["repair"], values[arm])]) for arm in arms[:-1]}
            line = " | ".join(f"{stats[key][arm]:.4f}" if key == "nll" else f"{stats[key][arm]:.2%}" for arm in arms)
            lines.append(f"| {task} | {key} | {line} |")
        choices = [r["nll"]["selected_native_patch"] for r in group]
        base = [r["nll"]["rank64"] for r in group]
        results[task] = dict(n=len(group), metrics=stats, paired=paired,
            selected_native_patch_gain=C.paired([a-b for a,b in zip(base,choices)]),
            rescued=sum(r["scores"]["repair"][binary]>r["scores"]["rank64"][binary] for r in group),
            harmed=sum(r["scores"]["repair"][binary]<r["scores"]["rank64"][binary] for r in group))
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    save("summary.json", status="completed", training=completed, tasks=results)
    status("completed")


if __name__ == "__main__":
    {"prepare":prepare, "label":label, "train":train, "evaluate":evaluate}[sys.argv[1]]()
