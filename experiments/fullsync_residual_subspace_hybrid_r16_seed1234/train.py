"""Fit and train the six locked RGST controls with legacy Full-Sync alignment."""

import gzip
import json
import random
import sys
from pathlib import Path
from statistics import mean
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "dst_kvridge_fullbody_v1_seed1234"))
import evaluate_fixed as fixed
from experiment_common import C, CALIBRATION, K, PILOT, Z, torch
from rgst import (BLOCKS, HEADS, SPECS, WIDTH, HeadMLP, normalized, pack_source, pack_target,
                  ridge_prediction, unpack_body)

REFERENCE = ROOT.parent / "fullsync_common_residual_diagnostic_seed1234"
diag = fixed.load_module("rgst_residual_fixture", REFERENCE / "run.py")
SEED = 1234
N_TRAIN = 384
N_VALID = 32
EPOCHS = 4
BATCH = 128
FUNCTION_BATCH = 8
POSITIONS = 8
TRANSLATION_CHUNK = 256
FUNCTIONAL = PILOT / "data/stage_b.json"


def save_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def pull_raw(engine, legacy, text, expected):
    src, dst = (legacy.unit(tok, text) for tok in (engine.sender_tok, engine.tok))
    assert dst["body_ids"] == expected
    index = legacy.align(src, dst)
    return tuple(x[:, index] for x in legacy.capture(engine.sender, src["ids"]))


def load_context():
    fixed.verify()
    engine = Z.DE.Engine()
    sys.path.insert(0, str(fixed.LEGACY))
    legacy = fixed.load_module("rgst_legacy", fixed.LEGACY / "experiment.py")
    assembler = fixed.load_module("rgst_cache_assembler", fixed.HERB / "evaluate.py")
    metadata = C.read(REFERENCE / "manifest.json")
    layer_file = PILOT / "stage_a/source_layers.json"
    assert metadata["taps_sha256"] == C.sha(layer_file)
    assert metadata["legacy_code_sha256"] == C.sha(legacy.__file__)
    return engine, legacy, assembler, torch.tensor(C.read(layer_file)["taps"], device="cuda"), metadata


def make_cache(engine, assembler, output):
    return C.DynamicCache(ddp_cache_data=assembler.assemble(
        engine.model, engine.header, unpack_body(output), normalized=True), config=engine.model.config)


@torch.no_grad()
def native_parity(engine, legacy, assembler):
    reader = fixed.BodyReader(engine)
    rows = fixed.rows("test")
    out = {}
    for task in ("arc", "hotpotqa"):
        row = next(r for r in rows if r["task"] == task)
        source = legacy.capture(engine.model, engine.prefix_ids + row["content_ids"])
        body = pack_target([x[:, len(engine.prefix_ids):].cuda() for x in source])
        replay = make_cache(engine, assembler, body)
        native = engine.native_cache(row, baseline=True)
        delta = [max(float((a.float() - b.float()).abs().max()) for a, b in zip(x, y))
                 for x, y in zip(K.pairs(replay), K.pairs(native))]
        a, b = [reader.logits(row, cache, rn=False) for cache in (replay, native)]
        kl = torch.nn.functional.kl_div(a.log_softmax(-1), b.log_softmax(-1),
                                       log_target=True, reduction="batchmean")
        assert max(delta) < 0.02 and a.argmax(-1).item() == b.argmax(-1).item() and float(kl) < 1e-3
        out[task] = dict(id=row["id"], key_value_max_abs=max(delta), next_logits_max_abs=float((a-b).abs().max()),
                         next_token_kl=float(kl), greedy_token_matches=True)
    C.save(ROOT / "native_replay.json", out)
    C.log("native cache replay passed", out)


def extra_documents(engine, legacy, excludes):
    with gzip.open(CALIBRATION, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    random.Random(SEED + 1).shuffle(rows)
    selected = []
    for row in rows:
        id_ = row["id"]
        if id_ in excludes:
            continue
        text = row["text"].lstrip()
        enc = engine.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        n = diag.LENGTH
        if len(ids) <= n or offsets[n - 1][1] > offsets[n][0]:
            continue
        text = text[:offsets[n - 1][1]]
        try:
            l = legacy.unit(engine.sender_tok, text)
            r = legacy.unit(engine.tok, text)
            if r["body_ids"] != ids[:n]:
                continue
            indices = legacy.align(l, r)
        except ValueError:
            continue
        selected.append(dict(id=id_, left=l["ids"], right=r["ids"], target_start=r["prefix_count"], indices=indices))
        excludes.add(id_)
        if len(selected) == N_TRAIN + N_VALID:
            break
    assert len(selected) == N_TRAIN + N_VALID, len(selected)
    return selected


def matrix_axes(covariance, rank):
    _, eigenvectors = torch.linalg.eigh(covariance.double())
    return eigenvectors[..., -rank:].flip(-1).float()


def estimate(source, target, taps):
    cut_a = diag.SPLITS["base_fit"] * diag.LENGTH
    cut_b = cut_a + diag.SPLITS["residual_fit"] * diag.LENGTH
    stats = {n: [] for n in ("center_x", "scale_x", "center_y", "ridge_w", "ridge_b",
                             "residual", "native", "random")}
    for role in range(2):
        for layer, indices in enumerate(taps.tolist()):
            raw = torch.cat([source[role][i, :cut_b] for i in indices], -1).transpose(0, 1).cuda().float()
            y = target[role][layer, :cut_b].transpose(0, 1).cuda().float()
            mx, sx = raw[:, :cut_a].mean(1, keepdim=True), raw[:, :cut_a].std(1, keepdim=True).clamp_min(1e-4)
            x = (raw - mx) / sx
            weight, bias = diag.ridge(x[:, :cut_a], y[:, :cut_a])
            cal = y[:, cut_a:]
            residual = cal - diag.predict(x[:, cut_a:], (weight, bias))
            residual = residual - residual.mean(1, keepdim=True)
            native = cal - cal.mean(1, keepdim=True)
            rng = torch.Generator(device="cuda").manual_seed(SEED + role * 1000 + layer)
            choose = torch.linalg.qr(torch.randn((HEADS, WIDTH, 16), device="cuda", generator=rng)).Q
            values = dict(center_x=mx, scale_x=sx, center_y=y[:, :cut_a].mean(1, keepdim=True),
                          ridge_w=weight, ridge_b=bias, residual=matrix_axes(diag.gram(residual, residual), 32),
                          native=matrix_axes(diag.gram(native, native), 16), random=choose)
            for name, value in values.items():
                stats[name].append(value)
    return {k: torch.cat(v) for k, v in stats.items()}


def initialize_models(stats):
    result = {}
    for name in SPECS:
        torch.manual_seed(SEED)
        result[name] = HeadMLP(name, stats)
    expected = sum(p.numel() for p in result["E3"].parameters())
    assert expected == 39_011_328
    return result


def minibatch(raw, native, ix, taps, stats):
    x = pack_source([t.index_select(1, ix).cuda() for t in raw], taps)
    y = pack_target([t.index_select(1, ix).cuda() for t in native]).float()
    return normalized(x, stats), y


def loss_rep(model, x, y):
    estimated = model.coordinates(x)
    expected = model.target_coordinates(y)
    k, v = (estimated - expected).square().chunk(2)
    return k.mean() + v.mean()


def evaluation_rep(raw, native, steps, taps, stats, models):
    values = {"E0": [], **{n: [] for n in models}}
    basis = stats["residual"][..., :16]
    with torch.no_grad():
        for indices in steps:
            x, y = minibatch(raw, native, indices, taps, stats)
            ridge = ridge_prediction(x, stats)
            for name, predicted in (("E0", ridge), *((n, m.reconstruct(x, ridge)) for n, m in models.items())):
                squared = (predicted - y).square().mean((1, 2))
                size = y.square().mean((1, 2)).clamp_min(1e-6)
                sub = ((predicted - y) @ basis).square().mean((1, 2))
                sub_size = ((y - stats["center_y"]) @ basis).square().mean((1, 2)).clamp_min(1e-6)
                cosine = torch.nn.functional.cosine_similarity(predicted, y, dim=-1).mean(1)
                rest = ((128 * squared - 16 * sub) / 112).clamp_min(0)
                y_rest = ((128 * size - 16 * ((y @ basis).square().mean((1, 2)))) / 112).clamp_min(1e-6)
                tensor = torch.stack((squared / size, sub / sub_size, rest / y_rest, cosine)).reshape(4, 2, -1).mean(-1)
                values[name].append(tensor.cpu())
    names = ("nmse", "residual16_nmse", "complement112_nmse", "cosine")
    return {name: {metric: dict(key=float(vals[0]), value=float(vals[1])) for metric, vals in
                   zip(names, torch.stack(data).mean(0))} for name, data in values.items()}


def phase1(stats, source, target, taps, models):
    first = sum(diag.SPLITS.values()) * diag.LENGTH
    valid_first = first + N_TRAIN * diag.LENGTH
    optimizer = {n: torch.optim.AdamW(m.parameters(), lr=3e-4, fused=True) for n, m in models.items()}
    best = {n: float("inf") for n in models}
    start = 1
    history = {}
    progress = ROOT / "phase1/epoch_state.pt"
    if progress.exists():
        state = torch.load(progress, map_location="cuda", weights_only=True)
        start, best, history = state["epoch"] + 1, state["best"], state["history"]
        for n, m in models.items():
            m.load_state_dict(state["models"][n])
            optimizer[n].load_state_dict(state["optimizer"][n])
    validation = torch.arange(valid_first, valid_first + N_VALID * diag.LENGTH).split(BATCH)
    for epoch in range(start, EPOCHS + 1):
        ids = first + torch.randperm(N_TRAIN * diag.LENGTH, generator=torch.Generator().manual_seed(SEED + epoch))
        for i, indices in enumerate(ids.split(BATCH), 1):
            x, y = minibatch(source, target, indices, taps, stats)
            for n, model in models.items():
                opt = optimizer[n]
                opt.zero_grad(set_to_none=True)
                loss = loss_rep(model, x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
            if i in (1, 8) or i % 32 == 0:
                C.log("Phase 1", "epoch", epoch, "batch", i, len(ids) // BATCH, "last loss", float(loss.detach()))
                C.save(ROOT / "status.json", dict(stage="phase1", epoch=epoch, batch=i, total=len(ids)//BATCH))
        validation_metrics = evaluation_rep(source, target, validation, taps, stats, models)
        history[str(epoch)] = validation_metrics
        for n, model in models.items():
            criterion = mean(validation_metrics[n]["nmse"].values())
            if criterion < best[n]:
                best[n] = criterion
                save_torch(ROOT / "phase1" / (n + ".pt"), dict(state=model.state_dict(), epoch=epoch,
                                                            validation=validation_metrics[n]))
        save_torch(progress, dict(epoch=epoch, best=best, history=history,
                                 models={n:m.state_dict() for n, m in models.items()},
                                 optimizer={n:opt.state_dict() for n, opt in optimizer.items()}))
        C.save(ROOT / "phase1/validation.json", history)
        C.log("Phase 1 validated", epoch, {n: round(v, 5) for n, v in best.items()})
    for n, model in models.items():
        model.load_state_dict(torch.load(ROOT / "phase1" / (n + ".pt"), map_location="cuda", weights_only=True)["state"])
    first_test = (diag.SPLITS["base_fit"] + diag.SPLITS["residual_fit"]) * diag.LENGTH
    steps = torch.arange(first_test, first).split(BATCH)
    C.save(ROOT / "phase1/reconstruction_test.json", evaluation_rep(source, target, steps, taps, stats, models))
    C.save(ROOT / "phase1/completed.json", dict(status="completed", epochs=EPOCHS))


def render_functional(engine, legacy, row):
    from heterofold.modeling import chat_affixes

    suffix = row["suffix"]
    header = []
    if row["mode"] == "chat":
        opening, ending = chat_affixes(engine.tok, thinking=False)
        header = engine.tok.encode(opening, add_special_tokens=False)
        suffix = ending + suffix
    encode = lambda text: engine.tok.encode(text, add_special_tokens=False)
    gold = row["candidates"][row["gold"]] if row["task"] == "choice" else row["gold"]
    if row["task"] == "generation" and row["mode"] == "chat":
        gold = gold + "<|im_end|>"
    token_ids = encode(row["body"])
    source = pull_raw(engine, legacy, row["body"], token_ids)
    prompt = dict(id=row["id"], content_ids=token_ids, suffix_ids=encode(suffix))
    assert prompt["suffix_ids"]
    labels = encode(gold)
    assert labels and len(labels) <= 224
    steps = sorted(set(round(i * (len(labels)-1) / (POSITIONS-1)) for i in range(POSITIONS)))
    return header, prompt, source, labels, steps


@torch.no_grad()
def set_header(engine, ids):
    engine.prefix_ids = ids
    if ids:
        engine.header = C.decoder(engine.model)(torch.tensor([ids], device="cuda"), use_cache=True).past_key_values
    else:
        engine.header = None


def functional_cache(engine, assembler, output):
    if engine.header is not None:
        return make_cache(engine, assembler, output)
    from heterofold.attention import rotate_half
    k, v = unpack_body(output)
    k = k.to(engine.model.dtype).unflatten(-1, (HEADS, WIDTH)).transpose(1, 2)
    v = v.to(engine.model.dtype).unflatten(-1, (HEADS, WIDTH)).transpose(1, 2)
    tables = C._rotary_tables(engine.model, torch.arange(k.shape[2], device="cuda"))
    pairs = []
    for l, (kl, vl) in enumerate(zip(k, v)):
        cos, sin = tables(l)
        pairs.append(((kl * cos + rotate_half(kl) * sin)[None], vl[None]))
    return C.DynamicCache(ddp_cache_data=pairs, config=engine.model.config)


def translate_body(source, taps, stats, model=None, native=None, oracle_rank=None):
    """Move only one slice to the accelerator and recompute its intermediates on backward."""
    def predict(key, value, *target):
        x = normalized(pack_source((key.cuda(), value.cuda()), taps), stats)
        ridge = ridge_prediction(x, stats)
        if oracle_rank is not None:
            from rgst import repair
            y = pack_target([t.cuda().float() for t in target])
            result = repair(y, ridge, stats, oracle_rank)
        else:
            result = ridge if model is None else model.reconstruct(x, ridge)
        return result.half()

    length = source[0].shape[1]
    chunks = []
    for start in range(0, length, TRANSLATION_CHUNK):
        inputs = [x[:, start:start + TRANSLATION_CHUNK] for x in (*source, *(native or ()))]
        if torch.is_grad_enabled() and model is not None:
            out = checkpoint(predict, *inputs, use_reentrant=False, preserve_rng_state=False)
        else:
            out = predict(*inputs)
        chunks.append(out)
    return torch.cat(chunks, dim=1)


@torch.no_grad()
def functional_sample(engine, legacy, row):
    header, config, source, labels, positions = render_functional(engine, legacy, row)
    set_header(engine, header)
    # Capture the complete native run once, with no gradient into Qwen.
    input_ids = header + config["content_ids"] + config["suffix_ids"] + labels[:-1]
    ids = torch.tensor([input_ids], device="cuda")
    hidden = C.decoder(engine.model)(ids, use_cache=False).last_hidden_state[0]
    offset = len(header) + len(config["content_ids"]) + len(config["suffix_ids"]) - 1
    teacher = engine.model.lm_head(hidden[[offset + i for i in positions]]).float().log_softmax(-1)
    selected = torch.tensor([labels[i] for i in positions], device="cuda")
    return config, labels, positions, teacher, selected, source


def objective(engine, assembler, reader, model, sample, taps, stats):
    row, labels, positions, teacher, selected, source = sample
    prediction = translate_body(source, taps, stats, model)
    cache = functional_cache(engine, assembler, prediction)
    logits = reader.logits(row, cache, labels[:-1], positions=positions, rn=False)
    kl = torch.nn.functional.kl_div(logits.log_softmax(-1), teacher, log_target=True, reduction="batchmean")
    ce = torch.nn.functional.cross_entropy(logits, selected)
    return ce, kl


def phase2(engine, legacy, assembler, stats, taps, models, training, validation):
    reader = fixed.BodyReader(engine)
    optim = {n: torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True) for n, m in models.items()}
    scaler = {n: torch.amp.GradScaler("cuda", init_scale=128) for n in models}
    progress = ROOT / "phase2/state.pt"
    start = 0
    if progress.exists():
        prior = torch.load(progress, map_location="cuda", weights_only=True)
        start = prior["step"]
        for n,m in models.items():
            m.load_state_dict(prior["models"][n])
            optim[n].load_state_dict(prior["optimizer"][n])
            scaler[n].load_state_dict(prior["scaler"][n])
    counts = {n: [0.0, 0.0] for n in models}
    for i, row in enumerate(training[start:], start + 1):
        sample = functional_sample(engine, legacy, row)
        for n, model in models.items():
            ce, kl = objective(engine, assembler, reader, model, sample, taps, stats)
            loss = ce + 0.5 * kl
            assert torch.isfinite(loss), (n, row["id"])
            scaler[n].scale(loss / FUNCTION_BATCH).backward()
            counts[n][0] += float(ce.detach())
            counts[n][1] += float(kl.detach())
            if i % FUNCTION_BATCH == 0:
                optimizer = optim[n]
                scaler[n].unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if i == FUNCTION_BATCH:
                    assert torch.isfinite(norm) and float(norm) > 0
                scaler[n].step(optimizer)
                scaler[n].update()
                optimizer.zero_grad(set_to_none=True)
        if i == 1 or i % 8 == 0:
            C.log("Phase 2", i, len(training), {n: [round(v / (i - start), 4) for v in values] for n,values in counts.items()})
            C.save(ROOT / "status.json", dict(stage="phase2", completed=i, total=len(training)))
        if i == 8 or i % 32 == 0 or i == len(training):
            save_torch(progress, dict(step=i, models={n:m.state_dict() for n,m in models.items()},
                       optimizer={n:o.state_dict() for n,o in optim.items()},
                       scaler={n:v.state_dict() for n,v in scaler.items()}))
    with torch.no_grad():
        results = {n: [] for n in ("E0", *models)}
        for row in validation:
            sample = functional_sample(engine, legacy, row)
            for n, model in (("E0", None), *models.items()):
                results[n].append(tuple(float(v) for v in objective(engine, assembler, reader, model, sample, taps, stats)))
    for name, model in models.items():
        save_torch(ROOT / "phase2" / (name + ".pt"), dict(state=model.state_dict(), updates=len(training)//FUNCTION_BATCH))
    C.save(ROOT / "phase2/validation.json", {n: dict(ce=mean(r[0] for r in values), kl=mean(r[1] for r in values))
                                             for n,values in results.items()})
    C.save(ROOT / "phase2/completed.json", dict(status="completed", rows=len(training)))


def run():
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    engine, legacy, assembler, taps, reference = load_context()
    native_parity(engine, legacy, assembler)
    previous, _ = diag.documents(legacy, engine.sender_tok, engine.tok, set(reference["tap_fit_documents_excluded"]))
    assert [r["id"] for r in previous] == [id_ for n in diag.SPLITS for id_ in reference["documents"][n]]
    func = C.read(FUNCTIONAL)
    training, validation = ([r for group in ("choice", "generation") for r in func[part][group]]
                            for part in ("train", "validation"))
    random.Random(SEED).shuffle(training)
    locked_path = ROOT.parent / "fullsync_value_subspace_oracle_seed1234/manifest.json"
    test = C.read(locked_path)["test"]
    feat_ids = [r["id"] for r in (*training, *validation)]
    locked_ids = {i for values in test.values() for i in values}
    assert len(set(feat_ids)) == len(feat_ids) and locked_ids.isdisjoint(feat_ids)
    excludes = set(feat_ids + [r["id"] for r in previous] + reference["tap_fit_documents_excluded"])
    extra = extra_documents(engine, legacy, excludes)
    manifest = dict(code={n: C.sha(ROOT/n) for n in ("rgst.py", "train.py", "evaluate.py")},
        reference_sha256=C.sha(REFERENCE/"manifest.json"), functional_source_sha256=C.sha(FUNCTIONAL),
        qwen_tokens=diag.LENGTH, phases=dict(representation_epochs=EPOCHS, phase1_lr=3e-4,
            phase2_lr=1e-4, phase1_token_batch=BATCH, phase2_batch=FUNCTION_BATCH, phase2_max_positions=POSITIONS,
            phase2_loss="CE + 0.5 full-vocabulary KL"),
        statistics=reference["documents"], neural_train=[r["id"] for r in extra[:N_TRAIN]],
        neural_validation=[r["id"] for r in extra[N_TRAIN:]],
        functional_train=[r["id"] for r in training], functional_validation=[r["id"] for r in validation],
        test=test, controls=["E0",*SPECS], seed=SEED, coordinate_space="post-k_norm, pre-RoPE")
    file = ROOT / "manifest.json"
    phase1_manifest = ROOT / "manifest_before_memory_fix.json"
    if phase1_manifest.exists():
        original = C.read(phase1_manifest)
        assert {k:v for k,v in manifest.items() if k != "code"} == {
            k:v for k,v in original.items() if k != "code"}
        manifest["recovery"] = dict(original_manifest_sha256=C.sha(phase1_manifest),
                                    translation_token_chunk=TRANSLATION_CHUNK,
                                    checkpointed_body_translation=True)
    if file.exists():
        assert C.read(file) == manifest
    else:
        C.save(file, manifest)

    stat_path = ROOT / "statistics.pt"
    p1_done = (ROOT/"phase1/completed.json").exists()
    if p1_done:
        stats = torch.load(stat_path, map_location="cuda", weights_only=True)
    else:
        C.save(ROOT/"status.json", dict(stage="capture_for_phase1", completed=0, total=len(previous)+len(extra)))
        source, target = diag.blocks(legacy, engine.sender, engine.model, previous + extra)
        stats = estimate(source, target, taps)
        save_torch(stat_path, {n:t.cpu() for n,t in stats.items()})
    models = initialize_models(stats)
    save_torch(ROOT / "taps.pt", taps.cpu())
    C.save(ROOT / "parameter_counts.json", {n: sum(p.numel() for p in m.parameters()) for n,m in models.items()})
    if p1_done:
        for n, model in models.items():
            model.load_state_dict(torch.load(ROOT/"phase1"/(n+".pt"), map_location="cuda", weights_only=True)["state"])
    else:
        phase1(stats, source, target, taps, models)
        del source, target
    if not (ROOT/"phase2/completed.json").exists():
        phase2(engine, legacy, assembler, stats, taps, models, training, validation)
    C.save(ROOT/"status.json", dict(stage="training_completed"))
    C.log("Both RGST training phases complete")


if __name__ == "__main__":
    run()
