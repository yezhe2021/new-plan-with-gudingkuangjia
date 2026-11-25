"""Run the two nested RGST widths on the locked DST training and scoring protocol."""

import gzip
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean

from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "dst_kvridge_fullbody_v1_seed1234"))
import evaluate_fixed as fixed
from experiment_common import C, CALIBRATION, CFG, K, N, PILOT, REFERENCE, Z, torch
RGST_OLD = ROOT.parent / "fullsync_residual_subspace_hybrid_r16_seed1234"
sys.path.append(str(RGST_OLD))
model_code = fixed.load_module("rgst_nested_model", ROOT / "model.py")
CHUNK, FrozenRidge, RGST, oracle, pack_native, unpack_body = (
    getattr(model_code, k) for k in ("CHUNK", "FrozenRidge", "RGST", "oracle", "pack_native", "unpack_body"))
helpers = fixed.load_module("rgst_locked_helpers", RGST_OLD / "train.py")
sampler = fixed.load_module("rgst_locked_dst_sampler", fixed.BASE / "runtime.py")
decode = fixed.load_module("rgst_locked_decoder", ROOT.parent / "fullsync_residual_generation_seed1234/run.py")
speed = fixed.load_module("rgst_nested_speed", ROOT / "speed.py")
RANKS = (16, 32)
RIDGE = PILOT / "stage_a/ridge.pt"
CAL_DOCS, CAL_LENGTH = 64, 512


def save(path, payload):
    helpers.save_torch(path, payload)


class Experiment:
    def __init__(self):
        fixed.verify()
        self.engine = Z.DE.Engine()
        self.reader = fixed.BodyReader(self.engine)
        sys.path.insert(0, str(fixed.LEGACY))
        self.legacy = fixed.load_module("rgst32_legacy_alignment", fixed.LEGACY / "experiment.py")
        self.assembler = fixed.load_module("rgst32_legacy_assembler", fixed.HERB / "evaluate.py")
        previous = C.read(RIDGE.parent / "summary.json")
        self.ridge_sha = C.sha(RIDGE)
        assert self.ridge_sha == previous["checkpoint_sha256"]
        assert previous["output_key_space"] == "Qwen post-k_norm, pre-RoPE"
        assert C.sha(CALIBRATION) == previous["source_sha256"]
        self.ridge_payload = torch.load(RIDGE, map_location="cpu", mmap=True, weights_only=True)
        self.ridge = FrozenRidge(self.ridge_payload)
        self.stats = None
        self.batch_decode = None
        self.fast_capture = False
        self.fast_decode = False
        self.fast_functional = False

    def source(self, row):
        if self.fast_capture:
            left, right = (self.legacy.unit(tok, row["prompt"]) for tok in (self.engine.sender_tok, self.engine.tok))
            assert right["body_ids"] == row["content_ids"]
            index = self.legacy.align(left, right)
            return speed.capture(self.legacy, self.engine.sender, left["ids"], index)
        return helpers.pull_raw(self.engine, self.legacy, row["prompt"], row["content_ids"])

    @torch.no_grad()
    def target(self, row):
        start = len(self.engine.prefix_ids)
        tokens = self.engine.prefix_ids + row["content_ids"]
        if self.fast_capture:
            return speed.capture(self.legacy, self.engine.model, tokens, range(start, len(tokens)))
        both = self.legacy.capture(self.engine.model, tokens)
        return tuple(t[:, start:] for t in both)

    def make_cache(self, packed):
        return C.DynamicCache(ddp_cache_data=self.assembler.assemble(
            self.engine.model, self.engine.header, unpack_body(packed), normalized=True),
            config=self.engine.model.config)

    @torch.no_grad()
    def cached_features(self, source):
        length = source[0].shape[1]
        free, _ = torch.cuda.mem_get_info()
        if length > 4096 or free < 8 * 2**30:
            return None
        return [self.ridge.features(*(x[:, start:start+CHUNK] for x in source))
                for start in range(0, length, CHUNK)]

    def predict(self, source, model=None, target=None, rank=None, prepared=None):
        def operation(k, v, *reference):
            x, old = self.ridge.features(k, v)
            if model is not None:
                old = model(x, old, self.stats)
            elif rank is not None:
                teacher = pack_native([r.cuda() for r in reference])
                old = oracle(old, teacher, self.stats, rank)
            return old.half()

        if prepared is not None:
            assert model is not None and rank is None
            def reconstruct(x, y):
                return model(x, y, self.stats).half()
            chunks = [checkpoint(reconstruct, *pair, use_reentrant=False, preserve_rng_state=False)
                      if torch.is_grad_enabled() else reconstruct(*pair) for pair in prepared]
            return torch.cat(chunks, 1)
        chunks = []
        for i in range(0, source[0].shape[1], CHUNK):
            args = [x[:, i:i + CHUNK] for x in (*source, *(target or ()))]
            if model is not None and torch.is_grad_enabled():
                out = checkpoint(operation, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                out = operation(*args)
            chunks.append(out)
        return torch.cat(chunks, 1)

    @torch.no_grad()
    def verify_writers(self):
        # Temperature / model parity use the same raw header and receiver body as training.
        row = next(r for r in fixed.rows("test") if r["task"] == "arc")
        source = self.source(row)
        library = fixed.load_module("rgst32_original_ridge_implementation", PILOT / "common.py")
        archive = library.RidgeStageA(self.ridge_payload).cuda().eval()
        inputs = [t[:, :16].cuda() for t in source]
        reference = pack_native([x[0] for x in archive(*(t[None] for t in inputs))])
        _, current = self.ridge.features(*inputs)
        torch.testing.assert_close(current, reference, atol=0.006, rtol=0.003)
        del archive
        target = self.target(row)
        native = self.engine.native_cache(row, baseline=True)
        replay = self.make_cache(pack_native([t.cuda() for t in target]))
        assert all(torch.equal(a, b) for p, q in zip(K.pairs(native), K.pairs(replay)) for a, b in zip(p, q))
        a, b = (self.reader.logits(row, c, rn=False) for c in (native, replay))
        assert torch.equal(a, b)
        C.save(ROOT / "parity.json", dict(id=row["id"], native_kv_exact=True, logits_exact=True,
            archived_ridge_max_abs_error=float((current.float() - reference.float()).abs().max())))

    def calibration_rows(self):
        fitted = {r["id"] for r in C.read(RIDGE.parent / "calibration_rows.json")}
        assert len(fitted) == 500
        with gzip.open(CALIBRATION, "rt", encoding="utf8") as stream:
            rows = [json.loads(r) for r in stream]
        random.Random(CFG["seed"]).shuffle(rows)
        examples = []
        for row in rows:
            if row["id"] in fitted:
                continue
            prompt = row["text"].lstrip()
            tokenized = self.engine.tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = tokenized["input_ids"], tokenized["offset_mapping"]
            if len(ids) <= CAL_LENGTH or offsets[CAL_LENGTH - 1][1] > offsets[CAL_LENGTH][0]:
                continue
            prompt = prompt[:offsets[CAL_LENGTH - 1][1]]
            try:
                left, right = (self.legacy.unit(tok, prompt) for tok in (self.engine.sender_tok, self.engine.tok))
                if right["body_ids"] != ids[:CAL_LENGTH]:
                    continue
                indices = self.legacy.align(left, right)
            except ValueError:
                continue
            assert len(indices) == CAL_LENGTH
            examples.append(dict(id=row["id"], prompt=prompt, content_ids=ids[:CAL_LENGTH]))
            fitted.add(row["id"])
            if len(examples) == CAL_DOCS:
                break
        assert len(examples) == CAL_DOCS
        return examples

    @torch.no_grad()
    def fit(self, examples):
        n = 0
        sx = xx = sy = sr = rr = 0
        for i, row in enumerate(examples, 1):
            source, target = self.source(row), self.target(row)
            for j in range(0, len(row["content_ids"]), CHUNK):
                x, y0 = self.ridge.features(*(t[:, j:j+CHUNK] for t in source))
                x = x.float()
                y = pack_native([t[:, j:j+CHUNK].cuda().float() for t in target])
                residual = y - y0.float()
                sx = sx + x.sum(1, keepdim=True)
                xx = xx + x.square().sum(1, keepdim=True)
                sy = sy + y.sum(1, keepdim=True)
                sr = sr + residual.sum(1, keepdim=True)
                rr = rr + residual.transpose(-1, -2) @ residual
                n += x.shape[1]
            if i % 16 == 0:
                C.log("PCA calibration", i, len(examples))
        center = sr / n
        covariance = rr / n - center.transpose(-1, -2) @ center
        covariance = (covariance + covariance.transpose(-1, -2)) / 2
        values, axes = torch.linalg.eigh(covariance.double())
        axes = axes[..., -32:].flip(-1).float()
        identity = torch.eye(32, device=axes.device)
        assert (axes.transpose(-1, -2) @ axes - identity).abs().max() < 1e-5
        self.stats = dict(center_x=sx / n, scale_x=(xx/n-(sx/n).square()).clamp_min(1e-8).sqrt(),
                          center_y=sy / n, axes=axes)
        energy = values.clamp_min(0)
        ratios = {str(rank): (energy[..., -rank:].sum(-1) / energy.sum(-1)).reshape(2, -1)
                  .mean(-1).tolist() for rank in RANKS}
        save(ROOT / "statistics.pt", {k:v.cpu() for k,v in self.stats.items()})
        C.save(ROOT / "calibration.json", dict(documents=len(examples), tokens_per_head=n,
                 overlap_with_archived_ridge=0, residual_energy_fraction_by_rank=ratios))

    def prepare(self):
        examples = self.calibration_rows()
        manifest = dict(code={name: C.sha(ROOT/name) for name in ("model.py", "train.py", "evaluate.py")},
            ridge_sha256=self.ridge_sha, fullsync_writer_sha256=C.sha(self.legacy.__file__),
            calibration_ids=[r["id"] for r in examples], calibration_source_sha256=C.sha(CALIBRATION),
            calibration_tokens=CAL_LENGTH, ranks=list(RANKS), seed=CFG["seed"],
            phase1=CFG["stage1"], phase2=CFG["stage2"], batch=CFG["effective_batch"],
            weight_decay=CFG["weight_decay"], neural_input=512, neural_hidden=128, chunk=CHUNK,
            train=CFG["train_counts"], validation=CFG["validation_counts"], test=CFG["test_counts"],
            test_ids=[r["id"] for r in fixed.rows("test")],
            data_hashes={part: C.sha(fixed.BASE/"prepared"/(part+".jsonl"))
                         for part in ("train", "validation", "test")},
            quotas=CFG["prefix_quotas_per_batch"],
            policy="legacy Full-Sync sender alignment, locked DST receiver prompts, post-k_norm/pre-RoPE")
        path = ROOT / "manifest.json"
        if path.exists():
            original = C.read(path)
            check_path = ROOT / "speed_check.json"
            if manifest != original:
                assert {k:v for k,v in manifest.items() if k != "code"} == {
                    k:v for k,v in original.items() if k != "code"}
                checked = C.read(check_path)
                assert checked["manifest_sha256"] == C.sha(path)
                self.fast_capture, self.fast_decode, self.fast_functional = (
                    checked["enabled"][key] for key in ("capture", "decoder", "functional"))
                C.save(ROOT / "runtime_revision.json", dict(original_manifest_sha256=C.sha(path),
                    code={**manifest["code"], "speed.py": C.sha(ROOT/"speed.py")},
                    check_sha256=C.sha(check_path), enabled=checked["enabled"]))
        else:
            C.save(path, manifest)
        self.identity = C.sha(path)
        if not (ROOT/"parity.json").exists():
            self.verify_writers()
        path = ROOT / "statistics.pt"
        if path.exists() and (ROOT/"calibration.json").exists():
            self.stats = torch.load(path, map_location="cuda", weights_only=True)
        else:
            self.fit(examples)

    @torch.no_grad()
    def native_prefix(self, row, native):
        sample = row["id"]
        paths = ((REFERENCE/"native_prefix"/(sample+".json"), N.identity()),
                 (fixed.BASE/"native_prefix"/(sample+".json"), C.sha(fixed.BASE/"manifest.json")),
                 (ROOT/"native_prefix"/(sample+".json"), self.identity))
        for path, expected in paths:
            if path.exists():
                data = C.read(path)
                assert data["id"] == sample and data["manifest_sha256"] == expected
                return data["tokens"]
        tokens = self.reader.rollout(row, native, rn=False, cap=CFG["prefix_max_new_tokens"])
        C.save(paths[-1][0], dict(id=sample, tokens=tokens, manifest_sha256=self.identity))
        return tokens

    @torch.no_grad()
    def student_prefixes(self, row, source, models, cached=None):
        prefixes = [self.make_cache(self.predict(source, model=m, prepared=cached)) for m in models.values()]
        limit = CFG["prefix_max_new_tokens"]
        if self.batch_decode is None:
            fixture = decode.rollout(self.engine, row, prefixes, 16)
            self.batch_decode = all(t == self.reader.rollout(row, c, rn=False, cap=16)
                                    for t,c in zip(fixture, prefixes))
            C.log("checked paired rollout", self.batch_decode)
        if self.batch_decode and self.fast_decode:
            values = speed.rollout(self.engine, row, prefixes, limit, C, K)
        else:
            values = (decode.rollout(self.engine, row, prefixes, limit) if self.batch_decode else
                      [self.reader.rollout(row, c, rn=False, cap=limit) for c in prefixes])
        return dict(zip(models, values))


def reconstruct(experiment, models, row, source, target, backward=None):
    energy = [t.float().square().mean((1, 2, 3)).clamp_min(1e-6).cuda() for t in target]
    energy = torch.stack(energy)[:, :, None]
    result = {k: 0.0 for k in models}
    size = len(row["content_ids"])
    for i in range(0, size, CHUNK):
        x, base = experiment.ridge.features(*(a[:, i:i+CHUNK] for a in source))
        native = pack_native([t[:, i:i+CHUNK].cuda().float() for t in target])
        for key, model in models.items():
            predicted = model(x, base, experiment.stats)
            term = (predicted - native).square().mean((1, 2)).reshape(2, 36, 8) / energy
            loss = term.mean() * (x.shape[1] / size)
            assert torch.isfinite(loss)
            if backward is not None:
                backward(key, loss)
            result[key] += float(loss.detach())
    return result


def functional(experiment, models, row, source, kind, epoch, backward=None):
    reader, engine = experiment.reader, experiment.engine
    gold = row["gold_tokens"]
    with torch.no_grad():
        prepared = experiment.cached_features(source) if experiment.fast_functional else None
        native = engine.native_cache(row)
        if kind == "student":
            tokens = experiment.student_prefixes(row, source, models, prepared)
        else:
            selected = gold if kind == "gold" else experiment.native_prefix(row, native)
            tokens = dict.fromkeys(models, selected)
    metrics = {}
    shared = {}
    for key, model in models.items():
        sequence = tokens[key]
        positions = C.positions_for(sequence, row, epoch, validation=epoch == 0)
        prefix = sequence[:max(positions)]
        signature = (tuple(positions), tuple(prefix))
        if signature not in shared:
            with torch.no_grad():
                shared[signature] = reader.logits(row, native, prefix, positions, rn=False)
        cache = experiment.make_cache(experiment.predict(source, model=model, prepared=prepared))
        labels = torch.tensor(gold, device="cuda")
        logits = reader.token_logits(row, cache, gold, rn=False)
        ce = torch.nn.functional.cross_entropy(logits, labels)
        if kind == "gold" and experiment.fast_functional:
            student = logits[positions]
        else:
            student = reader.logits(row, cache, prefix, positions, rn=False)
        del logits
        kl = C.vocab_kl(shared[signature], student)
        loss = ce + CFG["stage2"]["lambda_kl"] * kl
        assert torch.isfinite(loss)
        if backward is not None:
            backward(key, loss)
        metrics[key] = dict(loss=float(loss.detach()), ce=float(ce.detach()), kl=float(kl.detach()),
                            prefix_tokens=len(sequence), gold_tokens=len(gold))
    return metrics


def new_models():
    models = {}
    for k in RANKS:
        torch.manual_seed(CFG["seed"])
        models[str(k)] = RGST(k)
    assert [sum(p.numel() for p in m.parameters()) for m in models.values()] == [39_011_328, 40_200_192]
    return models


def average(entries):
    return {n: {k: mean(row[n][k] for row in entries) for k in entries[0][n]} for n in entries[0]}


@torch.no_grad()
def validate(experiment, models, stage, epoch):
    results = []
    for i, row in enumerate(fixed.rows("validation"), 1):
        src = experiment.source(row)
        if stage == "stage1":
            row_metrics = reconstruct(experiment, models, row, src, experiment.target(row))
            result = {key: dict(nmse=v) for key,v in row_metrics.items()}
        else:
            result = functional(experiment, models, row, src, "native" if i % 2 else "gold", 0)
        results.append(result)
    C.save(ROOT/stage/f"validation_epoch{epoch}.json", average(results))


def train(experiment, models, stage):
    output = ROOT / stage
    complete = output / "completed.json"
    if complete.exists():
        for n, m in models.items():
            last = torch.load(output/f"final_r{n}.pt", map_location="cuda", weights_only=True)
            assert last["manifest_sha256"] == experiment.identity
            m.load_state_dict(last["model"])
        return
    epochs = CFG[stage]["epochs"]
    per_epoch = CFG["updates_per_epoch"]
    optim = {n: torch.optim.AdamW(model.parameters(), lr=CFG[stage]["learning_rate"],
                                  weight_decay=CFG["weight_decay"], fused=True) for n,model in models.items()}
    scales = {n: torch.amp.GradScaler("cuda", init_scale=64., growth_interval=1000000) for n in models}
    previous = output / "last.pt"
    resume = 0
    if previous.exists():
        state = torch.load(previous, map_location="cuda", weights_only=True)
        assert state["manifest_sha256"] == experiment.identity
        resume = state["step"]
        for n, model in models.items():
            model.load_state_dict(state["model"][n])
            optim[n].load_state_dict(state["optimizer"][n])
            scales[n].load_state_dict(state["scaler"][n])
        del state
    timing_path = output / "timing.jsonl"
    if timing_path.exists():
        observations = C.read_jsonl(timing_path)
        accepted = [entry for entry in observations if entry["step"] <= resume]
        assert [entry["step"] for entry in accepted] == list(range(1, resume + 1))
        if len(accepted) != len(observations):
            uncommitted = observations[len(accepted):]
            C.write_jsonl(output/f"timing_uncommitted_after_{resume}.jsonl", uncommitted)
            C.write_jsonl(timing_path, accepted)
    total = per_epoch * epochs
    def store(step):
        save(previous, dict(manifest_sha256=experiment.identity, step=step,
             model={n:m.state_dict() for n,m in models.items()},
             optimizer={n:m.state_dict() for n,m in optim.items()},
             scaler={n:m.state_dict() for n,m in scales.items()}))
    for epoch in range(1, epochs+1):
        for batch_index, batch in enumerate(sampler.batches(epoch, stage), 1):
            step = (epoch-1) * per_epoch + batch_index
            if step <= resume:
                continue
            start = time.monotonic()
            for retry in range(9):
                for optimizer in optim.values():
                    optimizer.zero_grad(set_to_none=True)
                entries = []
                def backward(n, loss):
                    scales[n].scale(loss / len(batch)).backward()
                for row, kind in batch:
                    src = experiment.source(row)
                    if stage == "stage1":
                        losses = reconstruct(experiment, models, row, src, experiment.target(row), backward)
                        entries.append({key:dict(nmse=v) for key,v in losses.items()})
                    else:
                        entries.append(functional(experiment, models, row, src, kind, epoch, backward))
                for n, m in models.items():
                    scales[n].unscale_(optim[n])
                    assert all(p.grad is not None for p in m.parameters())
                norms = {n: torch.nn.utils.clip_grad_norm_(m.parameters(), CFG["gradient_clip"]) for n,m in models.items()}
                if all(bool(torch.isfinite(v)) and float(v)>0 for v in norms.values()):
                    for n in models:
                        scales[n].step(optim[n])
                        scales[n].update()
                    break
                # Keep both variants on the identical batch schedule when AMP overflows.
                for n in models:
                    scales[n].update(new_scale=scales[n].get_scale()/2 if not torch.isfinite(norms[n])
                                     else scales[n].get_scale())
                if retry == 8:
                    raise FloatingPointError(f"AMP retries exhausted at {stage} {step}")
            elapsed = time.monotonic() - start
            metrics = average(entries)
            status = dict(stage=stage, epoch=epoch, step=step, total=total, seconds=elapsed,
                          losses=metrics, retries=retry, peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20))
            C.save(ROOT/"status.json", status)
            with timing_path.open("a") as log:
                log.write(json.dumps(status)+"\n")
            if step <= 4 or step % 8 == 0:
                C.log("RGST", stage, step, total, round(elapsed,2), metrics)
            if step <= 4 or step % CFG["checkpoint_every"] == 0:
                store(step)
            experiment.engine.unchanged()
        validation = output/f"validation_epoch{epoch}.json"
        if not validation.exists():
            validate(experiment, models, stage, epoch)
    for n, model in models.items():
        save(output/f"final_r{n}.pt", dict(manifest_sha256=experiment.identity, model=model.state_dict(), step=total))
    C.save(complete, dict(status="completed", updates_per_rank=total))


def profile(experiment, models):
    path = ROOT / "profile.json"
    if path.exists():
        return
    pool = fixed.rows("train")
    profile = {}
    for task in CFG["train_counts"]:
        rows = sorted((r for r in pool if r["task"]==task), key=lambda x: len(x["content_ids"]))
        row = rows[len(rows)//2]
        item = dict(body=len(row["content_ids"]), gold=len(row["gold_tokens"]))
        source = experiment.source(row)
        for kind in ("gold", "student"):
            start = time.monotonic()
            metrics = functional(experiment, models, row, source, kind, 1,
                                 backward=lambda key, loss: loss.backward())
            for m in models.values():
                m.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            item[kind+"_seconds"] = time.monotonic()-start
            item[kind+"_prefix_lengths"] = [x["prefix_tokens"] for x in metrics.values()]
        profile[task] = item
        C.log("functional profile", task, item)
    C.save(path, profile)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")
    x = Experiment()
    x.prepare()
    models = new_models()
    (ROOT/"stage1").mkdir(exist_ok=True)
    (ROOT/"stage2").mkdir(exist_ok=True)
    profile(x, models)
    train(x, models, "stage1")
    train(x, models, "stage2")
    C.save(ROOT/"status.json", dict(stage="training_completed"))
