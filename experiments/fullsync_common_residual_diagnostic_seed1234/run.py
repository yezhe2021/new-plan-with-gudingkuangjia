"""Test shared, transferable Ridge/Recolor residuals under the legacy Full-Sync alignment."""

import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "dst_kvridge_fullbody_v1_seed1234"))
import evaluate_fixed as fixed
from experiment_common import C, CALIBRATION, PILOT, torch

SEED = 1234
LENGTH = 128
SPLITS = {"base_fit": 80, "residual_fit": 64, "test": 48}
RANK = 16
QUADRATIC_FEATURES = 128
REG = 0.01
METHODS = ("ridge", "recolor")
HEADS = 8
WIDTH = 128


def documents(legacy, sender, receiver, tap_fit_ids):
    with gzip.open(CALIBRATION, "rt", encoding="utf-8") as stream:
        source = [json.loads(line) for line in stream]
    random.Random(SEED).shuffle(source)
    chosen, excluded = [], []
    seen = set()
    for row in source:
        if row["id"] in tap_fit_ids:
            continue
        text = row["text"].lstrip()
        encoded = receiver(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if len(ids) <= LENGTH or offsets[LENGTH - 1][1] > offsets[LENGTH][0] or row["id"] in seen:
            continue
        body = text[:offsets[LENGTH - 1][1]]
        try:
            left, right = (legacy.unit(tok, body) for tok in (sender, receiver))
            if right["body_ids"] != ids[:LENGTH]:
                continue
            indices = legacy.align(left, right)
        except ValueError as exc:
            excluded.append(dict(id=row["id"], reason=str(exc)))
            continue
        assert len(indices) == LENGTH
        chosen.append(dict(id=row["id"], left=left["ids"], right=right["ids"],
                           target_start=right["prefix_count"], indices=indices))
        seen.add(row["id"])
        if len(chosen) == sum(SPLITS.values()):
            break
    assert len(chosen) == sum(SPLITS.values())
    return chosen, excluded


def blocks(legacy, source, receiver, rows):
    src, dst = ([], []), ([], [])
    for i, row in enumerate(rows, 1):
        a = legacy.capture(source, row["left"])
        b = legacy.capture(receiver, row["right"])
        start = row["target_start"]
        for role in range(2):
            src[role].append(a[role][:, row["indices"]].contiguous())
            dst[role].append(b[role][:, start:start + LENGTH].contiguous())
        if i % 32 == 0:
            C.log("Full-Sync paired capture", i, len(rows))
    return tuple(torch.cat(group, 1) for group in src), tuple(torch.cat(group, 1) for group in dst)


def gram(a, b):
    return a.transpose(-1, -2) @ b / a.shape[1]


def identity(x):
    return torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)


def ridge(x, y):
    mx, my = x.mean(1, keepdim=True), y.mean(1, keepdim=True)
    a, b = x - mx, y - my
    lhs = gram(a, a)
    scale = lhs.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    weight = torch.linalg.solve(lhs + REG * scale * identity(lhs), gram(a, b))
    return weight, my - mx @ weight


def roots(matrix):
    matrix = (matrix + matrix.transpose(-1, -2)) / 2
    eig, vectors = torch.linalg.eigh(matrix.double())
    floor = REG * eig.mean(-1, keepdim=True).clamp_min(1e-7)
    sqrt = (eig.clamp_min(0) + floor).sqrt()
    transposed = vectors.transpose(-1, -2)
    return ((vectors * sqrt.unsqueeze(-2)) @ transposed).float(), (
        (vectors / sqrt.unsqueeze(-2)) @ transposed).float()


def recolor(x, y):
    mx, my = x.mean(1, keepdim=True), y.mean(1, keepdim=True)
    a, b = x - mx, y - my
    _, wx = roots(gram(a, a))
    sy, wy = roots(gram(b, b))
    u, _, vh = torch.linalg.svd(wx @ gram(a, b) @ wy, full_matrices=False)
    weight = wx @ (u @ vh) @ sy
    return weight, my - mx @ weight


def predict(x, parameters):
    weight, bias = parameters
    return x @ weight + bias


def axes(covariance):
    _, vectors = torch.linalg.eigh(covariance.double())
    return vectors[..., -RANK:].float()


def doc_energy(t):
    return t.reshape(HEADS, SPLITS["test"], LENGTH, t.shape[-1]).square().sum((2, 3))


def analyze(x, y, role, layer):
    cut_a = SPLITS["base_fit"] * LENGTH
    cut_b = cut_a + SPLITS["residual_fit"] * LENGTH
    mu = x[:, :cut_a].mean(1, keepdim=True)
    scale = x[:, :cut_a].std(1, keepdim=True).clamp_min(1e-4)
    x = (x - mu) / scale
    x_a, x_b, x_t = x[:, :cut_a], x[:, cut_a:cut_b], x[:, cut_b:]
    y_a, y_b, y_t = y[:, :cut_a], y[:, cut_a:cut_b], y[:, cut_b:]
    weights = dict(ridge=ridge(x_a, y_a), recolor=recolor(x_a, y_a))
    res_b = {name: y_b - predict(x_b, w) for name, w in weights.items()}
    res_t = {name: y_t - predict(x_t, w) for name, w in weights.items()}
    center = {name: error.mean(1, keepdim=True) for name, error in res_b.items()}
    fit = {name: error - center[name] for name, error in res_b.items()}
    tested = {name: error - center[name] for name, error in res_t.items()}
    matrices = {name: gram(error, error) for name, error in fit.items()}
    bases = {name: axes(matrix) for name, matrix in matrices.items()}
    common = axes(sum(matrix / matrix.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-8)[:, None, None]
                      for matrix in matrices.values()))
    similarity = ((bases["ridge"].transpose(-1, -2) @ bases["recolor"]).square().sum((-1, -2)) / RANK)

    # Fit a shared rank-16 residual from the mean of the two estimators' errors.
    labels = torch.cat([(fit["ridge"] + fit["recolor"]) @ common / 2,
                        *(fit[name] @ bases[name] for name in METHODS)], dim=-1)
    linear = predict(x_t, ridge(x_b, labels)).split(RANK, dim=-1)
    rng = torch.Generator(device="cuda").manual_seed(SEED + role * 1000 + layer)
    rotation = torch.randn((HEADS, x.shape[-1], QUADRATIC_FEATURES), generator=rng,
                           device=x.device) / (x.shape[-1] ** 0.5)
    quadratic = (x @ rotation).square()
    phi_scale = quadratic[:, :cut_b].std(1, keepdim=True).clamp_min(1e-4)
    augmented = torch.cat((x, quadratic / phi_scale), -1)
    expanded_fit, expanded_test = augmented[:, cut_a:cut_b], augmented[:, cut_b:]
    nonlinear = predict(expanded_test, ridge(expanded_fit, labels)).split(RANK, dim=-1)
    pooled = y_t - predict(x_t, ridge(x[:, :cut_b], y[:, :cut_b]))
    pooled_nonlinear = y_t - predict(expanded_test, ridge(augmented[:, :cut_b], y[:, :cut_b]))

    metrics = {}
    y_energy = doc_energy(y_t).clamp_min(1e-8)
    for i, name in enumerate(METHODS, 1):
        alternate = METHODS[2 - i]
        residual = res_t[name]
        error = tested[name]
        e_energy = doc_energy(error).clamp_min(1e-8)
        metrics[name] = dict(
            base=doc_energy(residual) / y_energy,
            mean_only=doc_energy(error) / y_energy,
            shared_linear=doc_energy(error - linear[0] @ common.transpose(-1, -2)) / y_energy,
            shared_quadratic=doc_energy(error - nonlinear[0] @ common.transpose(-1, -2)) / y_energy,
            individual_linear=doc_energy(error - linear[i] @ bases[name].transpose(-1, -2)) / y_energy,
            individual_quadratic=doc_energy(error - nonlinear[i] @ bases[name].transpose(-1, -2)) / y_energy,
            ridge_more_data=doc_energy(pooled) / y_energy,
            quadratic_more_data=doc_energy(pooled_nonlinear) / y_energy,
            own_subspace=doc_energy(error @ bases[name]) / e_energy,
            other_subspace=doc_energy(error @ bases[alternate]) / e_energy,
            common_subspace=doc_energy(error @ common) / e_energy,
        )
    return metrics, similarity


def report(records):
    output = {}
    for method in METHODS:
        fields = records[0][method]
        by_doc = {field: torch.cat([r[method][field] for r in records], dim=0).mean(0).tolist()
                  for field in fields}
        comparisons = (("base", "shared_quadratic"), ("base", "individual_linear"),
                       ("base", "individual_quadratic"), ("shared_linear", "shared_quadratic"),
                       ("individual_linear", "individual_quadratic"),
                       ("ridge_more_data", "quadratic_more_data"))
        output[method] = dict(means={f: sum(v) / len(v) for f, v in by_doc.items()},
                              paired_improvement={a + "_minus_" + b:
                                  C.paired([x - y for x, y in zip(by_doc[a], by_doc[b])])
                                  for a, b in comparisons}, by_doc=by_doc)
    return output


@torch.no_grad()
def run():
    fixed.verify()
    torch.set_float32_matmul_precision("highest")
    sys.path.insert(0, str(fixed.LEGACY))
    legacy = fixed.load_module("legacy_diagnostic_fullsync", fixed.LEGACY / "experiment.py")
    sender, m_s = C.load_model(C.CFG["models"]["sender"], "cuda", dtype=torch.float16)
    receiver, m_r = C.load_model(C.CFG["models"]["receiver"], "cuda", dtype=torch.float16)
    layer_file = PILOT / "stage_a/source_layers.json"
    layer_selection_count = C.read(PILOT / "stage_a/summary.json")["probes"]
    tap_fit_ids = {r["id"] for r in C.read(PILOT / "stage_a/calibration_rows.json")[:layer_selection_count]}
    assert len(tap_fit_ids) == layer_selection_count
    rows, excluded = documents(legacy, sender, receiver, tap_fit_ids)
    split_ids = {}
    cursor = 0
    for name, n in SPLITS.items():
        split_ids[name] = [row["id"] for row in rows[cursor:cursor + n]]
        cursor += n
    taps = C.read(layer_file)["taps"]
    assert len(taps) == 36 and all(len(t) == 4 for t in taps)
    manifest = dict(seed=SEED, documents=split_ids, skipped_alignment=excluded, token_count=LENGTH,
                    code_sha256=C.sha(__file__), dataset_sha256=C.sha(CALIBRATION),
                    taps_sha256=C.sha(layer_file), tap_fit_documents_excluded=sorted(tap_fit_ids),
                    legacy_code_sha256=C.sha(legacy.__file__),
                    key_coordinates="post-k_norm, pre-RoPE", rank=RANK, random_overlap=RANK / WIDTH,
                    regularization=REG, quadratic_features=QUADRATIC_FEATURES,
                    baseline="paired refits on identical four source layers and same-head input",
                    output="split by document before fitting baseline, PCA or residual probes")
    C.save(ROOT / "manifest.json", manifest)
    src, dst = blocks(legacy, m_s, m_r, rows)
    all_results = {}
    for role, name in enumerate(("key", "value")):
        role_results, similarity = [], []
        for layer, chosen in enumerate(taps):
            x = torch.cat([src[role][i] for i in chosen], dim=-1).transpose(0, 1).cuda().float()
            y = dst[role][layer].transpose(0, 1).cuda().float()
            result, overlap = analyze(x, y, role, layer)
            cpu = {m: {f: value.cpu() for f, value in fields.items()} for m, fields in result.items()}
            role_results.append(cpu)
            similarity.append(overlap.cpu())
            C.save(ROOT / "layers" / name / f"{layer:02d}.json",
                   dict(layer=layer, taps=chosen, overlap=overlap.tolist(),
                        metrics=report([cpu])))
            if (layer + 1) % 6 == 0:
                C.log(name, layer + 1, len(taps))
        overlap = torch.cat(similarity)
        all_results[name] = dict(average_overlap=float(overlap.mean()), median_overlap=float(overlap.median()),
                                 quantiles=overlap.quantile(torch.tensor([.1, .9])).tolist(),
                                 metrics=report(role_results))
        C.save(ROOT / (name + "_summary.json"), all_results[name])
    C.save(ROOT / "summary.json", dict(status="completed", manifest_sha256=C.sha(ROOT / "manifest.json"),
           n_test_documents=SPLITS["test"], measured_heads_per_role=len(taps) * HEADS,
           results=all_results))
    C.log("Full-Sync residual diagnostic complete")


if __name__ == "__main__":
    run()
