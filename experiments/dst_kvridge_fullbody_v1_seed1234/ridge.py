"""Ridge calibration using the raw sender and raw Qwen teacher from DST."""

import collections
import gzip
import json
import sys

from experiment_common import CALIBRATION, C, D, RIDGE, ROOT, Z, done, load_module, mark, torch, verify


def previous_fit():
    original = sys.modules.get("common")
    try:
        load_module("common", ROOT.parent / "fullsync_kvridge_task_adapters_seed1234/common.py")
        return load_module("previous_ridge_fit", ROOT.parent / "fullsync_kvridge_task_adapters_seed1234/ridge_stage_a.py")
    finally:
        if original is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = original


class RawRidge(torch.nn.Module):
    def __init__(self, payload):
        super().__init__()
        assert payload["key_space"] == "pre-k_norm, pre-RoPE"
        self.taps = payload["taps"]
        self.register_buffer("weight", payload["weight"])
        self.register_buffer("bias", payload["bias"])

    @torch.no_grad()
    def forward(self, source_k, source_v, indices):
        outputs = []
        for role, source in enumerate((source_k, source_v)):
            take = torch.as_tensor(indices, dtype=torch.long, device=source[0].device)
            selected = torch.stack([x.index_select(0, take) for x in source]).to(self.weight).float()
            projections = []
            for layer, taps in enumerate(self.taps):
                features = torch.cat([selected[t] for t in taps], dim=-1)
                projections.append(features @ self.weight[role, layer] + self.bias[role, layer])
            outputs.append(torch.stack(projections))
        return tuple(outputs)


@torch.no_grad()
def capture_pair(engine, sample, stride):
    text = sample["text"]
    source, receiver = [
        tok(text, add_special_tokens=False, return_offsets_mapping=True)
        for tok in (engine.sender_tok, engine.tok)
    ]
    sender_offsets, target_offsets = [[tuple(s) for s in result["offset_mapping"]] for result in (source, receiver)]
    sc, tc = map(collections.Counter, (sender_offsets, target_offsets))
    sender_indices = {s: i for i, s in enumerate(sender_offsets) if sc[s] == 1}
    pairs = [(sender_indices[s], i) for i, s in enumerate(target_offsets)
             if tc[s] == 1 and s[0] < s[1] and s in sender_indices][::stride]
    assert pairs, sample["id"]
    s, t = map(list, zip(*pairs))
    row = dict(id=sample["id"], prompt=text, sender_ids=list(source["input_ids"]),
               content_ids=list(receiver["input_ids"]))
    sk, sv = engine.source(row)
    tk, tv, cache = engine.teacher(row)
    del cache
    return tuple((torch.stack([x[s] for x in left]).reshape(28, len(s), 8, 128),
                  right[:, t].reshape(36, len(t), 8, 128)) for left, right in ((sk, tk), (sv, tv)))


@torch.no_grad()
def main():
    verify()
    if done("stage_a/summary.json"):
        assert C.sha(RIDGE) == C.read("stage_a/summary.json")["checkpoint_sha256"]
        return
    previous = previous_fit()
    settings = C.read(ROOT / "manifest.json")["ridge"]
    torch.set_float32_matmul_precision("highest")
    engine = Z.DE.Engine()
    with gzip.open(CALIBRATION, "rt", encoding="utf-8") as stream:
        records = previous.select_texts([json.loads(line) for line in stream], engine.tok,
                                        settings["context"], settings["documents"])
    mark("stage_a/calibration_rows.json", documents=[{k: v for k, v in row.items() if k != "text"}
                                                     for row in records])
    buffer, statistics = [], None
    for i, record in enumerate(records, 1):
        buffer.append(capture_pair(engine, record, settings["stride"]))
        if statistics is None and i == settings["probes"]:
            taps, scores = previous.select_layers(buffer, settings["top_k"])
            mark("stage_a/source_layers.json", taps=taps, probe_r2=scores)
            statistics = previous.Covariance(taps)
        if statistics is not None and (len(buffer) >= 4 or i == len(records)):
            for start in range(0, len(buffer), 4):
                statistics.add(buffer[start:start + 4])
            buffer.clear()
        if i == 1 or i % 10 == 0 or i == len(records):
            C.log("Raw K/V Ridge calibration", i, len(records), statistics.count if statistics else "layer probes")
    assert statistics is not None and statistics.count > 1024 * settings["top_k"]
    engine.unchanged()
    del engine
    torch.cuda.empty_cache()
    weights, bias = statistics.solve(settings["ridge_lambda"])
    C.atomic_save(dict(weight=weights, bias=bias, taps=statistics.taps,
                       key_space=settings["key_space"], manifest_sha256=C.sha(ROOT / "manifest.json")), RIDGE)
    mark("stage_a/summary.json", status="completed", **settings, aligned_pairs=statistics.count,
         checkpoint_sha256=C.sha(RIDGE), capture="DST raw body, native teacher with the fixed public header",
         pairing="unique matching character spans sampled at stride 4", dtype="FP16 capture; FP32 coefficients")
    C.log("Raw K/V Ridge Stage-A complete", statistics.count)


if __name__ == "__main__":
    main()
