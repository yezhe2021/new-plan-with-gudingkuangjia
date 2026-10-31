# Wrong-donor KV probe: does "no change at layers 1–18" mean irrelevant or accurate?

Server B only. A/C/D and both LLMs frozen; no training, no Reader, no GSM8K.

**Question.** The cumulative prefix sweep showed that replacing layers 1–18 with
same-question native KV changes accuracy by at most a few points, while layers
19–30 carry the whole translated-to-native gap. Two explanations fit that
observation equally well:

- **A. those layers are not content-critical**, so swapping in native KV cannot
  change anything; or
- **B. those layers are content-critical, and the translated KV there is already
  accurate**, so there is nothing to gain.

The prefix experiment cannot separate them. This one does, with a third source.

## Design

For each of 6 layer groups (`1–6 … 31–36`), per arm, per question:

| condition | injected body K/V |
|---|---|
| `unmasked` | none (translated cache as produced) |
| `g{k}_native` | **same-question** native Qwen KV (oracle repair) |
| `g{k}_donor` | **different-question** native Qwen KV (wrong content) |

The donor is same-prompt-format native KV from another question of the same
task, chosen by closest body-token length with a deterministic id tiebreak;
HotpotQA additionally requires a different question string. It is placed on the
target's own body positions by nearest-neighbour resampling, so shape, dtype and
position grid match while the content is wrong.

Reading, fixed before inference:

- donor ≪ translated and native ≈ translated → **group matters, translation already accurate**
- donor ≈ translated → **group not content-critical**
- native ≫ translated and donor ≈ translated → **translation inaccurate here**

## Scope

ARC 128, HellaSwag 64, HotpotQA 64 — same immutable parent order as the earlier
64-question experiments; the ARC first-64 subset is reported separately so it can
be checked against them. HellaSwag and HotpotQA `unmasked` and `g*_native`
anchors are reused under oracle-manifest identity; ARC is recomputed in full
because it extends to 128 questions.

A zero-cost side measurement records the per-group relative K/V distance between
the translated cache and the same-question native cache for every arm.

## Gates

Smoke must pass before any evaluation: self-donor identity resampling is bitwise
equal to ordinary same-question replacement; donor id ≠ target id on every row;
header and suffix tensors preserved; resample indices in range; all logits
finite. The pipeline fails on error, keeps per-configuration results, runs
detached, and never uploads or monitors automatically.

```bash
python run_pipeline.py launch
```
