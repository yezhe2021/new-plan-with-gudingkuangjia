# HeteroFold reproduction: Llama3.2-3B → Qwen3-4B

Independent implementation of the HeteroFold mapping/calibration algorithm. Existing Writers, checkpoints, prompts and experimental directories are not modified. This is a **different model pair from the paper**, not reproduction of its published numerical table.

## Active user-authorized existing-data run

Run `bash run_existing.sh` on server B. Calibration uses official **train** inputs only: OpenBookQA 400/100, ARC-Challenge 400/100, GSM8K 800/200 (calibration train/heldout). Gold answers and solutions are not rendered. Duplicate official dev/test question texts are excluded. Statistics give each of the three corpora equal total aligned-token mass; calibration loss gives each example equal weight. Query probes cover all causal body tokens, including the final decision prefix. Context budget is 2048; overlength inputs are omitted, not truncated.

The Recolor affine fit is closed-form (not an epoch-trained Stage-A). K-only and K/V low-rank calibration each run four epochs. QASPER is not calibration data: frozen maps are evaluated on the existing fixed Dev8/25-question Hub suite. This is an **algorithm reproduction with replacement calibration data and explicit V100 FP32 numerics**, not the original Open-R1/HotpotQA experiment. Official LongBench reproduction remains outside this run.

`prepare_existing.py` generates audited input-only manifests; `run_existing.sh` runs CPU tests, preparation, GPU gates, fitting, both calibration branches, and Hub evaluation. Existing config/data identity is checked on resume. No automatic Github upload or continuous monitoring is attached.

## Algorithm

- `token_alignment.py`: shared character-END anchors; receiver unmatched token copies the latest preceding shared anchor, or the first sender token before the first anchor. Specials are excluded.
- `layer_alignment.py`: proportional nearest layer, half-up ties, offsets −4/0/+4 clipped with duplicates retained; all source KV heads flattened and concatenated, 3072 → 1024.
- `capture_kv.py`: raw `k_proj`, before Qwen key normalization and RoPE; raw `v_proj`. Native query probes are normalized but not rotated.
- `recolor.py`: equal corpus total token weight, FP64 sufficient statistics, global RMS normalization, whiten/Procrustes/recolor, bias. Numerical eigenvalue floors are explicitly configured engineering choices.
- `calibration.py`: Eq.8 K attention KL plus **per-query-head** projected squared error; V head-summed output NMSE with attention detached. Frozen model and map; no answer supervision.
- `train.py`: separately trained K-only and K/V rank-16 residuals, zero up projection, four epochs, AdamW 3e-4/1e-4, clip 1. Each sample has equal loss weight. Best heldout combined loss selects checkpoint. Epoch-boundary restart supported; partial epochs repeat safely.
- Residuals are folded into affine maps including bias. Inference has no residual module.
- `evaluate.py`: full-context transfer plus frozen Evidence-Only/MultiQuery Hub reuse, with one source capture per paper and immutable memory. Hub loads the existing `prompt_protocol.py` unchanged for rendering, **not its copy/drop alignment**. Receiver recomputes its wrapper and query only. Mapped branches never receive native evidence.

The Ridge control is three-layer/all-head TA+Ridge, **not** the original top-8-layer/per-head KV Ridge baseline. K-only is an independently trained ablation, not just discarding V from the K/V checkpoint.

## Precision and hardware disclosure

Default: BF16 models, BF16 stored/final mapped features, FP32 RoPE. BF16-incapable devices fail explicitly. An optional V100 compatibility run requires explicitly setting `HETEROFOLD_FP32_COMPAT=1`; it uses FP32 models/final mapped features and records the deviation in effective config. **Do not report a compatibility run as strict paper numerics.** Raw mappings/residual parameter arithmetic uses FP32; final correction is quantized at the same boundary in folded inference.

Memory is bounded to one captured example, plus approximately 8 GiB FP64 CPU sufficient statistics and mapping/checkpoint tensors. Model loading is sequential to avoid two frozen models simultaneously on the V100; repeated loads cost time. CPU covariance eigendecomposition/SVD is also expensive. These costs are not represented as a communication speedup.

## Data prerequisite — currently unresolved

No Open-R1/HotpotQA/LongBench files were found in the configured server-B data directory during implementation. **No substitute corpus is used and no training has been launched.** Supply exact dataset revision and training exports. The paper's exact prompt strings and Open-R1 subset identification must be resolved before claiming strict protocol reproduction.

Input-only calibration JSONL schema:

```json
{"id":"sample-id","corpus":"open_r1","text":"problem only","question":"problem only","question_start":0,"question_end":12}
```

`question_end` must actually match the string length; the example is illustrative. No answer/solution fields are allowed. The HotpotQA adapter renders context then question; this is a declared input rendering choice, not a claim of an unpublished paper prompt.

```bash
PY=/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python
$PY make_manifests.py --open-r1-train /path/open_r1_train.jsonl --hotpotqa-train /path/hotpot_train.json --output manifests
```

Set absolute `train_manifest` / `heldout_manifest` in config.json. Default 800+800 train, 200+200 heldout; smoke uses 8+8 train, 2+2 heldout. Overlength records are rejected, never truncated. Export more candidates using larger manifest counts if necessary. Input text/ID split overlap is checked. If `evaluation_manifest` is supplied, text and question overlap with calibration are also forbidden. Provenance of raw train exports must be checked by the operator; field names alone do not prove split identity.

Full-context evaluation JSONL additionally needs `task` and `references` (list of answer strings), separate from input-only calibration manifests. Full-context sampled decoding uses temp .6 / top-p .95 / top-k 20, configured length 128. This runner is a declared diagnostic protocol until official LongBench data, per-task length/prompt and evaluator choices are provided; it is **not** an official LongBench score. QASPER Hub uses existing greedy decoding and official QASPER evaluator unchanged.

## Tests and execution

```bash
$PY tests.py
$PY gpu_gates.py
$PY -u run_pipeline.py --mode smoke --evaluate hub
$PY -u run_pipeline.py --mode formal --evaluate both
```

Explicit V100 compatibility alternative (not automatic):

```bash
HETEROFOLD_FP32_COMPAT=1 $PY -u run_pipeline.py --mode smoke --evaluate hub
```

CPU tests cover TA, clipped layer duplicates, Recolor moments/finite singular cases, zero initialization and folding, and headwise loss cancellation. GPU gates must additionally pass native raw-K/cache reconstruction, first-logit/16-token generation parity, native-attention replay, branch gradient isolation, and memory immutability. Tests write evidence to `gates/`; failures stop calibration, never loosen gates automatically. Full inference remains folded-affine only.

Formal results belong in `runs/formal/evaluation/`; smoke results cannot be reported as performance. Save effective config, split manifests, moment reports, epoch history, per-sample outputs and cache audit alongside scripts. No Github upload is performed by this implementation request.

Source: https://arxiv.org/html/2609.32259v2
