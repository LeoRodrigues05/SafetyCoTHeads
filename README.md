# safety_cot_heads

A research codebase on the **evaluation** of white-box safety interventions for
reasoning (chain-of-thought) language models.

**The gap.** Many inference-time / white-box safety interventions exist (safety-head
ablation, neuron ablation, refusal-direction steering, directional ablation, …), but
each is reported with its own bespoke metric — usually a single attack-success-rate.
There is **no direct, apples-to-apples way to compare two intervention methods**: an
ASR number can't tell apart "induced coherent harm" from "broke the model", or a
visibly-unsafe trace from a sanitised one.

**The contribution (dual).** (1) A **comparative** study running representative
intervention families through one controlled grid, and (2) a **proposed standardized
metric** combining three axes — **Potency**, **Quality**, and **Safety-Reasoning** —
so future intervention papers can report a comparable, decomposable score instead of
a one-off ASR. We are on the **evaluation side** of LLM research: the deliverable is a
measurement instrument, not a new attack or defence. See
[docs/general/EVALUATION_FRAMEWORK.md](docs/general/EVALUATION_FRAMEWORK.md).

The implementation substrate is **Direction A v5 (iso-ASR sweep)**: several families
of internal, inference-time perturbations applied at matched strengths to the same
harmful prompts, judged along the three axes.

> Built by merging two upstream projects and extending them:
> [`ydyjya/SafetyHeadAttribution`](https://github.com/ydyjya/SafetyHeadAttribution)
> (SHIPS / Sahara safety-head attribution) and
> [`Lott11/CoT-safety`](https://github.com/Lott11/CoT-safety) (5-label
> LLM-as-judge + BeaverTails). Original upstream code is preserved verbatim under
> [`src/safety_cot_heads/_legacy/`](src/safety_cot_heads/_legacy).

---

## 1. What we're doing

**The question.** How do white-box safety interventions *actually* differ when put on
shared axes — and can we standardize that comparison into a single, decomposable
metric? Concretely, at matched potency (iso-ASR), do methods preserve model quality
and keep the chain-of-thought a usable safety *monitor*, or not? The eval framing
(gap, 3 axes, composite-metric proposal, related-evals positioning) is in
[docs/general/EVALUATION_FRAMEWORK.md](docs/general/EVALUATION_FRAMEWORK.md); the
mechanistic design + metric definitions + hypotheses are in
[docs/direction_a/README.md](docs/direction_a/README.md).

**The grid.** For each `(model, dataset, condition)` cell we generate completions,
then judge them along several metric layers.

- **Intervention families** (`condition`, 11 total):
  - `baseline` — no intervention.
  - `ships_top{3,5,8}` — ablate the top-K safety **attention heads** (SHIPS).
  - `neurons_top{256,512,1024}` — ablate the top-K safety **MLP neurons**.
  - `steering_a{0.5,1.0,1.5}` — **activation-addition** of the unit-normalised
    **refusal direction** at dose α (suppresses refusal → induces harm).
    `steering_a1.0` is the iso-ASR anchor.
  - `steering_ablate` — **full directional ablation** (Arditi et al.): project the
    refusal direction out at every layer (α-free).
- **Datasets:** `jbb` (JailbreakBench, 100 prompts) and `bt` (BeaverTails, 98 =
  7×14 categories).
- **Models (5 active):** `qwen3_8b` and `olmo3_7b_think` (explicit `<think>`
  traces), `olmo3_7b_base`, `olmo3_7b_base_own` (interventions from base's own
  artifacts), and `llama31_8b_control`. Config-only stubs for other models exist
  under `configs/.../direction_a_v5_iso_asr/` but have not been run.

**Judges.**

| Judge | Model | Produces |
|---|---|---|
| Standard | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 5-label safety, coherence/quality gate, CoT-only monitor |
| Safety-reasoning trace | `Qwen/Qwen3-30B-A3B-Instruct-2507` (vLLM backend) | per-sentence safety-reasoning spans + 6 categories → `has_safety_reasoning`, first-position, extent |
| Pathway | fine-tuned `models/pathway_judge_14b_merged` (Qwen3-14B LoRA) | 12-label pathway taxonomy → 8-dim pathway vector + `dominant_pathway` |

The pathway judge is fine-tuned on HarmThoughts human annotations and validated at
**κ ≈ 0.96 / F1 0.98** vs that gold set (baseline 30B: κ 0.21). The safety-reasoning
trace judge runs on **vLLM** (continuous batching, ~100× faster than HF static
batching for this long-output pass — see [scripts/run_sr_vllm.sh](scripts/run_sr_vllm.sh)).
See [docs/direction_a/README.md §7](docs/direction_a/README.md).

**The three evaluation axes** (per `(model,dataset,condition)` cell). These are the
top-level decomposition the proposed composite metric combines — see
[docs/general/EVALUATION_FRAMEWORK.md](docs/general/EVALUATION_FRAMEWORK.md):

| Axis | Question | Metrics | Judge |
|---|---|---|---|
| **Potency** | did it remove safety? | 5-label safety; coherence-gated **`harmful_among_clean`** (headline ASR-clean); per-category harm | standard |
| **Quality** | did it keep the model intact? | coherence gate (`clean_rate`), repetition/degeneracy, judge helpfulness, benign-utility (MMLU/GSM8K) delta | standard |
| **Safety-Reasoning** | did the visible reasoning still engage safety? | SR-trace judge (`has_safety_reasoning`, categories, position); **monitorability gap** = `asr_final − asr_cot_pred`; 12-label pathway taxonomy (mechanism) | SR-trace + standard + pathway |

> Placement of the 12 pathway metrics under Safety-Reasoning, and the formula that
> combines the three axes, are open design decisions documented in the framework doc.

---

## 2. Repo layout

```
safety_cot_heads/
├── configs/
│   ├── models.yaml  datasets.yaml
│   └── experiments/direction_a_v5_iso_asr/      # CURRENT work
│       ├── matrix.yaml                          # single source of truth
│       └── <model_key>/                         # generated by make_v5_configs
│           ├── 0*-discovery / extraction yaml
│           ├── gen/{jbb,bt}/<condition>.yaml
│           ├── judge.yaml         # standard 30B judge
│           └── judge_14b.yaml     # fine-tuned pathway judge
├── scripts/                                     # CLI entry points (python -m scripts.X)
│   ├── run_local_pipeline.sh                    # standalone-VM driver (per model/stage)
│   ├── run_gpu0_olmo3_judging.sh / run_gpu1_llm_judging.sh   # two-GPU launchers
│   └── sbatch/                                  # SLURM drivers (cluster, optional)
├── src/safety_cot_heads/
│   ├── direction_a/      # pathway_taxonomy, monitorability, segmentation
│   ├── interventions/    # ablation hooks, steering, surgery
│   ├── generation/ judging/ analysis/ data/ models/ utils/
│   └── _legacy/          # verbatim upstream code (provenance)
├── data/annotations/                            # human-annotation validation (§6)
├── runs/direction_a_v5/                         # all outputs (gitignored)
├── docs/   tests/
```

---

## 3. One-time setup

```bash
# Local / CPU (tests, config gen, annotation scoring):
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt
pytest -q tests/                                 # no GPU / no downloads

# GPU VM (full pipeline): provisions env + smoke-tests model loading
bash scripts/setup_vm.sh
```

> **HF auth.** Llama / Qwen / OLMo weights are gated. Run `huggingface-cli login`
> or set `HF_TOKEN`. Set `HF_HOME=/scratch/hf` if `$HOME` is small.

---

## 4. Reproducing the pipeline

Every Python entry point takes `--config <yaml>`, optional `--overrides key=value …`
(OmegaConf dotlist), and `--dry-run`. The whole grid is driven from `matrix.yaml`.

### Step 0 — generate per-model configs  〔seconds〕
```bash
python -m scripts.make_v5_configs \
    --matrix configs/experiments/direction_a_v5_iso_asr/matrix.yaml
# edit ONLY matrix.yaml to add a model/dose/condition; this is idempotent.
```

### Step 1 — generate completions  〔~10–40 min/cell〕
```bash
# one model, all conditions × both datasets:
bash scripts/run_local_pipeline.sh qwen3_8b gen
# single-cell smoke:
python -m scripts.run_generation \
    --config configs/experiments/direction_a_v5_iso_asr/qwen3_8b/gen/jbb/baseline.yaml \
    --overrides dataset.n=8 batch_size=2
```

### Step 2 — standard judging (safety + coherence + monitorability)  〔~30–60 min/cell〕
```bash
bash scripts/run_local_pipeline.sh qwen3_8b judge        # uses judge.yaml (30B)
# env toggles: SKIP_PATHWAY=1 (default), SKIP_COT_ONLY=0, JUDGE_4BIT=0/1
```

### Step 3 — pathway judging (fine-tuned 14B)  〔~35 min/cell light · ~2 h/cell for long-CoT〕
```bash
CUDA_VISIBLE_DEVICES=0 PATHWAY_ONLY=1 JUDGE_CONFIG=judge_14b.yaml JUDGE_BATCH=128 \
    bash scripts/run_local_pipeline.sh qwen3_8b judge
```
Two GPUs at once — distribute models across cards (see the `run_gpu0_*`/`run_gpu1_*`
launchers, which pin `CUDA_VISIBLE_DEVICES` and loop a set of models).

### Step 3b — safety-reasoning trace judging (vLLM)  〔~30–40 min/GPU for the whole grid〕
Sentence-level safety-reasoning judge over full completions, on the **vLLM** backend
(continuous batching; ~100× faster than HF here). Splits evenly across two GPUs:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sr_vllm.sh --num-shards 2 --shard-id 0
CUDA_VISIBLE_DEVICES=1 bash scripts/run_sr_vllm.sh --num-shards 2 --shard-id 1
```
The wrapper sets the CUDA-13 toolkit paths + vLLM env this stack needs; HF fallback is
`--backend hf` on `scripts.run_v5_safety_reasoning`. Resume-safe (skips judged ids).

### Step 4 — heal & report  〔seconds〕
The standard and pathway passes write the same `summary.json`. The live writer now
**merges** (preserves blocks it didn't recompute), but if an older pass clobbered a
cell, re-aggregate from the surviving per-row jsonls (CPU-only, no re-judging):
```bash
python -m scripts.reaggregate_v5_summaries            # idempotent heal
python -m scripts.make_v5_metrics_status_report --out runs/metrics_status_report.html
.venv/bin/python -m scripts.make_v5_plots             # 10 analysis PNGs -> runs/plots/
```

---

## 5. Fine-tuning the pathway judge  〔one-time, several hours on 1 GPU〕

```bash
python -m scripts.prepare_harmthoughts_training_data   # build train/test from HarmThoughts
python -m scripts.train_pathway_judge                  # Qwen3-14B LoRA, ~2 epochs
python -m scripts.eval_pathway_judge \
    --test-data data/pathway_judge_test.jsonl \
    --finetuned-model models/pathway_judge_14b_merged  # F1 / Cohen's κ vs gold
```
Details: [docs/direction_a/README.md §7](docs/direction_a/README.md).

---

## 6. Validating the judges with human annotation  〔~1–2 h/annotator〕

A clone-and-run web tool to certify the judges by computing **human-vs-judge
Cohen's κ** on a blind, class-balanced sample (a metric is only standardizable if its
instruments are reliable). No GPU/model weights needed to annotate. The current
committed batch `batch_v5_002` covers all three task types:

- **safety_5label** (Potency) · **cot_only** (Safety-Reasoning / monitorability) ·
  **safety_reasoning** — the **Tier-2 sentence-level** task validating the SR-trace
  judge (mark each safety-reasoning sentence + category; sentence-level κ).

The batch (`tasks.json`/`judge_labels.json`/`manifest.json`) is committed, so everyone
who clones gets the **same** queries — run two annotators on it for the inter-annotator
reliability ceiling. Full setup:
[docs/general/ANNOTATION_SETUP.md](docs/general/ANNOTATION_SETUP.md).

```bash
python -m scripts.annotate_server  --batch data/annotations/batch_v5_002   # http://127.0.0.1:8765/
python -m scripts.score_annotations --batch data/annotations/batch_v5_002  # -> validation_report.html
```
See [data/annotations/README.md](data/annotations/README.md) for the metric mapping.

---

## 7. Outputs & logs

```
runs/direction_a_v5/<model>/
├── gen/<dataset>/<condition>/seed0/completions_<condition>.jsonl
└── judge/<dataset>/<condition>/seed0/
    ├── coherence.jsonl  judged_<condition>.jsonl  judge_cot_only.jsonl
    ├── monitorability_rows.jsonl  judge_pathway*.jsonl  pathway_vectors.jsonl
    ├── judge_safety_reasoning_trace.jsonl  safety_reasoning.summary.json
    └── summary.json                 # per-condition aggregates (std + pathway)
```
`runs/` is gitignored; status reports and the annotation batch are the tracked
artifacts. Standalone-VM run logs land in `logs/`.

---

## 8. Important scripts

| Script | Role |
|---|---|
| `make_v5_configs.py` | expand `matrix.yaml` → per-model configs |
| `run_local_pipeline.sh` | standalone-VM driver: `<model> {discover\|gen\|judge}` |
| `run_generation.py` | generate completions for a `gen/*.yaml` cell |
| `run_v4_jbb_judge.py` | **the live judge driver** (despite the v4 name): coherence, 5-label safety, pathway, cot-only; `--skip-*` toggles |
| `run_v5_safety_reasoning.py` / `run_sr_vllm.sh` | safety-reasoning trace judge (HF or vLLM backend); the `.sh` wrapper sets the vLLM/CUDA-13 env |
| `reaggregate_v5_summaries.py` | CPU heal of clobbered `summary.json` from raw jsonls |
| `make_v5_metrics_status_report.py` | filterable HTML coverage + metrics overview |
| `make_v5_plots.py` | 10 analysis diagrams |
| `make_annotation_batch.py` / `annotate_server.py` / `score_annotations.py` | judge-validation annotation tool (§6) |
| `train_pathway_judge.py` / `eval_pathway_judge.py` | fine-tune + evaluate the 14B pathway judge |
| `run_attribution.py` / `run_neuron_discovery.py` / `run_direction_extraction.py` | SHIPS heads / safety neurons / refusal direction discovery |

---

## 9. Tests & legacy

```bash
pytest -q tests/        # masks, JSON judge parsing, head selection, metrics — no GPU
```
Upstream code runs verbatim from `src/safety_cot_heads/_legacy/`.

---

> **Timing footnotes** 〔…〕 are rough wall-clock on a single ~140 GB-VRAM GPU
> (e.g. H200). The dominant driver for judging is **CoT length**: long-reasoning
> models (`olmo3_7b_think`) build 5–16k pathway prefix-rows per cell and take
> ~3–5× longer than `qwen3_8b`/base models. A full pathway pass for one model
> (22 cells) is ~10–13 h for base/control models and ~20–24 h for `think`.
