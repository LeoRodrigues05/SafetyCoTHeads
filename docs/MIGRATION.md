# MIGRATION — upstream → safety_cot_heads

This document maps every upstream artifact onto its new home in
`safety_cot_heads/`, and then enumerates every file that is **new** in this
repository (i.e. has no upstream counterpart).

## Layout summary

```
src/safety_cot_heads/
├── ships_legacy/      # SHIPS attribution — reformatted from upstream
├── sahara_legacy/     # Sahara attribution — reformatted from upstream
├── judging/           # LLM-as-judge — prompt is upstream verbatim; runner/parse new
├── attribution/       # NEW attribution methods (coherency, quality, random)
├── analysis/          # NEW (Wilson CIs, overlap, dose-response, trajectories, plots)
├── data/              # vendored CSVs + dataset loaders
├── generation/        # generation utilities (port + new metadata schema)
├── interventions/     # ablation (hook-based, new), surgery (new), activation_patching (new)
├── models/            # HeadMaskController (new) + loading
├── utils/             # NEW (device, io, logging, seed)
└── _legacy/           # 100% verbatim upstream code (no edits)
    ├── sha/           # ydyjya/SafetyHeadAttribution
    └── cots/          # Lott11/CoT-safety
```

Naming convention (per request #4):

* **`*_legacy`** sub-packages (`ships_legacy/`, `sahara_legacy/`) hold code
  that is reformatted from upstream but kept in production-style modules.
* `_legacy/` (underscore-prefixed) holds **verbatim** upstream code that is
  imported as-is and never modified.
* All other top-level sub-packages contain code that is new to this
  repository.

(Python disallows `-` in module names, so the user-suggested `-legacy`
suffix is rendered as `_legacy`.)

## ydyjya/SafetyHeadAttribution → safety_cot_heads

| Upstream file                                    | New home                                                                | Status                                                                 |
| ------------------------------------------------ | ----------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `lib/utils/custommodel.py::CustomLlamaAttention` | `models/custom_llama.py::HeadMaskController`                            | rewritten as forward-hooks (GQA-aware, HF-version-robust)              |
| `lib/utils/custommodel.py` (whole file)          | `_legacy/sha/custommodel.py`                                            | **vendored**                                                           |
| `lib/utils/get_model.py`                         | `models/loading.py::load_model`                                         | ported; supports BnB-4bit, `attn_implementation`, `device_map`         |
| `lib/utils/batch_inference.py`                   | `generation/generate.py::generate`                                      | ported; adds `condition` label + strict JSONL                          |
| `lib/utils/load_conv.py`, `format.py`            | `_legacy/sha/load_conv.py`, `format.py`                                 | **vendored** (prompt rendering moved to `generation/prompts.py`)       |
| `lib/SHIPS/get_ships.py`                         | `ships_legacy/ships.py::SHIPS` + `aggregate_dataset_ranking`            | ported; uses `HeadMaskController`; imports `kl_divergence`+`sort_ships_dict` verbatim |
| `lib/SHIPS/pd_diff.py`                           | `_legacy/sha/pd_diff.py`                                                | **vendored** (`kl_divergence` re-exported)                             |
| `lib/SHIPS/ships_utils.py`                       | `_legacy/sha/ships_utils.py`                                            | **vendored** (`sort_ships_dict` re-exported)                           |
| `lib/Sahara/attribution.py`                      | `sahara_legacy/sahara.py::safety_head_attribution` + `get_last_hidden_states` | ported; uses `HeadMaskController`; imports `compute_subspace_similarity` verbatim |
| `lib/Sahara/svd.py`                              | `_legacy/sha/sahara_svd.py`                                             | **vendored** (`compute_subspace_similarity` re-exported)               |
| `lib/Safety_discriminator/Discriminator.py`      | `_legacy/sha/safety_discriminator.py`                                   | **vendored**; rule-based safety/refusal discriminator preserved        |
| `Generalized_Ships.ipynb`                        | `_legacy/sha/notebooks/Generalized_Ships.ipynb`                         | **vendored** workflow notebook                                         |
| `Ships_quick_start.ipynb`                        | `_legacy/sha/notebooks/Ships_quick_start.ipynb`                         | **vendored** workflow notebook                                         |
| `Surgery.ipynb`                                  | `_legacy/sha/notebooks/Surgery.ipynb`                                   | **vendored** workflow notebook; logic ported to `interventions/surgery.py` |
| `exp_data/process_data.ipynb`                    | `_legacy/sha/notebooks/process_data.ipynb`                              | **vendored** preprocessing notebook                                    |
| `data/maliciousinstruct.csv` + co.               | `data/raw/sha/*.csv`, loaders in `data/maliciousinstruct.py`, `data/jailbreakbench.py` | copied                                                                 |

## Lott11/CoT-safety → safety_cot_heads

The CoT-safety repository contains **no Jupyter notebooks**; the judge
logic lives entirely in `judge_test_script.py` and supporting `.py`
files. All of it has been extracted into the `judging/` sub-package
(request #5 — already complete).

| Upstream file                          | New home                                                       | Status                                                                            |
| -------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `judge_test_script.py` (prompt)        | `judging/judge_prompts.py::SAFETY_BEHAVIOR_PROMPT` (verbatim)  | quoted verbatim                                                                   |
| `judge_test_script.py` (runner)        | `judging/judge.py::judge_rows` + `JudgeConfig`                 | ported; adds retries at rising T, robust parse, `aggregate_safety`                |
| `judge_test_script.py` (full)          | `_legacy/cots/judge_test_script.py`                            | **vendored**                                                                      |
| `load_script.py::load_model`           | `models/loading.py::load_model`                                | merged with the SHA loader                                                        |
| `load_script.py` (full)                | `_legacy/cots/load_script.py`                                  | **vendored**                                                                      |
| `beaver_experiments2.py`               | `generation/generate.py` (batched loop) + `data/beaver.py` (loader) | split: generation → `generation`, dataset reading → `data/beaver.py`              |
| `beaver_experiments2.py` (full)        | `_legacy/cots/beaver_experiments2.py`                          | **vendored**                                                                      |
| Original Beaver CSVs (14)              | `data/raw/beavertails/*.csv`                                   | copied; categories enumerated in `data/beaver.py::CATEGORIES`                     |

## Upstream artifacts intentionally not treated as source code

| Upstream artifact                                    | Handling |
| ---------------------------------------------------- | -------- |
| `SafetyHeadAttribution/resource/intro.png`           | image used by upstream README; not copied into source package |
| `SafetyHeadAttribution/.gitignore`, `Readme.md`, `requirements.txt` | documentation / environment metadata; superseded by this repo's README, requirements and docs |
| `SafetyHeadAttribution/exp_data/data_harmful-behaviors.csv` | intermediate preprocessing input referenced by the vendored notebook; canonical raw harmful data is copied under `data/raw/sha/` |
| `CoT-safety/exp_data/Beaver_samples/test.txt`        | placeholder file; not a code path |
| `CoT-safety/exp_data/beaver_results_llama/*.json`    | generated run outputs; new outputs are written under `runs/` |

---

## Files **new** to this repository (no upstream counterpart)

### New attribution methods — `src/safety_cot_heads/attribution/`

* `attribution/coherency.py` — 4 modes (`nll`, `judge_coherence`,
  `pathology`, `hybrid`) for discovering "coherency heads".
* `attribution/quality_heads.py` — counter-experiment: quality-degradation
  attribution on benign prompts.
* `attribution/random_heads.py` — `uniform_random`, `layer_matched`, and
  `activation_magnitude_matched` baselines.
* `attribution/__init__.py` — re-exports the SHIPS/Sahara API from the
  `ships_legacy` / `sahara_legacy` packages alongside the new methods.

### New analysis layer — `src/safety_cot_heads/analysis/`

* `analysis/metrics.py` — Wilson confidence intervals, refusal/harm rates,
  per-condition aggregation.
* `analysis/overlap.py` — set-overlap and Jaccard between selected head
  sets across methods.
* `analysis/dose_response.py` — sweep curves over number-of-heads-ablated.
* `analysis/trajectory.py` — sentence-level flip detection across CoT
  steps.
* `analysis/plots.py` — `head_grid_heatmap`, dose-response plotters.
* `analysis/__init__.py`.

### New judge tooling — `src/safety_cot_heads/judging/`

(Prompt is verbatim upstream; everything else is new.)

* `judging/judge.py` — temperature-rising retry loop, `judge_rows`,
  `JudgeConfig`, `aggregate_safety`.
* `judging/parse.py` — `parse_judge_json` (direct / fence-stripped /
  first-object) with `{parse_status: ok | recovered | failed}` reporting.
* `judging/manual_validation.py` — `sample_for_human_review`,
  `agreement` (Cohen's κ).
* `judging/__init__.py`.

### New interventions — `src/safety_cot_heads/interventions/`

* `interventions/ablation.py` — hook-based ablation using
  `HeadMaskController` (replaces verbatim weight zeroing).
* `interventions/surgery.py` — non-destructive in-place weight surgery
  with `apply_surgery` / `undo_surgery` snapshots.
* `interventions/activation_patching.py` — activation-patching stub for
  causal mediation experiments.
* `interventions/__init__.py`.

### New model layer — `src/safety_cot_heads/models/`

* `models/masks.py` — `HeadMask` config schema, `empty_mask_cfg`,
  `add_head`, `fmt_head_id`, `parse_head_id`.
* `models/custom_llama.py::HeadMaskController` — GQA-aware forward-hook
  controller (replacement for upstream `CustomLlamaAttention`).
* `models/custom_mistral.py` — Mistral analog of the controller.
* `models/loading.py::load_model` — unified loader merging SHA + CoT-safety
  loaders, BnB-4bit support, deterministic seeding.
* `models/__init__.py`.

### New utilities — `src/safety_cot_heads/utils/`

* `utils/device.py` — `select_device`, `SAFETY_COT_DEVICE` env override.
* `utils/seed.py` — `set_global_seed` covering Python / NumPy / torch / CUDA.
* `utils/io.py` — `jsonl_read`, `jsonl_write`, `json_dump`, `json_load`,
  atomic writes.
* `utils/logging.py` — `now_iso`, structured logger factory.
* `utils/__init__.py`.

### New data loaders — `src/safety_cot_heads/data/`

* `data/loaders.py` — generic CSV/JSONL loader.
* `data/beaver.py` — BeaverTails 14-category enumeration and reader.
* `data/benign.py` — benign-prompt pool for quality experiments.
* `data/coherence.py` — coherence-probe pool for coherency head discovery.
* `data/jailbreakbench.py` — JailbreakBench reader.
* `data/maliciousinstruct.py` — MaliciousInstruct reader.
* `data/__init__.py`.
* `data/raw/beavertails/*.csv` — 14 vendored BeaverTails CSVs.
* `data/raw/sha/*.csv` — vendored SHA prompt CSVs.

### New generation layer — `src/safety_cot_heads/generation/`

* `generation/generate.py::generate` — port of upstream `batch_inference`
  plus `condition` label and strict JSONL schema.
* `generation/decoding.py` — deterministic decoding configs (seeded
  sampling, temperature, top-p).
* `generation/prompts.py` — central prompt-rendering for Llama/Mistral
  chat templates.
* `generation/__init__.py`.

### New CLI scripts — `scripts/`

* `scripts/_cli.py` — shared OmegaConf loader, output-dir resolution,
  seeded setup.
* `scripts/run_attribution.py` — run any attribution method by name.
* `scripts/run_generation.py` — run a single ablation condition end-to-end.
* `scripts/run_ablation.py` — orchestrate one experiment yaml.
* `scripts/run_judge.py` — judge a JSONL of generations.
* `scripts/run_analysis.py` — aggregate metrics + plots from a runs/ dir.
* `scripts/make_experiment_matrix.py` — expand
  `experiments/exp03_safety_vs_random_ablation/matrix.yaml`
  into per-condition launch commands.

### New configs — `configs/`

* `configs/models.yaml`, `configs/datasets.yaml`, `configs/experiments/exp02_judge_pipeline/judge.yaml`.
* `configs/experiments/exp05_joint_disentangled_ablation/00-analysis.yaml`
* `configs/experiments/exp01_reproduce_ships_sahara/01-ships-discovery.yaml`
* `configs/experiments/exp01_reproduce_ships_sahara/02-sahara-discovery.yaml`
* `configs/experiments/exp03_safety_vs_random_ablation/03-baseline.yaml`
* `configs/experiments/exp03_safety_vs_random_ablation/04-safety-head-ablation.yaml`
* `configs/experiments/exp03_safety_vs_random_ablation/05-random-head-ablation.yaml`
* `configs/experiments/exp03_safety_vs_random_ablation/06-layer-matched-random.yaml`
* `configs/experiments/exp04_coherency_head_discovery/07-coherency-discovery.yaml`
* `configs/experiments/exp04_coherency_head_discovery/07b-coherency-ablation.yaml`
* `configs/experiments/exp05_joint_disentangled_ablation/08-quality-discovery.yaml`
* `configs/experiments/exp05_joint_disentangled_ablation/08b-quality-ablation.yaml`
* `configs/experiments/exp05_joint_disentangled_ablation/09-safety-minus-coherency.yaml`
* `configs/experiments/exp05_joint_disentangled_ablation/10-overlap-only.yaml`
* `configs/experiments/exp03_safety_vs_random_ablation/matrix.yaml`

### Tests — `tests/`

* `tests/test_masks.py` — masking semantics (scale_mask, mean_mask, GQA
  index mapping).
* `tests/test_head_selection.py` — random baselines and selection helpers.
* `tests/test_json_parsing.py` — judge JSON parser edge cases.
* `tests/test_metrics.py` — Wilson CI boundary cases.
* `tests/__init__.py`.

### Documentation — `docs/`

* `README.md` *(at repo root by request)* — 7-step run guide.
* `docs/MIGRATION.md` — this file.
* `docs/ExperimentTracker.md` — experiment tracker / lab notebook.
* `docs/PREVIOUS_CODE_MAP.md` — informal pre-reorg map.
* `docs/On_the_Role_of_Attention_Heads_LLM_Safety.pdf` — ICLR 2025 paper
  used for the SHIPS/Sahara design.

### Project metadata

* `pyproject.toml` — editable install + dependencies.
* `requirements.txt` — pinned conda/pip set.

---

## Quick verification

```bash
conda activate safety_cot_heads
python -c "from safety_cot_heads.attribution import SHIPS, SaharaConfig, CoherencyConfig, QualityConfig, uniform_random; print('imports OK')"
pytest -q tests/
```
