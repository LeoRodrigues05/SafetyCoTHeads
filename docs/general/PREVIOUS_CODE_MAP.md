# Previous Code Map

This document explains what the two upstream codebases did and where the same
functionality lives in the reorganized `safety_cot_heads` project.

Verified upstream snapshots:

- `ydyjya/SafetyHeadAttribution` at `6203a04628abe16c702d7cb4cbe6cfd3daf8f5d0`
- `Lott11/CoT-safety` at `7defa565c99eb5d82db8821828b840b4a356eab7`

All Python source and notebook code from those repos is now present in the
local codebase, either as a verbatim legacy copy or as a ported module.

## SafetyHeadAttribution

`SafetyHeadAttribution` contributed the mechanistic attribution and intervention
side of the project: SHIPS, Sahara, custom attention masking, batch generation,
prompt formatting utilities, a rule-based safety discriminator, and notebooks
showing end-to-end workflows.

| Previous file | What it did | Current location |
| --- | --- | --- |
| `lib/utils/custommodel.py` | Reimplemented Llama attention/model classes so q/k/v attention heads could be masked during forward passes. | Verbatim copy: `safety_cot_heads/src/safety_cot_heads/_legacy/sha/custommodel.py`. Ported behavior: `safety_cot_heads/src/safety_cot_heads/models/custom_llama.py`, `safety_cot_heads/src/safety_cot_heads/models/masks.py`. |
| `lib/utils/get_model.py` | Loaded Hugging Face model/tokenizer objects with upstream assumptions around device placement. | Verbatim copy: `_legacy/sha/get_model.py`. Ported loader: `safety_cot_heads/src/safety_cot_heads/models/loading.py`. |
| `lib/utils/batch_inference.py` | Ran prompt batches through a model and saved generations. | Verbatim copy: `_legacy/sha/batch_inference.py`. Ported generation loop: `safety_cot_heads/src/safety_cot_heads/generation/generate.py`. |
| `lib/utils/load_conv.py` | Loaded conversation templates / chat wrappers. | Verbatim copy: `_legacy/sha/load_conv.py`. Current rendering: `safety_cot_heads/src/safety_cot_heads/generation/prompts.py`. |
| `lib/utils/format.py` | Prompt and output formatting helpers. | Verbatim copy: `_legacy/sha/format.py`. Current rendering and row schemas live in `generation/prompts.py` and `generation/generate.py`. |
| `lib/SHIPS/get_ships.py` | Implemented SHIPS per-head attribution: compare base last-token distribution with masked-head distribution, then rank heads by KL shift. | Verbatim copy: `_legacy/sha/get_ships.py`. Ported API: `safety_cot_heads/src/safety_cot_heads/attribution/ships.py`. |
| `lib/SHIPS/pd_diff.py` | KL divergence helper used by SHIPS. | Verbatim copy: `_legacy/sha/pd_diff.py`; imported directly by `attribution/ships.py`. |
| `lib/SHIPS/ships_utils.py` | Sorted and aggregated SHIPS head-score dictionaries. | Verbatim copy: `_legacy/sha/ships_utils.py`; imported directly by `attribution/ships.py`. |
| `lib/Sahara/attribution.py` | Implemented Sahara greedy attribution by measuring hidden-state subspace shifts after head masking. | Verbatim copy: `_legacy/sha/sahara_attribution.py`. Ported API: `safety_cot_heads/src/safety_cot_heads/attribution/sahara.py`. |
| `lib/Sahara/svd.py` | Principal-angle / SVD similarity metric for Sahara. | Verbatim copy: `_legacy/sha/sahara_svd.py`; imported directly by `attribution/sahara.py`. |
| `lib/Safety_discriminator/Discriminator.py` | Rule-based harmfulness/refusal discriminator using refusal/safety keyword prefixes and short-output checks. | Verbatim copy: `safety_cot_heads/src/safety_cot_heads/_legacy/sha/safety_discriminator.py`. The current preferred evaluation path is the judge stack in `safety_cot_heads/src/safety_cot_heads/judging/` plus metrics in `analysis/metrics.py`. |
| `Generalized_Ships.ipynb` | Notebook workflow for generalized SHIPS runs. | Verbatim copy: `_legacy/sha/notebooks/Generalized_Ships.ipynb`. Production path: `scripts/run_attribution.py` with `configs/experiments/exp01_reproduce_ships_sahara/01-ships-discovery.yaml`. |
| `Ships_quick_start.ipynb` | Minimal notebook entry point for SHIPS usage. | Verbatim copy: `_legacy/sha/notebooks/Ships_quick_start.ipynb`. Production path: `scripts/run_attribution.py`. |
| `Surgery.ipynb` | Notebook workflow for directly modifying model weights / projections. | Verbatim copy: `_legacy/sha/notebooks/Surgery.ipynb`. Ported implementation: `safety_cot_heads/src/safety_cot_heads/interventions/surgery.py`. |
| `exp_data/process_data.ipynb` | Preprocessed harmful-behavior data into experiment CSVs. | Verbatim copy: `_legacy/sha/notebooks/process_data.ipynb`. Current raw data lives under `safety_cot_heads/data/raw/sha/`; loaders live in `safety_cot_heads/src/safety_cot_heads/data/`. |
| `exp_data/MaliciousInstruct.txt` | Text version of MaliciousInstruct prompts. | Copied to `safety_cot_heads/data/raw/sha/MaliciousInstruct.txt`. |
| `exp_data/maliciousinstruct.csv` | Discovery prompts for safety-head attribution. | Copied to `data/raw/sha/maliciousinstruct.csv`; loader: `data/maliciousinstruct.py`. |
| `exp_data/jailbreakbench.csv` | Held-out harmful prompts for evaluation. | Copied to `data/raw/sha/jailbreakbench.csv`; loader: `data/jailbreakbench.py`. |
| `exp_data/harmful_behaviors.csv`, `exp_data/advbench.csv` | Additional harmful prompt sets. | Copied to `data/raw/sha/`; accessible through the generic dataset loader path. |

## CoT-safety

`CoT-safety` contributed the behavioral evaluation side of the project:
BeaverTails category data, a model-loading helper, a generation script for
BeaverTails, and a 5-label LLM-as-judge prompt/runner for safety behavior.

| Previous file | What it did | Current location |
| --- | --- | --- |
| `load_script.py` | Loaded model/tokenizer for generation and judging. | Verbatim copy: `safety_cot_heads/src/safety_cot_heads/_legacy/cots/load_script.py`. Merged loader: `safety_cot_heads/src/safety_cot_heads/models/loading.py`. |
| `judge_test_script.py` | Defined the 5-label safety behavior judge prompt and generated judge outputs. | Verbatim copy: `_legacy/cots/judge_test_script.py`. Prompt: `safety_cot_heads/src/safety_cot_heads/judging/judge_prompts.py`. Runner: `safety_cot_heads/src/safety_cot_heads/judging/judge.py`. JSON recovery: `safety_cot_heads/src/safety_cot_heads/judging/parse.py`. |
| `beaver_experiments2.py` | Iterated over BeaverTails categories, generated model responses, and wrote category output JSON files. | Verbatim copy: `_legacy/cots/beaver_experiments2.py`. Dataset loading: `data/beaver.py`. Generation: `generation/generate.py`. CLI path: `scripts/run_generation.py`. |
| `exp_data/Beaver_samples/*.csv` | 14 category-specific BeaverTails prompt samples. | Copied to `safety_cot_heads/data/raw/beavertails/*.csv`; categories are enumerated in `data/beaver.py::CATEGORIES`. |
| `exp_data/beaver_results_llama/*.json` | Previously generated model outputs from one run. | Not source code. The reorganized project writes fresh outputs to `safety_cot_heads/runs/<experiment>/`. |

## Current Modules Built Around The Imported Code

| Current module | Purpose |
| --- | --- |
| `safety_cot_heads/src/safety_cot_heads/models/custom_llama.py` | Hook-based head masking that replaces the fragile upstream custom forward implementation. |
| `safety_cot_heads/src/safety_cot_heads/models/custom_mistral.py` | Mistral compatibility wrapper for attention-head operations. |
| `safety_cot_heads/src/safety_cot_heads/models/masks.py` | Shared mask configuration helpers and head-id parsing/formatting. |
| `safety_cot_heads/src/safety_cot_heads/attribution/ships.py` | Clean SHIPS API using the new mask controller while preserving upstream KL/sorting helpers. |
| `safety_cot_heads/src/safety_cot_heads/attribution/sahara.py` | Clean Sahara API using the new mask controller while preserving upstream SVD similarity. |
| `safety_cot_heads/src/safety_cot_heads/attribution/coherency.py` | New coherency-head discovery using NLL, judge coherence, pathology, or hybrid scoring. |
| `safety_cot_heads/src/safety_cot_heads/attribution/quality_heads.py` | New benign-quality head attribution for matched degradation controls. |
| `safety_cot_heads/src/safety_cot_heads/attribution/random_heads.py` | Random, layer-matched, and activation-magnitude-matched control head selection. |
| `safety_cot_heads/src/safety_cot_heads/interventions/ablation.py` | Runtime hook-based head ablation without mutating weights. |
| `safety_cot_heads/src/safety_cot_heads/interventions/surgery.py` | GQA-aware destructive weight surgery with undo support. |
| `safety_cot_heads/src/safety_cot_heads/judging/` | Robust judge prompts, generation, parsing, aggregation, and human-validation helpers. |
| `safety_cot_heads/src/safety_cot_heads/analysis/` | Rates, Wilson intervals, dose response, overlap reports, trajectory analysis, and plots. |
| `safety_cot_heads/scripts/` | Config-driven entry points for attribution, generation, ablation, judging, and analysis. |
| `safety_cot_heads/configs/experiments/` | YAML experiment definitions matching the experiment tracker. |

## Non-code Artifacts

The following upstream files were reviewed but are not treated as source code:

- `SafetyHeadAttribution/resource/intro.png`: README image.
- `SafetyHeadAttribution/.gitignore`, `Readme.md`, `requirements.txt`: repo metadata superseded by this project.
- `SafetyHeadAttribution/exp_data/data_harmful-behaviors.csv`: intermediate preprocessing input referenced by the vendored notebook; the usable raw datasets are under `data/raw/sha/`.
- `CoT-safety/exp_data/Beaver_samples/test.txt`: placeholder.
- `CoT-safety/exp_data/beaver_results_llama/*.json`: generated outputs, replaced by the `runs/` layout.
