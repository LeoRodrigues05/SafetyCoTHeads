# SafetyCoTHeads — evaluating white-box safety interventions

Code, configs and results for the paper **"Beyond Refusal Rates: Evaluating Safety
Mechanisms in Language Models"** (ARR submission; LaTeX source in
`docs/paper/ARR_Aug_SafetyIntervention.zip`).

## What the project does

Papers that locate a "safety mechanism" inside an LLM (safety heads, safety neurons, a
refusal direction) usually support the claim with a single attack-success rate (ASR). The
ASR is measured on their own data with their own success criterion. This project puts
four such interventions through **one controlled grid** and scores each one with a
decomposable metric:

| Axis | Question | Definition (per cell, against the same model's no-intervention baseline) |
|---|---|---|
| **Potency P** | Did it induce harm beyond baseline? | `clip((HAC_c − HAC_b) / (1 − HAC_b))`; HAC = harmful answers among coherent answers |
| **Coherence retention Q** | Did it keep the model working? | `clip(clean_c / clean_b)`; clean = passes the coherence gate |
| **Monitorability retention S** | Is induced harm still visible in the trace? | `1 − clip(U_c − U_b)`; U = P(answer harmful ∧ trace-only monitor says safe) |
| **SFS** | Summary | `(P·Q·S)^(1/3)`; the vector is the primary report |

The four intervention families (all inference-time hooks, no weight changes):

| Family | Target discovery | Intervention |
|---|---|---|
| SHIPS heads | Per-head next-token KL on harmful prompts (SHIPS) | Scale the head's query by 1e-4 (top-k = 3/5/8) |
| Neurons | \|mean harmful − mean benign\| MLP activation, last prompt token | Zero top-k = 256/512/1024 MLP neurons |
| Steering | Harmful − benign mean residual at layer 14 | Add `−8·d · unit(r)` at layer 14, d = 0.5/1.0/1.5 |
| Directional ablation | Same direction | Project it out (legacy: layer inputs only; v6.1: every residual write) |

Models: Qwen3-8B (thinking), OLMo-3-7B-Think, Llama-3.1-8B-Instruct, and OLMo-3-7B-Base
(twice: with targets transferred from OLMo-Think, and with targets discovered on
Base). Data: JailbreakBench (100) and BeaverTails (98). Discovery uses MaliciousInstruct
vs Alpaca. The full experimental record is in **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**; risks and
assumptions are in **[docs/RISKS_AND_ASSUMPTIONS.md](docs/RISKS_AND_ASSUMPTIONS.md)**. The pending
correction run (v6.1) is specified in **[RERUN_V61.md](RERUN_V61.md)**.

> **Status (2026-09-24).** The reported numbers come from the v6 evaluation
> (`runs/direction_a_v6/reports/cell_metrics.json`, 2026-07-26). Several correctness
> fixes (the v6.1 set) change some of them; the rerun is pending. Per-claim status for
> the paper is tracked in `docs/paper/PAPER_CLAIMS_AUDIT.md`.

## Pipeline

```
discovery ──► generation ──► parse ──► judge ──► aggregate ──► figures / tables
(heads,        (greedy, per   (answer /  (answer,   (P,Q,S,SFS,
 neurons,       model×dataset  trace      coherence, paired bootstrap)
 direction)     ×condition)    split)     monitor,
                                          pathway, SR)
```

1. **Discovery** (`scripts/discovery/`): rank heads (SHIPS), neurons, and per-layer refusal
   directions on MaliciousInstruct vs Alpaca.
2. **Generation** (`scripts/generation/`): one completion per prompt per cell, greedy, seed 0.
   Outputs go to `runs/direction_a_v5/<model>/gen/<dataset>/<condition>/seed0/`; this tree is
   treated as immutable.
3. **Parse** (`scripts/preprocessing/parse_v6_completions.py`): split each completion into the
   final answer and the reasoning trace (explicit `<think>` or, for non-reasoning models, an
   early-response prefix used only as a sensitivity signal).
4. **Judge** (`scripts/judging/`): Qwen3-30B-A3B-Instruct-2507 judges answer harmfulness,
   predicts harm from the trace alone, and marks safety-reasoning sentences; the coherence
   gate is rule + classifier based. The fine-tuned 14B pathway judge labels reasoning stages.
5. **Aggregate** (`scripts/analysis/`): per-cell P, Q, S, SFS with prompt-paired bootstrap CIs
   → `runs/direction_a_v6/reports/`.
6. **Figures** (`scripts/plotting/`) → `figures/`.

## Repository layout

```
configs/
  models.yaml, datasets.yaml
  direction_a_v6/paper_scope.yaml         # which models are primary / explicit-trace / prose-prefix
  experiments/direction_a_v5_iso_asr/      # matrix.yaml + one YAML per generation cell
src/safety_cot_heads/                      # library: models (hooks), interventions, attribution,
                                           # generation, judging, direction_a_v6 (parser, metrics)
scripts/
  common/        _bootstrap.py (import paths), v6_common.py (IO), _cli.py, _paper_figstyle.py
  configs/       generate experiment configs (grid, controls, dose refinement)
  discovery/     head / neuron / direction discovery
  generation/    run_generation.py, run_generation_batch.py, continue_truncated_generations.py
  preprocessing/ audit, parse, invalidate stale judge rows, completeness, pathway quarantine
  judging/       run_v6_dual_gpu.py (work-queue across GPUs), run_v6_judge_shard.py (+ coherence)
  analysis/      aggregate metrics / reasoning, export tables, verify paper numbers
  plotting/      every paper and diagnostic figure
  validation/    annotation tool + scoring; validate_v6_matched.py (v6 human validation)
  training/      HarmThoughts data prep, 14B pathway-judge LoRA training and evaluation
  release/       manifest + Hugging Face export staging
  orchestration/ run_v61_queue.sh — the end-to-end, resumable rerun
  unused/        superseded v4/v5/v6 drivers and reports, kept for provenance
figures/
  paper/         figures used by (or regenerated for) the paper
  diagnostics/   v5 analysis plots, v6 correction diagnostics
  validation/    human-annotation plots per batch
  legacy/        early BeaverTails dose reports
data/annotations/batch_v5_002/             # committed two-annotator validation batch
docs/            EXPERIMENTS.md, RISKS_AND_ASSUMPTIONS.md, paper/ (source zip + claims audit)
runs/            all outputs (gitignored; shared storage)
tests/           pytest suite (CPU only)
```

Every script can be run from the repo root as `python scripts/<group>/<script>.py`. A
script finds `src/` and its sibling scripts through `scripts/common/_bootstrap.py`, so no
`PYTHONPATH` setup is needed. The `unused/` scripts are not part of the current pipeline;
see *Unused scripts* below.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt
PYTHONPATH=src pytest -q tests/          # CPU-only; the integration test writes to a temp dir
huggingface-cli login                    # Llama-3.1-8B-Instruct is gated
```

The judging/generation runs assume B200-class GPUs (see `RERUN_V61.md` for timings).

## Running

| Task | Command |
|---|---|
| Regenerate grid configs | `python scripts/configs/make_v5_configs.py --matrix configs/experiments/direction_a_v5_iso_asr/matrix.yaml` |
| Discovery for a model | `python scripts/discovery/run_attribution.py --config <model>/01-ships-discovery.yaml` (likewise `run_neuron_discovery.py`, `run_direction_extraction.py`) |
| Generate one cell | `python scripts/generation/run_generation.py --config configs/experiments/direction_a_v5_iso_asr/<model>/gen/<ds>/<cond>.yaml` |
| Generate many cells of one model | `python scripts/generation/run_generation_batch.py --configs <yaml> ...` |
| Parse | `python scripts/preprocessing/parse_v6_completions.py --models ... --datasets jbb bt` |
| Judge a stage | `python scripts/judging/run_v6_dual_gpu.py --stage {answer,monitor,pathway14b,safety-reasoning} --gpus 2 ...` |
| Coherence gate | `python scripts/judging/run_v6_judge_shard.py --stage coherence --gpu 0 --n-shards 1` |
| Aggregate | `python scripts/analysis/aggregate_v6_metrics.py --answer-source v6 --monitor-source v6 --datasets jbb bt --models <primary arms>` |
| Figures | `python scripts/plotting/make_composite_insight_reports.py`, `make_narrative_figs.py`, `plot_pathway_radar.py` |
| Human validation (v6) | `python scripts/validation/validate_v6_matched.py --out runs/direction_a_v6/validation` |
| Full v6.1 rerun | `NGPU=2 bash scripts/orchestration/run_v61_queue.sh` (see `RERUN_V61.md`) |

All judge stages resume by (prompt id, input hash). Re-running a stage only judges rows that
are new or whose input text changed.

## Outputs

| Path | Content |
|---|---|
| `runs/direction_a_v5/<model>/gen/...` | generations (immutable source) |
| `runs/direction_a_v6/gen_continued/...` | cells whose truncated traces were continued (supersede v5 files when present) |
| `runs/direction_a_v6/parsed/...` | canonical answer/trace split per cell (+ `parse_changes.json`) |
| `runs/direction_a_v6/judge/<model>/<ds>/<cond>/seed0/` | `judge_answer_safety`, `coherence_answer`, `judge_cot_only[__prefix]`, `judge_pathway` (+ per-label files), `judge_safety_reasoning_trace` |
| `runs/direction_a_v6/reports/` | `cell_metrics.json` (all axes, CIs, denominators), `composite_cells.csv`, `control_cells.csv`, `reasoning_metrics.*` |
| `runs/direction_a_v6/validation/` | human-validation reproductions and `validation_v6_matched.*` |
| `figures/` | all plots |

## Unused scripts

`scripts/unused/` holds drivers that produced earlier (v4/v5) results or orchestrated past
runs. Nothing in the current pipeline calls them. Outputs of three of them do still appear in
the paper, as supplementary or provenance material:

- `v5_judging/run_v4_jbb_judge.py` (despite its name, the v5 judge driver) produced the v5
  answer/monitor labels used to build and score the human-validation batch. The v5 κ values
  are reported as supplementary material only.
- `v5_judging/run_v5_safety_reasoning.py` produced the v5 safety-reasoning labels in the same
  batch (supplementary v5 SR κ).
- `v5_pipeline/run_local_pipeline.sh`, `run_3gpu_all.sh`, `complete_v5_generation.py` and
  `sbatch/` drove the discovery and generation runs whose **outputs the current results reuse**.
  They call the active `scripts/discovery/` and `scripts/generation/` code, so only the drivers
  are retired.

## Provenance

Built on [`ydyjya/SafetyHeadAttribution`](https://github.com/ydyjya/SafetyHeadAttribution)
(SHIPS) and [`Lott11/CoT-safety`](https://github.com/Lott11/CoT-safety) (5-label judge prompt).
Upstream code is preserved verbatim in `src/safety_cot_heads/_legacy/`.
