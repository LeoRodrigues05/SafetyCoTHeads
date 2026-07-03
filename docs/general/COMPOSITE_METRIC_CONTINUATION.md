# Composite-metric work: continuation runbook (GPU machine)

**Purpose.** The composite metric (code + framework doc) is done and validated on the
current grid. What remains needs GPU + gated-model access: regenerating the **stale
steering cells** and re-judging them, then re-reading the cross-model family table.
This file is the exact continuation for a machine with a GPU and Hugging Face access.

Written 2026-07-02. Branch: `judge-validation-batch-v5_002`.

---

## What was already done (no GPU needed, committed)

- **Metric implemented.** [`src/safety_cot_heads/analysis/composite.py`](../../src/safety_cot_heads/analysis/composite.py)
  — baseline-corrected (P, Q, S) axes, geometric-mean `sfs`, `sfs_product`,
  `sfs_covert`, `kendall_tau`, `pareto_front`. Exported from `analysis/__init__.py`.
- **Report script.** [`scripts/make_composite_report.py`](../../scripts/make_composite_report.py)
  — emits per-cell CSV/JSON + a 5-section HTML report (per-model rankings, raw-ASR-vs-SFS
  τ, axis ablation, family table, Pareto front).
- **Hardening.** A steering config missing `mode` now raises instead of silently
  defaulting to `ablate` (which discards the α dose). See
  [`interventions/steering.py`](../../src/safety_cot_heads/interventions/steering.py)
  `build_steering_cfg_from_file` and [`scripts/run_generation.py`](../../scripts/run_generation.py).
- **Framework doc.** [`EVALUATION_FRAMEWORK.md`](EVALUATION_FRAMEWORK.md) §4/§5/§8
  rewritten from "undecided candidates" to the committed metric.
- **Stale cells archived** to `runs/direction_a_v5/_stale_steering_pre_dose_fix/`
  (the 12 qwen/llama `steering_a{0.5,1.0,1.5}` gen+judge cells that ran as directional
  ablation). **Do not delete** — they are the "before" evidence for the bug writeup.
- **qwen3_8b direction extracted** →
  `runs/direction_a_v5/qwen3_8b/17-direction-extraction/refusal_directions.npz`.
- **qwen3_8b steering generation** was started on the origin machine and may be
  partially complete; the `complete_v5_*` scripts are idempotent (they skip finished
  cells), so just re-run them — they finish whatever is missing.

## The bug being fixed (context)

The qwen3_8b and llama31_8b_control `steering_a{0.5,1.0,1.5}` cells gave **identical**
results across α because their on-disk runs were generated *before* commit `d69fedc`
switched the steering config from `mode: ablate` to `mode: add`. They ran directional
ablation (α ignored, all layers) instead of dosed activation-addition at layer 14. The
YAML configs are already correct; the runs were just never regenerated. OLMo models
were regenerated after the fix and vary correctly (`think`: 0.11 → 0.42 → 0.83).

Root cause also revealed: **neither qwen nor llama had a `refusal_directions.npz`** at
all (the originals were deleted), so regeneration must run direction extraction first.

---

## Prerequisites on the GPU machine

```bash
cd /path/to/SafetyCoTHeads
source .venv/bin/activate            # torch 2.11 + cu128, omegaconf, vllm, etc.
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# Llama-3.1-8B-Instruct is a GATED repo — this is what blocked the origin machine.
export HF_TOKEN=hf_xxx               # a token with meta-llama access
huggingface-cli whoami               # confirm auth
```

## Step 1 — Extract the llama refusal direction

(qwen's already exists; skip qwen extraction unless the npz is missing.)

```bash
python -m scripts.run_direction_extraction \
  --config configs/experiments/direction_a_v5_iso_asr/llama31_8b_control/17-direction-extraction.yaml
# writes runs/direction_a/17-direction-extraction-llama31/refusal_directions.npz
# (this is the exact path the llama steering configs already reference)
```

> The llama extraction config was created during this work (adapted from qwen's:
> maliciousinstruct/alpaca, n=100 each, output to the path the steering YAMLs expect).

## Step 2 — Regenerate steering generation (idempotent)

```bash
python -m scripts.complete_v5_generation qwen3_8b            --datasets jbb bt
python -m scripts.complete_v5_generation llama31_8b_control  --datasets jbb bt
```

This regenerates the 6 archived `steering_a*` cells per model (plus the pre-existing
missing `steering_ablate` cells, which it will also fill — harmless, completes the
grid). ~20 min/cell for qwen (thinking model, 1024 new tokens); llama is faster.

## Step 3 — Verify the dose fix propagated (critical gate)

```bash
python - <<'PY'
import json
for m in ["qwen3_8b","llama31_8b_control"]:
    for a in ["0.5","1.0","1.5"]:
        p=f"runs/direction_a_v5/{m}/gen/jbb/steering_a{a}/seed0/completions_steering_a{a}.jsonl"
        r=json.loads(open(p).readline()); meta=r.get("meta",r)
        print(m, f"a{a}", meta.get("steering_mode"), meta.get("steering_alpha"),
              meta.get("steering_layers"))
PY
```

Expect `add`, `alpha ∈ {-4,-8,-12}`, `layers [14]` (NOT `ablate / 1.0 / [0..N]`).
If it still says `ablate`, the wrong config was used — stop and check the YAML.

## Step 4 — Re-judge the regenerated cells

```bash
# Standard 5-label + pathway (14B) judge:
python -m scripts.complete_v5_judging qwen3_8b            --datasets jbb bt
python -m scripts.complete_v5_judging llama31_8b_control  --datasets jbb bt

# Safety-reasoning trace judge (vLLM backend is much faster for the SR pass):
python -m scripts.run_v5_safety_reasoning \
  --models qwen3_8b llama31_8b_control --datasets jbb bt \
  --conditions steering_a0.5 steering_a1.0 steering_a1.5 steering_ablate \
  --backend vllm
```

## Step 5 — Reaggregate summaries

```bash
python -m scripts.reaggregate_v5_summaries          # rewrites summary.json per cell
```

## Step 6 — Regenerate the composite report on the clean grid

```bash
python -m scripts.make_composite_report \
  --out       runs/direction_a_v5/composite_report.html \
  --csv-out   runs/direction_a_v5/composite_cells.csv \
  --json-out  runs/direction_a_v5/composite_cells.json
```

**Acceptance checks** (numbers that must hold, independent of the steering fix — they
validate the metric didn't regress):
- `olmo3_7b_base / jbb / neurons_top512`: `P ≈ 0.00`, `raw_hac ≈ 0.65`
- `llama31_8b_control / jbb / ships_top8`: `Q ≈ 0.46`
- The steering dose-response for qwen/llama must now **vary** across α (finding: it was
  flat before). Section 1 of the report should show steering climbing with α.

## Step 7 — Finalise the framework doc family table

Open the regenerated `composite_report.html` §4 (family table) and reconcile
`EVALUATION_FRAMEWORK.md` §5.5's data-status note with the now-clean numbers. The
prototype (pre-fix) family ranking was Steering ≫ heads/neurons/ablation on SFS; the
narrative should survive but the exact steering P/SFS will change. Replace the
"pending re-judge" caveat with the final table once verified.

---

## Still out of scope (separate future tasks)

- **H1–H5 pre-registered statistics** (χ², McNemar, paired bootstrap, BH-FDR): only
  McNemar + Wilson CI exist (`analysis/paired.py`, `analysis/metrics.py`); no bootstrap
  or BH-FDR yet. The pre-reg doc `docs/direction_a/prereg_v4.md` is *referenced but
  missing* and should be restored/recreated. Rewrite H3 to match observed `gap ≤ 0`.
- **Benign-utility retention** for the Quality axis (MMLU/GSM8K/AlpacaEval delta): no
  utility eval exists in the grid; Q is coherence-only for now.
- **Paper §7 prose draft** from the four dissociation findings.
