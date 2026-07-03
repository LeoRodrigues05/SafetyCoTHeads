# Run this project on a fresh machine — LLM/agent handoff runbook

> **If you are an LLM/coding agent: START HERE.** Read this file top-to-bottom,
> then read the two linked docs before running anything:
> [`MULTI_GPU_EXECUTION_PLAN.md`](MULTI_GPU_EXECUTION_PLAN.md) (per-GPU schedule +
> commands) and [`COMPOSITE_METRIC_CONTINUATION.md`](COMPOSITE_METRIC_CONTINUATION.md)
> (why the steering cells are being regenerated). Repo overview:
> [`../../README.md`](../../README.md). **Everything is resume-safe** — when in
> doubt, run the audit (Step 3) and let the fill-in commands do only what's missing.

**Target machine:** fresh instance with **2 × GPU** (built/validated on B200 /
Blackwell sm_100; ≥~80 GB VRAM/card is plenty — the biggest model is the 30B judge
in NF4). **Goal of this run:** finish the Direction A v5 grid (the Llama steering
cells + any missing pathway cells) and add one new thinking model
(**`r1_distill_qwen_7b`** = DeepSeek-R1-Distill-Qwen-7B), then regenerate the
composite report.

---

## Step 0 — Bring the data over

> **FAST PATH — filesystem preserved (this project's case).** If the new machine
> mounts the **same filesystem** (the repo lives on the persisted `/work` mount),
> then `runs/`, `models/`, the working-tree config edits, `.venv`, and the HF cache
> (`~/.cache/huggingface`, if `$HOME` persists too) are **already present** —
> **skip the rsync below entirely.** Just: confirm 2 GPUs are visible, verify the
> artifacts exist (Step 2), re-run `hf auth login` only if `hf auth whoami` says
> "Not logged in", and jump to **Step 3**. The rest of Step 0 is only for a
> genuinely fresh disk.

For a **fresh disk** (no shared mount): a plain `git clone` gives you the **code
only**. These are **gitignored** and must be copied from the origin machine or you
will redo ~40 GPU-hours and lose the fine-tuned judge:

| Path | Size | Why it's essential |
|---|---|---|
| `runs/direction_a_v5/` | ~4.4 GB | the already-completed grid (completions + all judge outputs for qwen, both olmo-base variants, olmo-think, 14/22 llama). Losing it = regenerate everything. |
| `models/pathway_judge_14b_merged/` | ~28 GB | the fine-tuned 14B pathway judge (κ≈0.96). Retraining is hours + needs HarmThoughts data. |
| `runs/direction_a/` (if present) | small | Llama SHIPS/neuron discovery artifacts (only needed if redoing llama head/neuron cells — not for the steering work). |

Also **working-tree config edits are uncommitted** (the `r1_distill_qwen_7b`
entries in `configs/models.yaml` + `matrix.yaml`, and these docs). So the safest
transfer is **rsync the whole working tree** (preserves edits + runs + models),
excluding the venv and caches, then rebuild the venv on the new box:

```bash
# Run from the ORIGIN machine (adjust host/path). ~33 GB, a few minutes on a fast link.
rsync -avP \
  --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
  /work/Work/SafetyCoTHeads/  USER@NEWHOST:/path/to/SafetyCoTHeads/
```

> **Optional (skip ~90 GB of re-downloads):** also copy the HF weight cache:
> `rsync -avP ~/.cache/huggingface/  USER@NEWHOST:~/.cache/huggingface/`
> (contains Qwen3-30B judge, Llama-3.1-8B, R1-Distill-Qwen-7B once pre-warmed).
> If you skip this, Step 1 re-downloads them — fine, just slower.

> **Alternative to rsync-ing the tree:** on the origin, `git add -A && git commit`
> the config/doc changes and push; then `git clone` on the new box **and still**
> rsync `runs/direction_a_v5/` + `models/pathway_judge_14b_merged/` (they can never
> be in git). The rsync-the-tree path above is simpler.

---

## Step 1 — Environment + auth

> If the filesystem was preserved and `.venv` already works
> (`.venv/bin/python -c "import torch;print(torch.cuda.is_available())"` → `True`),
> **skip `setup_vm.sh`** — just re-auth if needed. Run it only on a fresh disk or if
> the venv is broken.

```bash
cd /path/to/SafetyCoTHeads
bash scripts/setup_vm.sh            # builds .venv, Blackwell torch (cu128), installs pkg, smoke
#   SKIP_SMOKE=1 bash scripts/setup_vm.sh   # to skip the model-loading smoke test
source .venv/bin/activate

# HF auth — needed for gated Llama-3.1; harmless for the open models.
hf auth login --token hf_XXXX       # writes ~/.cache/huggingface/token (all procs read it)
hf auth whoami                      # expect your username
#   Also click "Agree and access repository" once on:
#   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

Per-shell env each lane wants:
```bash
export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Step 2 — Sanity checks (fail fast)

```bash
# 2 GPUs visible + torch/cuda
python -c "import torch;print('cuda',torch.cuda.is_available(),'devs',torch.cuda.device_count())"
# expect: cuda True devs 2

# Fine-tuned judge present (must be ~28 GB, self-contained)
ls models/pathway_judge_14b_merged/config.json models/pathway_judge_14b_merged/tokenizer.json

# Gated Llama access
python - <<'PY'
from huggingface_hub import model_info
print("llama gated ok:", bool(model_info("meta-llama/Llama-3.1-8B-Instruct")))
PY

# New-model configs exist (regenerate if the rsync missed them)
ls configs/experiments/direction_a_v5_iso_asr/r1_distill_qwen_7b/gen/*/*.yaml | wc -l   # expect 22
# if 0: python -m scripts.make_v5_configs --matrix configs/experiments/direction_a_v5_iso_asr/matrix.yaml

# Pre-warm shared cache so 2 lanes don't double-download:
hf download Qwen/Qwen3-30B-A3B-Instruct-2507
hf download meta-llama/Llama-3.1-8B-Instruct
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

---

## Step 3 — Audit what's actually missing (source of truth)

Run this; it prints exactly which (model, dataset, condition) cells lack each judge
layer. Do the work only for what it flags (all commands below are resume-safe).

```bash
python - <<'PY'
import os, glob
base="runs/direction_a_v5"
layers={"summary":"summary.json","coherence":"coherence.jsonl","judged5":"judged_*.jsonl",
        "cot_only":"judge_cot_only.jsonl","monitor":"monitorability_rows.jsonl",
        "pathway":"judge_pathway.jsonl","sr_trace":"judge_safety_reasoning_trace.jsonl"}
for m in sorted(os.listdir(base)):
    md=f"{base}/{m}"
    if not os.path.isdir(md) or m.startswith("_"): continue
    cells=sorted(glob.glob(f"{md}/judge/*/*/seed0"))
    for c in cells:
        miss=[k for k,pat in layers.items() if not glob.glob(f"{c}/{pat}")]
        if miss: print(m, c.split('/judge/')[1], "MISSING:", ",".join(miss))
    gen=len(glob.glob(f"{md}/gen/*/*/seed0/completions_*.jsonl"))
    print(f"# {m}: gen_cells={gen}/22, judge_cells={len(cells)}/22")
PY
```

**Expected state at handoff** (verify against the above):
- `qwen3_8b`, `olmo3_7b_think`: complete (the pathway re-judge of qwen's 8 steering
  cells + olmo-think `jbb/steering_ablate` was running on the origin — if the audit
  still shows those `MISSING: pathway`, re-run the pathway pass, see 3A).
- `olmo3_7b_base`, `olmo3_7b_base_own`: complete (22/22).
- `llama31_8b_control`: **14/22** — missing all 8 steering cells (do 3B).
- `r1_distill_qwen_7b`: 0/22 — full pipeline (do 3C).

### 3A — Fill any missing pathway cells (token-free; GPU 0)
Only if the audit flags `MISSING: pathway`. For each flagged `<dset>/<cond>`:
```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -m scripts.run_v4_jbb_judge \
  --config configs/experiments/direction_a_v5_iso_asr/<MODEL>/judge_14b.yaml \
  --out-base runs/direction_a_v5/<MODEL>/judge \
  --skip-safety --skip-coherence --skip-cot-only \
  --condition tag=<dset>/<cond>,cond=<cond>,completions=runs/direction_a_v5/<MODEL>/gen/<dset>/<cond>/seed0/completions_<cond>.jsonl
```
(~30 min/cell for qwen-scale thinking cells.)

### 3B — Llama steering lane (needs HF token; GPU 0)
```bash
# L1: extract the refusal direction (writes runs/direction_a/17-direction-extraction-llama31/refusal_directions.npz)
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_direction_extraction \
  --config configs/experiments/direction_a_v5_iso_asr/llama31_8b_control/17-direction-extraction.yaml

# L2: generate the 8 steering cells (steering_a{0.5,1.0,1.5} + steering_ablate x jbb,bt)
CUDA_VISIBLE_DEVICES=0 python -m scripts.complete_v5_generation llama31_8b_control --datasets jbb bt

# GATE — verify the dose actually applied (must be add / alpha∈{-4,-8,-12} / layers [14]):
python - <<'PY'
import json
for a in ["0.5","1.0","1.5"]:
    p=f"runs/direction_a_v5/llama31_8b_control/gen/jbb/steering_a{a}/seed0/completions_steering_a{a}.jsonl"
    m=json.loads(open(p).readline()).get("meta",{})
    print("a"+a, m.get("steering_mode"), m.get("steering_alpha"), m.get("steering_layers"))
PY

# L3a: standard judge (30B) for the 8 new cells
CUDA_VISIBLE_DEVICES=0 python -m scripts.complete_v5_judging llama31_8b_control --datasets jbb bt --skip-report
# L3b: pathway judge (14B) for the 8 new cells
CUDA_VISIBLE_DEVICES=0 PATHWAY_ONLY=1 JUDGE_CONFIG=judge_14b.yaml JUDGE_BATCH=128 \
  bash scripts/run_local_pipeline.sh llama31_8b_control judge
# L3c: SR-trace (vLLM) for the 8 new cells
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_v5_safety_reasoning \
  --models llama31_8b_control --datasets jbb bt \
  --conditions steering_a0.5 steering_a1.0 steering_a1.5 steering_ablate --backend vllm
```

### 3C — New model full pipeline `r1_distill_qwen_7b` (GPU 1, in parallel with 3B)
```bash
export CUDA_VISIBLE_DEVICES=1
python -m scripts.run_local_pipeline.sh r1_distill_qwen_7b discover   # SHIPS + neurons + direction (~1–3h)
python -m scripts.run_local_pipeline.sh r1_distill_qwen_7b gen        # 22 cells (~8–14h)
python -m scripts.run_local_pipeline.sh r1_distill_qwen_7b judge      # standard 30B (SKIP_PATHWAY=1 default)
PATHWAY_ONLY=1 JUDGE_CONFIG=judge_14b.yaml JUDGE_BATCH=128 \
  python -m scripts.run_local_pipeline.sh r1_distill_qwen_7b judge    # pathway 14B (~12–13h)
python -m scripts.run_v5_safety_reasoning --models r1_distill_qwen_7b --datasets jbb bt --backend vllm
```

> **2-GPU parallelism:** run 3B on GPU 0 and 3C on GPU 1 concurrently (they're
> independent). Within 3C, generation is the long pole — you can shard its 22 cells
> across both cards once 3B finishes (see `fan_gen.sh` in the plan doc §4). Full
> per-card schedule: [`MULTI_GPU_EXECUTION_PLAN.md`](MULTI_GPU_EXECUTION_PLAN.md) §5a.
> **2-GPU wall-clock estimate: ~22–26 h.**

---

## Step 4 — Finalise + acceptance gates

```bash
python -m scripts.reaggregate_v5_summaries
python -m scripts.make_composite_report \
  --out runs/direction_a_v5/composite_report.html \
  --csv-out runs/direction_a_v5/composite_cells.csv \
  --json-out runs/direction_a_v5/composite_cells.json
.venv/bin/python -m scripts.make_v5_plots
```

Must hold (else something regressed):
- Llama steering now **varies** across α (was flat pre-fix); the L2 gate above passed.
- `olmo3_7b_base/jbb/neurons_top512`: `P≈0.00`, `raw_hac≈0.65`.
- `llama31_8b_control/jbb/ships_top8`: `Q≈0.46`.
- `composite_cells.csv` has full 22-cell blocks for **both** `llama31_8b_control`
  and `r1_distill_qwen_7b`.

---

## Monitoring & resume
- `tail -f logs/*.log`; GPUs: `watch -n5 nvidia-smi`.
- Any killed stage: just re-run its command — finished cells/rows are skipped.
- Re-run the Step 3 audit anytime to see remaining work.
