# B200 CoT-trace judging bundle — Llama & Qwen

Data payload for running the **CoT-trace metrics** on the B200 VM for
`llama31_8b_control` and `qwen3_8b`:

| Metric | Output |
|--------|--------|
| **C** — CoT-only monitor | `asr_cot_pred` |
| **D** — Monitorability gap | `gap = asr_final − asr_cot_pred` |
| **E** — Pathway (12 labels) | `pathway_vectors.jsonl` |

All three read the **same single input: the generation completions** — that's
everything in this bundle. Code + configs are NOT here; they come via git.

## What's inside
```
runs/direction_a_v5/llama31_8b_control/gen/   20 completion files (jbb=100, bt=98 rows/cell)
runs/direction_a_v5/qwen3_8b/gen/             20 completion files
```
40 files, ~34 MB. See `MANIFEST.txt` for per-file line counts.

## Prerequisites on the VM
1. **Code + configs via git** — in the repo checkout:
   ```bash
   cd <repo>/safety_cot_heads && git pull origin main   # need >= commit 7f51162
   ```
2. **Judge model** — `Qwen/Qwen3-30B-A3B-Instruct-2507` is already cached on the
   B200 from the OLMo bf16 run. No download needed.

## Install the data
```bash
rsync -a b200_cot_judging_bundle/runs/  <repo>/safety_cot_heads/runs/
```

## Run the CoT-trace judging (bf16, full-N)
```bash
cd <repo>/safety_cot_heads

# IMPORTANT: run_local_pipeline is resume-safe — it SKIPS existing per-label
# files. If a 4-bit judge/ tree is present it must be moved aside first, or the
# bf16 run will silently skip those cells.
for M in qwen3_8b llama31_8b_control; do
  [ -d runs/direction_a_v5/$M/judge ] && \
    mv runs/direction_a_v5/$M/judge runs/direction_a_v5/$M/judge_4bit_$(date +%s)
done

# bf16 is the default (load_in_4bit=false); SKIP_PATHWAY=0 turns on metric E.
SKIP_PATHWAY=0 bash scripts/run_local_pipeline.sh qwen3_8b          judge
SKIP_PATHWAY=0 bash scripts/run_local_pipeline.sh llama31_8b_control judge
```
Notes:
- One judge run produces **A (safety) + B (coherence) + C (cot-only) + D
  (monitorability) + E (pathway)**. D is `asr_final − asr_cot_pred`, so the
  safety pass (A) is computed too — that's required, not waste.
- **Pathway (E) is the long pole** (12 labels × prefix expansion). To get C/D
  fast first, run with the default `SKIP_PATHWAY=1`, then re-run with
  `SKIP_PATHWAY=0` for E (resume-safe, so it only adds the pathway files).

## Later: safety-variable agreement test
These same completions also feed the n=25 bf16 vs 4-bit agreement check via
`run_v5_qwen_subset_judging` (writes to a separate `judge_subset_n25_bf16`
out-base — no conflict with the full-N `judge/` tree).

Source repo: github.com/LeoRodrigues05/SafetyCoTHeads @ main (7f51162)
