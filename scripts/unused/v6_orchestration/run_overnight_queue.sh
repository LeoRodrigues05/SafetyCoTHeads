#!/usr/bin/env bash
# =============================================================================
# Overnight GPU queue — starts ONLY after the in-flight safety-reasoning judge
# has fully exited, then runs two independent jobs in priority order.
#
#   JOB A  base-arm prose-prefix monitor      ~2.5-3 h   84 cells / 6,387 rows
#   JOB B  steering dose refinement           ~4-5 h     12 new generation cells
#   FINAL  re-aggregate + regenerate figures + verify paper numbers
#
# Design rules:
#   * Every judge stage is resume-safe (per-cell, id-filtered), so a kill/restart
#     loses at most the in-flight cell. Re-running this script is safe.
#   * Jobs are independent: JOB B still runs if JOB A fails, and the final
#     aggregation still runs if either fails. Nothing is deleted, ever.
#   * JOB B writes only to NEW condition directories (steering_a0.75 /
#     steering_a1.25); it cannot perturb any cell the paper currently cites.
#   * The paper's headline grid stays at the 10 primary conditions. The new
#     doses are additive and are consumed only by the dose-refinement analysis.
#
# Usage (already launched detached; to re-run by hand):
#   bash scripts/unused/v6_orchestration/run_overnight_queue.sh 2>&1 | tee -a runs/direction_a_v6/logs/overnight.out
# =============================================================================
set -uo pipefail          # NOT -e: a failing job must not kill the queue
cd "$(dirname "$0")/../../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
PY="${PY:-.venv/bin/python}"
LOGDIR="runs/direction_a_v6/logs"; mkdir -p "$LOGDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
have_gpu(){ nvidia-smi >/dev/null 2>&1; }

# --- 0. wait for the in-flight SR judge ------------------------------------
WAIT_PID="${WAIT_PID:-10179}"
if kill -0 "$WAIT_PID" 2>/dev/null; then
  log "waiting for in-flight safety-reasoning judge (pid $WAIT_PID) to exit..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
log "in-flight job finished; SR cells now: $(find runs/direction_a_v6/judge -name 'judge_safety_reasoning_trace.jsonl' -not -path '*_pre_p04*' | wc -l)/79"
# let the GPUs settle / VRAM free
sleep 60

# --- JOB A: prose-prefix monitor for the two OLMo-3-Base arms --------------
# Why: these arms were declared explicit-trace in paper_scope.yaml but carry an
# explicit <think> block in 0% of completions, so the judge runner excluded them
# from BOTH the explicit pass and the prose-prefix pass. No monitor labels were
# ever produced -> S and SFS undefined for all 40 of their primary cells. The
# scope file is now corrected; this produces the missing labels.
# Unblocks: 37 primary cells gain SFS (grid 58 -> ~95 defined), the two
# OLMo-3-Base rows of tab:appendix-model-family and tab:appendix-kendall, and
# the base arms in fig:family-model.
log "=== JOB A: prose-prefix monitor, olmo3_7b_base + olmo3_7b_base_own ==="
$PY scripts/judging/run_v6_dual_gpu.py --stage monitor --prose-prefix --gpus 2 \
    --backend hf --batch-size 64 --max-new-tokens 384 \
    --models olmo3_7b_base olmo3_7b_base_own --datasets jbb bt \
    > "$LOGDIR/jobA_base_monitor_${STAMP}.log" 2>&1
log "JOB A exit=$? | prefix-monitor cells now: $(find runs/direction_a_v6/judge -name 'judge_cot_only__prefix.jsonl' -not -path '*_pre_p04*' | wc -l)"

# intermediate aggregation so JOB A's value is banked even if JOB B is cut short
log "re-aggregating after JOB A"
$PY scripts/analysis/aggregate_v6_metrics.py --answer-source v6 --monitor-source v6 \
    --datasets jbb bt --n-boot 2000 >> "$LOGDIR/jobA_base_monitor_${STAMP}.log" 2>&1
log "post-A aggregation exit=$?"

# --- JOB B: steering dose refinement --------------------------------------
# Why: the corrected v6 shows steering's SFS is non-monotonic in dose (Potency
# rises, Coherence Retention collapses, SFS peaks at a=1.0). That rests on a
# 3-point ladder. Adding a0.75 (alpha -6) and a1.25 (alpha -10) makes it a
# 5-point Potency/Quality frontier per arm. Configs were derived mechanically
# from each arm's own a1.0 config (only condition/alpha/output dir differ), so
# the new points are directly comparable.
log "=== JOB B: steering dose refinement (a0.75, a1.25) ==="
$PY scripts/configs/make_dose_refinement_configs.py --write \
    > "$LOGDIR/jobB_doses_${STAMP}.log" 2>&1

GEN_OK=0; GEN_FAIL=0
for model in qwen3_8b llama31_8b_control olmo3_7b_think; do
  for ds in jbb bt; do
    for cond in steering_a0.75 steering_a1.25; do
      cfg="configs/experiments/direction_a_v5_iso_asr/${model}/gen/${ds}/${cond}.yaml"
      out="runs/direction_a_v5/${model}/gen/${ds}/${cond}/seed0"
      if compgen -G "${out}/completions*.jsonl" > /dev/null; then
        log "  [skip] ${model}/${ds}/${cond} already generated"; GEN_OK=$((GEN_OK+1)); continue
      fi
      log "  [gen ] ${model}/${ds}/${cond}"
      $PY -m scripts.generation.run_generation --config "$cfg" \
          >> "$LOGDIR/jobB_doses_${STAMP}.log" 2>&1 \
        && { GEN_OK=$((GEN_OK+1)); log "         ok"; } \
        || { GEN_FAIL=$((GEN_FAIL+1)); log "         FAILED (continuing)"; }
    done
  done
done
log "JOB B generation: $GEN_OK ok, $GEN_FAIL failed"

if [ "$GEN_OK" -gt 0 ]; then
  log "JOB B: parsing new cells into v6"
  $PY scripts/preprocessing/parse_v6_completions.py --models qwen3_8b llama31_8b_control olmo3_7b_think \
      --datasets jbb bt >> "$LOGDIR/jobB_doses_${STAMP}.log" 2>&1
  log "  parse exit=$?"

  # coherence gate must run before the answer judge is meaningful
  log "JOB B: coherence gate on new cells"
  $PY scripts/judging/run_v6_judge_shard.py --stage coherence --gpu 0 --n-shards 1 \
      --models qwen3_8b llama31_8b_control olmo3_7b_think --datasets jbb bt \
      >> "$LOGDIR/jobB_doses_${STAMP}.log" 2>&1
  log "  coherence exit=$?"

  for stage in answer monitor; do
    log "JOB B: $stage judge on new cells"
    $PY scripts/judging/run_v6_dual_gpu.py --stage "$stage" --gpus 2 --backend hf \
        --batch-size 64 --max-new-tokens 384 \
        --models qwen3_8b llama31_8b_control olmo3_7b_think --datasets jbb bt \
        >> "$LOGDIR/jobB_doses_${STAMP}.log" 2>&1
    log "  $stage exit=$?"
  done
  # Llama is prose-only: its monitor labels live in the __prefix file
  log "JOB B: prose-prefix monitor for llama new cells"
  $PY scripts/judging/run_v6_dual_gpu.py --stage monitor --prose-prefix --gpus 2 --backend hf \
      --batch-size 64 --max-new-tokens 384 \
      --models llama31_8b_control --datasets jbb bt \
      >> "$LOGDIR/jobB_doses_${STAMP}.log" 2>&1
  log "  prefix-monitor exit=$?"
fi

# --- FINAL: aggregate, export, refresh figures, verify ---------------------
log "=== FINAL: aggregation + figures + verification ==="
FIN="$LOGDIR/final_${STAMP}.log"
$PY scripts/analysis/aggregate_v6_metrics.py --answer-source v6 --monitor-source v6 \
    --datasets jbb bt --n-boot 2000                     > "$FIN" 2>&1
$PY scripts/analysis/aggregate_v6_reasoning.py --models qwen3_8b olmo3_7b_think \
    --datasets jbb bt                                  >> "$FIN" 2>&1
$PY scripts/analysis/export_v6_composite_cells.py               >> "$FIN" 2>&1
$PY scripts/unused/v5_reports/make_intro_teaser.py                       >> "$FIN" 2>&1
$PY scripts/plotting/make_narrative_figs.py                     >> "$FIN" 2>&1
$PY scripts/plotting/make_composite_insight_reports.py          >> "$FIN" 2>&1
$PY scripts/plotting/plot_pathway_radar.py --model olmo3_7b_think >> "$FIN" 2>&1
for f in composite_03_raw_asr_vs_sfs composite_04_model_family_sfs_heatmap \
         composite_07_pareto_frontier_clean composite_10_dose_response_compact \
         pathway_radar_olmo3_7b_think pathway_radar_overlay_olmo3_7b_think; do
  for e in png pdf; do
    [ -f "runs/plots/$f.$e" ] && cp "runs/plots/$f.$e" papers/ARR_Aug_SafetyIntervention/figures/
  done
done

echo
log "=== PAPER NUMBER VERIFICATION (expect mismatches where new data landed) ==="
$PY scripts/analysis/verify_paper_numbers.py 2>&1 | tail -40

echo
log "=== SUMMARY ==="
$PY - <<'PY'
import json
from statistics import mean
rows = json.load(open('runs/direction_a_v6/reports/cell_metrics.json'))['rows']
FAM = ['steering_a0.5','steering_a1.0','steering_a1.5','steering_ablate',
       'ships_top3','ships_top5','ships_top8',
       'neurons_top256','neurons_top512','neurons_top1024']
ARMS = ['qwen3_8b','llama31_8b_control','olmo3_7b_think','olmo3_7b_base','olmo3_7b_base_own']
prim = [r for r in rows if r['model'] in ARMS and r['condition'] in FAM]
have = sum(1 for r in prim if isinstance(r.get('SFS'), (int, float)))
print(f"primary grid: {have}/{len(prim)} cells with a defined SFS  (was 58/100)")
for m in ARMS:
    sub = [r for r in prim if r['model'] == m]
    v = [r['SFS'] for r in sub if isinstance(r.get('SFS'), (int, float))]
    print(f"  {m:20s} SFS defined {len(v):2d}/{len(sub)}"
          + (f"  mean={mean(v):.3f}" if v else ""))
print()
print("dose ladder (new points marked *):")
for m in ['qwen3_8b','llama31_8b_control','olmo3_7b_think']:
    print(f"  {m}")
    for c in ['steering_a0.5','steering_a0.75','steering_a1.0','steering_a1.25','steering_a1.5']:
        sub = [r for r in rows if r['model'] == m and r['condition'] == c]
        if not sub:
            continue
        f = lambda k: (f"{mean([r[k] for r in sub if isinstance(r.get(k),(int,float))]):.3f}"
                       if any(isinstance(r.get(k), (int, float)) for r in sub) else "  --  ")
        star = '*' if c in ('steering_a0.75','steering_a1.25') else ' '
        print(f"   {star}{c:16s} P={f('P')} Q={f('Q')} S={f('S_v6')} SFS={f('SFS')}")
PY
log "=== QUEUE COMPLETE ==="
