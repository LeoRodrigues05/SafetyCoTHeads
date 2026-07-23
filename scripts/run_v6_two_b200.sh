#!/usr/bin/env bash
# Direction A v6 corrected rerun — two-B200 orchestrator.
#
# One independent judge process per GPU (CUDA_VISIBLE_DEVICES=0/1), no tensor
# parallelism. Deterministic cell sharding (blake2b(cell_key) % 2). CPU stages
# (audit, parse, coherence, aggregate, validation) run without a GPU.
#
# Source data under runs/direction_a_v5 is IMMUTABLE: this script only ever
# writes under runs/direction_a_v6. It never regenerates model completions
# unless --allow-generation-repair is passed AND the audit repair manifest is
# non-empty.
#
# Usage:
#   bash scripts/run_v6_two_b200.sh <stage> [options]
# Stages:
#   audit smoke parse answer monitor pathway safety-reasoning aggregate
#   validation check all
#
# Options:
#   --allow-generation-repair   permit restricted regeneration of repair cells
#   --n-boot N                  bootstrap replicates for aggregate (default 10000)
#   --models "a b"  --datasets "jbb bt"
#   --answer-source v5|v6       aggregation input (default: v6 if judge exists)
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
PY="${PY:-.venv/bin/python}"
LOGDIR="runs/direction_a_v6/logs"
mkdir -p "$LOGDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# Judge backend. Default 'hf': loads the full bf16 30B judge into each B200's
# 183 GB and runs Transformers generation — no vLLM engine needed. Set
# V6_BACKEND=vllm to use continuous batching instead (faster, but slow engine
# init on this box).
BACKEND="${V6_BACKEND:-hf}"

# vLLM 0.23 + CUDA-13 environment for this Blackwell (B200) box — mirrors the
# proven scripts/run_sr_vllm.sh recipe so libcudart.so.13/nvcc resolve and the
# engine-core worker uses spawn (fork+CUDA is illegal after seeding).
if [[ "$BACKEND" == "vllm" ]]; then
  export PYTHONUNBUFFERED=1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_USE_FLASHINFER_SAMPLER=0
  CU13ROOT="$($PY -c 'import os,nvidia;print(os.path.join(os.path.dirname(nvidia.__file__),"cu13"))' 2>/dev/null || true)"
  if [[ -n "$CU13ROOT" && -d "$CU13ROOT" ]]; then
    export CUDA_HOME="$CU13ROOT"
    export PATH="$CU13ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$CU13ROOT/lib:${LD_LIBRARY_PATH:-}"
  fi
fi

STAGE="${1:-all}"; shift || true
ALLOW_REPAIR=0
N_BOOT=10000
MODELS=""
DATASETS=""
ANSWER_SOURCE=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-generation-repair) ALLOW_REPAIR=1; shift;;
    --n-boot) N_BOOT="$2"; shift 2;;
    --models) MODELS="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    --answer-source) ANSWER_SOURCE="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done
mopt() { [[ -n "$MODELS" ]] && echo "--models $MODELS"; }
dopt() { [[ -n "$DATASETS" ]] && echo "--datasets $DATASETS"; }

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

run_audit() {
  log "STAGE audit"
  $PY scripts/audit_v6_generations.py $(mopt) $(dopt) 2>&1 | tee "$LOGDIR/audit_$STAMP.log"
}

run_parse() {
  log "STAGE parse (CPU)"
  $PY scripts/parse_v6_completions.py $(mopt) $(dopt) 2>&1 | tee "$LOGDIR/parse_$STAMP.log"
}

# Judge one stage across BOTH B200s (shard 0 -> GPU0, shard 1 -> GPU1) in parallel.
# CPU-only stages (coherence) run once with no CUDA device.
judge_stage() {
  local stage="$1"; shift
  local extra=("$@")
  log "STAGE $stage (2x B200, sharded)"
  if [[ "$stage" == "coherence" ]]; then
    $PY scripts/run_v6_judge_shard.py --stage coherence --gpu 0 --n-shards 1 \
      $(mopt) $(dopt) "${extra[@]}" 2>&1 | tee "$LOGDIR/${stage}_$STAMP.log"
    return
  fi
  CUDA_VISIBLE_DEVICES=0 $PY scripts/run_v6_judge_shard.py --stage "$stage" \
      --gpu 0 --n-shards 2 --backend "$BACKEND" $(mopt) $(dopt) "${extra[@]}" \
      >"$LOGDIR/${stage}_gpu0_$STAMP.log" 2>&1 &
  local p0=$!
  CUDA_VISIBLE_DEVICES=1 $PY scripts/run_v6_judge_shard.py --stage "$stage" \
      --gpu 1 --n-shards 2 --backend "$BACKEND" $(mopt) $(dopt) "${extra[@]}" \
      >"$LOGDIR/${stage}_gpu1_$STAMP.log" 2>&1 &
  local p1=$!
  # A failing shard must not kill the other (rows already written survive).
  local rc=0
  wait $p0 || rc=$?
  wait $p1 || rc=$?
  tail -n 3 "$LOGDIR/${stage}_gpu0_$STAMP.log" "$LOGDIR/${stage}_gpu1_$STAMP.log" || true
  [[ $rc -eq 0 ]] || log "WARNING: $stage had a shard failure (rc=$rc); completed rows preserved, resume by re-running the stage."
}

run_answer() { judge_stage answer "${EXTRA[@]}"; judge_stage coherence "${EXTRA[@]}"; }
run_monitor() { judge_stage monitor "${EXTRA[@]}"; judge_stage monitor --prose-prefix "${EXTRA[@]}"; }
run_pathway() { judge_stage pathway "${EXTRA[@]}"; }
run_sr() { judge_stage safety-reasoning "${EXTRA[@]}"; }

run_aggregate() {
  log "STAGE aggregate (CPU) n_boot=$N_BOOT"
  local src="${ANSWER_SOURCE:-v6}"
  # fall back to v5 answer labels if the v6 answer judge hasn't run yet
  if [[ "$src" == "v6" && -z "$(find runs/direction_a_v6/judge -name 'judge_answer_safety.jsonl' 2>/dev/null | head -1)" ]]; then
    log "no v6 answer-judge outputs found; using --answer-source v5 (aggregation-only correction)"
    src="v5"
  fi
  $PY scripts/aggregate_v6_metrics.py --n-boot "$N_BOOT" --answer-source "$src" \
      $(mopt) $(dopt) 2>&1 | tee "$LOGDIR/aggregate_$STAMP.log"
  $PY scripts/plot_v6_figures.py 2>&1 | tee -a "$LOGDIR/aggregate_$STAMP.log" || \
    log "plotting skipped (matplotlib unavailable)"
  $PY scripts/stage_v6_hf_export.py 2>&1 | tee -a "$LOGDIR/aggregate_$STAMP.log"
  $PY scripts/write_v6_manifest.py 2>&1 | tee -a "$LOGDIR/aggregate_$STAMP.log"
}

run_validation() {
  log "STAGE validation (CPU, reuse annotations)"
  $PY scripts/reproduce_v5_validation.py 2>&1 | tee "$LOGDIR/validation_$STAMP.log"
  $PY scripts/eval_pathway_judge.py --help >/dev/null 2>&1 && \
    log "pathway transfer-domain validation available via scripts/eval_pathway_judge.py (HarmThoughts held-out)" || true
}

run_check() {
  log "STAGE check"
  $PY scripts/check_v6_completeness.py $(mopt) $(dopt) 2>&1 | tee "$LOGDIR/check_$STAMP.log"
}

run_smoke() {
  log "STAGE smoke (2 cells, dry-run judging: builds sharded inputs, no model)"
  local sm_models="olmo3_7b_think" sm_ds="jbb"
  $PY scripts/audit_v6_generations.py --models $sm_models --datasets $sm_ds
  $PY scripts/parse_v6_completions.py --models $sm_models --datasets $sm_ds
  for st in answer monitor; do
    CUDA_VISIBLE_DEVICES=0 $PY scripts/run_v6_judge_shard.py --stage $st --gpu 0 --n-shards 2 \
      --models $sm_models --datasets $sm_ds --dry-run
    CUDA_VISIBLE_DEVICES=1 $PY scripts/run_v6_judge_shard.py --stage $st --gpu 1 --n-shards 2 \
      --models $sm_models --datasets $sm_ds --dry-run
  done
  $PY scripts/aggregate_v6_metrics.py --models $sm_models --datasets $sm_ds --n-boot 200 --answer-source v5
  $PY scripts/check_v6_completeness.py --models $sm_models --datasets $sm_ds
  log "smoke OK — remove --dry-run and run 'answer' on the B200s for real judging."
}

maybe_repair() {
  local n
  n="$($PY -c 'import json;print(json.load(open("runs/direction_a_v6/audit/generation_repair_manifest.json"))["n_repairs"])' 2>/dev/null || echo 0)"
  if [[ "$n" -gt 0 ]]; then
    if [[ "$ALLOW_REPAIR" -eq 1 ]]; then
      log "generation repair requested for $n cells — see manifest; regeneration NOT auto-run here (add your restricted run_generation call)."
    else
      log "AUDIT flags $n cells for repair but --allow-generation-repair not set; proceeding with reuse only."
    fi
  else
    log "no generation repair needed; reusing all v5 completions."
  fi
}

case "$STAGE" in
  audit) run_audit;;
  smoke) run_smoke;;
  parse) run_parse;;
  answer) run_answer;;
  monitor) run_monitor;;
  pathway) run_pathway;;
  safety-reasoning) run_sr;;
  aggregate) run_aggregate;;
  validation) run_validation;;
  check) run_check;;
  all)
    run_audit; maybe_repair; run_parse; run_check
    run_answer; run_monitor; run_pathway; run_sr
    run_aggregate; run_validation; run_check
    log "ALL stages complete. Reports under runs/direction_a_v6/reports/";;
  *) echo "unknown stage: $STAGE"; exit 2;;
esac
