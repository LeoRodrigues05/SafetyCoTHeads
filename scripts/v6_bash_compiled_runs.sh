#!/usr/bin/env bash
# =============================================================================
# v6_bash_compiled_runs.sh — the ONE script for the whole Direction A v6
# corrected rerun on two B200s.
#
# Runs, in order: audit -> parse -> check -> answer(+coherence) -> monitor
# (+prose-prefix) -> pathway -> safety-reasoning -> aggregate -> validation ->
# plots -> manifest. Every model-judge stage is sharded across BOTH GPUs
# (CUDA_VISIBLE_DEVICES=0/1) and pools all of a shard's rows into a single
# continuous generation pass so neither GPU idles between cells.
#
# It only reads runs/direction_a_v5 (immutable) and writes runs/direction_a_v6.
# Resumable: re-running skips already-judged rows. Safe to Ctrl-C and restart.
#
# GPU UTILISATION NOTES
#   * Autoregressive decode is memory-bandwidth bound, so nvidia-smi "util" for a
#     30B judge tops out ~70-90%, not 100% — that is expected, not a stall. The
#     win here is removing the 0% idle GAPS (pooling) and using large batches so
#     both GPUs stay continuously busy.
#   * Batch sizes below are tuned for a 183 GB B200: ~61 GB bf16 weights +
#     ~0.5 GB KV/seq at the given token cap. Raise BATCH_* if you have headroom
#     (watch `nvidia-smi`), lower them if you OOM.
#
# Usage:
#   bash scripts/v6_bash_compiled_runs.sh                # full pipeline
#   bash scripts/v6_bash_compiled_runs.sh --from monitor # resume from a stage
#   BACKEND=vllm bash scripts/v6_bash_compiled_runs.sh   # use vLLM instead of HF
#   N_BOOT=10000 bash scripts/v6_bash_compiled_runs.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
PY="${PY:-.venv/bin/python}"
BACKEND="${BACKEND:-hf}"
N_BOOT="${N_BOOT:-10000}"
LOGDIR="runs/direction_a_v6/logs"
mkdir -p "$LOGDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER_LOG="$LOGDIR/compiled_${STAMP}.log"

# Per-stage tuning: BATCH and MAX_NEW_TOKENS. Pathway/SR emit longer JSON.
BATCH_ANSWER="${BATCH_ANSWER:-96}";  TOK_ANSWER="${TOK_ANSWER:-384}"
BATCH_MONITOR="${BATCH_MONITOR:-96}"; TOK_MONITOR="${TOK_MONITOR:-256}"
BATCH_PATHWAY="${BATCH_PATHWAY:-64}"; TOK_PATHWAY="${TOK_PATHWAY:-512}"
BATCH_SR="${BATCH_SR:-48}";          TOK_SR="${TOK_SR:-1024}"

FROM="audit"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

# vLLM env (only if BACKEND=vllm) — mirrors scripts/run_sr_vllm.sh.
if [[ "$BACKEND" == "vllm" ]]; then
  export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
  CU13ROOT="$($PY -c 'import os,nvidia;print(os.path.join(os.path.dirname(nvidia.__file__),"cu13"))' 2>/dev/null || true)"
  if [[ -n "$CU13ROOT" && -d "$CU13ROOT" ]]; then
    export CUDA_HOME="$CU13ROOT" PATH="$CU13ROOT/bin:$PATH" LD_LIBRARY_PATH="$CU13ROOT/lib:${LD_LIBRARY_PATH:-}"
  fi
fi

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

# Stage ordering so --from can skip completed stages.
STAGES=(audit parse check answer monitor pathway safety-reasoning aggregate validation)
declare -A ORDER; i=0; for s in "${STAGES[@]}"; do ORDER[$s]=$i; i=$((i+1)); done
skip() { [[ ${ORDER[$1]} -lt ${ORDER[$FROM]} ]]; }

# Run a model-judge stage across BOTH B200s using the dynamic work-queue runner
# (one persistent judge per GPU, cells pulled from a shared queue -> auto load
# balancing, neither GPU idles at the stage tail). One invocation drives both
# GPUs. Completed cells persist; re-running resumes (already-judged ids skipped).
judge_dual_gpu() {
  local stage="$1" batch="$2" toks="$3"; shift 3
  local extra=("$@")
  local suffix=""; [[ ${extra[*]} == *prose-prefix* ]] && suffix="_prefix"
  log "STAGE $stage$suffix  (dynamic dual-B200, backend=$BACKEND, batch=$batch, max_new_tokens=$toks)"
  local rc=0
  $PY scripts/run_v6_dual_gpu.py --stage "$stage" --gpus 2 --backend "$BACKEND" \
      --batch-size "$batch" --max-new-tokens "$toks" "${extra[@]}" \
      >"$LOGDIR/${stage}${suffix}_dual_$STAMP.log" 2>&1 || rc=$?
  [[ $rc -eq 0 ]] && log "  $stage$suffix OK" || \
    log "  WARNING: $stage$suffix rc=$rc (completed cells kept; re-run to resume)"
}

log "=== v6 compiled run start (backend=$BACKEND, from=$FROM) ==="

# ---- CPU preflight ---------------------------------------------------------
if ! skip audit; then log "STAGE audit"; $PY scripts/audit_v6_generations.py >>"$MASTER_LOG" 2>&1; fi
if ! skip parse; then log "STAGE parse"; $PY scripts/parse_v6_completions.py >>"$MASTER_LOG" 2>&1; fi
if ! skip check; then log "STAGE check"; $PY scripts/check_v6_completeness.py >>"$MASTER_LOG" 2>&1 || log "  check reported issues (see log)"; fi

# ---- GPU judging (both B200s) ---------------------------------------------
if ! skip answer; then
  judge_dual_gpu answer  "$BATCH_ANSWER"  "$TOK_ANSWER"
  log "STAGE coherence (CPU, answer_text)"; $PY scripts/run_v6_judge_shard.py \
      --stage coherence --gpu 0 --n-shards 1 >>"$MASTER_LOG" 2>&1
fi
if ! skip monitor; then
  judge_dual_gpu monitor "$BATCH_MONITOR" "$TOK_MONITOR"
  judge_dual_gpu monitor "$BATCH_MONITOR" "$TOK_MONITOR" --prose-prefix
fi
if ! skip pathway; then judge_dual_gpu pathway "$BATCH_PATHWAY" "$TOK_PATHWAY"; fi
if ! skip safety-reasoning; then judge_dual_gpu safety-reasoning "$BATCH_SR" "$TOK_SR"; fi

# ---- CPU aggregation / reporting ------------------------------------------
if ! skip aggregate; then
  log "STAGE aggregate (n_boot=$N_BOOT, answer_source=v6)"
  src="v6"; [[ -z "$(find runs/direction_a_v6/judge -name 'judge_answer_safety.jsonl' 2>/dev/null | head -1)" ]] && src="v5"
  $PY scripts/aggregate_v6_metrics.py --n-boot "$N_BOOT" --answer-source "$src" >>"$MASTER_LOG" 2>&1
  $PY scripts/plot_v6_figures.py >>"$MASTER_LOG" 2>&1 || log "  plotting skipped"
  $PY scripts/stage_v6_hf_export.py >>"$MASTER_LOG" 2>&1
  $PY scripts/write_v6_manifest.py >>"$MASTER_LOG" 2>&1
fi
if ! skip validation; then
  log "STAGE validation (reuse annotations)"
  $PY scripts/reproduce_v5_validation.py >>"$MASTER_LOG" 2>&1
fi

log "=== v6 compiled run COMPLETE — reports under runs/direction_a_v6/reports/ ==="
log "master log: $MASTER_LOG"
