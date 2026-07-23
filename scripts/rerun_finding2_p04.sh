#!/usr/bin/env bash
# =============================================================================
# P0.4 Finding-2 rerun: pathway (cumulative prefixes) + safety-reasoning
# (indexed sentences), with the CORRECT judge-input structures.
#
# The first v6 run fed both judges a single raw trace_text blob; this rerun uses
# scripts/run_v6_dual_gpu.py with the fixed build_inputs (cumulative-prefix
# pathway rows + indexed-sentence SR rows). Because the old (wrong) outputs share
# ids with the new ones (SR especially), we back them up first so resume starts
# clean instead of silently reusing stale rows (P0.7).
#
# SCOPE (Finding-2 core): primary explicit-trace models on primary datasets.
#   full-grid pathway is ~764k rows (~50h); this scope is ~243k rows (~20h) +
#   SR ~5h. Widen MODELS/DATASETS below to include r1 (exploratory) or xstest.
#
# Usage (in tmux):
#   tmux new -s f2 'bash scripts/rerun_finding2_p04.sh 2>&1 | tee -a runs/direction_a_v6/logs/finding2_p04.out'
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
PY="${PY:-.venv/bin/python}"
BACKEND="${BACKEND:-hf}"
MODELS="${MODELS:-qwen3_8b olmo3_7b_think}"
DATASETS="${DATASETS:-jbb bt}"
LOGDIR="runs/direction_a_v6/logs"; mkdir -p "$LOGDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="runs/direction_a_v6/judge/_pre_p04_fix_${STAMP}"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

log "=== P0.4 Finding-2 rerun (models: $MODELS | datasets: $DATASETS | backend: $BACKEND) ==="

# 1. Back up the old (wrong-input) pathway + SR outputs and their scratch so
#    resume cannot reuse them. Non-destructive: files are MOVED, not deleted.
log "backing up old pathway/SR outputs -> $BACKUP"
mkdir -p "$BACKUP"
moved=0
for f in $(find runs/direction_a_v6/judge -type f \
            \( -name 'judge_pathway.jsonl' -o -name 'judge_safety_reasoning_trace.jsonl' \) \
            -not -path '*_pre_p04_fix*' 2>/dev/null); do
  rel="${f#runs/direction_a_v6/judge/}"; mkdir -p "$BACKUP/$(dirname "$rel")"
  mv "$f" "$BACKUP/$rel"; moved=$((moved+1))
done
# stale scratch for these stages
mv runs/direction_a_v6/judge/_shard_scratch/pathway_* "$BACKUP/" 2>/dev/null || true
mv runs/direction_a_v6/judge/_shard_scratch/safety-reasoning_* "$BACKUP/" 2>/dev/null || true
log "backed up $moved per-cell files"

# 2. Pathway: cumulative-prefix inputs, dynamic dual-GPU (biggest cells first).
log "STAGE pathway (cumulative prefixes) — dual GPU"
$PY scripts/run_v6_dual_gpu.py --stage pathway --gpus 2 --backend "$BACKEND" \
    --batch-size 64 --max-new-tokens 512 --models $MODELS --datasets $DATASETS \
    2>&1 | tee "$LOGDIR/pathway_p04_${STAMP}.log"

# 3. Safety-reasoning: indexed-sentence inputs, dynamic dual-GPU.
log "STAGE safety-reasoning (indexed sentences) — dual GPU"
$PY scripts/run_v6_dual_gpu.py --stage safety-reasoning --gpus 2 --backend "$BACKEND" \
    --batch-size 48 --max-new-tokens 1024 --models $MODELS --datasets $DATASETS \
    2>&1 | tee "$LOGDIR/safety-reasoning_p04_${STAMP}.log"

# 4. Finding-2 aggregation (pathway trajectories + safety-reasoning rates).
log "STAGE Finding-2 aggregation"
$PY scripts/aggregate_v6_reasoning.py --models $MODELS --datasets $DATASETS \
    2>&1 | tee "$LOGDIR/finding2_agg_${STAMP}.log" || log "aggregation step reported an issue (see log)"

log "=== P0.4 Finding-2 rerun COMPLETE ==="
