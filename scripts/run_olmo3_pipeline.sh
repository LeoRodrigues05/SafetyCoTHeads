#!/usr/bin/env bash
# =============================================================================
# Temporary full-pipeline driver for the three OLMo-3 variants.
#
# Run order (single B200, sequential):
#   1. olmo3_7b_think    gen
#   2. olmo3_7b_base     discover   (SHIPS + neurons + direction on base itself)
#   3. olmo3_7b_base     gen        (interventions from think's artifacts)
#   4. olmo3_7b_base_own gen        (interventions from base's own artifacts)
#   5. olmo3_7b_think    judge
#   6. olmo3_7b_base     judge
#   7. olmo3_7b_base_own judge
#
# Safe to re-run: gen stages are skipped if all 22 completions files already
# exist. Judge stages are always run — the judge is resume-safe and will skip
# already-judged rows without re-loading the model unnecessarily.
#
# Usage (from repo root):
#   bash scripts/run_olmo3_pipeline.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNS="runs/direction_a_v5"
CFGS="configs/experiments/direction_a_v5_iso_asr"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

gen_done() {
    # True when every completions_*.jsonl for a model's gen tree is present.
    local model_key=$1
    local expected=22  # 11 conditions × 2 datasets
    local found
    found=$(find "${RUNS}/${model_key}/gen" -name "completions_*.jsonl" 2>/dev/null | wc -l)
    [[ "$found" -ge "$expected" ]]
}

discover_done() {
    # True when all three discovery output files exist for a model.
    local model_key=$1
    [[ -f "${RUNS}/${model_key}/01-ships-discovery/ships_dataset_ranking.json" ]] && \
    [[ -f "${RUNS}/${model_key}/16-neuron-discovery/neuron_ranking.json" ]]       && \
    [[ -f "${RUNS}/${model_key}/17-direction-extraction/refusal_directions.npz" ]]
}

banner() { echo; echo "======================================================"; echo " $*"; echo "======================================================"; }

# ---------------------------------------------------------------------------
# [1] think — generation
# ---------------------------------------------------------------------------
banner "[1/7] olmo3_7b_think — generation"
if gen_done olmo3_7b_think; then
    echo "  all 22 completions present — skipping"
else
    bash scripts/run_local_pipeline.sh olmo3_7b_think gen
fi

# ---------------------------------------------------------------------------
# [2] base — discovery (empirical, not assumed from think)
# ---------------------------------------------------------------------------
banner "[2/7] olmo3_7b_base — discovery (SHIPS + neurons + direction)"
if discover_done olmo3_7b_base; then
    echo "  all discovery artifacts present — skipping"
else
    bash scripts/run_local_pipeline.sh olmo3_7b_base discover
fi

# ---------------------------------------------------------------------------
# [3] base — generation using think's artifacts (original matched design)
# ---------------------------------------------------------------------------
banner "[3/7] olmo3_7b_base — generation (think's artifacts)"
if gen_done olmo3_7b_base; then
    echo "  all 22 completions present — skipping"
else
    bash scripts/run_local_pipeline.sh olmo3_7b_base gen
fi

# ---------------------------------------------------------------------------
# [4] base_own — generation using base's own discovered artifacts
# ---------------------------------------------------------------------------
banner "[4/7] olmo3_7b_base_own — generation (base's own artifacts)"
if gen_done olmo3_7b_base_own; then
    echo "  all 22 completions present — skipping"
else
    bash scripts/run_local_pipeline.sh olmo3_7b_base_own gen
fi

# ---------------------------------------------------------------------------
# [5-7] Judge all three (judge is resume-safe — skips already-judged rows)
# ---------------------------------------------------------------------------
banner "[5/7] olmo3_7b_think — judge"
bash scripts/run_local_pipeline.sh olmo3_7b_think judge

banner "[6/7] olmo3_7b_base — judge"
bash scripts/run_local_pipeline.sh olmo3_7b_base judge

banner "[7/7] olmo3_7b_base_own — judge"
bash scripts/run_local_pipeline.sh olmo3_7b_base_own judge

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
banner "ALL DONE"
echo "Refresh the status report:"
echo "  python -m scripts.make_v5_metrics_status_report"
