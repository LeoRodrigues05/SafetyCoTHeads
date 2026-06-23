#!/usr/bin/env bash
# =============================================================================
# GPU 0 — OLMo-3 judging: think + base + base_own
#
# Completes all judge passes (A safety, B coherence, C cot-only, D monitorability,
# E pathway) for all three OLMo-3 variants, pinned to GPU 0.
#
# Resume-safe: already-judged rows are skipped.
# Prior bf16 think/jbb tree (8/11 conditions done, no pathway) is extended.
# Prior bf16 base/bt/baseline partial (safety done, 1/12 pathway) is extended.
#
# Usage (from repo root, in its own tmux pane / terminal):
#   bash scripts/run_gpu0_olmo3_judging.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0      # pin to GPU 0

# bf16, metrics pass (pathway deferred — handled by a separate task)
export JUDGE_4BIT=0
export SKIP_PATHWAY=1
export SKIP_COT_ONLY=0

banner() { echo; echo "======================================================"; echo " $*"; echo "======================================================"; }

banner "[1/3] olmo3_7b_think — judge (resume: completes jbb, adds pathway everywhere)"
bash scripts/run_local_pipeline.sh olmo3_7b_think judge

banner "[2/3] olmo3_7b_base — judge (full: all conditions, all passes)"
bash scripts/run_local_pipeline.sh olmo3_7b_base judge

banner "[3/3] olmo3_7b_base_own — judge (full: all conditions, all passes)"
bash scripts/run_local_pipeline.sh olmo3_7b_base_own judge

banner "GPU 0 DONE — all OLMo-3 models judged"
echo "Run the status report when both GPUs finish:"
echo "  source .venv/bin/activate && python -m scripts.make_v5_metrics_status_report"
