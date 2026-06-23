#!/usr/bin/env bash
# =============================================================================
# GPU 1 — Llama & Qwen judging: qwen3_8b + llama31_8b_control
#
# Runs full judge passes (A safety, B coherence, C cot-only, D monitorability,
# E pathway) for qwen3_8b and llama31_8b_control, pinned to GPU 1.
#
# Completions must already be present (synced from b200_cot_judging_bundle):
#   runs/direction_a_v5/qwen3_8b/gen/         (20 files)
#   runs/direction_a_v5/llama31_8b_control/gen/ (20 files)
#
# No existing judge/ trees to move — both models start fresh.
#
# Usage (from repo root, in its own tmux pane / terminal):
#   bash scripts/run_gpu1_llm_judging.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1      # pin to GPU 1

# bf16, metrics pass (pathway deferred — handled by a separate task)
export JUDGE_4BIT=0
export SKIP_PATHWAY=1
export SKIP_COT_ONLY=0

banner() { echo; echo "======================================================"; echo " $*"; echo "======================================================"; }

# Verify completions are present before starting
for M in qwen3_8b llama31_8b_control; do
    n=$(find "runs/direction_a_v5/${M}/gen" -name "completions_*.jsonl" 2>/dev/null | wc -l)
    if [[ "$n" -lt 20 ]]; then
        echo "ERROR: only $n completion files found for $M (expected 20)."
        echo "Sync the bundle first: rsync -a runs/b200_cot_judging_bundle/runs/ runs/"
        exit 1
    fi
    echo "  $M: $n completion files — OK"
done

banner "[1/2] qwen3_8b — judge (full: 10 conditions × 2 datasets, all passes)"
bash scripts/run_local_pipeline.sh qwen3_8b judge

banner "[2/2] llama31_8b_control — judge (full: 10 conditions × 2 datasets, all passes)"
bash scripts/run_local_pipeline.sh llama31_8b_control judge

banner "GPU 1 DONE — qwen3_8b and llama31_8b_control judged"
echo "Run the status report when both GPUs finish:"
echo "  source .venv/bin/activate && python -m scripts.make_v5_metrics_status_report"
