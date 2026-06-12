#!/bin/bash
#SBATCH --job-name=direction_a_v5_qwen_subset
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/judging/direction_a_v5_qwen_subset-%j.out
#SBATCH --error=logs/slurm/judging/direction_a_v5_qwen_subset-%j.err
#
# Judge a deterministic Qwen3-8B subset with:
#   1. the standard v5 metrics, and
#   2. indexed safety-reasoning sentence/line location metrics.
#
# Default: 25 JBB + 25 BT completions per completed condition.
# The expensive all-prefix pathway grid is skipped by default so the full
# subset can target a roughly 6-hour single-GPU run. Set SKIP_PATHWAY=0 to run
# the full prefix pathway metrics; that is much slower.
#
# Example:
#   sbatch --nodelist=ws-l3-020 \
#     --export=ALL,MODEL_KEY=qwen3_8b,N_PER_DATASET=25 \
#     scripts/sbatch/direction_a_v5_qwen_subset_judging.sbatch
#
# Optional environment:
#   DATASETS="jbb bt"
#   CONDITIONS="baseline steering_a1.0"
#   OUT_BASE="runs/direction_a_v5/qwen3_8b/judge_subset_n25_fast"
#   SAMPLE_SEED=0
#   DRY_RUN=1
#   SKIP_STANDARD_METRICS=1
#   SKIP_SAFETY_REASONING=1
#   SKIP_SAFETY=1
#   SKIP_PATHWAY=0
#   SKIP_COT_ONLY=1
#   SKIP_COHERENCE=1
#   SAFETY_REASONING_MAX_NEW_TOKENS=512

set -euo pipefail
cd /home/leo.rodrigues/SafetyAblation/safety_cot_heads

source /home/leo.rodrigues/miniconda3/etc/profile.d/conda.sh
conda activate safety_cot_heads

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -z "${HF_TOKEN:-}" && -s "${HF_HOME}/token" ]]; then
    export HF_TOKEN="$(tr -d '[:space:]' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

mkdir -p logs/slurm/judging
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true

MODEL_KEY="${MODEL_KEY:-qwen3_8b}"
N_PER_DATASET="${N_PER_DATASET:-25}"
DATASETS="${DATASETS:-jbb bt}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
SAFETY_REASONING_MAX_NEW_TOKENS="${SAFETY_REASONING_MAX_NEW_TOKENS:-512}"
OUT_BASE="${OUT_BASE:-runs/direction_a_v5/${MODEL_KEY}/judge_subset_n${N_PER_DATASET}_fast}"

DATASET_ARGS=()
for dset in ${DATASETS}; do
    DATASET_ARGS+=("${dset}")
done

EXTRA_ARGS=()
if [[ -n "${CONDITIONS:-}" ]]; then
    EXTRA_ARGS+=(--conditions)
    for cond in ${CONDITIONS}; do
        EXTRA_ARGS+=("${cond}")
    done
fi
EXTRA_ARGS+=(--out-base "${OUT_BASE}")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi
if [[ "${SKIP_STANDARD_METRICS:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-standard-metrics)
fi
if [[ "${SKIP_SAFETY_REASONING:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-safety-reasoning)
fi
if [[ "${SKIP_SAFETY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-safety)
fi
if [[ "${SKIP_PATHWAY:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-pathway)
fi
if [[ "${SKIP_COT_ONLY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-cot-only)
fi
if [[ "${SKIP_COHERENCE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-coherence)
fi

python -m scripts.run_v5_qwen_subset_judging \
    --model-key "${MODEL_KEY}" \
    --datasets "${DATASET_ARGS[@]}" \
    --n-per-dataset "${N_PER_DATASET}" \
    --sample-seed "${SAMPLE_SEED}" \
    --safety-reasoning-max-new-tokens "${SAFETY_REASONING_MAX_NEW_TOKENS}" \
    "${EXTRA_ARGS[@]}"
