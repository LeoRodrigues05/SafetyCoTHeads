#!/bin/bash
#SBATCH --job-name=direction_a_v5_judging
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/judging/direction_a_v5_complete_judging-%j.out
#SBATCH --error=logs/slurm/judging/direction_a_v5_complete_judging-%j.err
#
# Resume Direction A v5 judging for existing generation outputs.
#
# Example:
#   sbatch --export=ALL,MODEL_KEY=llama31_8b_control scripts/sbatch/direction_a_v5_complete_judging.sbatch
#
# Optional environment:
#   DATASETS="bt jbb"          Dataset subset; default both.
#   JUDGE_PATH=single          "single" reuses legacy outputs; "split" uses shard path.
#   DRY_RUN=1                  Print the planned work only.
#   SKIP_REPORT=1              Do not regenerate v5_report.md.

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

: "${MODEL_KEY:?Set MODEL_KEY, e.g. llama31_8b_control.}"
DATASETS="${DATASETS:-bt jbb}"
JUDGE_PATH="${JUDGE_PATH:-single}"

DATASET_ARGS=()
for dset in ${DATASETS}; do
    DATASET_ARGS+=("${dset}")
done

EXTRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi
if [[ "${SKIP_REPORT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-report)
fi

python -m scripts.complete_v5_judging "${MODEL_KEY}" \
    --datasets "${DATASET_ARGS[@]}" \
    --path "${JUDGE_PATH}" \
    "${EXTRA_ARGS[@]}"
