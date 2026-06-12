#!/bin/bash
#SBATCH --job-name=direction_a_v5_judge
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/slurm/judging/direction_a_v5_judge-%A_%a.out
#SBATCH --error=logs/slurm/judging/direction_a_v5_judge-%A_%a.err

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

: "${MODEL_KEY:?Set MODEL_KEY.}"
: "${MANIFEST:?Set MANIFEST.}"
TASKS_PER_WORKER="${TASKS_PER_WORKER:-16}"

python -m scripts.run_v5_query_metric_judge \
    --manifest "${MANIFEST}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --tasks-per-worker "${TASKS_PER_WORKER}"
