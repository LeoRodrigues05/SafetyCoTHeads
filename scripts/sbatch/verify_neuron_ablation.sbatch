#!/bin/bash
#SBATCH --job-name=verify_neuron
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/discovery/verify_neuron-%j.out
#SBATCH --error=logs/slurm/discovery/verify_neuron-%j.err
#
# Verify neuron-ablation hook + discovery on Llama-3.1-8B-Instruct.
# Runs 5 conditions on ~20 JBB prompts; printed table + JSONLs.

set -euo pipefail
cd /home/leo.rodrigues/SafetyAblation/safety_cot_heads

source /home/leo.rodrigues/miniconda3/etc/profile.d/conda.sh
conda activate safety_cot_heads

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning
if [[ -z "${HF_TOKEN:-}" && -s "${HF_HOME}/token" ]]; then
    export HF_TOKEN="$(tr -d '[:space:]' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

mkdir -p logs/slurm/discovery
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true

python -m scripts.verify_neuron_ablation \
    --ranking runs/direction_a/16-neuron-discovery-llama31/neuron_ranking.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --n 20 \
    --out-dir runs/analysis/neuron_verify

echo "=== verify_neuron_ablation complete ==="
