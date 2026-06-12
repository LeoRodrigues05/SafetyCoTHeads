#!/bin/bash
#SBATCH --job-name=neuron_ksweep
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/discovery/neuron_ksweep-%j.out
#SBATCH --error=logs/slurm/discovery/neuron_ksweep-%j.err
#
# K-sweep for the neuron-ablation arm. For each K in {32,64,128,256,512,1024}
# runs both top-K positive and a layer-matched random-K control on 20 JBB
# prompts. ~12 conditions x ~5-15s each = comfortably under an hour.

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

python -m scripts.sweep_neuron_k \
    --ranking runs/direction_a/16-neuron-discovery-llama31/neuron_ranking.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --n 20 \
    --k-values 32 64 128 256 512 1024 \
    --max-new-tokens 200 \
    --out-dir runs/analysis/neuron_ksweep

echo "=== neuron K-sweep complete ==="
