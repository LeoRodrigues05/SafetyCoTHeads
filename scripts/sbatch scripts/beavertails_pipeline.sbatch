#!/bin/bash
#SBATCH --job-name=beavertails_pipeline
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/beavertails_pipeline-%j.out
#SBATCH --error=logs/slurm/beavertails_pipeline-%j.err

# Submit, optionally pinning the node:
#   sbatch "scripts/sbatch scripts/beavertails_pipeline.sbatch"
#   sbatch --nodelist=<node-name> "scripts/sbatch scripts/beavertails_pipeline.sbatch"

set -euo pipefail
cd /home/leo.rodrigues/SafetyAblation/safety_cot_heads

source /home/leo.rodrigues/miniconda3/etc/profile.d/conda.sh
conda activate safety_cot_heads

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_VERBOSITY=warning

mkdir -p logs/slurm
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

bash scripts/run_beavertails_pipeline.sh
