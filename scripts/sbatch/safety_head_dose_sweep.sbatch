#!/bin/bash
#SBATCH --job-name=safety_head_dose_sweep
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/safety_head_dose_sweep-%j.out
#SBATCH --error=logs/slurm/safety_head_dose_sweep-%j.err

# Pick the target node at submit time, for example:
#   sbatch --nodelist=<node-name> scripts/sbatch/safety_head_dose_sweep.sbatch

set -euo pipefail
cd /home/leo.rodrigues/SafetyAblation/safety_cot_heads

source /home/leo.rodrigues/miniconda3/etc/profile.d/conda.sh
conda activate safety_cot_heads

# Cache model downloads on the shared filesystem so re-runs are instant.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_VERBOSITY=warning

mkdir -p logs/slurm
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

bash scripts/run_safety_head_dose_sweep.sh
