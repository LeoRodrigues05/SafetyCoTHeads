#!/bin/bash
#SBATCH --job-name=judge_qwen32b
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/judge_qwen32b-%j.out
#SBATCH --error=logs/slurm/judge_qwen32b-%j.err

# Pick the target node at submit time, for example:
#   sbatch --nodelist=<node-name> scripts/judge_qwen32b.sbatch

source /home/leo.rodrigues/miniconda3/etc/profile.d/conda.sh
conda activate safety_cot_heads

# Cache model downloads on the shared filesystem so re-runs are instant.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_VERBOSITY=warning

mkdir -p logs/slurm
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

bash scripts/judge_all_conditions.sh
