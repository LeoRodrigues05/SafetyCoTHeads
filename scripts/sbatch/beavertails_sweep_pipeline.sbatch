#!/bin/bash
#SBATCH --job-name=beavertails_dose_sweep
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/beavertails_dose_sweep-%j.out
#SBATCH --error=logs/slurm/beavertails_dose_sweep-%j.err

# Submit from the repo root:
#   sbatch scripts/sbatch/beavertails_sweep_pipeline.sbatch
#
# Optional:
#   BEAVERTAILS_TOP_KS="1 3 5 8" sbatch --export=ALL,BEAVERTAILS_TOP_KS scripts/sbatch/beavertails_sweep_pipeline.sbatch
#   sbatch scripts/sbatch/beavertails_sweep_pipeline.sbatch --dry-run

set -euo pipefail
cd /home/leo.rodrigues/SafetyAblation/safety_cot_heads
source /home/leo.rodrigues/miniconda3/etc/profile.d/conda.sh
conda activate safety_cot_heads

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning

mkdir -p logs/slurm
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
fi

bash scripts/run_beavertails_sweep_pipeline.sh "$@"
