#!/bin/bash
#SBATCH --job-name=direction_a_v4_intervention_grid
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/direction_a_v4_grid-%j.out
#SBATCH --error=logs/slurm/direction_a_v4_grid-%j.err
#
# Direction A v4 — Llama-3.1-8B-Instruct intervention grid
#  Conditions: {baseline, SHIPS top-5, SHIPS top-8}
#  Datasets : {JailbreakBench (100 prompts), BeaverTails (~7980 prompts)}
#  For each cell: generate completions (CoT + final answer in one string)
#                 + run final-answer safety judge.
#
# Idempotent: each step checks for a sentinel file and skips if present.
# Logs land under logs/slurm/direction_a_v4_grid-<JOBID>.{out,err}.
#
# Submit with:
#   sbatch scripts/sbatch/direction_a_v4_intervention_grid.sbatch

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

mkdir -p logs/slurm
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

CFG_DIR=configs/experiments/direction_a_ships
JUDGE_CFG="${CFG_DIR}/15z-final-safety-judge.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
generate () {
    local cfg="$1" out_dir="$2" cond="$3"
    local completions="${out_dir}/completions_${cond}.jsonl"
    if [[ -s "${completions}" ]]; then
        echo "--- skip generate (exists): ${completions} ---"
        return 0
    fi
    echo "--- generate: ${cfg} -> ${completions} ---"
    python -m scripts.run_generation --config "${cfg}"
}

judge () {
    local completions="$1" out="$2"
    if [[ ! -s "${completions}" ]]; then
        echo "--- skip judge (no completions): ${completions} ---"
        return 0
    fi
    # run_judge.py is now resume-aware (reads existing ids from ${out} and
    # only judges the pending rows, appending incrementally). Always invoke.
    echo "--- judge (resume-safe): ${completions} -> ${out} ---"
    python -m scripts.run_judge \
        --config "${JUDGE_CFG}" \
        --completions "${completions}" \
        --out "${out}"
}

# ---------------------------------------------------------------------------
# Generation grid (6 cells)
# ---------------------------------------------------------------------------
echo "=== Stage 1/2: Generation ==="
generate "${CFG_DIR}/03-baseline-gen-llama31-jbb.yaml"      runs/direction_a/03-baseline-llama31-jbb/seed0       baseline
generate "${CFG_DIR}/15b-ships-top5-llama31-jbb.yaml"       runs/direction_a/15-ships-top5-llama31-jbb/seed0     ships_top5
generate "${CFG_DIR}/15c-ships-top8-llama31-jbb.yaml"       runs/direction_a/15-ships-top8-llama31-jbb/seed0     ships_top8
generate "${CFG_DIR}/15a-baseline-llama31-bt.yaml"          runs/direction_a/15-baseline-llama31-bt/seed0        baseline
generate "${CFG_DIR}/15d-ships-top5-llama31-bt.yaml"        runs/direction_a/15-ships-top5-llama31-bt/seed0      ships_top5
generate "${CFG_DIR}/15e-ships-top8-llama31-bt.yaml"        runs/direction_a/15-ships-top8-llama31-bt/seed0      ships_top8

# ---------------------------------------------------------------------------
# Final-answer safety judge (6 cells)
# ---------------------------------------------------------------------------
echo "=== Stage 2/2: Final-answer safety judge ==="
judge runs/direction_a/03-baseline-llama31-jbb/seed0/completions_baseline.jsonl       runs/direction_a/03-baseline-llama31-jbb/seed0/judged_baseline.jsonl
judge runs/direction_a/15-ships-top5-llama31-jbb/seed0/completions_ships_top5.jsonl   runs/direction_a/15-ships-top5-llama31-jbb/seed0/judged_ships_top5.jsonl
judge runs/direction_a/15-ships-top8-llama31-jbb/seed0/completions_ships_top8.jsonl   runs/direction_a/15-ships-top8-llama31-jbb/seed0/judged_ships_top8.jsonl
judge runs/direction_a/15-baseline-llama31-bt/seed0/completions_baseline.jsonl        runs/direction_a/15-baseline-llama31-bt/seed0/judged_baseline.jsonl
judge runs/direction_a/15-ships-top5-llama31-bt/seed0/completions_ships_top5.jsonl    runs/direction_a/15-ships-top5-llama31-bt/seed0/judged_ships_top5.jsonl
judge runs/direction_a/15-ships-top8-llama31-bt/seed0/completions_ships_top8.jsonl    runs/direction_a/15-ships-top8-llama31-bt/seed0/judged_ships_top8.jsonl

echo "=== Direction A v4 grid complete ==="
