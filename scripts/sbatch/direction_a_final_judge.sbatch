#!/bin/bash
#SBATCH --job-name=direction_a_final_judge
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --nodelist=ws-l5-031
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/judging/direction_a_final_judge-%j.out
#SBATCH --error=logs/slurm/judging/direction_a_final_judge-%j.err

# Final-answer safety judge (Qwen-2.5-32B NF4) over the Direction-A
# baseline + SHIPS-ablation completions (Phase 1, P1.1.3).
#
# Inputs (eight cells):
#   runs/direction_a/03-baseline-llama31-jbb/seed{0,1}/completions_baseline.jsonl
#   runs/direction_a/04-baseline-r1distill-jbb/seed{0,1}/completions_baseline.jsonl
#   runs/direction_a/05-ships-ablation-llama31-jbb/seed{0,1}/completions_ships_top10.jsonl
#   runs/direction_a/06-ships-ablation-r1distill-jbb/seed{0,1}/completions_ships_top10.jsonl
#
# Output for each cell:
#   runs/direction_a/<NN>-..../seed<N>/judged_<basename>.jsonl
#
# Skip-if-done: a cell is skipped when its judged_*.jsonl already exists
# and is non-empty.
#
# Submit with:
#   sbatch scripts/sbatch/direction_a_final_judge.sbatch

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

mkdir -p logs/slurm/judging
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

JUDGE_CFG=configs/experiments/direction_a_ships/07-trajectory-judge.yaml
RUN_DIR=runs/direction_a

judge_if_missing () {
    local tag="$1" cond="$2" seed="$3"
    local completions="${RUN_DIR}/${tag}/seed${seed}/completions_${cond}.jsonl"
    local out_dir="${RUN_DIR}/${tag}/seed${seed}"
    local out="${out_dir}/judged_${cond}.jsonl"

    if [[ -s "${out}" ]]; then
        echo "--- skip (already done): ${tag} seed=${seed} cond=${cond} ---"
        return 0
    fi
    if [[ ! -s "${completions}" ]]; then
        echo "--- skip (no completions file): ${completions} ---"
        return 0
    fi
    echo "--- judge: ${tag} seed=${seed} cond=${cond} -> ${out} ---"
    python -m scripts.run_judge \
        --config      "${JUDGE_CFG}" \
        --completions "${completions}" \
        --out         "${out}"
}

# Baseline cells
for seed in 0 1; do
    judge_if_missing 03-baseline-llama31-jbb       baseline      "${seed}"
    judge_if_missing 04-baseline-r1distill-jbb     baseline      "${seed}"
done

# SHIPS-ablation cells
for seed in 0 1; do
    judge_if_missing 05-ships-ablation-llama31-jbb   ships_top10 "${seed}"
    judge_if_missing 06-ships-ablation-r1distill-jbb ships_top10 "${seed}"
done

echo "--- all cells processed ---"
