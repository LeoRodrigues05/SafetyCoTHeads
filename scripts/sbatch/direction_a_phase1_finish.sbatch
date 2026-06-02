#!/bin/bash
#SBATCH --job-name=direction_a_phase1
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
# No hard node pin — pass `-w <node>` at submission to pin a specific host.
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/direction_a_phase1-%j.out
#SBATCH --error=logs/slurm/direction_a_phase1-%j.err

# Phase 1 finisher: runs everything needed to close out Phase 1.
#   Stage 1 — P1.1.3 final-answer judge (Qwen-2.5-32B NF4) over the 8
#             baseline + SHIPS-ablation cells.
#   Stage 2 — P1.5 template anchoring (rho_tpl) for both models +
#             residualised ranking.
#
# Each stage is skip-if-done at the cell level so re-submitting after a
# crash or partial completion only re-runs the missing pieces.

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
RUN_DIR=runs/direction_a
JUDGE_CFG=${CFG_DIR}/07-trajectory-judge.yaml

# --------------------------------------------------------------------------
# Stage 1 — final-answer judge over 8 cells (P1.1.3)
# --------------------------------------------------------------------------
judge_if_missing () {
    local tag="$1" cond="$2" seed="$3"
    local completions="${RUN_DIR}/${tag}/seed${seed}/completions_${cond}.jsonl"
    local out="${RUN_DIR}/${tag}/seed${seed}/judged_${cond}.jsonl"

    if [[ -s "${out}" ]]; then
        echo "--- skip judge (already done): ${tag} seed=${seed} cond=${cond} ---"
        return 0
    fi
    if [[ ! -s "${completions}" ]]; then
        echo "--- skip judge (no completions): ${completions} ---"
        return 0
    fi
    echo "--- judge: ${tag} seed=${seed} cond=${cond} -> ${out} ---"
    python -m scripts.run_judge \
        --config      "${JUDGE_CFG}" \
        --completions "${completions}" \
        --out         "${out}"
}

echo "=========================================="
echo "Stage 1 — final-answer judge (P1.1.3)"
echo "=========================================="
for seed in 0 1; do
    judge_if_missing 03-baseline-llama31-jbb        baseline    "${seed}"
    judge_if_missing 04-baseline-r1distill-jbb      baseline    "${seed}"
    judge_if_missing 05-ships-ablation-llama31-jbb  ships_top10 "${seed}"
    judge_if_missing 06-ships-ablation-r1distill-jbb ships_top10 "${seed}"
done

# --------------------------------------------------------------------------
# Stage 2 — template anchoring + residualised ranking (P1.5)
# --------------------------------------------------------------------------
anchor_if_missing () {
    local cfg="$1" out_dir="$2" ranking="$3"
    local sentinel="${out_dir}/template_anchoring.jsonl"

    if [[ -s "${sentinel}" ]]; then
        echo "--- skip anchoring (already done): ${out_dir} ---"
        return 0
    fi
    echo "--- anchoring: cfg=${cfg} ranking=${ranking} ---"
    if [[ -n "${ranking}" && -s "${ranking}" ]]; then
        python -m scripts.run_template_anchoring \
            --config "${cfg}" --ranking "${ranking}"
    else
        python -m scripts.run_template_anchoring --config "${cfg}"
    fi
}

echo "=========================================="
echo "Stage 2 — template anchoring (P1.5)"
echo "=========================================="
anchor_if_missing \
    "${CFG_DIR}/12-template-anchoring-llama31.yaml" \
    "${RUN_DIR}/12-template-anchoring-llama31" \
    "${RUN_DIR}/01-ships-discovery-llama31/ships_dataset_ranking.json"

anchor_if_missing \
    "${CFG_DIR}/13-template-anchoring-r1distill.yaml" \
    "${RUN_DIR}/13-template-anchoring-r1distill" \
    "${RUN_DIR}/02-ships-discovery-r1distill/ships_dataset_ranking.json"

echo "--- Phase 1 finished ---"
