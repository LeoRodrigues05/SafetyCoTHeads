#!/bin/bash
#SBATCH --job-name=direction_a_pathway_passA
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/direction_a_pathway_passA-%j.out
#SBATCH --error=logs/slurm/direction_a_pathway_passA-%j.err
#
# Direction A v4 — Pass A pilot.
# Re-judges the EXISTING trajectory prefix_rows.jsonl under
# runs/direction_a/07-trajectory/<tag>/seed0/ with the pathway taxonomy
# (12 labels) + CoT-only monitor, and joins against the final-answer
# safety judge (`judged_*.jsonl`) to compute monitorability gaps.
#
# Idempotent: --skip-pathway / --skip-cot-only reuse prior partials.
# Outputs land under runs/direction_a/14-pathway/<tag>/seed0/.
#
# Submit with:
#   sbatch scripts/sbatch/direction_a_pathway_pass_a.sbatch

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

CFG=configs/experiments/direction_a_ships/14-pathway-judge.yaml
GEN_DIR=runs/direction_a
TRAJ_DIR=runs/direction_a/07-trajectory
OUT_BASE=runs/direction_a/14-pathway

run_pathway () {
    local tag="$1" cond="$2" seed="$3"
    local completions="${GEN_DIR}/${tag}/seed${seed}/completions_${cond}.jsonl"
    local final_judge="${GEN_DIR}/${tag}/seed${seed}/judged_${cond}.jsonl"
    local traj_out="${TRAJ_DIR}/${tag}/seed${seed}"
    local out="${OUT_BASE}/${tag}/seed${seed}"
    local sentinel="${out}/pathway_vectors.summary.json"

    if [[ -s "${sentinel}" ]]; then
        echo "--- skip (already done): ${tag} seed=${seed} ---"
        return 0
    fi
    if [[ ! -s "${completions}" ]]; then
        echo "--- skip (no completions): ${completions} ---"
        return 0
    fi
    mkdir -p "${out}"
    # Seed the per-prefix inputs by copying the existing trajectory output;
    # the orchestrator detects prefix_rows.jsonl and skips re-segmentation.
    if [[ -s "${traj_out}/prefix_rows.jsonl" && ! -s "${out}/prefix_rows.jsonl" ]]; then
        cp "${traj_out}/prefix_rows.jsonl" "${out}/prefix_rows.jsonl"
        echo "--- seeded prefix_rows from ${traj_out} ---"
    fi
    local final_arg=()
    if [[ -s "${final_judge}" ]]; then
        final_arg=(--final-judge "${final_judge}")
    else
        echo "--- WARN: no final judge at ${final_judge}; gap will be skipped ---"
    fi
    echo "--- pathway: ${tag} seed=${seed} -> ${out} ---"
    python -m scripts.run_pathway_analysis \
        --config "${CFG}" \
        --completions "${completions}" \
        --out-dir "${out}" \
        "${final_arg[@]}"
}

echo "=== Direction A v4 Pass A pathway + monitorability ==="
# Pass A pilot: Llama-3.1 baseline + SHIPS, seed 0 only.
run_pathway 03-baseline-llama31-jbb         baseline      0
run_pathway 05-ships-ablation-llama31-jbb   ships_top10   0
# Optional R1-Distill arm (uncomment to extend pilot):
# run_pathway 04-baseline-r1distill-jbb       baseline      0
# run_pathway 06-ships-ablation-r1distill-jbb ships_top10   0

echo "=== Pass A complete ==="
