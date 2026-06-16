#!/bin/bash
#SBATCH --job-name=direction_a_traj_resume
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
# No hard node pin — pass `-w <node>` at submission to pin a specific host.
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/analysis/direction_a_traj_resume-%j.out
#SBATCH --error=logs/slurm/analysis/direction_a_traj_resume-%j.err

# Resume-only: re-runs ONLY stage [4/4] (trajectory judge) over generation
# files that don't already have a finished trajectory_vectors.summary.json
# under runs/direction_a/07-trajectory/<tag>/seed<N>/.
#
# Uses the speedup-tuned config (batch_size=8, max_new_tokens=128) so the
# remaining ~6 R1-heavy files fit inside the 24h walltime.
#
# Submit with:
#   sbatch scripts/sbatch/direction_a_ships_resume_trajectory.sbatch

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

mkdir -p logs/slurm/analysis
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

CFG_DIR=configs/experiments/direction_a_ships
RUN_DIR=runs/direction_a

run_traj_if_missing () {
    local tag="$1" cond="$2" seed="$3"
    local completions="${RUN_DIR}/${tag}/seed${seed}/completions_${cond}.jsonl"
    local out="${RUN_DIR}/07-trajectory/${tag}/seed${seed}"
    local sentinel="${out}/trajectory_vectors.summary.json"

    if [[ -s "${sentinel}" ]]; then
        echo "--- skip (already done): ${tag} seed=${seed} cond=${cond} ---"
        return 0
    fi
    if [[ ! -s "${completions}" ]]; then
        echo "--- skip (no completions file): ${completions} ---"
        return 0
    fi
    # Wipe any partial outputs from a previous crash so we don't mix runs.
    rm -f "${out}/prefix_rows.jsonl" "${out}/judge_prefixes.jsonl" \
          "${out}/trajectory_vectors.jsonl" "${out}/trajectory.dryrun.json"
    echo "--- traj: ${tag} seed=${seed} cond=${cond} -> ${out} ---"
    python -m scripts.run_trajectory \
        --config "${CFG_DIR}/07-trajectory-judge.yaml" \
        --completions "${completions}" \
        --out-dir "${out}"
}

echo "=== Trajectory judge RESUME over remaining generation files ==="
for seed in 0 1; do
    run_traj_if_missing 03-baseline-llama31-jbb         baseline      "${seed}"
    run_traj_if_missing 04-baseline-r1distill-jbb       baseline      "${seed}"
    run_traj_if_missing 05-ships-ablation-llama31-jbb   ships_top10   "${seed}"
    run_traj_if_missing 06-ships-ablation-r1distill-jbb ships_top10   "${seed}"
done

echo "=== Direction A trajectory resume complete ==="
