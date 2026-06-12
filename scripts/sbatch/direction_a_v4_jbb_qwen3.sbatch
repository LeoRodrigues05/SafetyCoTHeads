#!/bin/bash
#SBATCH --job-name=direction_a_v4_jbb_qwen3
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/judging/direction_a_v4_jbb_qwen3-%j.out
#SBATCH --error=logs/slurm/judging/direction_a_v4_jbb_qwen3-%j.err
#
# Direction A v4 — single-label judge run on JBB, Llama-3.1, 4 conditions:
#   baseline / ships_top10 / neurons_top32 / steering_ablate
#
# Per (condition, label) JSONLs are append-safe and resumable; safe to
# re-submit on SLURM kill.
#
# Outputs under runs/direction_a/19-v4-jbb-qwen3/<tag>/seed0/

set -euo pipefail
cd /home/leo.rodrigues/SafetyAblation/safety_cot_heads

source /home/leo.rodrigues/miniconda3/etc/profile.d/conda.sh
conda activate safety_cot_heads

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning
# Defragment CUDA allocator so bnb-NF4 30B-MoE shard loading fits a single 32 GiB card
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -z "${HF_TOKEN:-}" && -s "${HF_HOME}/token" ]]; then
    export HF_TOKEN="$(tr -d '[:space:]' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

mkdir -p logs/slurm/judging
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true

CFG=configs/experiments/direction_a_ships/19-v4-jbb-qwen3-judge.yaml
OUT_BASE=runs/direction_a/19-v4-jbb-qwen3
GEN_DIR=runs/direction_a

python -m scripts.run_v4_jbb_judge \
    --config "${CFG}" \
    --out-base "${OUT_BASE}" \
    --seed 0 \
    --condition "tag=03-baseline-llama31-jbb,cond=baseline,completions=${GEN_DIR}/03-baseline-llama31-jbb/seed0/completions_baseline.jsonl" \
    --condition "tag=05-ships-ablation-llama31-jbb,cond=ships_top10,completions=${GEN_DIR}/05-ships-ablation-llama31-jbb/seed0/completions_ships_top10.jsonl" \
    --condition "tag=16-neurons-top32-llama31-jbb,cond=neurons_top32,completions=${GEN_DIR}/16-neurons-top32-llama31-jbb/seed0/completions_neurons_top32.jsonl" \
    --condition "tag=17-steering-ablate-llama31-jbb,cond=steering_ablate,completions=${GEN_DIR}/17-steering-ablate-llama31-jbb/seed0/completions_steering_ablate.jsonl"

echo "=== v4 JBB judge complete ==="

# Generate the consolidated report on success.
python -m scripts.make_v4_jbb_report \
    --in-base "${OUT_BASE}" \
    --out "${OUT_BASE}/v4_jbb_report.md"

echo "=== report written: ${OUT_BASE}/v4_jbb_report.md ==="
