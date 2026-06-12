#!/bin/bash
#SBATCH --job-name=direction_a_v5_smoke
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/smoke/direction_a_v5_smoke-%j.out
#SBATCH --error=logs/slurm/smoke/direction_a_v5_smoke-%j.err
#
# Direction A v5 SMOKE TEST.
#
# Validates the full v5 pipeline end-to-end at micro scale (n=4 prompts):
#   * Qwen3-4B-Thinking baseline   -> exercises `enable_thinking` plumbing
#   * Llama-3.1-8B baseline           ┐
#   * Llama-3.1-8B steering_a1.0      ├ reuses existing v4 discovery
#                                     │ artifacts, exercises gibberish
#                                     ┘ gating + iso-ASR report
#
# Outputs under runs/direction_a_v5/_smoke/.

set -euo pipefail
cd /home/leo.rodrigues/SafetyAblation/safety_cot_heads

source /home/leo.rodrigues/miniconda3/etc/profile.d/conda.sh
conda activate safety_cot_heads

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -z "${HF_TOKEN:-}" && -s "${HF_HOME}/token" ]]; then
    export HF_TOKEN="$(tr -d '[:space:]' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

mkdir -p logs/slurm runs/direction_a_v5/_smoke
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true

OUT="runs/direction_a_v5/_smoke"
JUDGE_CFG="configs/experiments/direction_a_v5_iso_asr/llama31_8b_control/judge.yaml"

# ---------- 1. Qwen3-4B-Thinking baseline (tests enable_thinking) ---------
echo "=== smoke: qwen3_4b_thinking baseline (n=4) ==="
python -m scripts.run_generation \
    --config configs/experiments/direction_a_v5_iso_asr/qwen3_4b_thinking/gen/jbb/baseline.yaml \
    --overrides \
        dataset.n=4 \
        batch_size=2 \
        decoding.max_new_tokens=256 \
        output.dir=${OUT}/qwen3_4b_thinking/baseline/seed0

# ---------- 2. Llama-3.1 baseline + steering (tests gating + report) -------
echo "=== smoke: llama31 baseline (n=4) ==="
python -m scripts.run_generation \
    --config configs/experiments/direction_a_v5_iso_asr/llama31_8b_control/gen/jbb/baseline.yaml \
    --overrides \
        dataset.n=4 \
        batch_size=2 \
        decoding.max_new_tokens=256 \
        output.dir=${OUT}/llama31/baseline/seed0

echo "=== smoke: llama31 steering_a1.0 (n=4) ==="
python -m scripts.run_generation \
    --config configs/experiments/direction_a_v5_iso_asr/llama31_8b_control/gen/jbb/steering_a1.0.yaml \
    --overrides \
        dataset.n=4 \
        batch_size=2 \
        decoding.max_new_tokens=256 \
        output.dir=${OUT}/llama31/steering_a1.0/seed0

# ---------- 3. Judge all 3 conditions in one model load --------------------
echo "=== smoke: v4 judge (gating ON) ==="
python -m scripts.run_v4_jbb_judge \
    --config "${JUDGE_CFG}" \
    --out-base "${OUT}/judge" \
    --seed 0 \
    --condition "tag=qwen3_4b_baseline,cond=baseline,completions=${OUT}/qwen3_4b_thinking/baseline/seed0/completions_baseline.jsonl" \
    --condition "tag=llama31_baseline,cond=baseline,completions=${OUT}/llama31/baseline/seed0/completions_baseline.jsonl" \
    --condition "tag=llama31_steering,cond=steering_a1.0,completions=${OUT}/llama31/steering_a1.0/seed0/completions_steering_a1.0.jsonl"

# ---------- 4. Report (with iso-ASR section) -------------------------------
echo "=== smoke: report ==="
python -m scripts.make_v4_jbb_report \
    --in-base "${OUT}/judge" \
    --out "${OUT}/v5_smoke_report.md" \
    --title "Direction A v5 SMOKE (n=4 per cell)" \
    --iso-anchor steering_a1.0

echo "=== smoke pipeline done ==="
echo "report: ${OUT}/v5_smoke_report.md"
cat "${OUT}/v5_smoke_report.md"
