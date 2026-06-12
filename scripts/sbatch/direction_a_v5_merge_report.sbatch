#!/bin/bash
#SBATCH --job-name=direction_a_v5_merge
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/analysis/direction_a_v5_merge-%j.out
#SBATCH --error=logs/slurm/analysis/direction_a_v5_merge-%j.err

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

: "${MODEL_KEY:?Set MODEL_KEY.}"
CFG_DIR="configs/experiments/direction_a_v5_iso_asr/${MODEL_KEY}"
OUT_BASE="runs/direction_a_v5/${MODEL_KEY}"

python -m scripts.merge_v5_query_metric_judge --model-key "${MODEL_KEY}"

for dset_dir in "${CFG_DIR}/gen"/*/; do
    dkey="$(basename "${dset_dir}")"
    JUDGE_OUT="${OUT_BASE}/judge/${dkey}"
    echo "=== [${MODEL_KEY}/${dkey}] report ==="
    python -m scripts.make_v4_jbb_report \
        --in-base "${JUDGE_OUT}" \
        --out "${JUDGE_OUT}/v5_report.md" \
        --title "Direction A v5 - ${MODEL_KEY} on ${dkey} (iso-ASR)" \
        --iso-anchor steering_a1.0
done

echo "=== split v5 merge/report done for ${MODEL_KEY} ==="
ls "${OUT_BASE}/judge"/*/v5_report.md 2>/dev/null || true
