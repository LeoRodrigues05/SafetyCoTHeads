#!/bin/bash
#SBATCH --job-name=v5_n25_final_judge
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/judging/v5_n25_final_judge-%j.out
#SBATCH --error=logs/slurm/judging/v5_n25_final_judge-%j.err
#
# FINAL n=25 (PAIRED) judging rerun — both policy models, in ONE 24h job.
#
# Judges the PAIRED subset (the SAME 25 prompts per dataset across EVERY
# condition; see `_subset_rows` in run_v5_qwen_subset_judging). Passes per cell:
#   * 5-label safety + coherence gate    (standard metrics)
#   * indexed safety-reasoning trace     (the CoT-trace analysis)
#   * CoT-only monitor                   (asr_cot_pred + monitorability gap)
#
# The 12-label PATHWAY pass is EXCLUDED on purpose: even at n=25 the
# prefix-expansion x 12-label grid runs tens of hours and will not fit 24h. Run
# the mechanism layer separately (B200 / longer wall-clock) if you want it.
#
# Models run sequentially: qwen3_8b (~16h, long CoT) then llama31_8b_control
# (~6-8h). Total ~22-24h. IDEMPOTENT / resume-safe: if the 24h cap cuts it off,
# just `sbatch` this same file again and it continues where it stopped.
#
# Output is the canonical n25 source `judge_subset_n25_fast`. Any PRE-EXISTING
# (unpaired) dir is moved aside to `*_unpaired_backup` on the first run; a
# `.paired_subset` marker keeps resubmits resuming instead of re-moving.
#
# Submit:  sbatch scripts/sbatch/direction_a_v5_n25_final_judging.sbatch

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

mkdir -p logs/slurm/judging
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true

MODELS="${MODELS:-qwen3_8b llama31_8b_control}"
N_PER_DATASET="${N_PER_DATASET:-25}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
SR_MAX_NEW_TOKENS="${SAFETY_REASONING_MAX_NEW_TOKENS:-512}"

for MODEL_KEY in ${MODELS}; do
    OUT_BASE="runs/direction_a_v5/${MODEL_KEY}/judge_subset_n${N_PER_DATASET}_fast"
    BACKUP="${OUT_BASE}_unpaired_backup"

    # One-time: preserve a pre-existing UNPAIRED subset so we never mix old+new.
    # The `.paired_subset` marker means a dir was produced by THIS (paired)
    # script, so resubmits skip the move and resume into the fresh dir.
    if [[ -d "${OUT_BASE}" && ! -f "${OUT_BASE}/.paired_subset" ]]; then
        [[ -d "${BACKUP}" ]] && BACKUP="${BACKUP}_$(date +%s)"
        echo "preserving pre-existing UNPAIRED ${OUT_BASE} -> ${BACKUP}"
        mv "${OUT_BASE}" "${BACKUP}"
    fi
    mkdir -p "${OUT_BASE}"
    touch "${OUT_BASE}/.paired_subset"

    echo "============================================================"
    echo "=== [${MODEL_KEY}] n=${N_PER_DATASET} PAIRED final judging ==="
    echo "============================================================"
    python -m scripts.run_v5_qwen_subset_judging \
        --model-key "${MODEL_KEY}" \
        --out-base "${OUT_BASE}" \
        --datasets jbb bt \
        --n-per-dataset "${N_PER_DATASET}" \
        --sample-seed "${SAMPLE_SEED}" \
        --safety-reasoning-max-new-tokens "${SR_MAX_NEW_TOKENS}" \
        --skip-pathway \
        || echo "WARNING: ${MODEL_KEY} judging exited non-zero; continuing to next model"
done

echo "=== n=${N_PER_DATASET} PAIRED final judging complete for: ${MODELS} ==="
ls "runs/direction_a_v5"/*/judge_subset_n"${N_PER_DATASET}"_fast/*/*/seed0/safety_reasoning.summary.json 2>/dev/null | wc -l || true
