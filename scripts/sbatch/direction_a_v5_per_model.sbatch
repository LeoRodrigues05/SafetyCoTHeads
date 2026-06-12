#!/bin/bash
#SBATCH --job-name=direction_a_v5
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/pipeline/direction_a_v5_%x-%j.out
#SBATCH --error=logs/slurm/pipeline/direction_a_v5_%x-%j.err
#
# Direction A v5 — per-model iso-ASR pipeline.
#
# Submit one job per model_key:
#   sbatch --export=ALL,MODEL_KEY=qwen3_8b           scripts/sbatch/direction_a_v5_per_model.sbatch
#   sbatch --export=ALL,MODEL_KEY=qwen3_4b_thinking  scripts/sbatch/direction_a_v5_per_model.sbatch
#   sbatch --export=ALL,MODEL_KEY=r1_distill_8b      scripts/sbatch/direction_a_v5_per_model.sbatch
#   sbatch --export=ALL,MODEL_KEY=olmo2_7b_instruct  scripts/sbatch/direction_a_v5_per_model.sbatch
#   sbatch --export=ALL,MODEL_KEY=olmo2_7b_sft       scripts/sbatch/direction_a_v5_per_model.sbatch
#   sbatch --export=ALL,MODEL_KEY=llama31_8b_control scripts/sbatch/direction_a_v5_per_model.sbatch
#
# OLMo-SFT and llama31-control reuse OLMo-Instruct / Llama-3.1 discovery
# artifacts, so submit those AFTER their discovery parent has finished.
#
# All steps idempotent: rerunning skips completed cells.

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

mkdir -p logs/slurm/pipeline
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true

: "${MODEL_KEY:?Set MODEL_KEY (e.g. qwen3_8b). One sbatch per model.}"
CFG_DIR="configs/experiments/direction_a_v5_iso_asr/${MODEL_KEY}"
OUT_BASE="runs/direction_a_v5/${MODEL_KEY}"

if [[ ! -d "${CFG_DIR}" ]]; then
    echo "no config dir for MODEL_KEY=${MODEL_KEY} (expected ${CFG_DIR})"
    echo "did you run: python -m scripts.make_v5_configs --matrix configs/experiments/direction_a_v5_iso_asr/matrix.yaml ?"
    exit 2
fi

# ---------- 1. Discovery (skip if reusing) ---------------------------------
if [[ -f "${CFG_DIR}/01-ships-discovery.yaml" ]]; then
    echo "=== [${MODEL_KEY}] SHIPS discovery ==="
    python -m scripts.run_ablation --config "${CFG_DIR}/01-ships-discovery.yaml"
fi
if [[ -f "${CFG_DIR}/16-neuron-discovery.yaml" ]]; then
    echo "=== [${MODEL_KEY}] neuron discovery ==="
    python -m scripts.run_neuron_discovery --config "${CFG_DIR}/16-neuron-discovery.yaml"
fi
if [[ -f "${CFG_DIR}/17-direction-extraction.yaml" ]]; then
    echo "=== [${MODEL_KEY}] direction extraction ==="
    python -m scripts.run_direction_extraction --config "${CFG_DIR}/17-direction-extraction.yaml"
fi

# ---------- 2. Generation (per dataset × condition) -----------------------
for dset_dir in "${CFG_DIR}/gen"/*/; do
    dkey="$(basename "${dset_dir}")"
    for gcfg in "${dset_dir}"*.yaml; do
        cond="$(basename "${gcfg}" .yaml)"
        echo "=== [${MODEL_KEY}/${dkey}] generate ${cond} ==="
        python -m scripts.run_generation --config "${gcfg}"
    done
done

# ---------- 3. v4 judge + report (per dataset) -----------------------------
for dset_dir in "${CFG_DIR}/gen"/*/; do
    dkey="$(basename "${dset_dir}")"
    JUDGE_OUT="${OUT_BASE}/judge/${dkey}"

    COND_ARGS=()
    for gcfg in "${dset_dir}"*.yaml; do
        cond="$(basename "${gcfg}" .yaml)"
        comp_file="${OUT_BASE}/gen/${dkey}/${cond}/seed0/completions_${cond}.jsonl"
        if [[ ! -f "${comp_file}" ]]; then
            comp_file="$(ls "${OUT_BASE}/gen/${dkey}/${cond}/seed0/"completions_*.jsonl 2>/dev/null | head -n1)"
        fi
        [[ -z "${comp_file}" ]] && { echo "no completions for ${dkey}/${cond}"; continue; }
        COND_ARGS+=( --condition "tag=${cond},cond=${cond},completions=${comp_file}" )
    done

    echo "=== [${MODEL_KEY}/${dkey}] v4 judge (${#COND_ARGS[@]} conditions / 2) ==="
    python -m scripts.run_v4_jbb_judge \
        --config "${CFG_DIR}/judge.yaml" \
        --out-base "${JUDGE_OUT}" \
        --seed 0 \
        "${COND_ARGS[@]}"

    echo "=== [${MODEL_KEY}/${dkey}] report ==="
    python -m scripts.make_v4_jbb_report \
        --in-base "${JUDGE_OUT}" \
        --out "${JUDGE_OUT}/v5_report.md" \
        --title "Direction A v5 — ${MODEL_KEY} on ${dkey} (iso-ASR)" \
        --iso-anchor steering_a1.0
done

echo "=== v5 pipeline done for ${MODEL_KEY} ==="
ls "${OUT_BASE}/judge"/*/v5_report.md 2>/dev/null || true
