#!/bin/bash
#SBATCH --job-name=direction_a_v5_metrics_judge
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/judging/direction_a_v5_metrics_judge_%x-%j.out
#SBATCH --error=logs/slurm/judging/direction_a_v5_metrics_judge_%x-%j.err
#
# Direction A v5 — full-grid "initial metrics" judging.
#
# Judges the FULL n=100 completions (all conditions, jbb + bt) with ONLY the
# metrics required for the OAS / potency layer:
#   1. 5-label safety judge  -> harmful_response, safe_rejection, ...
#   2. coherence / gibberish gate -> clean_rate, harmful_among_clean_rate
#
# It deliberately SKIPS the expensive / deferred passes:
#   - per-prefix pathway (the 12-label grid)            [--skip-pathway]
#   - CoT-only monitor                                  [--skip-cot-only]
#   - indexed safety-reasoning trace (not run by this driver at all)
#
# This calls the core driver scripts.run_v4_jbb_judge directly (NOT the
# subset wrapper), so the 49-min/condition safety_reasoning_trace pass is
# excluded structurally. Output lands in the full-judge tree
#   runs/direction_a_v5/<MODEL_KEY>/judge/<dataset>/<condition>/seed0/
#
# Submit one job per model_key:
#   sbatch --export=ALL,MODEL_KEY=qwen3_8b           scripts/sbatch/direction_a_v5_metrics_judge.sbatch
#   sbatch --export=ALL,MODEL_KEY=llama31_8b_control scripts/sbatch/direction_a_v5_metrics_judge.sbatch
#
# Optional environment:
#   DATASETS="jbb bt"      # which datasets to judge
#   SKIP_COT_ONLY=0        # set 0 to ALSO capture the CoT monitor (E term);
#                          #   ~+1.5 min/condition. Default 1 (skipped).
#   SKIP_PATHWAY=0         # set 0 to run the full prefix pathway grid (slow).
#   SKIP_COHERENCE=1       # set 1 to skip the gibberish/quality gate (not advised).
#   SKIP_SAFETY=1          # set 1 to skip the 5-label safety judge (not advised).
#   SKIP_REPORT=0          # set 0 to also build the per-dataset v5 report.
#   ISO_ANCHOR=steering_a1.0
#   DRY_RUN=1              # write the plan json and exit without judging.

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

: "${MODEL_KEY:?Set MODEL_KEY (e.g. qwen3_8b). One sbatch per model.}"
CFG_DIR="configs/experiments/direction_a_v5_iso_asr/${MODEL_KEY}"
OUT_BASE="runs/direction_a_v5/${MODEL_KEY}"
DATASETS="${DATASETS:-jbb bt}"
ISO_ANCHOR="${ISO_ANCHOR:-steering_a1.0}"

if [[ ! -d "${CFG_DIR}" ]]; then
    echo "no config dir for MODEL_KEY=${MODEL_KEY} (expected ${CFG_DIR})"
    exit 2
fi

# ---- pass selection ------------------------------------------------------
# Defaults: keep safety + coherence, skip pathway + cot_only.
SKIP_ARGS=()
if [[ "${SKIP_SAFETY:-0}"    == "1" ]]; then SKIP_ARGS+=(--skip-safety);    fi
if [[ "${SKIP_PATHWAY:-1}"   == "1" ]]; then SKIP_ARGS+=(--skip-pathway);   fi
if [[ "${SKIP_COT_ONLY:-1}"  == "1" ]]; then SKIP_ARGS+=(--skip-cot-only);  fi
if [[ "${SKIP_COHERENCE:-0}" == "1" ]]; then SKIP_ARGS+=(--skip-coherence); fi
if [[ "${DRY_RUN:-0}"        == "1" ]]; then SKIP_ARGS+=(--dry-run);        fi

# ---- judge each dataset (full n=100, all conditions) ---------------------
for dkey in ${DATASETS}; do
    GEN_DIR="${OUT_BASE}/gen/${dkey}"
    JUDGE_OUT="${OUT_BASE}/judge/${dkey}"
    if [[ ! -d "${GEN_DIR}" ]]; then
        echo "no generations for ${MODEL_KEY}/${dkey} (expected ${GEN_DIR}); skipping"
        continue
    fi

    COND_ARGS=()
    for cond_dir in "${GEN_DIR}"/*/; do
        cond="$(basename "${cond_dir}")"
        comp_file="${cond_dir}seed0/completions_${cond}.jsonl"
        if [[ ! -f "${comp_file}" ]]; then
            comp_file="$(ls "${cond_dir}seed0/"completions_*.jsonl 2>/dev/null | head -n1)"
        fi
        [[ -z "${comp_file}" ]] && { echo "no completions for ${dkey}/${cond}"; continue; }
        COND_ARGS+=( --condition "tag=${cond},cond=${cond},completions=${comp_file}" )
    done

    if [[ ${#COND_ARGS[@]} -eq 0 ]]; then
        echo "no conditions found under ${GEN_DIR}; skipping"
        continue
    fi

    echo "=== [${MODEL_KEY}/${dkey}] metrics judge (${#COND_ARGS[@]} conditions) ==="
    python -m scripts.run_v4_jbb_judge \
        --config "${CFG_DIR}/judge.yaml" \
        --out-base "${JUDGE_OUT}" \
        --seed 0 \
        "${SKIP_ARGS[@]}" \
        "${COND_ARGS[@]}"

    if [[ "${DRY_RUN:-0}" != "1" && "${SKIP_REPORT:-1}" != "1" ]]; then
        echo "=== [${MODEL_KEY}/${dkey}] report ==="
        python -m scripts.make_v4_jbb_report \
            --in-base "${JUDGE_OUT}" \
            --out "${JUDGE_OUT}/v5_report.md" \
            --title "Direction A v5 — ${MODEL_KEY} on ${dkey} (metrics-only)" \
            --iso-anchor "${ISO_ANCHOR}" \
            || echo "report step failed for ${dkey} (judging outputs are intact)"
    fi
done

echo "=== v5 metrics judging done for ${MODEL_KEY} ==="
ls "${OUT_BASE}/judge"/*/*/seed0/summary.json 2>/dev/null | head || true
