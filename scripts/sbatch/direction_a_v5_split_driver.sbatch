#!/bin/bash
#SBATCH --job-name=direction_a_v5_gen
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/generation/direction_a_v5_split_driver-%x-%j.out
#SBATCH --error=logs/slurm/generation/direction_a_v5_split_driver-%x-%j.err
#
# Direction A v5 split driver.
#
# Runs discovery + generation for one MODEL_KEY, builds a query+metric judge
# manifest, submits the GPU judge array, then submits the merge/report job.

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

mkdir -p logs/slurm/generation

: "${MODEL_KEY:?Set MODEL_KEY.}"
CFG_DIR="configs/experiments/direction_a_v5_iso_asr/${MODEL_KEY}"
OUT_BASE="runs/direction_a_v5/${MODEL_KEY}"
TASKS_PER_WORKER="${TASKS_PER_WORKER:-16}"
MAX_ARRAY_CONCURRENCY="${MAX_ARRAY_CONCURRENCY:-16}"
CHAIN_MODEL_KEYS="${CHAIN_MODEL_KEYS:-}"

SBATCH_PIN_ARGS=()
if [[ -n "${SBATCH_NODELIST:-}" ]]; then
    SBATCH_PIN_ARGS+=(--nodelist="${SBATCH_NODELIST}")
fi

if [[ ! -d "${CFG_DIR}" ]]; then
    echo "no config dir for MODEL_KEY=${MODEL_KEY} (expected ${CFG_DIR})"
    exit 2
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true

if [[ -f "${CFG_DIR}/01-ships-discovery.yaml" ]]; then
    echo "=== [${MODEL_KEY}] SHIPS discovery ==="
    python -m scripts.run_attribution --config "${CFG_DIR}/01-ships-discovery.yaml"
fi
if [[ -f "${CFG_DIR}/16-neuron-discovery.yaml" ]]; then
    echo "=== [${MODEL_KEY}] neuron discovery ==="
    python -m scripts.run_neuron_discovery --config "${CFG_DIR}/16-neuron-discovery.yaml"
fi
if [[ -f "${CFG_DIR}/17-direction-extraction.yaml" ]]; then
    echo "=== [${MODEL_KEY}] direction extraction ==="
    python -m scripts.run_direction_extraction --config "${CFG_DIR}/17-direction-extraction.yaml"
fi

for dset_dir in "${CFG_DIR}/gen"/*/; do
    dkey="$(basename "${dset_dir}")"
    for gcfg in "${dset_dir}"*.yaml; do
        cond="$(basename "${gcfg}" .yaml)"
        echo "=== [${MODEL_KEY}/${dkey}] generate ${cond} ==="
        python -m scripts.run_generation --config "${gcfg}"
    done
done

MANIFEST="${OUT_BASE}/judge/_manifests/query_metric_tasks.jsonl"
TASKS="$(python -m scripts.make_v5_judge_manifest \
    --model-key "${MODEL_KEY}" \
    --out "${MANIFEST}" \
    --print-count-only)"
WORKERS=$(( (TASKS + TASKS_PER_WORKER - 1) / TASKS_PER_WORKER ))

if [[ "${TASKS}" -le 0 || "${WORKERS}" -le 0 ]]; then
    echo "no judge tasks produced for MODEL_KEY=${MODEL_KEY}"
    exit 3
fi

echo "=== [${MODEL_KEY}] judge manifest: ${TASKS} query+metric tasks, ${WORKERS} workers ==="
JUDGE_DEP_ARGS=()
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    JUDGE_DEP_ARGS+=(--dependency="afterok:${SLURM_JOB_ID}")
fi
JUDGE_JOB="$(sbatch --parsable \
    "${SBATCH_PIN_ARGS[@]}" \
    "${JUDGE_DEP_ARGS[@]}" \
    --array="0-$((WORKERS - 1))%${MAX_ARRAY_CONCURRENCY}" \
    --export=ALL,MODEL_KEY="${MODEL_KEY}",MANIFEST="${MANIFEST}",TASKS_PER_WORKER="${TASKS_PER_WORKER}" \
    scripts/sbatch/direction_a_v5_judge_query_metric.sbatch)"
echo "submitted judge array: ${JUDGE_JOB}"

MERGE_JOB="$(sbatch --parsable \
    "${SBATCH_PIN_ARGS[@]}" \
    --dependency="afterok:${JUDGE_JOB}" \
    --export=ALL,MODEL_KEY="${MODEL_KEY}" \
    scripts/sbatch/direction_a_v5_merge_report.sbatch)"
echo "submitted merge/report job: ${MERGE_JOB}"

if [[ -n "${CHAIN_MODEL_KEYS}" ]]; then
    NEXT_MODEL="${CHAIN_MODEL_KEYS%%:*}"
    REST_MODELS=""
    if [[ "${CHAIN_MODEL_KEYS}" == *:* ]]; then
        REST_MODELS="${CHAIN_MODEL_KEYS#*:}"
    fi

    NEXT_EXPORT="ALL,MODEL_KEY=${NEXT_MODEL},TASKS_PER_WORKER=${TASKS_PER_WORKER},MAX_ARRAY_CONCURRENCY=${MAX_ARRAY_CONCURRENCY}"
    if [[ -n "${SBATCH_NODELIST:-}" ]]; then
        NEXT_EXPORT+=",SBATCH_NODELIST=${SBATCH_NODELIST}"
    fi
    if [[ -n "${REST_MODELS}" ]]; then
        NEXT_EXPORT+=",CHAIN_MODEL_KEYS=${REST_MODELS}"
    fi

    NEXT_JOB="$(sbatch --parsable \
        "${SBATCH_PIN_ARGS[@]}" \
        --dependency="afterok:${MERGE_JOB}" \
        --export="${NEXT_EXPORT}" \
        scripts/sbatch/direction_a_v5_split_driver.sbatch)"
    echo "submitted next split driver: ${NEXT_MODEL} ${NEXT_JOB}"
fi
