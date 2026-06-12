#!/bin/bash
# Submit Direction A v5 split drivers for one or more model keys.
#
# Usage:
#   bash scripts/submit_direction_a_v5_split.sh qwen3_8b qwen3_4b_thinking
#
# Optional environment:
#   TASKS_PER_WORKER=16 MAX_ARRAY_CONCURRENCY=16 bash scripts/submit_direction_a_v5_split.sh ...

set -euo pipefail
cd /home/leo.rodrigues/SafetyAblation/safety_cot_heads

if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 MODEL_KEY [MODEL_KEY ...]"
    exit 2
fi

declare -A JOB_BY_MODEL=()

for model_key in "$@"; do
    dep_args=()
    if [[ "${model_key}" == "olmo2_7b_sft" && -n "${JOB_BY_MODEL[olmo2_7b_instruct]:-}" ]]; then
        dep_args=(--dependency="afterok:${JOB_BY_MODEL[olmo2_7b_instruct]}")
    fi

    job_id="$(sbatch --parsable \
        "${dep_args[@]}" \
        --export=ALL,MODEL_KEY="${model_key}",TASKS_PER_WORKER="${TASKS_PER_WORKER:-16}",MAX_ARRAY_CONCURRENCY="${MAX_ARRAY_CONCURRENCY:-16}" \
        scripts/sbatch/direction_a_v5_split_driver.sbatch)"
    JOB_BY_MODEL["${model_key}"]="${job_id}"
    echo "${model_key}: split driver ${job_id}"
done
