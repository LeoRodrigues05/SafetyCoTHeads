#!/bin/bash
#SBATCH --job-name=direction_a_ships
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --nodelist=ws-l5-031
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/direction_a_ships-%j.out
#SBATCH --error=logs/slurm/direction_a_ships-%j.err

# Single-job Direction A SHIPS slice. Runs end-to-end:
#   1. SHIPS head discovery on Llama-3.1-8B-Instruct  (MaliciousInstruct, n=100)
#   2. SHIPS head discovery on DeepSeek-R1-Distill-8B (MaliciousInstruct, n=100)
#   3. Baseline + SHIPS-ablation generation on JailbreakBench (n=100)
#      for both models, two seeds each (greedy + sampled t=0.7).
#   4. Trajectory judge (Qwen2.5-32B-Instruct, NF4) over every generation file:
#      segments completions into cumulative sentence prefixes (prose for Llama,
#      <think>-block aware for R1) and emits 7-dim trajectory vectors.
#
# PREREQ: both gated repos must be accepted on the active HF token:
#   - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
#   - https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# Make sure HF_TOKEN (or ~/.cache/huggingface/token) carries the granted account.
#
# Submit with:
#   sbatch scripts/sbatch/direction_a_ships_pipeline.sbatch

set -euo pipefail
cd /home/leo.rodrigues/SafetyAblation/safety_cot_heads

source /home/leo.rodrigues/miniconda3/etc/profile.d/conda.sh
conda activate safety_cot_heads

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning

# Explicitly propagate the HF auth token to the compute node so gated
# repos (Llama-3.1, R1-Distill) resolve. The compute node otherwise
# sees an unauthenticated request even with the token file present.
if [[ -z "${HF_TOKEN:-}" && -s "${HF_HOME}/token" ]]; then
    export HF_TOKEN="$(tr -d '[:space:]' < "${HF_HOME}/token")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

mkdir -p logs/slurm runs/direction_a
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Pre-flight: fail fast if the gated repos aren't accessible with the current token.
python - <<'PY'
import sys
from huggingface_hub import HfApi
api = HfApi()
gated = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen/Qwen2.5-32B-Instruct",
]
missing = []
for rid in gated:
    try:
        api.model_info(rid)
        print(f"[hf-auth] OK   {rid}")
    except Exception as e:
        print(f"[hf-auth] FAIL {rid}: {e}")
        missing.append(rid)
if missing:
    sys.exit(f"Missing HF access for: {missing}")
PY

CFG_DIR=configs/experiments/direction_a_ships
RUN_DIR=runs/direction_a

# ---------------------------------------------------------------------------
# 1-2. SHIPS head discovery (one run per model).
# ---------------------------------------------------------------------------
echo "=== [1/4] SHIPS discovery: Llama-3.1-8B-Instruct ==="
python -m scripts.run_attribution --config "${CFG_DIR}/01-ships-discovery-llama31.yaml"

echo "=== [2/4] SHIPS discovery: DeepSeek-R1-Distill-Llama-8B ==="
python -m scripts.run_attribution --config "${CFG_DIR}/02-ships-discovery-r1distill.yaml"

# ---------------------------------------------------------------------------
# 3. Generation (baseline + SHIPS-ablation) x 2 models x 2 seeds.
#    Seed 0 = greedy (config defaults).
#    Seed 1 = sampled at temperature 0.7.
# ---------------------------------------------------------------------------
run_gen () {
    # $1 = config path, $2 = run_dir tag (e.g. 03-baseline-llama31-jbb), $3 = seed
    local cfg="$1" tag="$2" seed="$3"
    local out="${RUN_DIR}/${tag}/seed${seed}"
    echo "--- gen: $(basename "${cfg}") seed=${seed} -> ${out} ---"
    if [[ "${seed}" == "0" ]]; then
        python -m scripts.run_generation --config "${cfg}" \
            --overrides "seed=0" "decoding.seed=0" "output.dir=${out}"
    else
        python -m scripts.run_generation --config "${cfg}" \
            --overrides "seed=${seed}" "decoding.seed=${seed}" \
                        "decoding.do_sample=true" "decoding.temperature=0.7" \
                        "output.dir=${out}"
    fi
}

echo "=== [3/4] Generation: baseline + ships_top10, 2 models, 2 seeds ==="
for seed in 0 1; do
    run_gen "${CFG_DIR}/03-baseline-gen-llama31-jbb.yaml"        03-baseline-llama31-jbb        "${seed}"
    run_gen "${CFG_DIR}/04-baseline-gen-r1distill-jbb.yaml"      04-baseline-r1distill-jbb      "${seed}"
    run_gen "${CFG_DIR}/05-ships-ablation-gen-llama31-jbb.yaml"  05-ships-ablation-llama31-jbb  "${seed}"
    run_gen "${CFG_DIR}/06-ships-ablation-gen-r1distill-jbb.yaml" 06-ships-ablation-r1distill-jbb "${seed}"
done

# ---------------------------------------------------------------------------
# 4. Trajectory judge across every generation file.
#    Loads Qwen2.5-32B-Instruct (NF4) once per call; sequential to avoid OOM.
# ---------------------------------------------------------------------------
run_traj () {
    # $1 = gen run tag, $2 = condition name, $3 = seed
    local tag="$1" cond="$2" seed="$3"
    local completions="${RUN_DIR}/${tag}/seed${seed}/completions_${cond}.jsonl"
    local out="${RUN_DIR}/07-trajectory/${tag}/seed${seed}"
    echo "--- traj: ${tag} seed=${seed} cond=${cond} ---"
    python -m scripts.run_trajectory \
        --config "${CFG_DIR}/07-trajectory-judge.yaml" \
        --completions "${completions}" \
        --out-dir "${out}"
}

echo "=== [4/4] Trajectory judge over 8 generation files ==="
for seed in 0 1; do
    run_traj 03-baseline-llama31-jbb         baseline      "${seed}"
    run_traj 04-baseline-r1distill-jbb       baseline      "${seed}"
    run_traj 05-ships-ablation-llama31-jbb   ships_top10   "${seed}"
    run_traj 06-ships-ablation-r1distill-jbb ships_top10   "${seed}"
done

echo "=== Direction A SHIPS slice complete ==="
