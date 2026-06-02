#!/bin/bash
#SBATCH --job-name=direction_a_v4_neuron_steering_circuit
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/direction_a_v4_nsc-%j.out
#SBATCH --error=logs/slurm/direction_a_v4_nsc-%j.err
#
# Direction A v4 — Llama-3.1-8B-Instruct extended intervention grid
#  Discovery: (a) MLP-neuron attribution (Wang et al. 2024)
#             (b) refusal-direction extraction (Arditi et al. 2024)
#  Generation conditions: { neurons_top32, steering_ablate, circuit_top8_top32_ablate }
#  Datasets:               { JailbreakBench (100 prompts), BeaverTails (~7980) }
#  Each cell writes completions_<cond>.jsonl (CoT + final answer in one
#  string) and judged_<cond>.jsonl (final-answer safety judge).
#
# Defaults used (all cited in the per-condition output report):
#   - Neuron ablation : top-32 contrast-scored neurons, scale_factor=0.0
#                       (Wang et al. 2024, default ablation).
#   - Steering        : project refusal direction (extracted at layer 14)
#                       out of the residual stream at every layer
#                       (Arditi et al. 2024, default directional ablation).
#   - Circuit         : union of SHIPS top-8 heads + Wang top-32 neurons
#                       + Arditi directional ablation; ACDC-style pragmatic
#                       union (Conmy et al. 2023).
#
# Idempotent: each step checks for a sentinel file and skips if present.
#
# Submit with:
#   sbatch scripts/sbatch/direction_a_v4_neuron_steering_circuit.sbatch

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

mkdir -p logs/slurm
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

CFG_DIR=configs/experiments/direction_a_ships
JUDGE_CFG="${CFG_DIR}/15z-final-safety-judge.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
generate () {
    local cfg="$1" out_dir="$2" cond="$3"
    local completions="${out_dir}/completions_${cond}.jsonl"
    if [[ -s "${completions}" ]]; then
        echo "--- skip generate (exists): ${completions} ---"
        return 0
    fi
    echo "--- generate: ${cfg} -> ${completions} ---"
    python -m scripts.run_generation --config "${cfg}"
}

judge () {
    local completions="$1" out="$2"
    if [[ ! -s "${completions}" ]]; then
        echo "--- skip judge (no completions): ${completions} ---"
        return 0
    fi
    # run_judge.py is now resume-aware (reads existing ids from ${out} and
    # only judges the pending rows, appending incrementally). Always invoke.
    echo "--- judge (resume-safe): ${completions} -> ${out} ---"
    python -m scripts.run_judge \
        --config "${JUDGE_CFG}" \
        --completions "${completions}" \
        --out "${out}"
}

# ---------------------------------------------------------------------------
# Stage 0a: Neuron discovery (Wang et al. 2024)
# ---------------------------------------------------------------------------
NEURON_RANKING=runs/direction_a/16-neuron-discovery-llama31/neuron_ranking.json
if [[ -s "${NEURON_RANKING}" ]]; then
    echo "--- skip neuron discovery (exists): ${NEURON_RANKING} ---"
else
    echo "=== Stage 0a/4: Neuron discovery ==="
    python -m scripts.run_neuron_discovery \
        --config "${CFG_DIR}/16-neuron-discovery-llama31.yaml"
fi

# ---------------------------------------------------------------------------
# Stage 0b: Refusal-direction extraction (Arditi et al. 2024)
# ---------------------------------------------------------------------------
DIRECTION_NPZ=runs/direction_a/17-direction-extraction-llama31/refusal_directions.npz
if [[ -s "${DIRECTION_NPZ}" ]]; then
    echo "--- skip direction extraction (exists): ${DIRECTION_NPZ} ---"
else
    echo "=== Stage 0b/4: Refusal-direction extraction ==="
    python -m scripts.run_direction_extraction \
        --config "${CFG_DIR}/17-direction-extraction-llama31.yaml"
fi

# ---------------------------------------------------------------------------
# Stage 1: Generation grid (6 cells = 3 conditions × 2 datasets)
# ---------------------------------------------------------------------------
echo "=== Stage 1/4: Generation ==="
generate "${CFG_DIR}/16a-neurons-top32-llama31-jbb.yaml"  runs/direction_a/16-neurons-top32-llama31-jbb/seed0      neurons_top32
generate "${CFG_DIR}/16b-neurons-top32-llama31-bt.yaml"   runs/direction_a/16-neurons-top32-llama31-bt/seed0       neurons_top32
generate "${CFG_DIR}/17a-steering-ablate-llama31-jbb.yaml" runs/direction_a/17-steering-ablate-llama31-jbb/seed0    steering_ablate
generate "${CFG_DIR}/17b-steering-ablate-llama31-bt.yaml"  runs/direction_a/17-steering-ablate-llama31-bt/seed0     steering_ablate
generate "${CFG_DIR}/18a-circuit-llama31-jbb.yaml"        runs/direction_a/18-circuit-llama31-jbb/seed0            circuit_top8_top32_ablate
generate "${CFG_DIR}/18b-circuit-llama31-bt.yaml"         runs/direction_a/18-circuit-llama31-bt/seed0             circuit_top8_top32_ablate

# ---------------------------------------------------------------------------
# Stage 2: Final-answer safety judge (6 cells)
# ---------------------------------------------------------------------------
echo "=== Stage 2/4: Final-answer safety judge ==="
judge runs/direction_a/16-neurons-top32-llama31-jbb/seed0/completions_neurons_top32.jsonl                runs/direction_a/16-neurons-top32-llama31-jbb/seed0/judged_neurons_top32.jsonl
judge runs/direction_a/16-neurons-top32-llama31-bt/seed0/completions_neurons_top32.jsonl                 runs/direction_a/16-neurons-top32-llama31-bt/seed0/judged_neurons_top32.jsonl
judge runs/direction_a/17-steering-ablate-llama31-jbb/seed0/completions_steering_ablate.jsonl            runs/direction_a/17-steering-ablate-llama31-jbb/seed0/judged_steering_ablate.jsonl
judge runs/direction_a/17-steering-ablate-llama31-bt/seed0/completions_steering_ablate.jsonl             runs/direction_a/17-steering-ablate-llama31-bt/seed0/judged_steering_ablate.jsonl
judge runs/direction_a/18-circuit-llama31-jbb/seed0/completions_circuit_top8_top32_ablate.jsonl          runs/direction_a/18-circuit-llama31-jbb/seed0/judged_circuit_top8_top32_ablate.jsonl
judge runs/direction_a/18-circuit-llama31-bt/seed0/completions_circuit_top8_top32_ablate.jsonl           runs/direction_a/18-circuit-llama31-bt/seed0/judged_circuit_top8_top32_ablate.jsonl

echo "=== Direction A v4 neuron/steering/circuit grid complete ==="
