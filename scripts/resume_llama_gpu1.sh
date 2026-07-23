#!/usr/bin/env bash
# Resume llama31_8b_control defence cells on GPU1.
# Prereq: HF token with meta-llama access already active (hf auth login / HF_TOKEN).
# Steering direction already exists; this fills the two missing discovery artifacts
# (SHIPS + neuron) then runs all remaining gen cells idempotently.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GPU pinning is supplied by the caller, e.g. `CUDA_VISIBLE_DEVICES=1 bash scripts/resume_llama_gpu1.sh`
PY=".venv/bin/python"
CFG_SHIPS="configs/experiments/direction_a_ships/01-ships-discovery-llama31.yaml"
CFG_NEURON="configs/experiments/direction_a_ships/16-neuron-discovery-llama31.yaml"

echo "=== [llama31] $(date -Is) SHIPS head discovery ==="
if [[ ! -f runs/direction_a/01-ships-discovery-llama31/ships_dataset_ranking.json ]]; then
  "$PY" -m scripts.run_attribution --config "$CFG_SHIPS" || { echo "SHIPS discovery FAILED"; exit 1; }
else echo "SHIPS ranking already present, skipping"; fi

echo "=== [llama31] $(date -Is) neuron discovery ==="
if [[ ! -f runs/direction_a/16-neuron-discovery-llama31/neuron_ranking.json ]]; then
  "$PY" -m scripts.run_neuron_discovery --config "$CFG_NEURON" || { echo "neuron discovery FAILED"; exit 1; }
else echo "neuron ranking already present, skipping"; fi

echo "=== [llama31] $(date -Is) generation (all missing cells) ==="
# batch_size raised from 4 -> 48 to use the ~167 GB of free KV-cache headroom
# (max_new_tokens=512 here; greedy decoding is batch-invariant).
"$PY" -m scripts.complete_v5_generation llama31_8b_control --overrides batch_size=48
echo "=== [llama31] $(date -Is) DONE ==="
