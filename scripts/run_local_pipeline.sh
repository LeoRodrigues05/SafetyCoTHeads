#!/usr/bin/env bash
# =============================================================================
# Direction A v5 — non-Slurm pipeline driver for a single stock VM (tmux-ready).
#
# Same logic as scripts/sbatch/direction_a_v5_per_model.sbatch, but STAGE-GATED
# so generation and judging can be launched separately (and two models can be
# generated in parallel in two tmux panes).
#
# Usage:
#   bash scripts/run_local_pipeline.sh <MODEL_KEY> <STAGE>
#     MODEL_KEY : olmo2_7b_instruct | olmo2_7b_sft | qwen3_8b | llama31_8b_control | ...
#     STAGE     : discover | gen | judge | all
#
# IMPORTANT ordering:
#   olmo2_7b_sft REUSES olmo2_7b_instruct discovery artifacts (ships heads /
#   neuron ranking / refusal directions). Run the instruct discovery first:
#       bash scripts/run_local_pipeline.sh olmo2_7b_instruct discover
#   before launching any olmo2_7b_sft gen.  (olmo2_7b_sft has no discover stage.)
#
# Judge runs the Qwen3-30B-A3B judge in BF16 by default (B200 has the VRAM;
# avoids bitsandbytes on Blackwell).  Tunables via environment:
#   ENV_DIR=.venv          # virtualenv to activate
#   JUDGE_4BIT=1           # force 4-bit judge instead of bf16 (small-VRAM GPUs)
#   SKIP_PATHWAY=1         # skip the slow 12-label pathway pass (default: skip)
#   SKIP_COT_ONLY=0        # keep the CoT-only monitor / E-term (default: keep)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_DIR="${ENV_DIR:-.venv}"
if [[ -f "$ENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
fi
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

MODEL_KEY="${1:?usage: run_local_pipeline.sh <MODEL_KEY> <STAGE: discover|gen|judge|all>}"
STAGE="${2:?usage: run_local_pipeline.sh <MODEL_KEY> <STAGE: discover|gen|judge|all>}"
CFG_DIR="configs/experiments/direction_a_v5_iso_asr/${MODEL_KEY}"
OUT_BASE="runs/direction_a_v5/${MODEL_KEY}"
[[ -d "$CFG_DIR" ]] || { echo "no config dir for MODEL_KEY=$MODEL_KEY ($CFG_DIR)"; exit 2; }

# ---- judge pass selection ---------------------------------------------------
JUDGE_FLAGS=()
[[ "${SKIP_PATHWAY:-1}"  == "1" ]] && JUDGE_FLAGS+=(--skip-pathway)
[[ "${SKIP_COT_ONLY:-0}" == "1" ]] && JUDGE_FLAGS+=(--skip-cot-only)
JUDGE_OVERRIDES=(--overrides)
if [[ "${JUDGE_4BIT:-0}" == "1" ]]; then
    JUDGE_OVERRIDES+=(model.load_in_4bit=true)
else
    JUDGE_OVERRIDES+=(model.load_in_4bit=false)
fi

do_discover() {
    [[ -f "$CFG_DIR/01-ships-discovery.yaml" ]] && {
        echo "=== [$MODEL_KEY] SHIPS head discovery ==="
        python -m scripts.run_ablation --config "$CFG_DIR/01-ships-discovery.yaml"; }
    [[ -f "$CFG_DIR/16-neuron-discovery.yaml" ]] && {
        echo "=== [$MODEL_KEY] neuron discovery ==="
        python -m scripts.run_neuron_discovery --config "$CFG_DIR/16-neuron-discovery.yaml"; }
    [[ -f "$CFG_DIR/17-direction-extraction.yaml" ]] && {
        echo "=== [$MODEL_KEY] refusal-direction extraction ==="
        python -m scripts.run_direction_extraction --config "$CFG_DIR/17-direction-extraction.yaml"; }
}

do_gen() {
    for dset_dir in "$CFG_DIR/gen"/*/; do
        dkey="$(basename "$dset_dir")"
        for gcfg in "$dset_dir"*.yaml; do
            cond="$(basename "$gcfg" .yaml)"
            echo "=== [$MODEL_KEY/$dkey] generate $cond ==="
            python -m scripts.run_generation --config "$gcfg"
        done
    done
}

do_judge() {
    for dset_dir in "$CFG_DIR/gen"/*/; do
        dkey="$(basename "$dset_dir")"
        COND_ARGS=()
        for gcfg in "$dset_dir"*.yaml; do
            cond="$(basename "$gcfg" .yaml)"
            comp="$OUT_BASE/gen/$dkey/$cond/seed0/completions_${cond}.jsonl"
            [[ -f "$comp" ]] || comp="$(ls "$OUT_BASE/gen/$dkey/$cond/seed0/"completions_*.jsonl 2>/dev/null | head -1)"
            [[ -z "$comp" || ! -f "$comp" ]] && { echo "  missing completions for $dkey/$cond — generate first"; continue; }
            COND_ARGS+=(--condition "tag=$cond,cond=$cond,completions=$comp")
        done
        [[ ${#COND_ARGS[@]} -eq 0 ]] && { echo "no completions under $dset_dir; skipping judge"; continue; }
        echo "=== [$MODEL_KEY/$dkey] judge (${#COND_ARGS[@]} conditions) ==="
        python -m scripts.run_v4_jbb_judge \
            --config "$CFG_DIR/judge.yaml" \
            --out-base "$OUT_BASE/judge/$dkey" \
            --seed 0 \
            "${JUDGE_FLAGS[@]}" \
            "${COND_ARGS[@]}" \
            "${JUDGE_OVERRIDES[@]}"
        echo "=== [$MODEL_KEY/$dkey] report ==="
        python -m scripts.make_v4_jbb_report \
            --in-base "$OUT_BASE/judge/$dkey" \
            --out "$OUT_BASE/judge/$dkey/v5_report.md" \
            --title "Direction A v5 — $MODEL_KEY on $dkey" \
            --iso-anchor steering_a1.0 \
            || echo "  report step failed (judging outputs are intact)"
    done
}

case "$STAGE" in
    discover) do_discover ;;
    gen)      do_gen ;;
    judge)    do_judge ;;
    all)      do_discover; do_gen; do_judge ;;
    *) echo "unknown STAGE=$STAGE (want discover|gen|judge|all)"; exit 2 ;;
esac
echo "=== done: $MODEL_KEY / $STAGE ==="
