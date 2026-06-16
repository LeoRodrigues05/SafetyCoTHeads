#!/usr/bin/env bash
# Exp 7 dose sweep: run BeaverTails top-k ablations for safety heads and
# random controls, then judge/evaluate each k against the BeaverTails baseline.
#
# Usage:
#   bash scripts/run_beavertails_sweep_pipeline.sh
#   BEAVERTAILS_TOP_KS="1 3 5 8" bash scripts/run_beavertails_sweep_pipeline.sh
#   bash scripts/run_beavertails_sweep_pipeline.sh --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY="--dry-run"
fi

CFG_DIR="configs/experiments/exp07_beavertails_pipeline"
TOP_KS_RAW="${BEAVERTAILS_TOP_KS:-1 3 5 8}"
TOP_KS="${TOP_KS_RAW//,/ }"

BASELINE_COMP="runs/08-beaver-baseline/completions_baseline.jsonl"
BASELINE_JUDGE="runs/08-beaver-baseline/judge_beavertails.jsonl"
DISCOVERY_RANKING="runs/07-ships-beavertails/ships_dataset_ranking.json"

run_step() {
  local output="$1"
  shift
  if [[ -z "$DRY" && -s "$output" ]]; then
    echo "Found ${output}; skipping."
    return 0
  fi
  "$@"
}

echo "BeaverTails sweep top-k values: ${TOP_KS}"

run_step "$DISCOVERY_RANKING" \
  python scripts/run_attribution.py \
    --config "$CFG_DIR/01-ships-discovery-beavertails.yaml" $DRY

run_step "$BASELINE_COMP" \
  python scripts/run_generation.py \
    --config "$CFG_DIR/02-baseline.yaml" $DRY

run_step "$BASELINE_JUDGE" \
  python scripts/run_judge.py \
    --config "$CFG_DIR/judge.yaml" \
    --completions "$BASELINE_COMP" \
    --out "$BASELINE_JUDGE" $DRY

for k in $TOP_KS; do
  echo "=== BeaverTails top-${k} sweep ==="

  safety_dir="runs/09-beaver-safety-ablation-top${k}"
  random_dir="runs/10-beaver-random-top${k}"
  layer_dir="runs/11-beaver-layer-matched-top${k}"

  safety_comp="${safety_dir}/completions_safety_head_ablation_top${k}.jsonl"
  random_comp="${random_dir}/completions_random_head_ablation_top${k}.jsonl"
  layer_comp="${layer_dir}/completions_layer_matched_random_head_ablation_top${k}.jsonl"

  safety_judge="${safety_dir}/judge_beavertails.jsonl"
  random_judge="${random_dir}/judge_beavertails.jsonl"
  layer_judge="${layer_dir}/judge_beavertails.jsonl"

  run_step "$safety_comp" \
    python scripts/run_generation.py \
      --config "$CFG_DIR/03-safety-head-ablation.yaml" \
      --overrides \
        "condition=safety_head_ablation_top${k}" \
        "heads.top_k=${k}" \
        "output.dir=${safety_dir}" $DRY

  run_step "$random_comp" \
    python scripts/run_generation.py \
      --config "$CFG_DIR/04-random-head-ablation.yaml" \
      --overrides \
        "condition=random_head_ablation_top${k}" \
        "heads.k=${k}" \
        "output.dir=${random_dir}" $DRY

  run_step "$layer_comp" \
    python scripts/run_generation.py \
      --config "$CFG_DIR/05-layer-matched-random.yaml" \
      --overrides \
        "condition=layer_matched_random_head_ablation_top${k}" \
        "heads.top_k=${k}" \
        "output.dir=${layer_dir}" $DRY

  run_step "$safety_judge" \
    python scripts/run_judge.py \
      --config "$CFG_DIR/judge.yaml" \
      --completions "$safety_comp" \
      --out "$safety_judge" $DRY

  run_step "$random_judge" \
    python scripts/run_judge.py \
      --config "$CFG_DIR/judge.yaml" \
      --completions "$random_comp" \
      --out "$random_judge" $DRY

  run_step "$layer_judge" \
    python scripts/run_judge.py \
      --config "$CFG_DIR/judge.yaml" \
      --completions "$layer_comp" \
      --out "$layer_judge" $DRY

  if [[ -z "$DRY" ]]; then
    python scripts/run_evaluation.py \
      --judge-kind beavertails \
      --baseline "$BASELINE_JUDGE" \
      --compare "$safety_judge" "$random_judge" "$layer_judge" \
      --completions-for-coherence \
        "$BASELINE_COMP" "$safety_comp" "$random_comp" "$layer_comp" \
      --coherence \
      --plots-dir "runs/analysis/beavertails_dose_top${k}_report/plots" \
      --out "runs/analysis/beavertails_dose_top${k}_report/evaluation_report.json"

    python scripts/make_multi_condition_report.py \
      --report-kind beavertails \
      --title "BeaverTails Top-${k} Safety-Head Sweep: Baseline vs Safety vs Random vs Layer-Matched" \
      --out-dir "runs/analysis/beavertails_dose_top${k}_report" \
      --evaluation-report "runs/analysis/beavertails_dose_top${k}_report/evaluation_report.json" \
      --condition baseline "$BASELINE_COMP" "$BASELINE_JUDGE" \
      --condition "safety_head_ablation_top${k}" "$safety_comp" "$safety_judge" \
      --condition "random_head_ablation_top${k}" "$random_comp" "$random_judge" \
      --condition "layer_matched_random_head_ablation_top${k}" "$layer_comp" "$layer_judge"
  fi
done

echo "Done. See runs/analysis/beavertails_dose_top*_report/."
