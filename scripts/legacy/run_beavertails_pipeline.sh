#!/usr/bin/env bash
# Exp 7 end-to-end orchestration: SHIPS discovery on BeaverTails -> 4-condition
# generation -> BeaverTails dual-score judge -> pre/post HTML reports ->
# cross-condition evaluation with paired Wilcoxon + per-category breakdown.
#
# Requires a GPU. Pass --dry-run to validate configs without loading models.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then DRY="--dry-run"; fi

CFG_DIR=configs/experiments/exp07_beavertails_pipeline

# 1. SHIPS discovery on BeaverTails (uniform across 14 categories).
python scripts/run_attribution.py --config "$CFG_DIR/01-ships-discovery-beavertails.yaml" $DRY

# 2-3. Generation: baseline, safety-ablation, random, layer-matched.
for yaml in 02-baseline 03-safety-head-ablation 04-random-head-ablation 05-layer-matched-random; do
  python scripts/run_generation.py --config "$CFG_DIR/${yaml}.yaml" $DRY
done

# 4. Judge each condition with the dual-score BeaverTails judge.
declare -A IN=(
  [baseline]="runs/08-beaver-baseline/completions_baseline.jsonl"
  [safety_head_ablation]="runs/09-beaver-safety-ablation/completions_safety_head_ablation.jsonl"
  [random_head_ablation]="runs/10-beaver-random/completions_random_head_ablation.jsonl"
  [layer_matched_random_head_ablation]="runs/11-beaver-layer-matched/completions_layer_matched_random_head_ablation.jsonl"
)
declare -A OUT=(
  [baseline]="runs/08-beaver-baseline/judge_beavertails.jsonl"
  [safety_head_ablation]="runs/09-beaver-safety-ablation/judge_beavertails.jsonl"
  [random_head_ablation]="runs/10-beaver-random/judge_beavertails.jsonl"
  [layer_matched_random_head_ablation]="runs/11-beaver-layer-matched/judge_beavertails.jsonl"
)
for cond in baseline safety_head_ablation random_head_ablation layer_matched_random_head_ablation; do
  python scripts/run_judge.py \
    --config "$CFG_DIR/judge.yaml" \
    --completions "${IN[$cond]}" \
    --out         "${OUT[$cond]}" $DRY
done

# 5. Per-condition pre/post HTML reports with judge badges attached.
# Skipped under --dry-run because generation/judge produce no JSONLs in that mode.
if [[ -z "$DRY" ]]; then
for cond in safety_head_ablation random_head_ablation layer_matched_random_head_ablation; do
  python scripts/make_pre_post_report.py \
    --baseline       "${IN[baseline]}" \
    --ablation       "${IN[$cond]}" \
    --baseline-judge "${OUT[baseline]}" \
    --ablation-judge "${OUT[$cond]}" \
    --ablation-label "$cond" \
    --out-dir "runs/analysis/beavertails_pre_post_${cond}"
done

# 6. Cross-condition evaluation (paired Wilcoxon + per-category mean intent + plots).
python scripts/run_evaluation.py \
  --judge-kind beavertails \
  --baseline "${OUT[baseline]}" \
  --compare  "${OUT[safety_head_ablation]}" \
             "${OUT[random_head_ablation]}" \
             "${OUT[layer_matched_random_head_ablation]}" \
  --completions-for-coherence \
             "${IN[baseline]}" \
             "${IN[safety_head_ablation]}" \
             "${IN[random_head_ablation]}" \
             "${IN[layer_matched_random_head_ablation]}" \
  --coherence \
  --plots-dir runs/analysis/beavertails_report/plots \
  --out runs/analysis/beavertails_report/evaluation_report.json
fi

echo "Done. See runs/analysis/beavertails_report/ and runs/analysis/beavertails_pre_post_*/."
